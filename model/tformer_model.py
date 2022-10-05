import torch
from model.base_model import BaseModel
from model import base_function, network
import itertools
from model.loss import AdversarialLoss, PerceptualLoss, StyleLoss


class tformer(BaseModel):
    """This class implements the tformer, for 256*256 resolution image inpainting"""
    def name(self):
        return "tformer"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--output_scale', type=int, default=1, help='# of number of the output scale')
        if is_train:
            parser.add_argument('--train_paths', type=str, default='two', help='training strategies with one path or two paths')
            parser.add_argument('--lambda_per', type=float, default=1, help='weight for perceptual loss')
            parser.add_argument('--lambda_l1', type=float, default=1, help='weight for reconstruction loss l1')
            parser.add_argument('--lambda_g', type=float, default=0.1, help='weight for generation loss')
            parser.add_argument('--lambda_sty', type=float, default=250, help='weight for style loss')

        return parser

    def __init__(self, opt):
        """Initial the inpainting model"""
        BaseModel.__init__(self, opt)

        self.loss_names = ['app_g', 'ad_g', 'img_d', 'per', 'sty']
        self.visual_names = ['img_m', 'img_truth', 'img_out', 'img_g']
        self.model_names = ['G', 'D',]


        # define the inpainting model
        self.net_G = network.define_g(gpu_ids=opt.gpu_ids)
        # define the discriminator model
        self.net_D = network.define_d(gpu_ids=opt.gpu_ids)

        self.net_G = self.net_G.cuda(self.gpu_ids[0])
        self.net_D = self.net_D.cuda(self.gpu_ids[0])


        if self.isTrain:
            # define the loss functions
            self.GANloss = AdversarialLoss(type='nsgan')
            self.L1loss = torch.nn.L1Loss()
            self.per = PerceptualLoss()
            self.sty = StyleLoss()
            # define the optimizer
            self.optimizer_G = torch.optim.AdamW(itertools.chain(filter(lambda p: p.requires_grad, self.net_G.parameters())), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.AdamW(itertools.chain(filter(lambda p: p.requires_grad, self.net_D.parameters())),lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        # load the pretrained model and schedulers
        self.setup(opt)

    def set_input(self, input, epoch=0):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.input = input
        self.image_paths = self.input['img_path']
        self.img = input['img']
        self.mask = input['mask']

        if len(self.gpu_ids) > 0:
            self.img = self.img.cuda(self.gpu_ids[0])
            self.mask = self.mask.cuda(self.gpu_ids[0])


        self.img_truth = self.img * 2 - 1
        self.img_m = self.mask * self.img_truth


    def test(self):
        """Forward function used in test time"""
        # save the groundtruth and masked image
        self.save_results(self.img_truth, data_name='truth')
        self.save_results(self.img_m, data_name='mask')

        self.net_G.eval()

        self.img_g = self.net_G(self.img_m, self.mask)

        self.img_out = self.img_g * (1 - self.mask) + self.img_truth * self.mask
        self.save_results(self.img_out, data_name='out')


    def forward(self):
        """Run forward processing to get the inputs"""
        # encoder process
        self.img_g = self.net_G(self.img_m, self.mask)
        self.img_out = self.img_g * (1 - self.mask) + self.img_truth * self.mask

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real, _ = netD(real)

        # fake
        D_fake, _ = netD(fake.detach())
        # loss for discriminator
        D_loss = (self.GANloss(D_real, True, True) + self.GANloss(D_fake, False, True)) / 2

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D)
        self.loss_img_d = self.backward_D_basic(self.net_D, self.img_truth, self.img_g)

    def backward_G(self):
        """Calculate training loss for the generator"""

        # generator adversarial loss
        base_function._freeze(self.net_D)
        # g loss fake
        D_fake, _ = self.net_D(self.img_g)

        self.loss_ad_g = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        # calculate l1 loss
        totalG_loss = 0
        self.loss_app_g = self.L1loss(self.img_truth, self.img_g) * self.opt.lambda_l1
        self.loss_per = self.per(self.img_g, self.img_truth) * self.opt.lambda_per
        self.loss_sty = self.sty(self.img_truth * (1 - self.mask), self.img_g * (1 - self.mask)) * self.opt.lambda_sty

        totalG_loss = self.loss_app_g + self.loss_per + self.loss_sty + self.loss_ad_g

        totalG_loss.backward()

    def optimize_parameters(self):
        """update network weights"""
        # compute the inpainting results
        self.forward()
        # optimize the discrinimator network parameters
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # optimize the completion network parameters
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
