from torch import nn
import torch
import math
from torch.nn import functional as F
from model.base_function import init_net
from einops import rearrange as rearrange


def define_g(init_type='normal', gpu_ids=[]):
    net = Generator(ngf=48)
    return init_net(net, init_type, gpu_ids)


def define_d(init_type= 'normal', gpu_ids=[]):
    net = Discriminator(in_channels=3)
    return init_net(net, init_type, gpu_ids)



class Generator(nn.Module):
    def __init__(self, ngf=48, num_block=[1,2,3,4], num_head=[1,2,4,8], factor=2.66):
        super().__init__()
        self.start = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.GELU()
        )
        self.trane256 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf, head=num_head[0],expansion_factor=factor) for i in range(num_block[0])]
        )
        self.down128 = Downsample(num_ch=ngf) # B *2ngf * 128, 128
        self.trane128 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf*2, head=num_head[1],expansion_factor=factor) for i in range(num_block[1])]
        )
        self.down64 = Downsample(num_ch=ngf*2) # B *4ngf * 64, 64
        self.trane64 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf*4, head=num_head[2],expansion_factor=factor) for i in range(num_block[2])]
        )
        self.down32 = Downsample(num_ch=ngf*4)  # B *8ngf * 32, 32
        self.trane32 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf*8, head=num_head[3],expansion_factor=factor) for i in range(num_block[3])]
        )

        self.up64 = Upsample(ngf*8)  # B *4ngf * 64, 64
        self.fuse64 = nn.Conv2d(in_channels=ngf*4*2, out_channels=ngf*4, kernel_size=1, stride=1, bias=False)
        self.trand64 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf*4, head=num_head[2],expansion_factor=factor) for i in range(num_block[2])]
        )

        self.up128 = Upsample(ngf*4) # B *2ngf * 128, 128
        self.fuse128 = nn.Conv2d(in_channels=4*ngf, out_channels=2*ngf, kernel_size=1, stride=1, bias=False)
        self.trand128 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf*2, head=num_head[1],expansion_factor=factor) for i in range(num_block[1])]
        )

        self.up256 = Upsample(ngf*2) # B *ngf * 256, 256
        self.fuse256 = nn.Conv2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=1, stride=1)
        self.trand256 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf, head=num_head[0],expansion_factor=factor) for i in range(num_block[0])]
        )

        self.trand2562 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf, head=num_head[0],expansion_factor=factor) for i in range(num_block[0])]
        )

        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=7, padding=0)
        )

    def forward(self, x, mask=None):
        feature = torch.cat([x, mask], dim=1)
        feature256 = self.start(feature)

        feature256 = self.trane256(feature256)
        feature128 = self.down128(feature256)
        feature128 = self.trane128(feature128)
        feature64 = self.down64(feature128)
        feature64 = self.trane64(feature64)
        feature32 = self.down32(feature64)
        feature32 = self.trane32(feature32)

        out64 = self.up64(feature32)
        out64 = self.fuse64(torch.cat([feature64, out64], dim=1))
        out64 = self.trand64(out64)

        out128 = self.up128(out64)
        out128 = self.fuse128(torch.cat([feature128, out128], dim=1))
        out128 = self.trand128(out128)

        out256 = self.up256(out128)
        out256 = self.fuse256(torch.cat([feature256, out256], dim=1))
        out256 = self.trand256(out256)

        out = torch.tanh(self.out(out256))
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


# (H * W) * C -> (H/2 * C/2) * (4C) -> (H/4 * W/4) * 16C -> (H/8 * W/8) * 64C
class TransformerEncoder(nn.Module):
    def __init__(self, in_ch=256, head=4, expansion_factor=2.66):
        super().__init__()

        self.attn = mGAttn(in_ch=in_ch, num_head=head)
        self.feed_forward = FeedForward(dim=in_ch, expansion_factor=expansion_factor)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.feed_forward(x)
        return x


class Convblock(nn.Module):
    def __init__(self, in_ch=256, out_ch=None, kernel_size=3, padding=1, stride=1):
        super().__init__()

        if out_ch is None or out_ch == in_ch:
            out_ch = in_ch
            self.projection = nn.Identity()
        else:
            self.projection = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, dilation=1)

        self.norm = nn.InstanceNorm2d(num_features=out_ch, track_running_stats=False)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GELU()
        )
        self.linear = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        residual = self.projection(x)
        x1 = self.conv(x)
        x2 = self.gate(x)
        out = x1 * x2
        out = self.norm(out)
        out = self.linear(out)
        out = out + residual
        return out


class Downsample(nn.Module):
    def __init__(self, num_ch=32):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=num_ch, out_channels=num_ch*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=num_ch*2, track_running_stats=False),
            nn.GELU()
        )


    def forward(self, x):
        return self.body(x)



class Upsample(nn.Module):
    def __init__(self, num_ch=32):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=num_ch, out_channels=num_ch//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=num_ch//2, track_running_stats=False),
            nn.GELU()
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.body(x)



class FeedForward(nn.Module):
    def __init__(self, dim=64, expansion_factor=2.66):
        super().__init__()

        num_ch = int(dim * expansion_factor)
        self.norm = nn.InstanceNorm2d(num_features=dim, track_running_stats=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=num_ch*2, kernel_size=1, bias=False),
            nn.Conv2d(in_channels=num_ch*2, out_channels=num_ch*2, kernel_size=3, stride=1, padding=1, groups=num_ch*2, bias=False)
        )
        self.linear = nn.Conv2d(in_channels=num_ch, out_channels=dim, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.norm(x)
        x1, x2 = self.conv(out).chunk(2, dim=1)
        out = F.gelu(x1) * x2
        out = self.linear(out)
        out = out + x
        return out


class mGAttn(nn.Module):
    def __init__(self, in_ch=256, num_head=4):
        super().__init__()
        self.head = num_head
        self.query = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.Softplus(),
        )

        self.key = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.Softplus(),
        )

        self.value = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU()
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU()
        )
        self.output_linear = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        self.norm = nn.InstanceNorm2d(num_features=in_ch)

    def forward(self, x):
        """
        x: b * c * h * w
        """
        x = self.norm(x)
        Ba, Ca, He, We = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        g = self.gate(x)
        num_per_head = Ca // self.head

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.head)   # B * head * c * N
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.head)   # B * head * c * N
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.head)   # B * head * c * N
        kv = torch.matmul(k, v.transpose(-2, -1))
        z = torch.einsum('bhcn,bhc -> bhn', q, k.sum(dim=-1)) / math.sqrt(num_per_head)
        z = 1.0 / (z + He*We)   # b h n
        out = torch.einsum('bhcn, bhcd-> bhdn', q, kv)
        out = out / math.sqrt(num_per_head)    # b h c n
        out = out + v
        out = out * z.unsqueeze(2)
        out = rearrange(out,'b head c (h w) -> b (head c) h w', h=He)
        out = out * g
        out = self.output_linear(out)
        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
