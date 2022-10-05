from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of the test examples')
        parser.add_argument('--results_dir', type=str, default='/result/paris/', help='saves results here')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test')
        parser.add_argument('--save_number', type=int, default=1, help='choice # reasonable results based on the discriminator score')

        self.isTrain = False

        return parser
