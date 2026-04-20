from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--test_img_path', type=str, default='', help='path of single test image.')
        parser.add_argument('--test_upscale', type=int, default=1, help='upscale single test image.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--save_as_dir', type=str, default='', help='save results in different dir.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--pretrain_model_path', type=str, default='', help='load pretrain model path if specified')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        
        parser.add_argument('--blur_kernel_size', nargs='+', type=int, default=[19, 20], help='blur kernel size')
        parser.add_argument('--kernel_list', nargs='+', type=str, default=['iso', 'aniso'], help='kernel list')
        parser.add_argument('--kernel_prob', nargs='+', type=float, default=[0.5, 0.5], help='kernel probability')
        parser.add_argument('--blur_sigma', nargs='+', type=float, default=[0.1, 10], help='blur sigma')
        parser.add_argument('--downsample_range', nargs='+', type=float, default=[0.8, 8], help='downsample range')
        parser.add_argument('--noise_range', nargs='+', type=int, default=[0, 20], help='noise range')
        parser.add_argument('--jpeg_range', nargs='+', type=int, default=[60, 100], help='jpeg range')

        
        self.isTrain = False
        return parser
