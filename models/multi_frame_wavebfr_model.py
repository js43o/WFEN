import pyiqa
import torch
from torch import nn, optim

from helpers.arcface.models import resnet_face18
from models import loss, networks
from models.arch.multi_frame_wavebfr import MultiFrameWaveBFR
from models.arch.wfen import HaarWavelet
from utils import utils
from .base_model import BaseModel


class MultiFrameWaveBFRModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        losses = {
            "pix": 0.1,
            "ssim": 0.0,
            "vgg": 0.001,
            "adv": 0.001,
            "id": 0.01,
            "dists": 0.0,
            "lf": 1.0,
            "hf": 1.0,
        }
        for name, value in losses.items():
            parser.add_argument(f"--lambda_{name}", type=float, default=value)
        parser.add_argument("--scale_factor", type=int, default=8)
        parser.add_argument("--backbone_lr_scale", type=float, default=0.1)
        parser.add_argument("--pretrain_strict", action="store_true")
        parser.add_argument(
            "--new_module_names",
            nargs="+",
            default=[
                "feature_accumulator",
                "reference_fusion",
            ],
        )
        parser.add_argument(
            "--wavebfr_backbone", choices=("breeze", "wavebfr"), default="breeze"
        )
        parser.add_argument("--wavebfr_dim", type=int, default=24)
        parser.add_argument("--wavebfr_num_blocks", type=int, nargs="+", default=None)
        parser.add_argument("--wavebfr_heads", type=int, nargs="+", default=None)
        parser.add_argument("--wavebfr_num_refinement_blocks", type=int, default=2)
        parser.add_argument("--visualize_all_lr", action="store_true")
        for name, default in (
            ("min_frames", 5),
            ("max_frames", 10),
            ("min_lr_size", 16),
            ("max_lr_size", 112),
        ):
            parser.add_argument(f"--{name}", type=int, default=default)
        parser.add_argument("--allow_duplicate_frames", action="store_true")
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.netG = networks.define_network(
            opt,
            MultiFrameWaveBFR(
                backbone=opt.wavebfr_backbone,
                dim=opt.wavebfr_dim,
                num_blocks=opt.wavebfr_num_blocks,
                heads=opt.wavebfr_heads,
                num_refinement_blocks=opt.wavebfr_num_refinement_blocks,
            ),
        )
        self.wavelet_transform = HaarWavelet(3, grad=False).to(opt.data_device)
        self.model_names = ["G"]
        self.load_model_names = ["G"]
        self.loss_names = ["Pix", "LF", "HF"]
        self.visual_names = ["img_LR", "img_SR", "img_HR"]
        if not self.isTrain:
            return

        self.criterionL1 = nn.L1Loss()
        prefixes = tuple(opt.new_module_names)
        fresh, pretrained = [], []
        for name, param in self.bare_generator.named_parameters():
            if param.requires_grad:
                (fresh if name.startswith(prefixes) else pretrained).append(param)
        groups = [
            {"params": fresh, "lr": opt.lr, "name": "multi_frame_modules"},
            {
                "params": pretrained,
                "lr": opt.lr * opt.backbone_lr_scale,
                "name": "pretrained_backbone",
            },
        ]
        self.optimizer_G = optim.Adam(
            [g for g in groups if g["params"]], lr=opt.lr, betas=(opt.beta1, 0.99)
        )
        self.optimizers = [self.optimizer_G]
        print(
            f"Generator LR: fusion={opt.lr:.2e}, backbone={opt.lr * opt.backbone_lr_scale:.2e}"
        )

        if opt.lambda_ssim > 0:
            self.loss_names.append("SSIM")
            self.criterionSSIM = pyiqa.create_metric(
                "ssim", as_loss=True, device=opt.data_device
            )
        if opt.lambda_vgg > 0:
            self.loss_names.append("VGG")
            self.criterionPCP = loss.PCPLoss(opt)
            self.vgg19 = (
                networks.define_network(
                    opt,
                    loss.PCPFeat("./pretrain_models/vgg19-dcbb9e9d.pth", "vgg"),
                    isTrain=False,
                    init_network=False,
                )
                .eval()
                .requires_grad_(False)
            )
        if opt.lambda_adv > 0:
            self.model_names.append("D")
            self.load_model_names.append("D")
            self.loss_names += ["FM", "G", "D"]
            self.criterionFM = loss.FMLoss().to(opt.data_device)
            self.criterionGAN = loss.GANLoss(opt.gan_mode).to(opt.data_device)
            self.netD = networks.define_network(
                opt,
                networks.MultiScaleDiscriminator(
                    9, n_layers=opt.n_layers_D, norm_type=opt.Dnorm, num_D=opt.num_D
                ),
                use_norm="spectral_norm",
            )
            self.optimizer_D = optim.Adam(
                self.netD.parameters(), lr=opt.d_lr, betas=(opt.beta1, 0.99)
            )
            self.optimizers.append(self.optimizer_D)
        if opt.lambda_id > 0:
            self.loss_names.append("ID")
            self.criterionID = loss.IDLoss()
            self.arcface_model = resnet_face18(use_se=False, use_feature_maps=True).to(
                opt.data_device
            )
            self.arcface_model.load_state_dict(
                torch.load(
                    "helpers/arcface/weights/resnet18_110_wo_dist.pth",
                    map_location=opt.data_device,
                    weights_only=False,
                )
            )
            self.arcface_model.eval().requires_grad_(False)
        if opt.lambda_dists > 0:
            self.loss_names.append("DISTS")
            self.criterionDISTS = pyiqa.create_metric(
                "dists", as_loss=True, device=opt.data_device
            )

    @property
    def bare_generator(self):
        return self.netG.module if hasattr(self.netG, "module") else self.netG

    def load_pretrain_model(self):
        self.for_load_pretrain_model(self.opt.pretrain_model_path)

    def for_load_pretrain_model(self, path):
        report = self.bare_generator.load_single_frame_weights(
            path, self.opt.data_device
        )
        print(
            f"Pretrained weights: loaded={report['loaded']}, "
            f"skipped={len(report['skipped'])}, missing={len(report['missing'])}"
        )
        for key in list(report["skipped"])[:20]:
            print("  Skipped:", key, report["skipped"][key])
        if self.opt.pretrain_strict and (report["skipped"] or report["missing"]):
            raise RuntimeError("Checkpoint does not exactly match the generator")
        return report

    @staticmethod
    def _bands(image, wavelet):
        ll, lh, hl, hh = wavelet(image, rev=False).split(3, dim=1)
        return ll, torch.cat((lh, hl, hh), dim=1)

    def set_input(self, data, cur_iters=None):
        self.cur_iters = cur_iters
        device = self.opt.data_device
        self.img_LR_seq = data["LR"].to(device, non_blocking=True)
        self.img_LR_mask = data["LR_mask"].to(device, non_blocking=True).bool()
        self.img_HR = data["HR"].to(device, non_blocking=True)
        if (
            self.img_LR_seq.ndim != 5
            or self.img_LR_seq.shape[:2] != self.img_LR_mask.shape
        ):
            raise ValueError("Expected LR [B,10,3,H,W] and LR_mask [B,10]")
        counts = self.img_LR_mask.sum(1)
        if (counts < 1).any():
            raise ValueError("Input requires at least one valid frame")
        batch = torch.arange(self.img_LR_seq.shape[0], device=self.img_LR_seq.device)
        self.img_LR = self.img_LR_seq[batch, counts - 1]
        self.num_frames = data.get("num_frames", counts)
        self.img_lf_HR, self.img_hf_HR = self._bands(
            self.img_HR, self.wavelet_transform
        )

    def forward(self):
        self.img_SR = self.netG(self.img_LR_seq, frame_mask=self.img_LR_mask)
        self.img_lf_SR, self.img_hf_SR = self._bands(
            self.img_SR, self.wavelet_transform
        )
        if self.opt.lambda_vgg > 0:
            self.fake_vgg_feat = self.vgg19(self.img_SR)
            with torch.no_grad():
                self.real_vgg_feat = self.vgg19(self.img_HR)
        if self.opt.lambda_adv > 0:
            with torch.no_grad():
                self.real_D_results = self.netD(self.img_hf_HR, return_feat=True)
            self.fake_D_results = self.netD(self.img_hf_SR.detach(), return_feat=False)
            self.fake_G_results = self.netD(self.img_hf_SR, return_feat=True)

    def backward_G(self):
        self.loss_Pix = self.criterionL1(self.img_SR, self.img_HR) * self.opt.lambda_pix
        self.loss_LF = (
            self.criterionL1(self.img_lf_SR, self.img_lf_HR) * self.opt.lambda_lf
        )
        self.loss_HF = (
            self.criterionL1(self.img_hf_SR, self.img_hf_HR) * self.opt.lambda_hf
        )
        total = self.loss_Pix + self.loss_LF + self.loss_HF
        if self.opt.lambda_ssim > 0:
            self.loss_SSIM = (
                self.criterionSSIM(self.img_SR, self.img_HR) * self.opt.lambda_ssim
            )
            total += self.loss_SSIM
        if self.opt.lambda_vgg > 0:
            self.loss_VGG = (
                self.criterionPCP(self.fake_vgg_feat, self.real_vgg_feat)
                * self.opt.lambda_vgg
            )
            total += self.loss_VGG
        if self.opt.lambda_adv > 0:
            fm = sum(
                self.criterionFM(self.fake_G_results[i][1], self.real_D_results[i][1])
                for i in range(self.opt.num_D)
            )
            gan = sum(
                self.criterionGAN(
                    self.fake_G_results[i][0], True, for_discriminator=False
                )
                for i in range(self.opt.num_D)
            )
            self.loss_FM = fm * self.opt.lambda_adv * 10 / self.opt.num_D
            self.loss_G = gan * self.opt.lambda_adv / self.opt.num_D
            total += self.loss_FM + self.loss_G
        if self.opt.lambda_id > 0:
            pred = self.arcface_model(utils.process_arcface_input(self.img_lf_SR))
            with torch.no_grad():
                target = self.arcface_model(utils.process_arcface_input(self.img_lf_HR))
            self.loss_ID = self.criterionID(pred, target) * self.opt.lambda_id
            total += self.loss_ID
        if self.opt.lambda_dists > 0:
            self.loss_DISTS = (
                self.criterionDISTS(self.img_SR, self.img_HR) * self.opt.lambda_dists
            )
            total += self.loss_DISTS
        total.backward()

    def backward_D(self):
        self.loss_D = (
            sum(
                (
                    self.criterionGAN(self.fake_D_results[i], False)
                    + self.criterionGAN(self.real_D_results[i][0], True)
                )
                * 0.5
                for i in range(self.opt.num_D)
            )
            / self.opt.num_D
        )
        self.loss_D.backward()

    def optimize_parameters(self):
        if self.opt.lambda_adv > 0:
            self.netD.requires_grad_(False)
        self.optimizer_G.zero_grad(set_to_none=True)
        self.backward_G()
        self.optimizer_G.step()
        if self.opt.lambda_adv > 0:
            self.netD.requires_grad_(True)
            self.optimizer_D.zero_grad(set_to_none=True)
            self.backward_D()
            self.optimizer_D.step()

    def get_current_visuals(self, size=128):
        arrays = [
            utils.tensor_to_numpy(x) for x in (self.img_LR, self.img_SR, self.img_HR)
        ]
        return [utils.batch_numpy_to_image(x, size) for x in arrays]
