import os
from collections import OrderedDict

import pyiqa
import torch
import torch.nn as nn
import torch.optim as optim

from models import loss, networks
from .base_model import BaseModel
from utils import utils
from models.arch.wavebfr import WaveBFRBreeze
from models.arch.wfen import HaarWavelet
from helpers.arcface.models import resnet_face18


class WaveBFRModel(BaseModel):
    """Fine-tuning wrapper for multi-frame WaveBFRBreeze.

    Expected dataset output:
        LR:        [B, T_max, 3, H, W]
        LR_mask:   [B, T_max] (bool or 0/1)
        HR:        [B, 3, H, W]
        num_frames (optional): [B]

    The generator is expected to support:
        netG(LR, frame_mask=LR_mask) -> [B, 3, H, W]
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--scale_factor", type=int, default=8, help="upscale factor for model"
        )
        parser.add_argument(
            "--lambda_pix", type=float, default=0.1, help="weight for pixel loss"
        )
        parser.add_argument(
            "--lambda_ssim", type=float, default=0.0, help="weight for SSIM loss"
        )
        parser.add_argument(
            "--lambda_vgg", type=float, default=0.001, help="weight for VGG loss"
        )
        parser.add_argument(
            "--lambda_adv",
            type=float,
            default=0.001,
            help="weight for adversarial loss",
        )
        parser.add_argument(
            "--lambda_id", type=float, default=0.01, help="weight for identity loss"
        )
        parser.add_argument(
            "--lambda_dists", type=float, default=0.0, help="weight for DISTS loss"
        )
        parser.add_argument(
            "--lambda_lf",
            type=float,
            default=1.0,
            help="weight for low-frequency wavelet loss",
        )
        parser.add_argument(
            "--lambda_hf",
            type=float,
            default=1.0,
            help="weight for high-frequency wavelet loss",
        )

        # Multi-frame fine-tuning options
        parser.add_argument(
            "--backbone_lr_scale",
            type=float,
            default=0.1,
            help=(
                "learning-rate multiplier for pretrained WaveBFR parameters; "
                "new multi-frame modules use the base --lr"
            ),
        )
        parser.add_argument(
            "--pretrain_strict",
            action="store_true",
            help="strictly load generator checkpoint; normally keep disabled",
        )
        parser.add_argument(
            "--new_module_names",
            nargs="+",
            type=str,
            default=[
                "shallow_refine",
                "feature_accumulator",
                "reference_fusion",
            ],
            help="generator module-name prefixes trained with the full base LR",
        )
        parser.add_argument(
            "--visualize_all_lr",
            action="store_true",
            help="retain full LR sequence for custom visualization/debugging",
        )

        return parser

    def __init__(self, opt):
        super().__init__(opt)

        self.in_channels = 3

        # This must be the multi-frame version whose forward accepts frame_mask.
        self.netG = WaveBFRBreeze()
        self.netG = networks.define_network(opt, self.netG)

        self.wavelet_transform = HaarWavelet(
            in_channels=self.in_channels, grad=False
        ).to(device=opt.data_device)

        self.model_names = ["G"]
        self.load_model_names = ["G"]
        self.loss_names = ["Pix", "LF", "HF"]
        self.visual_names = ["img_LR", "img_SR", "img_HR"]

        if not self.isTrain:
            return

        self.criterionL1 = nn.L1Loss()

        # New fusion modules use opt.lr; pretrained body uses a smaller LR.
        new_params, backbone_params = self._split_generator_parameters()
        parameter_groups = []

        if new_params:
            parameter_groups.append(
                {
                    "params": new_params,
                    "lr": opt.lr,
                    "name": "multi_frame_modules",
                }
            )

        if backbone_params:
            parameter_groups.append(
                {
                    "params": backbone_params,
                    "lr": opt.lr * opt.backbone_lr_scale,
                    "name": "pretrained_backbone",
                }
            )

        self.optimizer_G = optim.Adam(
            parameter_groups,
            lr=opt.lr,
            betas=(opt.beta1, 0.99),
        )
        self.optimizers = [self.optimizer_G]

        print(
            "Generator LR groups: "
            f"new={opt.lr:.3e}, "
            f"backbone={opt.lr * opt.backbone_lr_scale:.3e}"
        )

        if opt.lambda_ssim > 0:
            print("➕ SSIM loss")
            self.loss_names.append("SSIM")
            self.criterionSSIM = pyiqa.create_metric(
                "ssim", as_loss=True, device=opt.data_device
            )

        if opt.lambda_vgg > 0:
            print("➕ VGG loss")
            self.loss_names.append("VGG")
            self.criterionPCP = loss.PCPLoss(opt)
            self.vgg19 = loss.PCPFeat(
                "./pretrain_models/vgg19-dcbb9e9d.pth", "vgg"
            )
            self.vgg19 = networks.define_network(
                opt, self.vgg19, isTrain=False, init_network=False
            )
            self.vgg19.requires_grad_(False)
            self.vgg19.eval()

        if opt.lambda_adv > 0:
            print("➕ adversarial loss")
            self.model_names.append("D")
            self.load_model_names.append("D")
            self.loss_names.extend(["FM", "G", "D"])

            self.criterionFM = loss.FMLoss().to(opt.data_device)
            self.criterionGAN = loss.GANLoss(opt.gan_mode).to(opt.data_device)

            # HF target contains H/V/D subbands: 3 input channels × 3.
            self.netD = networks.MultiScaleDiscriminator(
                self.in_channels * 3,
                n_layers=opt.n_layers_D,
                norm_type=opt.Dnorm,
                num_D=opt.num_D,
            )
            self.netD = networks.define_network(
                opt, self.netD, use_norm="spectral_norm"
            )

            self.optimizer_D = optim.Adam(
                self.netD.parameters(),
                lr=opt.d_lr,
                betas=(opt.beta1, 0.99),
            )
            self.optimizers.append(self.optimizer_D)

        if opt.lambda_id > 0:
            print("➕ identity loss")
            self.loss_names.append("ID")
            self.criterionID = loss.IDLoss()
            self.arcface_model = resnet_face18(
                use_se=False, use_feature_maps=True
            ).to(opt.data_device)
            self.arcface_model.load_state_dict(
                torch.load(
                    "helpers/arcface/weights/resnet18_110_wo_dist.pth",
                    map_location=opt.data_device,
                    weights_only=False,
                )
            )
            self.arcface_model.requires_grad_(False)
            self.arcface_model.eval()

        if opt.lambda_dists > 0:
            print("➕ DISTS loss")
            self.loss_names.append("DISTS")
            self.criterionDISTS = pyiqa.create_metric(
                "dists", as_loss=True, device=opt.data_device
            )

    # ------------------------------------------------------------------
    # Generator/checkpoint helpers
    # ------------------------------------------------------------------

    def _bare_netG(self):
        return self.netG.module if hasattr(self.netG, "module") else self.netG

    def _split_generator_parameters(self):
        """Split newly added multi-frame modules from pretrained parameters."""
        new_prefixes = tuple(self.opt.new_module_names)
        new_params = []
        backbone_params = []

        for name, parameter in self._bare_netG().named_parameters():
            if not parameter.requires_grad:
                continue

            if name.startswith(new_prefixes):
                new_params.append(parameter)
            else:
                backbone_params.append(parameter)

        print(
            "Fine-tuning parameters: "
            f"new={sum(p.numel() for p in new_params):,}, "
            f"backbone={sum(p.numel() for p in backbone_params):,}"
        )
        return new_params, backbone_params

    @staticmethod
    def _extract_state_dict(checkpoint):
        """Accept plain state dicts and common wrapped checkpoint formats."""
        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint must be a dict or state_dict-like object.")

        for key in ("state_dict", "netG", "generator", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

        cleaned = OrderedDict()
        for key, value in checkpoint.items():
            if not torch.is_tensor(value):
                continue

            while key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("netG."):
                key = key[len("netG.") :]

            cleaned[key] = value

        return cleaned

    def _load_generator_checkpoint(self, path):
        if not path:
            raise ValueError("A non-empty pretrained generator path is required.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Pretrained model not found: {path}")

        print("Loading pretrained generator:", path)
        checkpoint = torch.load(
            path,
            map_location=self.opt.data_device,
            weights_only=False,
        )
        state_dict = self._extract_state_dict(checkpoint)

        incompatible = self._bare_netG().load_state_dict(
            state_dict,
            strict=self.opt.pretrain_strict,
        )

        if not self.opt.pretrain_strict:
            print(f"Missing keys ({len(incompatible.missing_keys)}):")
            for key in incompatible.missing_keys:
                print("  -", key)
            print(f"Unexpected keys ({len(incompatible.unexpected_keys)}):")
            for key in incompatible.unexpected_keys:
                print("  -", key)

    def load_pretrain_model(self):
        self._load_generator_checkpoint(self.opt.pretrain_model_path)

    def for_load_pretrain_model(self, path):
        self._load_generator_checkpoint(path)

    # ------------------------------------------------------------------
    # Input / forward
    # ------------------------------------------------------------------

    def set_input(self, input, cur_iters=None):
        self.cur_iters = cur_iters

        self.img_LR_seq = input["LR"].to(
            self.opt.data_device, non_blocking=True
        )
        self.img_LR_mask = input["LR_mask"].to(
            self.opt.data_device, non_blocking=True
        ).bool()
        self.img_HR = input["HR"].to(
            self.opt.data_device, non_blocking=True
        )

        if self.img_LR_seq.ndim != 5:
            raise ValueError(
                "Multi-frame LR must have shape [B,T,C,H,W], "
                f"got {tuple(self.img_LR_seq.shape)}"
            )
        if self.img_LR_mask.ndim != 2:
            raise ValueError(
                "LR_mask must have shape [B,T], "
                f"got {tuple(self.img_LR_mask.shape)}"
            )
        if self.img_LR_seq.shape[:2] != self.img_LR_mask.shape:
            raise ValueError(
                "LR and LR_mask batch/time dimensions do not match: "
                f"{tuple(self.img_LR_seq.shape[:2])} vs "
                f"{tuple(self.img_LR_mask.shape)}"
            )

        valid_counts = self.img_LR_mask.long().sum(dim=1)
        if torch.any(valid_counts == 0):
            raise ValueError("Every sample must contain at least one valid LR frame.")

        # Last valid LR is the reference frame and is used for visualization.
        batch_indices = torch.arange(
            self.img_LR_seq.shape[0], device=self.img_LR_seq.device
        )
        last_indices = valid_counts - 1
        self.img_LR = self.img_LR_seq[batch_indices, last_indices]
        self.num_frames = input.get("num_frames", valid_counts)

        # Wavelet targets are computed once per batch.
        haar = self.wavelet_transform(self.img_HR, rev=False)
        self.img_lf_HR = haar.narrow(1, 0, self.in_channels)
        h = haar.narrow(1, self.in_channels, self.in_channels)
        v = haar.narrow(1, self.in_channels * 2, self.in_channels)
        d = haar.narrow(1, self.in_channels * 3, self.in_channels)
        self.img_hf_HR = torch.cat([h, v, d], dim=1)

    def forward(self):
        self.img_SR = self.netG(
            self.img_LR_seq,
            frame_mask=self.img_LR_mask,
        )

        haar = self.wavelet_transform(self.img_SR, rev=False)
        self.img_lf_SR = haar.narrow(1, 0, self.in_channels)
        h = haar.narrow(1, self.in_channels, self.in_channels)
        v = haar.narrow(1, self.in_channels * 2, self.in_channels)
        d = haar.narrow(1, self.in_channels * 3, self.in_channels)
        self.img_hf_SR = torch.cat([h, v, d], dim=1)

        if self.opt.lambda_vgg > 0:
            self.fake_vgg_feat = self.vgg19(self.img_SR)
            with torch.no_grad():
                self.real_vgg_feat = self.vgg19(self.img_HR)

        if self.opt.lambda_adv > 0:
            # Real branch does not need gradients for generator training.
            with torch.no_grad():
                self.real_D_results = self.netD(
                    self.img_hf_HR, return_feat=True
                )
            self.fake_D_results = self.netD(
                self.img_hf_SR.detach(), return_feat=False
            )
            self.fake_G_results = self.netD(
                self.img_hf_SR, return_feat=True
            )

    # ------------------------------------------------------------------
    # Losses and optimization
    # ------------------------------------------------------------------

    def backward_G(self):
        self.loss_Pix = (
            self.criterionL1(self.img_SR, self.img_HR) * self.opt.lambda_pix
        )
        self.loss_LF = (
            self.criterionL1(self.img_lf_SR, self.img_lf_HR) * self.opt.lambda_lf
        )
        self.loss_HF = (
            self.criterionL1(self.img_hf_SR, self.img_hf_HR) * self.opt.lambda_hf
        )

        total_loss = self.loss_Pix + self.loss_LF + self.loss_HF

        if self.opt.lambda_ssim > 0:
            self.loss_SSIM = (
                self.criterionSSIM(self.img_SR, self.img_HR)
                * self.opt.lambda_ssim
            )
            total_loss = total_loss + self.loss_SSIM

        if self.opt.lambda_vgg > 0:
            self.loss_VGG = (
                self.criterionPCP(self.fake_vgg_feat, self.real_vgg_feat)
                * self.opt.lambda_vgg
            )
            total_loss = total_loss + self.loss_VGG

        if self.opt.lambda_adv > 0:
            feature_matching = 0.0
            generator_gan = 0.0

            for i in range(self.opt.num_D):
                feature_matching = feature_matching + self.criterionFM(
                    self.fake_G_results[i][1],
                    self.real_D_results[i][1],
                )
                generator_gan = generator_gan + self.criterionGAN(
                    self.fake_G_results[i][0],
                    True,
                    for_discriminator=False,
                )

            self.loss_FM = (
                feature_matching
                * (self.opt.lambda_adv * 10.0)
                / self.opt.num_D
            )
            self.loss_G = (
                generator_gan * self.opt.lambda_adv / self.opt.num_D
            )
            total_loss = total_loss + self.loss_FM + self.loss_G

        if self.opt.lambda_id > 0:
            pred_embed = self.arcface_model(
                utils.process_arcface_input(self.img_lf_SR)
            )
            with torch.no_grad():
                hr_embed = self.arcface_model(
                    utils.process_arcface_input(self.img_lf_HR)
                )

            self.loss_ID = (
                self.criterionID(pred_embed, hr_embed) * self.opt.lambda_id
            )
            total_loss = total_loss + self.loss_ID

        if self.opt.lambda_dists > 0:
            self.loss_DISTS = (
                self.criterionDISTS(self.img_SR, self.img_HR)
                * self.opt.lambda_dists
            )
            total_loss = total_loss + self.loss_DISTS

        total_loss.backward()

    def backward_D(self):
        discriminator_loss = 0.0
        for i in range(self.opt.num_D):
            discriminator_loss = discriminator_loss + 0.5 * (
                self.criterionGAN(self.fake_D_results[i], False)
                + self.criterionGAN(self.real_D_results[i][0], True)
            )

        self.loss_D = discriminator_loss / self.opt.num_D
        self.loss_D.backward()

    @staticmethod
    def _set_requires_grad(network, requires_grad):
        if network is None:
            return
        for parameter in network.parameters():
            parameter.requires_grad = requires_grad

    def optimize_parameters(self):
        # The outer training loop already calls forward() once.
        if self.opt.lambda_adv > 0:
            self._set_requires_grad(self.netD, False)

        self.optimizer_G.zero_grad(set_to_none=True)
        self.backward_G()
        self.optimizer_G.step()

        if self.opt.lambda_adv > 0:
            self._set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad(set_to_none=True)
            self.backward_D()
            self.optimizer_D.step()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def get_current_visuals(self, size=128):
        # img_LR is the last valid/reference LR frame, not the full 5-D tensor.
        tensors = [self.img_LR, self.img_SR, self.img_HR]
        arrays = [utils.tensor_to_numpy(tensor) for tensor in tensors]
        return [utils.batch_numpy_to_image(array, size) for array in arrays]
