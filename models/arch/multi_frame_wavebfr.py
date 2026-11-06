from pathlib import Path

import torch
import torch.nn as nn
from models.arch.restormer import (
    OverlapPatchEmbed,
    Downsample,
    Upsample,
    TransformerBlock,
)
from models.arch.wfen import HaarWavelet


class MultiFrameWaveBFR(nn.Module):
    def __init__(
        self,
        backbone="breeze",
        inp_channels=3,
        out_channels=3,
        dim=24,
        num_blocks=None,
        num_refinement_blocks=2,
        heads=None,
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super().__init__()
        self.backbone, self.dim = backbone.lower(), dim
        defaults = {
            "breeze": ((2, 2, 4), (1, 2, 4)),
            "wavebfr": ((2, 2, 2, 3), (1, 2, 2, 4)),
        }
        if self.backbone not in defaults:
            raise ValueError("backbone must be 'breeze' or 'wavebfr'")
        default_blocks, default_heads = defaults[self.backbone]
        nb, nh = tuple(num_blocks or default_blocks), tuple(heads or default_heads)
        if len(nb) != len(default_blocks) or len(nh) != len(default_heads):
            raise ValueError(
                f"{self.backbone} expects {len(default_blocks)} block/head values"
            )

        def blocks(channels, count, head):
            return nn.Sequential(
                *[
                    TransformerBlock(
                        dim=channels,
                        num_heads=head,
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias,
                        LayerNorm_type=LayerNorm_type,
                    )
                    for _ in range(count)
                ]
            )

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.wavelet_transform = HaarWavelet(dim, grad=False)
        self.feature_accumulator = nn.ModuleDict(
            {
                "delta": nn.Sequential(
                    nn.Conv2d(dim * 3, dim, 3, padding=1, bias=bias),
                    nn.SiLU(),
                    nn.Conv2d(dim, dim, 3, padding=1, bias=bias),
                ),
                "gate": nn.Conv2d(dim * 3, dim, 1, bias=True),
            }
        )
        nn.init.constant_(self.feature_accumulator["gate"].bias, -2.0)
        self.reference_fusion = nn.ModuleDict(
            {
                "body": nn.Sequential(
                    nn.Conv2d(dim * 2, dim, 3, padding=1, bias=bias),
                    nn.SiLU(),
                    nn.Conv2d(dim, dim, 3, padding=1, bias=bias),
                )
            }
        )
        nn.init.zeros_(self.reference_fusion["body"][-1].weight)
        if self.reference_fusion["body"][-1].bias is not None:
            nn.init.zeros_(self.reference_fusion["body"][-1].bias)

        for prefix, multiplier in (
            ("lf", 1),
            ("hf", 3 if self.backbone == "wavebfr" else 1),
        ):
            c1, c2, c3 = (dim * multiplier * 2**k for k in range(3))
            modules = {
                "encoder_lv1": blocks(c1, nb[0], nh[0]),
                "down1_2": Downsample(c1),
                "encoder_lv2": blocks(c2, nb[1], nh[1]),
                "down2_3": Downsample(c2),
                "latent": blocks(c3, nb[-1], nh[-1]),
                "up3_2": Upsample(c3),
                "reduce_chan_lv2": nn.Conv2d(c3, c2, 1, bias=bias),
                "decoder_lv2": blocks(c2, nb[1], nh[1]),
                "up2_1": Upsample(c2),
                "reduce_chan_lv1": nn.Conv2d(c2, c1, 1, bias=bias),
                "decoder_lv1": blocks(c1, nb[0], nh[0]),
            }
            if self.backbone == "wavebfr":
                modules.update(
                    {
                        "encoder_lv3": blocks(c3, nb[2], nh[2]),
                        "reduce_chan_lv3": nn.Conv2d(c3 * 2, c3, 1, bias=bias),
                        "decoder_lv3": blocks(c3, nb[2], nh[2]),
                        "refinement": blocks(c1, num_refinement_blocks, nh[0]),
                    }
                )
            for name, module in modules.items():
                setattr(self, f"{prefix}_{name}", module)

        if self.backbone == "breeze":
            self.hf_project_in = nn.Conv2d(dim * 3, dim, 1, bias=bias)
            self.hf_project_out = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        else:
            self.last_refinement = blocks(dim, num_refinement_blocks, nh[0])
        self.output = nn.Sequential(
            nn.Conv2d(dim, out_channels, 3, padding=1, bias=bias)
        )

    def _accumulate(self, memory, current, reference):
        x = torch.cat((memory, current, reference), dim=1)
        return memory + torch.sigmoid(
            self.feature_accumulator["gate"](x)
        ) * self.feature_accumulator["delta"](x)

    def _fuse_reference(self, reference, memory):
        return reference + self.reference_fusion["body"](
            torch.cat((reference, memory), dim=1)
        )

    def fuse_frames(self, frames, frame_mask=None):
        """Fuse valid prefix frames; absent frames are near-zero after normalization."""
        if isinstance(frames, torch.Tensor):
            frames = frames.unbind(dim=1)

        if frame_mask is None:
            scores = torch.stack(
                [frame.abs().mean(dim=(1, 2, 3)) for frame in frames],
                dim=1,
            )

            # 평균 픽셀값 0.02~0.03 구간에서 마스크가 0에서 1로 전환되도록 설정
            x = (scores - 0.02) * 100.0
            frame_mask = torch.relu(x) - torch.relu(x - 1.0)

        mask = frame_mask.to(dtype=frames[0].dtype)

        features = [self.patch_embed(frame) for frame in frames]

        reference, reference_image = features[0], frames[0]
        for i in range(1, len(frames)):
            valid = mask[:, i].reshape(-1, 1, 1, 1)
            reference = reference + valid * (features[i] - reference)
            reference_image = reference_image + valid * (frames[i] - reference_image)

        memory = features[0]
        for i in range(1, len(frames) - 1):
            active = (mask[:, i] * mask[:, i + 1]).reshape(-1, 1, 1, 1)
            updated = self._accumulate(memory, features[i], reference)
            memory = memory + active * (updated - memory)

        return self._fuse_reference(reference, memory), reference_image

    def _branch(self, x, prefix):
        def layer(name):
            return getattr(self, f"{prefix}_{name}")

        e1 = layer("encoder_lv1")(x)
        e2 = layer("encoder_lv2")(layer("down1_2")(e1))
        e3 = layer("down2_3")(e2)
        if self.backbone == "wavebfr":
            e3 = layer("encoder_lv3")(e3)
        z = layer("latent")(e3)
        if self.backbone == "wavebfr":
            z = layer("decoder_lv3")(layer("reduce_chan_lv3")(torch.cat((z, e3), 1)))
        d2 = layer("decoder_lv2")(
            layer("reduce_chan_lv2")(torch.cat((layer("up3_2")(z), e2), 1))
        )
        d1 = layer("decoder_lv1")(
            layer("reduce_chan_lv1")(torch.cat((layer("up2_1")(d2), e1), 1))
        )
        return x + layer("refinement")(d1) if self.backbone == "wavebfr" else d1

    def restore(self, feature, reference_image):
        ll, lh, hl, hh = self.wavelet_transform(feature, rev=False).split(
            self.dim, dim=1
        )
        hf = torch.cat((lh, hl, hh), dim=1)
        if self.backbone == "breeze":
            hf = self.hf_project_in(hf)
        lf, hf = self._branch(ll, "lf"), self._branch(hf, "hf")
        if self.backbone == "breeze":
            hf = self.hf_project_out(hf)
        x = self.wavelet_transform(torch.cat((lf, hf), dim=1), rev=True)
        if self.backbone == "wavebfr":
            x = self.last_refinement(x)
        return self.output(x) + reference_image

    def forward(self, inp_img, frame_mask=None):
        if inp_img.ndim == 4:
            return self.restore(self.patch_embed(inp_img), inp_img)
        if inp_img.ndim != 5:
            raise ValueError("Expected [B,3,H,W] or [B,T,3,H,W]")
        feature, reference = self.fuse_frames(inp_img.unbind(dim=1), frame_mask)
        return self.restore(feature, reference)

    def forward_strip(self, strip):
        """EVB input: normalized RGB [B,3,112,1120], padded with neutral gray."""
        feature, reference = self.fuse_frames(strip.split(112, dim=3))
        return self.restore(feature, reference)

    def load_single_frame_weights(self, checkpoint, map_location="cpu"):
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(
                checkpoint, map_location=map_location, weights_only=False
            )
        for key in (
            "params_ema",
            "params",
            "state_dict",
            "model",
            "generator",
            "netG",
            "net_g",
        ):
            if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break
        if not isinstance(checkpoint, dict):
            raise TypeError("Expected state_dict or checkpoint dict")

        own, compatible, skipped = self.state_dict(), {}, {}
        for key, value in checkpoint.items():
            if not torch.is_tensor(value):
                continue
            while key.startswith(("module.", "model.", "netG.", "net_g.")):
                key = key.split(".", 1)[1]
            if key in own and own[key].shape == value.shape:
                compatible[key] = value
            else:
                skipped[key] = (
                    tuple(value.shape),
                    tuple(own[key].shape) if key in own else None,
                )
        missing = self.load_state_dict(compatible, strict=False).missing_keys
        return {
            "loaded": len(compatible),
            "skipped": skipped,
            "missing": missing,
            "all_source_weights_loaded": not skipped,
        }


MultiFrameWaveBFRBreeze = MultiFrameWaveBFR
