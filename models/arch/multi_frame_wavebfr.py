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
        dim=16,
        num_blocks=None,
        num_refinement_blocks=2,
        heads=None,
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super().__init__()

        self.backbone = backbone.lower()
        self.dim = dim

        if self.backbone == "wavebfr":
            print("🦴 WaveBFR backbone is loaded")
            num_blocks = tuple(num_blocks or (2, 2, 2, 4))
            heads = tuple(heads or (1, 2, 4, 8))
            if len(num_blocks) != 4 or len(heads) != 4:
                raise ValueError("WaveBFR requires 4 num_blocks/head values.")
        elif self.backbone == "breeze":
            print("🦴 Breeze backbone is loaded")
            num_blocks = tuple(num_blocks or (1, 1, 1))
            heads = tuple(heads or (1, 2, 4))
            if len(num_blocks) != 3 or len(heads) != 3:
                raise ValueError("WaveBFRBreeze requires 3 num_blocks/head values.")
        else:
            raise ValueError("backbone must be 'wavebfr' or 'breeze'.")

        def blocks(ch, n, h):
            return nn.Sequential(
                *[
                    TransformerBlock(
                        dim=ch,
                        num_heads=h,
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias,
                        LayerNorm_type=LayerNorm_type,
                    )
                    for _ in range(n)
                ]
            )

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.wavelet_transform = HaarWavelet(dim, grad=False)

        # Multi-frame modules: new weights, not present in single-frame checkpoints.
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

        if self.backbone == "wavebfr":
            self._build_wavebfr(
                blocks, dim, num_blocks, num_refinement_blocks, heads, bias
            )
        else:
            self._build_breeze(blocks, dim, num_blocks, heads, bias)

        self.output = nn.Sequential(
            nn.Conv2d(dim, out_channels, 3, stride=1, padding=1, bias=bias)
        )

    def _build_wavebfr(self, blocks, dim, nb, nr, heads, bias):
        for p, mul in (("lf", 1), ("hf", 3)):
            d1, d2, d3 = dim * mul, dim * mul * 2, dim * mul * 4

            setattr(self, f"{p}_encoder_lv1", blocks(d1, nb[0], heads[0]))
            setattr(self, f"{p}_down1_2", Downsample(d1))
            setattr(self, f"{p}_encoder_lv2", blocks(d2, nb[1], heads[1]))
            setattr(self, f"{p}_down2_3", Downsample(d2))
            setattr(self, f"{p}_encoder_lv3", blocks(d3, nb[2], heads[2]))

            setattr(self, f"{p}_latent", blocks(d3, nb[3], heads[3]))

            setattr(self, f"{p}_reduce_chan_lv3", nn.Conv2d(d3 * 2, d3, 1, bias=bias))
            setattr(self, f"{p}_decoder_lv3", blocks(d3, nb[2], heads[2]))

            setattr(self, f"{p}_up3_2", Upsample(d3))
            setattr(self, f"{p}_reduce_chan_lv2", nn.Conv2d(d3, d2, 1, bias=bias))
            setattr(self, f"{p}_decoder_lv2", blocks(d2, nb[1], heads[1]))

            setattr(self, f"{p}_up2_1", Upsample(d2))
            setattr(self, f"{p}_reduce_chan_lv1", nn.Conv2d(d2, d1, 1, bias=bias))
            setattr(self, f"{p}_decoder_lv1", blocks(d1, nb[0], heads[0]))

            setattr(self, f"{p}_refinement", blocks(d1, nr, heads[0]))

        self.last_refinement = blocks(dim, nr, heads[0])

    def _build_breeze(self, blocks, dim, nb, heads, bias):
        dims = (dim, dim * 2, dim * 4)

        for p in ("lf", "hf"):
            setattr(self, f"{p}_encoder_lv1", blocks(dims[0], nb[0], heads[0]))
            setattr(self, f"{p}_down1_2", Downsample(dims[0]))
            setattr(self, f"{p}_encoder_lv2", blocks(dims[1], nb[1], heads[1]))
            setattr(self, f"{p}_down2_3", Downsample(dims[1]))
            setattr(self, f"{p}_latent", blocks(dims[2], nb[2], heads[2]))

            setattr(self, f"{p}_up3_2", Upsample(dims[2]))
            setattr(
                self, f"{p}_reduce_chan_lv2", nn.Conv2d(dims[2], dims[1], 1, bias=bias)
            )
            setattr(self, f"{p}_decoder_lv2", blocks(dims[1], nb[1], heads[1]))

            setattr(self, f"{p}_up2_1", Upsample(dims[1]))
            setattr(
                self, f"{p}_reduce_chan_lv1", nn.Conv2d(dims[1], dims[0], 1, bias=bias)
            )
            setattr(self, f"{p}_decoder_lv1", blocks(dims[0], nb[0], heads[0]))

        self.hf_project_in = nn.Conv2d(dim * 3, dim, 1, bias=bias)
        self.hf_project_out = nn.Conv2d(dim, dim * 3, 1, bias=bias)

    def _accumulate(self, memory, current, reference):
        x = torch.cat([memory, current, reference], dim=1)
        gate = torch.sigmoid(self.feature_accumulator["gate"](x))
        return memory + gate * self.feature_accumulator["delta"](x)

    def _fuse_reference(self, reference, memory):
        residual = self.reference_fusion["body"](torch.cat([reference, memory], dim=1))
        return reference + residual

    def fuse_frames(self, frames, frame_mask=None):
        b, t, c, h, w = frames.shape

        if frame_mask is None:
            frame_mask = frames.abs().sum(dim=(2, 3, 4)) > 0
        else:
            frame_mask = frame_mask.bool()

        valid_counts = frame_mask.sum(dim=1)
        if (valid_counts == 0).any():
            raise ValueError("Every sample must contain at least one valid frame.")

        last = valid_counts - 1
        batch = torch.arange(b, device=frames.device)

        feat = torch.stack(
            [self.patch_embed(frames[:, i]) for i in range(t)],
            dim=1,
        )

        reference = feat[batch, last]
        reference_image = frames[batch, last]

        memory = torch.zeros_like(reference)
        has_memory = torch.zeros(b, dtype=torch.bool, device=frames.device)

        for i in range(t):
            active = frame_mask[:, i] & (i < last)
            first = active & ~has_memory
            later = active & has_memory
            current = feat[:, i]

            memory = torch.where(first[:, None, None, None], current, memory)
            updated = self._accumulate(memory, current, reference)
            memory = torch.where(later[:, None, None, None], updated, memory)
            has_memory = has_memory | active

        return self._fuse_reference(reference, memory), reference_image

    def _wavebfr_branch(self, x, p):
        e1 = getattr(self, f"{p}_encoder_lv1")(x)
        e2 = getattr(self, f"{p}_encoder_lv2")(getattr(self, f"{p}_down1_2")(e1))
        e3 = getattr(self, f"{p}_encoder_lv3")(getattr(self, f"{p}_down2_3")(e2))
        z = getattr(self, f"{p}_latent")(e3)

        d3 = getattr(self, f"{p}_decoder_lv3")(
            getattr(self, f"{p}_reduce_chan_lv3")(torch.cat([z, e3], dim=1))
        )
        d2 = getattr(self, f"{p}_up3_2")(d3)
        d2 = getattr(self, f"{p}_decoder_lv2")(
            getattr(self, f"{p}_reduce_chan_lv2")(torch.cat([d2, e2], dim=1))
        )
        d1 = getattr(self, f"{p}_up2_1")(d2)
        d1 = getattr(self, f"{p}_decoder_lv1")(
            getattr(self, f"{p}_reduce_chan_lv1")(torch.cat([d1, e1], dim=1))
        )
        return x + getattr(self, f"{p}_refinement")(d1)

    def _breeze_branch(self, x, p):
        e1 = getattr(self, f"{p}_encoder_lv1")(x)
        e2 = getattr(self, f"{p}_encoder_lv2")(getattr(self, f"{p}_down1_2")(e1))
        z = getattr(self, f"{p}_latent")(getattr(self, f"{p}_down2_3")(e2))

        d2 = getattr(self, f"{p}_up3_2")(z)
        d2 = getattr(self, f"{p}_decoder_lv2")(
            getattr(self, f"{p}_reduce_chan_lv2")(torch.cat([d2, e2], dim=1))
        )
        d1 = getattr(self, f"{p}_up2_1")(d2)
        return getattr(self, f"{p}_decoder_lv1")(
            getattr(self, f"{p}_reduce_chan_lv1")(torch.cat([d1, e1], dim=1))
        )

    def restore(self, feature, reference_image):
        ll, lh, hl, hh = self.wavelet_transform(feature, rev=False).split(
            self.dim, dim=1
        )
        hf = torch.cat([lh, hl, hh], dim=1)

        if self.backbone == "wavebfr":
            lf = self._wavebfr_branch(ll, "lf")
            hf = self._wavebfr_branch(hf, "hf")
            restored = self.wavelet_transform(torch.cat([lf, hf], dim=1), rev=True)
            restored = self.last_refinement(restored)
        else:
            lf = self._breeze_branch(ll, "lf")
            hf = self.hf_project_out(self._breeze_branch(self.hf_project_in(hf), "hf"))
            restored = self.wavelet_transform(torch.cat([lf, hf], dim=1), rev=True)

        return self.output(restored) + reference_image

    def forward(self, inp_img, frame_mask=None):
        if inp_img.ndim == 4:
            return self.restore(self.patch_embed(inp_img), inp_img)

        if inp_img.ndim != 5:
            raise ValueError(
                f"Expected [B,C,H,W] or [B,T,C,H,W], got {tuple(inp_img.shape)}"
            )

        feature, reference = self.fuse_frames(inp_img, frame_mask)
        return self.restore(feature, reference)

    def load_single_frame_weights(self, checkpoint, map_location="cpu"):
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(checkpoint, map_location=map_location)

        state = checkpoint
        for key in ("params_ema", "params", "state_dict", "model", "net_g"):
            if (
                isinstance(state, dict)
                and key in state
                and isinstance(state[key], dict)
            ):
                state = state[key]
                break

        cleaned = {}
        for key, value in state.items():
            while any(key.startswith(p) for p in ("module.", "model.", "net_g.")):
                for p in ("module.", "model.", "net_g."):
                    if key.startswith(p):
                        key = key[len(p) :]
                        break
            cleaned[key] = value

        own = self.state_dict()
        loaded, mismatched, unexpected = {}, {}, []

        for key, value in cleaned.items():
            if key not in own:
                unexpected.append(key)
            elif own[key].shape != value.shape:
                mismatched[key] = (tuple(value.shape), tuple(own[key].shape))
            else:
                loaded[key] = value

        self.load_state_dict(loaded, strict=False)

        return {
            "loaded": len(loaded),
            "source": len(cleaned),
            "all_source_weights_loaded": len(loaded) == len(cleaned),
            "mismatched": mismatched,
            "unexpected": unexpected,
        }


# Existing Breeze code can keep using this name.
MultiFrameWaveBFRBreeze = MultiFrameWaveBFR
