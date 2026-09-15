import torch
import torch.nn as nn

from models.arch.restormer import (
    OverlapPatchEmbed,
    Downsample,
    Upsample,
    TransformerBlock,
    FeatureAccumulator,
    ReferenceResidualFusion,
    ResidualShallowEncoder,
)
from models.arch.wfen import HaarWavelet


class MultiFrameWaveBFRBreezeV1(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=16,
        num_blocks=(1, 2, 2),
        heads=(1, 2, 4),
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super().__init__()
        print("🏷️ Multi-Frame WaveBFR Breeze V1")

        self.dim = dim
        dims = [dim, dim * 2, dim * 4]

        def blocks(level):
            return nn.Sequential(
                *[
                    TransformerBlock(
                        dim=dims[level],
                        num_heads=heads[level],
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias,
                        LayerNorm_type=LayerNorm_type,
                    )
                    for _ in range(num_blocks[level])
                ]
            )

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.shallow_refine = ResidualShallowEncoder(
            dim=dim,
            heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

        self.feature_accumulator = FeatureAccumulator(dim=dim, bias=bias)
        self.reference_fusion = ReferenceResidualFusion(dim=dim, bias=bias)
        self.wavelet_transform = HaarWavelet(dim, grad=False)

        # LF branch
        self.lf_encoder_lv1 = blocks(0)
        self.lf_down1_2 = Downsample(dims[0])
        self.lf_encoder_lv2 = blocks(1)
        self.lf_down2_3 = Downsample(dims[1])
        self.lf_latent = blocks(2)
        self.lf_up3_2 = Upsample(dims[2])
        self.lf_reduce_chan_lv2 = nn.Conv2d(dims[2], dims[1], 1, bias=bias)
        self.lf_decoder_lv2 = blocks(1)
        self.lf_up2_1 = Upsample(dims[1])
        self.lf_reduce_chan_lv1 = nn.Conv2d(dims[1], dims[0], 1, bias=bias)
        self.lf_decoder_lv1 = blocks(0)

        # HF branch
        self.hf_project_in = nn.Conv2d(dim * 3, dim, 1, bias=bias)
        self.hf_encoder_lv1 = blocks(0)
        self.hf_down1_2 = Downsample(dims[0])
        self.hf_encoder_lv2 = blocks(1)
        self.hf_down2_3 = Downsample(dims[1])
        self.hf_latent = blocks(2)
        self.hf_up3_2 = Upsample(dims[2])
        self.hf_reduce_chan_lv2 = nn.Conv2d(dims[2], dims[1], 1, bias=bias)
        self.hf_decoder_lv2 = blocks(1)
        self.hf_up2_1 = Upsample(dims[1])
        self.hf_reduce_chan_lv1 = nn.Conv2d(dims[1], dims[0], 1, bias=bias)
        self.hf_decoder_lv1 = blocks(0)
        self.hf_project_out = nn.Conv2d(dim, dim * 3, 1, bias=bias)

        # Sequential을 유지해 기존 output.0.* state_dict 키 보존
        self.output = nn.Sequential(
            nn.Conv2d(dim, out_channels, 3, padding=1, bias=bias)
        )

    def encode_shallow(self, image):
        return self.shallow_refine(self.patch_embed(image))

    def accumulate_multi_frame_features(self, frames, frame_mask):
        """frames: [B,T,C,H,W], frame_mask: [B,T] (유효 프레임은 앞쪽부터 연속)."""
        if frames.ndim != 5:
            raise ValueError(f"frames must be [B,T,C,H,W], got {tuple(frames.shape)}")
        if frame_mask.ndim != 2 or frame_mask.shape != frames.shape[:2]:
            raise ValueError(
                f"frame_mask must be {tuple(frames.shape[:2])}, "
                f"got {tuple(frame_mask.shape)}"
            )

        b, t, c, h, w = frames.shape
        frame_mask = frame_mask.bool()
        valid_counts = frame_mask.sum(dim=1)

        if (valid_counts == 0).any():
            raise ValueError("Every sample must contain at least one valid frame.")

        last_indices = valid_counts - 1
        batch_indices = torch.arange(b, device=frames.device)

        features = self.encode_shallow(frames.reshape(b * t, c, h, w))
        features = features.reshape(b, t, *features.shape[1:])

        reference_feature = features[batch_indices, last_indices]
        reference_image = frames[batch_indices, last_indices]

        memory = torch.zeros_like(reference_feature)
        has_memory = torch.zeros(b, dtype=torch.bool, device=frames.device)

        for i in range(t):
            is_aux = frame_mask[:, i] & (i < last_indices)

            current = features[:, i]
            first_mask = (is_aux & ~has_memory)[:, None, None, None]
            recurrent_mask = (is_aux & has_memory)[:, None, None, None]

            memory = torch.where(first_mask, current, memory)
            updated_memory = self.feature_accumulator(memory, current)
            memory = torch.where(recurrent_mask, updated_memory, memory)
            has_memory = has_memory | is_aux

        return self.reference_fusion(reference_feature, memory), reference_image

    def _run_branch(self, x, prefix):
        """LF/HF에 공통인 3-level encoder-decoder 실행."""
        enc1 = getattr(self, f"{prefix}_encoder_lv1")(x)
        enc2 = getattr(self, f"{prefix}_encoder_lv2")(
            getattr(self, f"{prefix}_down1_2")(enc1)
        )
        latent = getattr(self, f"{prefix}_latent")(
            getattr(self, f"{prefix}_down2_3")(enc2)
        )

        dec2 = getattr(self, f"{prefix}_up3_2")(latent)
        dec2 = getattr(self, f"{prefix}_reduce_chan_lv2")(
            torch.cat([dec2, enc2], dim=1)
        )
        dec2 = getattr(self, f"{prefix}_decoder_lv2")(dec2)

        dec1 = getattr(self, f"{prefix}_up2_1")(dec2)
        dec1 = getattr(self, f"{prefix}_reduce_chan_lv1")(
            torch.cat([dec1, enc1], dim=1)
        )
        return getattr(self, f"{prefix}_decoder_lv1")(dec1)

    def restore_from_feature(self, feature, reference_image):
        ll, lh, hl, hh = self.wavelet_transform(feature, rev=False).split(
            self.dim, dim=1
        )

        lf = self._run_branch(ll, "lf")
        hf = self.hf_project_in(torch.cat([lh, hl, hh], dim=1))
        hf = self.hf_project_out(self._run_branch(hf, "hf"))

        restored_feature = self.wavelet_transform(
            torch.cat([lf, hf], dim=1),
            rev=True,
        )
        return self.output(restored_feature) + reference_image

    def forward(self, inp_img, frame_mask=None):
        if inp_img.ndim == 4:
            return self.restore_from_feature(
                self.encode_shallow(inp_img),
                inp_img,
            )

        if inp_img.ndim == 5:
            if frame_mask is None:
                frame_mask = torch.ones(
                    inp_img.shape[:2],
                    dtype=torch.bool,
                    device=inp_img.device,
                )

            feature, reference_image = self.accumulate_multi_frame_features(
                inp_img,
                frame_mask,
            )
            return self.restore_from_feature(feature, reference_image)

        raise ValueError(
            f"inp_img must be [B,C,H,W] or [B,T,C,H,W], " f"got {tuple(inp_img.shape)}"
        )


class ZeroInitReferenceResidualFusion(nn.Module):
    """Reference를 보존하면서 memory residual만 추가한다."""

    def __init__(self, dim, bias=False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1, bias=bias),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=bias),
        )

        nn.init.zeros_(self.body[-1].weight)
        if self.body[-1].bias is not None:
            nn.init.zeros_(self.body[-1].bias)

    def forward(self, reference, memory):
        residual = self.body(torch.cat([reference, memory], dim=1))
        return reference + residual


class ReferenceGatedFeatureAccumulator(nn.Module):
    """Reference와의 관련성을 이용해 auxiliary feature를 누적한다."""

    def __init__(self, dim, bias=False):
        super().__init__()

        self.delta = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 3, padding=1, bias=bias),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=bias),
        )
        self.gate = nn.Conv2d(dim * 3, dim, 1, bias=True)

        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, memory, current, reference):
        feature = torch.cat([memory, current, reference], dim=1)
        gate = torch.sigmoid(self.gate(feature))
        return memory + gate * self.delta(feature)


class MultiFrameWaveBFRBreezeV2(MultiFrameWaveBFRBreezeV1):
    """
    V1 + zero-initialized residual reference fusion.

    입력:
        inp_img:    [B,C,H,W] 또는 [B,T,C,H,W]
        frame_mask: [B,T]

    출력:
        마지막 유효 프레임을 reference로 한 SR 이미지 [B,C,H,W]
    """

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=16,
        num_blocks=(1, 1, 1),
        heads=(1, 2, 4),
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        print("🏷️ Multi-Frame WaveBFR Breeze V2")

        self.reference_fusion = ZeroInitReferenceResidualFusion(
            dim=dim,
            bias=bias,
        )


class MultiFrameWaveBFRBreezeV3(MultiFrameWaveBFRBreezeV2):
    """
    V2 + reference-conditioned gated feature accumulation.

    입력:
        inp_img:    [B,C,H,W] 또는 [B,T,C,H,W]
        frame_mask: [B,T]

    출력:
        마지막 유효 프레임을 reference로 한 SR 이미지 [B,C,H,W]
    """

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=16,
        num_blocks=(1, 1, 1),
        heads=(1, 2, 4),
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )
        print("🏷️ Multi-Frame WaveBFR Breeze V3")

        self.feature_accumulator = ReferenceGatedFeatureAccumulator(
            dim=dim,
            bias=bias,
        )

    def accumulate_multi_frame_features(self, frames, frame_mask):
        """Auxiliary feature를 reference-conditioned 방식으로 누적한다."""
        if frames.ndim != 5:
            raise ValueError(f"frames must be [B,T,C,H,W], got {tuple(frames.shape)}")
        if frame_mask.ndim != 2 or frame_mask.shape != frames.shape[:2]:
            raise ValueError(
                f"frame_mask must be {tuple(frames.shape[:2])}, "
                f"got {tuple(frame_mask.shape)}"
            )

        b, t, c, h, w = frames.shape
        frame_mask = frame_mask.bool()
        valid_counts = frame_mask.sum(dim=1)

        if (valid_counts == 0).any():
            raise ValueError("Every sample must contain at least one valid frame.")

        expected_mask = (
            torch.arange(t, device=frames.device)[None] < valid_counts[:, None]
        )
        if not torch.equal(frame_mask, expected_mask):
            raise ValueError("Valid frames must be contiguous from the beginning.")

        last_indices = valid_counts - 1
        batch_indices = torch.arange(b, device=frames.device)

        features = self.encode_shallow(frames.reshape(b * t, c, h, w))
        features = features.reshape(
            b,
            t,
            *features.shape[1:],
        )

        reference_feature = features[
            batch_indices,
            last_indices,
        ]
        reference_image = frames[
            batch_indices,
            last_indices,
        ]

        memory = torch.zeros_like(reference_feature)
        has_memory = torch.zeros(
            b,
            dtype=torch.bool,
            device=frames.device,
        )

        for i in range(t):
            is_aux = frame_mask[:, i] & (i < last_indices)

            current = features[:, i]
            first_aux = is_aux & ~has_memory
            later_aux = is_aux & has_memory

            memory = torch.where(
                first_aux[:, None, None, None],
                current,
                memory,
            )

            updated_memory = self.feature_accumulator(
                memory,
                current,
                reference_feature,
            )

            memory = torch.where(
                later_aux[:, None, None, None],
                updated_memory,
                memory,
            )
            has_memory = has_memory | is_aux

        fused_feature = self.reference_fusion(
            reference_feature,
            memory,
        )
        return fused_feature, reference_image
