import torch
import torch.nn as nn
from models.arch.restormer import OverlapPatchEmbed, Downsample, Upsample, TransformerBlock, LightweightHFBlock, FeatureAccumulator, ReferenceResidualFusion, ResidualShallowEncoder
from models.arch.wfen import HaarWavelet


class WaveBFR(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=16,  # 임베딩 채널 수
        num_blocks=[2, 2, 2, 4],  # encoder/decoder 각 계층의 Transformer 블록 수
        num_refinement_blocks=2,  # 네트워크 마지막 refinement 단계의 Transformer 블록 수
        heads=[1, 2, 4, 8],  # 각 계층 내 Transformer의 multi-head 개수 정의
        ffn_expansion_factor=2.66,  # FFN 블록의 hidden 채널 확장 비율
        bias=False,  # attention 연산에 쓰이는 conv 레이어의 bias 사용 여부
        LayerNorm_type="WithBias",  ## Other option 'BiasFree'
    ):

        print("🌊 WaveBFR")
        super(WaveBFR, self).__init__()

        self.inp_channels = inp_channels
        self.dim = dim

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.wavelet_transform = HaarWavelet(dim, grad=False)

        # LF subbands network 📉
        self.lf_encoder_lv1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.lf_down1_2 = Downsample(dim)
        self.lf_encoder_lv2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.lf_down2_3 = Downsample(int(dim * 2**1))
        self.lf_encoder_lv3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        # bottleneck
        self.lf_latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[3])
            ]
        )

        self.lf_reduce_chan_lv3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias
        )
        self.lf_decoder_lv3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.lf_up3_2 = Upsample(int(dim * 2**2))
        self.lf_reduce_chan_lv2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.lf_decoder_lv2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.lf_up2_1 = Upsample(int(dim * 2**1))
        self.lf_reduce_chan_lv1 = nn.Conv2d(
            int(dim * 2**1), int(dim), kernel_size=1, bias=bias
        )
        self.lf_decoder_lv1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.lf_refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        # HF subbands network 📈
        self.hf_encoder_lv1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim * 3,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.hf_down1_2 = Downsample(dim * 3)
        self.hf_encoder_lv2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 3 * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.hf_down2_3 = Downsample(int(dim * 3 * 2**1))
        self.hf_encoder_lv3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 3 * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        # bottleneck
        self.hf_latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 3 * 2**2),
                    num_heads=heads[3],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[3])
            ]
        )

        self.hf_reduce_chan_lv3 = nn.Conv2d(
            int(dim * 3 * 2**3), int(dim * 3 * 2**2), kernel_size=1, bias=bias
        )
        self.hf_decoder_lv3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 3 * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.hf_up3_2 = Upsample(int(dim * 3 * 2**2))
        self.hf_reduce_chan_lv2 = nn.Conv2d(
            int(dim * 3 * 2**2), int(dim * 3 * 2**1), kernel_size=1, bias=bias
        )
        self.hf_decoder_lv2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 3 * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.hf_up2_1 = Upsample(int(dim * 3 * 2**1))
        self.hf_reduce_chan_lv1 = nn.Conv2d(
            int(dim * 3 * 2**1), int(dim * 3), kernel_size=1, bias=bias
        )
        self.hf_decoder_lv1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 3),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        self.hf_refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 3),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        # image-level 🖇️
        self.last_refinement = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_refinement_blocks)
            ]
        )

        self.output = nn.Sequential(
            nn.Conv2d(
                int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
            )
        )

    def forward(self, inp_img):

        x = self.patch_embed(inp_img)
        haar = self.wavelet_transform(x, rev=False)

        # refine LF subband 📉
        a = haar.narrow(1, 0, self.dim)

        ##### encoder #####
        lf_out_enc_lv1 = self.lf_encoder_lv1(a)

        lf_inp_enc_lv2 = self.lf_down1_2(lf_out_enc_lv1)
        lf_out_enc_lv2 = self.lf_encoder_lv2(lf_inp_enc_lv2)

        lf_inp_enc_lv3 = self.lf_down2_3(lf_out_enc_lv2)
        lf_out_enc_lv3 = self.lf_encoder_lv3(lf_inp_enc_lv3)

        ##### bottleneck #####
        lf_latent = self.lf_latent(lf_out_enc_lv3)

        ##### decoder #####
        lf_inp_dec_lv3 = torch.cat([lf_latent, lf_out_enc_lv3], 1)
        lf_inp_dec_lv3 = self.lf_reduce_chan_lv3(lf_inp_dec_lv3)
        lf_out_dec_lv3 = self.lf_decoder_lv3(lf_inp_dec_lv3)

        lf_inp_dec_lv2 = self.lf_up3_2(lf_out_dec_lv3)
        lf_inp_dec_lv2 = torch.cat([lf_inp_dec_lv2, lf_out_enc_lv2], 1)
        lf_inp_dec_lv2 = self.lf_reduce_chan_lv2(lf_inp_dec_lv2)
        lf_out_dec_lv2 = self.lf_decoder_lv2(lf_inp_dec_lv2)

        lf_inp_dec_lv1 = self.lf_up2_1(lf_out_dec_lv2)
        lf_inp_dec_lv1 = torch.cat([lf_inp_dec_lv1, lf_out_enc_lv1], 1)
        lf_inp_dec_lv1 = self.lf_reduce_chan_lv1(lf_inp_dec_lv1)
        lf_out_dec_lv1 = self.lf_decoder_lv1(lf_inp_dec_lv1)

        refined_lf = a + self.lf_refinement(lf_out_dec_lv1)

        # refine HF subbands 📈
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim * 2, self.dim)
        d = haar.narrow(1, self.dim * 3, self.dim)
        x_hf = torch.cat([h, v, d], 1)

        ##### encoder #####
        hf_out_enc_lv1 = self.hf_encoder_lv1(x_hf)

        hf_inp_enc_lv2 = self.hf_down1_2(hf_out_enc_lv1)
        hf_out_enc_lv2 = self.hf_encoder_lv2(hf_inp_enc_lv2)

        hf_inp_enc_lv3 = self.hf_down2_3(hf_out_enc_lv2)
        hf_out_enc_lv3 = self.hf_encoder_lv3(hf_inp_enc_lv3)

        ##### bottleneck #####
        hf_latent = self.hf_latent(hf_out_enc_lv3)

        ##### decoder #####
        hf_inp_dec_lv3 = torch.cat([hf_latent, hf_out_enc_lv3], 1)  # concat skip
        hf_inp_dec_lv3 = self.hf_reduce_chan_lv3(hf_inp_dec_lv3)  # skip 채널 수 축소
        hf_out_dec_lv3 = self.hf_decoder_lv3(hf_inp_dec_lv3)

        hf_inp_dec_lv2 = self.hf_up3_2(hf_out_dec_lv3)
        hf_inp_dec_lv2 = torch.cat([hf_inp_dec_lv2, hf_out_enc_lv2], 1)
        hf_inp_dec_lv2 = self.hf_reduce_chan_lv2(hf_inp_dec_lv2)
        hf_out_dec_lv2 = self.hf_decoder_lv2(hf_inp_dec_lv2)

        hf_inp_dec_lv1 = self.hf_up2_1(hf_out_dec_lv2)
        hf_inp_dec_lv1 = torch.cat([hf_inp_dec_lv1, hf_out_enc_lv1], 1)
        hf_inp_dec_lv1 = self.hf_reduce_chan_lv1(hf_inp_dec_lv1)
        hf_out_dec_lv1 = self.hf_decoder_lv1(hf_inp_dec_lv1)

        refined_hf = x_hf + self.hf_refinement(hf_out_dec_lv1)

        # merge LF and HF subbands 🖇️
        restored = self.wavelet_transform(
            torch.cat([refined_lf, refined_hf], 1), rev=True
        )
        restored = self.last_refinement(restored)
        restored = self.output(restored) + inp_img

        return restored


class WaveBFRAir(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=16,  # 임베딩 채널 수
        num_blocks=[1, 1, 2],  # encoder/decoder 각 계층의 Transformer 블록 수
        heads=[1, 2, 4],  # 각 계층 내 Transformer의 multi-head 개수 정의
        ffn_expansion_factor=2.66,  # FFN 블록의 hidden 채널 확장 비율
        bias=False,  # attention 연산에 쓰이는 conv 레이어의 bias 사용 여부
        LayerNorm_type="WithBias",  ## Other option 'BiasFree'
    ):

        print("🏷️ WaveBFR Air")
        super(WaveBFRAir, self).__init__()

        self.inp_channels = inp_channels
        self.dim = dim

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.wavelet_transform = HaarWavelet(dim, grad=False)

        # LF subbands branch 🌊
        self.lf_encoder_lv1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.lf_down1_2 = Downsample(dim)

        self.lf_encoder_lv2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.lf_down2_3 = Downsample(int(dim * 2**1))

        self.lf_latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.lf_up3_2 = Upsample(int(dim * 2**2))
        self.lf_reduce_chan_lv2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.lf_decoder_lv2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.lf_up2_1 = Upsample(int(dim * 2**1))
        self.lf_reduce_chan_lv1 = nn.Conv2d(
            int(dim * 2**1), int(dim), kernel_size=1, bias=bias
        )
        self.lf_decoder_lv1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        # HF subbands branch ☀️
        self.hf_refine = nn.Sequential(
            LightweightHFBlock(dim * 3),
            LightweightHFBlock(dim * 3),
        )

        # image-level 🖇️
        self.output = nn.Sequential(
            nn.Conv2d(
                int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
            )
        )

    def forward(self, inp_img):

        x = self.patch_embed(inp_img)
        haar = self.wavelet_transform(x, rev=False)

        # refine LF subband 🌊
        a = haar.narrow(1, 0, self.dim)

        ##### encoder #####
        lf_out_enc_lv1 = self.lf_encoder_lv1(a)
        lf_inp_enc_lv2 = self.lf_down1_2(lf_out_enc_lv1)

        lf_out_enc_lv2 = self.lf_encoder_lv2(lf_inp_enc_lv2)
        lf_inp_enc_lv3 = self.lf_down2_3(lf_out_enc_lv2)

        ##### bottleneck #####
        lf_latent = self.lf_latent(lf_inp_enc_lv3)

        ##### decoder #####
        lf_inp_dec_lv2 = self.lf_up3_2(lf_latent)
        lf_inp_dec_lv2 = torch.cat([lf_inp_dec_lv2, lf_out_enc_lv2], 1)
        lf_inp_dec_lv2 = self.lf_reduce_chan_lv2(lf_inp_dec_lv2)
        lf_out_dec_lv2 = self.lf_decoder_lv2(lf_inp_dec_lv2)

        lf_inp_dec_lv1 = self.lf_up2_1(lf_out_dec_lv2)
        lf_inp_dec_lv1 = torch.cat([lf_inp_dec_lv1, lf_out_enc_lv1], 1)
        lf_inp_dec_lv1 = self.lf_reduce_chan_lv1(lf_inp_dec_lv1)
        lf_out = self.lf_decoder_lv1(lf_inp_dec_lv1)

        # refine HF subbands ☀️
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim * 2, self.dim)
        d = haar.narrow(1, self.dim * 3, self.dim)
        x_hf = torch.cat([h, v, d], 1)

        hf_out = self.hf_refine(x_hf)

        # merge LF and HF subbands 🖇️
        restored = self.wavelet_transform(torch.cat([lf_out, hf_out], 1), rev=True)
        restored = self.output(restored) + inp_img

        return restored


class WaveBFRBreeze(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=16,  # 임베딩 채널 수
        num_blocks=[1, 2, 2],  # encoder/decoder 각 계층의 Transformer 블록 수
        heads=[1, 2, 4],  # 각 계층 내 Transformer의 multi-head 개수 정의
        ffn_expansion_factor=2.66,  # FFN 블록의 hidden 채널 확장 비율
        bias=False,  # attention 연산에 쓰이는 conv 레이어의 bias 사용 여부
        LayerNorm_type="WithBias",  ## Other option 'BiasFree'
    ):

        print("🏷️ WaveBFR Breeze")
        super(WaveBFRBreeze, self).__init__()

        self.inp_channels = inp_channels
        self.dim = dim

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.wavelet_transform = HaarWavelet(dim, grad=False)

        # LF subbands network 🌊
        self.lf_encoder_lv1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.lf_down1_2 = Downsample(dim)

        self.lf_encoder_lv2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.lf_down2_3 = Downsample(int(dim * 2**1))

        self.lf_latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.lf_up3_2 = Upsample(int(dim * 2**2))
        self.lf_reduce_chan_lv2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.lf_decoder_lv2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.lf_up2_1 = Upsample(int(dim * 2**1))
        self.lf_reduce_chan_lv1 = nn.Conv2d(
            int(dim * 2**1), int(dim), kernel_size=1, bias=bias
        )
        self.lf_decoder_lv1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        # HF subbands network ☀️
        self.hf_project_in = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=bias)
        self.hf_encoder_lv1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.hf_down1_2 = Downsample(dim)

        self.hf_encoder_lv2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.hf_down2_3 = Downsample(int(dim * 2**1))

        self.hf_latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.hf_up3_2 = Upsample(int(dim * 2**2))
        self.hf_reduce_chan_lv2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.hf_decoder_lv2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.hf_up2_1 = Upsample(int(dim * 2**1))
        self.hf_reduce_chan_lv1 = nn.Conv2d(
            int(dim * 2**1), int(dim), kernel_size=1, bias=bias
        )
        self.hf_decoder_lv1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.hf_project_out = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        # image-level 🖇️
        self.output = nn.Sequential(
            nn.Conv2d(
                int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
            )
        )

    def forward(self, inp_img):

        x = self.patch_embed(inp_img)
        haar = self.wavelet_transform(x, rev=False)

        # refine LF subband 🌊
        a = haar.narrow(1, 0, self.dim)

        ##### encoder #####
        lf_out_enc_lv1 = self.lf_encoder_lv1(a)
        lf_inp_enc_lv2 = self.lf_down1_2(lf_out_enc_lv1)

        lf_out_enc_lv2 = self.lf_encoder_lv2(lf_inp_enc_lv2)
        lf_inp_enc_lv3 = self.lf_down2_3(lf_out_enc_lv2)

        ##### bottleneck #####
        lf_latent = self.lf_latent(lf_inp_enc_lv3)

        ##### decoder #####
        lf_inp_dec_lv2 = self.lf_up3_2(lf_latent)
        lf_inp_dec_lv2 = torch.cat([lf_inp_dec_lv2, lf_out_enc_lv2], 1)
        lf_inp_dec_lv2 = self.lf_reduce_chan_lv2(lf_inp_dec_lv2)
        lf_out_dec_lv2 = self.lf_decoder_lv2(lf_inp_dec_lv2)

        lf_inp_dec_lv1 = self.lf_up2_1(lf_out_dec_lv2)
        lf_inp_dec_lv1 = torch.cat([lf_inp_dec_lv1, lf_out_enc_lv1], 1)
        lf_inp_dec_lv1 = self.lf_reduce_chan_lv1(lf_inp_dec_lv1)
        lf_out_dec_lv1 = self.lf_decoder_lv1(lf_inp_dec_lv1)

        # refine HF subbands ☀️
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim * 2, self.dim)
        d = haar.narrow(1, self.dim * 3, self.dim)
        x_hf = self.hf_project_in(torch.cat([h, v, d], 1))

        ##### encoder #####
        hf_out_enc_lv1 = self.hf_encoder_lv1(x_hf)
        hf_inp_enc_lv2 = self.hf_down1_2(hf_out_enc_lv1)

        hf_out_enc_lv2 = self.hf_encoder_lv2(hf_inp_enc_lv2)
        hf_inp_enc_lv3 = self.hf_down2_3(hf_out_enc_lv2)

        ##### bottleneck #####
        hf_latent = self.hf_latent(hf_inp_enc_lv3)

        ##### decoder #####
        hf_inp_dec_lv2 = self.hf_up3_2(hf_latent)
        hf_inp_dec_lv2 = torch.cat([hf_inp_dec_lv2, hf_out_enc_lv2], 1)
        hf_inp_dec_lv2 = self.hf_reduce_chan_lv2(hf_inp_dec_lv2)
        hf_out_dec_lv2 = self.hf_decoder_lv2(hf_inp_dec_lv2)

        hf_inp_dec_lv1 = self.hf_up2_1(hf_out_dec_lv2)
        hf_inp_dec_lv1 = torch.cat([hf_inp_dec_lv1, hf_out_enc_lv1], 1)
        hf_inp_dec_lv1 = self.hf_reduce_chan_lv1(hf_inp_dec_lv1)
        hf_out_dec_lv1 = self.hf_decoder_lv1(hf_inp_dec_lv1)
        
        hf_out_dec_lv1 = self.hf_project_out(hf_out_dec_lv1)

        # merge LF and HF subbands 🖇️
        restored = self.wavelet_transform(
            torch.cat([lf_out_dec_lv1, hf_out_dec_lv1], 1), rev=True
        )
        restored = self.output(restored) + inp_img

        return restored


class MultiFrameWaveBFRBreeze(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=16,  # 임베딩 채널 수
        num_blocks=[1, 2, 2],  # encoder/decoder 각 계층의 Transformer 블록 수
        heads=[1, 2, 4],  # 각 계층 내 Transformer의 multi-head 개수 정의
        ffn_expansion_factor=2.66,  # FFN 블록의 hidden 채널 확장 비율
        bias=False,  # attention 연산에 쓰이는 conv 레이어의 bias 사용 여부
        LayerNorm_type="WithBias",  ## Other option 'BiasFree'
    ):

        print("🏷️ Multi-Frame WaveBFR Breeze")
        super(MultiFrameWaveBFRBreeze, self).__init__()

        self.inp_channels = inp_channels
        self.dim = dim

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        # 요청한 추가 shallow Transformer block
        self.shallow_refine = ResidualShallowEncoder(
            dim=dim,
            heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

        # 이전 프레임 feature 누적
        self.feature_accumulator = FeatureAccumulator(
            dim=dim,
            bias=bias,
        )

        # 마지막 reference와 누적 feature 결합
        self.reference_fusion = ReferenceResidualFusion(
            dim=dim,
            bias=bias,
        )
        
        self.wavelet_transform = HaarWavelet(dim, grad=False)

        # LF subbands network 🌊
        self.lf_encoder_lv1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.lf_down1_2 = Downsample(dim)

        self.lf_encoder_lv2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.lf_down2_3 = Downsample(int(dim * 2**1))

        self.lf_latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.lf_up3_2 = Upsample(int(dim * 2**2))
        self.lf_reduce_chan_lv2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.lf_decoder_lv2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.lf_up2_1 = Upsample(int(dim * 2**1))
        self.lf_reduce_chan_lv1 = nn.Conv2d(
            int(dim * 2**1), int(dim), kernel_size=1, bias=bias
        )
        self.lf_decoder_lv1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )

        # HF subbands network ☀️
        self.hf_project_in = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=bias)
        self.hf_encoder_lv1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.hf_down1_2 = Downsample(dim)

        self.hf_encoder_lv2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )
        self.hf_down2_3 = Downsample(int(dim * 2**1))

        self.hf_latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[2])
            ]
        )

        self.hf_up3_2 = Upsample(int(dim * 2**2))
        self.hf_reduce_chan_lv2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )
        self.hf_decoder_lv2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[1])
            ]
        )

        self.hf_up2_1 = Upsample(int(dim * 2**1))
        self.hf_reduce_chan_lv1 = nn.Conv2d(
            int(dim * 2**1), int(dim), kernel_size=1, bias=bias
        )
        self.hf_decoder_lv1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=int(dim),
                    num_heads=heads[0],
                    ffn_expansion_factor=ffn_expansion_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks[0])
            ]
        )
        self.hf_project_out = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)

        # image-level 🖇️
        self.output = nn.Sequential(
            nn.Conv2d(
                int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
            )
        )

    def encode_shallow(self, image):
        """
        Args:
            image: [B, 3, H, W]

        Returns:
            feature: [B, dim, H, W]
        """

        feature = self.patch_embed(image)
        feature = self.shallow_refine(feature)

        return feature
    
    def accumulate_multi_frame_features(
        self,
        frames,
        frame_mask,
    ):
        """
        Args:
            frames:
                [B, T, 3, H, W]

            frame_mask:
                [B, T]
                유효 프레임이면 True.
                현재 데이터셋에서는 앞부분만 True이고,
                나머지 뒤쪽은 padding False.

        Returns:
            fused_feature:
                [B, dim, H, W]

            reference_image:
                [B, 3, H, W]
                각 sample의 마지막 유효 LR 프레임
        """

        if frames.ndim != 5:
            raise ValueError(
                f"frames must have shape [B,T,C,H,W], "
                f"but got {tuple(frames.shape)}"
            )

        if frame_mask.ndim != 2:
            raise ValueError(
                f"frame_mask must have shape [B,T], "
                f"but got {tuple(frame_mask.shape)}"
            )

        batch_size, num_slots, channels, height, width = frames.shape

        frame_mask = frame_mask.bool()

        num_valid_frames = frame_mask.long().sum(dim=1)

        if torch.any(num_valid_frames == 0):
            raise ValueError(
                "Every sample must contain at least one valid frame."
            )

        last_indices = num_valid_frames - 1

        # --------------------------------------------------------------
        # 모든 frame을 한 번에 shallow encode
        # --------------------------------------------------------------

        flat_frames = frames.reshape(
            batch_size * num_slots,
            channels,
            height,
            width,
        )

        flat_features = self.encode_shallow(flat_frames)

        _, feature_channels, feature_height, feature_width = (
            flat_features.shape
        )

        features = flat_features.reshape(
            batch_size,
            num_slots,
            feature_channels,
            feature_height,
            feature_width,
        )

        batch_indices = torch.arange(
            batch_size,
            device=frames.device,
        )

        # 각 sample의 마지막 유효 frame
        reference_feature = features[
            batch_indices,
            last_indices,
        ]

        reference_image = frames[
            batch_indices,
            last_indices,
        ]

        # --------------------------------------------------------------
        # 마지막 프레임을 제외한 auxiliary frame 누적
        # --------------------------------------------------------------

        memory = torch.zeros_like(reference_feature)

        has_memory = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=frames.device,
        )

        for frame_index in range(num_slots):
            current_feature = features[:, frame_index]

            # 현재 위치가 유효하면서 마지막 reference보다 앞선 경우
            is_auxiliary = (
                frame_mask[:, frame_index]
                & (frame_index < last_indices)
            )

            if not torch.any(is_auxiliary):
                continue

            update_mask = is_auxiliary.view(
                batch_size,
                1,
                1,
                1,
            )

            first_frame_mask = (
                is_auxiliary & (~has_memory)
            ).view(
                batch_size,
                1,
                1,
                1,
            )

            # 첫 auxiliary frame은 그대로 memory 초기값으로 사용
            memory = torch.where(
                first_frame_mask,
                current_feature,
                memory,
            )

            # 이미 memory가 있는 sample만 recurrent update
            recurrent_mask = (
                is_auxiliary & has_memory
            ).view(
                batch_size,
                1,
                1,
                1,
            )

            updated_memory = self.feature_accumulator(
                memory,
                current_feature,
            )

            memory = torch.where(
                recurrent_mask,
                updated_memory,
                memory,
            )

            has_memory = has_memory | is_auxiliary

        # auxiliary가 없는 sample은 memory=0
        # 현재 데이터셋은 최소 5장이므로 실제로는 항상 memory가 존재함.
        fused_feature = self.reference_fusion(
            reference_feature,
            memory,
        )

        return fused_feature, reference_image

    def restore_from_feature(
        self,
        feature,
        reference_image,
    ):
        """
        Args:
            feature:
                fusion이 완료된 dim-channel feature

            reference_image:
                마지막 LR 프레임.
                최종 residual connection에 사용.

        Returns:
            restored image
        """

        haar = self.wavelet_transform(
            feature,
            rev=False,
        )

        # ==============================================================
        # LF branch
        # ==============================================================

        a = haar.narrow(
            1,
            0,
            self.dim,
        )

        lf_out_enc_lv1 = self.lf_encoder_lv1(a)
        lf_inp_enc_lv2 = self.lf_down1_2(lf_out_enc_lv1)

        lf_out_enc_lv2 = self.lf_encoder_lv2(lf_inp_enc_lv2)
        lf_inp_enc_lv3 = self.lf_down2_3(lf_out_enc_lv2)

        lf_latent = self.lf_latent(lf_inp_enc_lv3)

        lf_inp_dec_lv2 = self.lf_up3_2(lf_latent)

        lf_inp_dec_lv2 = torch.cat(
            [lf_inp_dec_lv2, lf_out_enc_lv2],
            dim=1,
        )

        lf_inp_dec_lv2 = self.lf_reduce_chan_lv2(
            lf_inp_dec_lv2
        )

        lf_out_dec_lv2 = self.lf_decoder_lv2(
            lf_inp_dec_lv2
        )

        lf_inp_dec_lv1 = self.lf_up2_1(
            lf_out_dec_lv2
        )

        lf_inp_dec_lv1 = torch.cat(
            [lf_inp_dec_lv1, lf_out_enc_lv1],
            dim=1,
        )

        lf_inp_dec_lv1 = self.lf_reduce_chan_lv1(
            lf_inp_dec_lv1
        )

        lf_out_dec_lv1 = self.lf_decoder_lv1(
            lf_inp_dec_lv1
        )

        # ==============================================================
        # HF branch
        # ==============================================================

        h = haar.narrow(
            1,
            self.dim,
            self.dim,
        )

        v = haar.narrow(
            1,
            self.dim * 2,
            self.dim,
        )

        d = haar.narrow(
            1,
            self.dim * 3,
            self.dim,
        )

        x_hf = self.hf_project_in(
            torch.cat([h, v, d], dim=1)
        )

        hf_out_enc_lv1 = self.hf_encoder_lv1(x_hf)
        hf_inp_enc_lv2 = self.hf_down1_2(hf_out_enc_lv1)

        hf_out_enc_lv2 = self.hf_encoder_lv2(hf_inp_enc_lv2)
        hf_inp_enc_lv3 = self.hf_down2_3(hf_out_enc_lv2)

        hf_latent = self.hf_latent(hf_inp_enc_lv3)

        hf_inp_dec_lv2 = self.hf_up3_2(hf_latent)

        hf_inp_dec_lv2 = torch.cat(
            [hf_inp_dec_lv2, hf_out_enc_lv2],
            dim=1,
        )

        hf_inp_dec_lv2 = self.hf_reduce_chan_lv2(
            hf_inp_dec_lv2
        )

        hf_out_dec_lv2 = self.hf_decoder_lv2(
            hf_inp_dec_lv2
        )

        hf_inp_dec_lv1 = self.hf_up2_1(
            hf_out_dec_lv2
        )

        hf_inp_dec_lv1 = torch.cat(
            [hf_inp_dec_lv1, hf_out_enc_lv1],
            dim=1,
        )

        hf_inp_dec_lv1 = self.hf_reduce_chan_lv1(
            hf_inp_dec_lv1
        )

        hf_out_dec_lv1 = self.hf_decoder_lv1(
            hf_inp_dec_lv1
        )

        hf_out_dec_lv1 = self.hf_project_out(
            hf_out_dec_lv1
        )

        # ==============================================================
        # LF/HF merge
        # ==============================================================

        restored_feature = self.wavelet_transform(
            torch.cat(
                [lf_out_dec_lv1, hf_out_dec_lv1],
                dim=1,
            ),
            rev=True,
        )

        restored = (
            self.output(restored_feature)
            + reference_image
        )

        return restored


    def forward(
        self,
        inp_img,
        frame_mask=None,
    ):
        """
        Single-frame:
            inp_img: [B, 3, H, W]

        Multi-frame:
            inp_img:    [B, T, 3, H, W]
            frame_mask: [B, T]
        """

        # --------------------------------------------------------------
        # 기존 single-frame 사용
        # --------------------------------------------------------------
        if inp_img.ndim == 4:
            feature = self.encode_shallow(inp_img)

            return self.restore_from_feature(
                feature=feature,
                reference_image=inp_img,
            )

        # --------------------------------------------------------------
        # Multi-frame 사용
        # --------------------------------------------------------------
        if inp_img.ndim == 5:
            if frame_mask is None:
                frame_mask = torch.ones(
                    inp_img.shape[:2],
                    dtype=torch.bool,
                    device=inp_img.device,
                )

            fused_feature, reference_image = (
                self.accumulate_multi_frame_features(
                    frames=inp_img,
                    frame_mask=frame_mask,
                )
            )

            return self.restore_from_feature(
                feature=fused_feature,
                reference_image=reference_image,
            )

        raise ValueError(
            f"Unsupported input shape: {tuple(inp_img.shape)}"
        )
