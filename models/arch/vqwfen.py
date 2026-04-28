from models.arch.blocks import *
from torch import nn
from vector_quantize_pytorch import VectorQuantize

from models.arch.wfen import FDT


class VQWFEN(nn.Module):
    def __init__(
        self,
        inchannel=3,
        min_ch=40,
        is_pretrain=False,
    ):
        super(VQWFEN, self).__init__()

        self.is_pretrain = is_pretrain

        self.first_conv = nn.Conv2d(inchannel, min_ch, kernel_size=3, padding=1)

        self.Downsample1 = nn.Sequential(
            nn.Conv2d(
                min_ch, min_ch // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )
        self.Downsample2 = nn.Sequential(
            nn.Conv2d(
                min_ch * 2,
                min_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelUnshuffle(2),
        )
        self.Downsample3 = nn.Sequential(
            nn.Conv2d(
                min_ch * 4,
                min_ch * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelUnshuffle(2),
        )

        self.TransformerDown1 = nn.Sequential(
            FDT(inp_channels=min_ch, window_sizes=8, shifts=0, num_heads=4),
            FDT(inp_channels=min_ch, window_sizes=8, shifts=0, num_heads=4),
        )
        self.TransformerDown2 = nn.Sequential(
            FDT(inp_channels=min_ch * 2, window_sizes=4, shifts=1, num_heads=4),
        )
        self.TransformerDown3 = nn.Sequential(
            FDT(inp_channels=min_ch * 4, window_sizes=2, shifts=0, num_heads=8),
        )

        # 🧩 Codebook + VQ
        self.VQ = VectorQuantize(
            dim=min_ch * 8,
            codebook_size=1024,
            decay=0.8,
            commitment_weight=1.0,
            accept_image_fmap=True,
            freeze_codebook=not is_pretrain,
        )
        if is_pretrain:
            print("🍿 pretraining VQ-WFEN")
        else:
            print("🌽 fine-tuning VQ-WFEN")

        self.TransformerUp1 = nn.Sequential(
            FDT(inp_channels=min_ch * 4, window_sizes=8, shifts=1, num_heads=8),
        )
        self.TransformerUp2 = nn.Sequential(
            FDT(inp_channels=min_ch * 2, window_sizes=4, shifts=0, num_heads=4),
        )
        self.TransformerUp3 = nn.Sequential(
            FDT(inp_channels=min_ch, window_sizes=2, shifts=1, num_heads=4),
            FDT(inp_channels=min_ch, window_sizes=2, shifts=0, num_heads=4),
        )

        self.Upsample1 = nn.Sequential(
            nn.Conv2d(
                min_ch * 8,
                min_ch * 16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelShuffle(2),
        )
        self.Upsample2 = nn.Sequential(
            nn.Conv2d(
                min_ch * 4,
                min_ch * 8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelShuffle(2),
        )
        self.Upsample3 = nn.Sequential(
            nn.Conv2d(
                min_ch * 2,
                min_ch * 4,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelShuffle(2),
        )
        
        self.fuse_conv1 = nn.Conv2d(min_ch * 16, min_ch * 8, kernel_size=3, padding=1)
        self.fuse_conv2 = nn.Conv2d(min_ch * 8, min_ch * 4, kernel_size=3, padding=1)
        self.fuse_conv3 = nn.Conv2d(min_ch * 4, min_ch * 2, kernel_size=3, padding=1)

        self.out_conv = nn.Conv2d(min_ch, inchannel, kernel_size=3, padding=1)

    def forward(self, input_img):
        x_first = self.first_conv(input_img)

        # encoder
        x_skip1 = self.TransformerDown1(x_first)  # (128, 128, 40)
        x_down1 = self.Downsample1(x_skip1)  # (64, 64, 80)

        x_skip2 = self.TransformerDown2(x_down1)  # (64, 64, 80)
        x_down2 = self.Downsample2(x_skip2)  # (32, 32, 160)

        x_skip3 = self.TransformerDown3(x_down2)  # (32, 32, 160)
        x_down3 = self.Downsample3(x_skip3)  # (16, 16, 320)

        # bottleneck
        quantized, _indices, commit_loss = self.VQ(x_down3)  # VQ

        # decoder (use skip connection only when fine-tuning)
        x_up1 = self.Upsample1(quantized)
        if self.is_pretrain:
            x_up1 = self.TransformerUp1(x_up1)  # (32, 32, 160)
        else:
            x_up1 = self.TransformerUp1(x_up1 + x_skip3)  # (32, 32, 160)
        
        x_up2 = self.Upsample2(x_up1)
        if self.is_pretrain:
            x_up2 = self.TransformerUp2(x_up2)  # (64, 64, 80)
        else:
            x_up2 = self.TransformerUp2(x_up2 + x_skip2)  # (64, 64, 80)

        x_up3 = self.Upsample3(x_up2)
        if self.is_pretrain:
            x_up3 = self.TransformerUp3(x_up3)  # (128, 128, 40)
        else:
            x_up3 = self.TransformerUp3(x_up3 + x_skip1)  # (128, 128, 40)

        out_img = self.out_conv(x_up3 + x_first)

        return out_img, commit_loss
