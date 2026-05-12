from models.arch.blocks import *
from torch import nn
from vector_quantize_pytorch import VectorQuantize

from models.arch.wfen import WFD, WFU, FDT


class WFU_LFSkip(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(WFU_LFSkip, self).__init__()
        self.RB = nn.Sequential(
            nn.Conv2d(in_dim * 2, out_dim * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.PixelShuffle(2),
        )

    def forward(self, x_lf, x_small):
        out = torch.cat([x_lf, x_small], dim=1)
        out = self.RB(out)

        return out

class VQWFEN(nn.Module):
    def __init__(
        self,
        inchannel=3,
        min_ch=40,
        res_depth=6,
        is_pretrain=False,
    ):
        super(VQWFEN, self).__init__()
        
        self.is_pretrain = is_pretrain
        
        self.first_conv = nn.Conv2d(inchannel, min_ch, kernel_size=3, padding=1)

        self.HaarDownsample1 = WFD(min_ch, min_ch * 2, True)
        self.HaarDownsample2 = WFD(min_ch * 2, min_ch * 4, True)
        self.HaarDownsample3 = WFD(min_ch * 4, min_ch * 4, True)

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

        self.RB1 = nn.Sequential(
            nn.Conv2d(min_ch * 2, min_ch * 2, kernel_size=1, padding=0, groups=1),
            nn.ReLU(),
            nn.Conv2d(min_ch * 2, min_ch * 2, kernel_size=1, padding=0, groups=1),
        )
        self.RB2 = nn.Sequential(
            nn.Conv2d(
                min_ch * 4, min_ch * 4, kernel_size=1, padding=0, groups=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                min_ch * 4, min_ch * 4, kernel_size=1, padding=0, groups=1
            ),
        )
        self.RB3 = nn.Sequential(
            nn.Conv2d(
                min_ch * 4, min_ch * 4, kernel_size=1, padding=0, groups=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                min_ch * 4, min_ch * 4, kernel_size=1, padding=0, groups=1
            ),
        )

        self.Transformer0 = FDT(
            inp_channels=min_ch * 4, window_sizes=1, shifts=0, num_heads=8
        )
        Transformer = []
        for i in range(res_depth - 1):
            (
                Transformer.append(
                    FDT(
                        inp_channels=min_ch * 4,
                        window_sizes=1,
                        shifts=1,
                        num_heads=8
                    ),
                )
                if i % 2 == 0
                else Transformer.append(
                    FDT(
                        inp_channels=min_ch * 4,
                        window_sizes=1,
                        shifts=0,
                        num_heads=8
                    )
                )
            )
        self.Transformer = nn.Sequential(*Transformer)

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

        self.HaarFeatureFusion1 = WFU_LFSkip(min_ch * 4, min_ch * 4)
        self.HaarFeatureFusion2 = WFU_LFSkip(min_ch * 4, min_ch * 2)
        self.HaarFeatureFusion3 = WFU_LFSkip(min_ch * 2, min_ch)
        
        self.vq1 = VectorQuantize(dim=min_ch * 4, codebook_size=512, decay=0.95, accept_image_fmap=True, freeze_codebook=not self.is_pretrain)
        self.vq2 = VectorQuantize(dim=min_ch * 4, codebook_size=512, decay=0.95, accept_image_fmap=True, freeze_codebook=not self.is_pretrain)
        self.vq3 = VectorQuantize(dim=min_ch * 2, codebook_size=512, decay=0.95, accept_image_fmap=True, freeze_codebook=not self.is_pretrain)
        
        self.out_conv = nn.Conv2d(min_ch, inchannel, kernel_size=3, padding=1)

    def forward(self, input_img):
        x_first = self.first_conv(input_img)  # 첫 conv

        ############ encoder ############
        x1 = self.TransformerDown1(x_first)  # hw:128  c:40
        x1_a, x1_hvd = self.HaarDownsample1(x1)  # all hw:64 c:80

        x2 = self.TransformerDown2(x1_a)  # hw:64  c:80
        x1_hvd_enhanced = self.RB1(x1_hvd)
        x2 = x2 + x1_hvd_enhanced  # hw:64  c:160
        x2_a, x2_hvd = self.HaarDownsample2(x2)  # all hw:40 c:160

        x3 = self.TransformerDown3(x2_a)  # hw:40  c:160
        x2_hvd_enhanced = self.RB2(x2_hvd)
        x3 = x3 + x2_hvd_enhanced
        x3_a, x3_hvd = self.HaarDownsample3(x3)  # all hw:16 c:160
        ############ encoder ############

        x_trans0 = self.Transformer0(x3_a)  # hw:16 c:160
        x_trans = self.Transformer(x_trans0)  # hw:16 c:160
        x3_hvd_enhanced = self.RB3(x3_hvd)
        x_trans = x_trans + x3_hvd_enhanced  # hw:16 c:160

        ############ decoder ############
        x3_hvd_quant, _, commit_loss1 = self.vq1(x3_hvd_enhanced)
        x_up1 = self.HaarFeatureFusion1(x3_hvd_quant, x_trans)  # hw:40 c:160
        x_1 = self.TransformerUp1(x_up1)  # hw:40 c:160

        x2_hvd_quant, _, commit_loss2 = self.vq2(x2_hvd_enhanced)
        x_up2 = self.HaarFeatureFusion2(x2_hvd_quant, x_1)  # hw:64 c:80
        x_2 = self.TransformerUp2(x_up2)  # hw:64 c:80

        x1_hvd_quant, _, commit_loss3 = self.vq3(x1_hvd_enhanced)
        x_up3 = self.HaarFeatureFusion3(x1_hvd_quant, x_2)  # hw:128 c:40
        x_3 = self.TransformerUp3(x_up3)  # hw:32 c:40
        ############ decoder ############

        if self.is_pretrain:
            out_img = self.out_conv(x_3)
        else:
            out_img = self.out_conv(x_3 + x_first)
        
        commit_loss_total = commit_loss1 + commit_loss2 + commit_loss3

        return out_img, commit_loss_total
