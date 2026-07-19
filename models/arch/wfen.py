import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import numbers


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
# ---------------------------------------------------------------------------------------------------------------------


# FFN
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, input_resolution=None):
        super(FeedForward, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """
    def __init__(self, channels, normalize_shape=None, norm_type='bn'):
        super(NormLayer, self).__init__()
        norm_type = norm_type.lower()
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, channels, affine=True)
        elif norm_type == 'pixel':
            self.norm = lambda x: F.normalize(x, p=2, dim=1)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(normalize_shape)
        elif norm_type == 'none':
            self.norm = lambda x: x
        else:
            assert 1==0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        return self.norm(x)


class ReluLayer(nn.Module):
    """Relu Layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu 
            - SELU
            - none: direct pass
    """
    def __init__(self, channels, relu_type='relu'):
        super(ReluLayer, self).__init__()
        relu_type = relu_type.lower()
        if relu_type == 'relu':
            self.func = nn.ReLU(True)
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'prelu':
            self.func = nn.PReLU(channels)
        elif relu_type == 'selu':
            self.func = nn.SELU(True)
        elif relu_type == 'none':
            self.func = lambda x: x
        else:
            assert 1==0, 'Relu type {} not support.'.format(relu_type)

    def forward(self, x):
        return self.func(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale='none', norm_type='none', relu_type='none', use_pad=True):
        super(ConvLayer, self).__init__()
        self.use_pad = use_pad
        
        bias = True if norm_type in ['pixel', 'none'] else False 
        stride = 2 if scale == 'down' else 1

        self.scale_func = lambda x: x
        if scale == 'up':
            self.scale_func = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest')

        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2) 
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.relu = ReluLayer(out_channels, relu_type)
        self.norm = NormLayer(out_channels, norm_type=norm_type)

    def forward(self, x):
        out = self.scale_func(x)
        if self.use_pad:
            out = self.reflection_pad(out)
        out = self.conv2d(out)
        out = self.norm(out)
        out = self.relu(out)
        return out


class ResidualBlock(nn.Module):
    """
    Residual block recommended in: http://torch.ch/blog/2016/02/04/resnets.html
    ------------------
    # Args
        - hg_depth: depth of HourGlassBlock. 0: don't use attention map.
        - use_pmask: whether use previous mask as HourGlassBlock input.
    """
    def __init__(self, c_in, c_out, relu_type='prelu', norm_type='bn', scale='none', hg_depth=2, att_name='spar'):
        super(ResidualBlock, self).__init__()
        self.c_in      = c_in
        self.c_out     = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth  = hg_depth

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        self.preact_func = nn.Sequential(
                    NormLayer(c_in, norm_type=self.norm_type),
                    ReluLayer(c_in, self.relu_type),
                    )

        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']

        self.conv1 = ConvLayer(c_in, c_out, 3, scales[0], **kwargs) 
        self.conv2 = ConvLayer(c_out, c_out, 3, scales[1], norm_type=norm_type, relu_type='none')

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))

        self.att_func = HourGlassBlock(self.hg_depth, c_out, c_attn, **kwargs) 
        
    def forward(self, x):
        identity = self.shortcut_func(x)
        out = self.preact_func(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = identity + self.att_func(out)
        return out
        

class HourGlassBlock(nn.Module):
    """Simplified HourGlass block.
    Reference: https://github.com/1adrianb/face-alignment 
    --------------------------
    """
    def __init__(self, depth, c_in, c_out, 
            c_mid=64,
            norm_type='bn',
            relu_type='prelu',
            ):
        super(HourGlassBlock, self).__init__()
        self.depth     = depth
        self.c_in      = c_in
        self.c_mid     = c_mid
        self.c_out     = c_out
        self.kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        if self.depth:
            self._generate_network(self.depth)
            self.out_block = nn.Sequential(
                    ConvLayer(self.c_mid, self.c_out, norm_type='none', relu_type='none'),
                    nn.Sigmoid()
                    )
            
    def _generate_network(self, level):
        if level == self.depth:
            c1, c2 = self.c_in, self.c_mid
        else:
            c1, c2 = self.c_mid, self.c_mid

        self.add_module('b1_' + str(level), ConvLayer(c1, c2, **self.kwargs)) 
        self.add_module('b2_' + str(level), ConvLayer(c1, c2, scale='down', **self.kwargs)) 
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvLayer(self.c_mid, self.c_mid, **self.kwargs)) 

        self.add_module('b3_' + str(level), ConvLayer(self.c_mid, self.c_mid, scale='up', **self.kwargs))

    def _forward(self, level, in_x):
        up1 = self._modules['b1_' + str(level)](in_x)
        low1 = self._modules['b2_' + str(level)](in_x)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self._modules['b2_plus_' + str(level)](low1)

        up2 = self._modules['b3_' + str(level)](low2)
        if up1.shape[2:] != up2.shape[2:]:
            up2 = nn.functional.interpolate(up2, up1.shape[2:])

        return up1 + up2

    def forward(self, x, pmask=None):
        if self.depth == 0: return x
        input_x = x
        x = self._forward(self.depth, x)
        self.att_map = self.out_block(x)
        x = input_x * self.att_map
        return x


import math
class UpSampler(nn.Sequential):
    def __init__(self, scale, n_feats, out_feats):

        m = []
        if scale == 8:
            kernel_size = 3
        elif scale == 16:
            kernel_size = 5
        else:
            kernel_size = 1

        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(in_channels=n_feats, out_channels=4 * out_feats, kernel_size=kernel_size, stride=1,
                                   padding=kernel_size // 2))
                m.append(nn.PixelShuffle(upscale_factor=2))
                m.append(nn.PReLU())
        super(UpSampler, self).__init__(*m)
        

class GSA(nn.Module):
    # 전역 Self-Attention 연산 (채널 멀티헤드화 + 채널 축 어텐션)
    def __init__(self, channels, num_heads=8, bias=False):
        super(GSA, self).__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.act = nn.ReLU()

        # 1*1 point-wise convolution
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)

        # 3*3 depth-wise convolution
        self.qkv_dwconv = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels * 3,
            bias=bias,
        )

        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x, prev_atns=None):
        b, c, h, w = x.shape
        if prev_atns is None:  # 이전 어텐션 값 없을 경우
            qkv = self.qkv_dwconv(self.qkv(x))  # 1*1 conv + 3*3 conv
            q, k, v = qkv.chunk(
                3, dim=1
            )  # 3배 늘어난 채널 수를 3개씩 나눔 (= 원래 채널 수)

            # 채널 차원(c)을 멀티헤드로 쪼개고 공간 차원(h, w)을 flatten
            q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
            k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
            v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            # (b, head, c, hw) @ (b, head, hw, c) => (b, head, c, c) 어텐션 맵
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = self.act(attn)  # activation 통과 후 value와 곱함
            out = attn @ v  # => (b, head, c, hw) 결과

            # 공간 차원 역 flatten + 멀티헤드 병합
            y = rearrange(
                out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
            )
            y = rearrange(
                y, "b (head c) h w -> b (c head) h w", head=self.num_heads, h=h, w=w
            )

            y = self.project_out(y)  # 마무리

            return y, attn
        else:
            attn = prev_atns  # 이전 어텐션 값 있을 경우, 이를 어텐션 맵(query & key)으로 취급

            # 현재 입력을 value로 삼아 flatten 후 이전 어텐션 맵과 곱함
            v = rearrange(x, "b (head c) h w -> b head c (h w)", head=self.num_heads)
            out = attn @ v

            y = rearrange(
                out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
            )
            y = rearrange(
                y, "b (head c) h w -> b (c head) h w", head=self.num_heads, h=h, w=w
            )

            y = self.project_out(y)  # 마무리

            return y


class RSA(nn.Module):
    # Regional Self-Attention 연산 (윈도우 시프팅 + 채널 축 어텐션)
    def __init__(
        self, channels, num_heads, shifts=1, window_sizes=[4, 8, 12], bias=False
    ):
        super(RSA, self).__init__()
        self.channels = channels
        self.shifts = shifts
        self.window_sizes = window_sizes

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.act = nn.ReLU()

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x, prev_atns=None):
        b, c, h, w = x.shape
        if prev_atns is None:
            wsize = self.window_sizes
            x_ = x

            if self.shifts > 0:  # shifts는 0 또는 1
                # 윈도우 크기 절반 만큼 공간 차원을 대각선 왼쪽 위 방향으로 시프팅(평행이동)
                x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))

            qkv = self.qkv_dwconv(self.qkv(x_))
            q, k, v = qkv.chunk(3, dim=1)

            # 윈도우 크기(8, 4, 2 중 하나) 단위로 공간 차원(h, w)을 쪼개고, 윈도우 차원에 flatten 적용
            q = rearrange(
                q, "b c (h dh) (w dw) -> b (h w) (dh dw) c", dh=wsize, dw=wsize
            )
            k = rearrange(
                k, "b c (h dh) (w dw) -> b (h w) (dh dw) c", dh=wsize, dw=wsize
            )
            v = rearrange(
                v, "b c (h dh) (w dw) -> b (h w) (dh dw) c", dh=wsize, dw=wsize
            )

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            # (b, hw, c, dh*dw) @ (b, hw, dh*dw, c) => (b, hw, c, c) 어텐션 맵
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = self.act(attn)
            out = v @ attn  # => (b, hw, dh*dw, c) 결과

            out = rearrange(
                out,
                "b (h w) (dh dw) c-> b (c) (h dh) (w dw)",
                h=h // wsize,
                w=w // wsize,
                dh=wsize,
                dw=wsize,
            )  # 윈도우 해제, 원래 공간 차원으로 통합

            if self.shifts > 0:  # 시프팅 윈도우 어텐션 후 복구
                out = torch.roll(out, shifts=(wsize // 2, wsize // 2), dims=(2, 3))

            y = self.project_out(out)

            return y, attn
        else:
            wsize = self.window_sizes

            if self.shifts > 0:
                x = torch.roll(x, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))

            atn = prev_atns
            v = rearrange(
                x, "b (c) (h dh) (w dw) -> b (h w) (dh dw) c", dh=wsize, dw=wsize
            )
            y_ = v @ atn
            y_ = rearrange(
                y_,
                "b (h w) (dh dw) c-> b (c) (h dh) (w dw)",
                h=h // wsize,
                w=w // wsize,
                dh=wsize,
                dw=wsize,
            )

            if self.shifts > 0:
                y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))

            y = self.project_out(y_)

            return y


class FDT(nn.Module):
    # Full-Domain Transformer (RSA -> FFN -> residual -> GSA -> FFN -> residual)
    def __init__(
        self,
        inp_channels,
        window_sizes,
        shifts,
        num_heads,
        shared_depth=1,
        ffn_expansion_factor=2.66,
    ):
        super(FDT, self).__init__()
        self.shared_depth = shared_depth

        # 전체 블록 중 전반부(RSA + FFN)
        modules_ffd = {}
        modules_att = {}
        modules_norm = {}
        for i in range(shared_depth):  # 웬만하면 1인 듯. 그냥 한 번씩만 추가됨
            modules_ffd["ffd{}".format(i)] = FeedForward(
                inp_channels, ffn_expansion_factor, bias=False
            )
            modules_att["att_{}".format(i)] = RSA(
                channels=inp_channels,
                num_heads=num_heads,
                shifts=shifts,
                window_sizes=window_sizes,
            )
            modules_norm["norm_{}".format(i)] = LayerNorm(inp_channels, "WithBias")
            modules_norm["norm_{}".format(i + 2)] = LayerNorm(inp_channels, "WithBias")

        self.modules_ffd = nn.ModuleDict(modules_ffd)
        self.modules_att = nn.ModuleDict(modules_att)
        self.modules_norm = nn.ModuleDict(modules_norm)

        # 전체 블록 중 후반부(GSA + FFN)
        modulec_ffd = {}
        modulec_att = {}
        modulec_norm = {}
        for i in range(shared_depth):
            modulec_ffd["ffd{}".format(i)] = FeedForward(
                inp_channels, ffn_expansion_factor, bias=False
            )
            modulec_att["att_{}".format(i)] = GSA(
                channels=inp_channels, num_heads=num_heads
            )
            modulec_norm["norm_{}".format(i)] = LayerNorm(inp_channels, "WithBias")
            modulec_norm["norm_{}".format(i + 2)] = LayerNorm(inp_channels, "WithBias")

        self.modulec_ffd = nn.ModuleDict(modulec_ffd)
        self.modulec_att = nn.ModuleDict(modulec_att)
        self.modulec_norm = nn.ModuleDict(modulec_norm)

    def forward(self, x):
        atn = None
        B, C, H, W = x.size()
        for i in range(self.shared_depth):
            if i == 0:  ## only calculate attention for the 1-st module
                # LayerNorm -> Self-Attn -> resuidal + LayerNorm -> FFN -> residual
                x_, atn = self.modules_att["att_{}".format(i)](
                    self.modules_norm["norm_{}".format(i)](x), None
                )
                x = (
                    self.modules_ffd["ffd{}".format(i)](
                        self.modules_norm["norm_{}".format(i + 2)](x_ + x)
                    )
                    + x_
                )
            else:
                x_ = self.modules_att["att_{}".format(i)](
                    self.modules_norm["norm_{}".format(i)](x), atn
                )
                x = (
                    self.modules_ffd["ffd{}".format(i)](
                        self.modules_norm["norm_{}".format(i + 2)](x_ + x)
                    )
                    + x_
                )

        for i in range(self.shared_depth):
            if i == 0:  ## only calculate attention for the 1-st module
                x_, atn = self.modulec_att["att_{}".format(i)](
                    self.modulec_norm["norm_{}".format(i)](x), None
                )
                x = (
                    self.modulec_ffd["ffd{}".format(i)](
                        self.modulec_norm["norm_{}".format(i + 2)](x_ + x)
                    )
                    + x_
                )
            else:
                x = self.modulec_att["att_{}".format(i)](
                    self.modulec_norm["norm_{}".format(i)](x), atn
                )
                x = (
                    self.modulec_ffd["ffd{}".format(i)](
                        self.modulec_norm["norm_{}".format(i + 2)](x_ + x)
                    )
                    + x_
                )

        return x


class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=False):
        super(HaarWavelet, self).__init__()
        self.in_channels = in_channels

        self.haar_weights = torch.ones(4, 1, 2, 2)  # XX
        # XX (0번째는 평균(저주파))
        # h
        self.haar_weights[1, 0, 0, 1] = -1  # XO
        self.haar_weights[1, 0, 1, 1] = -1  # XO    (세로)
        # v
        self.haar_weights[2, 0, 1, 0] = -1  # XX
        self.haar_weights[2, 0, 1, 1] = -1  # OO    (가로)
        # d
        self.haar_weights[3, 0, 1, 0] = -1  # XO
        self.haar_weights[3, 0, 0, 1] = -1  # OX    (대각선)

        # 입력 채널 수만큼 복사(depth-wise conv 방식처럼 따로따로 계산)
        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = grad

    def forward(self, x, rev=False):  # rev는 inverse transform 여부
        if not rev:
            # kernel_size=2 & stride=2: 입력을 겹치지 않는 2*2 블록으로 쪼갬
            # groups=self.in_channels: 입력 텐서의 각 채널마다 4개의 필터를 적용하여 4개의 값 생성(B, C*4, H/2, H/2)
            out = (
                F.conv2d(
                    x, self.haar_weights, bias=None, stride=2, groups=self.in_channels
                )
                / 4.0
            )  # 4.0은 그냥 평균 맞추는 스케일러

            out = out.reshape(
                [x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2]
            )  # 4배로 늘어난 채널을 각각 분리하여 4개의 서브밴드 형성

            out = torch.transpose(
                out, 1, 2
            )  # (C, 4) -> (4, C), wavelet 종류별로 채널 묶음

            out = out.reshape(
                [x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2]
            )  # conv 연산을 위해 4개를 다시 하나로 통합

            return out
        else:
            # 역변환! 입력 형상은 (B, C*4, H', W')
            out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape(
                [x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3]]
            )

            return F.conv_transpose2d(
                out, self.haar_weights, bias=None, stride=2, groups=self.in_channels
            )  # 채널 4개를 다시 하나로, 공간 해상도는 2배로 늘어남


class WFU(nn.Module):  # Wavelet Feature Upgrade (Upsample)
    # 현재 디코더에서 처리 중인 특징(small)과
    # 인코더에서 건너온 특징(big)을 웨이블릿 역변환으로 병합
    # (중간에 HF/LF 각자 강화 과정도 거치고)
    def __init__(self, dim_big, dim_small):
        super(WFU, self).__init__()
        self.dim = dim_big
        self.HaarWavelet = HaarWavelet(dim_big, grad=False)
        self.InverseHaarWavelet = HaarWavelet(dim_big, grad=False)
        self.RB = nn.Sequential(
            nn.Conv2d(dim_big, dim_big, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim_big, dim_big, kernel_size=3, padding=1),
        )  # HF 요소 전용 residual 블록

        self.channel_tranformation = nn.Sequential(  # concat된 LF 요소 강화
            nn.Conv2d(
                dim_big + dim_small, dim_big + dim_small // 1, kernel_size=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(dim_big + dim_small // 1, dim_big * 3, kernel_size=1, padding=0),
        )

    def forward(self, x_big, x_small):
        haar = self.HaarWavelet(
            x_big, rev=False
        )  # 인코더 feature (큰 거) 웨이블릿 변환
        a = haar.narrow(1, 0, self.dim)  # LF
        h = haar.narrow(1, self.dim, self.dim)  # HF1
        v = haar.narrow(1, self.dim * 2, self.dim)  # HF2
        d = haar.narrow(1, self.dim * 3, self.dim)  # HF3

        hvd = self.RB(h + v + d)  # HF들은 그냥 더해서 residual block에 제공
        a_ = self.channel_tranformation(
            torch.cat([x_small, a], dim=1)
        )  # LF끼리는 concat해서 enhancing
        out = self.InverseHaarWavelet(
            torch.cat([hvd, a_], dim=1), rev=True
        )  # 결과 HF랑 LF랑 다시 하나로 묶어서 역변환

        return out


class WFD(nn.Module):  # Wavelet Feature Downsample
    # 특징 하나를 LF 1개랑 HF 3개로 쪼개서(해상도 절반) 각각 반환
    # 논문 그림에 나온 HR 요소들의 residual은 여기 바깥에서 진행함(RB 블록들)
    def __init__(self, dim_in, dim, need=False):
        super(WFD, self).__init__()
        self.need = need
        if need:
            self.first_conv = nn.Conv2d(
                dim_in, dim, kernel_size=1, padding=0
            )  # 첫 conv
            self.HaarWavelet = HaarWavelet(dim, grad=False)
            self.dim = dim
        else:
            self.HaarWavelet = HaarWavelet(dim_in, grad=False)
            self.dim = dim_in

    def forward(self, x):
        if self.need:
            x = self.first_conv(x)

        haar = self.HaarWavelet(x, rev=False)
        a = haar.narrow(1, 0, self.dim)
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim * 2, self.dim)
        d = haar.narrow(1, self.dim * 3, self.dim)

        return a, h + v + d


class WFEN(nn.Module):
    def __init__(
        self,
        inchannel=3,
        min_ch=40,
        res_depth=6,
    ):
        super(WFEN, self).__init__()
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
            nn.Conv2d(min_ch * 4, min_ch * 4, kernel_size=1, padding=0, groups=1),
            nn.ReLU(),
            nn.Conv2d(min_ch * 4, min_ch * 4, kernel_size=1, padding=0, groups=1),
        )
        self.RB3 = nn.Sequential(
            nn.Conv2d(min_ch * 4, min_ch * 4, kernel_size=1, padding=0, groups=1),
            nn.ReLU(),
            nn.Conv2d(min_ch * 4, min_ch * 4, kernel_size=1, padding=0, groups=1),
        )

        self.Transformer0 = FDT(
            inp_channels=min_ch * 4, window_sizes=1, shifts=0, num_heads=8
        )
        Transformer = []
        for i in range(res_depth - 1):
            (
                Transformer.append(
                    FDT(inp_channels=min_ch * 4, window_sizes=1, shifts=1, num_heads=8),
                )
                if i % 2 == 0
                else Transformer.append(
                    FDT(inp_channels=min_ch * 4, window_sizes=1, shifts=0, num_heads=8)
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

        self.HaarFeatureFusion1 = WFU(min_ch * 4, min_ch * 4)
        self.HaarFeatureFusion2 = WFU(min_ch * 2, min_ch * 4)
        self.HaarFeatureFusion3 = WFU(min_ch, min_ch * 2)

        self.out_conv = nn.Conv2d(min_ch, inchannel, kernel_size=3, padding=1)

    def forward(self, input_img):
        x_first = self.first_conv(input_img)  # 첫 conv

        ############ encoder ############
        x1 = self.TransformerDown1(x_first)  # hw:128  c:40
        x1_a, x1_hvd = self.HaarDownsample1(x1)  # all hw:64 c:80

        x2 = self.TransformerDown2(x1_a)  # hw:64  c:80
        x2 = x2 + self.RB1(x1_hvd)  # hw:64  c:160
        x2_a, x2_hvd = self.HaarDownsample2(x2)  # all hw:40 c:160

        x3 = self.TransformerDown3(x2_a)  # hw:40  c:160
        x3 = x3 + self.RB2(x2_hvd)
        x3_a, x3_hvd = self.HaarDownsample3(x3)  # all hw:16 c:160
        ############ encoder ############

        x_trans0 = self.Transformer0(x3_a)  # hw:16 c:160
        x_trans = self.Transformer(
            x_trans0
        )  # hw:16 c:160 => 중앙 블록. 여러 개 존재 가능
        x_trans = x_trans + self.RB3(x3_hvd)  # hw:16 c:160

        ############ decoder ############
        x_up1 = self.HaarFeatureFusion1(x3, x_trans)  # hw:40 c:160
        x_1 = self.TransformerUp1(x_up1)  # hw:40 c:160

        x_up2 = self.HaarFeatureFusion2(x2, x_1)  # hw:64 c:80
        x_2 = self.TransformerUp2(x_up2)  # hw:64 c:80

        x_up3 = self.HaarFeatureFusion3(x1, x_2)  # hw:128 c:40
        x_3 = self.TransformerUp3(x_up3)  # hw:32 c:40
        ############ decoder ############

        out_img = self.out_conv(x_3 + x_first)  # 마무리 conv

        return out_img


###############################


class WFEN_no_Wavelet(nn.Module):
    def __init__(
        self,
        inchannel=3,
        min_ch=40,
        res_depth=6,
    ):
        print("🍊 WFEN_no_Wavelet")
        super(WFEN_no_Wavelet, self).__init__()
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
                min_ch * 2 // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelUnshuffle(2),
        )
        self.Downsample3 = nn.Sequential(
            nn.Conv2d(
                min_ch * 2 * 2,
                min_ch * 2 // 2,
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
            FDT(inp_channels=min_ch * 2 * 2, window_sizes=2, shifts=0, num_heads=8),
        )

        self.Transformer0 = FDT(
            inp_channels=min_ch * 2 * 2, window_sizes=1, shifts=0, num_heads=8
        )
        Transformer = []
        for i in range(res_depth - 1):
            (
                Transformer.append(
                    FDT(
                        inp_channels=min_ch * 2 * 2,
                        window_sizes=1,
                        shifts=1,
                        num_heads=8,
                    ),
                )
                if i % 2 == 0
                else Transformer.append(
                    FDT(
                        inp_channels=min_ch * 2 * 2,
                        window_sizes=1,
                        shifts=0,
                        num_heads=8,
                    )
                )
            )
        self.Transformer = nn.Sequential(*Transformer)

        self.TransformerUp1 = nn.Sequential(
            FDT(inp_channels=min_ch * 2 * 2, window_sizes=8, shifts=1, num_heads=8),
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
                min_ch * 2 * 2,
                min_ch * 2 * 2 * 2 * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelShuffle(2),
        )
        self.Upsample2 = nn.Sequential(
            nn.Conv2d(
                min_ch * 2 * 2,
                min_ch * 2 * 2 * 2,
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
                min_ch * 2 * 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.PixelShuffle(2),
        )

        self.out_conv = nn.Conv2d(min_ch, inchannel, kernel_size=3, padding=1)

    def forward(self, input_img):
        x_first = self.first_conv(input_img)  # 첫 conv

        ############ encoder ############
        x1 = self.TransformerDown1(x_first)  # hw:128  c:40
        x1 = self.Downsample1(x1)  # all hw:64 c:80

        x2 = self.TransformerDown2(x1)  # hw:64  c:80
        x2 = self.Downsample2(x2)  # all hw:32 c:160

        x3 = self.TransformerDown3(x2)  # hw:32  c:160
        x3 = self.Downsample3(x3)  # all hw:16 c:160 (채널 수 그대로, 해상도만 2배)
        ############ encoder ############

        x_trans0 = self.Transformer0(x3)  # hw:16 c:160
        x_trans = self.Transformer(
            x_trans0
        )  # hw:16 c:160 => 중앙 블록. 여러 개 존재 가능

        ############ decoder ############
        x_up1 = self.Upsample1(
            x3 + x_trans
        )  # hw:32 c:160   (채널 수 그대로, 해상도만 2배)
        x_1 = self.TransformerUp1(x_up1)  # hw:32 c:160

        x_up2 = self.Upsample2(x2 + x_1)  # hw:64 c:80
        x_2 = self.TransformerUp2(x_up2)  # hw:64 c:80

        x_up3 = self.Upsample3(x1 + x_2)  # hw:128 c:40
        x_3 = self.TransformerUp3(x_up3)  # hw:128 c:40
        ############ decoder ############

        out_img = self.out_conv(x_3)  # 마무리 conv

        return out_img
