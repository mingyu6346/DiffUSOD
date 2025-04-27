import warnings
from functools import partial

import math
import torch
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from pytorch_wavelets import DWTForward
from scipy import randn

from denoising_diffusion_pytorch.simple_diffusion import ResnetBlock

from torch.nn import Module
import torch.nn as nn
from model.SMT import smt_t


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class conv(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()

        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.proj(x)
        # x = x.flatten(2).transpose(1, 2)
        return x


def Downsample(
        dim,
        dim_out=None,
        factor=2
):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=factor, p2=factor),
        nn.Conv2d(dim * (factor ** 2), dim if dim_out is None else dim_out, 1)
    )


class Upsample(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            factor=2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = dim if dim_out is None else dim_out
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.GELU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


# https://github.com/apple1986/HWD
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch, init_wt_weight_zero=False):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        if init_wt_weight_zero:
            # 将内部卷积权重初始化为0
            for param in self.wt.parameters():
                torch.nn.init.zeros_(param)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x


class WaveletT(nn.Module):
    def __init__(self):
        super(WaveletT, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        for param in self.wt.parameters():
            torch.nn.init.zeros_(param)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]

        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)

        # return yL, y_HL, y_LH, y_HH
        return yL, yH


class CBG(Module):
    def __init__(self, inc, outc, k, s, p, g=1, b=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=k, stride=s, padding=p, groups=g, bias=b)
        self.bn = nn.BatchNorm2d(num_features=outc)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class EmptyObject(object):
    def __init__(self, *args, **kwargs):
        pass


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MultiScaleConv(nn.Module):
    def __init__(self, inc, outc):
        super(MultiScaleConv, self).__init__()
        self.conv1 = nn.Conv2d(inc // 4, outc // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(inc // 4, outc // 4, kernel_size=3, padding=1, groups=outc // 4)
        self.conv3 = nn.Conv2d(inc // 4, outc // 4, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(inc // 4, outc // 4, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        # return x1 + x2 + x3
        return torch.cat((x1, x2, x3, x4), dim=1)


class up_sample(nn.Module):
    def __init__(self, inc, outc, ks=3, bias=True):
        super(up_sample, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc * 2, kernel_size=1, stride=1, bias=bias),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(in_channels=inc // 2, out_channels=outc, kernel_size=ks, stride=1, padding=ks // 2, bias=bias),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

    def forward(self, x):
        return self.up(x)


class down_sample(nn.Module):
    def __init__(self, inc, outc):
        super(down_sample, self).__init__()
        self.down = nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=2),
            nn.Conv2d(in_channels=inc * 4, out_channels=outc, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

    def forward(self, x):
        return self.down(x)


class convbngelu(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, padding=1, groups=1, bias=True):
        super(convbngelu, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=groups, bias=bias),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

    def forward(self, x):
        return self.up(x)


class KernelConv(nn.Module):
    def __init__(self, dim, HW, window_size=11, use_ca=True, use_pw=True):
        super(KernelConv, self).__init__()
        self.down_factor = HW // window_size
        self.hidden_dim = dim * (self.down_factor ** 2)
        self.pixel_unshuffle = nn.PixelUnshuffle(
            downscale_factor=self.down_factor) if self.down_factor > 1 else nn.Identity()
        self.dw = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=window_size, stride=1,
                            padding=window_size // 2, groups=self.hidden_dim)
        self.bn_1 = nn.BatchNorm2d(self.hidden_dim)
        self.act_1 = nn.GELU()

        self.pw_2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1, 1) if use_pw else nn.Identity()
        self.bn_2 = nn.BatchNorm2d(self.hidden_dim)
        self.act_2 = nn.GELU()

        self.up = nn.PixelShuffle(upscale_factor=self.down_factor) if self.down_factor > 1 else nn.Identity()

        self.use_ca = use_ca
        if self.use_ca:
            self.ca = ChannelAttention(dim)

    def forward(self, x):
        if self.use_ca:
            att = self.ca(x)
        x = self.pixel_unshuffle(x)
        x = self.dw(x)
        x = self.bn_1(x)
        x = self.act_1(x)
        x = self.pw_2(x)
        x = self.bn_2(x)
        x = self.act_2(x)

        x = self.up(x)
        if self.use_ca:
            x = x * att
        return x


class HighFusion(nn.Module):
    def __init__(self, inc_high, outc_low, HW, up=False, window_size=7, use_ca=True, use_pw=True):
        super().__init__()

        self.up = up_sample(inc_high, outc_low) if up else nn.Identity()
        self.fwa = KernelConv(outc_low, HW=HW, window_size=window_size, use_ca=use_ca, use_pw=use_pw)
        self.cbg_1 = convbngelu(outc_low * 2, outc_low, kernel_size=3, stride=1, padding=1, groups=outc_low)
        self.cbg_2 = convbngelu(outc_low, outc_low, kernel_size=3, stride=1, padding=1, groups=outc_low)

    def forward(self, x_high, x_low):
        x_high = self.up(x_high)
        x_cat = torch.cat((x_high, x_low), dim=1)
        x_cat = self.cbg_1(x_cat)
        x_cat = self.fwa(x_cat)
        x_cat = self.cbg_2(x_cat)
        x_cat = x_low * x_cat

        return x_cat


class MultiScaleFusion(nn.Module):
    def __init__(self, inc_high, outc_low):
        super().__init__()

        self.up = up_sample(inc_high, outc_low, ks=1, bias=False)
        self.msconv = MultiScaleConv(outc_low * 2, outc_low)
        self.cbg_1 = convbngelu(outc_low, outc_low)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(outc_low)

        self.down = down_sample(outc_low, inc_high)
        self.sa_2 = SpatialAttention()
        self.msconv_2 = MultiScaleConv(inc_high * 2, inc_high)
        self.cbg_2 = convbngelu(inc_high, inc_high, kernel_size=1, stride=1, padding=0, groups=1, bias=False)

        self.last_up = up_sample(inc_high, outc_low, ks=1, bias=False)

    def forward(self, x_high, x_low):
        x_high_clone = x_high.clone()
        x_high = self.up(x_high)  # stage 4 -> stage 3
        x_cat_1 = torch.cat((x_high, x_low), dim=1)
        x_cat_1 = self.msconv(x_cat_1)

        x_cat_1 = self.sa(x_cat_1) * x_cat_1
        x_cat_1 = x_low + self.cbg_1(x_cat_1)

        x_low_down = self.down(x_low)  # inc_high
        x_cat_2 = torch.cat((x_low_down, x_high_clone), dim=1)
        x_cat_2 = self.msconv_2(x_cat_2)
        x_cat_2 = self.sa_2(x_cat_2) * x_cat_2
        x_cat_2 = x_high_clone + self.cbg_2(x_cat_2)

        fuse = self.last_up(x_cat_2) + x_cat_1

        return fuse


def fft_2d(x):
    return torch.torch.fft.fft2(x)


def ifft_2d(x):
    return torch.torch.fft.ifft2(x).real


class FreqFusion(nn.Module):
    def __init__(self, dim=64):
        super().__init__()

        self.wave_p1 = WaveletT()
        self.ps_1 = nn.PixelShuffle(2)
        self.conv_3 = CBG(24, 24, k=3, s=1, p=1, g=1, b=False)
        self.conv_5 = CBG(24, 24, k=5, s=1, p=2, g=1, b=False)
        self.conv_7 = CBG(24, 24, k=7, s=1, p=3, g=1, b=False)
        self.conv_9 = CBG(24, 24, k=9, s=1, p=4, g=1, b=False)

        self.wave_p3 = WaveletT()
        self.ps_3 = nn.PixelShuffle(8)

        self.last_conv = nn.Sequential(
            CBG(96, 96, k=3, s=1, p=1, g=96),
            CBG(96, 96, k=1, s=1, p=0, g=1),
        )

    def forward(self, x_1, x_2, x_3, x_4):
        # yL_1, y_HL_1, y_LH_1, y_HH_1 = self.wave_p1(x_1)
        # yL_3, y_HL_3, y_LH_3, y_HH_3 = self.wave_p3(x_3)
        x_1_l, x_1_h = self.wave_p1(x_1)
        x_3_l, x_3_h = self.wave_p3(x_3)

        x_12 = torch.cat([x_1_l, x_1_h, x_2], dim=1)
        x_12 = self.ps_1(x_12)
        _p1, _p2, _p3, _p4 = torch.chunk(x_12, 4, dim=1)

        x_34 = torch.cat([x_3_l, x_3_h, x_4], dim=1)
        x_34 = self.ps_3(x_34)

        _p1 = self.conv_3(_p1) + x_34
        _p2 = self.conv_5(_p2) + x_34
        _p3 = self.conv_7(_p3) + x_34
        _p4 = self.conv_9(_p4) + x_34

        x_12 = torch.cat([_p1, _p2, _p3, _p4], dim=1) + x_12

        x_12 = self.last_conv(x_12)
        return x_12


class Decoder(Module):
    def __init__(self, dims, dim, class_num=1):
        super(Decoder, self).__init__()
        self.num_classes = class_num  # 设置输出类别数

        # 输入的4个特征层通道数
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]
        embedding_dim = dim  # 嵌入维度

        #
        self.msf_c4_c4 = HighFusion(c4_in_channels, c4_in_channels, HW=12, up=False, use_ca=True, use_pw=True)
        self.msf_c4_c3 = HighFusion(c4_in_channels, c3_in_channels, HW=12, up=True, use_ca=True, use_pw=True)
        self.msf_c3_c2 = MultiScaleFusion(c3_in_channels, c2_in_channels)
        self.msf_c2_c1 = MultiScaleFusion(c2_in_channels, c1_in_channels)

        # 将每个特征层通过卷积层映射到 embedding_dim 维度
        self.linear_c4 = nn.Sequential(
            nn.Conv2d(c4_in_channels, c1_in_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU())
        self.linear_c3 = nn.Sequential(
            nn.Conv2d(c3_in_channels, c1_in_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU())
        self.up_2 = Upsample(c2_in_channels, c1_in_channels, 2)

        # 时间嵌入部分：用于处理时间步的线性嵌入
        self.time_embed_dim = embedding_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # # 下采样网络模块
        self.down = nn.Sequential(
            Down_wt(1, embedding_dim // 8),  # 初步卷积
            # # resnet_block(embedding_dim//4, embedding_dim//4, time_emb_dim=embedding_dim//4),  # ResNet块用于特征提取
            DownSampleBlock(embedding_dim // 8, embedding_dim // 4),
            DownSampleBlock(embedding_dim // 4, embedding_dim // 2),
            DownSampleBlock(embedding_dim // 2, embedding_dim),
            ResnetBlock(embedding_dim, embedding_dim, time_emb_dim=self.time_embed_dim),
            DownSampleBlock(embedding_dim, embedding_dim * 2),
        )

        self.freq_fuse = FreqFusion(dim=64)

        # 上采样网络模块
        self.up = nn.Sequential(
            CBG(embedding_dim + 96, embedding_dim, k=1, s=1, p=0),
            Upsample(embedding_dim, embedding_dim // 4, factor=2),  # 上采样因子为2
            CBG(embedding_dim // 4, embedding_dim // 4, k=3, s=1, p=1),  # 上采样后卷积
            Upsample(embedding_dim // 4, embedding_dim // 8, factor=2),  # 再次上采样
            CBG(embedding_dim // 8, embedding_dim // 8, k=3, s=1, p=1),  # 上采样后卷积
        )

        # 最终预测部分
        self.pred = nn.Sequential(
            # nn.Dropout(0.1),  # 防止过拟合
            nn.Conv2d(embedding_dim // 8, embedding_dim // 4, kernel_size=1),
            nn.BatchNorm2d(embedding_dim // 4),
            nn.GELU(),
            nn.Conv2d(embedding_dim // 4, self.num_classes, kernel_size=1)  # 预测类别
        )

    def forward(self, x, timesteps, inputs):
        # x.shape : [B, 1, 384,384] 表示输入条件(训练时是GT，sample时是随机噪声)
        t = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))

        c1, c2, c3, c4 = inputs  # 四个输入特征层C1, C2, C3, C4

        ##############################################
        # 处理输入特征 x 的下采样流程
        _x = []
        for i, blk in enumerate(self.down):  # 遍历下采样模块
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)

            if i in [1, 2, 3, 5]:
                if i == 5:
                    i = 4
                x = x + inputs[i - 1]
                _x.append(x)

        x = self.freq_fuse(*_x)
        #
        # ############## C1-C4特征层 ###########
        _c4 = self.msf_c4_c4(c4, c4)  # 512,12,12
        _c3 = self.msf_c4_c3(_c4, c3)  # 256,24,24
        _c2 = self.msf_c3_c2(_c3, c2)  # 128,48,48
        _c1 = self.msf_c2_c1(_c2, c1)  # 64,96,96

        # # 512 11
        _c4 = resize(_c4, size=_c1.shape[2], mode='bilinear', align_corners=False)
        _c4 = self.linear_c4(_c4)
        _c3 = resize(_c3, size=_c1.shape[2], mode='bilinear', align_corners=False)
        _c3 = self.linear_c3(_c3)
        _c2 = self.up_2(_c2)

        # # to _c1 -> B,256,96,96
        # # curr _c1 -> B,64,96,96
        _c1_cat = torch.cat((_c1, _c2, _c3, _c4), dim=1)

        # _c1 = F.interpolate(self.baseline(c4), size=c1.size()[2:], mode='bilinear', align_corners=False)

        # 将融合后的特征与下采样后的 x 结合，进入上采样流程
        x = torch.cat([_c1_cat, x], dim=1)
        for blk in self.up:  # 遍历上采样模块
            x = blk(x)

        # 最后预测输出
        return self.pred(x)


class DownSampleBlock(nn.Module):
    def __init__(self, inc, outc):
        super(DownSampleBlock, self).__init__()
        self.wdown = Down_wt(inc, outc)  # 下采样
        self.dwc = CBG(outc, outc, k=3, s=1, p=1, g=outc)  # 卷积
        self.pwc = CBG(outc, outc, k=1, s=1, p=0, g=1)

    def forward(self, x):
        return self.pwc(self.dwc(self.wdown(x)))


class net(nn.Module):
    def __init__(self, class_num=1, mask_chans=1, **kwargs):
        super(net, self).__init__()
        self.class_num = class_num
        # self.backbone = pvt_v2_b4_m(in_chans=3, mask_chans=mask_chans)
        self.backbone = smt_t(mask_chans=mask_chans)
        self.decode_head = Decoder(dims=[64, 128, 256, 512], dim=256, class_num=class_num)
        self._init_weights()  # load pretrain

    def forward(self, x, timesteps, cond_img):
        features = self.backbone(x, timesteps, cond_img)
        pred = self.decode_head(x, timesteps, features)
        return pred

    def _init_weights(self):
        # pretrained_dict = torch.load("ckpts/smt_tiny.pth")  # for save mem
        from utils.affiliated_utils import load_param
        try:
            load_param('./ckpts/smt_tiny.pth', self.backbone, mode='model')
        except Exception as e:
            try:
                load_param('../ckpts/smt_tiny.pth', self.backbone, mode='model')
            except Exception as e:
                load_param('../../ckpts/smt_tiny.pth', self.backbone, mode='model')

    @torch.inference_mode()
    def sample_unet(self, x, timesteps, cond_img):
        return self.forward(x, timesteps, cond_img)

    def extract_features(self, cond_img):
        # do nothing
        return cond_img

    @torch.inference_mode()
    def get_params_flops(self, gpu=2, batchsize=1, get_fps=False, size=224):
        from utils.affiliated_utils import param, flops, fps
        param(self)
        flops(self, gpu=gpu, count=3, batchsize=batchsize, size=size)
        fps(self, epoch_num=5, size=size, gpu=gpu, count=3) if get_fps else None


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from utils.affiliated_utils import param

    train_size = 384
    model = net().eval()

    cond_image = torch.randn(1, 3, train_size, train_size)
    t = torch.randn(size=(1,))
    noise = torch.randn(1, 1, train_size, train_size)

    f_ls = model(noise, t, cond_image)

    for i in f_ls:
        print(i.shape)

    model.get_params_flops(gpu=2, get_fps=True, size=train_size)
