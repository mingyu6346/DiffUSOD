from functools import partial
import math
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
from timm.models.layers import DropPath

from model.down_up import Down_wt, DySample


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, ca_num_heads=4, sa_num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()

        self.ca_attention = ca_attention
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.split_groups = self.dim // ca_num_heads

        if ca_attention == 1:
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.s = nn.Linear(dim, dim, bias=qkv_bias)
            for i in range(self.ca_num_heads):
                local_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3 + i * 2),
                                       padding=(1 + i), stride=1,
                                       groups=dim // self.ca_num_heads)  # kernel_size 3,5,7,9 大核dw卷积，padding 1,2,3,4
                setattr(self, f"local_conv_{i + 1}", local_conv)
            self.proj0 = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1,
                                   groups=self.split_groups)
            self.bn = nn.BatchNorm2d(dim * expand_ratio)
            self.proj1 = nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)

        else:
            head_dim = dim // sa_num_heads
            self.scale = qk_scale or head_dim ** -0.5
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.ca_attention == 1:
            v = self.v(x)
            time_token = x[:, 0, :].reshape(B, 1, C)
            x_ = x[:, 1:, :]

            s = self.s(x_).reshape(B, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(3, 0, 4, 1,
                                                                                               2)  # num_heads,B,C//num_heads,H,W
            for i in range(self.ca_num_heads):
                local_conv = getattr(self, f"local_conv_{i + 1}")
                s_i = s[i]  # B,C,H,W
                s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)
                if i == 0:
                    s_out = s_i
                else:
                    s_out = torch.cat([s_out, s_i], 2)
            s_out = s_out.reshape(B, C, H, W)
            s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))
            self.modulator = s_out
            s_out = s_out.reshape(B, C, N - 1).permute(0, 2, 1)
            s_out = torch.cat((time_token, s_out), dim=1)
            x = s_out * v

        else:
            time_token = x[:, 0, :].reshape(B, 1, C)
            x_ = x[:, 1:, :]
            q = self.q(x_).reshape(B, N - 1, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x_).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_ = (attn @ v).transpose(1, 2).reshape(B, N - 1, C)

            x_conv = self.local_conv(v.transpose(1, 2).reshape(B, N - 1, C).transpose(1, 2).view(B, C, H, W))
            x_conv = x_conv.view(B, C, N - 1).transpose(1, 2)
            x_ = x_ + x_conv
            x = torch.cat((time_token, x_), dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, ca_num_heads, sa_num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 use_layerscale=False, layerscale_value=1e-4, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ca_attention=1, expand_ratio=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            ca_num_heads=ca_num_heads, sa_num_heads=sa_num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, ca_attention=ca_attention,
            expand_ratio=expand_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=3, stride=2, in_chans=3, embed_dim=768, mask_chans=0):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        img_size = to_2tuple(img_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        # self.proj = Down_wt(in_chans, embed_dim)
        if mask_chans != 0:
            # self.mask_proj = nn.Conv2d(mask_chans, embed_dim, kernel_size=patch_size, stride=stride,
            #                            padding=(patch_size[0] // 2, patch_size[1] // 2))
            self.mask_proj = Down_wt(in_chans, embed_dim, init_wt_weight_zero=True)
            # set mask_proj weight to 0
            # self.mask_proj.weight.data.zero_()
            # self.mask_proj.bias.data.zero_()

            self.mask_proj.conv_bn_relu[0].weight.data.zero_()
            self.mask_proj.conv_bn_relu[0].bias.data.zero_()

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        x = self.proj(x)
        # Do a zero conv to get the mask
        if mask is not None:
            mask = self.mask_proj(mask)
            x = x + mask
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class Head(nn.Module):
    def __init__(self, head_conv, dim, mask_chans=0):
        super(Head, self).__init__()
        stem = [nn.Conv2d(3, dim, head_conv, 2, padding=3 if head_conv == 7 else 1, bias=False), nn.BatchNorm2d(dim),
                nn.ReLU(True)]
        stem.append(nn.Conv2d(dim, dim, kernel_size=2, stride=2))
        # stem = [Down_wt(3, dim // 2), Down_wt(dim // 2, dim)]
        self.conv = nn.Sequential(*stem)
        if mask_chans != 0:
            # self.mask_proj = nn.Sequential(nn.Conv2d(mask_chans, dim, kernel_size=head_conv, stride=2, padding=(1, 1)),
            #                                nn.Conv2d(dim, dim, kernel_size=2, stride=2)
            #                                )
            self.mask_proj = nn.Sequential(Down_wt(1, dim // 4, init_wt_weight_zero=True),
                                           Down_wt(dim // 4, dim, init_wt_weight_zero=True))
            # set mask_proj weight to 0
            # self.mask_proj[0].weight.data.zero_()
            # self.mask_proj[0].bias.data.zero_()
            # self.mask_proj[1].weight.data.zero_()
            # self.mask_proj[1].bias.data.zero_()

            self.mask_proj[0].conv_bn_relu[0].weight.data.zero_()
            self.mask_proj[0].conv_bn_relu[0].bias.data.zero_()

            self.mask_proj[1].conv_bn_relu[0].weight.data.zero_()
            self.mask_proj[1].conv_bn_relu[0].bias.data.zero_()

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.conv(x)
        _, _, H, W = x.shape
        # Do a zero conv to get the mask
        if mask is not None:
            mask = self.mask_proj(mask)
            # print(f"{mask.shape=}")
            x = x + mask
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


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


class SMT(nn.Module):
    def __init__(self, img_size=384, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[8, 6, 4, 2],
                 qkv_bias=False, qk_scale=None, use_layerscale=False, layerscale_value=1e-4, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[2, 2, 8, 1], ca_attentions=[1, 1, 1, 0], num_stages=4, head_conv=3, expand_ratio=2,
                 mask_chans=1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.embed_dims = embed_dims  # 设置每个阶段的嵌入维度
        self.mask_chans = mask_chans  # 设置掩码通道数

        # 时间嵌入层
        self.time_embed = nn.ModuleList()  # 定义时间嵌入的ModuleList，用于存放不同阶段的时间嵌入模块
        for i in range(0, num_stages):  # 根据嵌入维度的数量创建时间嵌入层
            self.time_embed.append(nn.Sequential(  # 每个时间嵌入由两个线性层和一个SiLU激活函数组成
                nn.Linear(embed_dims[i], embed_dims[i]),
                nn.GELU(),
                nn.Linear(embed_dims[i], embed_dims[i]),
            ))

        # —————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
        # 图像块嵌入层
        self.patch_embed1 = Head(head_conv, embed_dims[0], mask_chans=mask_chans)  # 阶段1的图像块嵌入
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4,
                                              patch_size=3,
                                              stride=2,
                                              in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8,
                                              patch_size=3,
                                              stride=2,
                                              in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16,
                                              patch_size=3,
                                              stride=2,
                                              in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        # Stage 1
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], ca_num_heads=ca_num_heads[0], sa_num_heads=sa_num_heads[0], mlp_ratio=mlp_ratios[0],
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            use_layerscale=use_layerscale,
            layerscale_value=layerscale_value,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
            ca_attention=ca_attentions[0], expand_ratio=expand_ratio)
            for j in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])  # 阶段1的归一化层

        # Stage 2
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], ca_num_heads=ca_num_heads[1], sa_num_heads=sa_num_heads[1], mlp_ratio=mlp_ratios[1],
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            use_layerscale=use_layerscale,
            layerscale_value=layerscale_value,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
            ca_attention=ca_attentions[1], expand_ratio=expand_ratio)
            for j in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])  # 阶段2的归一化层

        # Stage 3
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], ca_num_heads=ca_num_heads[2], sa_num_heads=sa_num_heads[2], mlp_ratio=mlp_ratios[2],
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            use_layerscale=use_layerscale,
            layerscale_value=layerscale_value,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
            ca_attention=0 if j % 2 != 0 else ca_attentions[2], expand_ratio=expand_ratio)
            for j in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])  # 阶段3的归一化层

        # Stage 4
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], ca_num_heads=ca_num_heads[3], sa_num_heads=sa_num_heads[3], mlp_ratio=mlp_ratios[3],
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            use_layerscale=use_layerscale,
            layerscale_value=layerscale_value,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
            ca_attention=ca_attentions[3], expand_ratio=expand_ratio)
            for j in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])  # 阶段4的归一化层

    ##########################################################################################################################

    # for i in range(3,num_stages):
    #     if i == 0:
    #         patch_embed = Head(head_conv, embed_dims[i])  #
    #     else:
    #         patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
    #                                         patch_size=3,
    #                                         stride=2,
    #                                         in_chans=embed_dims[i - 1],
    #                                         embed_dim=embed_dims[i])
    #
    #     block = nn.ModuleList([Block(
    #         dim=embed_dims[i], ca_num_heads=ca_num_heads[i], sa_num_heads=sa_num_heads[i], mlp_ratio=mlp_ratios[i],
    #         qkv_bias=qkv_bias, qk_scale=qk_scale,
    #         use_layerscale=use_layerscale,
    #         layerscale_value=layerscale_value,
    #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
    #         ca_attention=0 if i == 2 and j % 2 != 0 else ca_attentions[i], expand_ratio=expand_ratio)
    #         for j in range(depths[i])])
    #     norm = norm_layer(embed_dims[i])
    #     cur += depths[i]
    #
    #     setattr(self, f"patch_embed{i + 1}", patch_embed)
    #     setattr(self, f"block{i + 1}", block)
    #     setattr(self, f"norm{i + 1}", norm)
    ##########################################################################################################################
    # classification head
    # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

    # def freeze_patch_emb(self):
    #     self.patch_embed1.requires_grad = False

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    # def get_classifier(self):
    #     return self.head

    # def reset_classifier(self, num_classes, global_pool=''):
    #     self.num_classes = num_classes
    #     self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, timesteps, cond_img):
        # x         -> GT(train phase) or Noise(sample phase)
        # cond_img  ->  input color image

        B = x.shape[0]
        f_list = []
        time_token_ls = []

        # Stage 1
        time_token = self.time_embed[0](timestep_embedding(timesteps, self.embed_dims[0]))  # 获取阶段1的时间嵌入
        time_token = time_token.unsqueeze(dim=1)  # 增加一个维度，使其与输入图像保持一致
        x, H, W = self.patch_embed1(cond_img, x)
        x = torch.cat([time_token, x], dim=1)  # 将时间嵌入与图像块嵌入拼接
        for i, blk in enumerate(self.block1):  # 遍历阶段1的编码块
            x = blk(x, H, W)  # 通过每个编码块处理
        x = self.norm1(x)  # 阶段1的归一化操作
        time_token = x[:, 0]  # 更新时间嵌入
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 调整x的形状以符合下一阶段
        f_list.append(x)
        time_token_ls.append(time_token)

        # Stage 2
        time_token = self.time_embed[1](timestep_embedding(timesteps, self.embed_dims[1]))  # 获取阶段2的时间嵌入
        time_token = time_token.unsqueeze(dim=1)
        x, H, W = self.patch_embed2(x)  # 执行阶段2的图像块嵌入操作
        x = torch.cat([time_token, x], dim=1)  # 拼接时间嵌入
        for i, blk in enumerate(self.block2):  # 遍历阶段2的编码块
            x = blk(x, H, W)
        x = self.norm2(x)  # 阶段2的归一化操作
        time_token = x[:, 0]  # 更新时间嵌入
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 调整x的形状
        f_list.append(x)
        time_token_ls.append(time_token)

        # Stage 3
        time_token = self.time_embed[2](timestep_embedding(timesteps, self.embed_dims[2]))  # 获取阶段3的时间嵌入
        time_token = time_token.unsqueeze(dim=1)
        x, H, W = self.patch_embed3(x)  # 执行阶段3的图像块嵌入操作
        x = torch.cat([time_token, x], dim=1)  # 拼接时间嵌入
        for i, blk in enumerate(self.block3):  # 遍历阶段3的编码块
            x = blk(x, H, W)
        x = self.norm3(x)  # 阶段3的归一化操作
        time_token = x[:, 0]  # 更新时间嵌入
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 调整x的形状
        f_list.append(x)  # 将阶段3的输出加入列表
        time_token_ls.append(time_token)

        # 阶段4
        time_token = self.time_embed[3](timestep_embedding(timesteps, self.embed_dims[3]))  # 获取阶段4的时间嵌入
        time_token = time_token.unsqueeze(dim=1)
        x, H, W = self.patch_embed4(x)  # 执行阶段4的图像块嵌入操作
        x = torch.cat([time_token, x], dim=1)  # 拼接时间嵌入
        for i, blk in enumerate(self.block4):  # 遍历阶段4的编码块
            x = blk(x, H, W)
        x = self.norm4(x)  # 阶段4的归一化操作
        time_token = x[:, 0]  # 更新时间嵌入
        x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # 调整x的形状
        f_list.append(x)  # 将阶段4的输出加入列表
        time_token_ls.append(time_token)

        # for i in range(1, self.num_stages):
        #     patch_embed = getattr(self, f"patch_embed{i + 1}")
        #     block = getattr(self, f"block{i + 1}")
        #     norm = getattr(self, f"norm{i + 1}")
        #     x, H, W = patch_embed(x)
        #     for blk in block:
        #         x = blk(x, H, W)
        #     x = norm(x)
        #     # if i != self.num_stages - 1:
        #     #     x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #     x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #     f_list.append(x)

        # return x.mean(dim=1), f_list
        # return f_list,time_token_ls
        return f_list

    def forward(self, x, timesteps, cond_img):
        f_list = self.forward_features(x, timesteps, cond_img)
        # f_list,time_token_ls = self.forward_features(x, timesteps, cond_img)
        # x, f_list = self.forward_features(x)
        # x = self.head(x)

        return f_list
        # return f_list, time_token_ls


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        time_token = x[:, 0, :].reshape(B, 1, C)
        x_ = x[:, 1:, :]
        x_ = x_.transpose(1, 2).view(B, C, H, W)
        x_ = self.dwconv(x_)
        x_ = x_.flatten(2).transpose(1, 2)
        x = torch.cat((time_token, x_), dim=1)

        return x


def smt_t(pretrained=False, **kwargs):
    model = SMT(
        embed_dims=[64, 128, 256, 512], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[4, 4, 4, 2],
        qkv_bias=True, depths=[2, 2, 8, 1], ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2, **kwargs)
    model.default_cfg = _cfg()

    return model


def smt_s(pretrained=False, **kwargs):
    model = SMT(
        embed_dims=[64, 128, 256, 512], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[4, 4, 4, 2],
        qkv_bias=True, depths=[3, 4, 18, 2], ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2, **kwargs)
    model.default_cfg = _cfg()
    return model


def smt_b(pretrained=False, **kwargs):
    model = SMT(
        embed_dims=[64, 128, 256, 512], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[8, 6, 4, 2],
        qkv_bias=True, depths=[4, 6, 28, 2], ca_attentions=[1, 1, 1, 0], head_conv=7, expand_ratio=2, **kwargs)
    model.default_cfg = _cfg()

    return model


def smt_l(pretrained=False, **kwargs):
    model = SMT(
        embed_dims=[96, 192, 384, 768], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[8, 6, 4, 2],
        qkv_bias=True, depths=[4, 6, 28, 4], ca_attentions=[1, 1, 1, 0], head_conv=7, expand_ratio=2, **kwargs)
    model.default_cfg = _cfg()

    return model


if __name__ == '__main__':
    train_size = 384
    model = smt_t().eval()

    cond_image = torch.randn(1, 3, train_size, train_size)
    t = torch.randn(size=(1,))
    noise = torch.randn(1, 1, train_size, train_size)

    # f_ls = model(noise, t, cond_image)
    f_ls, time_token_ls = model(noise, t, cond_image)

    for i in f_ls:
        print(i.shape)

    for i in time_token_ls:
        print(i.shape)
