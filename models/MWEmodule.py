import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Callable

from dinov3.models.vision_transformer import DinoVisionTransformer

from .util import wavelet

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class WTConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',padding='same'):
        super(WTConv2d, self).__init__()

        self.channel_proj = None
        if in_channels != out_channels:

             self.channel_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
             temp_in_channels = in_channels
        else:
             temp_in_channels = in_channels

        self.in_channels = temp_in_channels
        self.out_channels = out_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter = nn.Parameter(torch.rand(4, 1, 2, 2).repeat(self.in_channels, 1, 1, 1), requires_grad=False)
        self.iwt_filter = nn.Parameter(torch.rand(4, 1, 2, 2).repeat(self.in_channels, 1, 1, 1), requires_grad=False)

        self.base_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=self.in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,self.in_channels,1,1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(self.in_channels*4, self.in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=self.in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,self.in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:

            self.do_stride = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
        else:
            self.do_stride = None

    def forward(self, x):

        if self.channel_proj is not None:
             x_temp = x
        else:
             x_temp = x

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x_temp

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)

            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = torch.zeros(curr_x_ll.shape[0], curr_x_ll.shape[1], 4, curr_x_ll.shape[2]//2, curr_x_ll.shape[3]//2).to(curr_x_ll.device)

            curr_x_ll = curr_x[:,:,0,:,:]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])

        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)

            next_x_ll = torch.zeros(curr_x.shape[0], curr_x.shape[1], curr_shape[2], curr_shape[3]).to(curr_x.device)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x_temp = self.base_scale(self.base_conv(x_temp))
        x_temp = x_temp + x_tag

        if self.do_stride is not None:
            x_temp = self.do_stride(x_temp)

        if self.channel_proj is not None:
             x = self.channel_proj(x_temp)
        else:
             x = x_temp

        return x

class MultiReceptiveFieldBranch(nn.Module):

    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()
        self.base_channels = base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.fine_branch = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),

            WTConv2d(
                in_channels=base_channels,
                out_channels=base_channels,
                kernel_size=3, padding='same',
                wt_levels=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )

        self.medium_branch = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),

            nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),

            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
        )

        self.large_branch = nn.Sequential(

            WTConv2d(
                in_channels=base_channels,
                out_channels=base_channels * 2,
                kernel_size=5, padding='same',
                wt_levels=2, stride=2, bias=False
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),

            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),

            nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
        )

    def forward(self, x):
        stem_out = self.stem(x)
        fine_feat = self.fine_branch(stem_out)
        medium_feat = self.medium_branch(stem_out)
        large_feat = self.large_branch(stem_out)
        return [fine_feat, medium_feat, large_feat]

class concat(nn.Module):
    def __init__(self, sem_channels, det_channels, hidden_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(sem_channels + det_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

    def forward(self, semantic_feat, detail_feat):
        fused = torch.cat([semantic_feat, detail_feat], dim=1)
        return self.fusion(fused)

class CrossVSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            guide_channels: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0.1,
            norm_layer: Callable[..., nn.Module] = partial(LayerNorm2d, eps=1e-6),
            ssm_d_state: int = 16,
            ssm_ratio: float = 2.0,
            ssm_rank_ratio: float = 2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer: Callable = nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias: bool = True,
            ssm_drop_rate: float = 0.0,
            forward_type: str = "v2",
            guide_upsample_scale: float = 2.0,
            **kwargs,
    ):
        super().__init__()

        self.ssm_branch = ssm_ratio > 0
        self.hidden_dim = hidden_dim
        self.guide_upsample_scale = guide_upsample_scale

        self.guide_adapt = nn.Sequential(
            nn.Conv2d(guide_channels, hidden_dim, kernel_size=1, bias=False),
            nn.Upsample(
                scale_factor=guide_upsample_scale,
                mode="bilinear",
                align_corners=False
            ),
            norm_layer(hidden_dim),
            ssm_act_layer()
        )

        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(hidden_dim),
            ssm_act_layer()
        )

        self.lsblock = LSBlock(
            in_features=hidden_dim,
            hidden_features=hidden_dim,
            act_layer=ssm_act_layer,
            drop=ssm_drop_rate
        )

        if self.ssm_branch:
            self.norm_x = norm_layer(hidden_dim)
            self.norm_guide = norm_layer(hidden_dim)

            self.cross_ssm = CrossSS2D(
                guide_dim=hidden_dim,
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                forward_type=forward_type,
                **kwargs
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, guide_x: torch.Tensor):
        x, guide_x = sync_devices(x, guide_x)

        x_proj = self.proj_conv(x)
        guide_adapted = self.guide_adapt(guide_x)

        if guide_adapted.shape[2:] != x_proj.shape[2:]:
            guide_adapted = F.interpolate(
                guide_adapted,
                size=x_proj.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        x_local = self.lsblock(x_proj)

        if self.ssm_branch:
            x_norm = self.norm_x(x_local)
            guide_norm = self.norm_guide(guide_adapted)

            x_norm, guide_norm = sync_devices(x_norm, guide_norm)

            x_ssm = self.cross_ssm(
                x=x_norm,
                guide_x=guide_norm
            )

            x_proj = x_proj + self.drop_path(x_ssm)

        return x_proj



class CrossScaleStateBlock(nn.Module):

    """优化版跨尺度状态增强模块 - P5→P4→P3语义指导"""
    def __init__(self,
                 in_channels: int = 384,
                 cross_ssm_d_state: int = 16,
                 cross_ssm_ratio: float = 2.0,
                 cross_ssm_rank_ratio: float = 2.0,
                 cross_ssm_dt_rank: Any = "auto",
                 cross_ssm_act_layer: Callable = nn.SiLU,
                 cross_ssm_conv: int = 3,
                 cross_ssm_conv_bias: bool = True,
                 cross_ssm_drop_rate: float = 0.0,
                 cross_mlp_ratio: float = 4.0,
                 cross_mlp_act_layer: Callable = nn.GELU,
                 cross_mlp_drop_rate: float = 0.0,
                 cross_drop_path: float = 0.1):
        super().__init__()
        self.in_channels = in_channels

        self.cross_vss_blocks = nn.ModuleList([
            CrossVSSBlock(
                in_channels=in_channels,
                guide_channels=in_channels,
                hidden_dim=in_channels,
                drop_path=cross_drop_path,
                ssm_d_state=cross_ssm_d_state,
                ssm_ratio=cross_ssm_ratio,
                ssm_rank_ratio=cross_ssm_rank_ratio,
                ssm_dt_rank=cross_ssm_dt_rank,
                ssm_act_layer=cross_ssm_act_layer,
                ssm_conv=cross_ssm_conv,
                ssm_conv_bias=cross_ssm_conv_bias,
                ssm_drop_rate=cross_ssm_drop_rate,
                forward_type="v2",
                mlp_ratio=cross_mlp_ratio,
                mlp_act_layer=cross_mlp_act_layer,
                mlp_drop_rate=cross_mlp_drop_rate,
                guide_upsample_scale=2.0,
            ),
            CrossVSSBlock(
                in_channels=in_channels,
                guide_channels=in_channels,
                hidden_dim=in_channels,
                drop_path=cross_drop_path,
                ssm_d_state=cross_ssm_d_state,
                ssm_ratio=cross_ssm_ratio,
                ssm_rank_ratio=cross_ssm_rank_ratio,
                ssm_dt_rank=cross_ssm_dt_rank,
                ssm_act_layer=cross_ssm_act_layer,
                ssm_conv=cross_ssm_conv,
                ssm_conv_bias=cross_ssm_conv_bias,
                ssm_drop_rate=cross_ssm_drop_rate,
                forward_type="v2",
                mlp_ratio=cross_mlp_ratio,
                mlp_act_layer=cross_mlp_act_layer,
                mlp_drop_rate=cross_mlp_drop_rate,
                guide_upsample_scale=2.0,
            )
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(feats) != 3:
            raise ValueError(f"要求输入3个尺度特征，实际收到{len(feats)}个")
        P3, P4, P5 = feats

        for i, feat in enumerate([P3, P4, P5]):
            if feat.shape[1] != self.in_channels:
                raise RuntimeError(f"第{i+1}个特征通道数{feat.shape[1]}与配置{self.in_channels}不匹配")

        P4_enhanced = self.cross_vss_blocks[0](x=P4, guide_x=P5)
        P3_enhanced = self.cross_vss_blocks[1](x=P3, guide_x=P4_enhanced)

        return [P3_enhanced, P4_enhanced, P5]
