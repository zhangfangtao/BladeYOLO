

from typing import Callable, List, Optional, Tuple, Any

import torch
from torch import Tensor, nn
import torch.fft as fft

from dinov3.utils import cat_keep_shapes, uncat_with_shapes

from .attention import CausalSelfAttention, SelfAttention
from .ffn_layers import Mlp
from .layer_scale import LayerScale

torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.accumulated_cache_size_limit = 1024


class AquaStyleExtractor(nn.Module):
\
\
\

    def __init__(
        self,
        in_chans: int = 3,
        base_channels: int = 16,
        style_vec_dim: int = 384,
        device: torch.device = None
    ):
        super().__init__()
        self.style_encoder = nn.Sequential(
            nn.Conv2d(in_chans, base_channels, kernel_size=3, padding=1, bias=False, device=device),
            nn.BatchNorm2d(base_channels, device=device),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False, device=device),
            nn.BatchNorm2d(base_channels * 2, device=device),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        ).to(device) if device is not None else nn.Sequential(
            nn.Conv2d(in_chans, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.style_proj = nn.Linear(base_channels * 2, style_vec_dim, device=device)
        self.output_norm = nn.LayerNorm(style_vec_dim, device=device) if device is not None else nn.LayerNorm(style_vec_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape


        if torch.isnan(x).any() or torch.isinf(x).any():

            x_clean = self._clean_input_data(x)
        else:
            x_clean = x

        try:
            x_fft = fft.fft2(x_clean, dim=(-2, -1))
            amplitude = torch.abs(x_fft)
            phase = torch.angle(x_fft)

            avg_phase = phase.mean(dim=(0, 2, 3), keepdim=True)

            style_fft = amplitude * torch.exp(1j * avg_phase)
            style_img = torch.abs(fft.ifft2(style_fft, dim=(-2, -1)))

            style_max = style_img.max()
            style_min = style_img.min()
            diff = style_max - style_min

            if torch.isnan(style_img).any() or torch.isinf(style_img).any():
                return torch.zeros((B, self.style_proj.out_features), device=x.device)

            if diff.abs() < 1e-6:
                style_img = torch.zeros_like(style_img)
            else:
                style_img = (style_img - style_min) / (diff + 1e-6)

            style_feat = self.style_encoder(style_img)
            style_vec = self.style_proj(style_feat)
            style_vec = self.output_norm(style_vec)

            return style_vec

        except Exception as e:

            return torch.zeros((B, self.style_proj.out_features), device=x.device)

    def _clean_input_data(self, x: torch.Tensor) -> torch.Tensor:

        x_clean = x.clone()


        if torch.isnan(x_clean).any():
            x_clean = torch.nan_to_num(x_clean, nan=0.0)


        if torch.isinf(x_clean).any():
            x_clean = torch.clamp(x_clean, min=-10.0, max=10.0)

        return x_clean


class PaperAquaStyleInjector(nn.Module):
\
\
\

    def __init__(self, embed_dim: int, num_heads: int = 6, drop_rate: float = 0.1, init_values: float = 1e-5):
        super().__init__()
        self.embed_dim = embed_dim

        self.injection_scale = 0.001


        self.style_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )


        self.style_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=drop_rate,
            batch_first=True
        )

        self.ls_style_attn = LayerScale(embed_dim, init_values=1e-6, inplace=False)


        self.style_ff_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        self.ls_style_ff = LayerScale(embed_dim, init_values=1e-6, inplace=False)

    def forward(self, V_in: Tensor, mha_output: Tensor, ffn_output: Tensor, style_vec: Tensor) -> Tuple[Tensor, Tensor]:
        B, N, C = V_in.shape


        if torch.isnan(mha_output).any() or torch.isinf(mha_output).any():
            return mha_output, ffn_output
        if torch.isnan(ffn_output).any() or torch.isinf(ffn_output).any():
            return mha_output, ffn_output


        projected_style = self.style_proj(style_vec)
        style_vec_expand = projected_style.unsqueeze(1).expand(B, N, C)

        try:

            style_attn_output, _ = self.style_cross_attn(
                query=V_in,
                key=style_vec_expand,
                value=style_vec_expand,
                need_weights=False
            )
        except Exception as e:
            return mha_output, ffn_output


        scaled_style_attn = style_attn_output * self.injection_scale
        omega1 = mha_output + self.ls_style_attn(scaled_style_attn)

        if torch.isnan(omega1).any() or torch.isinf(omega1).any():
            return mha_output, ffn_output

        try:

            style_ff_output = self.style_ff_adapter(omega1)
        except Exception as e:
            return omega1, ffn_output


        scaled_style_ff = style_ff_output * self.injection_scale
        omega2 = ffn_output + self.ls_style_ff(scaled_style_ff)

        if torch.isnan(omega2).any() or torch.isinf(omega2).any():
            return omega1, ffn_output

        return omega1, omega2


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,
        use_aqua_style: bool = False,
        aqua_drop_rate: float = 0.1
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, inplace=False, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, inplace=False, device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

        self.use_aqua_style = use_aqua_style
        if self.use_aqua_style:
            self.paper_injector = PaperAquaStyleInjector(
                embed_dim=dim,
                num_heads=num_heads,
                drop_rate=aqua_drop_rate,
                init_values=init_values if isinstance(init_values, float) else 1e-5
            )
        else:
            self.paper_injector = None

    @staticmethod
    def _maybe_index_rope(rope: tuple[Tensor, Tensor] | None, indices: Tensor) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            return sin[indices], cos[indices]
        else:
            return sin, cos

    def _forward(self, x: Tensor, rope=None, style_vec: Optional[Tensor] = None) -> Tensor:
        V_in = x

        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size


        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]
            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)
            mha_output = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )
        else:
            mha_output = x + self.ls1(self.attn(self.norm1(x), rope=rope))


        if self.use_aqua_style and style_vec is not None and self.paper_injector is not None:
            omega1, _ = self.paper_injector(
                V_in=V_in,
                mha_output=mha_output,
                ffn_output=mha_output,
                style_vec=style_vec
            )
            mha_output_for_ffn = omega1
        else:
            mha_output_for_ffn = mha_output


        if self.training and self.sample_drop_ratio > 0.0:
            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]
            x_subset_2 = mha_output_for_ffn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))
            ffn_output = torch.index_add(
                mha_output_for_ffn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            ffn_output = mha_output_for_ffn + self.ls2(self.mlp(self.norm2(mha_output_for_ffn)))


        if self.use_aqua_style and style_vec is not None and self.paper_injector is not None:

            _, final_output = self.paper_injector(
                V_in=V_in,
                mha_output=mha_output_for_ffn,
                ffn_output=ffn_output,
                style_vec=style_vec
            )
        else:
            final_output = ffn_output

        return final_output

    def _forward_list(self, x_list: List[Tensor], rope_list=None, style_vec_list: Optional[List[Tensor]] = None) -> List[Tensor]:
        if style_vec_list is None:
            style_vec_list = [None] * len(x_list)

        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        mha_output_list = []
        mha_output_for_ffn_list = []
        ffn_output_list = []

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            mha_output_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]


            if self.use_aqua_style and style_vec_list is not None and self.paper_injector is not None:
                for i, (V_in, mha_output, style_vec) in enumerate(zip(x_list, mha_output_list, style_vec_list)):
                    if style_vec is not None:
                        omega1, _ = self.paper_injector(
                            V_in=V_in,
                            mha_output=mha_output,
                            ffn_output=mha_output,
                            style_vec=style_vec
                        )
                        mha_output_for_ffn_list.append(omega1)
                    else:
                        mha_output_for_ffn_list.append(mha_output)
            else:
                mha_output_for_ffn_list = mha_output_list

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(mha_output_for_ffn_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(mha_output_for_ffn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            ffn_output_list = [
                torch.index_add(
                    mha_output,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for mha_output, residual_2, indices_2, residual_scale_factor in zip(
                    mha_output_for_ffn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            for x, rope, style_vec in zip(x_list, rope_list, style_vec_list):
                mha_output = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                mha_output_list.append(mha_output)


                if self.use_aqua_style and style_vec is not None and self.paper_injector is not None:
                    omega1, _ = self.paper_injector(
                        V_in=x,
                        mha_output=mha_output,
                        ffn_output=mha_output,
                        style_vec=style_vec
                    )
                    mha_output_for_ffn_list.append(omega1)
                else:
                    mha_output_for_ffn_list.append(mha_output)

                ffn_output = mha_output_for_ffn_list[-1] + self.ls2(self.mlp(self.norm2(mha_output_for_ffn_list[-1])))
                ffn_output_list.append(ffn_output)


        if self.use_aqua_style and style_vec_list is not None and self.paper_injector is not None:
            final_output_list = []
            for i, (V_in, mha_output_for_ffn, ffn_output, style_vec) in enumerate(zip(x_list, mha_output_for_ffn_list, ffn_output_list, style_vec_list)):
                if style_vec is not None:

                    _, final_output = self.paper_injector(
                        V_in=V_in,
                        mha_output=mha_output_for_ffn,
                        ffn_output=ffn_output,
                        style_vec=style_vec
                    )
                    final_output_list.append(final_output)
                else:
                    final_output_list.append(ffn_output)
            return final_output_list
        else:
            return ffn_output_list

    def forward(self, x_or_x_list, rope_or_rope_list=None, style_vec_or_list: Optional[Any] = None) -> Tensor | List[Tensor]:
        if isinstance(x_or_x_list, Tensor):
            style_vec = style_vec_or_list if (style_vec_or_list is None or isinstance(style_vec_or_list, Tensor)) else None
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list], style_vec_list=[style_vec])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]

            if not isinstance(style_vec_or_list, list) and style_vec_or_list is not None:
                if isinstance(style_vec_or_list, Tensor):
                    style_vec_list = [style_vec_or_list] * len(x_or_x_list)
                else:
                    style_vec_list = [None] * len(x_or_x_list)
            else:
                style_vec_list = style_vec_or_list or [None] * len(x_or_x_list)

            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list, style_vec_list=style_vec_list)
        else:
            raise AssertionError


class CausalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        is_causal: bool = True,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
        use_aqua_style: bool = False,
        aqua_drop_rate: float = 0.1
    ):
        super().__init__()

        self.dim = dim
        self.is_causal = is_causal
        self.ls1 = LayerScale(dim, init_values=ls_init_value, inplace=False) if ls_init_value else nn.Identity()
        self.attention_norm = norm_layer(dim)
        self.attention = CausalSelfAttention(dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob)

        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = LayerScale(dim, init_values=ls_init_value, inplace=False) if ls_init_value else nn.Identity()

        self.use_aqua_style = use_aqua_style
        if self.use_aqua_style:
            self.paper_injector = PaperAquaStyleInjector(
                embed_dim=dim,
                num_heads=num_heads,
                drop_rate=aqua_drop_rate,
                init_values=ls_init_value if ls_init_value else 1e-5
            )
        else:
            self.paper_injector = None

    def init_weights(
        self,
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        init_fc_std: float | None = None,
        factor: float = 1.0,
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        init_fc_std = init_fc_std or (2 * self.dim) ** -0.5
        self.attention.init_weights(init_attn_std, init_proj_std)
        self.attention_norm.reset_parameters()
        nn.init.normal_(self.feed_forward.fc1.weight, std=init_fc_std)
        nn.init.normal_(self.feed_forward.fc2.weight, std=init_proj_std)
        self.ffn_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        style_vec: Optional[Tensor] = None
    ):
        V_in = x

        mha_output = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))


        if self.use_aqua_style and style_vec is not None and self.paper_injector is not None:
            omega1, _ = self.paper_injector(
                V_in=V_in,
                mha_output=mha_output,
                ffn_output=mha_output,
                style_vec=style_vec
            )
            mha_output_for_ffn = omega1
        else:
            mha_output_for_ffn = mha_output

        ffn_output = mha_output_for_ffn + self.ls2(self.feed_forward(self.ffn_norm(mha_output_for_ffn)))


        if self.use_aqua_style and style_vec is not None and self.paper_injector is not None:

            _, final_output = self.paper_injector(
                V_in=V_in,
                mha_output=mha_output_for_ffn,
                ffn_output=ffn_output,
                style_vec=style_vec
            )
        else:
            final_output = ffn_output

        return final_output
