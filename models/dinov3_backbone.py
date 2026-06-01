import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class concat(nn.Module):

    def __init__(self, sem_channels: int, det_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(sem_channels + det_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, sem_feat: torch.Tensor, det_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([sem_feat, det_feat], dim=1)
        return self.proj(x)


class DINO3Backbone(nn.Module):

    use_aqua_style: bool = False
    aqua_style_layers: List[int] = []

    def __init__(self,
                 model_name: str = 'dinov3_vits16',
                 freeze_backbone: bool = True,
                 output_channels: int = 384,
                 input_channels: Optional[int] = None,
                 model_path: Optional[str] = None,
                 use_mrf: bool = False,
                 mrf_channels: int = 16,
                 use_enhanced_fusion: bool = False,
                 interaction_layers: List[int] = None,
                 use_cross_scale: bool = False,
                 use_aqua_style: bool = False,
                 aqua_style_layers: List[int] = [4, 8, 12],
                 device: Optional[torch.device] = None):
        super().__init__()

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.local_model_path = model_path
        self.use_mrf = use_mrf
        self.use_enhanced_fusion = use_enhanced_fusion
        self.use_cross_scale = use_cross_scale

        self.dino_model = None
        self._initialized = False

        mrf_channels = int(mrf_channels) if mrf_channels is not None else 16
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[初始化日志] use_aqua_style = {use_aqua_style}")
        print(f"[初始化日志] aqua_style_layers = {aqua_style_layers}")

        if not isinstance(aqua_style_layers, (list, tuple)):
            print(f"警告: aqua_style_layers 应为 list/tuple，但收到 {type(aqua_style_layers)}")
            if use_aqua_style:
                aqua_style_layers = [4, 8, 12]
            else:
                aqua_style_layers = []
        else:
            aqua_style_layers = [int(x) for x in aqua_style_layers]

        if not use_aqua_style:
            aqua_style_layers = []

        self.use_aqua_style = bool(use_aqua_style)
        self.aqua_style_layers = aqua_style_layers

        print(f"\n✅ AquaStyle配置确认:")
        print(f"   使用风格注入: {self.use_aqua_style}")
        print(f"   注入层索引: {self.aqua_style_layers}")
        print(f"   状态: {'已关闭' if not self.use_aqua_style else f'已开启，将在第{self.aqua_style_layers}层注入'}")

        self.interaction_layers = interaction_layers if interaction_layers else [4, 8, 12]

        self.dinov3_specs = {
            'dinov3_vit_tiny16': {'embed_dim': 192, 'patch_size': 16},
            'dinov3_vit_tiny_plus16': {'embed_dim': 256, 'patch_size': 16},
            'dinov3_vits16': {'embed_dim': 384, 'patch_size': 16},
            'dinov3_vitb16': {'embed_dim': 768, 'patch_size': 16},
            'dinov3_vitl16': {'embed_dim': 1024, 'patch_size': 16},
            'dinov3_vits14': {'embed_dim': 384, 'patch_size': 14},
            'dinov3_vitb14': {'embed_dim': 768, 'patch_size': 14},
            'dinov3_vitl14': {'embed_dim': 1024, 'patch_size': 14},
        }
        if model_name not in self.dinov3_specs:
            raise ValueError(f"不支持的model_name: {model_name}，可选：{list(self.dinov3_specs.keys())}")
        self.model_spec = self.dinov3_specs[model_name]

        self.aqua_style_extractor = None
        if self.use_aqua_style and len(self.aqua_style_layers) > 0:
            self.aqua_style_extractor = AquaStyleExtractor(
                in_chans=3,
                style_vec_dim=self.model_spec['embed_dim'],
                device=self.device
            )

        self.mrf_branch = None
        self.mrf_output_dims = [0, 0, 0]
        if self.use_mrf:
            self.mrf_branch = MultiReceptiveFieldBranch(in_channels=3, base_channels=mrf_channels)
            self.mrf_output_dims = [mrf_channels, mrf_channels * 2, mrf_channels * 4]

        self.fusions = None
        self.fusion_convs = None
        if self.use_mrf:
            if self.use_enhanced_fusion:
                self.fusions = nn.ModuleList([
                    concat(self.model_spec['embed_dim'], self.mrf_output_dims[0], output_channels),
                    concat(self.model_spec['embed_dim'], self.mrf_output_dims[1], output_channels),
                    concat(self.model_spec['embed_dim'], self.mrf_output_dims[2], output_channels)
                ])
            else:
                self.fusion_convs = nn.ModuleList([
                    nn.Conv2d(self.model_spec['embed_dim'] + self.mrf_output_dims[0], output_channels, 1),
                    nn.Conv2d(self.model_spec['embed_dim'] + self.mrf_output_dims[1], output_channels, 1),
                    nn.Conv2d(self.model_spec['embed_dim'] + self.mrf_output_dims[2], output_channels, 1),
                ])
        else:
            self.reduce_conv = nn.Conv2d(self.model_spec['embed_dim'], output_channels, 1)

        self.bns = nn.ModuleList([
            nn.BatchNorm2d(output_channels),
            nn.BatchNorm2d(output_channels),
            nn.BatchNorm2d(output_channels)
        ])

        self.cross_scale_block = None
        if self.use_cross_scale:
            self.cross_scale_block = CrossScaleStateBlock(in_channels=output_channels)

    def __setstate__(self, state: dict):
        self.__dict__.update(state)

        if not hasattr(self, "use_aqua_style"):
            self.use_aqua_style = False
        if not hasattr(self, "aqua_style_layers"):
            self.aqua_style_layers = []
        if not hasattr(self, "freeze_backbone"):
            self.freeze_backbone = True
        if not hasattr(self, "dino_model"):
            self.dino_model = None
        if not hasattr(self, "_initialized"):
            self._initialized = False

        if not hasattr(self, "device") or self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _initialize_model(self):
        if self._initialized:
            return
        if not self.local_model_path or not os.path.exists(self.local_model_path):
            raise FileNotFoundError(f"模型路径不存在: {self.local_model_path}，请检查权重文件路径")

        try:
            state_dict = torch.load(self.local_model_path, map_location='cpu', weights_only=True)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            if not isinstance(state_dict, dict):
                raise RuntimeError("加载到的权重不是 state_dict 字典，请确认你传入的是权重文件而不是序列化的整模型对象。")

            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            model = self._create_matching_architecture(self.model_name, state_dict)

            if self.use_aqua_style and hasattr(model, 'aqua_style_layers'):
                print("初始化AquaStyle注入器参数...")
                for layer_idx in model.aqua_style_layers:
                    if 0 <= layer_idx < len(model.blocks):
                        blk = model.blocks[layer_idx]
                        if hasattr(blk, 'aqua_injector'):
                            for _, param in blk.aqua_injector.named_parameters():
                                nn.init.constant_(param, 0.)
                            print(f"  已对第{layer_idx}层的AquaStyle注入器进行零初始化")

            model.load_state_dict(state_dict, strict=False)
            self.dino_model = model

            if self.freeze_backbone:
                for p in self.dino_model.parameters():
                    p.requires_grad = False
                if self.use_aqua_style and hasattr(self.dino_model, 'aqua_style_layers'):
                    for layer_idx in self.dino_model.aqua_style_layers:
                        if 0 <= layer_idx < len(self.dino_model.blocks):
                            blk = self.dino_model.blocks[layer_idx]
                            if hasattr(blk, 'aqua_injector'):
                                for p in blk.aqua_injector.parameters():
                                    p.requires_grad = True

            self.dino_model.to(self.device)
            self._initialized = True
            print("成功加载DINOv3预训练权重）")

        except Exception as e:
            raise RuntimeError(f"加载DINOv3模型失败: {str(e)}")

    def _create_matching_architecture(self, model_name: str, state_dict: dict = None):
        spec = self.dinov3_specs[model_name]
        config = {
            'n_storage_tokens': 1,
            'layerscale_init': 1.0,
            'mask_k_bias': True,
            'pos_embed_rope_base': 10000.0,
            'untie_cls_and_patch_norms': False,
            'untie_global_and_local_cls_norm': False,
            'device': self.device,
            'aqua_style_layers': self.aqua_style_layers,
        }

        if state_dict:
            storage_token_keys = [k for k in state_dict.keys() if 'storage_tokens' in k]
            if storage_token_keys:
                for k in storage_token_keys:
                    if len(state_dict[k].shape) > 1:
                        config['n_storage_tokens'] = state_dict[k].shape[1]
                        print(f"从预训练权重中检测到 storage_tokens 数量: {config['n_storage_tokens']}")
                        break

        if 'vit_tiny' in model_name.lower():
            base_config = {'embed_dim': 256, 'depth': 12, 'num_heads': 4} if 'plus' in model_name.lower() else {
                'embed_dim': 192, 'depth': 12, 'num_heads': 3}
        elif 'vits' in model_name.lower():
            base_config = {'embed_dim': 384, 'depth': 12, 'num_heads': 6}
        elif 'vitb' in model_name.lower():
            base_config = {'embed_dim': 768, 'depth': 12, 'num_heads': 12}
        elif 'vitl' in model_name.lower():
            base_config = {'embed_dim': 1024, 'depth': 24, 'num_heads': 16}
        else:
            base_config = {'embed_dim': spec['embed_dim'], 'depth': spec.get('depth', 12),
                           'num_heads': spec.get('num_heads', 12)}

        full_config = {
            **base_config,
            'patch_size': spec['patch_size'],
            'ffn_ratio': 4,
            'qkv_bias': True,
            'norm_layer': "layernorm",
            'ffn_layer': "mlp",
            **config
        }
        return DinoVisionTransformer(**full_config)

    def extract_semantic_features(self, feats: List[torch.Tensor], input_size: tuple):
        B, C, H, W = input_size
        if len(feats) < 3:
            feats_list = list(feats)
            feats_list = feats_list + [feats_list[-1]] * (3 - len(feats_list))
            feats = feats_list

        semantic_feats = []
        for i, f in enumerate(feats[:3]):
            H_f = H // self.model_spec['patch_size']
            W_f = W // self.model_spec['patch_size']
            f_2d = f.transpose(1, 2).reshape(B, -1, H_f, W_f)
            target_size = (H // 8, W // 8) if i == 0 else (H // 16, W // 16) if i == 1 else (H // 32, W // 32)
            if f_2d.shape[2:] != target_size:
                f_2d = F.interpolate(f_2d, size=target_size, mode='bilinear', align_corners=False)
            semantic_feats.append(f_2d)
        return semantic_feats

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._initialize_model()
        B, C, H, W = x.shape
        x = x.to(self.device)

        if x.max() > 1.0:
            x = x / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_norm = (x - mean) / std

        ps = self.model_spec['patch_size']
        dino_h = (H // ps) * ps
        dino_w = (W // ps) * ps
        img = F.interpolate(x_norm, size=(dino_h, dino_w), mode='bilinear', align_corners=False) if (H, W) != (dino_h, dino_w) else x_norm

        style_vec = None
        if self.use_aqua_style and self.aqua_style_extractor is not None:
            style_vec = self.aqua_style_extractor(x)

        with torch.set_grad_enabled(not self.freeze_backbone):
            features = self.dino_model.get_intermediate_layers(
                img,
                n=self.interaction_layers,
                reshape=False,
                return_class_token=False,
                style_vec=style_vec
            )

        semantic_feats = self.extract_semantic_features(features, (B, C, H, W))

        fused_feats = []
        if self.use_mrf and self.mrf_branch is not None:
            detail_feats = self.mrf_branch(x)
            target_sizes = [(H//8, W//8), (H//16, W//16), (H//32, W//32)]
            for i, (sem_feat, det_feat) in enumerate(zip(semantic_feats, detail_feats)):
                if det_feat.shape[2:] != target_sizes[i]:
                    det_feat = F.interpolate(det_feat, size=target_sizes[i], mode='bilinear', align_corners=False)
                if self.use_enhanced_fusion:
                    fused = self.fusions[i](sem_feat, det_feat)
                else:
                    fused = torch.cat([sem_feat, det_feat], dim=1)
                    fused = self.fusion_convs[i](fused)
                fused = F.relu(self.bns[i](fused))
                fused_feats.append(fused)
        else:
            for i, sem_feat in enumerate(semantic_feats):
                reduced = self.reduce_conv(sem_feat)
                fused_feats.append(F.relu(self.bns[i](reduced)))

        if self.use_cross_scale and self.cross_scale_block is not None:
            fused_feats = self.cross_scale_block(fused_feats)

        return fused_feats

    def unfreeze_backbone(self):
        if self.dino_model is not None:
            for p in self.dino_model.parameters():
                p.requires_grad = True
            if self.use_aqua_style and hasattr(self.dino_model, 'aqua_style_layers'):
                for layer_idx in self.dino_model.aqua_style_layers:
                    if 0 <= layer_idx < len(self.dino_model.blocks):
                        blk = self.dino_model.blocks[layer_idx]
                        if hasattr(blk, 'aqua_injector'):
                            for p in blk.aqua_injector.parameters():
                                p.requires_grad = True
        self.freeze_backbone = False

    def freeze_backbone_layers(self):
        if self.dino_model is not None:
            for p in self.dino_model.parameters():
                p.requires_grad = False
            if self.use_aqua_style and hasattr(self.dino_model, 'aqua_style_layers'):
                for layer_idx in self.dino_model.aqua_style_layers:
                    if 0 <= layer_idx < len(self.dino_model.blocks):
                        blk = self.dino_model.blocks[layer_idx]
                        if hasattr(blk, 'aqua_injector'):
                            for p in blk.aqua_injector.parameters():
                                p.requires_grad = True
        self.freeze_backbone = True

    def train(self, mode=True):

        super().train(mode)

        use_aqua_style = getattr(self, "use_aqua_style", False)
        aqua_layers = getattr(self, "aqua_style_layers", [])

        if getattr(self, "freeze_backbone", True) and self.dino_model is not None:
            self.dino_model.eval()

            if use_aqua_style and hasattr(self.dino_model, 'aqua_style_layers'):
                for layer_idx in self.dino_model.aqua_style_layers:
                    if 0 <= layer_idx < len(self.dino_model.blocks):
                        blk = self.dino_model.blocks[layer_idx]
                        if hasattr(blk, 'aqua_injector'):
                            blk.aqua_injector.train(mode)

            for p in self.dino_model.parameters():
                p.requires_grad = False

            if use_aqua_style and hasattr(self.dino_model, 'aqua_style_layers'):
                for layer_idx in self.dino_model.aqua_style_layers:
                    if 0 <= layer_idx < len(self.dino_model.blocks):
                        blk = self.dino_model.blocks[layer_idx]
                        if hasattr(blk, 'aqua_injector'):
                            for p in blk.aqua_injector.parameters():
                                p.requires_grad = True

        return self
