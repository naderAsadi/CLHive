import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, act_layer: Callable = nn.GELU):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, n_heads: int, act_layer: Callable = nn.GELU
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(width, n_heads, act_layer=act_layer)
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x


class VisualTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        output_dim: int = 512,
        act_layer: Callable = nn.GELU,
    ):

        super().__init__()

        self.image_size = image_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = embed_dim ** -0.5
        self.num_patches = (image_size // patch_size) ** 2

        self.class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_patches + 1, embed_dim)
        )
        self.ln_pre = LayerNorm(embed_dim)

        self.transformer = Transformer(
            embed_dim, n_layers, n_heads, act_layer=act_layer
        )

        self.ln_post = LayerNorm(embed_dim)
        self.proj = nn.Parameter(scale * torch.randn(embed_dim, output_dim))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert (
            unlocked_groups == 0
        ), "partial locking not currently supported for this model"
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, forward_head: Optional[bool] = True, *args):
        x = self.conv1(x)  # shape = [*, embed_dim, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, embed_dim, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, embed_dim]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, embed_dim]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        cls_embeds = self.ln_post(x[:, 0, :])

        if forward_head:
            cls_embeds = cls_embeds @ self.proj

        return cls_embeds


@register_model("vit_tiny_patch16")
def vit_tiny_patch16(
    image_size: Optional[int] = 224,
    output_dim: Optional[int] = 512,
    pretrained: Optional[bool] = False,
    **kwargs
):
    """ViT-Tiny (Vit-Ti/16)"""
    return VisualTransformer(
        image_size=image_size,
        patch_size=16,
        embed_dim=192,
        n_layers=12,
        n_heads=3,
        output_dim=output_dim,
        **kwargs
    )


@register_model("vit_small_patch16")
def vit_small_patch16(
    image_size: Optional[int] = 224,
    output_dim: Optional[int] = 512,
    pretrained: Optional[bool] = False,
    **kwargs
):
    """ViT-Small (ViT-S/16)"""
    return VisualTransformer(
        image_size=image_size,
        patch_size=16,
        embed_dim=384,
        n_layers=12,
        n_heads=6,
        output_dim=output_dim,
        **kwargs
    )


@register_model("vit_small_patch32")
def vit_small_patch32(
    image_size: Optional[int] = 224,
    output_dim: Optional[int] = 512,
    pretrained: Optional[bool] = False,
    **kwargs
):
    """ViT-Small (ViT-S/32)"""
    return VisualTransformer(
        image_size=image_size,
        patch_size=32,
        embed_dim=384,
        n_layers=12,
        n_heads=6,
        output_dim=output_dim,
        **kwargs
    )


@register_model("vit_base_patch16")
def vit_base_patch16(
    image_size: Optional[int] = 224,
    output_dim: Optional[int] = 512,
    pretrained: Optional[bool] = False,
    **kwargs
):
    """ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    return VisualTransformer(
        image_size=image_size,
        patch_size=16,
        embed_dim=768,
        n_layers=12,
        n_heads=12,
        output_dim=output_dim,
        **kwargs
    )


@register_model("vit_base_patch32")
def vit_base_patch32(
    image_size: Optional[int] = 224,
    output_dim: Optional[int] = 512,
    pretrained: Optional[bool] = False,
    **kwargs
):
    """ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
    """
    return VisualTransformer(
        image_size=image_size,
        patch_size=32,
        embed_dim=768,
        n_layers=12,
        n_heads=12,
        output_dim=output_dim,
        **kwargs
    )


@register_model("vit_large_patch14")
def vit_large_patch14(
    image_size: Optional[int] = 224,
    output_dim: Optional[int] = 512,
    pretrained: Optional[bool] = False,
    **kwargs
):
    """ViT-Large model (ViT-L/14)"""
    return VisualTransformer(
        image_size=image_size,
        patch_size=14,
        embed_dim=1024,
        n_layers=24,
        n_heads=16,
        output_dim=output_dim,
        **kwargs
    )


@register_model("vit_large_patch16")
def vit_large_patch16(
    image_size: Optional[int] = 224,
    output_dim: Optional[int] = 512,
    pretrained: Optional[bool] = False,
    **kwargs
):
    """ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """

    return VisualTransformer(
        image_size=image_size,
        patch_size=16,
        embed_dim=1024,
        n_layers=24,
        n_heads=16,
        output_dim=output_dim,
        **kwargs
    )


@register_model("vit_large_patch32")
def vit_large_patch32(
    image_size: Optional[int] = 224,
    output_dim: Optional[int] = 512,
    pretrained: Optional[bool] = False,
    **kwargs
):
    """ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights."""
    return VisualTransformer(
        image_size=image_size,
        patch_size=32,
        embed_dim=1024,
        n_layers=24,
        n_heads=16,
        output_dim=output_dim,
        **kwargs
    )


@register_model("vit_huge_patch14")
def vit_huge_patch14(
    image_size: Optional[int] = 224,
    output_dim: Optional[int] = 512,
    pretrained: Optional[bool] = False,
    **kwargs
):
    """ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929)."""
    return VisualTransformer(
        image_size=image_size,
        patch_size=14,
        embed_dim=1280,
        n_layers=32,
        n_heads=16,
        output_dim=output_dim,
        **kwargs
    )


@register_model("vit_giant_patch14")
def vit_giant_patch14(
    image_size: Optional[int] = 224,
    output_dim: Optional[int] = 512,
    pretrained: Optional[bool] = False,
    **kwargs
):
    """ViT-Giant model (ViT-g/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560"""
    return VisualTransformer(
        image_size=image_size,
        patch_size=14,
        embed_dim=1408,
        n_layers=40,
        n_heads=16,
        output_dim=output_dim,
        **kwargs
    )
