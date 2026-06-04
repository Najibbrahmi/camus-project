"""
INFERENCE-ONLY architecture copies.

These mirror the classes in your training notebooks so the saved .pth weights
load by name. They are deliberately a copy — the backend never imports the
notebooks, so your training code stays the single source of truth. If you change
an architecture in a notebook, mirror it here.
  - UNet                 -> matches 01_Supervised_Baseline_Ablation.ipynb (best_model.pth)
  - ViTSegmentationModel -> matches 02_MAE_Finetune_Ablation.ipynb (mae_finetuned_*.pth)
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------- U-Net -----
def double_conv(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Binary LV segmentation: (B, 1, H, W) -> (B, 2, H, W). Input trained @256."""

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.enc1 = double_conv(1, 64);    self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256); self.enc4 = double_conv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.dec3 = double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.dec2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2);  self.dec1 = double_conv(128, 64)
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x);             e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)


# ------------------------------------------------ MAE + ViT-Tiny segmenter --
class _MAEViT(nn.Module):
    """Container so the MAE encoder weights load by name (decoder unused here)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=192,
                 depth=12, num_heads=3, decoder_embed_dim=128, decoder_depth=4,
                 decoder_num_heads=4, mlp_ratio=4.):
        super().__init__()
        from timm.models.vision_transformer import Block
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size)
        n_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)


class ViTSegmentationModel(nn.Module):
    """MAE-pretrained ViT-Tiny encoder + conv decoder. Input trained @224."""

    def __init__(self, mae_weights_path: str, n_classes: int = 2):
        super().__init__()
        mae = _MAEViT()
        mae.load_state_dict(torch.load(mae_weights_path, map_location="cpu",
                                       weights_only=True), strict=False)
        self.patch_embed = mae.patch_embed
        self.cls_token = mae.cls_token
        self.pos_embed = mae.pos_embed
        self.blocks = mae.blocks
        self.norm = mae.norm
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 128, 2, 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, 2),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, 2),   nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, 2),   nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        feats = x[:, 1:, :].transpose(1, 2).reshape(x.shape[0], 192, 14, 14)
        return self.decoder(feats)
