"""
Qualitative + clinical visualization suite (CAMUS).

Standalone, READ-ONLY — does NOT modify or import your training/model notebooks.
Architectures are re-declared for INFERENCE only, to load your saved checkpoints.

Three modular figures (call any independently):
  1. plot_model_grid(...)  — Original | GT | U-Net | MAE+ViT-Tiny [| KAN if a real
                             KAN *segmenter* exists]; real Dice under each column.
  2. plot_overlay(...)     — Original | GT | Pred-vs-GT (pred green over GT yellow).
  3. plot_ef_analysis(...) — ED/ES x (Input,GT,Pred) grid; title = biplane EF%.

Nothing is fabricated: a model column appears only if its checkpoint loads, and
Dice is COMPUTED against ground truth unless you explicitly enable DICE_OVERRIDE.

Run:  python qualitative_comparison.py
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

try:
    from scipy.ndimage import label as cc_label
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_PATH = "database/"
DISPLAY_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
MAE_BASE = "mae_pretrained_full.pth"
SPACING = (0.3, 0.3)               # mm/px on the grid — your notebook's EF constant
NUM_DISKS = 20

# Colours
COL_GT = "#c0392b"                 # ground truth (red) in the grid
COL_PRED = "#27ae60"               # predictions (green), per request
COL_OVERLAY_GT = "#f1c40f"         # ground truth (yellow) in the overlay plot
COL_OVERLAY_PRED = "#27ae60"       # prediction (green) in the overlay plot

# Rows = the 4 frame types of one patient (label, view, phase).
ROWS = [
    ("2 ch-ed-view", "2CH", "ED"),
    ("2 ch-es-view", "2CH", "ES"),
    ("4 ch-ed-view", "4CH", "ED"),
    ("4 ch-es-view", "4CH", "ES"),
]
PATIENT = None                     # None -> first patient of the test split

# --- Manual Dice override (OFF by default) --------------------------------
# Real Dice is computed from the masks shown. Only enable this if the numbers
# below come from the SAME model+sample you are displaying; otherwise the label
# will not match the mask (misleading). Keys: (model_name, view, phase).
USE_DICE_OVERRIDE = False
DICE_OVERRIDE = {
    ("U-Net", "2CH", "ED"): 0.9393, ("U-Net", "2CH", "ES"): 0.9284,
    ("U-Net", "4CH", "ED"): 0.9212, ("U-Net", "4CH", "ES"): 0.9308,
    ("MAE+ViT-Tiny", "2CH", "ED"): 0.924, ("MAE+ViT-Tiny", "2CH", "ES"): 0.923,
    ("MAE+ViT-Tiny", "4CH", "ED"): 0.932, ("MAE+ViT-Tiny", "4CH", "ES"): 0.930,
}


# ==========================================================================
# Architectures (INFERENCE-ONLY copies — notebooks untouched)
# ==========================================================================
def double_conv(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.enc1 = double_conv(1, 64);    self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256); self.enc4 = double_conv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.dec3 = double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.dec2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2);  self.dec1 = double_conv(128, 64)
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x);             e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)


def load_unet(path):
    model = UNet(n_classes=2).to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


class _MAEViT(nn.Module):
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
    def __init__(self, mae_weights_path, n_classes=2):
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

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        feats = x[:, 1:, :].transpose(1, 2).reshape(x.shape[0], 192, 14, 14)
        return self.decoder(feats)


def load_vit(seg_ckpt):
    if not os.path.exists(MAE_BASE):
        raise FileNotFoundError(f"MAE encoder weights '{MAE_BASE}' not found.")
    model = ViTSegmentationModel(MAE_BASE, n_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(seg_ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def kan_segmenter_available():
    """KAN check. results/ef_head_kan.pth is an EF-REGRESSION head (outputs a
    scalar), not a segmenter — so it cannot contribute a mask column. Returns
    False unless a real KAN *segmentation* checkpoint is found."""
    kan_seg = "kan_segmentation.pth"  # no such artifact exists in this project
    if os.path.exists(kan_seg):
        return True
    if os.path.exists("results/ef_head_kan.pth"):
        print("[info] KAN found, but it is an EF-regression head (no masks) — "
              "excluded from the segmentation grid; see plot_ef_analysis().")
    return False


MODEL_REGISTRY = [
    {"name": "U-Net",        "ckpt": "best_model.pth",                        "load": load_unet, "size": 256},
    {"name": "MAE+ViT-Tiny", "ckpt": "results/mae_finetuned_10pct_seed42.pth", "load": load_vit,  "size": 224},
]


# ==========================================================================
# Data + inference helpers
# ==========================================================================
def get_test_patients():
    pts = sorted(d for d in os.listdir(BASE_PATH)
                 if os.path.isdir(os.path.join(BASE_PATH, d)))
    random.seed(SEED)
    random.shuffle(pts)
    return pts[450:500]


def load_frame(patient, view, phase, size):
    pdir = os.path.join(BASE_PATH, patient)
    img = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(pdir, f"{patient}_{view}_{phase}.nii.gz")))).astype(np.float32)
    gt = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(pdir, f"{patient}_{view}_{phase}_gt.nii.gz")))).astype(np.float32)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img, (gt == 1).astype(np.uint8)


def dice(pred, gt):
    pred, gt = (pred > 0.5), (gt > 0.5)
    inter = np.logical_and(pred, gt).sum()
    return (2.0 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6)


@torch.no_grad()
def predict(model, img):
    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    return torch.argmax(model(x), dim=1)[0].cpu().numpy().astype(np.uint8)


def load_models():
    models = []
    for m in MODEL_REGISTRY:
        if not os.path.exists(m["ckpt"]):
            print(f"[skip] {m['name']}: '{m['ckpt']}' not found.");  continue
        try:
            models.append({**m, "model": m["load"](m["ckpt"])})
            print(f"[ok]   {m['name']}: loaded {m['ckpt']}")
        except Exception as e:
            print(f"[skip] {m['name']}: load failed ({e}).")
    return models


def _dice_label(name, view, phase, computed):
    if USE_DICE_OVERRIDE and (name, view, phase) in DICE_OVERRIDE:
        return f"Dice: {DICE_OVERRIDE[(name, view, phase)]:.3f}*"  # * = manual
    return f"Dice: {computed:.3f}"


# ==========================================================================
# Biplane Simpson EF (inference-only copy of your notebook logic)
# ==========================================================================
def get_largest_cc(mask):
    if not _HAS_SCIPY or mask.sum() == 0:
        return mask
    lbl, n = cc_label(mask)
    if n == 0:
        return mask
    return (lbl == np.bincount(lbl.flat)[1:].argmax() + 1).astype(np.uint8)


def _disk_diameters(mask, spacing=SPACING, num_disks=NUM_DISKS):
    ys, _ = np.where(mask > 0.5)
    if len(ys) == 0:
        return None, None
    y0, y1 = ys.min(), ys.max()
    h_px = y1 - y0
    if h_px == 0:
        return None, None
    h_mm = (h_px / num_disks) * spacing[0]
    diam = [int((mask[int(y0 + (i + 0.5) * h_px / num_disks), :] > 0.5).sum()) * spacing[1]
            for i in range(num_disks)]
    return diam, h_mm


def biplane_volume(m2, m4):
    d2, h2 = _disk_diameters(m2)
    d4, h4 = _disk_diameters(m4)
    if d2 is None or d4 is None:
        return 0.0
    h = 0.5 * (h2 + h4)
    return sum((np.pi / 4) * a * b * h for a, b in zip(d2, d4)) / 1000.0


def compute_ef(model, size, patient):
    """Biplane EF% from a model's masks across all 4 frames (your real method)."""
    masks = {}
    for v in ("2CH", "4CH"):
        for ph in ("ED", "ES"):
            img, _ = load_frame(patient, v, ph, size)
            masks[(v, ph)] = get_largest_cc(predict(model, img))
    edv = biplane_volume(masks[("2CH", "ED")], masks[("4CH", "ED")])
    esv = biplane_volume(masks[("2CH", "ES")], masks[("4CH", "ES")])
    ef = (edv - esv) / edv * 100.0 if edv > 0 else float("nan")
    return ef, edv, esv


# ==========================================================================
# 1) MODEL COMPARISON GRID
# ==========================================================================
def overlay(ax, gray, mask, color, alpha=0.45):
    ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(mask == 0, mask),
              cmap=ListedColormap([color]), alpha=alpha, vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])


def plot_model_grid(patient, models, out="fig_model_grid.png"):
    include_kan = kan_segmenter_available()  # honest check (currently False)
    n_cols = 2 + len(models) + (1 if include_kan else 0)
    fig, axes = plt.subplots(len(ROWS), n_cols, figsize=(2.4 * n_cols, 2.6 * len(ROWS)))
    if len(ROWS) == 1:
        axes = axes[None, :]
    headers = ["Original", "Ground Truth"] + [m["name"] for m in models] + (["KAN"] if include_kan else [])

    for r, (row_label, view, phase) in enumerate(ROWS):
        disp_img, disp_gt = load_frame(patient, view, phase, DISPLAY_SIZE)
        axes[r, 0].imshow(disp_img, cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(row_label, fontsize=12, fontweight="bold", labelpad=10)
        overlay(axes[r, 1], disp_img, disp_gt, COL_GT)

        for c, m in enumerate(models, start=2):
            img_m, gt_m = load_frame(patient, view, phase, m["size"])
            pred = predict(m["model"], img_m)
            d = dice(pred, gt_m)
            pred_disp = cv2.resize(pred, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
            overlay(axes[r, c], disp_img, pred_disp, COL_PRED)   # predictions in green
            axes[r, c].set_xlabel(_dice_label(m["name"], view, phase, d), fontsize=10)

        if r == 0:
            for c in range(n_cols):
                axes[r, c].set_title(headers[c], fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")


# ==========================================================================
# 2) OVERLAY COMPARISON (pred green over GT yellow)
# ==========================================================================
def plot_overlay(patient, view, phase, model, out="fig_overlay.png"):
    disp_img, disp_gt = load_frame(patient, view, phase, DISPLAY_SIZE)
    img_m, _ = load_frame(patient, view, phase, model["size"])
    pred = cv2.resize(predict(model["model"], img_m), (DISPLAY_SIZE, DISPLAY_SIZE),
                      interpolation=cv2.INTER_NEAREST)

    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    ax[0].imshow(disp_img, cmap="gray"); ax[0].set_title("Original")
    overlay(ax[1], disp_img, disp_gt, COL_OVERLAY_GT); ax[1].set_title("Ground Truth")

    ax[2].imshow(disp_img, cmap="gray", vmin=0, vmax=1)
    ax[2].imshow(np.ma.masked_where(disp_gt == 0, disp_gt),
                 cmap=ListedColormap([COL_OVERLAY_GT]), alpha=0.45)   # GT yellow
    ax[2].imshow(np.ma.masked_where(pred == 0, pred),
                 cmap=ListedColormap([COL_OVERLAY_PRED]), alpha=0.45)  # pred green
    ax[2].set_title(f"Prediction vs GT  (Dice {dice(pred, disp_gt):.3f})")
    ax[2].legend(handles=[Patch(color=COL_OVERLAY_GT, label="Ground Truth"),
                          Patch(color=COL_OVERLAY_PRED, label="Prediction")],
                 loc="lower right", fontsize=8)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{patient} · {view}-{phase} · {model['name']}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")


# ==========================================================================
# 3) EJECTION-FRACTION ANALYSIS
# ==========================================================================
def plot_ef_analysis(patient, model, view="4CH", out="fig_ef_analysis.png"):
    ef, edv, esv = compute_ef(model["model"], model["size"], patient)

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for r, phase in enumerate(("ED", "ES")):
        disp_img, disp_gt = load_frame(patient, view, phase, DISPLAY_SIZE)
        img_m, _ = load_frame(patient, view, phase, model["size"])
        pred = cv2.resize(predict(model["model"], img_m), (DISPLAY_SIZE, DISPLAY_SIZE),
                          interpolation=cv2.INTER_NEAREST)
        axes[r, 0].imshow(disp_img, cmap="gray"); axes[r, 0].set_ylabel(phase, fontsize=13, fontweight="bold")
        overlay(axes[r, 1], disp_img, disp_gt, COL_GT)
        overlay(axes[r, 2], disp_img, pred, COL_PRED)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if r == 0:
            for c, t in enumerate((f"{view} Input", "Ground Truth", "Prediction")):
                axes[r, c].set_title(t, fontsize=12, fontweight="bold")

    fig.suptitle(f"Ejection Fraction: {ef:.1f}%   (EDV {edv:.1f} mL · ESV {esv:.1f} mL · biplane Simpson)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}  (EF={ef:.1f}%)")


# ==========================================================================
def main():
    models = load_models()
    if not models:
        raise SystemExit("No models loaded — nothing to visualize.")
    patient = PATIENT or get_test_patients()[0]
    print(f"Patient: {patient}\n")

    plot_model_grid(patient, models)
    plot_overlay(patient, "4CH", "ED", models[0])     # U-Net by default
    plot_ef_analysis(patient, models[0], view="4CH")


if __name__ == "__main__":
    main()
