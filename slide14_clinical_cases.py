"""
Slide 14 - Clinical Case Studies figure.
Real outputs only: segmentation by the trained U-Net (best_model.pth); EF by
biplane Simpson's method computed from the EXPERT masks (GT) and the PREDICTED
masks (Pred). Three real CAMUS test patients spanning the clinical spectrum.

  python slide14_clinical_cases.py   ->  outputs/slide14_clinical_cases.png
"""
import os, sys
sys.path.insert(0, "backend")
import numpy as np
import torch
import cv2
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from inference.models import UNet
from inference.simpson import get_largest_cc, biplane_volume

BASE = "database/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GT_GREEN, PRED_RED = "#2ecc71", "#e74c3c"
DISPLAY_VIEW = "4CH"   # frames shown (EF itself is biplane: 2CH + 4CH)

# Three real test patients (selected to span the clinical spectrum).
CASES = [
    {"title": "Normal (High EF)",        "pid": "patient0173"},
    {"title": "Heart Failure (Low EF)",  "pid": "patient0113"},
    {"title": "Adverse Acoustics (Hard)", "pid": "patient0120"},
]

model = UNet().to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE, weights_only=True))
model.eval()


def load(pid, v, ph, size=256):
    img = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(f"{BASE}{pid}/{pid}_{v}_{ph}.nii.gz"))).astype(np.float32)
    gt = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(f"{BASE}{pid}/{pid}_{v}_{ph}_gt.nii.gz"))).astype(np.float32)
    img = cv2.resize(img, (size, size))
    gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img, (gt == 1).astype(np.uint8)


@torch.no_grad()
def seg(img):
    x = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    return get_largest_cc(torch.argmax(model(x), 1)[0].cpu().numpy().astype(np.uint8))


def overlay(ax, gray, mask, color):
    ax.imshow(gray, cmap="gray")
    ax.imshow(np.ma.masked_where(mask == 0, mask), cmap=ListedColormap([color]), alpha=0.45)
    ax.set_xticks([]); ax.set_yticks([])


# Compute real biplane EF (GT and Pred) + cache display frames/masks per case.
for c in CASES:
    pid = c["pid"]
    gm, pm, disp = {}, {}, {}
    for v in ("2CH", "4CH"):
        for ph in ("ED", "ES"):
            img, g = load(pid, v, ph)
            gm[(v, ph)] = g; pm[(v, ph)] = seg(img)
            if v == DISPLAY_VIEW:
                disp[ph] = (img, g, pm[(v, ph)])
    gedv = biplane_volume(gm[("2CH", "ED")], gm[("4CH", "ED")]); gesv = biplane_volume(gm[("2CH", "ES")], gm[("4CH", "ES")])
    pedv = biplane_volume(pm[("2CH", "ED")], pm[("4CH", "ED")]); pesv = biplane_volume(pm[("2CH", "ES")], pm[("4CH", "ES")])
    c["gt_ef"] = (gedv - gesv) / gedv * 100
    c["pred_ef"] = (pedv - pesv) / pedv * 100
    c["disp"] = disp

# --- Figure: 2 rows (ED/ES) x 9 cols (3 cases x [Original|Ground Truth|Prediction]) ---
fig, axes = plt.subplots(2, 9, figsize=(15, 5.6))
sub = ["Original", "Ground Truth", "Prediction"]
for ci, c in enumerate(CASES):
    base = ci * 3
    for ri, ph in enumerate(("ED", "ES")):
        img, gt, pred = c["disp"][ph]
        axes[ri, base + 0].imshow(img, cmap="gray"); axes[ri, base + 0].set_xticks([]); axes[ri, base + 0].set_yticks([])
        overlay(axes[ri, base + 1], img, gt, GT_GREEN)
        overlay(axes[ri, base + 2], img, pred, PRED_RED)
        if ri == 0:
            for k in range(3):
                axes[0, base + k].set_title(sub[k], fontsize=9, color="#444")
        if base == 0:
            axes[ri, 0].set_ylabel(ph, fontsize=12, fontweight="bold", rotation=0, labelpad=18, va="center")
    # Case header spanning its 3 columns
    x_center = (axes[0, base].get_position().x0 + axes[0, base + 2].get_position().x1) / 2
    fig.text(x_center, 0.965, c["title"], ha="center", fontsize=13, fontweight="bold")
    fig.text(x_center, 0.935, f"GT {c['gt_ef']:.1f}%  |  Pred {c['pred_ef']:.1f}%",
             ha="center", fontsize=11, color="#2c3e50")

# Legend + caption
fig.legend(handles=[plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=GT_GREEN, markersize=11, label="Ground Truth"),
                    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=PRED_RED, markersize=11, label="Prediction")],
           loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.02))
fig.text(0.5, -0.01,
         "Robust across the clinical spectrum; failures remain localized and explainable. "
         "EF by biplane Simpson's method (4-chamber view shown).",
         ha="center", fontsize=9, style="italic", color="#555")

plt.tight_layout(rect=[0.0, 0.06, 1.0, 0.92])
os.makedirs("outputs", exist_ok=True)
out = "outputs/slide14_clinical_cases.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved:", out)
for c in CASES:
    print(f"  {c['title']:26s} {c['pid']}  GT {c['gt_ef']:.1f}%  Pred {c['pred_ef']:.1f}%")
