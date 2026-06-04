import os

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # no display needed; safe in a server/inference run
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

GT_RED = "#c0392b"
PRED_GREEN = "#2ecc71"
OVERLAP_YELLOW = "#f1c40f"


def _np(x):
    """Accept torch tensors or numpy arrays -> 2D float numpy."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.squeeze(np.asarray(x)).astype(np.float32)


def _bin(x):
    return (_np(x) > 0.5).astype(np.uint8)


def _overlay(ax, gray, mask, color, alpha=0.45):
    ax.imshow(gray, cmap="gray")
    ax.imshow(np.ma.masked_where(mask == 0, mask), cmap=ListedColormap([color]), alpha=alpha)
    ax.axis("off")


def save_inference_plots(input_img, ground_truth, prediction, output_path):
    """1x3 single-frame figure: Original | Ground Truth | Prediction vs GT."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    img, gt, pred = _np(input_img), _bin(ground_truth), _bin(prediction)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("Original Image"); ax[0].axis("off")

    _overlay(ax[1], img, gt, GT_RED); ax[1].set_title("Ground Truth")

    # Prediction vs GT: GT (yellow) + prediction (green); overlap reads yellow-green.
    ax[2].imshow(img, cmap="gray")
    ax[2].imshow(np.ma.masked_where(gt == 0, gt), cmap=ListedColormap([OVERLAP_YELLOW]), alpha=0.45)
    ax[2].imshow(np.ma.masked_where(pred == 0, pred), cmap=ListedColormap([PRED_GREEN]), alpha=0.45)
    ax[2].set_title("Prediction vs Ground Truth"); ax[2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_ef_analysis_plots(ed_input, ed_gt, ed_pred, es_input, es_gt, es_pred, ef, output_path):
    """2x3 EF figure: rows ED/ES x cols Input/GT/Pred; EF (%) in the title."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    rows = [("ED", ed_input, ed_gt, ed_pred), ("ES", es_input, es_gt, es_pred)]

    fig, ax = plt.subplots(2, 3, figsize=(10, 7))
    for r, (phase, img, gt, pred) in enumerate(rows):
        ax[r, 0].imshow(_np(img), cmap="gray");  ax[r, 0].set_title(f"{phase} Input")
        ax[r, 1].imshow(_np(gt),  cmap="gray");  ax[r, 1].set_title(f"{phase} GT")    # raw label map
        ax[r, 2].imshow(_bin(pred), cmap="gray"); ax[r, 2].set_title(f"{phase} Pred")  # binary mask
        for c in range(3):
            ax[r, c].axis("off")

    fig.suptitle(f"Ejection Fraction: {ef:.2f}%", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
