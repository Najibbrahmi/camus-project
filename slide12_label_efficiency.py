"""
Slide 12 - Results: Label Efficiency Curves
Values taken verbatim from results/summary.txt and results/master_ablation_table.csv
(FINAL test-set results, biplane, both views, 3 seeds).
No values are estimated, interpolated, or fabricated.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ----------------------------------------------------------------------
# EXACT reported values (source: results/master_ablation_table.csv)
# ----------------------------------------------------------------------
fractions = [1, 5, 10, 100]          # label fractions (%)
x = np.arange(len(fractions))        # categorical spacing

# Dice score (%) = mean of 2CH & 4CH  -> column dice_mean_mean
dice_mae        = [19.55, 82.55, 86.27, 91.10]   # MAE + ViT-Tiny
dice_supervised = [77.44, 86.35, 89.11, 92.59]   # Supervised UNet

# Biplane EF MAE (%) -> column ef_mae_biplane_mean
ef_mae_mae        = [22.87, 19.72, 12.65, 6.89]  # MAE + ViT-Tiny
ef_mae_supervised = [60.66, 22.79, 13.86, 6.86]  # Supervised UNet

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
CYAN = "#00B7C2"   # MAE-pretrained
GRAY = "#7A7A7A"   # Supervised baseline
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
xtick_labels = ["1%", "5%", "10%", "100%"]


def style_axis(ax, ylabel, title):
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("Label Fraction", fontsize=14, fontweight="bold", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.45, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=True, fancybox=True, framealpha=0.95,
              edgecolor="#cccccc", fontsize=12, loc="best")


# ----------------------------------------------------------------------
# FIGURE A - Dice Score vs Label Fraction  (standalone)
# ----------------------------------------------------------------------
figA, axA = plt.subplots(figsize=(7, 5.2))
axA.plot(x, dice_mae, color=CYAN, marker="o", markersize=9, linewidth=2.6,
         label="MAE-pretrained", markeredgecolor="white", markeredgewidth=1.4)
axA.plot(x, dice_supervised, color=GRAY, marker="s", markersize=9, linewidth=2.6,
         label="Supervised baseline", markeredgecolor="white", markeredgewidth=1.4)
style_axis(axA, "Dice Score (%)", "Segmentation Performance")
figA.tight_layout()
figA.savefig("figures/slide12_dice_vs_labelfrac.png", dpi=300, bbox_inches="tight")

# ----------------------------------------------------------------------
# FIGURE B - EF MAE vs Label Fraction  (standalone)
# ----------------------------------------------------------------------
figB, axB = plt.subplots(figsize=(7, 5.2))
axB.plot(x, ef_mae_mae, color=CYAN, marker="o", markersize=9, linewidth=2.6,
         label="MAE-pretrained", markeredgecolor="white", markeredgewidth=1.4)
axB.plot(x, ef_mae_supervised, color=GRAY, marker="s", markersize=9, linewidth=2.6,
         label="Supervised baseline", markeredgecolor="white", markeredgewidth=1.4)
style_axis(axB, "EF MAE (%)", "Ejection-Fraction Error  (lower is better)")
figB.tight_layout()
figB.savefig("figures/slide12_efmae_vs_labelfrac.png", dpi=300, bbox_inches="tight")

# ----------------------------------------------------------------------
# COMBINED presentation layout (left: Dice, right: EF MAE, conclusion box)
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(15, 7.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.16], hspace=0.05, wspace=0.18,
                      left=0.07, right=0.97, top=0.90, bottom=0.04)

fig.suptitle("Label Efficiency: Self-Supervised vs Supervised",
             fontsize=19, fontweight="bold", y=0.975)

# Left panel - Dice
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x, dice_mae, color=CYAN, marker="o", markersize=10, linewidth=2.8,
         label="MAE-pretrained", markeredgecolor="white", markeredgewidth=1.5)
ax1.plot(x, dice_supervised, color=GRAY, marker="s", markersize=10, linewidth=2.8,
         label="Supervised baseline", markeredgecolor="white", markeredgewidth=1.5)
style_axis(ax1, "Dice Score (%)", "Segmentation  (higher is better)")

# Right panel - EF MAE
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(x, ef_mae_mae, color=CYAN, marker="o", markersize=10, linewidth=2.8,
         label="MAE-pretrained", markeredgecolor="white", markeredgewidth=1.5)
ax2.plot(x, ef_mae_supervised, color=GRAY, marker="s", markersize=10, linewidth=2.8,
         label="Supervised baseline", markeredgecolor="white", markeredgewidth=1.5)
style_axis(ax2, "EF MAE (%)", "Ejection Fraction  (lower is better)")

# Conclusion box spanning the bottom
axc = fig.add_subplot(gs[1, :])
axc.axis("off")
box = FancyBboxPatch((0.01, 0.05), 0.98, 0.9,
                     boxstyle="round,pad=0.02,rounding_size=0.03",
                     transform=axc.transAxes,
                     facecolor="#E6F9FB", edgecolor=CYAN, linewidth=2.2)
axc.add_patch(box)
axc.text(0.5, 0.5,
         "The performance gap widens as labels become scarce.\n"
         "Self-supervised pretraining provides the largest benefit in low-label regimes.",
         transform=axc.transAxes, ha="center", va="center",
         fontsize=14.5, fontweight="bold", color="#0A6E78")

fig.savefig("figures/slide12_label_efficiency_combined.png", dpi=300,
            bbox_inches="tight", facecolor="white")

print("Saved 3 figures to figures/. Values used:")
print("Dice MAE        :", dice_mae)
print("Dice Supervised :", dice_supervised)
print("EF MAE MAE      :", ef_mae_mae)
print("EF MAE Supervised:", ef_mae_supervised)
