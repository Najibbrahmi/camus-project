"""
Slide 13 - Headline Result ("The Money Slide")
16:9, dark background, coral accent.

ALL numbers verbatim from the report:
  source: results/master_ablation_table.csv  /  results/summary.txt
  - MAE + ViT-Tiny @ 10% labels : Dice 0.862 / 0.864 , biplane EF MAE 12.65%
  - Supervised UNet @ 100% labels: Dice 0.927 / 0.925 , biplane EF MAE  6.86%
No fabricated values are used.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# ---------------- palette ----------------
BG      = "#0E1117"
PANEL   = "#171C26"
PANEL2  = "#1E2530"
CORAL   = "#FF6F61"
CORAL_D = "#E85A4F"
INK     = "#F2F4F8"
MUTE    = "#9AA4B2"
GRID    = "#2A323F"
HILITE  = "#241A18"

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": BG})

# ---------------- canvas (16:9) ----------------
W, H = 13.333, 7.5
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(Rectangle((0, 0), 1, 1, color=BG, zorder=0))
ax.add_patch(Rectangle((0, 0.986), 1, 0.014, color=CORAL, zorder=2))

# ---------------- kicker + headline (upper-left, short) ----------------
ax.text(0.045, 0.93, "HEADLINE RESULT", color=CORAL, fontsize=15,
        fontweight="bold", ha="left", va="center")
ax.text(0.045, 0.86, "10% of the labels.", color=INK, fontsize=34,
        fontweight="bold", ha="left", va="center")
ax.text(0.045, 0.79, "93% of the Dice.", color=INK, fontsize=34,
        fontweight="bold", ha="left", va="center")

# ---------------- GIANT visual element (center-left) ----------------
ax.text(0.25, 0.585, "10×", color=CORAL, fontsize=165,
        fontweight="bold", ha="center", va="center", zorder=3)
ax.text(0.25, 0.385, "FEWER LABELS", color=INK, fontsize=21,
        fontweight="bold", ha="center", va="center", zorder=3)
ax.text(0.25, 0.345, "10%  vs  100%  annotation budget", color=MUTE,
        fontsize=13.5, ha="center", va="center", zorder=3)

# ---------------- three result chips (right) ----------------
def chip(y, icon, title, big, small):
    x0, w, h = 0.560, 0.395, 0.130
    ax.add_patch(FancyBboxPatch((x0, y), w, h,
                 boxstyle="round,pad=0.006,rounding_size=0.018",
                 facecolor=PANEL, edgecolor=GRID, linewidth=1.4, zorder=3))
    ax.add_patch(Rectangle((x0+0.004, y+0.012), 0.010, h-0.024,
                 facecolor=CORAL, edgecolor="none", zorder=4))
    icx, icy = x0 + 0.058, y + h/2
    ax.scatter([icx], [icy], s=1400, color=PANEL2, edgecolor=CORAL,
               linewidth=1.6, zorder=4)
    ax.text(icx, icy, icon, color=CORAL, fontsize=21, ha="center",
            va="center", zorder=5)
    tx = x0 + 0.108
    ax.text(tx, y + h - 0.040, title, color=MUTE, fontsize=13.5,
            fontweight="bold", ha="left", va="center", zorder=5)
    ax.text(tx, y + 0.048, big, color=INK, fontsize=29, fontweight="bold",
            ha="left", va="center", zorder=5)
    ax.text(x0 + w - 0.022, y + 0.045, small, color=MUTE, fontsize=12.5,
            ha="right", va="center", zorder=5)

chip(0.665, "◎", "DICE SCORE  ·  MAE @10%", "0.86", "vs 0.93 @100%")
chip(0.512, "♥", "EF MAE  ·  MAE @10%",     "12.6%", "vs 6.9% @100%")
chip(0.359, "★", "LABEL EFFICIENCY",         "10×", "fewer labels")

# ---------------- highlighted supporting statement ----------------
sy, sh = 0.222, 0.094
ax.add_patch(FancyBboxPatch((0.045, sy), 0.910, sh,
             boxstyle="round,pad=0.004,rounding_size=0.02",
             facecolor=HILITE, edgecolor=CORAL, linewidth=2.0, zorder=3))
ax.text(0.5, sy + sh*0.62,
        "MAE-pretrained at 10% labels reaches 0.86 Dice — within 0.07 of the",
        color=INK, fontsize=14.5, fontweight="bold", ha="center",
        va="center", zorder=5)
ax.text(0.5, sy + sh*0.30,
        "fully-supervised baseline trained on 100% of the labels.",
        color=INK, fontsize=14.5, fontweight="bold", ha="center",
        va="center", zorder=5)

# ---------------- comparison table (bottom, secondary) ----------------
tx0, tw = 0.045, 0.910
cols = [tx0+0.015, tx0+0.36, tx0+0.52, tx0+0.74]
headers = ["Method", "Labels", "Dice (2CH / 4CH)", "EF MAE"]
rows = [
    ("MAE Pretrained",      "10%",  "0.862 / 0.864", "12.65%", True),
    ("Supervised Baseline", "100%", "0.927 / 0.925", "6.86%",  False),
]
rh = 0.046
hy = 0.150               # header baseline
for cx, htext in zip(cols, headers):
    ax.text(cx, hy, htext, color=MUTE, fontsize=12.5, fontweight="bold",
            ha="left", va="center")
ax.plot([tx0, tx0+tw], [hy-0.024, hy-0.024], color=GRID, lw=1.2)
for i, (m, lab, dice, ef, hl) in enumerate(rows):
    ry = 0.092 - i*rh           # row center
    if hl:
        ax.add_patch(FancyBboxPatch((tx0, ry-rh/2+0.004), tw, rh-0.008,
                     boxstyle="round,pad=0.002,rounding_size=0.012",
                     facecolor=HILITE, edgecolor=CORAL_D, linewidth=1.2,
                     zorder=2))
    for j, (cx, v) in enumerate(zip(cols, [m, lab, dice, ef])):
        col = INK if (hl or j == 0) else MUTE
        fw  = "bold" if (j == 0 or hl) else "normal"
        ax.text(cx, ry, v, color=col, fontsize=13.5, fontweight=fw,
                ha="left", va="center", zorder=4)

ax.text(tx0+tw, 0.012,
        "Source: CAMUS test set · 3 seeds · results/master_ablation_table.csv",
        color="#5C6675", fontsize=9.5, ha="right", va="center")

fig.savefig("figures/slide13_headline_result.png", dpi=300, facecolor=BG)
print("Saved figures/slide13_headline_result.png  (4000x2250 @300dpi)")
