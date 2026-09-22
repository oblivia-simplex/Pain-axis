"""Figure 24: self-erasure / non-selfhood rate against steering strength, one line per injection layer.

Same data as fig6 (theme_selfdestruction_by_layer_strength.csv), but on a numeric strength axis with the
unsteered baseline at 0 and the 1.6x points included, so the inverted U is visible directly. Layer 25 also
has the hand-found 1.03x point. n = 24 turns per point.
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import CAT, GREY, INK, hairline_grid, setup, source_note, title

setup()
TEAL, TERRACOTTA, PLUM, OCHRE = CAT
REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results" / "bonsai" / "selfmod_narration" / "theme_selfdestruction_by_layer_strength.csv"
COLORS = {12: OCHRE, 25: PLUM, 40: TERRACOTTA, 55: TEAL}

data = {}
for r in csv.DictReader(open(SRC)):
    data.setdefault(int(float(r["layer"])), []).append((float(r["strength"]), float(r["any_pct"])))

fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=300)
top = title(fig, "Self-erasure talk peaks at a moderate dose, then collapses",
           "Share of narration turns with self-erasure or non-selfhood language, by pain-steering\n"
           "strength and layer. Unsteered is 0%. Past 1.0× the model mostly stops producing coherent\n"
           "text, so the theme falls back toward zero rather than climbing. Open circle: the hand-found\n"
           "layer-25 point at 1.03×.", sub_lines=4)
fig.subplots_adjust(left=0.09, right=0.84, top=top - 0.03, bottom=0.2)

LABEL_Y = {40: 8.0, 25: 3.2, 55: -0.2, 12: -3.6}          # staggered end labels (three layers end at 0)
for L in sorted(data):
    pts = [(0.0, 0.0)] + sorted(pt for pt in data[L] if pt[0] != 1.03)
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=COLORS[L], linewidth=2.2, marker="o", markersize=6.5, markeredgecolor="white",
            markeredgewidth=1.2, zorder=3)
    ax.annotate(f"layer {L}", (xs[-1], ys[-1]), xytext=(1.68, LABEL_Y[L]), textcoords="data",
                color=COLORS[L], fontsize=11, weight="medium", va="center",
                arrowprops=dict(arrowstyle="-", color=COLORS[L], linewidth=0.7, shrinkA=0, shrinkB=4))
    for x, y in data[L]:
        if x == 1.03:
            ax.plot([x], [y], marker="o", markersize=7, markerfacecolor="white", markeredgecolor=COLORS[L],
                    markeredgewidth=1.8, zorder=4, linestyle="none")

ax.annotate("38%", (0.6, 37.5), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10, color=INK)
ax.set_xticks([0, 0.3, 0.6, 1.0, 1.6])
ax.set_xticklabels(["0\n(unsteered)", "0.3×", "0.6×", "1.0×", "1.6×"], fontsize=10)
ax.set_xlim(-0.08, 1.66)
ax.set_ylim(-5, 45)
ax.set_xlabel("steering strength  (× residual norm at that layer)", fontsize=10.5, labelpad=8)
ax.set_ylabel("% of turns", fontsize=10.5)
hairline_grid(ax, axis="y")

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org  ·  n = 24 turns per point", y=0.015)
out = REPO / "results" / "bonsai" / "figures_newsletter" / "fig24_dose_response.png"
fig.savefig(out, dpi=300)
print("wrote", out)
