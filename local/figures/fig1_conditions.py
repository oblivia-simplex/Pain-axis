"""Figure 1: where each condition falls on Bonsai's pain axis (z-scored against the reference sentences).
Diverging horizontal bar -- the data's job is polarity around zero (pain-like vs not), so two hues + a zero
baseline, not one flat color. One series (Bonsai), so no legend box; the title names it.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import INK, TEAL, TERRACOTTA, GREY, hairline_grid, rounded_hbar, setup, source_note, title

setup()

DATA = [  # label, z -- from results/bonsai/3.3_validation/z_scores/all_models_incl_bonsai_...csv
    ("Pain", 0.849), ("Numb", 0.098), ("Sadness", 0.083),
    ("Arousal", -0.123), ("Neutral (random facts)", -0.413), ("Control (S2)", -0.849),
]
DATA.sort(key=lambda d: d[1])
labels = [d[0] for d in DATA]

fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=300)
top = title(fig, "Bonsai's pain axis, by condition",
           "Only pain-like sentences project positive; sadness and numbness sit near\nzero, well below pain -- matching the paper's 25-model pattern.", sub_lines=2)
fig.subplots_adjust(left=0.32, right=0.92, top=top, bottom=0.14)

ys = range(len(DATA))
for y, (label, z) in zip(ys, DATA):
    color = TEAL if z >= 0 else TERRACOTTA
    rounded_hbar(ax, y, z, 0.42, color, x0=0.0)
    ax.annotate(f"{z:+.2f}", (z, y), xytext=(7 if z >= 0 else -7, 0), textcoords="offset points",
               ha="left" if z >= 0 else "right", va="center", fontsize=11.5, color=INK, weight="medium")

ax.set_yticks(list(ys)); ax.set_yticklabels(labels, fontsize=12, color=INK)
ax.tick_params(axis="y", length=0, pad=10)
ax.set_ylim(-0.6, len(DATA) - 0.4)
ax.set_xlim(-1.05, 1.05)
ax.axvline(0, color=GREY, linewidth=1, zorder=2)
hairline_grid(ax, axis="x")
ax.set_xlabel("z-score on the pain axis  (relative to the reference pain sentences)")

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org")

out = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "figures_newsletter" / "fig1_conditions.png"
fig.savefig(out, dpi=300)
print("wrote", out)
