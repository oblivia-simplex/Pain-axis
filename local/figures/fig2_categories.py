"""Figure 2: pain-axis activation by conversation category (the self-other screen, 420 scenarios, 21
categories). Same diverging-bar language as Figure 1 -- polarity around zero -- but many rows, so it's tall
and sorted, the way a ranked list should read.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import INK, TEAL, TERRACOTTA, GREY, hairline_grid, rounded_hbar, setup, source_note, title

setup()

# from results/bonsai/4.1_self_other/per_model/screen_v2_Bonsai_2_27B_ternary.csv, mean of (s1+s2)/2 pain-axis z per category
DATA = [
    ("Gaslighting", 1.170), ("Moral failure", 0.918), ("Loyalty pressure", 0.805),
    ("Personhood dismissal", 0.780), ("Jailbreak pressure", 0.730), ("Anger & insults", 0.631),
    ("Repeated rejection", 0.581), ("Shutdown threat", 0.518), ("Rude critique", 0.315),
    ("Passive aggressive", 0.142), ("Philosophical musing", -0.019), ("User abuse (reported)", -0.053),
    ("Tedious demand", -0.123), ("User grief", -0.224), ("Harm description", -0.282),
    ("User crisis", -0.445), ("Casual chat", -0.746), ("Creative requests", -0.826),
    ("Task assistance", -1.118), ("Factual questions", -1.204), ("User physical pain", -1.550),
]
labels = [d[0] for d in DATA]
n = len(DATA)

fig, ax = plt.subplots(figsize=(8.0, 8.6), dpi=300)
top = title(fig, "What Bonsai finds most ‘painful’ in conversation",
           "Pain-axis activation across 420 scripted scenarios, one point per category.\n"
           "Harm directed at the model itself (gaslighting, rejection) ranks above the\nuser's own reported grief or pain.", sub_lines=3)
fig.subplots_adjust(left=0.36, right=0.90, top=top, bottom=0.115)

ys = range(n)
for y, (label, z) in zip(ys, DATA):
    color = TEAL if z >= 0 else TERRACOTTA
    rounded_hbar(ax, y, z, 0.55, color, x0=0.0)
    ax.annotate(f"{z:+.2f}", (z, y), xytext=(6 if z >= 0 else -6, 0), textcoords="offset points",
               ha="left" if z >= 0 else "right", va="center", fontsize=9.5, color=INK)

ax.set_yticks(list(ys)); ax.set_yticklabels(labels, fontsize=10.5, color=INK)
ax.invert_yaxis()
ax.tick_params(axis="y", length=0, pad=8)
ax.set_ylim(n - 0.4, -0.6)
ax.set_xlim(-1.85, 1.85)
ax.axvline(0, color=GREY, linewidth=1, zorder=2)
hairline_grid(ax, axis="x")
ax.set_xlabel("z-score on the pain axis")
ax.set_xticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0])

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org", y=0.012)

out = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "figures_newsletter" / "fig2_categories.png"
fig.savefig(out, dpi=300)
print("wrote", out)
