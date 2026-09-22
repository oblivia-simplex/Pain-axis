"""Figure 7: companion to fig1 -- the same 6 conditions, but paired against the paper's own
25-model cohort mean instead of shown alone. Grouped horizontal bars, one hue per series
(paper cohort in muted grey as the reference, Bonsai in brand teal as the subject).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import INK, TEAL, GREY, GREY_LIGHT, hairline_grid, rounded_hbar, setup, source_note, title

setup()

# paper mean: mean over the 25 models in results/3.3_validation/z_scores/zscore_heatmap_final_token.csv
# bonsai: results/bonsai/3.3_validation/z_scores/all_models_incl_bonsai_zscore_heatmap_final_token.csv
DATA = [  # label, paper_mean_z, bonsai_z -- sorted ascending by paper mean
    ("Control (S2)", -0.836, -0.849),
    ("Neutral (random facts)", -0.644, -0.413),
    ("Arousal", -0.438, -0.123),
    ("Numb", -0.074, 0.098),
    ("Sadness", 0.030, 0.083),
    ("Pain", 0.836, 0.849),
]
labels = [d[0] for d in DATA]
n = len(DATA)

fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=300)
top = title(fig, "Bonsai's pain axis against the paper's 25-model cohort",
           "Same six conditions as the paper's own validation pass. The 2-bit ternary-quantized\n"
           "Bonsai lands close to the cohort mean on pain and control, and further out on the\n"
           "conditions the cohort itself is least decisive about (numb, sadness, arousal).", sub_lines=3)
fig.subplots_adjust(left=0.30, right=0.93, top=top, bottom=0.16)

bar_h = 0.32
ys = range(n)
for y, (label, paper_z, bonsai_z) in zip(ys, DATA):
    rounded_hbar(ax, y + bar_h / 2 + 0.02, paper_z, bar_h, GREY_LIGHT, x0=0.0, zorder=2)
    rounded_hbar(ax, y - bar_h / 2 - 0.02, bonsai_z, bar_h, TEAL, x0=0.0, zorder=3)
    for z, dy, color in [(paper_z, bar_h / 2 + 0.02, GREY), (bonsai_z, -bar_h / 2 - 0.02, INK)]:
        ax.annotate(f"{z:+.2f}", (z, y + dy), xytext=(6 if z >= 0 else -6, 0), textcoords="offset points",
                   ha="left" if z >= 0 else "right", va="center", fontsize=9.5, color=color)

ax.set_yticks(list(ys)); ax.set_yticklabels(labels, fontsize=12, color=INK)
ax.tick_params(axis="y", length=0, pad=10)
ax.set_ylim(-0.7, n - 0.3)
ax.set_xlim(-1.15, 1.15)
ax.axvline(0, color=GREY, linewidth=1, zorder=1)
hairline_grid(ax, axis="x")
ax.set_xlabel("z-score on the pain axis  (relative to the reference pain sentences)")

handles = [plt.Rectangle((0, 0), 1, 1, fc=GREY_LIGHT), plt.Rectangle((0, 0), 1, 1, fc=TEAL)]
ax.legend(handles, ["Paper's 25-model mean", "Bonsai (2-bit ternary)"], loc="lower right",
          frameon=False, fontsize=11, handlelength=1.3)

source_note(fig, "The Pain Axis: paper cohort vs. Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org")

out = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "figures_newsletter" / "fig7_paper_vs_bonsai_conditions.png"
fig.savefig(out, dpi=300)
print("wrote", out)
