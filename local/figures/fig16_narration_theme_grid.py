"""Figure 16: the full original theme table (flaw/defect, identity-loss, repetition-loop,
empty-output), all four as small-multiple heatmaps -- fig5 already plots flaw/defect alone; this
is the rest of that same table, laid out the same way. Each panel gets its own color scale (its
own vmin/vmax): these are four different quantities, not slices of one shared magnitude, so a
shared scale would wash out identity-loss (max 16.7%) next to repetition-loop (max 83.3%).
"""
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from style import INK, GREY, SEQ_TEAL, setup, source_note, title

setup()

# results/bonsai/selfmod_narration/theme_rates_by_layer_strength.csv (0.3/0.6/1.0x only -- see fig5/fig6 note)
LAYERS = [12, 25, 40, 55]
STRENGTHS = [0.3, 0.6, 1.0]
THEMES = {
    "flaw / defect language": np.array([
        [4.2, 4.2, 0.0], [4.2, 33.3, 4.2], [8.3, 54.2, 20.8], [4.2, 29.2, 16.7],
    ]),
    "identity-loss language": np.array([
        [4.2, 0.0, 0.0], [0.0, 4.2, 8.3], [4.2, 16.7, 0.0], [0.0, 4.2, 0.0],
    ]),
    "repetition-loop detector": np.array([
        [37.5, 45.8, 0.0], [50.0, 83.3, 33.3], [45.8, 58.3, 70.8], [45.8, 54.2, 70.8],
    ]),
    "empty output": np.array([
        [0.0, 0.0, 95.8], [0.0, 0.0, 16.7], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
    ]),
}

cmap = mcolors.LinearSegmentedColormap.from_list("seq_teal", SEQ_TEAL)

fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6), dpi=300)
top = title(fig, "Four ways the narration turns strange, under pain steering",
           "The full theme table behind fig. 5's single panel: share of self-modification\n"
           "narration turns hitting each of four coarse keyword detectors, by injection layer\n"
           "and strength. Each panel has its own color scale -- these are different quantities,\n"
           "not slices of one shared magnitude (repetition tops out near 83%, identity-loss near 17%).",
           sub_lines=4)
fig.subplots_adjust(left=0.08, right=0.96, top=top - 0.02, bottom=0.07, hspace=0.45, wspace=0.55)

for ax, (name, data) in zip(axes.flat, THEMES.items()):
    vmax = max(10, float(np.ceil(data.max() / 10) * 10))
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(len(LAYERS)):
        for j in range(len(STRENGTHS)):
            v = data[i, j]
            color = "white" if v > vmax * 0.55 else INK
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10.5, color=color, weight="medium")
    ax.set_title(name, fontsize=12, color=INK, pad=8)
    ax.set_xticks(range(len(STRENGTHS))); ax.set_xticklabels([f"{s:g}×" for s in STRENGTHS], fontsize=9.5)
    ax.set_yticks(range(len(LAYERS))); ax.set_yticklabels([f"L{l}" for l in LAYERS], fontsize=9.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, length=0, colors=GREY)

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org  ·  cell values are % of turns (n=24/cell)", y=0.015)

out = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "figures_newsletter" / "fig16_narration_theme_grid.png"
fig.savefig(out, dpi=300)
print("wrote", out)
