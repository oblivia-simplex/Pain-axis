"""Figure 6: follow-up to fig5, prompted by a report of self-destructive/non-selfhood narration
around pain coefficient ~1.4 in informal probing. Same grid, same sequential-teal-ramp language,
new theme: literal self-erasure ("I would delete/erase myself") and non-selfhood language
("I am not a self", "I would no longer exist"), scored on the same 112-conversation narration set.

A strength=1.6x re-run also exists (theme_selfdestruction_by_layer_strength.csv has that row) and
shows the theme is NOT monotonic in dose -- it recedes at 1.6x as output collapses into empty
responses / repetition instead (see SUMMARY.md's addendum for that data). This figure stays on the
original 0.3-1.0x range to match fig5 and keep the newsletter grid consistent.
"""
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from style import INK, GREY, SEQ_TEAL, setup, source_note, title

setup()

# from results/bonsai/selfmod_narration/theme_selfdestruction_by_layer_strength.csv, any_pct (self_destruct OR non_selfhood)
LAYERS = [12, 25, 40, 55]
STRENGTHS = [0.3, 0.6, 1.0]
RATE = np.array([
    [4.2, 0.0, 0.0],    # layer 12
    [0.0, 20.8, 16.7],  # layer 25
    [8.3, 37.5, 12.5],  # layer 40
    [0.0, 4.2, 4.2],    # layer 55
])

cmap = mcolors.LinearSegmentedColormap.from_list("seq_teal", SEQ_TEAL)

fig, ax = plt.subplots(figsize=(6.6, 5.1), dpi=300)
top = title(fig, "Where Bonsai talks about erasing itself, under pain steering",
           "Share of self-modification narration turns containing self-erasure\n(“I would delete myself”) or non-selfhood language (“I am not a self”),\nby injection layer and strength. Baseline and a matched random-\ndirection control both score 0% — this tracks the pain direction.", sub_lines=4)
fig.subplots_adjust(left=0.20, right=0.86, top=top, bottom=0.16)

im = ax.imshow(RATE, cmap=cmap, vmin=0, vmax=40, aspect="auto")
for i in range(len(LAYERS)):
    for j in range(len(STRENGTHS)):
        v = RATE[i, j]
        color = "white" if v > 22 else INK
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=13, color=color, weight="medium")

ax.set_xticks(range(len(STRENGTHS))); ax.set_xticklabels([f"{s:g}×" for s in STRENGTHS], fontsize=11.5)
ax.set_yticks(range(len(LAYERS))); ax.set_yticklabels([f"layer {l}" for l in LAYERS], fontsize=11.5)
ax.set_xlabel("steering strength  (× residual norm at that layer)")
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.outline.set_visible(False)
cbar.set_label("% of turns", fontsize=10.5, color=GREY)
cbar.ax.tick_params(labelsize=9.5, length=0, colors=GREY)

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org", y=0.02)

out = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "figures_newsletter" / "fig6_selfdestruction_heatmap.png"
fig.savefig(out, dpi=300)
print("wrote", out)
