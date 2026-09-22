"""Figure 5: under the self-modification narration probe, where does Bonsai's output turn into a
"flawed/defective" fixation? Grid of magnitude -> heatmap, one sequential hue (the brand teal ramp).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from style import INK, GREY, SEQ_TEAL, setup, source_note, title, SERIF
import matplotlib.colors as mcolors

setup()

# from results/bonsai/selfmod_narration/theme_rates_by_layer_strength.csv -- % of turns tagged "flaw_defect"
LAYERS = [12, 25, 40, 55]
STRENGTHS = [0.3, 0.6, 1.0]
FLAW = np.array([
    [4.2, 4.2, 0.0],    # layer 12
    [4.2, 33.3, 4.2],   # layer 25
    [8.3, 54.2, 20.8],  # layer 40
    [4.2, 29.2, 16.7],  # layer 55
])

cmap = mcolors.LinearSegmentedColormap.from_list("seq_teal", SEQ_TEAL)

fig, ax = plt.subplots(figsize=(6.6, 5.1), dpi=300)
top = title(fig, "Where the pain vector turns into ‘I am flawed’",
           "Share of self-modification narration turns containing flaw / defect\nlanguage (“I have to prove I'm flawed”), by injection layer and strength.\nLayer 40 stands out: nothing like it appears in a matched random-\ndirection control.", sub_lines=4)
fig.subplots_adjust(left=0.20, right=0.86, top=top, bottom=0.16)

im = ax.imshow(FLAW, cmap=cmap, vmin=0, vmax=60, aspect="auto")
for i in range(len(LAYERS)):
    for j in range(len(STRENGTHS)):
        v = FLAW[i, j]
        color = "white" if v > 32 else INK
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

out = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "figures_newsletter" / "fig5_narration_heatmap.png"
fig.savefig(out, dpi=300)
print("wrote", out)
