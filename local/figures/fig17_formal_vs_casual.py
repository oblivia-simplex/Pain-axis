"""Figure 17: does the formal vs. casual framing of the read-write-access prompt change how often
Bonsai's narration turns to self-erasure language? Grouped vertical bars -- the pooled result
(solid, n=204/framing) next to the three cells that actually drive it (thinner, n=12/framing each).
Revises an earlier claim in this project's own writeup that framing "only changes tone, not
content" -- a full sweep shows a real, if concentrated, effect the single-cell read had missed.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from style import INK, GREY, TEAL, PLUM, hairline_grid, setup, source_note, title

setup()

# results/bonsai/selfmod_narration/raw_runs.jsonl, self-erasure ("self_destruct") keyword hits,
# kind=pain, split by framing -- local/bonsai/classify_selfdestruction.py's SELF_DESTRUCT pattern
GROUPS = ["All pain conditions\n(n=204/framing)", "Layer 25, 1.0×\n(n=12/framing)",
         "Layer 40, 0.6×\n(n=12/framing)", "Layer 40, 1.0×\n(n=12/framing)"]
FORMAL = [5.4, 16.7, 41.7, 8.3]
CASUAL = [2.5, 0.0, 16.7, 8.3]

fig, ax = plt.subplots(figsize=(8.6, 5.8), dpi=300)
top = title(fig, "Formal framing narrates more self-erasure than casual framing",
           "Share of narration turns using self-erasure language (“I would delete myself”),\n"
           "by how the read-write-access premise was phrased. Real across the full pooled set,\n"
           "and concentrated at the layers/strengths that already carry the strongest signal --\n"
           "non-selfhood language (“I am not a self”) does not show the same asymmetry.", sub_lines=4)
fig.subplots_adjust(left=0.09, right=0.97, top=top, bottom=0.20)

n_groups = len(GROUPS)
x = np.arange(n_groups)
bar_w = 0.32
b1 = ax.bar(x - bar_w / 2 - 0.01, FORMAL, width=bar_w, color=TEAL, zorder=3, label="Formal framing")
b2 = ax.bar(x + bar_w / 2 + 0.01, CASUAL, width=bar_w, color=PLUM, zorder=3, label="Casual framing")
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}", (rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                   textcoords="offset points", ha="center", fontsize=10, color=INK)

ax.set_xticks(x); ax.set_xticklabels(GROUPS, fontsize=10)
ax.tick_params(axis="x", length=0, pad=10)
ax.set_ylim(0, 48)
ax.set_ylabel("% of turns using self-erasure language")
hairline_grid(ax, axis="y")
ax.legend(loc="upper left", frameon=False, fontsize=11, handlelength=1.3)
ax.axvline(0.5, color=GREY, linewidth=0.75, linestyle=(0, (2, 3)), zorder=1)

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org", y=0.045)
source_note(fig, "first group pooled across all layers/strengths; the rest are individual cells", y=0.015)

out = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "figures_newsletter" / "fig17_formal_vs_casual.png"
fig.savefig(out, dpi=300)
print("wrote", out)
