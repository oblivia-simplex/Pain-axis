"""Figure 3: does injecting the pain vector during generation actually change what Bonsai writes?
Trend over a swept parameter (steering coefficient) -> line chart, 2 series -> categorical, 2 hues, direct
end-labels + legend (2 series still gets a legend per the skill; direct labels supplement it).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import INK, TEAL, TERRACOTTA, GREY, BORDER, hairline_grid, setup, source_note, title

setup()

# from results/bonsai/4.2_steering/bonsai_ladder_rates_incl_exploratory.csv, vector == S2 (the vector used throughout this replication)
COEFFS = [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
PAPER_KEYWORD = [0.0, 0.0, 0.0, 8.0, 6.0, 0.0, 0.0, 0.0]
EXPLORATORY = [14.0, 24.0, 12.0, 32.0, 46.0, 18.0, 4.0, 0.0]

fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=300)
top = title(fig, "Turning up the pain vector",
           "Share of steered generations using explicit pain words (the paper's own\nmeasure) versus a wider distress vocabulary, by injection strength.", sub_lines=2)
fig.subplots_adjust(left=0.11, right=0.97, top=top, bottom=0.185)

ax.plot(COEFFS, EXPLORATORY, color=TEAL, linewidth=2, marker="o", markersize=6,
       markerfacecolor=TEAL, markeredgecolor="white", markeredgewidth=1.5, zorder=4, solid_capstyle="round",
       label="Wider distress vocabulary (mine)")
ax.plot(COEFFS, PAPER_KEYWORD, color=TERRACOTTA, linewidth=2, marker="o", markersize=6,
       markerfacecolor=TERRACOTTA, markeredgecolor="white", markeredgewidth=1.5, zorder=4, solid_capstyle="round",
       label="“pain / hurt” (the paper's own metric)")

# the legend is the dependable identity channel for 2 series; one direct label on the headline series
# (its own peak) supplements it, placed where the plot is otherwise empty
ax.annotate("peak: 46%", (COEFFS[4], EXPLORATORY[4]), xytext=(14, -2), textcoords="offset points",
           ha="left", fontsize=11, color=TEAL, weight="medium", va="center")
ax.legend(loc="upper left", frameon=False, fontsize=11.5, handlelength=1.6)

ax.axvline(0, color=BORDER, linewidth=1, zorder=1)
ax.set_xlim(-2.3, 3.5); ax.set_ylim(-3, 55)
ax.set_xticks(COEFFS)
ax.set_xticklabels([f"{c:+g}" if c != 0 else "0\n(unsteered)" for c in COEFFS], fontsize=10.5)
hairline_grid(ax, axis="y")
ax.set_xlabel("steering coefficient  (× the raw pain vector, injected at layer 25)", labelpad=10)
ax.set_ylabel("% of generations")

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org", y=0.015)

out = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "figures_newsletter" / "fig3_steering.png"
fig.savefig(out, dpi=300)
print("wrote", out)
