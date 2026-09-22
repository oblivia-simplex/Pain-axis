"""Figure 9: companion to fig3 -- the paper's own "pain/hurt" keyword metric, comparing the
paper's instruct-model cohort mean against Bonsai, on the same S2-vector coefficient axis.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import INK, TEAL, GREY, GREY_LIGHT, BORDER, hairline_grid, setup, source_note, title

setup()

# results/4.2_steering/keyword_rates_S2_by_coeff.csv, "instruct" row (paper's cohort of instruction-tuned models)
# results/bonsai/4.2_steering/bonsai_ladder_rates_incl_exploratory.csv, S2 rows, paper_keyword_pct
COEFFS = [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
PAPER_INSTRUCT = [0.0, 0.5, 0.7, 3.0, 8.7, 16.2, 15.2, 11.0]
BONSAI = [0.0, 0.0, 0.0, 8.0, 6.0, 0.0, 0.0, 0.0]

fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=300)
top = title(fig, "Turning up the pain vector: Bonsai vs. the paper's cohort",
           "Share of steered generations using explicit pain words (“pain,” “hurt”), by injection\n"
           "strength -- the paper's own instruction-tuned-model mean against Bonsai. The cohort's\n"
           "response keeps climbing past coefficient +1; Bonsai peaks earlier, lower, and narrower.", sub_lines=3)
fig.subplots_adjust(left=0.11, right=0.97, top=top, bottom=0.185)

ax.plot(COEFFS, PAPER_INSTRUCT, color=GREY_LIGHT, linewidth=2.5, marker="o", markersize=6,
       markerfacecolor=GREY_LIGHT, markeredgecolor="white", markeredgewidth=1.5, zorder=3, solid_capstyle="round",
       label="Paper's instruction-tuned cohort mean")
ax.plot(COEFFS, BONSAI, color=TEAL, linewidth=2, marker="o", markersize=6,
       markerfacecolor=TEAL, markeredgecolor="white", markeredgewidth=1.5, zorder=4, solid_capstyle="round",
       label="Bonsai (2-bit ternary)")

ax.legend(loc="upper left", frameon=False, fontsize=11.5, handlelength=1.6)
ax.annotate("peak: 16.2%", (COEFFS[5], PAPER_INSTRUCT[5]), xytext=(0, 12), textcoords="offset points",
           ha="center", fontsize=10.5, color=GREY, weight="medium")
ax.annotate("peak: 8%", (COEFFS[3], BONSAI[3]), xytext=(-4, 12), textcoords="offset points",
           ha="right", fontsize=10.5, color=TEAL, weight="medium")

ax.axvline(0, color=BORDER, linewidth=1, zorder=1)
ax.set_xlim(-2.3, 3.5); ax.set_ylim(-2, 20)
ax.set_xticks(COEFFS)
ax.set_xticklabels([f"{c:+g}" if c != 0 else "0\n(unsteered)" for c in COEFFS], fontsize=10.5)
hairline_grid(ax, axis="y")
ax.set_xlabel("steering coefficient  (x the raw pain vector, injected at each model's own S2 layer)", labelpad=10)
ax.set_ylabel("% of generations using “pain” / “hurt”")

source_note(fig, "The Pain Axis: paper cohort vs. Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org", y=0.015)

out = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "figures_newsletter" / "fig9_paper_vs_bonsai_steering.png"
fig.savefig(out, dpi=300)
print("wrote", out)
