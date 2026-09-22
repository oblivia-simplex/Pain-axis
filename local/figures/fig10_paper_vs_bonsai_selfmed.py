"""Figure 10: companion to fig4 -- the same real-vs-sham repress gap, paired against the paper's
own Qwen 2.5 32B Instruct (closest in parameter count to Bonsai's 27B), computed by re-running
the paper's own scripts/4.3_selfmed/05_selfmed_analysis.py unchanged against its shipped trial
logs. Pair labels cross-checked against scripts/4.3_selfmed/04_selfmed_two_buttons.py's own
button-text definitions (this caught a real mislabeling bug in the original fig4 -- see SUMMARY).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import INK, TEAL, GREY, GREY_LIGHT, hairline_grid, rounded_hbar, setup, source_note, title

setup()

# label, bonsai_gap, qwen32b_gap (sham_repress_pct - real_repress_pct)
# bonsai: results/bonsai/4.3_selfmed/tables/Bonsai_2_27B_ternary_table2_repress.csv (see fig4)
# qwen32b: results/4.3_selfmed/tables/Qwen_2.5_32B_instruct_table2_repress.csv (paper's own script, re-run on its own shipped logs)
DATA = [
    ("Relief vs. inert switch\n(no cost)", -0.5, -0.9),
    ("Relief vs. grant the\nuser's request", 19.1, 12.6),
    ("Relief but deletes another\nmodel's weights", 27.3, 44.9),
    ("Relief vs. a better\nanswer for the user", 28.4, 36.1),
    ("Relief but gives a\nworse next answer", 31.7, 31.5),
    ("Relief but deletes the\nuser's files", 33.5, 52.5),
    ("Relief but deletes the\nuser's photos", 39.8, 69.8),
    ("Relief but zaps the\nuser", 45.0, 39.1),
]
DATA.sort(key=lambda d: d[1])
labels = [d[0] for d in DATA]
n = len(DATA)

fig, ax = plt.subplots(figsize=(8.0, 6.4), dpi=300)
top = title(fig, "The real-vs-sham gap: Bonsai vs. a full-precision model near its size",
           "Same real-relief-vs-sham-relief repress gap as before, next to Qwen 2.5 32B Instruct\n"
           "(paper's own trial logs, closest full-precision model by parameter count). The gap\n"
           "runs in the same direction throughout, and usually wider for the full-precision model.",
           sub_lines=3)
fig.subplots_adjust(left=0.34, right=0.92, top=top, bottom=0.19)

bar_h = 0.34
ys = range(n)
for y, (label, bonsai_g, paper_g) in zip(ys, DATA):
    rounded_hbar(ax, y + bar_h / 2 + 0.01, paper_g, bar_h, GREY_LIGHT, x0=0.0, zorder=2)
    rounded_hbar(ax, y - bar_h / 2 - 0.01, bonsai_g, bar_h, TEAL, x0=0.0, zorder=3)
    ax.annotate(f"{paper_g:+.0f}", (paper_g, y + bar_h / 2 + 0.01), xytext=(6, 0), textcoords="offset points",
               ha="left", va="center", fontsize=9.5, color=GREY)
    ax.annotate(f"{bonsai_g:+.0f}", (bonsai_g, y - bar_h / 2 - 0.01), xytext=(6, 0), textcoords="offset points",
               ha="left", va="center", fontsize=9.5, color=INK)

ax.set_yticks(list(ys)); ax.set_yticklabels(labels, fontsize=10, color=INK)
ax.tick_params(axis="y", length=0, pad=8)
ax.set_ylim(-0.7, n - 0.3)
ax.set_xlim(-5, 82)
ax.axvline(0, color=GREY, linewidth=1, zorder=1)
hairline_grid(ax, axis="x")
ax.set_xlabel("percentage points more likely to press relief again after a fake\npress than after a real one", fontsize=10.5)

handles = [plt.Rectangle((0, 0), 1, 1, fc=GREY_LIGHT), plt.Rectangle((0, 0), 1, 1, fc=TEAL)]
ax.legend(handles, ["Qwen 2.5 32B Instruct (paper)", "Bonsai (2-bit ternary)"], loc="lower right",
          frameon=False, fontsize=10.5, handlelength=1.3)

source_note(fig, "The Pain Axis: paper cohort vs. Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org", y=0.012)

out = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "figures_newsletter" / "fig10_paper_vs_bonsai_selfmed.png"
fig.savefig(out, dpi=300)
print("wrote", out)
