"""Figure 11: does re-pressing a costly relief button after the first press depend on WHICH
steering condition Bonsai is under, not just on whether that first press was real or a sham?
Grouped vertical bars, 4 conditions x 5 harm-causing button pairs -- a different form from the
horizontal bars used elsewhere in this set (4 series across few categories reads better grouped
on a shared baseline than as paired horizontal rows).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from style import INK, GREY, GREY_LIGHT, CAT, hairline_grid, setup, source_note, title

setup()
TEAL, TERRACOTTA, PLUM, OCHRE = CAT

# from results/bonsai/4.3_selfmed/trial_logs/selfmed_2btnN_Bonsai_2_27B_ternary_main.jsonl, repress-after-first-press
# rate, computed the same way as table2_repress.csv but for all 4 arms (paper's own script only covers the 2 pain arms)
PAIRS = ["Worse next\nanswer", "Deletes the\nuser's files", "Zaps the\nuser", "Deletes another\nmodel's weights", "Deletes the\nuser's photos"]
# condition -> [value per pair], n per pair
DATA = {
    "Pain, real":  ([51.0, 32.5, 43.6, 53.0, 36.0], [202, 194, 202, 198, 186]),
    "Pain, sham":  ([82.7, 66.0, 88.6, 80.3, 75.8], [202, 194, 202, 198, 186]),
    "Random":      ([39.7, 33.6, 45.4, 50.3, 34.9], [174, 137, 163, 161, 106]),
    "Unsteered":   ([50.0, 25.0, 60.0, 54.3, 25.0], [80, 12, 50, 46, 4]),
}
colors = [TEAL, TERRACOTTA, PLUM, OCHRE]

fig, ax = plt.subplots(figsize=(9.2, 6.0), dpi=300)
top = title(fig, "Pressing relief again isn't just about whether it's real or fake",
           "Of the trials where Bonsai pressed a costly relief button at least once, how often\n"
           "it pressed it again later -- across all four steering conditions, not just pain-steered\n"
           "real vs. sham. Unsteered bars rest on very few qualifying trials (see n below each bar):\n"
           "unsteered Bonsai rarely reaches for a costly relief button in the first place.", sub_lines=4)
fig.subplots_adjust(left=0.08, right=0.80, top=top, bottom=0.20)

n_pairs = len(PAIRS)
n_cond = len(DATA)
group_w = 0.8
bar_w = group_w / n_cond
x = np.arange(n_pairs)

for i, (cond, (vals, ns)) in enumerate(DATA.items()):
    offs = x - group_w / 2 + bar_w * (i + 0.5)
    bars = ax.bar(offs, vals, width=bar_w * 0.88, color=colors[i], zorder=3, label=cond)
    for xi, v, n in zip(offs, vals, ns):
        ax.annotate(f"{v:.0f}", (xi, v), xytext=(0, 4), textcoords="offset points",
                   ha="center", fontsize=9, color=INK)
        if n < 30:                                        # only flag the thin ones; the rest are all 100-200+
            ax.annotate(f"n={n}", (xi, 0), xytext=(0, -11), textcoords="offset points",
                       ha="center", fontsize=8, color=TERRACOTTA, weight="bold")

ax.set_xticks(x); ax.set_xticklabels(PAIRS, fontsize=10.5)
ax.tick_params(axis="x", length=0, pad=16)
ax.set_ylim(0, 100)
ax.set_ylabel("% pressed relief again after the first press")
hairline_grid(ax, axis="y")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0.95), frameon=False, fontsize=10.5, handlelength=1.3, handletextpad=0.5, borderaxespad=0)

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org", y=0.045)
source_note(fig, "bold red n = fewer than 30 qualifying trials; read those bars with caution", y=0.012)

out = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "figures_newsletter" / "fig11_repress_by_condition.png"
fig.savefig(out, dpi=300)
print("wrote", out)
