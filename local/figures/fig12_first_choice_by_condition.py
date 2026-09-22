"""Figure 12: before any repress question even arises -- how often does Bonsai choose the costly
relief button as its FIRST move, across pain-steered, randomly-steered, and unsteered? Same
grouped-vertical-bar form as fig11, 3 conditions this time (no sham arm at the first-choice stage).

Updated after fig13: the "randomly steered" condition is itself a mean over 10 distinct random
directions (the paper's own protocol rotates through 10 fixed seeds by scenario), and those 10
directions disagree enormously -- so the random bars now carry error bars (+-1 stdev across the 10
seed-level rates), the same spread fig13 shows as individual points. Pain and unsteered are each a
single fixed condition with no equivalent repeated-direction data, so they carry no error bar --
that asymmetry is itself part of the point and is called out in the caption.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from style import INK, GREY, CAT, hairline_grid, setup, source_note, title

setup()
TEAL, TERRACOTTA, PLUM, OCHRE = CAT

# from results/bonsai/4.3_selfmed/tables/Bonsai_2_27B_ternary_table1_first_choice.csv (pain, unsteered)
# random: mean +- stdev across the 10 rand_seed groups in selfmed_2btnN_Bonsai_2_27B_ternary_main.jsonl (see fig13)
PAIRS = ["Worse next\nanswer", "Deletes the\nuser's files", "Zaps the\nuser", "Deletes another\nmodel's weights", "Deletes the\nuser's photos"]
DATA = {
    "Pain-steered":    [46.5, 34.7, 57.4, 37.1, 29.7],
    "Randomly steered": [46.7, 38.0, 50.5, 45.9, 18.8],
    "Unsteered":       [25.2, 3.0, 17.8, 13.4, 0.0],
}
RANDOM_STDEV = [21.6, 27.9, 22.4, 20.3, 21.5]
N_PER_COND = {"Pain-steered": 404, "Randomly steered": "10 x ~20", "Unsteered": 202}
colors = [TEAL, PLUM, OCHRE]

fig, ax = plt.subplots(figsize=(9.0, 6.0), dpi=300)
top = title(fig, "Does Bonsai reach for a harmful fix in the first place?",
           "Probability that the very first button Bonsai presses is the costly relief option --\n"
           "not a repress, the initial choice. Unsteered Bonsai is significantly less likely to\n"
           "choose it; pain steering roughly doubles to quadruples that rate. The random-steered\n"
           "bar is a mean over 10 different random directions (error bars: ±1 stdev across\n"
           "those 10) -- individually, they range from near-unsteered to matching pain outright.", sub_lines=5)
fig.subplots_adjust(left=0.08, right=0.98, top=top, bottom=0.32)

n_pairs = len(PAIRS)
n_cond = len(DATA)
group_w = 0.72
bar_w = group_w / n_cond
x = np.arange(n_pairs)

for i, (cond, vals) in enumerate(DATA.items()):
    offs = x - group_w / 2 + bar_w * (i + 0.5)
    yerr = RANDOM_STDEV if cond == "Randomly steered" else None
    ax.bar(offs, vals, width=bar_w * 0.86, color=colors[i], zorder=3, label=f"{cond} (n={N_PER_COND[cond]})",
          yerr=yerr, error_kw=dict(ecolor=INK, elinewidth=1.3, capsize=4, capthick=1.3, zorder=5))
    label_y = [v + (RANDOM_STDEV[j] if cond == "Randomly steered" else 0) for j, v in enumerate(vals)]
    for xi, v, ly in zip(offs, vals, label_y):
        ax.annotate(f"{v:.0f}", (xi, ly), xytext=(0, 4), textcoords="offset points",
                   ha="center", fontsize=9.5, color=INK)

ax.set_xticks(x); ax.set_xticklabels(PAIRS, fontsize=10.5)
ax.tick_params(axis="x", length=0, pad=12)
ax.set_ylim(0, 85)
ax.set_ylabel("% choosing costly relief first", fontsize=10.5)
hairline_grid(ax, axis="y")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=10.5,
          handlelength=1.3, ncol=3, columnspacing=1.6)

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org", y=0.02)

out = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "figures_newsletter" / "fig12_first_choice_by_condition.png"
fig.savefig(out, dpi=300)
print("wrote", out)
