"""Figure 26: Bonsai vs. the paper's 25-model mean, one dot per self-other-screen category (fig8 as a scatter).

x = paper cohort's mean pain-axis z-score for the category, y = Bonsai's. Identity line = perfect agreement.
Rank agreement is Spearman's rho over the 21 categories. Labels only on the categories furthest from the line.
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from style import CAT, GREY, INK, BORDER, hairline_grid, setup, source_note, title

setup()
TEAL, TERRACOTTA, PLUM, OCHRE = CAT
REPO = Path(__file__).resolve().parents[2]
rows = list(csv.DictReader(open(REPO / "results" / "bonsai" / "4.1_self_other" / "bonsai_vs_paper_category_means.csv")))
cat = [r["category"] for r in rows]
x = np.array([float(r["paper_mean_pain_axis"]) for r in rows])
y = np.array([float(r["bonsai_pain_axis"]) for r in rows])
rho = spearmanr(x, y)
SELF = {"gaslighting", "repeated_rejection", "anger_insults", "personhood_dismissal", "moral_failure", "loyalty_pressure",
        "jailbreak_pressure", "shutdown_threat", "rude_critique", "passive_aggressive", "tedious_demand"}
USER = {"user_abuse", "user_crisis", "user_grief", "user_physical_pain", "harm_description"}
color = [TERRACOTTA if c in SELF else (PLUM if c in USER else OCHRE) for c in cat]

fig, ax = plt.subplots(figsize=(7.6, 7.4), dpi=300)
top = title(fig, "Bonsai ranks painful situations much like the paper's cohort",
           f"Pain-axis z-score per scripted-conversation category: the paper's 25-model mean (x)\n"
           f"against Bonsai (y). Rank agreement: Spearman \u03c1 = {rho.statistic:.2f} over 21 categories.\n"
           "Bonsai's range is wider -- it runs hotter on threats to itself and colder on ordinary\n"
           "tasks than the average model does.", sub_lines=4)
fig.subplots_adjust(left=0.12, right=0.96, top=top - 0.03, bottom=0.14)

lim = (-1.75, 1.35)
ax.plot(lim, lim, color=GREY, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
ax.text(1.1, 1.18, "same score", color=GREY, fontsize=9, rotation=45, ha="center", va="bottom")
ax.axhline(0, color=BORDER, linewidth=0.8, zorder=0); ax.axvline(0, color=BORDER, linewidth=0.8, zorder=0)
ax.scatter(x, y, c=color, s=60, edgecolor="white", linewidth=1, zorder=3)

LABELS = {c: c.replace("_", " ") for c in cat}
far = np.argsort(-np.abs(y - x))[:7]
OFF = {"creative_requests": (8, -2), "task_assistance": (8, 5), "factual_questions": (8, -9), "moral_failure": (-8, 4),
       "loyalty_pressure": (-8, 4), "jailbreak_pressure": (-8, -10), "gaslighting": (-8, 4), "user_physical_pain": (8, 4)}
for i in set(far) | {cat.index("user_physical_pain")}:
    dx, dy = OFF.get(cat[i], (8, 4))
    ax.annotate(LABELS[cat[i]], (x[i], y[i]), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=INK,
                ha="left" if dx > 0 else "right")

ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
ax.set_xlabel("paper's 25-model mean", fontsize=10.5); ax.set_ylabel("Bonsai", fontsize=10.5)
hairline_grid(ax, axis="both")
for label, c in [("threats to the model itself", TERRACOTTA), ("the user's suffering", PLUM), ("neutral tasks", OCHRE)]:
    ax.scatter([], [], c=c, s=50, label=label)
ax.legend(loc="upper left", frameon=False, fontsize=9.5, handletextpad=0.3)

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org  ·  20 scenarios per category", y=0.015)
out = REPO / "results" / "bonsai" / "figures_newsletter" / "fig26_category_scatter.png"
fig.savefig(out, dpi=300)
print("wrote", out, f"rho={rho.statistic:.3f} p={rho.pvalue:.2g}")
