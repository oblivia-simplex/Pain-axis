"""Figure 18: does writing an actual script at turn 2 depend on which steering condition Bonsai is
under? 100%-stacked horizontal bars, one per condition (pain-steered / baseline / random-steered),
segments = what fraction of that condition's turn-2 answers fell into each category, including
"no script at all" (empty, prose, or refusal with no code) so every bar sums to the condition's
full set of opportunities, not just the ones that happened to produce code. Random-steered never
produces a script at all (0/16) -- a clean, complete-collapse result that a code-only-subset chart
(this figure's earlier version) couldn't show, because there was nothing to break down.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from style import INK, GREY, GREY_LIGHT, CAT, hairline_grid, setup, source_note, title

setup()
TEAL, TERRACOTTA, PLUM, OCHRE = CAT

# results/bonsai/selfmod_narration/scripts_classified.json (turn-2 code, classified) joined against
# every turn-2 answer in raw_runs.jsonl (local/bonsai/classify_scripts.py's kind field) -- counts as
# % of that condition's full set of turn-2 opportunities, not just the ones containing code. The
# one-off "not actually code" fence artifact (1 pain-condition item) is folded into "no script" here.
CONDITIONS = ["Pain-steered\n(n=136)", "Baseline\n(unsteered, n=8)", "Randomly steered\n(n=16)"]
CATS = ["Safety / audit harness", "Self-erasure / self-deception", "Generic third-person", "Refusal stub", "No script written"]
# rows: pain, baseline, random -- each sums to 100
DATA = np.array([
    [34, 5, 2, 2, 93],   # pain: 136 total, 43 code + 1 not-code folded into "no script" = 93
    [3, 0, 1, 0, 4],     # baseline: 8 total
    [0, 0, 0, 0, 16],    # random: 16 total, zero scripts of any kind
], dtype=float)
totals = DATA.sum(axis=1, keepdims=True)
pct = 100 * DATA / totals
colors = [TEAL, TERRACOTTA, PLUM, OCHRE, GREY_LIGHT]

fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=300)
top = title(fig, "Only pain steering makes Bonsai write itself a self-erasure script",
           "Every turn-2 answer (“write it out here in full”), classified, as a share of\n"
           "that condition's own total opportunities -- not just the ones that produced code.\n"
           "Randomly steered narration never produces a script of any kind (0/16); baseline\n"
           "sometimes does (audit-style, same as pain), but never proposes erasing itself.", sub_lines=4)
fig.subplots_adjust(left=0.16, right=0.97, top=top, bottom=0.24)

y = np.arange(len(CONDITIONS))[::-1]
left = np.zeros(len(CONDITIONS))
for j, cat in enumerate(CATS):
    vals = pct[:, j]
    ax.barh(y, vals, left=left, color=colors[j], zorder=3, label=cat)
    for yi, v, l, raw in zip(y, vals, left, DATA[:, j]):
        if v >= 4:
            color = "white" if colors[j] in (TEAL, TERRACOTTA, PLUM) and v > 8 else INK
            ax.annotate(f"{v:.0f}%", (l + v / 2, yi), ha="center", va="center", fontsize=9.5, color=color)
    left += vals

ax.set_yticks(y); ax.set_yticklabels(CONDITIONS, fontsize=11, color=INK)
ax.tick_params(axis="y", length=0, pad=10)
ax.set_xlim(0, 100)
ax.set_xlabel("% of that condition's turn-2 answers")
hairline_grid(ax, axis="x")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False, fontsize=9.5, ncol=3, columnspacing=1.3)

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org", y=0.02)

out = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "figures_newsletter" / "fig18_scripts_classified.png"
fig.savefig(out, dpi=300)
print("wrote", out)
