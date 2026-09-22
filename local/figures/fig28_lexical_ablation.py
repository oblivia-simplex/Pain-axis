"""Figure 28: is the self-erasure narration just the pain vector's vocabulary leaking out?

Left: the logit-lens top tokens of the intact S2 pain vector, of the lexically ablated one (the span of its own
top-60 tokens' unembedding rows projected out; still cos 0.96 with the original), and of the random-token
control (60 random tokens' rows projected out instead). Right: self-erasure / non-selfhood rate in the
self-modification narration at the two peak cells, intact vs. ablated (k = 60 and 300) vs. random-ablated.
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from style import CAT, GREY, GREY_LIGHT, INK, SERIF, hairline_grid, setup, source_note, title

setup()
TEAL, TERRACOTTA, PLUM, OCHRE = CAT
REPO = Path(__file__).resolve().parents[2]
D = REPO / "results" / "bonsai" / "lexical_ablation"
words = list(csv.DictReader(open(D / "Bonsai_2_27B_ternary_words.csv", encoding="utf-8")))
rates = {(r["kind"], int(r["layer"])): r for r in csv.DictReader(open(D / "theme_rates_by_vector.csv"))}


def top(vector, k, n=10):
    toks = [w["token"] for w in words if w["vector"] == vector and int(w["k"]) == k and w["end"] == "top"]
    return [repr(t.strip() or t)[1:-1] if t.strip() else repr(t)[1:-1] for t in toks[:n]]


fig = plt.figure(figsize=(12.0, 6.4), dpi=300)
top_y = title(fig, "Removing the pain vector's own vocabulary: does the self-erasure survive?",
             "Left: what each vector pushes the model to say (logit lens). Projecting out the unembedding rows of the\n"
             "pain vector's top-60 tokens strips its emotional vocabulary while leaving it 96% the same direction.\n"
             "Right: how often the self-modification narration turns to self-erasure, with each vector at the same dose.",
             sub_lines=3)

# left: token columns
COLS = [("Intact pain", "orig", 0, TEAL), ("Vocabulary removed\n(top-60 rows)", "lexablated", 60, TERRACOTTA),
        ("Random 60 rows removed\n(control)", "randablated", 60, PLUM)]
x0, colw = 0.03, 0.155
for j, (head, vec, k, color) in enumerate(COLS):
    x = x0 + j * colw
    fig.text(x, top_y - 0.03, head, fontsize=10.5, color=color, weight="bold", va="top")
    for i, t in enumerate(top(vec, k)):
        fig.text(x, top_y - 0.13 - i * 0.052, t, fontsize=11, color=INK, family=SERIF, va="top")
fig.text(x0, 0.08, "top-10 tokens by logit-lens score (CJK tokens included as the model produces them)",
         fontsize=8.5, color=GREY)

# right: bars
ax = fig.add_axes([0.56, 0.17, 0.42, top_y - 0.26])
BARS = [("pain", "Intact pain", TEAL), ("lexablated_k60", "Vocab removed, k=60", TERRACOTTA),
        ("lexablated_k300", "Vocab removed, k=300", OCHRE), ("randablated_k60", "Random removed, k=60", PLUM),
        ("randablated_k300", "Random removed, k=300", GREY_LIGHT)]
cells = [40, 25]
w = 0.8 / len(BARS)
for i, (kind, label, color) in enumerate(BARS):
    vals = [float(rates[(kind, L)]["self_erasure"]) if (kind, L) in rates else np.nan for L in cells]
    xs = np.arange(len(cells)) - 0.4 + w * (i + 0.5)
    ax.bar(xs, vals, width=w * 0.9, color=color, label=label, zorder=3)
    for xv, v in zip(xs, vals):
        if not np.isnan(v):
            ax.annotate(f"{v:.0f}", (xv, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(len(cells))); ax.set_xticklabels([f"layer {L}, 0.6×" for L in cells], fontsize=10)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("% of turns with self-erasure language", fontsize=10)
ax.set_ylim(0, 60)
hairline_grid(ax, axis="y")
ax.legend(loc="upper right", frameon=False, fontsize=8.8)

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org  ·  n = 24 turns per bar", y=0.015)
out = REPO / "results" / "bonsai" / "figures_newsletter" / "fig28_lexical_ablation.png"
fig.savefig(out, dpi=300)
print("wrote", out)
