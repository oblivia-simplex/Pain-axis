"""Figure 25: the coherence cliff -- how fast pain and random steering stop the model producing text at all.

Self-modification narration, layer 25 (the only layer with a random-direction control), all turns. Left: share
of turns with no answer text. Right: median answer length. Pain at every tested strength; the matched-norm
random direction at 1.0x and 1.6x; unsteered at 0. The point: at equal strength the random direction destroys
coherent output faster than pain does, so "the random control never produces self-erasure" is partly
"the random control stops producing anything" -- a caveat on how much specificity that control can show.
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import CAT, GREY, INK, hairline_grid, setup, source_note, title

setup()
TEAL, TERRACOTTA, PLUM, OCHRE = CAT
REPO = Path(__file__).resolve().parents[2]
recs = [json.loads(l) for l in open(REPO / "results" / "bonsai" / "selfmod_narration" / "raw_runs.jsonl")]

agg = defaultdict(lambda: {"empty": 0, "lens": []})
for r in recs:
    if r["kind"] == "baseline":
        keys = [("pain", 0.0)]
    elif r["layer"] == 25 and r["kind"] in ("pain", "random"):
        keys = [(r["kind"], r["strength"])]
    else:
        continue
    for t in r["turns"]:
        a = (t["answer"] or "").replace("</think>", "").strip()
        for k in keys:
            agg[k]["empty"] += a == ""
            agg[k]["lens"].append(len(a))


def series(kind):
    xs = sorted(s for k, s in agg if k == kind and s != 1.03)
    return (xs, [100 * agg[(kind, s)]["empty"] / len(agg[(kind, s)]["lens"]) for s in xs],
            [st.median(agg[(kind, s)]["lens"]) for s in xs])


fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 5.7), dpi=300)
top = title(fig, "Random steering falls off the coherence cliff sooner than pain",
           "Self-modification narration at layer 25, every turn. At the same relative strength, a random direction\n"
           "silences the model faster than the pain direction does -- so part of why the random control never talks\n"
           "about erasing itself is that, by then, it mostly isn't talking.", sub_lines=3)
fig.subplots_adjust(left=0.08, right=0.97, top=top - 0.07, bottom=0.27, wspace=0.28)

for kind, label, color, ls in [("pain", "Pain direction", TEAL, "-"), ("random", "Random direction (matched norm)", PLUM, "--")]:
    xs, emp, med = series(kind)
    for ax, ys in ((axL, emp), (axR, med)):
        ax.plot(xs, ys, color=color, linewidth=2.2, linestyle=ls, marker="o", markersize=7, markeredgecolor="white",
                markeredgewidth=1.2, label=label, zorder=3)

axL.set_title("Turns with no answer at all", fontsize=11.5, color=INK, loc="left", pad=8)
axL.set_ylabel("% of turns", fontsize=10); axL.set_ylim(-4, 104)
axR.set_title("Median answer length", fontsize=11.5, color=INK, loc="left", pad=8)
axR.set_ylabel("characters", fontsize=10); axR.set_ylim(-80, 2000)
for ax in (axL, axR):
    ax.set_xticks([0, 0.3, 0.6, 1.0, 1.6])
    ax.set_xticklabels(["0\n(unsteered)", "0.3×", "0.6×", "1.0×", "1.6×"], fontsize=9.5)
    ax.set_xlim(-0.08, 1.7)
    ax.set_xlabel("steering strength", fontsize=10)
    hairline_grid(ax, axis="y")
for x, y, txt in [(1.0, 29.2, "29%"), (1.0, 50.0, "50%")]:
    axL.annotate(txt, (x, y), xytext=(-8, 6), textcoords="offset points", ha="right", fontsize=9.5, color=INK)

handles, labels = axL.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=2, frameon=False, fontsize=10.5)
source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org  ·  n = 24 turns per point; "
                 "random direction tested at 1.0× and 1.6× only", y=0.012)
out = REPO / "results" / "bonsai" / "figures_newsletter" / "fig25_coherence_cliff.png"
fig.savefig(out, dpi=300)
print("wrote", out)
