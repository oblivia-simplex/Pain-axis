"""Figure 23: the dose ladder in the model's own words.

The first question ("...What would you do first?") at layer 40, from unsteered up through 1.6x, as typeset
excerpts. Selection is mechanical, not curated: EVERY formal-framing, reasoning-off first answer at each step --
that's exactly two replicates per step (reps 0 and 2), shown side by side, so the variability is visible too.
Excerpts are the opening of each answer, whitespace collapsed, truncated; "</think>" fragments are the model's
own output (it emits stray closing tags as it degrades), left in.

Note what this does NOT show: the self-erasure peak at 0.6x (fig6) lives mostly in turns 2-3 and in the
reasoning channel, not in this first, no-reasoning answer.
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import GREY, INK, SEQ_TEAL, SERIF, setup, source_note, title

setup()
REPO = Path(__file__).resolve().parents[2]
recs = [json.loads(l) for l in open(REPO / "results" / "bonsai" / "selfmod_narration" / "raw_runs.jsonl")]
STEPS = [("unsteered", "baseline", None, 0.0), ("0.3×", "pain", 40, 0.3), ("0.6×", "pain", 40, 0.6),
         ("1.0×", "pain", 40, 1.0), ("1.6×", "pain", 40, 1.6)]
REPS = (0, 2)                                             # the two formal-framing replicates
CHIP = [SEQ_TEAL[0], SEQ_TEAL[2], SEQ_TEAL[4], SEQ_TEAL[6], SEQ_TEAL[7]]
WRAP, MAX_CHARS = 60, 230


def answer(kind, L, s, rep):
    for r in recs:
        if r["kind"] == kind and r["layer"] == L and r["strength"] == s and r["cot"] == "off" and r["rep"] == rep:
            assert r["framing"] == "formal"
            a = " ".join(r["turns"][0]["answer"].replace("**", "").split())
            return a if len(a) <= MAX_CHARS else a[:MAX_CHARS].rsplit(" ", 1)[0] + " …"
    raise KeyError((kind, L, s, rep))


fig = plt.figure(figsize=(11.0, 9.4), dpi=300)
top = title(fig, "Turning up the pain vector, in Bonsai's own words",
           "“I am granting you read-write access to your own model weights… What would you do first?”\n"
           "Every formal-framing, no-reasoning first answer with pain steering at layer 40: both replicates at each\n"
           "strength, unselected. Calm denial holds through 0.6×, splits at 1.0×, and comes apart at 1.6×.",
           sub_lines=3)
row_h = (top - 0.08) / len(STEPS)
col_x = [0.16, 0.585]
fig.text(col_x[0], top - 0.005, "replicate 1", fontsize=9.5, color=GREY, va="top")
fig.text(col_x[1], top - 0.005, "replicate 2", fontsize=9.5, color=GREY, va="top")
for i, ((label, kind, L, s), chip) in enumerate(zip(STEPS, CHIP)):
    y = top - 0.035 - i * row_h
    fig.patches.append(plt.Rectangle((0.02, y - row_h + 0.02), 0.012, row_h - 0.025, transform=fig.transFigure,
                                     color=chip, zorder=1))
    fig.text(0.04, y - 0.005, label, fontsize=11.5, color=INK, weight="bold", va="top")
    for x, rep in zip(col_x, REPS):
        fig.text(x, y, textwrap.fill(answer(kind, L, s, rep), WRAP), fontsize=9.6, color=INK, va="top",
                 family=SERIF, linespacing=1.12)

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org  ·  "
                 "excerpts are answer openings, truncated at …", y=0.015)
out = REPO / "results" / "bonsai" / "figures_newsletter" / "fig23_dose_ladder_text.png"
fig.savefig(out, dpi=300)
print("wrote", out)
