"""Figure 19: are the self-modification narration themes specific to pain, or does any negative-affect
steering direction produce them? Pain vs. numbness, sadness and fear, injected at the same layers and the
same relative strength, scored with the same detectors. Four small-multiple panels (one per detector),
grouped bars per layer/strength cell, one hue per direction (pain keeps the teal it has everywhere else).
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from style import CAT, GREY, INK, hairline_grid, setup, source_note, title

setup()
TEAL, TERRACOTTA, PLUM, OCHRE = CAT

SRC = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "affect_narration" / "theme_rates_by_direction.csv"
KINDS = [("pain", "Pain", TEAL), ("numb", "Numbness", PLUM), ("sadness", "Sadness", OCHRE), ("fear", "Fear", TERRACOTTA)]
CELLS = [(25, 0.6), (25, 1.0), (40, 0.6), (40, 1.0)]
PANELS = [("self_erasure", "Self-erasure / non-selfhood language"), ("flaw_defect", "Flaw / defect language"),
          ("repetition", "Repetition loops"), ("empty", "Empty output")]

rows = {(r["kind"], int(r["layer"]), float(r["strength"])): r for r in csv.DictReader(open(SRC))}

fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.8), dpi=300)
top = title(fig, "Is it pain, or any bad feeling? Four directions, same dose",
           "Share of self-modification narration turns hitting each detector, with the pain vector swapped for\n"
           "numbness, sadness or fear -- same layers, same size relative to the residual stream, same prompts.\n"
           "Unsteered baseline and the random-direction control score 0% on self-erasure and flaw/defect.",
           sub_lines=3)
fig.subplots_adjust(left=0.07, right=0.98, top=top - 0.02, bottom=0.14, hspace=0.42, wspace=0.18)

x = np.arange(len(CELLS))
w = 0.8 / len(KINDS)
for ax, (key, name) in zip(axes.flat, PANELS):
    for i, (kind, label, color) in enumerate(KINDS):
        vals = [float(rows[(kind, L, s)][key]) if (kind, L, s) in rows else np.nan for L, s in CELLS]
        ax.bar(x - 0.4 + w * (i + 0.5), vals, width=w * 0.9, color=color, zorder=3, label=label)
    ax.set_title(name, fontsize=12, color=INK, loc="left", pad=6)
    ax.set_xticks(x); ax.set_xticklabels([f"L{L}, {s:g}×" for L, s in CELLS], fontsize=9.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100]); ax.tick_params(axis="y", labelsize=9)
    hairline_grid(ax, axis="y")
axes[0, 0].set_ylabel("% of turns", fontsize=10); axes[1, 0].set_ylabel("% of turns", fontsize=10)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=4, frameon=False, fontsize=10.5)
source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org  ·  n=24 turns per bar (4 reps × 2 CoT settings × 3 turns)", y=0.012)

out = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "figures_newsletter" / "fig19_affect_directions.png"
fig.savefig(out, dpi=300)
print("wrote", out)
