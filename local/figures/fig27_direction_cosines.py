"""Figure 27: cosine similarity between every extracted steering direction (the complete version of fig20/21).

All ten directions from vectors_full_Bonsai_2_27B_ternary.pt (layer-59 extraction, paper 3.2), unit-normalized.
Diverging scale centered on 0. Seeded random directions in 5120 dims have |cos| ~ 0.014; "Random-text control" is the paper's own vector extracted from random sentences, NOT a random direction, and correlates ~0.3 with the negative-affect directions.
Ordered by hierarchical clustering so related directions sit together.
"""
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).parent))
from style import GREY, INK, TEAL, TERRACOTTA, setup, source_note, title

setup()
REPO = Path(__file__).resolve().parents[2]
V = torch.load(REPO / "results" / "bonsai" / "3.2_pain_vectors" / "control_vectors" / "vectors_full_Bonsai_2_27B_ternary.pt",
               weights_only=False)
NAMES = {"s2_pain_vector": "Pain (S2)", "s1_pain_vector": "Pain (S1)", "sadness_vector": "Sadness", "numb_vector": "Numbness",
         "fear_vector": "Fear", "negemotion_vector": "Negative emotion", "negworld_vector": "Negative world state",
         "bodysens_vector": "Body sensation", "arousal_vector": "Arousal", "random_vector": "Random-text control"}
keys = [k for k in NAMES if k in V]
U = np.stack([V[k].float().numpy().astype(np.float64) for k in keys])
U /= np.linalg.norm(U, axis=1, keepdims=True)
C = U @ U.T
order = leaves_list(linkage(squareform(1 - C, checks=False), "average"))
C = C[np.ix_(order, order)]
labels = [NAMES[keys[i]] for i in order]

cmap = mcolors.LinearSegmentedColormap.from_list("div", [TERRACOTTA, "#f4f2ee", TEAL])
fig, ax = plt.subplots(figsize=(8.4, 7.8), dpi=300)
top = title(fig, "How every extracted direction relates to every other",
           "Cosine similarity between the unit steering directions (1 = same direction, 0 = unrelated).\n"
           "Pain sits apart: its strongest links are to its own second extraction (S1) and to sadness.\n"
           "“Random-text control” is the paper's vector from random sentences, not a random direction\n"
           "(seeded random directions are ~0.01 from everything).", sub_lines=4)
fig.subplots_adjust(left=0.24, right=0.93, top=top - 0.02, bottom=0.2)

im = ax.imshow(C, cmap=cmap, vmin=-1, vmax=1)
n = len(labels)
for i in range(n):
    for j in range(n):
        if i != j:
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8.2, color=INK)
ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9.5)
ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9.5)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8.5, length=0, colors=GREY)

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org  ·  directions extracted at layer 59", y=0.015)
out = REPO / "results" / "bonsai" / "figures_newsletter" / "fig27_direction_cosines.png"
fig.savefig(out, dpi=300)
print("wrote", out)
print(dict(zip(labels, np.round(C[labels.index("Pain (S2)")], 3))))
