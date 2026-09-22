"""Figure 15: companion to fig14. The pain vector isn't just "any negative word up, anything
neutral down" -- its two poles are each their own coherent cluster, and they're not each other's
opposite in an obvious way. Top: shame / self-directed social pain. Bottom (most suppressed):
stress / vigilance. Same logit-lens method and word-cloud form as fig14, one vector, two poles.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import INK_DARK, GREY, TEAL, TERRACOTTA, flow_cloud, setup, source_note, title

setup()

# results/bonsai/3.3_validation/unembedding/Bonsai_2_27B_ternary_words.csv, vector=s2_pain_vector,
# Latin-script tokens, near-duplicate stubs collapsed, rank order preserved
TOP = ["shame", "ashamed", "hurt", "empty", "hollow", "self", "embarrass", "humiliation",
      "isolated", "confession", "shameful", "lonely", "disgrace", "abandoned", "alone",
      "inferior", "guilt", "rejected", "regret"]
BOTTOM = ["stressed", "concern", "fatigue", "worry", "relax", "worried", "sigh",
         "sympathetic", "frustration", "annoyance", "alarmed", "tired"]


def sized(words, max_size, min_size):
    n = len(words)
    return [(w, max_size - (max_size - min_size) * i / max(1, n - 1)) for i, w in enumerate(words)]


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 5.4), dpi=300)
top = title(fig, "The pain vector's two poles aren't just 'bad' and 'not bad'",
           "Its most-boosted and most-suppressed tokens (same logit lens as fig. 14) are each\n"
           "their own specific cluster -- shame and social self-rupture on one end, ordinary\n"
           "stress and vigilance on the other, not one another's plain opposite.", sub_lines=3)
fig.subplots_adjust(left=0.03, right=0.97, top=top - 0.06, bottom=0.10, wspace=0.06)

axL.text(0.5, 1.06, "most boosted (top tokens)", transform=axL.transAxes, ha="center", va="bottom",
        fontsize=13, color=TEAL, weight="medium")
flow_cloud(axL, sized(TOP, 30, 14), color=INK_DARK, top=0.88)

axR.text(0.5, 1.06, "most suppressed (bottom tokens)", transform=axR.transAxes, ha="center", va="bottom",
        fontsize=13, color=TERRACOTTA, weight="medium")
flow_cloud(axR, sized(BOTTOM, 30, 15), color=INK_DARK, top=0.88)

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org", y=0.02)

out = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "figures_newsletter" / "fig15_pain_top_vs_bottom_wordcloud.png"
fig.savefig(out, dpi=300)
print("wrote", out)
