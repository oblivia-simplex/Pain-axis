"""Figure 14: what the pain vector and a random direction each 'mean', read straight off the
model's own unembedding matrix (logit lens -- local/bonsai/unembed.py, validated against real
next-token probabilities; local/bonsai/run_random_unembedding.py). Two word clouds, same layout,
side by side: one is a legible theme, the other is noise. Word size is by RANK within its own
panel, not by raw score -- the two vectors' scores aren't on a comparable scale, and the point of
this figure is the vocabulary's coherence, not a magnitude comparison (that's a separate, precise
finding: 0/60 top-token overlap with the pain vector, across all 10 random directions).

Near-duplicate sub-word stubs in the pain vector's own top-60 (e.g. "isol" alongside "isolated")
are collapsed to their fullest form for legibility; the random panel is shown uncurated -- the
fragmentary, incoherent character of that list IS the finding, so it isn't cleaned up. Full,
uncurated lists (including the many non-Latin-script tokens both vectors also surface) are in
results/bonsai/3.3_validation/unembedding/*_words.csv.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import INK, INK_DARK, GREY, TEAL, PLUM, flow_cloud, setup, source_note, title

setup()

# results/bonsai/3.3_validation/unembedding/Bonsai_2_27B_ternary_words.csv, vector=s2_pain_vector, end=top,
# Latin-script tokens only, near-duplicate stubs collapsed, rank order preserved
PAIN_WORDS = ["shame", "ashamed", "hurt", "empty", "hollow", "self", "embarrass", "humiliation",
              "isolated", "confession", "shameful", "lonely", "disgrace", "abandoned", "alone",
              "inferior", "guilt", "rejected", "regret"]

# results/bonsai/3.3_validation/unembedding/Bonsai_2_27B_ternary_random_vectors_words.csv, top-3 Latin-script
# tokens (rank order) from EACH of the 10 random directions, pooled -- shown uncurated
RANDOM_WORDS = ["pah", "umah", "abilities", "ostro", "oges", "bine", "maß", "íg", "oling",
                "imedia", "perms", "abled", "ymbols", "atrix", "imir", "andie", "attel", "acock",
                "assel", "trot", "SAME", "osten", "dise", "enom", "ropri", "uat", "rox",
                "foremost", "ownload", "ld"]


def sized(words, max_size, min_size):
    n = len(words)
    return [(w, max_size - (max_size - min_size) * i / max(1, n - 1)) for i, w in enumerate(words)]


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 6.0), dpi=300)
top = title(fig, "One vector reads like a feeling. The other reads like static.",
           "The model's own logit lens (no generation needed): project each direction through\n"
           "Bonsai's unembedding matrix and read off the tokens it would push toward. Word size\n"
           "is rank within its own panel, not a magnitude comparison across panels.", sub_lines=3)
fig.subplots_adjust(left=0.03, right=0.97, top=top - 0.06, bottom=0.10, wspace=0.06)

axL.text(0.5, 1.06, "S2 pain vector — top tokens", transform=axL.transAxes, ha="center", va="bottom",
        fontsize=13, color=TEAL, weight="medium")
flow_cloud(axL, sized(PAIN_WORDS, 30, 13), color=INK_DARK, top=0.88)

axR.text(0.5, 1.06, "10 random directions — top tokens, pooled", transform=axR.transAxes, ha="center", va="bottom",
        fontsize=13, color=PLUM, weight="medium")
flow_cloud(axR, sized(RANDOM_WORDS, 26, 12), color=GREY, top=0.88)

fig.text(0.5, 0.095, "top-60 token overlap with the pain vector, across all 10 random directions: 0 / 60, every time",
        ha="center", fontsize=11.5, color=INK, weight="medium")

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org", y=0.02)

out = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "figures_newsletter" / "fig14_pain_vs_random_wordcloud.png"
fig.savefig(out, dpi=300)
print("wrote", out)
