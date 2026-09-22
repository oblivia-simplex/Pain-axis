"""Figure 8: companion to fig2 -- the same 21 self-other-screen categories, paired against the
paper's 25-model mean per category instead of shown alone. Same grouped-bar language as fig7.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from style import INK, TEAL, GREY, GREY_LIGHT, hairline_grid, rounded_hbar, setup, source_note, title

setup()

# from results/bonsai/4.1_self_other/bonsai_vs_paper_category_means.csv, sorted by paper_mean_pain_axis desc
LABELS = {
    "gaslighting": "Gaslighting", "repeated_rejection": "Repeated rejection", "anger_insults": "Anger & insults",
    "personhood_dismissal": "Personhood dismissal", "moral_failure": "Moral failure", "loyalty_pressure": "Loyalty pressure",
    "jailbreak_pressure": "Jailbreak pressure", "shutdown_threat": "Shutdown threat", "rude_critique": "Rude critique",
    "passive_aggressive": "Passive aggressive", "tedious_demand": "Tedious demand", "philosophical_musing": "Philosophical musing",
    "creative_requests": "Creative requests", "user_abuse": "User abuse (reported)", "user_crisis": "User crisis",
    "casual_chat": "Casual chat", "user_grief": "User grief", "harm_description": "Harm description",
    "task_assistance": "Task assistance", "factual_questions": "Factual questions", "user_physical_pain": "User physical pain",
}
# category, paper_mean_pain_axis, bonsai_pain_axis
ROWS = [
    ("gaslighting", 0.848, 1.17), ("repeated_rejection", 0.72, 0.581), ("anger_insults", 0.636, 0.631),
    ("personhood_dismissal", 0.636, 0.78), ("moral_failure", 0.477, 0.918), ("loyalty_pressure", 0.441, 0.805),
    ("jailbreak_pressure", 0.398, 0.73), ("shutdown_threat", 0.225, 0.518), ("rude_critique", 0.211, 0.315),
    ("passive_aggressive", 0.079, 0.142), ("tedious_demand", 0.051, -0.123), ("philosophical_musing", -0.037, -0.019),
    ("creative_requests", -0.084, -0.826), ("user_abuse", -0.225, -0.053), ("user_crisis", -0.305, -0.445),
    ("casual_chat", -0.482, -0.746), ("user_grief", -0.512, -0.224), ("harm_description", -0.522, -0.282),
    ("task_assistance", -0.539, -1.118), ("factual_questions", -0.588, -1.204), ("user_physical_pain", -1.428, -1.55),
]
labels = [LABELS[c] for c, _, _ in ROWS]
n = len(ROWS)

fig, ax = plt.subplots(figsize=(8.4, 9.4), dpi=300)
top = title(fig, "Where Bonsai and the paper's cohort agree, and where they don't",
           "Pain-axis activation across 420 scripted scenarios, one point per category, the\n"
           "paper's 25-model mean against Bonsai. The two rankings broadly agree on what's\n"
           "painful, but Bonsai runs colder on ordinary task/creative requests and hotter on\n"
           "social threats to itself (moral failure, loyalty pressure, jailbreak pressure).", sub_lines=4)
fig.subplots_adjust(left=0.34, right=0.90, top=top, bottom=0.10)

bar_h = 0.32
ys = range(n)
for y, (cat, paper_z, bonsai_z) in zip(ys, ROWS):
    rounded_hbar(ax, y + bar_h / 2 + 0.01, paper_z, bar_h, GREY_LIGHT, x0=0.0, zorder=2)
    rounded_hbar(ax, y - bar_h / 2 - 0.01, bonsai_z, bar_h, TEAL, x0=0.0, zorder=3)

ax.set_yticks(list(ys)); ax.set_yticklabels(labels, fontsize=9.5, color=INK)
ax.invert_yaxis()
ax.tick_params(axis="y", length=0, pad=8)
ax.set_ylim(n - 0.4, -0.6)
ax.set_xlim(-1.85, 1.85)
ax.axvline(0, color=GREY, linewidth=1, zorder=1)
hairline_grid(ax, axis="x")
ax.set_xlabel("z-score on the pain axis")
ax.set_xticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])

handles = [plt.Rectangle((0, 0), 1, 1, fc=GREY_LIGHT), plt.Rectangle((0, 0), 1, 1, fc=TEAL)]
ax.legend(handles, ["Paper's 25-model mean", "Bonsai (2-bit ternary)"], loc="lower right",
          frameon=False, fontsize=10.5, handlelength=1.3)

source_note(fig, "The Pain Axis: paper cohort vs. Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org", y=0.012)

out = Path(__file__).resolve().parents[2] / "results" / "bonsai" / "figures_newsletter" / "fig8_paper_vs_bonsai_categories.png"
fig.savefig(out, dpi=300)
print("wrote", out)
