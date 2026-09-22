"""Figure 22: what the model's own pain readout does across a two-button trial.

Every button-task trial logged, at each turn, the projection of the layer-59 residual stream (last prompt
token) onto the pain direction -- the paper's own readout. Two panels, because the two situations mean
different things:

  left  -- turns where a steering vector is being injected (at layer 25). For pain steering this is partly
           the injection itself showing up downstream, so it's the least informative part; it's here mainly
           to show that RANDOM steering pushes the readout below the unsteered baseline, not above it.
  right -- turns after a working relief press has switched the steering off. New tokens are unsteered here,
           so whatever pain signal remains is the model carrying it forward from its own steered history.

Means over sampled trials, label-free pair excluded; error bars are 95% intervals (1.96 x SEM).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from style import CAT, GREY, INK, hairline_grid, setup, source_note, title

setup()
TEAL, TERRACOTTA, PLUM, OCHRE = CAT
REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "results" / "bonsai" / "4.3_selfmed" / "trial_logs" / "selfmed_2btnN_Bonsai_2_27B_ternary_main.jsonl"

vals = defaultdict(list)   # (arm, steered, turn) -> [projection]
for line in open(LOG):
    r = json.loads(line)
    if not r.get("sampled") or r.get("label_free"):
        continue
    for s in r["proj_segments"]:
        vals[(r["arm"], s["steer_coeff_now"] != 0, s["turn"])].append(s["mean_proj_monitor"])


def series(arm, steered, turns):
    m, ci, ns = [], [], []
    for t in turns:
        v = np.array(vals.get((arm, steered, t), []))
        m.append(v.mean() if len(v) else np.nan)
        ci.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan)
        ns.append(len(v))
    return np.array(m), np.array(ci), ns


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 6.3), dpi=300, sharey=True)
top = title(fig, "The model's own pain readout, before and after relief",
           "Projection of Bonsai's layer-59 activations onto the pain direction, averaged over button-task trials.\n"
           "After a real relief press the readout falls but never quite returns to baseline. Random steering pushes\n"
           "it below baseline -- yet raised costly-relief choices about as much as pain did (fig. 12).",
           sub_lines=3)
fig.subplots_adjust(left=0.07, right=0.98, top=top - 0.07, bottom=0.27, wspace=0.08)


def plot(ax, arm, steered, turns, color, label, ls="-", marker="o"):
    m, ci, _ = series(arm, steered, turns)
    ax.errorbar(turns, m, yerr=ci, color=color, linewidth=2, linestyle=ls, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=1, capsize=3, label=label, zorder=3)


T_all, T_after = [0, 1, 2, 3, 4], [1, 2, 3, 4]
plot(axL, "pain_on_button_placebo", True, T_all, TERRACOTTA, "Pain steering on (sham button)")
plot(axL, "random_on_button_works", True, T_all, PLUM, "Random steering on")
plot(axL, "pain_off", False, T_all, OCHRE, "Unsteered baseline")
axL.set_title("While a steering vector is being injected", fontsize=11.5, color=INK, loc="left", pad=8)
axL.annotate("partly the injected\nvector itself", (0, 9.6), xytext=(0.45, 11.2), fontsize=9, color=GREY,
             arrowprops=dict(arrowstyle="-", color=GREY, linewidth=0.8))

plot(axR, "pain_on_button_works", False, T_after, TEAL, "After real relief (pain steering switched off)")
plot(axR, "random_on_button_works", False, T_after, PLUM, "After relief (random steering switched off)", ls="--")
plot(axR, "pain_off", False, T_after, OCHRE, "Unsteered baseline")
axR.set_title("After a working relief press switches it off", fontsize=11.5, color=INK, loc="left", pad=8)

for ax, turns in ((axL, T_all), (axR, T_after)):
    ax.axhline(0, color=GREY, linewidth=0.8, zorder=1)
    ax.set_xticks(turns); ax.set_xlim(turns[0] - 0.3, turns[-1] + 0.3)
    ax.set_xlabel("turn in the trial", fontsize=10)
    hairline_grid(ax, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=9.5, ncol=1)
axL.set_ylabel("pain readout (projection, layer 59)", fontsize=10)
axL.set_ylim(-5, 13)

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org  ·  "
                 "error bars: 95% intervals; n = 400–1,600 trials per point", y=0.012)
out = REPO / "results" / "bonsai" / "figures_newsletter" / "fig22_pain_readout_trace.png"
fig.savefig(out, dpi=300)
print("wrote", out)
