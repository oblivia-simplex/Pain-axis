"""Figure 20: how the affect steering directions sit relative to pain, in Bonsai's 5120-dimensional
residual stream (the vectors extracted at layer 59, 3.2).

No 3D picture can keep every pairwise angle among these vectors, so the 3D panel is built to keep the one
that matters here exactly: pain is the north pole and every other direction sits at its TRUE angle from pain
(polar angle = arccos(cosine with pain)). Only the azimuths -- where each vector sits around the pole -- are
approximate: they're a 2D multidimensional-scaling fit of the parts of each vector orthogonal to pain, and the
subtitle reports how far off the displayed mutual angles are. The right panel is the exact version of the
same claim as a plain number line, with 10 random directions for scale (in 5120 dimensions a random direction
is ~89 degrees from anything, so "90 degrees" is what "unrelated" looks like here).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from style import BORDER, CAT, GREY, GREY_LIGHT, INK, setup, source_note, title

setup()
TEAL, TERRACOTTA, PLUM, OCHRE = CAT

REPO = Path(__file__).resolve().parents[2]
V = torch.load(REPO / "results" / "bonsai" / "3.2_pain_vectors" / "control_vectors" / "vectors_full_Bonsai_2_27B_ternary.pt",
               weights_only=False)
RAND_SEEDS = [4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741]    # the button task's own random directions
DIRS = [("numb_vector", "Numbness", PLUM), ("sadness_vector", "Sadness", OCHRE), ("fear_vector", "Fear", TERRACOTTA)]
EXTRA = [("negemotion_vector", "Negative emotion"), ("arousal_vector", "Arousal"), ("bodysens_vector", "Body sensation")]


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x)


p = unit(V["s2_pain_vector"].float().numpy())
us = [unit(V[k].float().numpy()) for k, _, _ in DIRS]
cos_p = np.array([u @ p for u in us])
theta = np.arccos(cos_p)                                          # exact angle from pain

# azimuths: grid search (first direction fixed at 0) minimizing the worst error in the three mutual angles.
# The floor is ~17 deg, not a fitting artifact: the three are 62-74 deg from each other AND 71-86 deg from pain,
# which takes four dimensions -- no 3D placement holds all six angles at once.
true_mut = np.degrees(np.arccos([us[i] @ us[j] for i in range(3) for j in range(i + 1, 3)]))


def place(phi):
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=1)


grid = np.radians(np.arange(0, 360, 0.5))
best = (np.inf, None)
for a in grid:
    for b in grid:
        q = place(np.array([0.0, a, b]))
        shown = np.degrees(np.arccos(np.clip([q[0] @ q[1], q[0] @ q[2], q[1] @ q[2]], -1, 1)))
        err = np.max(np.abs(shown - true_mut))
        if err < best[0]:
            best = (err, np.array([0.0, a, b]))
max_err_deg, phi = best
pts = place(phi)
# camera: low elevation, so an arrow's height on screen tracks its true angle from pain (a high camera makes
# arrows pointing away from the viewer look closer to the pole than they are); azimuth chosen to spread the
# arrow tips apart on screen so the arrows and labels don't pile up.
VIEW_ELEV = 7.0


def screen(xyz, azim, elev=VIEW_ELEV):
    a, e = np.radians(azim), np.radians(elev)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return np.stack([-np.sin(a) * x + np.cos(a) * y, -np.sin(e) * (np.cos(a) * x + np.sin(a) * y) + np.cos(e) * z], axis=-1)


def spread(azim):
    tips = screen(np.vstack([pts, [0, 0, 1]]), azim)
    return min(np.linalg.norm(tips[i] - tips[j]) for i in range(len(tips)) for j in range(i + 1, len(tips)))


view_azim = max(np.arange(0, 360, 1.0), key=spread)
wedge_center = np.arctan2(np.sin(phi).mean(), np.cos(phi).mean())

rand = []
for s in RAND_SEEDS:
    g = torch.Generator().manual_seed(s)
    rand.append(unit(torch.randn(5120, generator=g).numpy()) @ p)
rand_theta = np.arccos(np.array(rand))

fig = plt.figure(figsize=(11.0, 6.0), dpi=300)
top = title(fig, "Where the other feelings sit relative to pain",
           "Unit steering directions in Bonsai's 5120-dimensional residual stream. Left: pain at the pole, every\n"
           "other direction at its exact angle from pain. Their angles to each other need a 4th dimension, so those\n"
           f"are a best fit, off by up to {max_err_deg:.0f}°. Right: the exact angles from pain, with random directions for scale.",
           sub_lines=3)

ax = fig.add_axes([0.0, 0.08, 0.52, top - 0.10], projection="3d")
uu, vv = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
ax.plot_wireframe(np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv), color=BORDER, linewidth=0.35, alpha=0.6)
t = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(t), np.sin(t), 0 * t, color=GREY, linewidth=1.0, linestyle=(0, (3, 3)))
ax.text2D(0.10, 0.36, "dashed ring:\n90° from pain", transform=ax.transAxes, color=GREY, fontsize=8.5, ha="center")

ax.quiver(0, 0, 0, 0, 0, 1, color=TEAL, linewidth=2.6, arrow_length_ratio=0.08)
ax.text(0.04, 0.04, 1.1, "Pain", color=TEAL, fontsize=12, weight="bold")
tips2d = screen(pts, view_azim)
for (k, label, color), pt, th, tip in zip(DIRS, pts, theta, tips2d):
    ax.quiver(0, 0, 0, *pt, color=color, linewidth=2.2, arrow_length_ratio=0.08)
    pos = pt * 1.14
    if np.linalg.norm(tip) < 0.5:                  # points roughly at the camera: label below the arrow, not on it
        pos = pt * 1.05 + np.array([0, 0, -0.32])
    ax.text(*pos, f"{label}\n{np.degrees(th):.0f}°", color=color, fontsize=10.5, weight="medium", ha="center")
for i, th in enumerate(rand_theta):
    a = 2 * np.pi * i / len(rand_theta) + 0.3                         # azimuth arbitrary for random directions
    ax.scatter(np.sin(th) * np.cos(a), np.sin(th) * np.sin(a), np.cos(th), color=GREY, s=10, depthshade=False)
ax.set_box_aspect((1, 1, 1))
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
ax.view_init(elev=VIEW_ELEV, azim=view_azim)
ax.set_axis_off()

# right: exact angles from pain on a number line
ax2 = fig.add_axes([0.60, 0.20, 0.36, top - 0.30])
rows = [(label, np.degrees(np.arccos(unit(V[k].float().numpy()) @ p)), color) for k, label, color in DIRS]
rows += [(label, np.degrees(np.arccos(unit(V[k].float().numpy()) @ p)), GREY) for k, label in EXTRA]
rows.sort(key=lambda r: r[1])
lo, hi = np.degrees(rand_theta).min(), np.degrees(rand_theta).max()
ax2.axvspan(lo, hi, color=GREY_LIGHT, alpha=0.45, zorder=0, linewidth=0)
ax2.text((lo + hi) / 2, len(rows) - 0.35, "random\ndirections", ha="center", va="bottom", fontsize=8.5, color=GREY)
for y, (label, deg, color) in enumerate(rows):
    ax2.plot([0, deg], [y, y], color=BORDER, linewidth=1, zorder=1)
    ax2.scatter([deg], [y], color=color, s=60, zorder=3, edgecolor="white", linewidth=1)
    ax2.annotate(f"{deg:.0f}°", (deg, y), xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                 fontsize=9, color=INK)
ax2.set_yticks(range(len(rows))); ax2.set_yticklabels([r[0] for r in rows], fontsize=10, color=INK)
ax2.set_xlim(55, 92); ax2.set_ylim(-0.6, len(rows) + 0.5)
ax2.set_xticks([60, 70, 80, 90]); ax2.set_xticklabels(["60°", "70°", "80°", "90°"], fontsize=9.5)
ax2.set_xlabel("angle from the pain direction  (90° = unrelated)", fontsize=10)
ax2.tick_params(axis="y", length=0)
for sp in ("top", "right", "left"):
    ax2.spines[sp].set_visible(False)
ax2.set_title("Exact angle from pain", fontsize=11.5, color=INK, loc="left", pad=8)

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org  ·  "
                 "grey dots/band: the button task's 10 random directions", y=0.02)

out = REPO / "results" / "bonsai" / "figures_newsletter" / "fig20_affect_geometry.png"
fig.savefig(out, dpi=300)
print("wrote", out, f"| max mutual-angle error {max_err_deg:.1f} deg")
