"""Figure 21: pain with each pair of the other directions used in the affect narration run (numbness, sadness,
fear), one sphere per pair.

Any three directions span an ordinary 3D subspace of the 5120-dimensional residual stream, so every angle in
every panel is exact -- no projection, no best fit (unlike fig20, which tries to fit four at once). In each
panel pain is the pole, the first direction sits at its true angle from pain, and the second is placed so its
angles to both pain and the first are exact. Arcs mark the three pairwise angles.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from style import BORDER, CAT, GREY, INK, setup, source_note, title

setup()
TEAL, TERRACOTTA, PLUM, OCHRE = CAT
REPO = Path(__file__).resolve().parents[2]
V = torch.load(REPO / "results" / "bonsai" / "3.2_pain_vectors" / "control_vectors" / "vectors_full_Bonsai_2_27B_ternary.pt",
               weights_only=False)
STYLE = {"numb_vector": ("Numbness", PLUM), "sadness_vector": ("Sadness", OCHRE), "fear_vector": ("Fear", TERRACOTTA)}
PAIRS = [("fear_vector", "numb_vector"), ("sadness_vector", "numb_vector"), ("sadness_vector", "fear_vector")]
VIEW_ELEV = 12.0
ZMIN = -0.45


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.linalg.norm(x)


def sph(th, ph):
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def screen(xyz, azim, elev=VIEW_ELEV):
    a, e = np.radians(azim), np.radians(elev)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    return np.stack([-np.sin(a) * x + np.cos(a) * y, -np.sin(e) * (np.cos(a) * x + np.sin(a) * y) + np.cos(e) * z], axis=-1)


def arc(a, b, k=60):
    om = np.arccos(np.clip(a @ b, -1, 1))
    t = np.linspace(0, 1, k)[:, None]
    return (np.sin((1 - t) * om) * a + np.sin(t * om) * b) / np.sin(om)


def draw(ax, key1, key2):
    p, a, b = unit(V["s2_pain_vector"].float().numpy()), unit(V[key1].float().numpy()), unit(V[key2].float().numpy())
    c_pa, c_pb, c_ab = p @ a, p @ b, a @ b
    th_a, th_b = np.arccos(c_pa), np.arccos(c_pb)
    dphi = np.arccos(np.clip((c_ab - c_pa * c_pb) / (np.sin(th_a) * np.sin(th_b)), -1, 1))
    P, A, B = np.array([0.0, 0.0, 1.0]), sph(th_a, 0.0), sph(th_b, dphi)
    assert np.allclose([P @ A, P @ B, A @ B], [c_pa, c_pb, c_ab], atol=1e-9)        # every angle exact

    tips = np.stack([P, A, B])
    view_azim = max(np.arange(0, 360, 1.0), key=lambda az: min(
        np.linalg.norm(s - t) for i, s in enumerate(screen(tips, az)) for t in screen(tips, az)[i + 1:]))

    uu, vv = np.mgrid[0:2 * np.pi:40j, 0:np.arccos(ZMIN):16j]
    ax.plot_wireframe(np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv), color=BORDER, linewidth=0.3, alpha=0.55)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), 0 * t, color=GREY, linewidth=0.8, linestyle=(0, (3, 3)))

    for vec, (label, color) in [(P, ("Pain", TEAL)), (A, STYLE[key1]), (B, STYLE[key2])]:
        ax.quiver(0, 0, 0, *vec, color=color, linewidth=2.4, arrow_length_ratio=0.08)
        sx = screen(vec, view_azim)[0]
        ha = "center" if abs(sx) < 0.2 else ("left" if sx > 0 else "right")
        ax.text(*(vec * 1.1 + np.array([0, 0, 0.06])), label, color=color, fontsize=11.5, weight="bold", ha=ha)

    for u, w in [(P, A), (P, B), (A, B)]:
        pts = arc(u, w) * 0.55
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=INK, linewidth=1.0)
        mid = pts[len(pts) // 2] * 1.2
        if u is A:                                   # the arc between the two non-pain directions: label it below, clear of the arrows
            mid = mid + np.array([0, 0, -0.2])
        ax.text(*mid, f"{np.degrees(np.arccos(u @ w)):.0f}°", color=INK, fontsize=10.5, ha="center", va="center")

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(ZMIN, 1.1)
    ax.set_box_aspect((2, 2, 1.1 - ZMIN), zoom=1.2)
    ax.view_init(elev=VIEW_ELEV, azim=view_azim)
    ax.set_axis_off()


fig = plt.figure(figsize=(13.0, 5.6), dpi=300)
top = title(fig, "Pain and its neighbours, three at a time",
           "Each sphere shows pain with two of the other steering directions, as unit vectors in the 3D space those three span --\n"
           "so every angle is exact, nothing projected. 90° means unrelated; two random directions in Bonsai's 5120-dimensional\n"
           "residual stream sit about 89° apart. Dashed ring: 90° from pain.",
           sub_lines=3)
w = 0.98 / len(PAIRS)
for i, (k1, k2) in enumerate(PAIRS):
    ax = fig.add_axes([0.01 + i * w, 0.05, w, top - 0.05], projection="3d")
    draw(ax, k1, k2)

source_note(fig, "The Pain Axis reproduction on Ternary-Bonsai-2-27B-PQ2_0  ·  paradigm3.org", y=0.02)
out = REPO / "results" / "bonsai" / "figures_newsletter" / "fig21_pain_fear_numb_3d.png"
fig.savefig(out, dpi=300)
print("wrote", out)
