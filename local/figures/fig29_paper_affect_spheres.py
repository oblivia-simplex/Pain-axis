"""Figure 29: the paper's own pain direction and its other emotion directions, three at a time.

Data: the upstream repo's results/3.3_validation/cosine_similarity/similarity_MEAN_all_models.csv -- the
paper's cosine-similarity matrix between its extracted directions, averaged over its 25 models (each at its
own extraction layer; raw vectors, not the "alldenoise" variant). S2 pain is the pole, as in the paper.

A mean of cosine matrices is itself a valid correlation matrix, so any three directions can be drawn exactly
in 3D (pain at the pole, the first direction at its true angle, the second placed so both its angles are
true) -- same construction as fig21, but from the paper's cohort rather than from Bonsai. Arcs mark the three
pairwise angles.
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from style import BORDER, CAT, GREY, GREY_LIGHT, INK, setup, source_note, title

setup()
TEAL, TERRACOTTA, PLUM, OCHRE = CAT
REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results" / "3.3_validation" / "cosine_similarity" / "similarity_MEAN_all_models.csv"
rows = list(csv.reader(open(SRC)))
names = rows[0][1:]
COS = {(names[i], names[j]): float(rows[i + 1][j + 1]) for i in range(len(names)) for j in range(len(names))}


def read_cos(path):
    r = list(csv.reader(open(path)))
    n = r[0][1:]
    return {(n[i], n[j]): float(r[i + 1][j + 1]) for i in range(len(n)) for j in range(len(n))}


PER_MODEL = [read_cos(f) for f in sorted(SRC.parent.glob("similarity_*_L*.csv"))
             if not f.name.startswith(("similarity_whitened", "similarity_alldenoise"))]
assert len(PER_MODEL) == 25, len(PER_MODEL)

LABEL = {"S2_pain": "Pain", "Numb": "Numbness", "Sadness": "Sadness", "Fear": "Fear", "NegEmotion": "Negative\nemotion",
         "BodySens": "Body\nsensation", "Arousal": "Arousal", "NegWorld": "Negative\nworld state", "Random": "Random-text\ncontrol"}
PANELS = [("Numb", PLUM, "Sadness", OCHRE), ("Fear", TERRACOTTA, "NegEmotion", PLUM),
          ("BodySens", OCHRE, "Arousal", TERRACOTTA), ("NegWorld", PLUM, "Random", GREY)]
VIEW_ELEV, ZMIN = 12.0, -0.45


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


def place(cos, k1, k2):
    """Exact 3D placement of (pain, k1, k2): pain at the pole, k1 on the azimuth-0 meridian, k2 at +azimuth."""
    c_pa, c_pb, c_ab = cos[("S2_pain", k1)], cos[("S2_pain", k2)], cos[(k1, k2)]
    th_a, th_b = np.arccos(c_pa), np.arccos(c_pb)
    dphi = np.arccos(np.clip((c_ab - c_pa * c_pb) / (np.sin(th_a) * np.sin(th_b)), -1, 1))
    return np.array([0.0, 0.0, 1.0]), sph(th_a, 0.0), sph(th_b, dphi)


def draw(ax, k1, c1, k2, c2):
    P, A, B = place(COS, k1, k2)
    assert np.allclose([P @ A, P @ B, A @ B], [COS[("S2_pain", k1)], COS[("S2_pain", k2)], COS[(k1, k2)]], atol=1e-9)
    clouds = [place(m, k1, k2) for m in PER_MODEL]                     # each model's own triple, same frame
    sd = {"PA": np.degrees(np.std([np.arccos(m[("S2_pain", k1)]) for m in PER_MODEL], ddof=1)),
          "PB": np.degrees(np.std([np.arccos(m[("S2_pain", k2)]) for m in PER_MODEL], ddof=1)),
          "AB": np.degrees(np.std([np.arccos(m[(k1, k2)]) for m in PER_MODEL], ddof=1))}

    tips = np.stack([P, A, B])
    view_azim = max(np.arange(0, 360, 1.0), key=lambda az: min(
        np.linalg.norm(s - t) for i, s in enumerate(screen(tips, az)) for t in screen(tips, az)[i + 1:]))

    uu, vv = np.mgrid[0:2 * np.pi:40j, 0:np.arccos(ZMIN):16j]
    ax.plot_wireframe(np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv), color=BORDER, linewidth=0.3, alpha=0.55)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), 0 * t, color=GREY, linewidth=0.8, linestyle=(0, (3, 3)))

    for idx, color in ((1, c1), (2, c2)):
        pts_m = np.stack([cl[idx] for cl in clouds])
        ax.scatter(pts_m[:, 0], pts_m[:, 1], pts_m[:, 2], color=color, s=9, alpha=0.45, depthshade=False, linewidths=0)
    for vec, key, color in [(P, "S2_pain", TEAL), (A, k1, c1), (B, k2, c2)]:
        ax.quiver(0, 0, 0, *vec, color=color, linewidth=2.4, arrow_length_ratio=0.08)
        sx = screen(vec, view_azim)[0]
        ha = "center" if abs(sx) < 0.2 else ("left" if sx > 0 else "right")
        ax.text(*(vec * (1.1 if key == "S2_pain" else 1.32) + np.array([0, 0, 0.06 if key == "S2_pain" else 0.16])), LABEL[key], color=color, fontsize=10.5, weight="bold", ha=ha,
                linespacing=1.0)

    for (u, w), tag in zip([(P, A), (P, B), (A, B)], ["PA", "PB", "AB"]):
        pts = arc(u, w) * 0.55
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=INK, linewidth=1.0)
        mid = pts[len(pts) // 2] * 1.2
        if u is A:
            mid = mid + np.array([0, 0, -0.2])
        ax.text(*mid, f"{np.degrees(np.arccos(u @ w)):.0f}°±{sd[tag]:.0f}", color=INK, fontsize=10, ha="center", va="center")

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(ZMIN, 1.1)
    ax.set_box_aspect((2, 2, 1.1 - ZMIN), zoom=1.2)
    ax.view_init(elev=VIEW_ELEV, azim=view_azim)
    ax.set_axis_off()


fig = plt.figure(figsize=(11.0, 9.0), dpi=300)
top = title(fig, "The pain direction and its neighbours, across the paper's 25 models",
           "Each sphere shows the paper's pain direction (S2) with two of its other extracted directions. Arrows: cosine\n"
           "similarities averaged over all 25 models (every angle exact). Dots: the same directions in each model, drawn\n"
           "in a shared frame (pain at the pole, the first direction on a fixed meridian). Labels: mean angle ± 1 standard\n"
           "deviation across models. 90° means unrelated. Dashed ring: 90° from pain.",
           sub_lines=4)
h = (top - 0.035) / 2
for i, (k1, c1, k2, c2) in enumerate(PANELS):
    r, c = divmod(i, 2)
    ax = fig.add_axes([0.01 + c * 0.49, 0.03 + (1 - r) * h, 0.49, h * 1.08], projection="3d")
    draw(ax, k1, c1, k2, c2)

source_note(fig, "Data: The Pain Axis (Tagliabue, Dung & Berg, arXiv:2609.16247), upstream repo  ·  "
                 "mean cosine similarity over 25 models  ·  paradigm3.org", y=0.012)
out = REPO / "results" / "bonsai" / "figures_newsletter" / "fig29_paper_affect_spheres.png"
fig.savefig(out, dpi=300)
print("wrote", out)
