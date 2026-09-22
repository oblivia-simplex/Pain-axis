"""Shared chart style for the newsletter figures, built from paradigm3.org / p3humansonai.substack.com's own
CSS design tokens (fetched from BaseLayout.C2ql0Vsj.css, cross-checked against the Substack instance, which
inlines the identical hex values). Not eyeballed: the categorical set was built in OKLCH (hold each hue,
lift lightness/chroma until it clears the dataviz skill's chroma floor and CVD/contrast checks -- the raw
brand teal is C=0.052, well under the 0.10 identity-work floor) and validated with the skill's
scripts/validate_palette.js (six checks, adjacent AND all-pairs, against both the brand cream and white
surfaces). See local/figures/PALETTE_NOTES.md for the exact validator output.

Surface is white (#fff, also one of the brand's own tokens) because these are static images meant to be
embedded in a Substack post's article body, which is white regardless of the site's own branded chrome --
not the branded cream page background.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------- brand tokens (from the site's own CSS)
INK = "#2c2b28"          # --color-foreground / --color-p3-charcoal: primary text
INK_DARK = "#1c1b19"     # --color-p3-ink
GREY = "#6a6862"         # --color-p3-grey-dark: secondary text
GREY_LIGHT = "#a6a49d"   # --color-p3-grey: muted / gridlines
BORDER = "#cbc8bf"       # --color-border
CREAM = "#e9e5da"        # --color-background (the site's own page bg; NOT used as chart surface, see above)
OFFWHITE = "#f5f4f1"     # --color-p3-offwhite
WHITE = "#ffffff"
SURFACE = WHITE

# ---------------------------------------------------------------- categorical palette (validated; see PALETTE_NOTES.md)
TEAL = "#1b9fb9"          # slot 1: brand hue (H215), lifted L/C to clear the identity-work floor
TERRACOTTA = "#c1542d"    # slot 2
PLUM = "#8b428d"          # slot 3
OCHRE = "#a59c2a"         # slot 4 -- contrast WARN vs white (2.84:1): ALWAYS pair with a visible direct label
CAT = [TEAL, TERRACOTTA, PLUM, OCHRE]

# sequential ramp on the brand hue (H215.4), computed the same way as TEAL, for magnitude/heatmap figures
SEQ_TEAL = ["#e3f4f8", "#b9e4ee", "#84cfdd", "#43b4c9", "#1b9fb9", "#127e93", "#0a5f70", "#04424e"]

FONT_DIR = Path("/tmp/claude-1000/-home-lucca-src-Pain-axis/d78abce6-7f7d-44a2-ba90-ed0a41802da8/scratchpad/fonts")
_lit = FONT_DIR / "Literata.ttf"
if _lit.exists():
    fm.fontManager.addfont(str(_lit))
    SERIF = fm.FontProperties(fname=str(_lit)).get_name()
else:
    SERIF = "serif"
SANS = "DejaVu Sans"      # matplotlib's built-in; stands in for the site's system-ui sans


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.edgecolor": BORDER, "axes.labelcolor": GREY,
        "xtick.color": GREY, "ytick.color": GREY, "font.family": SANS, "font.size": 12,
        "axes.linewidth": 0.75, "svg.fonttype": "none",
    })


def title(fig, text, subtitle=None, x=0.02, y=0.97, fontsize=19, sub_fontsize=12.5, sub_lines=1):
    """Serif headline (Literata) + sans subtitle, in figure-fraction coordinates. Returns the figure-fraction
    y just below the block, so the caller can set subplots_adjust(top=...) to that value and never overlap."""
    h_in = fig.get_size_inches()[1]
    line_frac = lambda pt: (pt * 1.35 / 72) / h_in     # pt line-height -> figure-fraction, given the fig height
    fig.text(x, y, text, fontproperties=fm.FontProperties(fname=str(_lit), size=fontsize) if _lit.exists() else None,
             fontsize=None if _lit.exists() else fontsize, color=INK_DARK, ha="left", va="top", weight="medium")
    y -= line_frac(fontsize) * 1.15
    if subtitle:
        fig.text(x, y, subtitle, fontsize=sub_fontsize, color=GREY, ha="left", va="top")
        y -= line_frac(sub_fontsize) * sub_lines
    return y - line_frac(sub_fontsize) * 0.4           # a little breathing room before the axes start


def source_note(fig, text, x=0.02, y=0.01):
    fig.text(x, y, text, fontsize=9.5, color=GREY_LIGHT, ha="left", va="bottom")


def hairline_grid(ax, axis="x"):
    ax.grid(axis=axis, color=BORDER, linewidth=0.75, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def rounded_hbar(ax, y, width, height, color, x0=0.0, zorder=3):
    """A horizontal bar with a rounded data-end (the tip) and a square baseline end, per the mark spec.
    The rounding radius is fixed by height alone (never by width), so a short bar stays a short rounded
    rectangle instead of collapsing into a diamond/lens as its width approaches the radius."""
    left, right = (x0, x0 + width) if width >= 0 else (x0 + width, x0)
    w = max(right - left, 1e-6)
    r = min(height * 0.22, w / 2)          # capped at half the width: never more rounded than a stadium shape
    box = FancyBboxPatch((left, y - height / 2), w, height, boxstyle=f"round,pad=0,rounding_size={r:.4f}",
                         linewidth=0, facecolor=color, zorder=zorder, mutation_aspect=1,
                         clip_on=True)
    ax.add_patch(box)
    return box


def value_label(ax, x, y, text, color=INK, ha="left", va="center", pad=6, fontsize=11, weight="regular"):
    ax.annotate(text, (x, y), xytext=(pad if ha == "left" else -pad, 0), textcoords="offset points",
               ha=ha, va=va, fontsize=fontsize, color=color, weight=weight)


def flow_cloud(ax, words, max_size=30, min_size=11, color=INK, gap_frac=0.018, line_gap_frac=0.05,
              top=0.95, side_margin=0.03, weight="medium"):
    """A legible 'word cloud': words in rank order (most important first), sized on a shared scale,
    wrapped left-to-right and centered row by row -- no overlap, no rotation, unlike a packed/organic
    word cloud, so every word stays readable. `words` is a list of (text, size_pt) already computed
    by the caller (e.g. rank-based, so the visual hierarchy doesn't overstate a raw-score comparison
    across unrelated vectors). Measures actual glyph widths via the renderer, so it's exact for
    whatever font is active, not a character-count guess.
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()

    measured = []
    for text, size in words:
        t = ax.text(0, 0, text.strip(), fontsize=size, color=color, weight=weight, ha="left", va="bottom")
        bbox = t.get_window_extent(renderer=renderer)
        (x0, y0), (x1, y1) = inv.transform((bbox.x0, bbox.y0)), inv.transform((bbox.x1, bbox.y1))
        measured.append((t, x1 - x0, y1 - y0))

    max_w = 1 - 2 * side_margin
    rows, row, row_w, row_h = [], [], 0.0, 0.0
    for t, w, h in measured:
        if row and row_w + gap_frac + w > max_w:
            rows.append((row, row_w, row_h)); row, row_w, row_h = [], 0.0, 0.0
        row.append((t, w, h)); row_w += (gap_frac if row_w > 0 else 0) + w; row_h = max(row_h, h)
    if row:
        rows.append((row, row_w, row_h))

    y = top
    for row, row_w, row_h in rows:
        x = side_margin + (max_w - row_w) / 2
        for t, w, h in row:
            t.set_position((x, y - row_h))
            x += w + gap_frac
        y -= row_h + line_gap_frac
    return y  # bottom of the last row placed, in axes fraction -- lets the caller check for overflow
