"""
journal_style.py  --  one figure style for every journal on the ladder

Every figure script imports this module. It sets the fonts, sizes and line
widths, holds the fixed model colours and markers, and saves each figure as a
vector PDF (fonts embedded), a 600 dpi LZW TIFF and a 600 dpi PNG.

Notes:
  * a marker shape per model, so identity never rests on colour alone
    (three of the Okabe-Ito colours are light on white);
  * save() checks every figure before writing it: no visible text below 7 pt,
    no line thinner than 0.5 pt, width = 88/120/180 mm, height <= 225 mm,
    and no two text labels overlapping;
  * Arial is used when installed (your PC); Liberation Sans, which has the same
    letter widths, is the fallback on Linux.
"""

import itertools
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

MM = 1 / 25.4
WIDTH = {"single": 88 * MM, "onehalf": 120 * MM, "double": 180 * MM}
MAX_HEIGHT = 225 * MM
MIN_FONT = 7.0          # pt; 6 pt is allowed only for sub/superscripts (none used)
MIN_LINE = 0.5          # pt

MODEL_ORDER = ["Persistence", "XGBoost", "CNN", "Bi-LSTM", "CNN-BiLSTM", "CEEMDAN-BiLSTM"]
# Model colours = the thesis manuscript scheme (replaces the
# brief's set, where CNN and CNN-BiLSTM were two similar blues). Okabe-Ito based;
# CEEMDAN-BiLSTM is charcoal grey, so reference lines are drawn in black or light grey.
MODEL_COLORS = {"Persistence": "#000000", "XGBoost": "#E69F00", "CNN": "#009E73",
                "Bi-LSTM": "#0072B2", "CNN-BiLSTM": "#D55E00", "CEEMDAN-BiLSTM": "#4D4D4D"}
MODEL_MARKERS = {"Persistence": "X", "XGBoost": "s", "CNN": "^", "Bi-LSTM": "o",
                 "CNN-BiLSTM": "D", "CEEMDAN-BiLSTM": "v"}
SPEI_CMAP = "RdBu"      # use with TwoSlopeNorm(vcenter=0)
INK = "#222222"         # dark text/edge ink
GREY = "#8C8C8C"        # reference lines, recessive guides
LIGHT = "#D9D9D9"       # grid / contour lines


def _pick_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"):
        if cand in names:
            return cand
    return "DejaVu Sans"


FONT = _pick_font()


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "mathtext.fontset": "custom", "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
        "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "axes.labelpad": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.labelcolor": INK, "text.color": INK,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5, "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.2,
        "patch.linewidth": 0.5, "hatch.linewidth": 0.5,
        "errorbar.capsize": 0,
        "figure.constrained_layout.use": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 600,
    })


def panel_labels(axes, start="a", dx=-14, dy=4, labels=None):
    """8 pt bold lowercase labels, top-left, offset in points from the axes corner."""
    fig = axes[0].figure
    labels = labels or [chr(ord(start) + i) for i in range(len(axes))]
    for ax, lab in zip(axes, labels):
        tr = ax.transAxes + ScaledTranslation(dx / 72, dy / 72, fig.dpi_scale_trans)
        ax.text(0, 1, lab, transform=tr, fontsize=8, fontweight="bold",
                va="bottom", ha="left")


# ---------------------------------------------------------------------------
# checks run on every figure before it is written
# ---------------------------------------------------------------------------
def _undrawn_tick_labels(fig):
    """Tick labels that matplotlib keeps but does not draw (ticks outside the axis limits)."""
    skip = set()
    for ax in fig.axes:
        axes_list = [ax.xaxis, ax.yaxis]
        for axis in axes_list:
            try:
                drawn = axis._update_ticks()
            except Exception:
                continue
            drawn_ids = {id(t.label1) for t in drawn} | {id(t.label2) for t in drawn}
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                for lab in (tk.label1, tk.label2):
                    if id(lab) not in drawn_ids:
                        skip.add(id(lab))
    return skip


def _visible_texts(fig):
    out = []
    skip = _undrawn_tick_labels(fig)
    for t in fig.findobj(Text):
        if id(t) in skip:
            continue
        if not t.get_visible() or not t.get_text().strip():
            continue
        # skip texts whose parent axes is hidden
        ax = t.axes
        if ax is not None and not ax.get_visible():
            continue
        out.append(t)
    return out


def check_figure(fig, width_key, allow_overlap=()):
    """Return a list of problems (empty = clean)."""
    problems = []
    w, h = fig.get_size_inches()
    if abs(w - WIDTH[width_key]) > 1e-6:
        problems.append(f"width {w / MM:.1f} mm is not {width_key} ({WIDTH[width_key] / MM:.0f} mm)")
    if h > MAX_HEIGHT + 1e-6:
        problems.append(f"height {h / MM:.1f} mm exceeds 225 mm")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = _visible_texts(fig)
    for t in texts:
        if t.get_fontsize() < MIN_FONT - 1e-6:
            problems.append(f"text {t.get_text()!r} is {t.get_fontsize():.1f} pt (< 7 pt)")
    for ln in fig.findobj(Line2D):
        if ln.get_visible() and ln.get_linestyle() not in ("None", "", " ") \
                and len(ln.get_xdata()) > 1 and 0 < ln.get_linewidth() < MIN_LINE - 1e-6:
            problems.append(f"line {ln.get_label()!r} is {ln.get_linewidth():.2f} pt (< 0.5 pt)")

    # text-on-text overlap (labels colliding); figure-pixel boxes, 0.5 px tolerance
    boxes = []
    for t in texts:
        try:
            bb = t.get_window_extent(renderer)
        except Exception:
            continue
        if bb.width > 0 and bb.height > 0:
            boxes.append((t, bb))
    fig_bb = fig.bbox
    for t, bb in boxes:
        if bb.x0 < fig_bb.x0 - 1 or bb.y0 < fig_bb.y0 - 1 or bb.x1 > fig_bb.x1 + 1 or bb.y1 > fig_bb.y1 + 1:
            problems.append(f"text {t.get_text()!r} runs off the page")
    for (t1, b1), (t2, b2) in itertools.combinations(boxes, 2):
        # two labels rotated by the same non-zero angle (e.g. 45 deg tick labels) have
        # overlapping upright bounding boxes even when the glyphs are apart; skip them
        # (they are checked by eye in the preview)
        r1, r2 = t1.get_rotation() % 360, t2.get_rotation() % 360
        if r1 == r2 and r1 not in (0, 180):
            continue
        ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
        oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
        if ox > 0.5 and oy > 0.5:
            pair = (t1.get_text(), t2.get_text())
            if pair in allow_overlap or pair[::-1] in allow_overlap:
                continue
            problems.append(f"texts overlap: {t1.get_text()!r} / {t2.get_text()!r}")
    return problems


def save(fig, stem, outdir, width_key="double", allow_overlap=(), png=True):
    """Check, then write <stem>.pdf (vector, embedded fonts), <stem>.tif (600 dpi, LZW) and
    <stem>.png (600 dpi; the easy file to paste into Word, also fine for submission)."""
    outdir.mkdir(parents=True, exist_ok=True)
    problems = check_figure(fig, width_key, allow_overlap)
    for p in problems:
        warnings.warn(f"[{stem}] {p}")
    fig.savefig(outdir / f"{stem}.pdf")
    fig.savefig(outdir / f"{stem}.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    if png:
        fig.savefig(outdir / f"{stem}.png", dpi=600)
    w, h = fig.get_size_inches()
    print(f"  wrote {stem}.pdf/.tif/.png  ({w / MM:.0f} x {h / MM:.0f} mm, font {FONT}, "
          f"{len(problems)} check problem(s))")
    plt.close(fig)
    return problems
