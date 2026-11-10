"""
Fig S5  --  Model architectures (manuscript Figs 4-6 and Appendix Figs A9-A13, §3.9)

One figure with five panels (FigS05: a XGBoost, b CNN, c Bi-LSTM, d CNN-BiLSTM,
e CEEMDAN-BiLSTM) and the same five panels as single files (FigS05a ... FigS05e).

Drawn from the Stage 2 code, not from the thesis drawings:
  dl_models.py         layer order of the four deep-learning models (build_cnn, build_bilstm,
                       build_cnn_bilstm, build_ceemdan_bilstm)
  xgb_model.py         XGBoost (SPEI level predicted directly; early stopping on 2017-2018,
                       up to 3000 trees)
  ceemdan_features.py  CEEMDAN of each window's SPEI channel, 4 IMF channels
  run2_data_prep.py    input layout: 10 sequence channels per month (7 meteorological
                       anomalies at 1-month lag, SPEI(t), month sine and cosine) + 8
                       division one-hots; XGBoost feature vector
Every number in a box is read from the run record (stage2_runs_all.csv: window and
tuned sizes, identical in all seeds and leads) or the data-prep log (feature counts),
and checked against Tables S8 and S9. Dropout rates and learning rates are
left to those tables.

Colour code: white square boxes = inputs and outputs; boxes in the model colour =
layers specific to that model; light grey = the dense head shared by all four deep-learning models.
"""

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Circle

import journal_style as js
import fig_data as D
import p1_fig_config as C

PT = 25.4 / 72                 # mm per point
LSP = 1.15                     # line spacing
LH = 7 * PT * LSP              # line height (mm) at 7 pt
PAD = 1.0                      # inner vertical padding (mm)
GAP = 3.2                      # vertical gap between boxes = arrow length (mm)
SIDE_W = 17                    # width of side-input boxes (mm)
TITLE_H = 5.0                  # panel title band (mm)
HEAD_FILL, HEAD_EDGE = "#F2F2F2", "#8C8C8C"
NAME = {"XGBoost": "XGBoost", "CNN": "CNN", "BiLSTM": "Bi-LSTM", "CNN-BiLSTM": "CNN-BiLSTM",
        "CEEMDAN-BiLSTM": "CEEMDAN-BiLSTM"}

# Tables S8 / S9 as printed (used only as a check)
MS_TABLE5 = {  # model, ts: filters, kernel, units (recurrent or 2nd conv), dense, dropout, lr
    ("CNN", 3): (16, 5, 80, 48, 0.108, 0.008706), ("CNN", 6): (96, 3, 64, 32, 0.280, 0.000359),
    ("CNN", 12): (32, 4, 112, 24, 0.110, 0.001660),
    ("BiLSTM", 3): (None, None, 16, 48, 0.429, 0.003810), ("BiLSTM", 6): (None, None, 16, 64, 0.495, 0.002713),
    ("BiLSTM", 12): (None, None, 128, 32, 0.488, 0.001904),
    ("CNN-BiLSTM", 3): (112, 4, 96, 40, 0.431, 0.000173), ("CNN-BiLSTM", 6): (80, 2, 16, 64, 0.486, 0.004138),
    ("CNN-BiLSTM", 12): (96, 4, 112, 40, 0.419, 0.000205),
    ("CEEMDAN-BiLSTM", 3): (None, None, 32, 48, 0.434, 0.005893),
    ("CEEMDAN-BiLSTM", 6): (None, None, 48, 40, 0.273, 0.000382),
    ("CEEMDAN-BiLSTM", 12): (None, None, 32, 48, 0.429, 0.004019),
}
MS_TABLE6 = {  # max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, lambda, alpha
    3: (3, 0.0441, 0.6138, 0.9637, 3, 0.4468, 0.0177),
    6: (3, 0.1716, 0.9863, 0.9234, 3, 0.0025, 0.5457),
    12: (3, 0.1151, 0.8827, 0.8916, 7, 0.0020, 0.0272),
}
XGB_COLS = ["max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight",
            "reg_lambda", "reg_alpha"]
DL_COLS = ["window", "filters", "kernel_size", "conv2_filters", "lstm_units", "dense_units", "dropout", "lr"]


# =============================================================================
# numbers: run record + data-prep log, checked against the manuscript tables
# =============================================================================

def load_numbers(log):
    runs = pd.read_csv(C.OUT_ROOT / "_tables" / "stage2_runs_all.csv")
    runs = runs[runs.pipeline == "main"]
    P = {}
    for (m, ts), g in runs.groupby(["model", "timescale"]):
        cols = XGB_COLS + ["gamma"] if m == "XGBoost" else DL_COLS
        row = {}
        for c in cols:
            v = g[c].dropna().unique()
            if len(v) > 1:
                log.fail(f"{m} SPEI-{ts}: '{c}' differs between runs ({v})")
            row[c] = v[0] if len(v) else None
        P[(m, int(ts))] = row
    log.ok("window and tuned values read from stage2_runs_all.csv; each is identical in all "
           f"seeds and leads ({len(runs)} runs)")

    worst = []
    for (m, ts), ref in MS_TABLE5.items():
        p = P[(m, ts)]
        units = p["conv2_filters"] if m == "CNN" else p["lstm_units"]
        got = (p["filters"], p["kernel_size"], units, p["dense_units"], p["dropout"], p["lr"])
        for g, r, nd in zip(got, ref, (0, 0, 0, 0, 3, 6)):
            if r is None:
                if g is not None and not (isinstance(g, float) and np.isnan(g)):
                    worst.append(f"{m} SPEI-{ts}: has a value {g} where Table S8 has '-'")
                continue
            if round(float(g), nd) != r:
                worst.append(f"{m} SPEI-{ts}: {g} vs Table S8 {r}")
    for ts, ref in MS_TABLE6.items():
        p = P[("XGBoost", ts)]
        for c, r, nd in zip(XGB_COLS, ref, (0, 4, 4, 4, 0, 4, 4)):
            if round(float(p[c]), nd) != r:
                worst.append(f"XGBoost SPEI-{ts} {c}: {p[c]} vs Table S9 {r}")
    if worst:
        log.fail("tuned values differ from Tables S8/S9: " + "; ".join(worst))
    log.ok("tuned values equal Table S8 (deep-learning models) and Table S9 (XGBoost) as printed")
    log.note("XGBoost also has a tuned gamma (min split loss) = "
             + " / ".join(f"{P[('XGBoost', t)]['gamma']:.4f}" for t in D.TIMESCALES)
             + " (SPEI-3/6/12), listed in Table S9")

    win = {ts: int(P[("CNN", ts)]["window"]) for ts in D.TIMESCALES}
    for m in ("BiLSTM", "CNN-BiLSTM", "CEEMDAN-BiLSTM"):
        if any(int(P[(m, ts)]["window"]) != win[ts] for ts in D.TIMESCALES):
            log.fail(f"{m}: input window differs from the CNN window")
    log.ok("input window T (months) = " + " / ".join(str(win[t]) for t in D.TIMESCALES)
           + " for SPEI-3/6/12, the same in all four deep-learning models (SPEI-6 cut to the last 6 months)")

    nfeat, nch = {}, {}
    lp = C.OUT_ROOT / "_logs" / "run2_data_prep_log.txt"
    if lp.exists():
        txt = lp.read_text(encoding="utf-8", errors="replace")
        for ts, n in re.findall(r"\[SPEI-(\d+) lead 1\] 2D arrays\s*\n\s*X_train=\(\d+, (\d+)\)", txt):
            nfeat[int(ts)] = int(n)
        for ts, n in re.findall(r"\[SPEI-(\d+) lead 1\] 3D arrays\s+\(window=\d+, per-timestep features=(\d+)\)", txt):
            nch[int(ts)] = int(n)
    if set(nfeat) != set(D.TIMESCALES):
        log.note(f"{lp.name} not found or incomplete; XGBoost feature counts taken from the "
                 "pipeline rule 21 + (SPEI lags + 1) + 2 + 8")
        nfeat = {3: 39, 6: 45, 12: 45}
    else:
        log.ok("XGBoost feature vector length (data-prep log, lead 1) = "
               + " / ".join(str(nfeat[t]) for t in D.TIMESCALES))
    if nch and any(v != 18 for v in nch.values()):
        log.fail(f"3D arrays do not have 18 channels per month: {nch}")
    if nch:
        log.ok("3D windows have 18 channels per month (10 sequence + 8 division one-hots)")
    ar = {t: nfeat[t] - 21 - 2 - 8 - 1 for t in D.TIMESCALES}      # highest SPEI lag
    log.ok("XGBoost features = 21 meteorological (7 x lags 1-3) + SPEI lags 0-"
           + " / 0-".join(str(ar[t]) for t in D.TIMESCALES) + " + 2 month terms + 8 division one-hots")
    return P, win, nfeat, ar


def tri(P, model, col, fmt="{:.0f}"):
    return " / ".join(fmt.format(float(P[(model, t)][col])) for t in D.TIMESCALES)


# =============================================================================
# drawing primitives (axes in millimetres)
# =============================================================================

def tint(c, a=0.24):
    r, g, b = to_rgb(c)
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


class Canvas:
    def __init__(self, fig, rect_mm, fig_mm):
        x0, y0, w, h = rect_mm
        fw, fh = fig_mm
        self.ax = fig.add_axes([x0 / fw, y0 / fh, w / fw, h / fh])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.axis("off")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.W, self.H = w, h
        self.boxes = []

    def box(self, xc, ytop, w, title, lines=(), kind="layer", color=None, bold=True):
        n = 1 + len(lines)
        h = n * LH + 2 * PAD
        if kind == "data":
            fc, ec, lw, style = "white", js.INK, 0.7, "square,pad=0"
        elif kind == "head":
            fc, ec, lw, style = HEAD_FILL, HEAD_EDGE, 0.7, "round,pad=0,rounding_size=1.0"
        else:
            fc, ec, lw, style = tint(color), color, 0.9, "round,pad=0,rounding_size=1.0"
        p = FancyBboxPatch((xc - w / 2, ytop - h), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, zorder=2)
        self.ax.add_patch(p)
        texts = [self.ax.text(xc, ytop - PAD, title, ha="center", va="top", zorder=3,
                              fontweight="bold" if bold else "normal", linespacing=LSP)]
        if lines:
            texts.append(self.ax.text(xc, ytop - PAD - LH, "\n".join(lines), ha="center", va="top",
                                      zorder=3, linespacing=LSP))
        b = dict(x0=xc - w / 2, x1=xc + w / 2, y0=ytop - h, y1=ytop, xc=xc, yc=ytop - h / 2, h=h,
                 patch=p, texts=texts)
        self.boxes.append(b)
        return b

    def arrow(self, x0, y0, x1, y1, color=js.INK):
        self.ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                         arrowprops=dict(arrowstyle="-|>", lw=0.7, color=color, mutation_scale=5,
                                         shrinkA=0, shrinkB=0))

    def plus(self, xc, yc, r=1.9):
        self.ax.add_patch(Circle((xc, yc), r, fc="white", ec=js.INK, lw=0.8, zorder=2))
        self.ax.text(xc, yc, "+", ha="center", va="center_baseline", fontsize=8, fontweight="bold",
                     zorder=3)
        return dict(xc=xc, yc=yc, y0=yc - r, y1=yc + r, x0=xc - r, x1=xc + r)

    def title(self, letter, model, ts_sel=None):
        x = 0.0
        if letter:
            self.ax.text(0, self.H, letter, ha="left", va="top", fontsize=8, fontweight="bold")
            x = 3.6
        t = self.ax.text(x, self.H - 0.35, NAME[model], ha="left", va="top", fontweight="bold")
        if ts_sel:
            fig = self.ax.figure
            bb = t.get_window_extent(fig.canvas.get_renderer())
            w_mm = bb.width / fig.dpi * 25.4
            self.ax.text(x + w_mm + 1.2, self.H - 0.35, f"(selected model, SPEI-{ts_sel})", ha="left",
                         va="top")


def box_h(nlines):
    return nlines * LH + 2 * PAD


# =============================================================================
# the shared dense head (bottom-anchored, so the heads line up across panels)
# =============================================================================

HEAD_STACK = (box_h(1), 3.8, box_h(2), box_h(2), box_h(1))   # output, +, dense(1), dense, concat
HEAD_H = sum(HEAD_STACK) + 4 * GAP + 0.6


def draw_head(cv, xc, mw, P, model, concat_w=None, concat_xc=None):
    """Concatenate (+ division one-hots) -> Dense+ReLU, dropout -> Dense(1) -> + SPEI(t) -> forecast.
    Returns the concatenate box (arrows from the model part end on its top edge)."""
    xs = xc + mw / 2 + 3 + SIDE_W / 2                       # side-input column
    y = 0.6
    out = cv.box(xc, y + box_h(1), mw, "Forecast SPEI(t + L)", kind="data")
    y = out["y1"] + GAP
    pl = cv.plus(xc, y + 1.9)
    pers = cv.box(xs, pl["yc"] + box_h(2) / 2, SIDE_W, "SPEI(t)", ["persistence"], kind="data")
    cv.arrow(pers["x0"], pl["yc"], pl["x1"], pl["yc"])
    cv.arrow(xc, pl["y0"], xc, out["y1"])
    y = pl["y1"] + GAP
    d1 = cv.box(xc, y + box_h(2), mw, "Dense (1 unit)", ["predicted ΔSPEI"], kind="head")
    cv.arrow(xc, d1["y0"], xc, pl["y1"])
    y = d1["y1"] + GAP
    dn = cv.box(xc, y + box_h(2), mw, "Dense + ReLU, dropout", [f"units {tri(P, model, 'dense_units')}"],
                kind="head")
    cv.arrow(xc, dn["y0"], xc, d1["y1"])
    y = dn["y1"] + GAP
    cw = concat_w or mw
    cx = concat_xc if concat_xc is not None else xc
    cc = cv.box(cx, y + box_h(1), cw, "Concatenate", kind="head")
    cv.arrow(xc, cc["y0"], xc, dn["y1"])
    xs2 = cc["x1"] + 3 + SIDE_W / 2 if concat_w else xs
    dv = cv.box(xs2, cc["yc"] + box_h(2) / 2, SIDE_W, "Division", ["one-hot (8)"], kind="data")
    cv.arrow(dv["x0"], cc["yc"], cc["x1"], cc["yc"])
    return cc


# =============================================================================
# the five panels
# =============================================================================

def dl_layout(W):
    mw = min(W - SIDE_W - 3 - 2.5, 46)
    left = (W - (mw + 3 + SIDE_W)) / 2
    return left + mw / 2, mw


def stack_down(cv, xc, ytop, w, specs, color):
    """Model-specific boxes from the top down, joined by arrows; returns the last box."""
    prev = None
    y = ytop
    for kind, title, lines in specs:
        b = cv.box(xc, y, w, title, lines, kind=kind, color=color)
        if prev is not None:
            cv.arrow(xc, prev["y0"], xc, b["y1"])
        prev = b
        y = b["y0"] - GAP
    return prev


def input_spec(win):
    return ("data", "Input window", ["T × 10 channels",
                                     "T = " + " / ".join(str(win[t]) for t in D.TIMESCALES) + " months"])


def specs_for(model, P, win):
    if model == "CNN":
        return [input_spec(win),
                ("layer", "Conv1D, batch norm, ReLU", [f"filters {tri(P, model, 'filters')}",
                                                       f"kernel {tri(P, model, 'kernel_size')}"]),
                ("layer", "Conv1D, batch norm, ReLU", [f"filters {tri(P, model, 'conv2_filters')}",
                                                       "same kernel as above"]),
                ("layer", "Spatial dropout, then", ["global average pooling", "over time"])]
    if model == "BiLSTM":
        return [input_spec(win),
                ("layer", "Bidirectional LSTM", [f"units {tri(P, model, 'lstm_units')}",
                                                 "per direction; last states"])]
    if model == "CNN-BiLSTM":
        return [input_spec(win),
                ("layer", "Conv1D, batch norm, ReLU", [f"filters {tri(P, model, 'filters')}",
                                                       f"kernel {tri(P, model, 'kernel_size')}"]),
                ("layer", "Bidirectional LSTM", [f"units {tri(P, model, 'lstm_units')}",
                                                 "per direction; last states"])]
    raise KeyError(model)


def spec_height(specs):
    return sum(box_h(1 + len(s[2])) for s in specs) + GAP * (len(specs) - 1)


def draw_dl(cv, model, P, win, letter):
    color = js.MODEL_COLORS[NAME[model]]
    sel = [t for t, m in D.SELECTED.items() if m == model]
    cv.title(letter, model, sel[0] if sel else None)
    xc, mw = dl_layout(cv.W)
    cc = draw_head(cv, xc, mw, P, model)
    last = stack_down(cv, xc, cv.H - TITLE_H, mw, specs_for(model, P, win), color)
    cv.arrow(xc, last["y0"], xc, cc["y1"])


def ceemdan_geometry(W):
    sw = 38
    group = 48 + 9 + 3 + SIDE_W + 19                      # raw box left edge .. division box right edge
    x0 = (W - group) / 2
    xr = x0 + sw / 2
    xi = xr + 48
    return xr, xi, (xr + xi) / 2, sw


def ceemdan_spec_height():
    return box_h(3) + GAP + 1.2 + box_h(3) + GAP + box_h(2) + GAP + box_h(1)


def draw_ceemdan(cv, P, win, letter):
    model = "CEEMDAN-BiLSTM"
    color = js.MODEL_COLORS[NAME[model]]
    cv.title(letter, model)
    xr, xi, xh, sw = ceemdan_geometry(cv.W)
    mw = 40
    cc = draw_head(cv, xh, mw, P, model, concat_w=(xi - xr) + 18, concat_xc=xh)
    top = cv.H - TITLE_H
    inp = cv.box(xr, top, sw, *input_spec(win)[1:], kind="data")
    cem = cv.box(xi, top, sw, "CEEMDAN", ["SPEI channel of each window",
                                          "4 IMF channels (T × 4)"], kind="layer", color=color)
    cv.arrow(inp["x1"], inp["yc"], cem["x0"], inp["yc"])
    units = tri(P, model, "lstm_units")
    for x, src, lab in ((xr, inp, "raw sequence"), (xi, cem, "IMF sequence")):
        y = src["y0"] - GAP - 1.2
        b1 = cv.box(x, y, sw, "Bidirectional LSTM", [f"units {units}",
                                                    "per direction; every month"], color=color)
        cv.arrow(x, src["y0"], x, b1["y1"])
        cv.ax.text(x + 1.2, (src["y0"] + b1["y1"]) / 2, lab, ha="left", va="center", color="#555555")
        b2 = cv.box(x, b1["y0"] - GAP, sw, "Temporal attention", ["weighted sum over months"],
                    color=color)
        cv.arrow(x, b1["y0"], x, b2["y1"])
        b3 = cv.box(x, b2["y0"] - GAP, sw, "Layer normalization", color=color)
        cv.arrow(x, b2["y0"], x, b3["y1"])
        cv.arrow(x, b3["y0"], x, cc["y1"])


XGB_TREE_H = PAD + 4 * LH + 1.2 + 7.5 + 0.8 + LH + 1.2      # ensemble box: text, glyphs, labels


def xgb_height():
    return TITLE_H + box_h(5) + GAP + XGB_TREE_H + GAP + 3.8 + GAP + box_h(2) + 0.6


def draw_xgb(cv, P, nfeat, ar, letter):
    model = "XGBoost"
    color = js.MODEL_COLORS[model]
    cv.title(letter, model)
    W = cv.W
    w = min(W - 2, 58)
    xc = W / 2
    top = cv.H - TITLE_H
    f = " / ".join(str(nfeat[t]) for t in D.TIMESCALES)
    lags = " / ".join(f"0–{ar[t]}" for t in D.TIMESCALES)
    inp = cv.box(xc, top, w, f"Feature vector, {f} values",
                 ["7 meteorological anomalies × lags 1–3 (21)", f"SPEI(t) and lags, {lags}",
                  "month sine and cosine (2)", "division one-hot (8)"], kind="data")
    # output and sum node at the bottom, level with the models' output and + node
    out = cv.box(xc, 0.6 + box_h(2), w, "Forecast SPEI(t + L)", ["level predicted directly (no ΔSPEI)"],
                 kind="data")
    sig_y = out["y1"] + GAP + 1.9
    cv.ax.add_patch(Circle((xc, sig_y), 1.9, fc="white", ec=js.INK, lw=0.8, zorder=2))
    cv.ax.text(xc, sig_y, "Σ", ha="center", va="center_baseline", fontsize=8, fontweight="bold", zorder=3)
    cv.ax.text(xc - 3.0, sig_y, "sum of all trees", ha="right", va="center_baseline")
    cv.arrow(xc, sig_y - 1.9, xc, out["y1"])
    # tree-ensemble box centred in the space between the input and the sum node
    free = (inp["y0"] - (sig_y + 1.9)) - XGB_TREE_H
    t_top = inp["y0"] - free / 2
    t_bot = t_top - XGB_TREE_H
    eb = FancyBboxPatch((xc - w / 2, t_bot), w, XGB_TREE_H, boxstyle="round,pad=0,rounding_size=1.0",
                        fc=tint(color), ec=color, lw=0.9, zorder=1.5)
    cv.ax.add_patch(eb)
    t1 = cv.ax.text(xc, t_top - PAD, "Gradient-boosted trees", ha="center", va="top", fontweight="bold",
                    zorder=3, linespacing=LSP)
    t2 = cv.ax.text(xc, t_top - PAD - LH, f"max depth {tri(P, model, 'max_depth')}; up to 3000 trees,\n"
                    "early stopping on 2017–2018;\neach tree fits what the earlier trees missed",
                    ha="center", va="top", zorder=3, linespacing=LSP)
    cv.arrow(xc, inp["y0"], xc, t_top)
    cv.arrow(xc, t_bot, xc, sig_y + 1.9)
    # tree glyphs (depth 3, as tuned)
    g_top = t_top - PAD - 4 * LH - 1.2
    g_h = 7.5
    gw = 10.0
    span = w - 10
    xs = [xc - span / 2 + gw / 2 + i * (span - gw) / 3 for i in range(4)]
    labs = []
    for x, nm in zip(xs, ["tree 1", "tree 2", "…", "tree M"]):
        if nm == "…":
            cv.ax.text(x, g_top - g_h / 2, "…", ha="center", va="center", fontsize=8, fontweight="bold")
            continue
        draw_tree(cv.ax, x, g_top, gw, g_h, color)
        labs.append(cv.ax.text(x, g_top - g_h - 0.8, nm, ha="center", va="top"))
    cv.boxes.append(dict(x0=xc - w / 2, x1=xc + w / 2, y0=t_bot, y1=t_top, patch=eb, texts=[t1, t2] + labs))


def draw_tree(ax, xc, ytop, w, h, color, depth=3):
    dy = h / depth
    for lev in range(depth + 1):
        n = 2 ** lev
        xs = [xc - w / 2 + (i + 0.5) * w / n for i in range(n)]
        y = ytop - lev * dy
        if lev < depth:
            n2 = 2 ** (lev + 1)
            xs2 = [xc - w / 2 + (i + 0.5) * w / n2 for i in range(n2)]
            for i, x in enumerate(xs):
                for x2 in (xs2[2 * i], xs2[2 * i + 1]):
                    ax.plot([x, x2], [y, y - dy], color=js.INK, lw=0.5, zorder=3, solid_capstyle="round")
        ax.plot(xs, [y] * n, ls="none", marker="o", ms=2.0 if lev < depth else 1.7,
                mfc=color if lev < depth else "white", mec=js.INK, mew=0.5, zorder=4)


# =============================================================================
# key (notation + colour code), flowed to the figure width
# =============================================================================

def text_w(fig, s, bold=False):
    t = fig.text(0, 0, s, fontweight="bold" if bold else "normal")
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi * 25.4
    t.remove()
    return w


def key_items(models):
    dl = any(m != "XGBoost" for m in models)
    items = [("data", "Input or output")]
    cols = [js.MODEL_COLORS[NAME[m]] for m in models]
    items.append(("layer", "Layers of this model" if len(models) == 1 else "Model-specific layers (model colour)"))
    if dl:
        items.append(("head", "Dense head shared by the four deep-learning models"))
    notes = ["Numbers: SPEI-3 / SPEI-6 / SPEI-12", "L = lead (1–3 months)"]
    if dl:
        notes += ["T = input window (months)",
                  "10 channels per month: 7 meteorological anomalies (1-month lag), SPEI(t), month sine and cosine",
                  "ΔSPEI = SPEI(t + L) − SPEI(t)".replace(" ", "\u00a0")]   # kept on one line
    return items, cols, notes


def key_height(fig, W, models):
    rows, lines, _ = layout_key(fig, W, models)
    return len(rows) * LH * 1.15 + 0.4 + 0.6 + len(lines) * LH + 0.8


def layout_key(fig, W, models):
    items, cols, notes = key_items(models)
    # swatch row (one line if it fits, else wraps)
    rows, cur, x = [], [], 0.0
    for kind, lab in items:
        sw = 4.0 + 1.2 * (len(cols) if kind == "layer" else 1) - (1.2 if kind == "layer" else 0)
        wid = sw + 1.2 + text_w(fig, lab) + 4.0
        if cur and x + wid > W:
            rows.append(cur)
            cur, x = [], 0.0
        cur.append((kind, lab, x, sw))
        x += wid
    rows.append(cur)
    # notes joined by ';', greedy word wrap to the width
    words = []
    for i, n in enumerate(notes):
        ws = n.split(" ")
        if i < len(notes) - 1:
            ws[-1] += ";"
        words += ws
    lines, cur = [], ""
    for wd in words:
        trial = f"{cur} {wd}" if cur else wd
        if cur and text_w(fig, trial) > W - 0.5:
            lines.append(cur)
            cur = wd
        else:
            cur = trial
    lines.append(cur)
    return rows, lines, cols


def draw_key(fig, rect_mm, fig_mm, models):
    x0, y0, w, h = rect_mm
    cv = Canvas(fig, rect_mm, fig_mm)
    ax = cv.ax
    rows, lines, cols = layout_key(fig, w, models)
    y = h - 0.4
    for row in rows:
        yc = y - LH / 2
        for kind, lab, x, sw in row:
            if kind == "data":
                ax.add_patch(FancyBboxPatch((x, yc - 1.4), 4.0, 2.8, boxstyle="square,pad=0", fc="white",
                                            ec=js.INK, lw=0.7))
            elif kind == "head":
                ax.add_patch(FancyBboxPatch((x, yc - 1.4), 4.0, 2.8, boxstyle="round,pad=0,rounding_size=0.6",
                                            fc=HEAD_FILL, ec=HEAD_EDGE, lw=0.7))
            else:
                for i, c in enumerate(cols):
                    ax.add_patch(FancyBboxPatch((x + 1.2 * i, yc - 1.4), 4.0 if len(cols) == 1 else 2.6, 2.8,
                                                boxstyle="round,pad=0,rounding_size=0.5", fc=tint(c), ec=c,
                                                lw=0.7, zorder=2 + i))
            ax.text(x + sw + 1.2, yc, lab, ha="left", va="center_baseline")
        y -= LH * 1.15
    ax.text(0, y - 0.6, "\n".join(lines), ha="left", va="top", linespacing=LSP)
    return cv


# =============================================================================
# checks: every text inside its box, no two boxes overlapping
# =============================================================================

def check_boxes(fig, canvases, log, stem):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    mm = fig.dpi / 25.4
    bad = []
    for cv in canvases:
        for b in cv.boxes:
            pb = b["patch"].get_window_extent(r)
            for t in b["texts"]:
                tb = t.get_window_extent(r)
                if tb.x0 < pb.x0 + 0.4 * mm or tb.x1 > pb.x1 - 0.4 * mm or tb.y0 < pb.y0 + 0.2 * mm \
                        or tb.y1 > pb.y1 - 0.2 * mm:
                    bad.append(f"text {t.get_text()!r} does not fit its box")
        for b1, b2 in ((a, b) for i, a in enumerate(cv.boxes) for b in cv.boxes[i + 1:]):
            ox = min(b1["x1"], b2["x1"]) - max(b1["x0"], b2["x0"])
            oy = min(b1["y1"], b2["y1"]) - max(b1["y0"], b2["y0"])
            if ox > 0 and oy > 0:
                bad.append(f"boxes overlap: {b1['texts'][0].get_text()!r} / {b2['texts'][0].get_text()!r}")
    for p in bad:
        log.fail(f"{stem}: {p}")
    return bad


# =============================================================================
# figures
# =============================================================================

def panel_height(model, P, win):
    if model == "XGBoost":
        return xgb_height()
    if model == "CEEMDAN-BiLSTM":
        return TITLE_H + ceemdan_spec_height() + GAP + HEAD_H
    return TITLE_H + spec_height(specs_for(model, P, win)) + GAP + HEAD_H


def draw_panel(cv, model, P, win, nfeat, ar, letter):
    if model == "XGBoost":
        draw_xgb(cv, P, nfeat, ar, letter)
    elif model == "CEEMDAN-BiLSTM":
        draw_ceemdan(cv, P, win, letter)
    else:
        draw_dl(cv, model, P, win, letter)


def new_fig(w_mm, h_mm):
    js.apply()
    fig = plt.figure(figsize=(w_mm * js.MM, h_mm * js.MM))
    fig.set_layout_engine("none")              # axes are placed in millimetres by hand
    return fig


def make_combined(outdir, log, P, win, nfeat, ar):
    W = 180.0
    row1 = ["XGBoost", "CNN", "BiLSTM"]
    row2 = ["CNN-BiLSTM", "CEEMDAN-BiLSTM"]
    h1 = max(panel_height(m, P, win) for m in row1)
    h2 = max(panel_height(m, P, win) for m in row2)
    probe = new_fig(W, 100)
    kh = key_height(probe, W, D.MODELS)
    plt.close(probe)
    H = kh + 2.0 + h1 + 4.0 + h2 + 0.5
    fig = new_fig(W, H)
    cvs = [draw_key(fig, (0, H - kh, W, kh), (W, H), D.MODELS)]
    y1 = H - kh - 2.0 - h1
    for i, m in enumerate(row1):
        cv = Canvas(fig, (i * 60, y1, 60, h1), (W, H))
        draw_panel(cv, m, P, win, nfeat, ar, "abc"[i])
        cvs.append(cv)
    y2 = y1 - 4.0 - h2
    for (x, w), m, lab in zip(((0, 60), (60, 120)), row2, "de"):
        cv = Canvas(fig, (x, y2, w, h2), (W, H))
        draw_panel(cv, m, P, win, nfeat, ar, lab)
        cvs.append(cv)
    bad = check_boxes(fig, cvs, log, "FigS05")
    log.ok(f"FigS05: {sum(len(c.boxes) for c in cvs)} boxes, every label inside its box, no boxes overlap "
           f"({W:.0f} x {H:.0f} mm)")
    return bad + js.save(fig, "FigS05", outdir, width_key="double")


def make_single(outdir, log, model, suffix, P, win, nfeat, ar):
    wkey = "onehalf" if model == "CEEMDAN-BiLSTM" else "single"
    W = 120.0 if wkey == "onehalf" else 88.0
    pw = W - 4.0
    ph = panel_height(model, P, win)
    probe = new_fig(W, 100)
    kh = key_height(probe, pw, [model])
    plt.close(probe)
    H = kh + 3.0 + ph + 1.0
    fig = new_fig(W, H)
    cvs = [draw_key(fig, (2.0, H - kh - 0.5, pw, kh), (W, H), [model])]
    cv = Canvas(fig, (2.0, 1.0, pw, ph), (W, H))
    draw_panel(cv, model, P, win, nfeat, ar, None)
    cvs.append(cv)
    stem = f"FigS05{suffix}"
    bad = check_boxes(fig, cvs, log, stem)
    return bad + js.save(fig, stem, outdir, width_key=wkey)


def make(outdir, log):
    log.section("Fig S5 - model architectures")
    P, win, nfeat, ar = load_numbers(log)
    problems = make_combined(outdir, log, P, win, nfeat, ar)
    for suffix, m in zip("abcde", D.MODELS):
        problems += make_single(outdir, log, m, suffix, P, win, nfeat, ar)
    log.ok("FigS05a-e written (single panels; 88 mm wide, CEEMDAN-BiLSTM 120 mm)")
    log.note("Fig. S5a-e are the single panels of Fig. S5; the Bi-LSTM is trained with shuffled batches (Table S8).")
    return problems


if __name__ == "__main__":
    log = D.CheckLog()
    make(C.FIG_OUT, log)
    print("\n".join(log.lines))
