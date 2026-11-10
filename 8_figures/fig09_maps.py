"""
Fig 9   --  Spatial verification, worst observed drought month (main text)
Fig S8  --  Spatial verification, near-normal month (SI)

Layout: 2 x 3 grid. Columns SPEI-3/6/12; top row observed, bottom
row the lead-1 forecast of the selected model from its median-skill seed (C8):
CNN-BiLSTM seed 44, Bi-LSTM seed 42, CNN seed 44. Months are those of the thesis
maps (Table S16 (summary)): the most negative and the nearest-to-zero national
mean of the observed index. Each division shows its name and SPEI value. RdBu
centred at 0 (dry = red), one colour scale per figure. Bottom-left text:
national mean (and, for the forecast, the spatial correlation across the 8
divisions), equal to Table S16 (per seed) for the seed shown.

Sources: pred_<model>_SPEI<ts>_lead1_seed<s>.csv, bd_div_wgs.shp,
Table S16 (summary), Table S16 (per seed).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm

import journal_style as js
import fig_data as D
import p1_fig_config as C

CASES = {"worst": "Fig09", "near_normal": "FigS08"}
ASPECT = 1 / np.cos(np.radians(23.7))
LAT_TICKS, LON_TICKS = [21, 22, 23, 24, 25, 26], [88, 89, 90, 91, 92]
# Division labels (name + value, plain black or white text, no halo, as in the thesis maps).
# (label lon, lat), leader-line anchor or None. Five divisions have a position where the
# whole two-line label clears every border; Mymensingh, Barisal and Chittagong are too
# narrow, so their label sits just outside the country with a thin leader line.
# Inside positions are checked with the real label size in place_labels(); the outside
# positions are found there at run time (nearest free spot to the anchor).
LABELS = {"Rangpur": ((89.30, 25.65), None), "Rajshahi": ((88.95, 24.70), None),
          "Dhaka": ((90.15, 23.70), None), "Khulna": ((89.35, 22.80), None),
          "Sylhet": ((91.70, 24.75), None),
          # outside labels: position found at run time (nearest free spot); anchor inside
          "Mymensingh": (None, (90.40, 24.95)),
          "Chittagong": (None, (91.95, 22.60)),
          "Barisal": (None, (90.30, 22.40))}


def load_divisions():
    import geopandas as gpd
    if not C.SHAPEFILE.exists():
        raise FileNotFoundError(f"{C.SHAPEFILE} not found (check SHAPEFILE in p1_fig_config.py)")
    g = gpd.read_file(C.SHAPEFILE)
    if g.crs is not None and g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    col = next(c for c in g.columns if c != "geometry" and g[c].astype(str).isin(
        ["Barisal", "Chittagong", "Dhaka", "Khulna"]).sum() >= 4)
    g = g.rename(columns={col: "division"})[["division", "geometry"]]
    # The shapefile coastline is very detailed (17 MB per figure). Simplify to 0.003 deg
    # (~300 m), which is < 0.04 mm at this map scale, i.e. below one 600 dpi pixel.
    g["geometry"] = g.geometry.simplify(0.003, preserve_topology=True)
    return g


def case_values(case):
    """{ts: (date, model, seed, DataFrame[division, observed, predicted])}"""
    summ = D.table("TableS16_spatial_cases_summary.csv")
    out = {}
    for ts in D.TIMESCALES:
        m = D.SELECTED[ts]
        sd = D.median_seed(m, ts)
        date = pd.Timestamp(summ[(summ.timescale == ts) & (summ.case == case)].date.iloc[0])
        d = D.pred(m, ts, 1, sd)
        v = d[d.target_date == date][["division", "observed", "predicted"]].reset_index(drop=True)
        if len(v) != 8:
            raise ValueError(f"SPEI-{ts} {date:%Y-%m}: {len(v)} divisions, expected 8")
        out[ts] = (date, m, sd, v)
    return out


def checks(log, case, vals):
    per = D.table("TableS16_spatial_cases_per_seed.csv")
    worst = 0.0
    for ts, (date, m, sd, v) in vals.items():
        r = per[(per.timescale == ts) & (per.case == case) & (per.seed == sd)].iloc[0]
        nat_o, nat_p = v.observed.mean(), v.predicted.mean()
        sr = np.corrcoef(v.observed, v.predicted)[0, 1]
        worst = max(worst, abs(nat_o - r.national_obs), abs(nat_p - r.national_pred),
                    abs(sr - r.spatial_r))
        dry_o = v.loc[v.observed.idxmin(), "division"]
        dry_p = v.loc[v.predicted.idxmin(), "division"]
        rev = int(((np.sign(v.observed) * np.sign(v.predicted)) < 0).sum())
        if dry_o != r.driest_obs or dry_p != r.driest_pred or rev != int(r.sign_reversals):
            log.fail(f"{case} SPEI-{ts}: driest/sign-reversal mismatch with Table S16 (per seed)")
        log.ok(f"{case} SPEI-{ts} ({date:%b %Y}, {D.DISPLAY[m]} seed {sd}): national mean obs "
               f"{nat_o:.3f}, pred {nat_p:.3f}; spatial r {sr:.3f}; driest obs {dry_o}, pred {dry_p}; "
               f"sign reversals {rev}")
    if worst > 1e-5:
        log.fail(f"{case}: values differ from Table S16 (per seed) (max {worst})")
    log.ok(f"{case}: national means and spatial r equal Table S16 (per seed) "
           f"(max |diff| {worst:.1e})")


def fmt(v):
    return f"{v:+.2f}".replace("-", "−")


def _measure(ax, text):
    """Size (width, height) of a 7 pt two-line label in data units of this axes."""
    fig = ax.figure
    t = ax.text(0, 0, text, fontsize=7, linespacing=1.05, ha="center", va="center")
    bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
    t.remove()
    return bb.width, bb.height


def place_labels(ax, gdf, natbox, log):
    """Check the inside labels against the borders with their real size, and find the
    nearest free spot outside the country for the three narrow divisions."""
    import itertools
    from shapely.geometry import box, Point
    from shapely.prepared import prep
    land = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.unary_union
    pland = prep(land)
    pos = {}
    for _, r in gdf.iterrows():
        xy, anchor = LABELS[r.division]
        w, h = _measure(ax, f"{r.division}\n−0.00")
        if anchor is None:
            x, y = xy
            b = box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            if not r.geometry.contains(Point(x, y)) or b.intersects(r.geometry.boundary):
                log.fail(f"{r.division}: inside label crosses a border")
            pos[r.division] = (x, y)
    outside = [d for d, (_, a) in LABELS.items() if a is not None]
    sizes = {d: _measure(ax, f"{d}\n−0.00") for d in outside}
    xs = np.arange(88.05, 92.75, 0.05)
    ys = np.arange(20.6, 26.68, 0.05)
    best = None
    for order in itertools.permutations(outside):
        placed, total, trial = [natbox.buffer(0.05)], 0.0, {}
        for d in order:
            w, h = sizes[d]
            ax_, ay_ = LABELS[d][1]
            cand = None
            for x in xs:
                for y in ys:
                    b = box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)
                    if b.bounds[0] < 88.02 or b.bounds[2] > 92.73 or b.bounds[1] < 20.57 \
                            or b.bounds[3] > 26.68:
                        continue
                    if pland.intersects(b) or any(b.intersects(p) for p in placed):
                        continue
                    dist = b.distance(Point(ax_, ay_))
                    if cand is None or dist < cand[0]:
                        cand = (dist, x, y, b)
            if cand is None:
                break
            placed.append(cand[3].buffer(0.06))
            total += cand[0]
            trial[d] = (cand[1], cand[2])
        if len(trial) == len(outside) and (best is None or total < best[0]):
            best = (total, trial)
    if best is None:
        log.fail("no free position for the outside map labels")
    pos.update(best[1])
    log.ok("map labels: Rangpur, Rajshahi, Dhaka, Khulna and Sylhet sit inside their division "
           "without touching a border (checked with the real label size); "
           + ", ".join(f"{d} at {x:.2f}°E {y:.2f}°N" for d, (x, y) in best[1].items())
           + " sit outside the country with a leader line")
    return pos


def draw_case(case, stem, outdir, log, gdf):
    vals = case_values(case)
    checks(log, case, vals)
    allv = np.concatenate([np.r_[v.observed, v.predicted] for _, _, _, v in vals.values()])
    vlim = np.ceil(np.abs(allv).max() / 0.25) * 0.25
    norm = TwoSlopeNorm(0, -vlim, vlim)
    cmap = plt.get_cmap(js.SPEI_CMAP)
    outline = gdf.dissolve()
    js.apply()
    fig, axes = plt.subplots(2, 3, figsize=(js.WIDTH["double"], 196 * js.MM))
    # pass 1: maps, frames, titles, national-mean text, colour bar
    for ci, ts in enumerate(D.TIMESCALES):
        date, m, sd, v = vals[ts]
        for ri, key in enumerate(["observed", "predicted"]):
            ax = axes[ri, ci]
            g = gdf.merge(v, on="division")
            g.plot(column=key, cmap=cmap, norm=norm, edgecolor="white", linewidth=0.6, ax=ax)
            outline.boundary.plot(ax=ax, color=js.INK, linewidth=0.6)
            nat = v[key].mean()
            txt = f"National mean {fmt(nat)}"
            if key == "predicted":
                txt += f"\nSpatial r {np.corrcoef(v.observed, v.predicted)[0, 1]:.2f}"
            ax.text(0.03, 0.03, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=7)
            ax.set_xlabel("")            # geopandas adds CRS axis names; not wanted
            ax.set_ylabel("")
            ax.set_xlim(88.0, 92.75)
            ax.set_ylim(20.55, 26.7)
            ax.set_aspect(ASPECT)
            ax.set_xticks(LON_TICKS)
            ax.set_yticks(LAT_TICKS)
            ax.set_xticklabels([f"{x}°E" for x in LON_TICKS] if ri == 1 else [])
            ax.set_yticklabels([f"{y}°N" for y in LAT_TICKS] if ci == 0 else [])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.6)
            if ri == 0:
                ax.set_title(f"SPEI-{ts} · {date:%B %Y}\nObserved", fontsize=7, pad=3,
                             fontweight="bold")
            else:
                ax.set_title(f"Forecast: {D.DISPLAY[m]} (seed {sd})", fontsize=7, pad=3)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    ticks = np.arange(-vlim, vlim + 1e-9, 0.5 if vlim > 1 else 0.25)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.035, pad=0.04,
                      aspect=40, ticks=ticks)
    cb.set_label("SPEI (red = dry, blue = wet)")
    cb.set_ticklabels([f"{t:g}".replace("-", "−") for t in ticks])
    cb.outline.set_linewidth(0.5)
    js.panel_labels(list(axes.ravel()), dx=-4, dy=14)
    fig.canvas.draw()                     # fix the layout before measuring labels
    # pass 2: division labels (all panels share the same map geometry)
    a0 = axes[1, 0]                       # a panel with the two-line national-mean text
    nb = [t for t in a0.texts if t.get_text().startswith("National mean")][0]
    bb = nb.get_window_extent(fig.canvas.get_renderer()).transformed(a0.transData.inverted())
    from shapely.geometry import box as _box
    pos = place_labels(a0, gdf, _box(bb.x0, bb.y0, bb.x1, bb.y1), log)
    for ci, ts in enumerate(D.TIMESCALES):
        date, m, sd, v = vals[ts]
        for ri, key in enumerate(["observed", "predicted"]):
            ax = axes[ri, ci]
            g = gdf.merge(v, on="division")
            for _, row in g.iterrows():
                x, y = pos[row.division]
                anchor = LABELS[row.division][1]
                val = row[key]
                lab = f"{row.division}\n{fmt(val)}"
                if anchor is None:
                    ax.text(x, y, lab, ha="center", va="center", fontsize=7, linespacing=1.05,
                            color="white" if abs(val) > 0.55 * vlim else js.INK)
                else:
                    ax.annotate(lab, xy=anchor, xytext=(x, y), ha="center", va="center",
                                fontsize=7, linespacing=1.05, color=js.INK,
                                arrowprops=dict(arrowstyle="-", lw=0.5, color=js.INK,
                                                shrinkA=1, shrinkB=0))
                    ax.plot(*anchor, marker="o", ms=1.6, color=js.INK, zorder=5)
    log.ok(f"{stem}: one colour scale for all six maps, ±{vlim:g}")
    return js.save(fig, stem, outdir)


def make(outdir, log):
    log.section("Figs 9 and S8 - spatial verification maps")
    gdf = load_divisions()
    if sorted(gdf.division) != ["Barisal", "Chittagong", "Dhaka", "Khulna", "Mymensingh",
                                "Rajshahi", "Rangpur", "Sylhet"]:
        log.fail(f"shapefile divisions are {sorted(gdf.division)}")
    log.ok("shapefile has the 8 divisions with the same names as the prediction files")
    problems = []
    for case, stem in CASES.items():
        problems += draw_case(case, stem, outdir, log, gdf)
    return problems


if __name__ == "__main__":
    log = D.CheckLog()
    make(C.FIG_OUT, log)
    print("\n".join(log.lines))
