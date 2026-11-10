"""
Correlation of climate drivers vs SPEI (SPEI-1/3/6/12), divisional monthly data.

Reference layout: Mamun et al. 2024, Sci. Rep. 14:566, Fig. 3.

WHAT THIS PRODUCES
drivers_vs_spei_pooled.png   - annotated heatmap: variables rows x SPEI-1/3/6/12 columns,
                               pooled across divisions, Pearson r, p<0.05 marked with *.
                               Constitutive rows PET and D are drawn below a divider.

full_correlation_matrix.png  - Mamun-style all-vs-all symmetric matrix.

correlation_values.csv       - r and p for every variable-SPEI pair.

optional per-division panel if RUN_PER_DIVISION = True.

DESIGN NOTES
Pearson matches Mamun for direct contrast. Spearman is also saved in the CSV.
Concurrent correlation only, same-month. Association only.

p-values are naive and assume independent samples. Effective-N is printed as a caveat.

PET and D correlate strongly with SPEI partly because SPEI is constructed from D.
"""

import os
import re
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats


# Force serif / Times New Roman-style fonts where available.
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402  (one relative-path config for the whole package)
SPEI_CSV = str(P.SPEI_CSV)
VARS_CSV = str(P.VARS_CSV)
PET_CSV  = str(P.PETWB_CSV)

OUT_DIR  = str(P.WORK / "3_predictor_screening" / "correlation")

RUN_PER_DIVISION = False
METHOD_PRIMARY   = "pearson"


# ----------------------------------------------------------------------------
# PPT-VISIBLE TEXT SIZES
# ----------------------------------------------------------------------------
# If the indicator names still look small in PowerPoint, increase FS_INDICATOR
# and, if needed, FS_TICK.
# ----------------------------------------------------------------------------
FS_TITLE       = 20   # main figure title
FS_INDICATOR   = 18   # indicator / variable names on y-axis
FS_TICK        = 16   # SPEI-1/3/6/12 labels and colorbar ticks
FS_CELL        = 15   # correlation values inside heatmap cells
FS_NOTE        = 13   # footnote / divider note
FS_CBAR_LABEL  = 17   # colorbar label

FS_MATRIX_TICK = 15   # labels in full symmetric matrix
FS_MATRIX_CELL = 13   # cell values in full symmetric matrix


# ----------------------------------------------------------------------------
# Column detection helpers
# ----------------------------------------------------------------------------
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_division_col(df):
    hints = {
        "division",
        "div",
        "divname",
        "name",
        "region",
        "zone",
    }

    for c in df.columns:
        if _norm(c) in hints:
            return c

    for c in df.columns:
        if df[c].dtype == object and 2 <= df[c].nunique() <= 20:
            return c

    raise ValueError("Could not detect a division column. Columns: %s" % list(df.columns))


def build_datetime(df):
    """
    Return a pandas datetime Series, monthly, from whatever date encoding is present.
    """
    cols = {_norm(c): c for c in df.columns}

    for key in ("date", "time", "datetime", "yearmonth", "ym"):
        if key in cols:
            return pd.to_datetime(df[cols[key]], errors="coerce")

    ycol = next((cols[k] for k in cols if k in ("year", "yr", "yyyy")), None)
    mcol = next((cols[k] for k in cols if k in ("month", "mon", "mm")), None)

    if ycol and mcol:
        tmp = pd.DataFrame({
            "year": df[ycol].astype(int),
            "month": df[mcol].astype(int),
            "day": 1
        })
        return pd.to_datetime(tmp, errors="coerce")

    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.9:
            return parsed

    raise ValueError("Could not detect a date/time column. Columns: %s" % list(df.columns))


def detect_spei_cols(df):
    """
    Return dict {1:'SPEI_1', 3:..., 6:..., 12:...}, excluding class/category columns.
    """
    out = {}

    for c in df.columns:
        n = _norm(c)

        if "class" in n or "categ" in n or "flag" in n:
            continue

        m = re.fullmatch(r"spei(\d{1,2})", n)
        if m:
            ts = int(m.group(1))
            if ts in (1, 3, 6, 12) and df[c].dtype != object:
                out[ts] = c

    return out


def pick(df, *aliases):
    """
    Return the first column whose normalized name matches any alias, else None.
    """
    cols = {_norm(c): c for c in df.columns}

    for a in aliases:
        if _norm(a) in cols:
            return cols[_norm(a)]

    return None


# ----------------------------------------------------------------------------
# Load + diagnose
# ----------------------------------------------------------------------------
def load_and_report(path, label):
    if not os.path.exists(path):
        sys.exit("FILE NOT FOUND (%s): %s" % (label, path))

    df = pd.read_csv(path)

    print("\n[%s] %s" % (label, path))
    print("  shape:", df.shape)
    print("  columns:", list(df.columns))
    print("  dtypes:\n   ", df.dtypes.to_dict())
    print("  head:\n", df.head(3).to_string())

    return df


def standardize_keys(df, name):
    dcol = find_division_col(df)
    dt = build_datetime(df)

    out = df.copy()
    out["_division"] = out[dcol].astype(str).str.strip()

    # Normalize to month-start so all files merge cleanly by month.
    dt = pd.to_datetime(dt, errors="coerce")
    out["_date"] = dt.dt.to_period("M").dt.to_timestamp()

    n_bad = out["_date"].isna().sum()
    if n_bad:
        print("  WARNING [%s]: %d rows with unparseable dates" % (name, n_bad))

    print(
        "  detected division col: '%s' (%d unique: %s)"
        % (dcol, out["_division"].nunique(), sorted(out["_division"].unique())[:12])
    )
    print("  date range: %s -> %s" % (out["_date"].min(), out["_date"].max()))

    return out


# ----------------------------------------------------------------------------
# Correlation with p-value, pairwise-complete
# ----------------------------------------------------------------------------
def corr_with_p(x, y, method="pearson"):
    m = np.isfinite(x) & np.isfinite(y)

    if m.sum() < 10:
        return np.nan, np.nan, int(m.sum())

    if method == "pearson":
        r, p = stats.pearsonr(x[m], y[m])
    else:
        r, p = stats.spearmanr(x[m], y[m])

    return r, p, int(m.sum())


def effective_n_note(x, y):
    """
    Rough Bretherton effective N using lag-1 autocorrelation, printed as a caveat.
    """
    m = np.isfinite(x) & np.isfinite(y)

    if m.sum() < 30:
        return None

    xs, ys = x[m], y[m]

    def ac1(v):
        v = np.asarray(v, dtype=float)
        v = v - np.nanmean(v)

        if len(v) < 2 or np.allclose(v, 0.0):
            return 0.0

        r = np.corrcoef(v[:-1], v[1:])[0, 1]
        return 0.0 if not np.isfinite(r) else r

    rx, ry = ac1(xs), ac1(ys)
    n = m.sum()

    denom = (1 + rx * ry)
    if denom <= 0:
        return None

    return n * (1 - rx * ry) / denom


# ----------------------------------------------------------------------------
# Plot: annotated heatmap variables x SPEI timescales
# ----------------------------------------------------------------------------
def plot_driver_heatmap(rmat, pmat, row_labels, ts_cols, n_construct, out_png, title):
    # Larger figure for larger PPT text
    fig, ax = plt.subplots(figsize=(10.5, 0.95 * len(row_labels) + 3.2))

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(rmat, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(len(ts_cols)))
    ax.set_xticklabels(["SPEI-%d" % t for t in ts_cols], fontsize=FS_TICK)

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=FS_INDICATOR)

    ax.xaxis.tick_top()

    for i in range(rmat.shape[0]):
        for j in range(rmat.shape[1]):
            r = rmat[i, j]

            if np.isnan(r):
                continue

            star = "*" if (not np.isnan(pmat[i, j]) and pmat[i, j] < 0.05) else ""
            txt = "%.2f%s" % (r, star)

            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=FS_CELL,
                color="white" if abs(r) > 0.55 else "black"
            )

    # Divider separating climate drivers from constitutive PET and D.
    if 0 < n_construct < len(row_labels):
        y = len(row_labels) - n_construct - 0.5
        ax.axhline(y, color="black", lw=1.8)

        ax.text(
            len(ts_cols) - 0.5,
            y + 0.02,
            " constitutive mechanical \u2193",
            ha="right",
            va="top",
            fontsize=FS_NOTE,
            style="italic",
            color="dimgray"
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
    cbar.set_label("Pearson r", fontsize=FS_CBAR_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK)

    # Slightly smaller than FS_TITLE to avoid long-title clipping
    ax.set_title(title, fontsize=max(16, FS_TITLE - 2), pad=36)

    fig.text(
        0.5,
        0.01,
        "* p < 0.05 naive; see console for effective-N caveat. Concurrent, same-month.",
        ha="center",
        fontsize=FS_NOTE,
        color="dimgray"
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("  saved:", out_png)


def plot_full_matrix(corr, labels, out_png):
    n = len(labels)

    # Larger square figure for larger PPT text
    fig, ax = plt.subplots(figsize=(0.95 * n + 4.0, 0.95 * n + 4.0))

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(corr, cmap="RdBu_r", norm=norm)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=FS_MATRIX_TICK)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=FS_MATRIX_TICK)

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr[i, j]

            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    "%.2f" % v,
                    ha="center",
                    va="center",
                    fontsize=FS_MATRIX_CELL,
                    color="white" if abs(v) > 0.55 else "black"
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=FS_CBAR_LABEL)
    cbar.ax.tick_params(labelsize=FS_TICK)

    ax.set_title("")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("  saved:", out_png)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("LOADING")
    print("=" * 70)

    spei = standardize_keys(load_and_report(SPEI_CSV, "SPEI"), "SPEI")
    vars_ = standardize_keys(load_and_report(VARS_CSV, "VARS"), "VARS")
    pet = standardize_keys(load_and_report(PET_CSV, "PET"), "PET")

    # --- detect target columns ---
    spei_cols = detect_spei_cols(spei)
    ts_present = sorted(spei_cols)

    if not ts_present:
        sys.exit("No SPEI_1/3/6/12 columns detected in the SPEI file.")

    print("\nDetected SPEI targets:", {t: spei_cols[t] for t in ts_present})

    # --- detect the 9 ERA5 variables in the vars file ---
    v = {
        "tp":   pick(vars_, "tp", "precip", "precipitation", "tp_mm"),
        "t2m":  pick(vars_, "t2m", "t2m_c", "tmean", "meantemp", "temperature", "temp"),
        "mx2t": pick(vars_, "mx2t", "tmax", "maxtemp", "mx2t_c"),
        "mn2t": pick(vars_, "mn2t", "tmin", "mintemp", "mn2t_c"),
        "d2m":  pick(vars_, "d2m", "dewpoint", "d2m_c"),
        "ssrd": pick(vars_, "ssrd", "radiation", "solar", "ssrd_mj"),
        "tcc":  pick(vars_, "tcc", "cloud", "cloudcover"),
        "u10":  pick(vars_, "u10", "u_wind", "uwind"),
        "v10":  pick(vars_, "v10", "v_wind", "vwind"),
    }

    print("\nDetected ERA5 variable columns:")
    for k, c in v.items():
        print("  %-5s -> %s" % (k, c))

    missing = [k for k, c in v.items() if c is None]
    if missing:
        print("  NOTE: missing/undetected:", missing, "they will be skipped")

    # --- detect PET and D ---
    pet_col = pick(pet, "pet_mm_month", "pet", "et0", "eto", "pet_mm", "et0_mm", "petmm")

    d_col_petfile = pick(
        pet,
        "d_mm_month",
        "d",
        "d_month",
        "waterbalance",
        "water_balance",
        "wb",
        "d_mm"
    )

    d_col_speifile = pick(
        spei,
        "d_mm_month",
        "d_month",
        "d",
        "waterbalance",
        "wb"
    )

    print(
        "\nPET col:", pet_col,
        "| D in PET file:", d_col_petfile,
        "| D in SPEI file:", d_col_speifile
    )

    # --- assemble a single merged frame on division and date ---
    base = spei[["_division", "_date"] + [spei_cols[t] for t in ts_present]].copy()

    if d_col_speifile:
        base["D"] = spei[d_col_speifile].values

    keep_v = {k: c for k, c in v.items() if c is not None}
    vsub = vars_[["_division", "_date"] + list(keep_v.values())].copy()
    vsub = vsub.rename(columns={c: k for k, c in keep_v.items()})

    psub_cols = ["_division", "_date"]
    if pet_col:
        psub_cols.append(pet_col)
    if d_col_petfile:
        psub_cols.append(d_col_petfile)

    psub = pet[psub_cols].copy()

    ren = {}
    if pet_col:
        ren[pet_col] = "PET"
    if d_col_petfile and "D" not in base.columns:
        ren[d_col_petfile] = "D"

    psub = psub.rename(columns=ren)

    df = (
        base
        .merge(vsub, on=["_division", "_date"], how="inner")
        .merge(psub, on=["_division", "_date"], how="inner")
    )

    print(
        "\nMerged frame shape:", df.shape,
        "| divisions:", df["_division"].nunique(),
        "| months:", df["_date"].nunique()
    )

    # derive wind speed
    if {"u10", "v10"}.issubset(df.columns):
        df["wind_speed"] = np.hypot(df["u10"], df["v10"])

    # --- NaN audit ---
    print("\nNaN audit, leading NaNs in SPEI-3/6/12 are expected by construction:")
    for t in ts_present:
        c = spei_cols[t]
        print("  %s: %d NaN of %d" % (c, df[c].isna().sum(), len(df)))

    # --- ordered driver rows: climate first, then constitutive PET and D ---
    climate_order = [
        "tp",
        "t2m",
        "mx2t",
        "mn2t",
        "d2m",
        "ssrd",
        "tcc",
        "wind_speed",
    ]

    construct_order = ["PET", "D"]

    # Updated label: Temperature -> Temperature (2m)
    label_map = {
        "tp": "Total Precipitation",
        "t2m": "Temperature (2m)",
        "mx2t": "Maximum Temperature",
        "mn2t": "Minimum Temperature",
        "d2m": "Dew Point Temperature",
        "ssrd": "Solar Radiation",
        "tcc": "Total Cloud Cover",
        "wind_speed": "Wind Speed",
        "PET": "PET",
        "D": "Water Balance",
    }

    climate_rows = [c for c in climate_order if c in df.columns]
    construct_rows = [c for c in construct_order if c in df.columns]

    rows = climate_rows + construct_rows
    row_labels = [label_map[c] for c in rows]

    # --- compute correlation matrices, pooled ---
    rmat = np.full((len(rows), len(ts_present)), np.nan)
    pmat = np.full((len(rows), len(ts_present)), np.nan)
    records = []

    print("\nEffective-N check, worst case shown per SPEI column:")

    for j, t in enumerate(ts_present):
        yc = spei_cols[t]
        y = df[yc].to_numpy(dtype=float)

        raw_n = int(np.isfinite(y).sum())
        eff_list = []

        for i, rc in enumerate(rows):
            x = df[rc].to_numpy(dtype=float)

            r, p, n = corr_with_p(x, y, METHOD_PRIMARY)
            rs, ps, _ = corr_with_p(x, y, "spearman")

            rmat[i, j], pmat[i, j] = r, p

            eff = effective_n_note(x, y)
            if eff:
                eff_list.append(eff)

            records.append(dict(
                variable=label_map.get(rc, rc),
                target="SPEI-%d" % t,
                n=n,
                pearson_r=r,
                pearson_p=p,
                spearman_r=rs,
                spearman_p=ps,
                effective_n=eff,
                constitutive=(rc in construct_rows)
            ))

        if eff_list:
            print(
                "  SPEI-%d: raw N~%d, effective N~%d, p-values inflated if ignored"
                % (t, raw_n, int(np.nanmin(eff_list)))
            )

    # --- save CSV ---
    csv_path = os.path.join(OUT_DIR, "correlation_values.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print("\n  saved:", csv_path)

    # --- primary figure ---
    plot_driver_heatmap(
        rmat,
        pmat,
        row_labels,
        ts_present,
        len(construct_rows),
        os.path.join(OUT_DIR, "drivers_vs_spei_pooled.png"),
        "Concurrent correlation: climate drivers vs SPEI, pooled, 8 divisions"
    )

    # --- full symmetric matrix ---
    fulldf = df[rows].copy()

    for t in ts_present:
        fulldf["SPEI-%d" % t] = df[spei_cols[t]].values

    corr = fulldf.corr(method=METHOD_PRIMARY).to_numpy()

    plot_full_matrix(
        corr,
        [label_map.get(c, c) for c in rows] + ["SPEI-%d" % t for t in ts_present],
        os.path.join(OUT_DIR, "full_correlation_matrix.png")
    )

    # --- optional per-division panel ---
    if RUN_PER_DIVISION:
        divs = sorted(df["_division"].unique())
        ncol = 4
        nrow = int(np.ceil(len(divs) / ncol))

        fig, axes = plt.subplots(
            nrow,
            ncol,
            figsize=(5.2 * ncol, 4.2 * nrow),
            squeeze=False
        )

        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        for k, dv in enumerate(divs):
            ax = axes[k // ncol][k % ncol]
            sub = df[df["_division"] == dv]

            rm = np.full((len(rows), len(ts_present)), np.nan)

            for j, t in enumerate(ts_present):
                y = sub[spei_cols[t]].to_numpy(dtype=float)

                for i, rc in enumerate(rows):
                    r, _, _ = corr_with_p(
                        sub[rc].to_numpy(dtype=float),
                        y,
                        METHOD_PRIMARY
                    )
                    rm[i, j] = r

            ax.imshow(rm, cmap="RdBu_r", norm=norm, aspect="auto")

            ax.set_title(dv, fontsize=16)
            ax.set_xticks(range(len(ts_present)))
            ax.set_xticklabels(["S%d" % t for t in ts_present], fontsize=FS_NOTE)

            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels(row_labels, fontsize=max(10, FS_NOTE - 1))

        for k in range(len(divs), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")

        fig.suptitle("Concurrent correlation by division", fontsize=FS_TITLE)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        pth = os.path.join(OUT_DIR, "drivers_vs_spei_per_division.png")
        fig.savefig(pth, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print("  saved:", pth)

    print("\nDONE. Review the console diagnostics above before trusting the figures.")


if __name__ == "__main__":
    main()