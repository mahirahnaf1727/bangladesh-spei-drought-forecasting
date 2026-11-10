"""
Lead-lag (predictive precedence) analysis: do PAST climate variables predict FUTURE SPEI?
Three complementary tests, each run on RAW predictors and DESEASONALIZED anomalies:
  1. Lagged cross-correlation   - at which lead time (1..MAXLAG months) does |r| peak?
  2. Lagged mutual information   - same, but captures nonlinear lead-lag relationships.
  3. Granger causality           - does a variable's past significantly improve prediction of
                                   future SPEI beyond SPEI's own past? (p-value; controls for
                                   SPEI autocorrelation). Run per-division then pooled (Fisher),
                                   so one division's series never bleeds into another's.

CONTRAST WITH REFERENCE PAPER: Mamun et al. (2024) tested CONCURRENT (same-month) correlation.
This tests LAGGED predictive precedence - the appropriate evidence for a FORECASTING task.

HONEST CAVEATS (read):
  - "Predictive precedence" is NOT physical causation. Report as "leads"/"predicts", never "causes".
  - At SPEI-6/12 an apparent lead is PARTLY rolling-window overlap, not genuine climate precedence.
    SPEI-1 and SPEI-3 give the cleanest evidence. Interpretation is weighted to short timescales.
  - D and PET relate to SPEI partly by CONSTRUCTION. Their lead is expected; the informative
    result is the INDEPENDENT climate drivers' lead.
  - Granger's standard form assumes (roughly) linearity + stationarity. SPEI is stationary by
    design; predictors are checked (ADF). The nonlinear MI test complements the linear Granger.

Reads only; writes figures + CSVs. Never modifies raw data.
"""

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG - your exact paths
# ----------------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402  (one relative-path config for the whole package)
SPEI_CSV = str(P.SPEI_CSV)
VARS_CSV = str(P.VARS_CSV)
PET_CSV  = str(P.PETWB_CSV)
OUT_DIR  = str(P.WORK / "3_predictor_screening" / "leadlag")

MAXLAG      = 12           # months of lead to test
SPEI_SCALES = [1, 3, 6, 12]
USE_ANOMALY_BOTH = True    # compute raw AND anomaly; compare

# ----------------------------------------------------------------------------
# Column detection (same logic already verified against your real files)
# ----------------------------------------------------------------------------
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def find_division_col(df):
    hints = {"division", "div", "divname", "name", "region", "zone"}
    for c in df.columns:
        if _norm(c) in hints:
            return c
    for c in df.columns:
        if df[c].dtype == object and 2 <= df[c].nunique() <= 20:
            return c
    raise ValueError("No division column. Columns: %s" % list(df.columns))

def build_datetime(df):
    cols = {_norm(c): c for c in df.columns}
    for key in ("date", "time", "datetime", "yearmonth", "ym"):
        if key in cols:
            return pd.to_datetime(df[cols[key]], errors="coerce")
    ycol = next((cols[k] for k in cols if k in ("year", "yr", "yyyy")), None)
    mcol = next((cols[k] for k in cols if k in ("month", "mon", "mm")), None)
    if ycol and mcol:
        return pd.to_datetime(dict(year=df[ycol].astype(int), month=df[mcol].astype(int), day=1),
                              errors="coerce")
    for c in df.columns:
        p = pd.to_datetime(df[c], errors="coerce")
        if p.notna().mean() > 0.9:
            return p
    raise ValueError("No date column. Columns: %s" % list(df.columns))

def detect_spei_cols(df):
    out = {}
    for c in df.columns:
        n = _norm(c)
        if "class" in n or "categ" in n or "flag" in n:
            continue
        m = re.fullmatch(r"spei_?(\d{1,2})", n)
        if m and int(m.group(1)) in (1, 3, 6, 12) and df[c].dtype != object:
            out[int(m.group(1))] = c
    return out

def pick(df, *aliases):
    cols = {_norm(c): c for c in df.columns}
    for a in aliases:
        if _norm(a) in cols:
            return cols[_norm(a)]
    return None

def standardize_keys(df):
    dcol = find_division_col(df)
    out = df.copy()
    out["_division"] = out[dcol].astype(str).str.strip()
    out["_date"] = build_datetime(df)
    return out

# ----------------------------------------------------------------------------
# Deseasonalize: subtract each calendar-month mean, per division
# ----------------------------------------------------------------------------
def deseasonalize(df, cols):
    out = df.copy()
    out["_month"] = out["_date"].dt.month
    for c in cols:
        out[c + "_anom"] = out.groupby(["_division", "_month"])[c].transform(lambda s: s - s.mean())
    return out

# ----------------------------------------------------------------------------
# Test 1: lagged cross-correlation (predictor at t-k vs SPEI at t), pooled
# ----------------------------------------------------------------------------
def lagged_xcorr(df, xcol, ycol, maxlag):
    rs = []
    for k in range(1, maxlag + 1):
        vals = []
        for _, g in df.groupby("_division"):
            g = g.sort_values("_date")
            x = g[xcol].shift(k).to_numpy(dtype=float)   # PAST predictor
            y = g[ycol].to_numpy(dtype=float)            # CURRENT SPEI
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() > 10:
                vals.append((x[m], y[m]))
        if not vals:
            rs.append(np.nan); continue
        X = np.concatenate([v[0] for v in vals]); Y = np.concatenate([v[1] for v in vals])
        r, _ = stats.pearsonr(X, Y) if len(X) > 10 else (np.nan, np.nan)
        rs.append(r)
    return np.array(rs)

# ----------------------------------------------------------------------------
# Test 2: lagged mutual information (nonlinear), pooled
# ----------------------------------------------------------------------------
def lagged_mi(df, xcol, ycol, maxlag):
    from sklearn.feature_selection import mutual_info_regression
    mis = []
    for k in range(1, maxlag + 1):
        Xs, Ys = [], []
        for _, g in df.groupby("_division"):
            g = g.sort_values("_date")
            x = g[xcol].shift(k).to_numpy(dtype=float)
            y = g[ycol].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            Xs.append(x[m]); Ys.append(y[m])
        X = np.concatenate(Xs).reshape(-1, 1); Y = np.concatenate(Ys)
        if len(Y) > 50:
            mi = mutual_info_regression(X, Y, random_state=0)[0]
        else:
            mi = np.nan
        mis.append(mi)
    return np.array(mis)

# ----------------------------------------------------------------------------
# Test 3: Granger causality, per-division then Fisher-combined p
# ----------------------------------------------------------------------------
def granger_pooled(df, xcol, ycol, maxlag):
    """Return (min p-value across lags, best lag) combining divisions via Fisher's method."""
    from statsmodels.tsa.stattools import grangercausalitytests
    best_p, best_lag = 1.0, None
    for lag in range(1, maxlag + 1):
        pvals = []
        for _, g in df.groupby("_division"):
            g = g.sort_values("_date")
            sub = g[[ycol, xcol]].dropna()
            if len(sub) < 3 * lag + 20:
                continue
            try:
                res = grangercausalitytests(sub.values, maxlag=[lag], verbose=False)
                p = res[lag][0]["ssr_ftest"][1]
                if np.isfinite(p):
                    pvals.append(p)
            except Exception:
                continue
        if len(pvals) < 3:
            continue
        # Fisher: -2 sum(ln p) ~ chi2(2k)
        stat = -2.0 * np.sum(np.log(np.clip(pvals, 1e-300, 1.0)))
        p_comb = stats.chi2.sf(stat, df=2 * len(pvals))
        if p_comb < best_p:
            best_p, best_lag = p_comb, lag
    return best_p, best_lag

def adf_note(df, col):
    """Fraction of divisions where the predictor is stationary (ADF p<0.05) - a caveat print."""
    from statsmodels.tsa.stattools import adfuller
    ok = 0; tot = 0
    for _, g in df.groupby("_division"):
        s = g[col].dropna().values
        if len(s) < 30:
            continue
        tot += 1
        try:
            if adfuller(s, autolag="AIC")[1] < 0.05:
                ok += 1
        except Exception:
            pass
    return ok, tot

# ----------------------------------------------------------------------------
# Plot: lead-lag curve panel per SPEI scale
# ----------------------------------------------------------------------------
def plot_leadlag(results, drivers, scale, mode, out_png):
    fig, ax = plt.subplots(figsize=(8, 5))
    lags = np.arange(1, MAXLAG + 1)
    for d in drivers:
        r = results[d]
        ax.plot(lags, r, marker="o", ms=3, label=d)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Lead time (months before SPEI)")
    ax.set_ylabel("Lagged Pearson r")
    ax.set_title("Lead-lag: predictors leading SPEI-%d  [%s]" % (scale, mode))
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=180, bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for p, lab in [(SPEI_CSV, "SPEI"), (VARS_CSV, "VARS"), (PET_CSV, "PET")]:
        if not os.path.exists(p):
            sys.exit("MISSING (%s): %s" % (lab, p))

    spei = standardize_keys(pd.read_csv(SPEI_CSV))
    vars_ = standardize_keys(pd.read_csv(VARS_CSV))
    pet   = standardize_keys(pd.read_csv(PET_CSV))

    spei_cols = detect_spei_cols(spei)
    scales = [s for s in SPEI_SCALES if s in spei_cols]
    print("SPEI targets:", {s: spei_cols[s] for s in scales})

    v = {k: pick(vars_, *al) for k, al in {
        "tp":   ("tp", "precip", "precipitation"),
        "t2m":  ("t2m", "tmean", "meantemp"),
        "mx2t": ("mx2t", "tmax", "maxtemp"),
        "mn2t": ("mn2t", "tmin", "mintemp"),
        "d2m":  ("d2m", "dewpoint"),
        "ssrd": ("ssrd", "radiation", "solar"),
        "tcc":  ("tcc", "cloud", "cloudcover"),
        "u10":  ("u10", "uwind"),
        "v10":  ("v10", "vwind"),
    }.items()}
    keep = {k: c for k, c in v.items() if c is not None}
    print("Detected variables:", keep)

    pet_col = pick(pet, "pet_mm_month", "pet", "et0", "eto", "pet_mm")
    d_spei  = pick(spei, "d_mm_month", "d_month", "d")
    d_pet   = pick(pet, "d_mm_month", "d_month", "d")
    print("PET:", pet_col, "| D(spei file):", d_spei, "| D(pet file):", d_pet)

    # merge
    base = spei[["_division", "_date"] + [spei_cols[s] for s in scales]].copy()
    if d_spei: base["D"] = spei[d_spei].values
    vsub = vars_[["_division", "_date"] + list(keep.values())].rename(
        columns={c: k for k, c in keep.items()})
    pcols = ["_division", "_date"] + ([pet_col] if pet_col else []) + \
            ([d_pet] if (d_pet and "D" not in base.columns) else [])
    psub = pet[pcols].rename(columns={**({pet_col: "PET"} if pet_col else {}),
                                      **({d_pet: "D"} if (d_pet and "D" not in base.columns) else {})})
    df = base.merge(vsub, on=["_division", "_date"], how="inner") \
             .merge(psub, on=["_division", "_date"], how="inner")
    if {"u10", "v10"}.issubset(df.columns):
        df["wind"] = np.hypot(df["u10"], df["v10"])
    df = df.sort_values(["_division", "_date"]).reset_index(drop=True)
    print("Merged:", df.shape, "| divisions:", df["_division"].nunique(),
          "| months:", df["_date"].nunique())

    # driver list: climate first, constitutive (D, PET) flagged
    # t2m dropped per the VIF collinearity decision (redundant average of mx2t/mn2t).
    # Lead-lag now runs on the same 7-var decomposed set used downstream, for sequence consistency.
    climate = [c for c in ["tp", "mx2t", "mn2t", "d2m", "ssrd", "tcc", "wind"] if c in df.columns]
    construct = [c for c in ["PET", "D"] if c in df.columns]
    drivers = climate + construct

    # deseasonalize
    df = deseasonalize(df, drivers)

    modes = ["raw", "anom"] if USE_ANOMALY_BOTH else ["raw"]
    all_records = []

    for scale in scales:
        ycol = spei_cols[scale]
        for mode in modes:
            suffix = "" if mode == "raw" else "_anom"
            xcr, xmi = {}, {}
            print("\n=== SPEI-%d [%s] ===" % (scale, mode))
            for d in drivers:
                xc = d + suffix
                r = lagged_xcorr(df, xc, ycol, MAXLAG)
                mi = lagged_mi(df, xc, ycol, MAXLAG)
                gp, gl = granger_pooled(df, xc, ycol, MAXLAG)
                xcr[d], xmi[d] = r, mi
                bestlag_r = int(np.nanargmax(np.abs(r))) + 1 if np.isfinite(r).any() else None
                peak_r = r[bestlag_r - 1] if bestlag_r else np.nan
                print("  %-5s | peak r=%+.3f @lag%s | Granger p=%.4g @lag%s | %s"
                      % (d, peak_r, bestlag_r, gp, gl, "CONSTITUTIVE" if d in construct else ""))
                all_records.append(dict(scale=scale, mode=mode, driver=d,
                                        constitutive=(d in construct),
                                        peak_xcorr=peak_r, peak_xcorr_lag=bestlag_r,
                                        peak_mi=np.nanmax(mi), peak_mi_lag=int(np.nanargmax(mi))+1,
                                        granger_p=gp, granger_lag=gl))
            # plot climate drivers only (constitutive shown separately in CSV)
            plot_leadlag(xcr, climate, scale, mode,
                         os.path.join(OUT_DIR, "leadlag_xcorr_SPEI%d_%s.png" % (scale, mode)))

    # stationarity caveat (raw drivers)
    print("\nADF stationarity check (raw predictors, fraction of divisions stationary):")
    for d in drivers:
        ok, tot = adf_note(df, d)
        print("  %-5s: %d/%d stationary" % (d, ok, tot))

    out_csv = os.path.join(OUT_DIR, "leadlag_summary.csv")
    pd.DataFrame(all_records).to_csv(out_csv, index=False)
    print("\nSaved summary:", out_csv)
    print("Figures + CSV in:", OUT_DIR)
    print("\nREAD: interpret SPEI-1/3 most strongly; SPEI-6/12 leads are partly window overlap.")
    print("      Compare raw vs anom rows in the CSV to settle the deseasonalize decision.")

if __name__ == "__main__":
    main()
