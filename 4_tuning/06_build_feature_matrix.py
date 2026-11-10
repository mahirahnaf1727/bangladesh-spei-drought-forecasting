"""
Feature matrix builder for 1-month-ahead SPEI forecasting.

WHAT IT PRODUCES
  Four feature matrices (one per SPEI timescale: 1/3/6/12). Each row = one (division, month)
  observation. Columns:
    - lagged DESEASONALIZED climate drivers (7-var decomposed set) at short lags
    - lagged own-SPEI (autoregressive) at that timescale's ACF/PACF depth
    - division one-hot + month sin/cos (seasonality)
    - persistence_baseline = SPEI at time t (the naive "next = current" forecast)
    - target = SPEI at time t + HORIZON   <-- the answer column (1-month-ahead)

DECISIONS APPLIED (all settled earlier)
  - Horizon = 1 month ahead (adjustable via FORECAST_HORIZON).
  - Decomposed 7-var climate set: tp, mx2t, mn2t, ssrd, tcc, wind, d2m  (t2m, D, PET excluded).
  - Climate variables DESEASONALIZED (per-division calendar-month mean removed). SPEI used as-is.
  - Climate lag depths: short (from lead-lag). SPEI lag depths: from ACF/PACF (per timescale).
  - Pooled across divisions (division-ID feature), per-timescale matrices.

CRITICAL SAFETY
  All predictors are known as of the target's ORIGIN month t (the target itself is at t+HORIZON).
  Climate predictors are at lag >= 1 relative to t. spei_lag0 is the exception: it is SPEI AT t
  (identical to persistence_baseline) - legitimate because it is known at forecast time, and it is
  the most informative AR feature, not future leakage. spei_lag1..N are SPEI at t-1..t-N.
  A built-in leakage check confirms no predictor column is accidentally aligned with the future
  target. Lags are computed WITHIN each division (no bleed across divisions) and the date is parsed
  explicitly (the source 'date' carries a 06:00:00 timestamp).

OUTPUT: one CSV per timescale + a console leakage/scale report.
"""

import os
import re
import sys
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402  (one relative-path config for the whole package)
SPEI_CSV = str(P.SPEI_CSV)
VARS_CSV = str(P.VARS_CSV)
OUT_DIR  = str(P.WORK / "4_tuning" / "feature_matrix")

FORECAST_HORIZON = 1        # months ahead (1 = predict next month). Change here to add longer leads.

# climate lag depth (short, from lead-lag) - same for all timescales
CLIMATE_MAXLAG = 3
# autoregressive SPEI lag depth per timescale (from ACF/PACF; big ones are an upper bound to test)
SPEI_LAGS = {1: 1, 3: 7, 6: 13, 12: 13}

CLIMATE_DRIVERS = ["tp", "mx2t", "mn2t", "ssrd", "tcc", "wind", "d2m"]

# ----------------------------------------------------------------------------
# Column detection (verified against the real files)
# ----------------------------------------------------------------------------
def _norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())

def find_division_col(df):
    for c in df.columns:
        if _norm(c) in {"division", "div", "divname", "name", "region", "zone"}:
            return c
    for c in df.columns:
        if df[c].dtype == object and 2 <= df[c].nunique() <= 20:
            return c
    raise ValueError("No division column: %s" % list(df.columns))

def build_datetime(df):
    # source 'date' carries a 06:00:00 timestamp - parse then normalize to month start
    cols = {_norm(c): c for c in df.columns}
    if "date" in cols:
        dt = pd.to_datetime(df[cols["date"]], errors="coerce")
        return dt.dt.to_period("M").dt.to_timestamp()   # snap to first of month, drop time
    if "year" in cols and "month" in cols:
        return pd.to_datetime(dict(year=df[cols["year"]].astype(int),
                                   month=df[cols["month"]].astype(int), day=1))
    raise ValueError("No date/year+month columns: %s" % list(df.columns))

def detect_spei_cols(df):
    out = {}
    for c in df.columns:
        n = _norm(c)
        if "class" in n:
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

# ----------------------------------------------------------------------------
# Load, merge, deseasonalize
# ----------------------------------------------------------------------------
def load():
    for p, lab in [(SPEI_CSV, "SPEI"), (VARS_CSV, "VARS")]:
        if not os.path.exists(p):
            sys.exit("MISSING (%s): %s" % (lab, p))
    spei = pd.read_csv(SPEI_CSV)
    vars_ = pd.read_csv(VARS_CSV)
    for d in (spei, vars_):
        d["_division"] = d[find_division_col(d)].astype(str).str.strip()
        d["_date"] = build_datetime(d)
    return spei, vars_

def deseasonalize(df, cols):
    out = df.copy()
    out["_month"] = out["_date"].dt.month
    for c in cols:
        out[c + "_anom"] = out.groupby(["_division", "_month"])[c].transform(lambda s: s - s.mean())
    return out

# ----------------------------------------------------------------------------
# Build one timescale's matrix
# ----------------------------------------------------------------------------
def build_matrix(spei, vars_, scale, target_col, drivers_anom, horizon):
    # merge climate anomalies onto the SPEI frame by (division, date)
    keep = ["_division", "_date", target_col] + drivers_anom
    vsub = vars_[["_division", "_date"] + drivers_anom]
    df = spei[["_division", "_date", target_col]].merge(vsub, on=["_division", "_date"], how="inner")
    df = df.sort_values(["_division", "_date"]).reset_index(drop=True)

    parts = []
    for div, g in df.groupby("_division"):
        g = g.sort_values("_date").reset_index(drop=True)
        cols = {"_division": div, "_date": g["_date"].values}
        # climate lags 1..CLIMATE_MAXLAG (past predictors)
        for k in range(1, CLIMATE_MAXLAG + 1):
            for d in drivers_anom:
                cols["%s_lag%d" % (d.replace("_anom", ""), k)] = g[d].shift(k).values
        # current-month SPEI (lag0) - known at forecast time, same value as persistence_baseline
        cols["spei_lag0"] = g[target_col].values
        # autoregressive SPEI lags 1..SPEI_LAGS[scale] (past predictors)
        for k in range(1, SPEI_LAGS[scale] + 1):
            cols["spei_lag%d" % k] = g[target_col].shift(k).values
        # persistence baseline = SPEI at current month t (the "next = current" naive forecast)
        cols["persistence_baseline"] = g[target_col].values
        # seasonality
        m = g["_date"].dt.month.values
        cols["month_sin"] = np.sin(2 * np.pi * m / 12)
        cols["month_cos"] = np.cos(2 * np.pi * m / 12)
        # TARGET = SPEI at t + horizon  (shift NEGATIVE = look forward)
        cols["target"] = g[target_col].shift(-horizon).values
        parts.append(pd.DataFrame(cols))
    mat = pd.concat(parts, ignore_index=True)

    # division one-hot
    dummies = pd.get_dummies(mat["_division"], prefix="div")
    mat = pd.concat([mat, dummies], axis=1)
    return mat

# ----------------------------------------------------------------------------
# Leakage check: no predictor column may equal the target (future) value
# ----------------------------------------------------------------------------
def leakage_check(mat, scale):
    tgt = mat["target"].to_numpy(dtype=float)
    problems = []
    feat_cols = [c for c in mat.columns
                 if c not in ("_division", "_date", "target") and mat[c].dtype != object]
    for c in feat_cols:
        x = mat[c].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(tgt)
        if m.sum() < 30:
            continue
        # a predictor identical (or near-identical) to the future target => leakage
        if np.corrcoef(x[m], tgt[m])[0, 1] > 0.999:
            problems.append(c)
    return problems

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    spei, vars_ = load()
    spei_cols = detect_spei_cols(spei)
    scales = [s for s in SPEI_LAGS if s in spei_cols]
    print("SPEI targets:", {s: spei_cols[s] for s in scales})
    print("Horizon: %d month(s) ahead | climate lags: 1..%d" % (FORECAST_HORIZON, CLIMATE_MAXLAG))

    # detect + build wind, then deseasonalize climate drivers
    for raw, out in [("u10", None), ("v10", None)]:
        if pick(vars_, raw) is None:
            print("WARNING: %s missing" % raw)
    u, v = pick(vars_, "u10"), pick(vars_, "v10")
    if u and v:
        vars_["wind"] = np.hypot(vars_[u], vars_[v])
    present = [d for d in CLIMATE_DRIVERS if d in vars_.columns]
    missing = [d for d in CLIMATE_DRIVERS if d not in vars_.columns]
    if missing:
        print("WARNING: drivers missing:", missing)
    vars_ = deseasonalize(vars_, present)
    drivers_anom = [d + "_anom" for d in present]

    for scale in scales:
        tcol = spei_cols[scale]
        mat = build_matrix(spei, vars_, scale, tcol, drivers_anom, FORECAST_HORIZON)

        # drop rows with any NaN (from lag creation + target shift + SPEI leading NaNs)
        before = len(mat)
        feat_and_target = [c for c in mat.columns if c not in ("_division", "_date")]
        clean = mat.dropna(subset=feat_and_target).reset_index(drop=True)

        # leakage check on the clean matrix
        leaks = leakage_check(clean, scale)
        n_feat = len([c for c in clean.columns
                      if c not in ("_division", "_date", "target") and clean[c].dtype != object])

        print("\n=== SPEI-%d ===" % scale)
        print("  spei lags: 1..%d | climate lags: 1..%d x %d vars" %
              (SPEI_LAGS[scale], CLIMATE_MAXLAG, len(present)))
        print("  rows: %d raw -> %d after NaN drop | feature columns: %d" % (before, len(clean), n_feat))
        print("  target range: %.2f..%.2f (should look like SPEI, ~ -3..3)" %
              (clean["target"].min(), clean["target"].max()))
        if leaks:
            print("  !! LEAKAGE WARNING - predictor(s) match the future target:", leaks)
        else:
            print("  leakage check PASSED - no predictor equals the future target.")

        out_path = os.path.join(OUT_DIR, "feature_matrix_SPEI%d.csv" % scale)
        clean.to_csv(out_path, index=False)
        print("  saved:", out_path)

    print("\nAll feature matrices written to:", OUT_DIR)
    print("Each is model-ready: predictors + persistence_baseline + target (SPEI at t+%d)." % FORECAST_HORIZON)

if __name__ == "__main__":
    main()
