"""
================================================================================
run1_build_matrices.py  --  STEP 1 of the lead-time extension
================================================================================
Builds one feature matrix per (SPEI timescale, lead). This is the LEAD-aware
version of build_feature_matrix.py.

WHAT CHANGED FROM THE ORIGINAL
  The original had FORECAST_HORIZON = 1 as a module constant. Here the horizon
  is a loop variable, so leads 1, 2 and 3 are all built in one pass and written
  to separate files. Nothing else about the matrix changes.

WHAT DOES *NOT* CHANGE WITH LEAD
  persistence_baseline stays SPEI at time t. For an h-month lead the naive
  forecast is still "carry the last observed value forward", so the baseline
  column is identical across leads -- only the target moves away from it. That
  is exactly why the skill score gets harder to beat as the lead grows.

STAGE 2 CHANGE (15 Oct 2026)
  deseasonalize() takes the reference years from common.DESEASON_REF:
    "all"   (main pipeline)  -> calendar-month means over all years, as in the thesis
    "train" (strict pipeline) -> calendar-month means over 1980-2018 only (fix F2)
  Nothing else in this script changed.

OUTPUT
  <OUT_ROOT>/_feature_matrix/feature_matrix_SPEI{3,6,12}_lead{1,2,3}.csv
  <OUT_ROOT>/_logs/run1_build_matrices_log.txt
================================================================================
"""

import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import common as C


# =============================================================================
# column detection  (unchanged from the original script)
# =============================================================================

def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_division_col(df):
    for c in df.columns:
        if _norm(c) in {"division", "div", "divname", "name", "region", "zone"}:
            return c
    for c in df.columns:
        if df[c].dtype == object and 2 <= df[c].nunique() <= 20:
            return c
    raise ValueError(f"No division column found in: {list(df.columns)}")


def build_datetime(df):
    """The source 'date' column carries a 06:00:00 timestamp -- parse, then snap
    to the first of the month."""
    cols = {_norm(c): c for c in df.columns}
    if "date" in cols:
        dt = pd.to_datetime(df[cols["date"]], errors="coerce")
        return dt.dt.to_period("M").dt.to_timestamp()
    if "year" in cols and "month" in cols:
        return pd.to_datetime(dict(year=df[cols["year"]].astype(int),
                                   month=df[cols["month"]].astype(int), day=1))
    raise ValueError(f"No date (or year+month) columns in: {list(df.columns)}")


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


# =============================================================================
# load + deseasonalize
# =============================================================================

def load():
    for p, lab in [(C.SPEI_CSV, "SPEI"), (C.VARS_CSV, "VARS")]:
        if not os.path.exists(p):
            sys.exit(f"MISSING ({lab}): {p}")
    spei = pd.read_csv(C.SPEI_CSV)
    vars_ = pd.read_csv(C.VARS_CSV)
    for d in (spei, vars_):
        d["_division"] = d[find_division_col(d)].astype(str).str.strip()
        d["_date"] = build_datetime(d)
    return spei, vars_


def deseasonalize(df, cols, ref="all"):
    """Remove the per-division calendar-month mean from each climate driver.
    SPEI is left as-is (already standardised by construction).

    ref="all"   : means over every year (thesis behaviour, unchanged)
    ref="train" : means over years <= TRAIN_END_YEAR only, then subtracted from
                  every year (F2: no test-period information in the anomalies)"""
    out = df.copy()
    out["_month"] = out["_date"].dt.month
    if ref == "all":
        for c in cols:
            out[c + "_anom"] = out.groupby(["_division", "_month"])[c].transform(
                lambda s: s - s.mean()
            )
    elif ref == "train":
        in_train = out["_date"].dt.year <= C.TRAIN_END_YEAR
        clim = (out[in_train].groupby(["_division", "_month"])[cols].mean()
                .add_suffix("_clim").reset_index())
        out = out.merge(clim, on=["_division", "_month"], how="left")
        for c in cols:
            out[c + "_anom"] = out[c] - out[c + "_clim"]
        out = out.drop(columns=[c + "_clim" for c in cols])
    else:
        raise ValueError(f"deseasonalize: ref must be 'all' or 'train', got {ref!r}")
    return out


# =============================================================================
# build one (timescale, lead) matrix
# =============================================================================

def build_matrix(spei, vars_, scale, target_col, drivers_anom, lead):
    vsub = vars_[["_division", "_date"] + drivers_anom]
    df = (spei[["_division", "_date", target_col]]
          .merge(vsub, on=["_division", "_date"], how="inner")
          .sort_values(["_division", "_date"])
          .reset_index(drop=True))

    parts = []
    for div, g in df.groupby("_division"):
        g = g.sort_values("_date").reset_index(drop=True)
        cols = {"_division": div, "_date": g["_date"].values}

        # climate anomaly lags 1..CLIMATE_MAXLAG -- all strictly past
        for k in range(1, C.CLIMATE_MAXLAG + 1):
            for d in drivers_anom:
                cols[f"{d.replace('_anom', '')}_lag{k}"] = g[d].shift(k).values

        # SPEI at time t. Known at forecast time, so it is a legitimate
        # predictor, not leakage. Same value as persistence_baseline.
        cols["spei_lag0"] = g[target_col].values

        # autoregressive SPEI lags 1..SPEI_LAGS[scale]
        for k in range(1, C.SPEI_LAGS[scale] + 1):
            cols[f"spei_lag{k}"] = g[target_col].shift(k).values

        # persistence baseline: SPEI at t, INDEPENDENT of lead
        cols["persistence_baseline"] = g[target_col].values

        # seasonality
        m = g["_date"].dt.month.values
        cols["month_sin"] = np.sin(2 * np.pi * m / 12)
        cols["month_cos"] = np.cos(2 * np.pi * m / 12)

        # >>> THE LEAD CHANGE <<<  target = SPEI at t + lead
        cols["target"] = g[target_col].shift(-lead).values

        parts.append(pd.DataFrame(cols))

    mat = pd.concat(parts, ignore_index=True)
    mat = pd.concat([mat, pd.get_dummies(mat["_division"], prefix="div")], axis=1)
    return mat


def leakage_check(mat):
    """No predictor column may be (near-)identical to the future target."""
    tgt = mat["target"].to_numpy(dtype=float)
    problems = []
    feat_cols = [c for c in mat.columns
                 if c not in ("_division", "_date", "target") and mat[c].dtype != object]
    for c in feat_cols:
        x = mat[c].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(tgt)
        if m.sum() < 30:
            continue
        if np.corrcoef(x[m], tgt[m])[0, 1] > 0.999:
            problems.append(c)
    return problems


# =============================================================================
# main
# =============================================================================

def main():
    C.ensure_dirs()
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    C.banner("STEP 1  --  BUILD FEATURE MATRICES (leads "
             + ", ".join(str(l) for l in C.LEADS) + ")")
    log(f"run: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"output: {C.FEATURE_DIR}")

    spei, vars_ = load()
    spei_cols = detect_spei_cols(spei)

    scales = [s for s in C.TIMESCALES if s in spei_cols]
    if not scales:
        sys.exit(f"None of {C.TIMESCALES} found in the SPEI file. Found: {spei_cols}")
    log(f"SPEI target columns: { {s: spei_cols[s] for s in scales} }")

    # wind speed from the u10 / v10 components
    u, v = pick(vars_, "u10"), pick(vars_, "v10")
    if u and v:
        vars_["wind"] = np.hypot(vars_[u], vars_[v])
    else:
        log("WARNING: u10 / v10 missing -- 'wind' cannot be built")

    present = [d for d in C.CLIMATE_DRIVERS if d in vars_.columns]
    missing = [d for d in C.CLIMATE_DRIVERS if d not in vars_.columns]
    if missing:
        log(f"WARNING: drivers missing: {missing}")

    vars_ = deseasonalize(vars_, present, ref=C.DESEASON_REF)
    log(f"pipeline: {C.PIPELINE} | SPEI file: {C.SPEI_CSV}")
    log(f"climate anomalies: calendar-month means from "
        + ("ALL years (thesis)" if C.DESEASON_REF == "all" else
           f"training years only (<= {C.TRAIN_END_YEAR})"))
    drivers_anom = [d + "_anom" for d in present]
    log(f"climate drivers used ({len(present)}): {present}")

    summary = []
    for lead in C.LEADS:
        log(f"\n{'-' * 70}\nLEAD = {lead} month(s)\n{'-' * 70}")
        for scale in scales:
            mat = build_matrix(spei, vars_, scale, spei_cols[scale],
                               drivers_anom, lead)

            before = len(mat)
            feat_and_target = [c for c in mat.columns if c not in ("_division", "_date")]
            clean = mat.dropna(subset=feat_and_target).reset_index(drop=True)

            leaks = leakage_check(clean)
            n_feat = len([c for c in clean.columns
                          if c not in ("_division", "_date", "target")
                          and clean[c].dtype != object])

            log(f"\n  SPEI-{scale}  (spei lags 1..{C.SPEI_LAGS[scale]}, "
                f"climate lags 1..{C.CLIMATE_MAXLAG} x {len(present)} vars)")
            log(f"    rows: {before} raw -> {len(clean)} after NaN drop | "
                f"feature columns: {n_feat}")
            log(f"    target range: {clean['target'].min():.2f} .. "
                f"{clean['target'].max():.2f}  (should look like SPEI, ~ -3..3)")

            if leaks:
                log(f"    !! LEAKAGE WARNING -- predictor(s) match the future target: {leaks}")
            else:
                log("    leakage check PASSED")

            out_path = C.feature_matrix_path(scale, lead)
            clean.to_csv(out_path, index=False)
            log(f"    saved: {out_path.name}")

            summary.append(dict(timescale=scale, lead=lead, rows=len(clean),
                                features=n_feat, leaks=len(leaks)))

    # Row counts must fall as the lead grows: a longer target shift drops more
    # months off the end of each division's record.
    log(f"\n{'=' * 70}\nROW-COUNT SANITY CHECK\n{'=' * 70}")
    sdf = pd.DataFrame(summary)
    ok = True
    for scale in scales:
        sub = sdf[sdf.timescale == scale].sort_values("lead")
        counts = sub["rows"].tolist()
        good = all(a > b for a, b in zip(counts, counts[1:]))
        ok &= good
        log(f"  SPEI-{scale}: rows by lead {sub['lead'].tolist()} = {counts}  "
            + ("OK (decreasing)" if good else "*** NOT DECREASING -- CHECK ***"))
    log("\n" + ("All checks passed." if ok else "*** SOMETHING IS WRONG -- stop and investigate ***"))

    sdf.to_csv(C.LOG_DIR / "run1_matrix_summary.csv", index=False)
    with open(C.LOG_DIR / "run1_build_matrices_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))



if __name__ == "__main__":
    main()
