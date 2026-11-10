"""
================================================================================
 data_prep.py
 Modeling data preparation for deep-learning SPEI drought forecasting (Bangladesh)
================================================================================

WHAT THIS SCRIPT DOES (and does NOT do)
---------------------------------------
This is the DATA-PREP stage. No models are trained here. Its only job is to turn
the four verified feature matrices into clean, model-ready arrays -- exactly once,
correctly -- so that every model downstream (baselines, LSTM, CNN-LSTM,
CEEMD-LSTM) loads from the same leak-free source.

For each timescale (SPEI-1/3/6/12) it:
  1. Loads the feature matrix and normalises the date column to month-start.
  2. Verifies the temporal convention of the matrix (see VALIDATION below) so we
     KNOW the lag alignment is what we think it is, on the real data.
  3. Splits into train / test by the TARGET month's date (leak-free; see NOTE 1).
        train  = targets in 1980..2018
        test   = targets in 2019..2024   (pseudo-future hold-out)
  4. Fits a StandardScaler on the TRAINING rows only, using the climate-anomaly
     columns only (see NOTE 2), and applies it to all rows.
  5. Builds 2D arrays (rows x features) for Linear Regression / XGBoost.
     For SPEI-6/12 it also builds a reduced-lag ("altlag6") 2D set so the
     lag-depth sensitivity test can be run on the fast models first.
  6. Builds 3D arrays (samples x timesteps x features) for the LSTM family.
     *** PROVISIONAL *** -- see NOTE 3. The window length and the windowing
     representation are the one remaining LSTM-design choice; finalise before the
     LSTM phase. The 2D arrays above are final and are what the next phase needs.
  7. Writes a validation log (the "receipt") you can quote in the methods section.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
  - It does not normalise the CSV files themselves.
  - It does not drop NaNs (the matrices are already NaN-dropped; it only checks).
  - It does not shuffle (shuffling a time series = leakage).

--------------------------------------------------------------------------------
NOTE 1 -- WHY SPLIT BY TARGET DATE, NOT FEATURE DATE
  A row dated month t predicts SPEI at t+1 (the target). If we split by the row's
  own date, the row dated Dec-2018 (whose target is Jan-2019, a TEST month) would
  land in TRAIN -- so the model would train on a test-period outcome. That is a
  small (8-row) but real boundary leak. Splitting by the TARGET month's date
  removes it entirely. Effect on results is negligible; the point is a clean,
  defensible "no test outcome was ever a training label".

NOTE 2 -- WHY SCALE CLIMATE COLUMNS ONLY
  The 7 climate predictors are anomalies but still in wildly different physical
  units (precip mm, solar MJ/m2, cloud fraction, temp degC). They MUST be put on
  a common scale so no variable dominates by units and so SHAP magnitudes are
  comparable. SPEI is already standardised (~N(0,1)) by construction, so its lag
  columns are left on their native scale -- this keeps SHAP values on SPEI lags in
  interpretable SPEI units and preserves the persistence relationship. The
  cyclical month_sin/cos (bounded [-1,1]) and the div_* one-hots (0/1) are left
  as-is; standardising them would distort the encoding.

NOTE 3 -- THE 3D ARRAYS ARE PROVISIONAL
  Two things about the 3D representation are still open and should be confirmed
  before building the LSTM:
    (a) SPEI-6/12 window length depends on the lag-depth sensitivity result from
        the fast models (13 vs 6). Provisionally set to the full depth here.
    (b) Representation. This script uses a "proper" windowed LSTM: each timestep
        is a single month's feature vector (current-month climate + current SPEI
        + seasonality + division id), and the window supplies the memory. An
        alternative is the "matched-1-timestep" reshape (feed XGBoost's exact lag
        matrix as a length-1 sequence). The choice affects whether the LSTM is a
        genuine sequence model or an MLP-in-disguise. Provisionally: proper
        windowed, W = max(AR depth, climate-lag depth = 3).

--------------------------------------------------------------------------------
LOCKED DECISIONS honoured here (do not reopen without a reason):
  8 divisions pooled + division-id one-hots; 4 timescales as 4 separate tasks;
  7 decomposed climate predictors as anomalies; 1-month-ahead target;
  temporal hold-out (train <=2018 / test 2019-2024); normalise on train only.

Author: (thesis) -- generated for the modeling data-prep phase.
================================================================================
"""

import os
import re
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIG  -- edit paths here if your folder names differ
# =============================================================================
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402  (one relative-path config for the whole package)
INPUT_DIR  = str(P.WORK / "4_tuning" / "feature_matrix")
OUTPUT_DIR = str(P.WORK / "4_tuning" / "data_prep")

TRAIN_END_YEAR  = 2018   # training targets: years <= this
TEST_START_YEAR = 2019   # test targets:     years >= this  (should be TRAIN_END_YEAR + 1)

# The 7 decomposed climate predictors (locked) and the climate lag depth (locked).
CLIMATE_VARS = ["tp", "mx2t", "mn2t", "ssrd", "tcc", "wind", "d2m"]
CLIMATE_LAGS = [1, 2, 3]

# Per-timescale autoregressive (own-past-SPEI) lag depth, and the reduced depth
# to test on the fast models (only where a reduced depth is in question).
TIMESCALES = {
    "SPEI1":  {"file": "feature_matrix_SPEI1.csv",  "ar_depth": 1},
    "SPEI3":  {"file": "feature_matrix_SPEI3.csv",  "ar_depth": 7},
    "SPEI6":  {"file": "feature_matrix_SPEI6.csv",  "ar_depth": 13, "ar_depth_alt": 6},
    "SPEI12": {"file": "feature_matrix_SPEI12.csv", "ar_depth": 13, "ar_depth_alt": 6},
}

# Identifier / helper columns that are NOT model features.
COL_DIVISION = "_division"
COL_DATE     = "_date"
COL_TARGET   = "target"
COL_PERSIST  = "persistence_baseline"

FLOAT_TOL = 1e-6   # tolerance for the temporal-convention equality checks


# =============================================================================
# small helpers
# =============================================================================
def setup_dirs(output_dir):
    dirs = {
        "arrays":  os.path.join(output_dir, "arrays"),
        "scalers": os.path.join(output_dir, "scalers"),
        "meta":    os.path.join(output_dir, "meta"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def spei_lag_columns(df, ar_depth):
    """Return spei_lag1..spei_lag{ar_depth} present in df, ordered by lag number."""
    found = []
    for c in df.columns:
        m = re.fullmatch(r"spei_lag(\d+)", c)
        if m and int(m.group(1)) <= ar_depth:
            found.append((int(m.group(1)), c))
    return [c for _, c in sorted(found)]


def climate_lag_columns(df):
    return [f"{v}_lag{l}" for v in CLIMATE_VARS for l in CLIMATE_LAGS
            if f"{v}_lag{l}" in df.columns]


def cyclical_columns(df):
    return [c for c in ["month_sin", "month_cos"] if c in df.columns]


def division_onehot_columns(df):
    return sorted([c for c in df.columns if c.startswith("div_")])


def feature_columns_2d(df, ar_depth):
    """Ordered feature list for the 2D matrix (excludes ids / target / persistence)."""
    return (climate_lag_columns(df)
            + spei_lag_columns(df, ar_depth)
            + cyclical_columns(df)
            + division_onehot_columns(df))


def _iso_array(dt_series):
    """Return a numpy fixed-width '<U10' array of YYYY-MM-DD strings.
    Explicit build avoids pandas 3.0 extension-array dtypes that would otherwise
    save as pickled object arrays (which need allow_pickle=True to reload)."""
    return np.array([d.strftime("%Y-%m-%d") for d in dt_series], dtype="U10")


def timestep_columns_3d(df):
    """Per-timestep feature vector for the windowed LSTM: current-month values only.
    current climate = *_lag1 (earliest available climate lag), current SPEI = spei_lag0,
    plus seasonality + div id."""
    clim_cur = [f"{v}_lag1" for v in CLIMATE_VARS if f"{v}_lag1" in df.columns]
    spei_cur = ["spei_lag0"] if "spei_lag0" in df.columns else []
    return clim_cur + spei_cur + cyclical_columns(df) + division_onehot_columns(df)


# =============================================================================
# load + prepare a single matrix
# =============================================================================
def load_and_prepare(path):
    df = pd.read_csv(path)

    if COL_DATE not in df.columns:
        raise ValueError(f"Expected a '{COL_DATE}' column in {path}")
    if COL_DIVISION not in df.columns:
        raise ValueError(f"Expected a '{COL_DIVISION}' column in {path}")

    # Normalise date to month-start (drops any 06:00:00-type timestamp).
    df[COL_DATE] = (pd.to_datetime(df[COL_DATE])
                      .dt.to_period("M").dt.to_timestamp())

    # Target month = feature month + 1 (1-month-ahead forecast).
    df["_target_date"] = df[COL_DATE] + pd.offsets.MonthBegin(1)
    df["_target_year"] = df["_target_date"].dt.year

    # Integer month index for contiguity checks and windowing.
    df["_midx"] = df[COL_DATE].dt.year * 12 + df[COL_DATE].dt.month

    # Split label by TARGET year (see NOTE 1).
    df["_split"] = np.where(df["_target_year"] <= TRAIN_END_YEAR, "train", "test")

    df = df.sort_values([COL_DIVISION, "_midx"]).reset_index(drop=True)
    return df


# =============================================================================
# validation checks (the "receipt")
# =============================================================================
def validate_matrix(df, ts, ar_depth, log):
    log(f"\n[{ts}] INPUT CHECKS")
    log(f"  rows={len(df)}  cols={df.shape[1]}  "
        f"divisions={df[COL_DIVISION].nunique()}")
    log(f"  date span: {df[COL_DATE].min().date()} .. {df[COL_DATE].max().date()}")

    # NaN audit
    n_nan = int(df.isnull().sum().sum())
    log(f"  NaNs in matrix: {n_nan}  " + ("OK" if n_nan == 0 else "*** NOT 0 ***"))

    # persistence_baseline should equal spei_lag0 (both = SPEI at feature month t)
    if COL_PERSIST in df.columns and "spei_lag0" in df.columns:
        d = float(np.max(np.abs(df[COL_PERSIST].values - df["spei_lag0"].values)))
        log(f"  persistence_baseline == spei_lag0 ?  max|diff|={d:.2e}  "
            + ("(identical -> held out of features)" if d < FLOAT_TOL
               else "*** DIFFER -- check builder ***"))

    # Temporal convention: within a division, target[m] == spei_lag0[m+1]
    # (target(m)=SPEI(m+1); spei_lag0(m+1)=SPEI(m+1)). Only over contiguous months.
    worst = 0.0
    n_pairs = 0
    if "spei_lag0" in df.columns:
        for _, g in df.groupby(COL_DIVISION):
            g = g.sort_values("_midx")
            midx = g["_midx"].values
            tgt  = g[COL_TARGET].values
            sl0  = g["spei_lag0"].values
            contig = (midx[1:] == midx[:-1] + 1)
            if contig.any():
                diffs = np.abs(tgt[:-1][contig] - sl0[1:][contig])
                worst = max(worst, float(diffs.max()))
                n_pairs += int(contig.sum())
        log(f"  temporal convention target[m]==spei_lag0[m+1]: "
            f"checked {n_pairs} pairs, max|diff|={worst:.2e}  "
            + ("OK" if worst < FLOAT_TOL else "*** CONVENTION MISMATCH ***"))

    # Month contiguity within divisions (needed for valid windowing).
    gaps = 0
    for _, g in df.groupby(COL_DIVISION):
        midx = np.sort(g["_midx"].values)
        gaps += int(np.sum(np.diff(midx) != 1))
    log(f"  within-division month gaps: {gaps}  "
        + ("(contiguous)" if gaps == 0 else "(windowing will respect runs)"))

    # One-hot sanity: each row's div_* should sum to exactly 1.
    div_cols = division_onehot_columns(df)
    if div_cols:
        rowsum = df[div_cols].sum(axis=1).values
        bad = int(np.sum(np.abs(rowsum - 1.0) > FLOAT_TOL))
        log(f"  division one-hot rows summing to 1: "
            f"{len(df) - bad}/{len(df)}  " + ("OK" if bad == 0 else "*** BAD ***"))


def validate_split(df, ts, log):
    tr = df[df["_split"] == "train"]
    te = df[df["_split"] == "test"]
    n = len(df)
    log(f"\n[{ts}] SPLIT CHECKS  (split by TARGET year)")
    log(f"  train rows={len(tr)} ({100*len(tr)/n:.1f}%)  "
        f"test rows={len(te)} ({100*len(te)/n:.1f}%)")
    log(f"  train target years: {tr['_target_year'].min()}..{tr['_target_year'].max()}")
    log(f"  test  target years: {te['_target_year'].min()}..{te['_target_year'].max()}")
    overlap = set(tr["_target_year"]).intersection(set(te["_target_year"]))
    log(f"  target-year overlap train/test: {sorted(overlap)}  "
        + ("OK (none)" if not overlap else "*** OVERLAP ***"))
    # no train target should reach into test years
    leak = int((tr["_target_year"] >= TEST_START_YEAR).sum())
    log(f"  train targets in test years: {leak}  "
        + ("OK" if leak == 0 else "*** LEAK ***"))


# =============================================================================
# scaling
# =============================================================================
def fit_scaler_transform(df, ts, log):
    """Fit StandardScaler on TRAIN climate columns only; transform all rows."""
    clim = climate_lag_columns(df)
    train_mask = (df["_split"] == "train").values
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, clim].values)

    out = df.copy()
    out[clim] = scaler.transform(out[clim].values)

    # report
    log(f"\n[{ts}] SCALING  (StandardScaler on {len(clim)} climate cols, fit on TRAIN only)")
    tr = out.loc[train_mask, clim].values
    te = out.loc[~train_mask, clim].values
    log(f"  train climate cols after scaling: mean~{tr.mean():+.3f}  std~{tr.std():.3f}  "
        "(should be ~0 / ~1)")
    log(f"  test  climate cols after scaling: mean~{te.mean():+.3f}  std~{te.std():.3f}  "
        "(near 0/1 if distribution stable; not forced)")
    return out, scaler, clim


# =============================================================================
# 2D arrays
# =============================================================================
def build_2d(df, ar_depth):
    feats = feature_columns_2d(df, ar_depth)
    tr = df[df["_split"] == "train"]
    te = df[df["_split"] == "test"]
    pack = {
        "X_train": tr[feats].values.astype(np.float32),
        "y_train": tr[COL_TARGET].values.astype(np.float32),
        "persist_train": tr[COL_PERSIST].values.astype(np.float32),
        "target_date_train": _iso_array(tr["_target_date"]),
        "X_test": te[feats].values.astype(np.float32),
        "y_test": te[COL_TARGET].values.astype(np.float32),
        "persist_test": te[COL_PERSIST].values.astype(np.float32),
        "target_date_test": _iso_array(te["_target_date"]),
    }
    return pack, feats


# =============================================================================
# 3D arrays (PROVISIONAL -- see NOTE 3)
# =============================================================================
def build_3d(df, window_len):
    ts_cols = timestep_columns_3d(df)
    Xs, ys, ps, splits, tds = [], [], [], [], []

    for _, g in df.groupby(COL_DIVISION):
        g = g.sort_values("_midx")
        midx = g["_midx"].values
        Xg = g[ts_cols].values.astype(np.float32)
        yg = g[COL_TARGET].values.astype(np.float32)
        pg = g[COL_PERSIST].values.astype(np.float32)
        sg = g["_split"].values
        tg = _iso_array(g["_target_date"])
        n = len(g)

        # window only within contiguous month runs (robust to any gaps)
        run_start = 0
        for i in range(1, n + 1):
            if i == n or midx[i] != midx[i - 1] + 1:
                for end in range(run_start + window_len - 1, i):
                    Xs.append(Xg[end - window_len + 1:end + 1])
                    ys.append(yg[end])
                    ps.append(pg[end])
                    splits.append(sg[end])
                    tds.append(tg[end])
                run_start = i

    if not Xs:
        return None, ts_cols

    X = np.stack(Xs)                       # (samples, W, F)
    y = np.asarray(ys, dtype=np.float32)
    p = np.asarray(ps, dtype=np.float32)
    s = np.asarray(splits)
    td = np.asarray(tds, dtype="U10")

    tr = s == "train"
    te = s == "test"
    pack = {
        "X_train": X[tr], "y_train": y[tr],
        "persist_train": p[tr], "target_date_train": td[tr],
        "X_test": X[te], "y_test": y[te],
        "persist_test": p[te], "target_date_test": td[te],
    }
    return pack, ts_cols


# =============================================================================
# per-timescale orchestration
# =============================================================================
def run_timescale(ts, cfg, input_dir, dirs, log):
    log("\n" + "=" * 70)
    log(f"TIMESCALE: {ts}   (AR depth = {cfg['ar_depth']})")
    log("=" * 70)

    path = os.path.join(input_dir, cfg["file"])
    df = load_and_prepare(path)

    validate_matrix(df, ts, cfg["ar_depth"], log)
    validate_split(df, ts, log)

    df_s, scaler, clim = fit_scaler_transform(df, ts, log)

    # ---- 2D (full depth) ----
    pack2d, feats = build_2d(df_s, cfg["ar_depth"])
    np.savez_compressed(os.path.join(dirs["arrays"], f"{ts}_2D.npz"), **pack2d)
    log(f"\n[{ts}] 2D arrays (full depth)")
    log(f"  X_train={pack2d['X_train'].shape}  X_test={pack2d['X_test'].shape}  "
        f"features={len(feats)}")
    log(f"  NaNs in X_train/X_test: "
        f"{int(np.isnan(pack2d['X_train']).sum())}/{int(np.isnan(pack2d['X_test']).sum())}")

    meta = {
        "timescale": ts, "ar_depth": cfg["ar_depth"],
        "feature_names_2d": feats, "n_features_2d": len(feats),
        "train_end_year": TRAIN_END_YEAR, "test_start_year": TEST_START_YEAR,
        "n_train_2d": int(pack2d["X_train"].shape[0]),
        "n_test_2d": int(pack2d["X_test"].shape[0]),
        "scaled_columns": clim,
    }

    # ---- 2D (reduced depth) for the sensitivity test ----
    if "ar_depth_alt" in cfg:
        pack2d_alt, feats_alt = build_2d(df_s, cfg["ar_depth_alt"])
        np.savez_compressed(
            os.path.join(dirs["arrays"], f"{ts}_2D_altlag{cfg['ar_depth_alt']}.npz"),
            **pack2d_alt)
        log(f"\n[{ts}] 2D arrays (reduced depth = {cfg['ar_depth_alt']})  "
            "-- for lag-depth sensitivity on fast models")
        log(f"  X_train={pack2d_alt['X_train'].shape}  features={len(feats_alt)}  "
            f"({len(feats) - len(feats_alt)} fewer spei_lag cols)")
        meta["ar_depth_alt"] = cfg["ar_depth_alt"]
        meta["feature_names_2d_alt"] = feats_alt
        meta["n_features_2d_alt"] = len(feats_alt)

    # ---- 3D (PROVISIONAL) ----
    window_len = max(cfg["ar_depth"], max(CLIMATE_LAGS))   # see NOTE 3
    pack3d, ts_cols = build_3d(df_s, window_len)
    if pack3d is not None:
        np.savez_compressed(os.path.join(dirs["arrays"], f"{ts}_3D.npz"), **pack3d)
        log(f"\n[{ts}] 3D arrays  *** PROVISIONAL *** (window={window_len}, "
            f"per-timestep features={len(ts_cols)})")
        log(f"  X_train={pack3d['X_train'].shape}  X_test={pack3d['X_test'].shape}  "
            "(samples, timesteps, features)")
        log(f"  NaNs in 3D X_train/X_test: "
            f"{int(np.isnan(pack3d['X_train']).sum())}/{int(np.isnan(pack3d['X_test']).sum())}")
        meta["window_len_3d_provisional"] = int(window_len)
        meta["timestep_feature_names_3d"] = ts_cols
        meta["n_train_3d"] = int(pack3d["X_train"].shape[0])
        meta["n_test_3d"] = int(pack3d["X_test"].shape[0])

    # ---- save scaler + meta ----
    with open(os.path.join(dirs["scalers"], f"{ts}_scaler.pkl"), "wb") as f:
        pickle.dump({"scaler": scaler, "scaled_columns": clim}, f)
    with open(os.path.join(dirs["meta"], f"{ts}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"\n[{ts}] saved: 2D/3D arrays, scaler.pkl, meta.json")


# =============================================================================
# main
# =============================================================================
def main(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    dirs = setup_dirs(output_dir)
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("#" * 70)
    log("# DATA PREP -- SPEI drought forecasting (modeling stage)")
    log(f"# run: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"# input : {input_dir}")
    log(f"# output: {output_dir}")
    log(f"# split : train targets <= {TRAIN_END_YEAR} | test targets >= {TEST_START_YEAR}")
    log("#" * 70)

    for ts, cfg in TIMESCALES.items():
        run_timescale(ts, cfg, input_dir, dirs, log)

    log("\n" + "#" * 70)
    log("# DONE. Next phase: baselines (persistence, climatology, linear reg, XGBoost).")
    log("# 2D arrays are final. 3D arrays are PROVISIONAL (confirm windowing before LSTM).")
    log("#" * 70)

    with open(os.path.join(output_dir, "prep_validation_log.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
