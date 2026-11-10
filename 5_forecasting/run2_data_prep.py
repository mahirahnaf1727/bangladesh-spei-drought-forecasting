"""
================================================================================
run2_data_prep.py  --  STEP 2 of the lead-time extension
================================================================================
Turns each feature matrix into model-ready arrays. This is the LEAD-aware
version of data_prep.py.

THE TWO PLACES THE ORIGINAL HARD-CODED A 1-MONTH LEAD
  1. The target date was computed as  _date + MonthBegin(1).
     Left alone, a lead-3 row dated Oct-2018 would still be labelled as
     targeting Nov-2018 and would land in TRAIN -- while it actually predicts
     Jan-2019, a TEST month. That is a real boundary leak, and it gets worse as
     the lead grows. It is now  _date + MonthBegin(LEAD).

  2. The temporal-convention check asserted  target[m] == spei_lag0[m+1].
     At lead 2 or 3 that assertion is simply the wrong statement, so it is now
     generalised to  target[m] == spei_lag0[m+LEAD], checked only over runs of
     contiguous months. It is NOT deleted -- a failure here means the target
     shift went wrong and everything downstream is meaningless.

GUARDRAILS ADDED
  A. Hard assertion that every training target month strictly precedes every
     test target month, recomputed from the shifted dates.
  B. Windows are formed only within contiguous runs of months inside a
     division, and the window never reaches past the forecast origin t.

OUTPUT
  <OUT_ROOT>/_data_prep/arrays/SPEI{ts}_{2D,3D}_lead{L}.npz
  <OUT_ROOT>/_data_prep/scalers/SPEI{ts}_scaler_lead{L}.pkl
  <OUT_ROOT>/_data_prep/meta/SPEI{ts}_meta_lead{L}.json
  <OUT_ROOT>/_logs/run2_data_prep_log.txt
================================================================================
"""

import json
import pickle
import re
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import common as C

COL_DIVISION = "_division"
COL_DATE     = "_date"
COL_TARGET   = "target"
COL_PERSIST  = "persistence_baseline"

FLOAT_TOL = 1e-6


# =============================================================================
# column groups
# =============================================================================

def spei_lag_columns(df, ar_depth):
    found = []
    for c in df.columns:
        m = re.fullmatch(r"spei_lag(\d+)", c)
        if m and int(m.group(1)) <= ar_depth:
            found.append((int(m.group(1)), c))
    return [c for _, c in sorted(found)]


def climate_lag_columns(df):
    return [f"{v}_lag{l}" for v in C.CLIMATE_DRIVERS for l in C.CLIMATE_LAGS
            if f"{v}_lag{l}" in df.columns]


def cyclical_columns(df):
    return [c for c in ["month_sin", "month_cos"] if c in df.columns]


def division_onehot_columns(df):
    return sorted([c for c in df.columns if c.startswith("div_")])


def feature_columns_2d(df, ar_depth):
    return (climate_lag_columns(df)
            + spei_lag_columns(df, ar_depth)
            + cyclical_columns(df)
            + division_onehot_columns(df))


def timestep_columns_3d(df):
    """Per-timestep vector for the sequence models: current-month climate
    (*_lag1), current SPEI (spei_lag0), seasonality, division id. 18 channels."""
    clim_cur = [f"{v}_lag1" for v in C.CLIMATE_DRIVERS if f"{v}_lag1" in df.columns]
    spei_cur = ["spei_lag0"] if "spei_lag0" in df.columns else []
    return clim_cur + spei_cur + cyclical_columns(df) + division_onehot_columns(df)


def _iso_array(dt_series):
    """Fixed-width '<U10' YYYY-MM-DD array. Built explicitly so the npz saves as
    a plain string array rather than a pickled object array."""
    return np.array([d.strftime("%Y-%m-%d") for d in dt_series], dtype="U10")


# =============================================================================
# load + split  --  where the LEAD actually matters
# =============================================================================

def load_and_prepare(path, lead):
    df = pd.read_csv(path)

    for col in (COL_DATE, COL_DIVISION):
        if col not in df.columns:
            raise ValueError(f"Expected a '{col}' column in {path}")

    df[COL_DATE] = pd.to_datetime(df[COL_DATE]).dt.to_period("M").dt.to_timestamp()

    # >>> THE LEAD CHANGE <<<  was MonthBegin(1)
    df["_target_date"] = df[COL_DATE] + pd.offsets.MonthBegin(lead)
    df["_target_year"] = df["_target_date"].dt.year

    df["_midx"] = df[COL_DATE].dt.year * 12 + df[COL_DATE].dt.month

    # Split on the TARGET year, recomputed from the shifted date.
    df["_split"] = np.where(df["_target_year"] <= C.TRAIN_END_YEAR, "train", "test")

    return df.sort_values([COL_DIVISION, "_midx"]).reset_index(drop=True)


# =============================================================================
# validation
# =============================================================================

def validate_matrix(df, ts, lead, log):
    log(f"\n[SPEI-{ts} lead {lead}] INPUT CHECKS")
    log(f"  rows={len(df)}  cols={df.shape[1]}  divisions={df[COL_DIVISION].nunique()}")
    log(f"  feature-date span: {df[COL_DATE].min().date()} .. {df[COL_DATE].max().date()}")
    log(f"  target-date  span: {df['_target_date'].min().date()} .. "
        f"{df['_target_date'].max().date()}")

    n_nan = int(df.isnull().sum().sum())
    log(f"  NaNs in matrix: {n_nan}  " + ("OK" if n_nan == 0 else "*** NOT 0 ***"))

    # persistence_baseline must equal spei_lag0 (both are SPEI at t), at EVERY lead
    if COL_PERSIST in df.columns and "spei_lag0" in df.columns:
        d = float(np.max(np.abs(df[COL_PERSIST].values - df["spei_lag0"].values)))
        log(f"  persistence_baseline == spei_lag0 ?  max|diff|={d:.2e}  "
            + ("OK (held out of features)" if d < FLOAT_TOL else "*** DIFFER ***"))

    # ---- generalised temporal convention: target[m] == spei_lag0[m + lead] ----
    worst, n_pairs = 0.0, 0
    if "spei_lag0" in df.columns:
        for _, g in df.groupby(COL_DIVISION):
            g = g.sort_values("_midx")
            midx = g["_midx"].values
            tgt  = g[COL_TARGET].values
            sl0  = g["spei_lag0"].values
            if len(g) <= lead:
                continue
            # only compare rows exactly `lead` months apart
            contig = (midx[lead:] == midx[:-lead] + lead)
            if contig.any():
                diffs = np.abs(tgt[:-lead][contig] - sl0[lead:][contig])
                worst = max(worst, float(diffs.max()))
                n_pairs += int(contig.sum())
        status = "OK" if worst < FLOAT_TOL else "*** CONVENTION MISMATCH -- STOP ***"
        log(f"  temporal convention target[m] == spei_lag0[m+{lead}]: "
            f"checked {n_pairs} pairs, max|diff|={worst:.2e}  {status}")
        if worst >= FLOAT_TOL:
            raise AssertionError(
                f"SPEI-{ts} lead {lead}: target is not SPEI at t+{lead}. "
                "The target shift is wrong -- do not proceed."
            )

    gaps = 0
    for _, g in df.groupby(COL_DIVISION):
        midx = np.sort(g["_midx"].values)
        gaps += int(np.sum(np.diff(midx) != 1))
    log(f"  within-division month gaps: {gaps}  "
        + ("(contiguous)" if gaps == 0 else "(windowing respects runs)"))

    div_cols = division_onehot_columns(df)
    if div_cols:
        rowsum = df[div_cols].sum(axis=1).values
        bad = int(np.sum(np.abs(rowsum - 1.0) > FLOAT_TOL))
        log(f"  division one-hot rows summing to 1: {len(df) - bad}/{len(df)}  "
            + ("OK" if bad == 0 else "*** BAD ***"))


def validate_split(df, ts, lead, log):
    tr = df[df["_split"] == "train"]
    te = df[df["_split"] == "test"]
    n = len(df)

    log(f"\n[SPEI-{ts} lead {lead}] SPLIT CHECKS  (split by TARGET year)")
    log(f"  train rows={len(tr)} ({100 * len(tr) / n:.1f}%)   "
        f"test rows={len(te)} ({100 * len(te) / n:.1f}%)")
    log(f"  train target years: {tr['_target_year'].min()}..{tr['_target_year'].max()}")
    log(f"  test  target years: {te['_target_year'].min()}..{te['_target_year'].max()}")

    # ---- GUARDRAIL A -------------------------------------------------------
    max_train = tr["_target_date"].max()
    min_test  = te["_target_date"].min()
    log(f"  latest train target = {max_train.date()} | "
        f"earliest test target = {min_test.date()}")
    assert max_train < min_test, (
        f"SPEI-{ts} lead {lead}: a training target ({max_train.date()}) reaches "
        f"the test period ({min_test.date()}). The split was not recomputed "
        "from the shifted target date."
    )
    log("  GUARDRAIL A: max(train target) < min(test target)  OK")

    overlap = set(tr["_target_year"]) & set(te["_target_year"])
    log(f"  target-year overlap: {sorted(overlap)}  "
        + ("OK (none)" if not overlap else "*** OVERLAP ***"))


# =============================================================================
# scaling
# =============================================================================

def fit_scaler_transform(df, ts, lead, log):
    """StandardScaler on the climate anomaly columns only, fit on TRAIN rows only.

    SPEI lag columns are left on their native scale: SPEI is already ~N(0,1) by
    construction, so scaling it would distort the persistence relationship and
    make SHAP values uninterpretable. month_sin/cos and the one-hots are
    bounded encodings and are left alone."""
    clim = climate_lag_columns(df)
    train_mask = (df["_split"] == "train").values

    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, clim].values)

    out = df.copy()
    out[clim] = scaler.transform(out[clim].values)

    tr = out.loc[train_mask, clim].values
    te = out.loc[~train_mask, clim].values
    log(f"\n[SPEI-{ts} lead {lead}] SCALING  "
        f"({len(clim)} climate cols, fit on TRAIN only)")
    log(f"  train after scaling: mean~{tr.mean():+.3f}  std~{tr.std():.3f}  (expect ~0 / ~1)")
    log(f"  test  after scaling: mean~{te.mean():+.3f}  std~{te.std():.3f}  (not forced)")
    return out, scaler, clim


# =============================================================================
# array builders
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
        "division_train": np.asarray(tr[COL_DIVISION].values, dtype="U32"),
        "X_test": te[feats].values.astype(np.float32),
        "y_test": te[COL_TARGET].values.astype(np.float32),
        "persist_test": te[COL_PERSIST].values.astype(np.float32),
        "target_date_test": _iso_array(te["_target_date"]),
        "division_test": np.asarray(te[COL_DIVISION].values, dtype="U32"),
    }
    return pack, feats


def build_3d(df, window_len):
    """Sliding windows within contiguous month runs inside each division.

    GUARDRAIL B: the window ends at the forecast origin t, and every channel in
    it is known at or before t (current climate is *_lag1, current SPEI is
    spei_lag0). The target sits at t+lead and is never inside the window."""
    ts_cols = timestep_columns_3d(df)
    Xs, ys, ps, splits, tds, dvs = [], [], [], [], [], []

    for div, g in df.groupby(COL_DIVISION):
        g = g.sort_values("_midx")
        midx = g["_midx"].values
        Xg = g[ts_cols].values.astype(np.float32)
        yg = g[COL_TARGET].values.astype(np.float32)
        pg = g[COL_PERSIST].values.astype(np.float32)
        sg = g["_split"].values
        tg = _iso_array(g["_target_date"])
        n = len(g)

        run_start = 0
        for i in range(1, n + 1):
            if i == n or midx[i] != midx[i - 1] + 1:
                for end in range(run_start + window_len - 1, i):
                    Xs.append(Xg[end - window_len + 1:end + 1])
                    ys.append(yg[end])
                    ps.append(pg[end])
                    splits.append(sg[end])
                    tds.append(tg[end])
                    dvs.append(div)
                run_start = i

    if not Xs:
        return None, ts_cols

    X  = np.stack(Xs)
    y  = np.asarray(ys, dtype=np.float32)
    p  = np.asarray(ps, dtype=np.float32)
    s  = np.asarray(splits)
    td = np.asarray(tds, dtype="U10")
    dv = np.asarray(dvs, dtype="U32")

    tr, te = (s == "train"), (s == "test")
    pack = {
        "X_train": X[tr], "y_train": y[tr], "persist_train": p[tr],
        "target_date_train": td[tr], "division_train": dv[tr],
        "X_test": X[te], "y_test": y[te], "persist_test": p[te],
        "target_date_test": td[te], "division_test": dv[te],
    }
    return pack, ts_cols


# =============================================================================
# per (timescale, lead)
# =============================================================================

def run_one(ts, lead, log):
    ar_depth = C.SPEI_LAGS[ts]

    log("\n" + "=" * 74)
    log(f"TIMESCALE SPEI-{ts}   LEAD {lead}   (AR depth = {ar_depth})")
    log("=" * 74)

    path = C.feature_matrix_path(ts, lead)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run run1_build_matrices.py first.")

    df = load_and_prepare(path, lead)
    validate_matrix(df, ts, lead, log)
    validate_split(df, ts, lead, log)

    df_s, scaler, clim = fit_scaler_transform(df, ts, lead, log)

    # ---- 2D (XGBoost) ----
    pack2d, feats = build_2d(df_s, ar_depth)
    np.savez_compressed(C.npz_path(ts, lead, "2D"), **pack2d)
    log(f"\n[SPEI-{ts} lead {lead}] 2D arrays")
    log(f"  X_train={pack2d['X_train'].shape}  X_test={pack2d['X_test'].shape}  "
        f"features={len(feats)}")

    # ---- 3D (sequence models) ----
    window_len = max(ar_depth, max(C.CLIMATE_LAGS))
    pack3d, ts_cols = build_3d(df_s, window_len)
    if pack3d is None:
        raise RuntimeError(f"SPEI-{ts} lead {lead}: no 3D windows could be formed.")

    np.savez_compressed(C.npz_path(ts, lead, "3D"), **pack3d)
    log(f"[SPEI-{ts} lead {lead}] 3D arrays  "
        f"(window={window_len}, per-timestep features={len(ts_cols)})")
    log(f"  X_train={pack3d['X_train'].shape}  X_test={pack3d['X_test'].shape}  "
        "(samples, timesteps, channels)")

    n_ch = pack3d["X_train"].shape[2]
    if n_ch != 18:
        log(f"  *** WARNING: {n_ch} channels, expected 18. "
            "SEQ_SLICE / STATIC_SLICE in common.py assume 18. ***")

    meta = {
        "timescale": ts, "lead": lead, "ar_depth": ar_depth,
        "feature_names_2d": feats, "n_features_2d": len(feats),
        "timestep_feature_names_3d": ts_cols, "window_len_3d": int(window_len),
        "train_end_year": C.TRAIN_END_YEAR, "test_start_year": C.TEST_START_YEAR,
        "n_train_2d": int(pack2d["X_train"].shape[0]),
        "n_test_2d":  int(pack2d["X_test"].shape[0]),
        "n_train_3d": int(pack3d["X_train"].shape[0]),
        "n_test_3d":  int(pack3d["X_test"].shape[0]),
        "scaled_columns": clim,
    }
    with open(C.meta_path(ts, lead), "w") as f:
        json.dump(meta, f, indent=2)
    with open(C.scaler_path(ts, lead), "wb") as f:
        pickle.dump({"scaler": scaler, "scaled_columns": clim}, f)

    # persistence RMSE on the test set -- the number the sanity check uses
    obs = pack2d["y_test"]
    per = pack2d["persist_test"]
    rmse_p = float(np.sqrt(np.mean((per - obs) ** 2)))
    log(f"  persistence RMSE on test = {rmse_p:.4f}")

    return dict(timescale=ts, lead=lead,
                n_train_3d=meta["n_train_3d"], n_test_3d=meta["n_test_3d"],
                n_train_2d=meta["n_train_2d"], n_test_2d=meta["n_test_2d"],
                window=window_len, persistence_rmse=rmse_p)


# =============================================================================
# main
# =============================================================================

def main():
    C.ensure_dirs()
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    C.banner("STEP 2  --  DATA PREP (lead-aware)")
    log(f"run: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"split: train targets <= {C.TRAIN_END_YEAR} | test targets >= {C.TEST_START_YEAR}")

    rows = []
    for lead in C.LEADS:
        for ts in C.TIMESCALES:
            rows.append(run_one(ts, lead, log))

    # ---- SANITY CHECK: persistence RMSE must rise with lead ----------------
    # This costs nothing and catches the single most likely error. If a longer
    # lead is not harder for the naive forecast, the target shift is wrong.
    log("\n" + "=" * 74)
    log("SANITY CHECK  --  persistence RMSE must INCREASE with lead")
    log("=" * 74)

    sdf = pd.DataFrame(rows)
    all_ok = True
    for ts in C.TIMESCALES:
        sub = sdf[sdf.timescale == ts].sort_values("lead")
        vals = [round(v, 4) for v in sub["persistence_rmse"].tolist()]
        good = all(a < b for a, b in zip(vals, vals[1:]))
        all_ok &= good
        log(f"  SPEI-{ts:<2} leads {sub['lead'].tolist()} -> persistence RMSE {vals}  "
            + ("OK (increasing)" if good else "*** NOT INCREASING -- CHECK THE SHIFT ***"))

    sdf.to_csv(C.LOG_DIR / "run2_prep_summary.csv", index=False)
    with open(C.LOG_DIR / "run2_data_prep_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    if all_ok:
        log("\nAll checks passed.")
    else:
        log("\n*** STOP. Persistence RMSE did not increase monotonically with lead. ***")
        log("*** Do not train anything until this is resolved. ***")


if __name__ == "__main__":
    main()
