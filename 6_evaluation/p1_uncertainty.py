"""
================================================================================
p1_uncertainty.py  --  F4: confidence intervals and Diebold-Mariano tests
================================================================================
Reproduces stage2_bootstrap_ci.csv, stage2_bootstrap_diff.csv and
stage2_dm_tests.csv from the 135 run prediction files (checked 7 Nov 2026).

1) CIRCULAR BLOCK BOOTSTRAP (Politis & Romano, 1992)
   The 576 test pairs are 72 months x 8 divisions and are correlated in time and
   across divisions, so plain resampling would make intervals too narrow.
   * resample MONTHS in blocks of 12 consecutive months; a block may start at
     any month and wraps around the end of the test period (circular), keeping
     all 8 divisions of a month together;
   * 2,000 resamples; the SAME resampled months are used for every model and
     for persistence (paired), so differences between them get intervals too;
   * for each model the metric is computed for every seed on every resample and
     the values are pooled, so the 95% interval covers sampling AND seed
     uncertainty.
   Metrics: RMSE, NSE, skill vs persistence, POD, CSI; plus the paired
   differences (model - persistence) for CSI and POD, and (model - XGBoost) for skill.

2) DIEBOLD-MARIANO TEST (model vs persistence, squared-error loss)
   Loss differential per month = mean over divisions of (e_model^2 - e_persist^2),
   giving 72 values. Long-run variance by Newey-West (bandwidth max(h-1, 3));
   Harvey-Leybourne-Newbold small-sample correction; two-sided p from t(T-1).
   Reported for each seed and for the seed-ensemble mean forecast.

OUTPUT (OUT_ROOT/_tables/)
   stage2_bootstrap_ci.csv, stage2_bootstrap_diff.csv, stage2_dm_tests.csv
================================================================================
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "5_forecasting"))
import common as C  # noqa: E402

B = 2000
BLOCK = 12
THR = C.DROUGHT_THRESHOLD
RNG_SEED = 20261015


# ------------------------------------------------------------------- loading
def load_predictions(root=None):
    """{(model, ts, lead): {seed: DataFrame sorted by (target_date, division)}}"""
    root = Path(root) if root else C.OUT_ROOT
    out = {}
    pat = re.compile(r"pred_(.+)_SPEI(\d+)_lead(\d)_seed(\d+)\.csv")
    for f in root.glob("*/runs/pred_*.csv"):
        m = pat.match(f.name)
        if not m:
            continue
        model, ts, lead, seed = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        df = pd.read_csv(f).sort_values(["target_date", "division"]).reset_index(drop=True)
        out.setdefault((model, ts, lead), {})[seed] = df
    return out


# ------------------------------------------------------------------- metrics
def metric_set(o, p, q):
    """o, p, q: (n_rep, n) arrays of observed, predicted, persistence."""
    e2 = (p - o) ** 2
    mse = e2.mean(axis=1)
    mse_p = ((q - o) ** 2).mean(axis=1)
    var = ((o - o.mean(axis=1, keepdims=True)) ** 2).mean(axis=1)
    oe, pe = o <= THR, p <= THR
    h = (oe & pe).sum(axis=1); mi = (oe & ~pe).sum(axis=1); fa = (~oe & pe).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pod = np.where(h + mi > 0, h / (h + mi), np.nan)
        csi = np.where(h + mi + fa > 0, h / (h + mi + fa), np.nan)
    return dict(RMSE=np.sqrt(mse), NSE=1 - mse / var, Skill=1 - mse / mse_p, POD=pod, CSI=csi)


def block_indices(n_months, rng):
    """Row index arrays (over months) for B circular-block resamples."""
    starts = rng.integers(0, n_months, size=(B, int(np.ceil(n_months / BLOCK))))
    idx = ((starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % n_months).reshape(B, -1)
    return idx[:, :n_months]


def expand(month_idx, n_div):
    """Month indices -> row indices (all divisions of each month)."""
    return (month_idx[:, :, None] * n_div + np.arange(n_div)[None, None, :]).reshape(
        month_idx.shape[0], -1)


# ------------------------------------------------------------------ bootstrap
def bootstrap(preds):
    rng = np.random.default_rng(RNG_SEED)
    ci_rows, diff_rows = [], []
    by_ts_lead = {}
    for (model, ts, lead), seeds in preds.items():
        by_ts_lead.setdefault((ts, lead), {})[model] = seeds

    for (ts, lead), models in sorted(by_ts_lead.items()):
        ref = next(iter(next(iter(models.values())).values()))
        n_div = ref["division"].nunique()
        n_months = len(ref) // n_div
        rows = expand(block_indices(n_months, rng), n_div)          # (B, 576)
        o = ref["observed"].values; q = ref["persistence"].values
        O, Q = o[rows], q[rows]

        pers = metric_set(O, Q, Q)
        full_p = metric_set(o[None], q[None], q[None])
        for k in ["RMSE", "NSE", "POD", "CSI"]:
            ci_rows.append(dict(model="Persistence", timescale=ts, lead=lead, metric=k,
                                point=float(full_p[k][0]),
                                lo=np.nanpercentile(pers[k], 2.5), hi=np.nanpercentile(pers[k], 97.5),
                                n_seeds=0))

        boot_by_model, point_by_model = {}, {}
        for model, seeds in models.items():
            pooled = {k: [] for k in ["RMSE", "NSE", "Skill", "POD", "CSI"]}
            points = {k: [] for k in pooled}
            for s, df in seeds.items():
                assert np.allclose(df["observed"].values, o), "observed series differ"
                p = df["predicted"].values
                ms = metric_set(O, p[rows], Q)
                fp = metric_set(o[None], p[None], q[None])
                for k in pooled:
                    pooled[k].append(ms[k]); points[k].append(fp[k][0])
            boot_by_model[model] = {k: np.stack(v) for k, v in pooled.items()}   # (n_seeds, B)
            point_by_model[model] = {k: float(np.mean(v)) for k, v in points.items()}
            for k in pooled:
                arr = boot_by_model[model][k].ravel()
                ci_rows.append(dict(model=model, timescale=ts, lead=lead, metric=k,
                                    point=float(np.mean(points[k])),
                                    lo=np.nanpercentile(arr, 2.5), hi=np.nanpercentile(arr, 97.5),
                                    n_seeds=len(seeds)))
            for k in ["CSI", "POD"]:
                d = (boot_by_model[model][k] - pers[k][None, :]).ravel()
                diff_rows.append(dict(model=model, vs="Persistence", timescale=ts, lead=lead,
                                      metric=k, point=point_by_model[model][k] - float(full_p[k][0]),
                                      mean=np.nanmean(d),
                                      lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5),
                                      share_above_0=float(np.nanmean(d > 0))))
        if "XGBoost" in boot_by_model:
            xg = boot_by_model["XGBoost"]["Skill"].mean(axis=0)              # seed-mean per resample
            for model, bm in boot_by_model.items():
                if model == "XGBoost":
                    continue
                d = (bm["Skill"] - xg[None, :]).ravel()
                diff_rows.append(dict(model=model, vs="XGBoost", timescale=ts, lead=lead,
                                      metric="Skill",
                                      point=point_by_model[model]["Skill"] - point_by_model["XGBoost"]["Skill"],
                                      mean=np.nanmean(d),
                                      lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5),
                                      share_above_0=float(np.nanmean(d > 0))))
    return pd.DataFrame(ci_rows), pd.DataFrame(diff_rows)


# ------------------------------------------------------------- Diebold-Mariano
def dm_test(o, p, q, n_div, h):
    d = ((p - o) ** 2 - (q - o) ** 2).reshape(-1, n_div).mean(axis=1)   # per month
    T = len(d)
    dbar = d.mean()
    bw = max(h - 1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    dc = d - dbar
    lrv = dc @ dc / T
    for k in range(1, bw + 1):
        w = 1 - k / (bw + 1)
        lrv += 2 * w * (dc[k:] @ dc[:-k]) / T
    dm = dbar / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return float(stat), float(p), float(dbar)


def dm_all(preds):
    rows = []
    for (model, ts, lead), seeds in sorted(preds.items()):
        ref = next(iter(seeds.values()))
        n_div = ref["division"].nunique()
        o, q = ref["observed"].values, ref["persistence"].values
        for s, df in sorted(seeds.items()):
            stat, p, dbar = dm_test(o, df["predicted"].values, q, n_div, lead)
            rows.append(dict(model=model, timescale=ts, lead=lead, forecast=f"seed {s}",
                             dm_stat=stat, p_value=p, mean_loss_diff=dbar))
        ens = np.mean([df["predicted"].values for df in seeds.values()], axis=0)
        stat, p, dbar = dm_test(o, ens, q, n_div, lead)
        rows.append(dict(model=model, timescale=ts, lead=lead, forecast="seed-ensemble mean",
                         dm_stat=stat, p_value=p, mean_loss_diff=dbar))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="Stage 2 output folder (default: p1_config)")
    ap.add_argument("--out", default=None, help="folder for the three tables (default: <root>/_tables)")
    a = ap.parse_args()
    preds = load_predictions(a.root)
    if not preds:
        raise SystemExit("No run predictions found.")
    root = Path(a.root) if a.root else C.OUT_ROOT
    out = Path(a.out) if a.out else root / "_tables"
    out.mkdir(parents=True, exist_ok=True)
    ci, diff = bootstrap(preds)
    dm = dm_all(preds)
    ci.to_csv(out / "stage2_bootstrap_ci.csv", index=False, float_format="%.4f")
    diff.to_csv(out / "stage2_bootstrap_diff.csv", index=False, float_format="%.4f")
    dm.to_csv(out / "stage2_dm_tests.csv", index=False, float_format="%.4f")
    print(f"{len(preds)} model x timescale x lead cells; tables written to {out}")


if __name__ == "__main__":
    main()
