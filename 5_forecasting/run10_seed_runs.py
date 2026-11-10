"""
================================================================================
run10_seed_runs.py  --  Stage 2 seed runs (timing test, Set B, Set C)
================================================================================
Trains every (model, timescale, lead, seed) with the tuned hyperparameters
FIXED (no new Optuna search) and saves, for every run:
    <model>/runs/pred_<model>_SPEI<ts>_lead<L>_seed<s>.csv     test predictions
    <model>/runs/metrics_<model>_SPEI<ts>_lead<L>_seed<s>.json  metrics + timing
    <model>/models/<...>.keras | .json                          lead-1 models (all seeds; for SHAP)

RESUMABLE: a run whose prediction AND metrics files exist is skipped. You can
stop the script (Ctrl+C) at any time and start it again later.

USAGE  (run from this folder, after run1 and run2)
    python run10_seed_runs.py --set timing     # 5 runs + 1 repeat; prints an ETA
    python run10_seed_runs.py --set B          # lead 1: 5 models x 3 timescales x 5 seeds
    python run10_seed_runs.py --set C          # leads 2-3: selected models + XGBoost, 5 seeds
Optional filters:  --models CNN BiLSTM   --timescales 6 12   --jobs 4 (CEEMDAN cores)

SET C picks its models with the rule fixed in advance (p1_selection.py) from the
Set B results, and prints the verdict before training anything.
================================================================================
"""

import argparse
import gc
import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import common as C


# --------------------------------------------------------------------------- log
class Tee:
    """Print to screen and to a log file at the same time."""
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.f.write(msg + "\n"); self.f.flush()


def versions():
    v = dict(python=platform.python_version(), os=platform.platform())
    for mod in ["numpy", "pandas", "tensorflow", "keras", "xgboost", "PyEMD", "sklearn"]:
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "?")
        except Exception:
            v[mod] = "not installed"
    try:
        import tensorflow as tf
        v["devices"] = [d.device_type for d in tf.config.list_physical_devices()]
    except Exception:
        v["devices"] = "?"
    return v


# ---------------------------------------------------------------------- one run
def done(model, ts, lead, seed):
    return (C.run_predictions_path(model, ts, lead, seed).exists()
            and C.run_metrics_path(model, ts, lead, seed).exists())


def train_one(model, ts, lead, seed, n_jobs=None, pred_path=None, metrics_path=None,
              save_model=None):
    """Train + score one run; write its prediction CSV and metrics JSON."""
    if n_jobs is not None:
        os.environ["P1_CEEMDAN_JOBS"] = str(n_jobs)
    if save_model is None:
        save_model = (lead == 1)          # lead-1 models of every seed (small; used by SHAP)
    t0 = time.time()
    if model == "XGBoost":
        import xgb_model
        mpath = C.run_model_path(model, ts, lead, seed, "json") if save_model else None
        row, preds = xgb_model.train_and_evaluate(ts, lead, seed, model_path=mpath)
    else:
        import dl_models
        import tensorflow as tf
        mpath = C.run_model_path(model, ts, lead, seed, "keras") if save_model else None
        row, preds = dl_models.train_and_evaluate(
            model, ts, lead, dict(C.PARAMS_BY_MODEL[model][ts]), seed=seed,
            save_model=save_model, verbose=False, model_path=mpath)
        tf.keras.backend.clear_session()
        gc.collect()
    elapsed = time.time() - t0

    pred_path = pred_path or C.run_predictions_path(model, ts, lead, seed)
    metrics_path = metrics_path or C.run_metrics_path(model, ts, lead, seed)
    C.save_predictions(model, ts, lead, preds["target_date"], preds["division"],
                       preds["obs"], preds["pred"], preds["persist"], path=pred_path)
    row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    row.update(pipeline=C.PIPELINE, elapsed_s=round(elapsed, 1),
               finished=datetime.now().isoformat(timespec="seconds"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(row, indent=1), encoding="utf-8")
    return row


# ------------------------------------------------------------------- run lists
def plan(set_name, log, models=None, timescales=None):
    ts_list = timescales or C.TIMESCALES
    if set_name == "B":
        runs = [(m, ts, 1, s) for m in C.MODELS for ts in ts_list for s in C.SEEDS]
    elif set_name == "C":
        import p1_selection as S
        summ = S.summarise(S.collect_runs(lead=1))
        selected, report, table = S.apply_rule(summ)
        C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(C.TABLE_DIR / "stage2_selection.csv", index=False, float_format="%.6f")
        (C.TABLE_DIR / "stage2_selection.txt").write_text("\n".join(report), encoding="utf-8")
        log("\nSELECTION RULE, FIXED IN ADVANCE (applied to Set B):")
        for line in report:
            log("  " + line)
        runs = []
        for ts in ts_list:
            for lead in (2, 3):
                for s in C.SEEDS:
                    runs.append((selected[ts], ts, lead, s))
                    runs.append(("XGBoost", ts, lead, s))
    else:
        raise ValueError(set_name)
    runs = list(dict.fromkeys(runs))          # no duplicates (e.g. if XGBoost were selected)
    if models:
        runs = [r for r in runs if r[0] in models]
    # XGBoost first (fast), then deep-learning models cell by cell
    runs.sort(key=lambda r: (r[0] != "XGBoost", C.MODELS.index(r[0]), r[1], r[2], r[3]))
    return runs


# ------------------------------------------------------------------ summaries
def write_summaries(log):
    import p1_selection as S
    runs = S.collect_runs()
    if not len(runs):
        return
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(C.TABLE_DIR / "stage2_runs_all.csv", index=False, float_format="%.6f")
    summ = S.summarise(runs)
    summ.to_csv(C.TABLE_DIR / "stage2_summary.csv", index=False, float_format="%.6f")

    log("\nSUMMARY (mean +/- SD across seeds)")
    for _, r in summ.sort_values(["lead", "timescale", "model"]).iterrows():
        log(f"  {r.model:<15} SPEI-{int(r.timescale):<2} lead {int(r.lead)}  "
            f"n={int(r.n_seeds)}  skill {r.skill_mean:+.3f} +/- {r.skill_sd:.3f}  "
            f"NSE {r.NSE_mean:.3f}  CSI {r.CSI_mean:.3f}")

    # seed 42 at lead 1 vs the thesis run of the same model
    rows = []
    for (m, ts), ref in C.THESIS_LEAD1_SEED42.items():
        got = runs[(runs.model == m) & (runs.timescale == ts) & (runs.lead == 1)
                   & (runs.seed == 42)]
        if len(got):
            g = got.iloc[0]
            rows.append(dict(model=m, timescale=ts,
                             **{f"{k}_now": g[k] for k in ["RMSE", "NSE", "Skill"]},
                             **{f"{k}_thesis": ref[k] for k in ["RMSE", "NSE", "Skill"]},
                             max_abs_diff=max(abs(g[k] - ref[k]) for k in ["RMSE", "NSE", "Skill"])))
    if rows:
        pd.DataFrame(rows).to_csv(C.TABLE_DIR / "stage2_seed42_vs_thesis.csv",
                                  index=False, float_format="%.6f")


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["timing", "B", "C"])
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--timescales", nargs="*", type=int)
    ap.add_argument("--jobs", type=int, default=None,
                    help="processes for the one-time CEEMDAN decomposition")
    a = ap.parse_args()

    C.ensure_dirs()
    log = Tee(C.LOG_DIR / f"run10_set{a.set}_{datetime.now():%Y%m%d_%H%M%S}.txt")
    C.banner(f"RUN10  --  set {a.set}  --  pipeline {C.PIPELINE}")
    log(json.dumps(versions()))
    log(f"output root: {C.OUT_ROOT}")
    log(f"batch shuffle by model: {C.SHUFFLE_BY_MODEL}")
    if a.jobs:
        os.environ["P1_CEEMDAN_JOBS"] = str(a.jobs)

    # ---------------- timing test ----------------
    if a.set == "timing":
        times = {}
        for m in ["XGBoost", "CNN", "BiLSTM", "CNN-BiLSTM", "CEEMDAN-BiLSTM"]:
            t0 = time.time()
            if done(m, 12, 1, 42):
                row = json.loads(C.run_metrics_path(m, 12, 1, 42).read_text())
                log(f"  {m:<15} SPEI-12 lead 1 seed 42: already done ({row['elapsed_s']} s)")
            else:
                row = train_one(m, 12, 1, 42)
                log(f"  {m:<15} SPEI-12 lead 1 seed 42: {time.time() - t0:6.1f} s  "
                    f"skill {row['Skill']:+.4f}  NSE {row['NSE']:.4f}")
            times[m] = row["elapsed_s"]
        # determinism: the same CNN run again, into a side folder
        side = C.OUT_ROOT / "_timing"
        side.mkdir(exist_ok=True)
        row2 = train_one("CNN", 12, 1, 42, pred_path=side / "pred_CNN_repeat.csv",
                         metrics_path=side / "metrics_CNN_repeat.json", save_model=False)
        p1 = pd.read_csv(C.run_predictions_path("CNN", 12, 1, 42))["predicted"].values
        p2 = pd.read_csv(side / "pred_CNN_repeat.csv")["predicted"].values
        diff = float(np.max(np.abs(p1 - p2)))
        log(f"\n  DETERMINISM: same seed twice -> max |prediction difference| = {diff:.2e}  "
            + ("OK (identical)" if diff < 1e-5 else "NOT identical"))

        dl = [times[m] for m in ["CNN", "BiLSTM", "CNN-BiLSTM"]]
        per_dl = float(np.mean(dl))
        # CEEMDAN's first run = one-time decomposition + one training. Its training
        # alone is taken as ~2 x the Bi-LSTM (two BiLSTM streams).
        ceem_train = 2.0 * times["BiLSTM"]
        decomp = max(times["CEEMDAN-BiLSTM"] - ceem_train, 0.0)
        log(f"\n  CEEMDAN one-time decomposition (SPEI-12) ~{decomp / 60:.0f} min; "
            f"SPEI-3 and SPEI-6 need one each during Set B")
        est_B = (45 * per_dl + 15 * times["XGBoost"] + 15 * ceem_train
                 + 2 * decomp)                                   # SPEI-12 already cached
        est_C = 30 * per_dl + 30 * times["XGBoost"]
        est_S = 5 * times["BiLSTM"] + 120
        log(f"\n  ESTIMATE (rough): Set B ~{est_B / 3600:.1f} h | Set C ~{est_C / 3600:.1f} h "
            f"| sensitivity ~{est_S / 60:.0f} min")
        log("  (rough: SPEI-3/6 windows are shorter than SPEI-12, so real times are usually lower)")
        log("\nLog file:  " + str(log.f.name))
        return

    # ---------------- Set B / Set C ----------------
    runs = plan(a.set, log, a.models, a.timescales)
    todo = [r for r in runs if not done(*r)]
    log(f"\n{len(runs)} runs in set {a.set}; {len(runs) - len(todo)} already done; "
        f"{len(todo)} to train.")
    elapsed = []
    for i, (m, ts, lead, s) in enumerate(todo, 1):
        t0 = time.time()
        try:
            row = train_one(m, ts, lead, s)
        except KeyboardInterrupt:
            log("\nStopped by you. Start the same command again to continue.")
            sys.exit(0)
        dt = time.time() - t0
        elapsed.append(dt)
        eta = np.mean(elapsed[-10:]) * (len(todo) - i)
        log(f"  [{i}/{len(todo)}] {m:<15} SPEI-{ts:<2} lead {lead} seed {s}: "
            f"skill {row['Skill']:+.4f}  NSE {row['NSE']:.4f}  CSI {row['CSI']:.3f}  "
            f"({dt:.0f} s, ~{eta / 60:.0f} min left)")

    write_summaries(log)
    log(f"\nSet {a.set} finished. Tables in {C.TABLE_DIR}")


if __name__ == "__main__":
    main()
