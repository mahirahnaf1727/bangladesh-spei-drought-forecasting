"""
================================================================================
XGBoost (Optuna-tuned) - Bangladesh SPEI drought forecasting (ERA5 predictors)
================================================================================
Optuna-tuned version of the XGBoost baseline, brought into PARITY with the
LSTM / CNN-LSTM / CEEMDAN-LSTM tuning protocol:

  - 20-trial Optuna per (timescale, variant)   [matches the neural models]
  - MedianPruner (n_startup_trials=5, n_warmup_steps=10)  [matches]
  - Temporal validation split for early stopping (leak-free)  [unchanged]
  - Same 6 metrics, same test period (2019-2024), same 2x2 figure style  [unchanged]
  - FULL (baseline) + WEATHER-ONLY (nonlinearity evidence)  [unchanged]
  - Lag-depth sensitivity (SPEI-6/12, 13 vs 6) preserved, using tuned params

HONEST DESIGN NOTES (state these in the thesis so the comparison is defensible):
  * XGBoost predicts SPEI DIRECTLY (no delta-vs-persistence target) and uses the
    flat 2D feature tables - this is the natural, correct form for a tree model,
    exactly as in the original baseline. The neural models use a delta target on
    3D sequences. This is a deliberate, standard difference: each model family in
    its proper form, compared on identical data splits, metrics and test period.
  * Trees are far more deterministic run-to-run than neural nets, so no separate
    seed-stability study is needed for XGBoost (the LSTM seed study already
    validates the shared tuning protocol as the toughest, noisiest case).

What Optuna searches (sensible ranges for ~3.5k rows, small data -> regularise):
    max_depth, learning_rate, subsample, colsample_bytree,
    min_child_weight, reg_lambda, gamma, reg_alpha
  n_estimators stays high (3000) with early stopping deciding the real count.

Run locally:  python xgboost_optuna.py
--------------------------------------------------------------------------------
EDIT ONLY the CONFIG block (paths/filenames) if they differ.
================================================================================
"""

import os
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================== CONFIG ========================================
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402  (one relative-path config for the whole package)
DATA_DIR = P.WORK / "4_tuning" / "data_prep" / "arrays"
META_DIR = P.WORK / "4_tuning" / "data_prep" / "meta"

FILES = {
    1:  {"npz": "SPEI1_2D.npz",  "meta": "SPEI1_meta.json"},
    3:  {"npz": "SPEI3_2D.npz",  "meta": "SPEI3_meta.json"},
    6:  {"npz": "SPEI6_2D.npz",  "meta": "SPEI6_meta.json",  "alt": "SPEI6_2D_altlag6.npz"},
    12: {"npz": "SPEI12_2D.npz", "meta": "SPEI12_meta.json", "alt": "SPEI12_2D_altlag6.npz"},
}

OUTPUT_DIR = P.WORK / "4_tuning" / "XGBoost" / "output"
MODEL_DIR  = P.WORK / "4_tuning" / "XGBoost" / "models"

TIMESCALES = [1, 3, 6, 12]

# Temporal early-stopping / Optuna validation slice (by DATE, keeps months whole).
VAL_START_YEAR = 2017

CLIMATE_STEMS = {"tp", "mx2t", "mn2t", "ssrd", "tcc", "wind", "d2m"}

# ---- Optuna / training protocol (parity with neural models) ----
SEED                  = 42
N_TRIALS              = 20            # per (timescale, variant)
N_ESTIMATORS_CAP      = 3000         # high cap; early stopping picks the real count
EARLY_STOPPING_ROUNDS = 50
USE_PRUNING           = True
PRUNER_STARTUP        = 5
PRUNER_WARMUP         = 10

# Fixed params not searched (kept from baseline)
FIXED_PARAMS = dict(
    n_estimators=N_ESTIMATORS_CAP,
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
)

# Reference: weather-only LINEAR regression R2 (for the nonlinearity evidence table).
LR_WEATHER_ONLY_R2 = {1: 0.015, 3: 0.313, 6: 0.414, 12: 0.169}

# ============================= METRICS ========================================
def compute_metrics(obs, pred, persist):
    obs     = np.asarray(obs, dtype=float)
    pred    = np.asarray(pred, dtype=float)
    persist = np.asarray(persist, dtype=float)
    err   = pred - obs
    rmse  = float(np.sqrt(np.mean(err ** 2)))
    mae   = float(np.mean(np.abs(err)))
    obar   = obs.mean()
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obar) ** 2)
    nse    = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    wi_denom = np.sum((np.abs(pred - obar) + np.abs(obs - obar)) ** 2)
    wi       = float(1.0 - ss_res / wi_denom) if wi_denom > 0 else np.nan
    if np.std(pred) > 0 and np.std(obs) > 0:
        r  = np.corrcoef(pred, obs)[0, 1]; r2 = float(r ** 2)
    else:
        r2 = np.nan
    mse_model   = np.mean(err ** 2)
    mse_persist = np.mean((persist - obs) ** 2)
    skill = float(1.0 - mse_model / mse_persist) if mse_persist > 0 else np.nan
    return {"RMSE": rmse, "MAE": mae, "NSE": nse, "WI": wi,
            "R2": r2, "Skill_vs_Persist": skill}

# ============================ XGBOOST HELPERS =================================
def _xgb_ver():
    return tuple(int(x) for x in xgb.__version__.split(".")[:2])

_ES_MODE = "constructor" if _xgb_ver() >= (1, 6) else "fit"

def fit_xgb(params, X_tr, y_tr, X_val, y_val):
    """Fit XGBRegressor with temporal early stopping (API-version safe)."""
    if _ES_MODE == "constructor":
        model = xgb.XGBRegressor(**params,
                                 early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                                 eval_metric="rmse")
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                  eval_metric="rmse", verbose=False)
    return model

def _suggest_params(trial):
    """Optuna search space (small-data-friendly regularisation)."""
    p = dict(FIXED_PARAMS)
    p.update(
        max_depth        = trial.suggest_int("max_depth", 3, 6),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 8),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        gamma            = trial.suggest_float("gamma", 0.0, 5.0),
    )
    return p

def tune_xgb(X_fit, y_fit, X_val, y_val, tag):
    """Run Optuna to minimise validation RMSE; return best full param dict + study."""
    def objective(trial):
        params = _suggest_params(trial)
        model = fit_xgb(params, X_fit, y_fit, X_val, y_val)
        pred_val = model.predict(X_val)
        rmse = float(np.sqrt(np.mean((pred_val - y_val) ** 2)))
        # report for pruning: single-shot (XGBoost trains internally), so report once
        if USE_PRUNING:
            trial.report(rmse, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return rmse

    pruner = (optuna.pruners.MedianPruner(n_startup_trials=PRUNER_STARTUP,
                                          n_warmup_steps=PRUNER_WARMUP)
              if USE_PRUNING else optuna.pruners.NopPruner())
    study = optuna.create_study(direction="minimize",
                                sampler=TPESampler(seed=SEED), pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = dict(FIXED_PARAMS); best.update(study.best_params)
    print(f"    [{tag}] best val RMSE={study.best_value:.4f}  params={study.best_params}")
    return best, study

# =============================== IO / SETUP ===================================
def load_arrays(npz_name):
    z = np.load(DATA_DIR / npz_name)
    d = {k: z[k] for k in z.files}
    z.close()
    return d

def load_meta(meta_name):
    with open(META_DIR / meta_name, "r", encoding="utf-8") as f:
        return json.load(f)

def weather_col_indices(feature_names):
    return [i for i, n in enumerate(feature_names)
            if ("_lag" in n) and (n.split("_lag")[0] in CLIMATE_STEMS)]

def div_col_indices(feature_names):
    pairs = sorted((n, i) for i, n in enumerate(feature_names) if n.startswith("div"))
    return [n for n, _ in pairs], [i for _, i in pairs]

def recover_divisions(X, feature_names):
    names, idx = div_col_indices(feature_names)
    if not idx:
        return np.array(["NA"] * X.shape[0])
    a = np.argmax(X[:, idx], axis=1)
    return np.array([names[j] for j in a])

# ================================ PLOTTING ====================================
def scatter_2x2(results, title, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, ts in zip(axes.ravel(), TIMESCALES):
        r = results[ts]
        obs, pred, m = r["obs"], r["pred"], r["metrics"]
        lo = float(min(obs.min(), pred.min()))
        hi = float(max(obs.max(), pred.max()))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        lo, hi = lo - pad, hi + pad
        ax.scatter(obs, pred, s=12, alpha=0.5, edgecolor="none", label="test points")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="1:1 line")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed SPEI"); ax.set_ylabel("Predicted SPEI")
        ax.set_title(f"SPEI-{ts}")
        box = (f"RMSE = {m['RMSE']:.3f}\nNSE  = {m['NSE']:.3f}\n"
               f"R2   = {m['R2']:.3f}\nSkill= {m['Skill_vs_Persist']:.3f}")
        ax.text(0.04, 0.96, box, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# ================================= MAIN =======================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"XGBoost {xgb.__version__} | Optuna {optuna.__version__} | "
          f"ES mode: {_ES_MODE} | trials/model: {N_TRIALS}")
    print("-" * 70)

    full_results, weather_results = {}, {}
    full_rows, weather_rows, param_rows = [], [], []

    for ts in TIMESCALES:
        f = FILES[ts]
        data = load_arrays(f["npz"])
        meta = load_meta(f["meta"])
        feat = meta["feature_names_2d"]

        X_train, y_train = data["X_train"], data["y_train"]
        X_test,  y_test  = data["X_test"],  data["y_test"]
        persist_test     = data["persist_test"]
        tdate_train      = [str(x) for x in data["target_date_train"]]
        tdate_test       = [str(x) for x in data["target_date_test"]]

        assert X_train.shape[1] == len(feat), \
            f"SPEI-{ts}: X_train has {X_train.shape[1]} cols but meta lists {len(feat)}"
        w_idx = weather_col_indices(feat)
        if len(w_idx) != 21:
            print(f"  [WARN] SPEI-{ts}: found {len(w_idx)} climate cols (expected 21).")

        yr_tr   = np.array([int(d[:4]) for d in tdate_train])
        val_msk = yr_tr >= VAL_START_YEAR
        fit_msk = ~val_msk
        if val_msk.sum() == 0 or fit_msk.sum() == 0:
            raise RuntimeError(f"SPEI-{ts}: empty fit/val split (VAL_START_YEAR={VAL_START_YEAR}).")

        print(f"SPEI-{ts}: features={len(feat)}  fit={fit_msk.sum()}  "
              f"val={val_msk.sum()}  test={len(y_test)}")

        divisions = recover_divisions(X_test, feat)

        # ---------------- XGBoost-FULL (tuned) ----------------
        best_full, _ = tune_xgb(X_train[fit_msk], y_train[fit_msk],
                                X_train[val_msk], y_train[val_msk], tag=f"FULL SPEI-{ts}")
        # retrain best config on fit, early-stop on val (same protocol as tuning)
        m_full = fit_xgb(best_full, X_train[fit_msk], y_train[fit_msk],
                         X_train[val_msk], y_train[val_msk])
        pred_full = m_full.predict(X_test)
        met_full  = compute_metrics(y_test, pred_full, persist_test)
        best_it_full = getattr(m_full, "best_iteration", None)

        m_full.save_model(str(MODEL_DIR / f"xgb_FULL_spei{ts}.json"))
        _save_predictions(OUTPUT_DIR / f"predictions_FULL_spei{ts}.csv",
                          tdate_test, divisions, y_test, pred_full, persist_test)

        full_results[ts] = {"obs": np.asarray(y_test, float),
                            "pred": np.asarray(pred_full, float),
                            "metrics": met_full}
        full_rows.append(_row(ts, met_full, best_it_full))
        param_rows.append(_param_row(ts, "FULL", best_full, best_it_full))

        # ---------------- XGBoost-WEATHER-ONLY (tuned) ----------------
        Xw_fit  = X_train[fit_msk][:, w_idx]
        Xw_val  = X_train[val_msk][:, w_idx]
        Xw_test = X_test[:, w_idx]

        best_w, _ = tune_xgb(Xw_fit, y_train[fit_msk], Xw_val, y_train[val_msk],
                             tag=f"WEATHERONLY SPEI-{ts}")
        m_w = fit_xgb(best_w, Xw_fit, y_train[fit_msk], Xw_val, y_train[val_msk])
        pred_w = m_w.predict(Xw_test)
        met_w  = compute_metrics(y_test, pred_w, persist_test)
        best_it_w = getattr(m_w, "best_iteration", None)

        m_w.save_model(str(MODEL_DIR / f"xgb_WEATHERONLY_spei{ts}.json"))
        _save_predictions(OUTPUT_DIR / f"predictions_WEATHERONLY_spei{ts}.csv",
                          tdate_test, divisions, y_test, pred_w, persist_test)

        weather_results[ts] = {"obs": np.asarray(y_test, float),
                               "pred": np.asarray(pred_w, float),
                               "metrics": met_w}
        row = _row(ts, met_w, best_it_w)
        row["LR_weatheronly_R2"] = LR_WEATHER_ONLY_R2.get(ts, np.nan)
        row["dR2_xgb_minus_lr"]  = met_w["R2"] - LR_WEATHER_ONLY_R2.get(ts, np.nan)
        weather_rows.append(row)
        param_rows.append(_param_row(ts, "WEATHERONLY", best_w, best_it_w))

        print(f"   FULL:          RMSE={met_full['RMSE']:.3f}  NSE={met_full['NSE']:.3f}"
              f"  R2={met_full['R2']:.3f}  Skill={met_full['Skill_vs_Persist']:.3f}"
              f"  (best_iter={best_it_full})")
        print(f"   WEATHER-ONLY:  R2={met_w['R2']:.3f}  (LR R2={LR_WEATHER_ONLY_R2.get(ts):.3f}"
              f"  -> dR2={met_w['R2']-LR_WEATHER_ONLY_R2.get(ts):+.3f})")

    # ------------------------ tables ------------------------
    _write_table(OUTPUT_DIR / "metrics_FULL.csv", full_rows,
                 ["timescale", "RMSE", "MAE", "NSE", "WI", "R2",
                  "Skill_vs_Persist", "best_iteration"])
    _write_table(OUTPUT_DIR / "metrics_WEATHERONLY.csv", weather_rows,
                 ["timescale", "RMSE", "MAE", "NSE", "WI", "R2",
                  "Skill_vs_Persist", "best_iteration",
                  "LR_weatheronly_R2", "dR2_xgb_minus_lr"])
    _write_table(OUTPUT_DIR / "best_params.csv", param_rows,
                 ["timescale", "variant", "max_depth", "learning_rate", "subsample",
                  "colsample_bytree", "min_child_weight", "reg_lambda", "reg_alpha",
                  "gamma", "best_iteration"])

    # ------------------------ figures ------------------------
    scatter_2x2(full_results,
                "XGBoost-FULL (Optuna-tuned): predicted vs observed",
                OUTPUT_DIR / "scatter_FULL_2x2.png")
    scatter_2x2(weather_results,
                "XGBoost WEATHER-ONLY (Optuna-tuned): predicted vs observed",
                OUTPUT_DIR / "scatter_WEATHERONLY_2x2.png")

    # ------------------ lag-depth sensitivity (SPEI-6/12) ------------------
    _lag_depth_sensitivity(full_results)

    print("-" * 70)
    print("Done.")
    print(f"  Outputs -> {OUTPUT_DIR}")
    print(f"  Models  -> {MODEL_DIR}")

# --------------------------- lag-depth test -----------------------------------
def _lag_depth_sensitivity(full_results):
    rows = []
    for ts in (6, 12):
        alt = FILES[ts].get("alt")
        if alt is None or not (DATA_DIR / alt).exists():
            print(f"[lag-depth] SPEI-{ts}: alt file '{alt}' not found -> skipped.")
            continue
        data = load_arrays(alt)
        X_train, y_train = data["X_train"], data["y_train"]
        X_test,  y_test  = data["X_test"],  data["y_test"]
        persist_test     = data["persist_test"]
        tdate_train      = [str(x) for x in data["target_date_train"]]

        yr_tr   = np.array([int(d[:4]) for d in tdate_train])
        val_msk = yr_tr >= VAL_START_YEAR
        fit_msk = ~val_msk

        # tune on the alt-lag features too (fair: same protocol)
        best_alt, _ = tune_xgb(X_train[fit_msk], y_train[fit_msk],
                               X_train[val_msk], y_train[val_msk], tag=f"FULL(alt6) SPEI-{ts}")
        m = fit_xgb(best_alt, X_train[fit_msk], y_train[fit_msk],
                    X_train[val_msk], y_train[val_msk])
        met6 = compute_metrics(y_test, m.predict(X_test), persist_test)
        m.save_model(str(MODEL_DIR / f"xgb_FULL_spei{ts}_altlag6.json"))

        met13 = full_results[ts]["metrics"]
        rows.append({
            "timescale": ts,
            "RMSE_lag13": met13["RMSE"], "RMSE_lag6": met6["RMSE"],
            "NSE_lag13":  met13["NSE"],  "NSE_lag6":  met6["NSE"],
            "R2_lag13":   met13["R2"],   "R2_lag6":   met6["R2"],
            "Skill_lag13": met13["Skill_vs_Persist"],
            "Skill_lag6":  met6["Skill_vs_Persist"],
        })
        print(f"[lag-depth] SPEI-{ts}: Skill lag13={met13['Skill_vs_Persist']:.3f} "
              f"vs lag6={met6['Skill_vs_Persist']:.3f}")

    if rows:
        cols = ["timescale", "RMSE_lag13", "RMSE_lag6", "NSE_lag13", "NSE_lag6",
                "R2_lag13", "R2_lag6", "Skill_lag13", "Skill_lag6"]
        _write_table(OUTPUT_DIR / "lag_depth_sensitivity.csv", rows, cols)

# ------------------------------ small utils -----------------------------------
def _row(ts, m, best_it):
    return {"timescale": ts, **m, "best_iteration": best_it}

def _param_row(ts, variant, params, best_it):
    return {"timescale": ts, "variant": variant,
            "max_depth": params.get("max_depth"),
            "learning_rate": params.get("learning_rate"),
            "subsample": params.get("subsample"),
            "colsample_bytree": params.get("colsample_bytree"),
            "min_child_weight": params.get("min_child_weight"),
            "reg_lambda": params.get("reg_lambda"),
            "reg_alpha": params.get("reg_alpha"),
            "gamma": params.get("gamma"),
            "best_iteration": best_it}

def _save_predictions(path, dates, divs, obs, pred, persist):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["target_date", "division", "observed", "predicted", "persistence"])
        for d, dv, o, p, ps in zip(dates, divs, obs, pred, persist):
            w.writerow([d, dv, f"{o:.6f}", f"{p:.6f}", f"{ps:.6f}"])

def _write_table(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([("" if r.get(c) is None else
                         (f"{r[c]:.4f}" if isinstance(r.get(c), float) else r.get(c)))
                        for c in cols])

# ==============================================================================
if __name__ == "__main__":
    main()
