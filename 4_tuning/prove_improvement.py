"""
================================================================================
LSTM - SEED-STABILITY PROOF  (baseline vs Optuna-tuned, FULL variant)
================================================================================
PURPOSE
  Show the Optuna improvement is REAL, not a lucky single run. Each model is
  retrained across several random seeds; we report mean +/- std per timescale.
  If the tuned model beats the baseline across ALL seeds, the gain is robust.

WHAT THIS SCRIPT DOES
  - Trains 2 configs (BASELINE fixed params  vs  TUNED Optuna params),
    each across N_SEEDS seeds, for all 4 timescales, FULL variant only.
  - No Optuna search here (tuning is already done). Just the final retrain,
    repeated per seed. => (2 configs x N_SEEDS x 4 timescales) trainings.
  - Applies the SPEED FIX: recurrent_dropout removed so the fast LSTM kernel
    is used. BOTH configs use the identical architecture -> fair comparison.

OUTPUTS (to OUTPUT_DIR)
  - seed_stability_FULL.csv   : mean +/- std of RMSE/NSE/R2/Skill per timescale,
                                baseline vs tuned  (THIS is your defense table)
  - seed_runs_raw_FULL.csv    : every individual run (for the appendix)
  - seed_stability_FULL.png   : bar chart with error bars (optional visual)

HOW TO USE
  1. SMOKE TEST FIRST: set SMOKE_TEST = True, run once. It trains ONE timescale,
     ONE seed, one config, and prints the wall-clock time. Multiply that by
     (2 x N_SEEDS x 4) to estimate the full run. Confirm it works + time it.
  2. Then set SMOKE_TEST = False and run the full thing (ideally overnight).

  Everything else (data layout, delta-target, metrics, val split) is copied
  verbatim from your try_3 / try_5 scripts so results stay comparable.
================================================================================
"""

# ---- determinism: set BEFORE importing tensorflow ----
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import json
import csv
import time
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================== CONFIG ========================================
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402  (one relative-path config for the whole package)
ARRAY_DIR  = P.WORK / "4_tuning" / "data_prep" / "arrays"
META_DIR   = P.WORK / "4_tuning" / "data_prep" / "meta"
OUTPUT_DIR = P.WORK / "4_tuning" / "BiLSTM_tuning_check" / "output"
MODEL_DIR  = P.WORK / "4_tuning" / "BiLSTM_tuning_check" / "models"

TIMESCALES = [1, 3, 6, 12]
NPZ  = {ts: f"SPEI{ts}_3D.npz"    for ts in TIMESCALES}
META = {ts: f"SPEI{ts}_meta.json" for ts in TIMESCALES}

WINDOW_OVERRIDE = {6: 6}

SEQ_SLICE     = slice(0, 10)
STATIC_SLICE  = slice(10, 18)
WEATHER_SLICE = slice(0, 7)

VAL_START_YEAR = 2014

# ---- training controls (matched to your final-retrain settings) ----
BATCH      = 32
EPOCHS_CAP = 200
PATIENCE   = 40          # matches try_3 PATIENCE and try_5 PATIENCE_FINAL

# ---- seeds for the stability check ----
SEEDS = [42, 43, 44]     # 3 seeds; add more if you want a tighter estimate

# ---- SMOKE TEST switch: True = one quick run to test + time it ----
SMOKE_TEST = False       # <-- set False for the full run

# ============================== PARAMS ========================================
# BASELINE = your try_3 fixed config (SAME for all timescales)
BASELINE_PARAMS = {ts: {"lstm_units": 48, "dense_units": 16, "dropout": 0.2, "lr": 1e-3}
                   for ts in TIMESCALES}

# TUNED = your try_5 Optuna best params, FULL variant (per timescale)
TUNED_PARAMS = {
    1:  {"lstm_units": 96,  "dense_units": 24, "dropout": 0.3913, "lr": 0.002918},
    3:  {"lstm_units": 16,  "dense_units": 48, "dropout": 0.4290, "lr": 0.003810},
    6:  {"lstm_units": 16,  "dense_units": 64, "dropout": 0.4948, "lr": 0.002713},
    12: {"lstm_units": 128, "dense_units": 32, "dropout": 0.4878, "lr": 0.001904},
}

CONFIGS = {"baseline": BASELINE_PARAMS, "tuned": TUNED_PARAMS}

# ============================= METRICS ========================================
def compute_metrics(obs, pred, persist):
    obs = np.asarray(obs, float); pred = np.asarray(pred, float); persist = np.asarray(persist, float)
    err = pred - obs
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    obar = obs.mean()
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obar) ** 2)
    nse = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    wi_den = np.sum((np.abs(pred - obar) + np.abs(obs - obar)) ** 2)
    wi  = float(1 - ss_res / wi_den) if wi_den > 0 else np.nan
    r2  = float(np.corrcoef(pred, obs)[0, 1] ** 2) if (pred.std() > 0 and obs.std() > 0) else np.nan
    mse_m = np.mean(err ** 2)
    mse_p = np.mean((persist - obs) ** 2)
    skill = float(1 - mse_m / mse_p) if mse_p > 0 else np.nan
    return {"RMSE": rmse, "MAE": mae, "NSE": nse, "WI": wi, "R2": r2, "Skill_vs_Persist": skill}

# ============================= HELPERS ========================================
def load_npz(ts):
    z = np.load(ARRAY_DIR / NPZ[ts], allow_pickle=True)
    d = {k: z[k] for k in z.files}
    z.close()
    return d

def temporal_val_mask(target_date_train):
    yrs = np.array([int(str(d)[:4]) for d in target_date_train])
    return yrs >= VAL_START_YEAR

def build_model(T, lstm_units, dense_units, dropout, lr):
    """FULL-variant BiLSTM. SPEED FIX: no recurrent_dropout -> fast kernel."""
    seq_in = Input(shape=(T, 10), name="seq")
    x = Bidirectional(LSTM(lstm_units, dropout=dropout), name="bilstm")(seq_in)   # <-- fix
    st_in = Input(shape=(8,), name="static")
    x = Concatenate()([x, st_in])
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout)(x)
    out = Dense(1)(x)
    m = Model([seq_in, st_in], out)
    m.compile(optimizer=Adam(lr), loss=tf.keras.losses.Huber(delta=1.0))
    return m

def make_inputs(X):
    return [X[:, :, SEQ_SLICE], X[:, 0, STATIC_SLICE]]

# ============================= ONE TRAINING ===================================
def train_one(ts, params, seed):
    """Train FULL model once for (timescale, params, seed); return test metrics + model."""
    # seed everything for THIS run
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    d = load_npz(ts)
    Xtr, ytr, ptr, dtr = d["X_train"], d["y_train"], d["persist_train"], d["target_date_train"]
    Xte, yte, pte      = d["X_test"],  d["y_test"],  d["persist_test"]

    ytr_delta = ytr - ptr

    if ts in WINDOW_OVERRIDE:
        w = WINDOW_OVERRIDE[ts]
        Xtr, Xte = Xtr[:, -w:, :], Xte[:, -w:, :]

    T = Xtr.shape[1]
    val_mask = temporal_val_mask(dtr)
    fit_mask = ~val_mask

    Xin = make_inputs(Xtr)
    Xfit = [Xin[0][fit_mask], Xin[1][fit_mask]]
    Xval = [Xin[0][val_mask], Xin[1][val_mask]]
    yfit, yval = ytr_delta[fit_mask], ytr_delta[val_mask]

    model = build_model(T, params["lstm_units"], params["dense_units"],
                        params["dropout"], params["lr"])
    es  = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-5, verbose=0)
    model.fit(Xfit, yfit, validation_data=(Xval, yval),
              epochs=EPOCHS_CAP, batch_size=BATCH, verbose=0, callbacks=[es, rlr])

    pred_delta = model.predict(make_inputs(Xte), verbose=0).ravel()
    final_pred = pred_delta + pte
    met = compute_metrics(yte, final_pred, pte)
    return met, model

# ============================= AGGREGATE + SAVE ===============================
METRICS_TO_REPORT = ["RMSE", "NSE", "R2", "Skill_vs_Persist"]

def summarise_and_save(raw_rows):
    """raw_rows: list of dicts {config, timescale, seed, <metrics>}. Writes CSVs + PNG."""
    # raw dump (appendix)
    raw_path = OUTPUT_DIR / "seed_runs_raw_FULL.csv"
    with open(raw_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "timescale", "seed"] + METRICS_TO_REPORT)
        for r in raw_rows:
            w.writerow([r["config"], r["timescale"], r["seed"]]
                       + [f"{r[m]:.4f}" for m in METRICS_TO_REPORT])

    # mean +/- std summary (the defense table)
    sum_path = OUTPUT_DIR / "seed_stability_FULL.csv"
    summary = {}  # (config, ts) -> {metric: (mean, std)}
    with open(sum_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["config", "timescale"]
        for m in METRICS_TO_REPORT:
            header += [f"{m}_mean", f"{m}_std", f"{m}_mean_pm_std"]
        w.writerow(header)
        for cfg in CONFIGS:
            for ts in TIMESCALES:
                vals = {m: np.array([r[m] for r in raw_rows
                                     if r["config"] == cfg and r["timescale"] == ts])
                        for m in METRICS_TO_REPORT}
                summary[(cfg, ts)] = {}
                row = [cfg, ts]
                for m in METRICS_TO_REPORT:
                    mu, sd = float(np.mean(vals[m])), float(np.std(vals[m]))
                    summary[(cfg, ts)][m] = (mu, sd)
                    row += [f"{mu:.4f}", f"{sd:.4f}", f"{mu:.3f} +/- {sd:.3f}"]
                w.writerow(row)

    # bar chart with error bars (NSE + Skill; the two headline metrics)
    if not SMOKE_TEST:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, metric in zip(axes, ["NSE", "Skill_vs_Persist"]):
            x = np.arange(len(TIMESCALES)); width = 0.38
            for i, cfg in enumerate(["baseline", "tuned"]):
                means = [summary[(cfg, ts)][metric][0] for ts in TIMESCALES]
                stds  = [summary[(cfg, ts)][metric][1] for ts in TIMESCALES]
                ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=4,
                       label=cfg)
            ax.set_xticks(x); ax.set_xticklabels([f"SPEI-{t}" for t in TIMESCALES])
            ax.set_ylabel(metric); ax.set_title(f"{metric}: baseline vs tuned ({len(SEEDS)} seeds)")
            ax.legend(); ax.axhline(0, color="0.7", lw=0.8)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "seed_stability_FULL.png", dpi=150)
        plt.close(fig)

    return sum_path

# =============================== MAIN =========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"TensorFlow {tf.__version__}  |  "
          f"devices: {[d.device_type for d in tf.config.list_physical_devices()]}")

    if SMOKE_TEST:
        print("\n*** SMOKE TEST: 1 config x 1 seed x 1 timescale (SPEI-12) ***")
        t0 = time.time()
        met, _ = train_one(12, TUNED_PARAMS[12], SEEDS[0])
        dt = time.time() - t0
        n_full = 2 * len(SEEDS) * len(TIMESCALES)
        print(f"  one run took {dt:.1f}s -> full run ~ {dt * n_full / 60:.1f} min "
              f"({n_full} trainings)")
        print(f"  sample metrics: RMSE={met['RMSE']:.3f}  NSE={met['NSE']:.3f}  "
              f"R2={met['R2']:.3f}  Skill={met['Skill_vs_Persist']:+.3f}")
        print("  Looks good? Set SMOKE_TEST = False and run again for the real thing.")
        return

    raw_rows = []
    total = 2 * len(SEEDS) * len(TIMESCALES); done = 0
    t0 = time.time()
    for cfg, param_map in CONFIGS.items():
        for ts in TIMESCALES:
            for seed in SEEDS:
                met, model = train_one(ts, param_map[ts], seed)
                model.save(MODEL_DIR / f"lstm_FULL_{cfg}_spei{ts}_seed{seed}.keras")
                raw_rows.append({"config": cfg, "timescale": ts, "seed": seed, **met})
                done += 1
                print(f"  [{done:>2}/{total}] {cfg:<8} SPEI-{ts:<2} seed{seed}  "
                      f"NSE={met['NSE']:.3f}  Skill={met['Skill_vs_Persist']:+.3f}")

    sum_path = summarise_and_save(raw_rows)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")
    print(f"  Defense table -> {sum_path}")
    print(f"  Raw runs      -> {OUTPUT_DIR / 'seed_runs_raw_FULL.csv'}")
    print(f"  Bar chart     -> {OUTPUT_DIR / 'seed_stability_FULL.png'}")
    print(f"  Models        -> {MODEL_DIR}")

if __name__ == "__main__":
    main()
