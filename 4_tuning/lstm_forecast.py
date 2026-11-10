"""
================================================================================
LSTM - Bangladesh SPEI drought forecasting (ERA5 predictors, 1-month lead)
================================================================================
TRY_5: Optuna tuning, SPEED-OPTIMIZED (target: ~5 h -> under ~45 min).

Speed changes vs try_4 (only these; everything else identical):
  - N_TRIALS 50 -> 20.
  - Dual patience: PATIENCE_TUNING=15 during trial search (fail fast),
    PATIENCE_FINAL=40 when retraining the final model on best_params.
  - Optuna MedianPruner(n_startup_trials=5, n_warmup_steps=10) + a per-epoch
    TFKerasPruningCallback so hopeless trials stop mid-training.

Search space (minimise val_loss on 2014-2018 validation years):
  LSTM_UNITS 16..128 step16 | DENSE_UNITS 8..64 step8 | DROPOUT 0.1..0.5 | LR 1e-4..1e-2 log

Leakage note: tuner only ever sees train/val years (<=2018); 2019-2024 test set
untouched until the final model is scored.

Comparison note: tuned per timescale/variant -> NOT config-identical to
CNN-LSTM/CEEMD-LSTM unless those are tuned with the SAME budget + space.

Preserved (unchanged): delta-target + reconstruction, metrics
(RMSE/MAE/NSE/WI/R2/Skill), static-vs-sequence input splitting, plotting, CSV/PNG/keras saving.

Run locally:  python lstm_forecast.py
--------------------------------------------------------------------------------
EDIT ONLY the CONFIG block if paths/filenames differ.
================================================================================
"""

# ---- determinism: set BEFORE importing tensorflow ----
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # steadier CPU numerics

import json
import random
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass  # older TF; seeds still set

from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # quiet per-trial spam

# --- Optuna Keras pruning callback moved to `optuna-integration` in recent versions.
#     Import defensively so the run never dies over a package-location change. ---
_PRUNING_CB = None
try:
    from optuna.integration import TFKerasPruningCallback as _PRUNING_CB
except Exception:
    try:
        from optuna_integration import TFKerasPruningCallback as _PRUNING_CB
    except Exception:
        _PRUNING_CB = None  # mid-training pruning unavailable; study-level pruning still applies
_HAS_PRUNING_CB = _PRUNING_CB is not None

# ============================== CONFIG ========================================
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402  (one relative-path config for the whole package)
ARRAY_DIR  = P.WORK / "4_tuning" / "data_prep" / "arrays"
META_DIR   = P.WORK / "4_tuning" / "data_prep" / "meta"   # optional
OUTPUT_DIR = P.WORK / "4_tuning" / "BiLSTM" / "output"
MODEL_DIR  = P.WORK / "4_tuning" / "BiLSTM" / "models"

TIMESCALES = [1, 3, 6, 12]
NPZ = {ts: f"SPEI{ts}_3D.npz"  for ts in TIMESCALES}
META = {ts: f"SPEI{ts}_meta.json" for ts in TIMESCALES}

# SPEI-6 ships 13 timesteps but the locked lag-depth decision is 6 -> slice last 6.
WINDOW_OVERRIDE = {6: 6}

# Channel layout (verified from the arrays)
SEQ_SLICE      = slice(0, 10)    # FULL sequence: climate(0-6)+SPEI(7)+month(8-9)
STATIC_SLICE   = slice(10, 18)   # division one-hots
WEATHER_SLICE  = slice(0, 7)     # weather-only sequence: 7 climate vars

VAL_START_YEAR = 2014            # last 5 yrs of train (2014-2018) -> early-stopping/tuning val

# Fixed training controls (only the 4 searched params vary per trial)
BATCH           = 32
EPOCHS_CAP      = 200
PATIENCE_TUNING = 15             # fail-fast during Optuna trials
PATIENCE_FINAL  = 40             # generous patience for the final retrain
N_TRIALS        = 20             # Optuna trials per (timescale, variant)
OPTUNA_SEED     = 42             # reproducible sampler

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

def div_names(ts):
    """Real division names from meta if available, else div_0..div_7."""
    p = META_DIR / META[ts]
    if p.exists():
        try:
            feat = json.load(open(p, encoding="utf-8"))["feature_names_2d"]
            names = sorted(n for n in feat if n.startswith("div"))
            if len(names) == 8:
                return names
        except Exception:
            pass
    return [f"div_{k}" for k in range(8)]

def recover_div(static_block, names):
    return np.array([names[k] for k in np.argmax(static_block, axis=1)])

def temporal_val_mask(target_date_train):
    yrs = np.array([int(str(d)[:4]) for d in target_date_train])
    return yrs >= VAL_START_YEAR, yrs  # (val_mask, years)

def build_model(T, weather_only, lstm_units, dense_units, dropout, lr):
    """Bidirectional LSTM parameterised by the (tuned) hyperparameters."""
    seq_feats = 7 if weather_only else 10
    seq_in = Input(shape=(T, seq_feats), name="seq")
    x = Bidirectional(LSTM(lstm_units, dropout=dropout, recurrent_dropout=dropout), name="bilstm")(seq_in)
    if weather_only:
        inputs = seq_in
    else:
        st_in = Input(shape=(8,), name="static")
        x = Concatenate()([x, st_in])
        inputs = [seq_in, st_in]
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout)(x)
    out = Dense(1)(x)
    m = Model(inputs, out)
    m.compile(optimizer=Adam(lr), loss=tf.keras.losses.Huber(delta=1.0))
    return m

def make_inputs(X, weather_only):
    if weather_only:
        return X[:, :, WEATHER_SLICE]
    return [X[:, :, SEQ_SLICE], X[:, 0, STATIC_SLICE]]

# ============================= PLOTTING =======================================
def scatter_2x2(results, title, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, ts in zip(axes.ravel(), TIMESCALES):
        obs, pred, m = results[ts]["obs"], results[ts]["pred"], results[ts]["metrics"]
        lo = float(min(obs.min(), pred.min())); hi = float(max(obs.max(), pred.max()))
        pad = 0.05 * (hi - lo if hi > lo else 1.0); lo -= pad; hi += pad
        ax.scatter(obs, pred, s=12, alpha=0.5, edgecolor="none", label="test points")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="1:1 line")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed SPEI"); ax.set_ylabel("Predicted SPEI"); ax.set_title(f"SPEI-{ts}")
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

def save_predictions(path, dates, divs, obs, pred, persist):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["target_date", "division", "observed", "predicted", "persistence"])
        for d, dv, o, p, ps in zip(dates, divs, obs, pred, persist):
            w.writerow([str(d), dv, f"{o:.6f}", f"{p:.6f}", f"{ps:.6f}"])

def write_metrics(path, rows):
    cols = ["timescale", "RMSE", "MAE", "NSE", "WI", "R2", "Skill_vs_Persist", "stop_epoch",
            "lstm_units", "dense_units", "dropout", "lr"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in rows:
            w.writerow([r["timescale"]] + [f"{r[c]:.4f}" if isinstance(r.get(c), float) else r.get(c)
                                           for c in cols[1:]])

def write_best_params(path, rows):
    """Separate log of the chosen hyperparameters per timescale (for the thesis appendix)."""
    cols = ["timescale", "lstm_units", "dense_units", "dropout", "lr", "best_val_loss", "n_trials"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in rows:
            w.writerow([r["timescale"], r["lstm_units"], r["dense_units"],
                        f"{r['dropout']:.4f}", f"{r['lr']:.6f}",
                        f"{r['best_val_loss']:.6f}", r["n_trials"]])

# =============================== TUNING =======================================
def fit_once(T, weather_only, Xfit, yfit, Xval, yval, params, patience, extra_callbacks=None):
    """Build+train one model with given params and patience; return (model, history)."""
    tf.keras.utils.set_random_seed(42)
    model = build_model(T, weather_only,
                        params["lstm_units"], params["dense_units"],
                        params["dropout"], params["lr"])
    es  = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-5, verbose=0)
    cbs = [es, rlr]
    if extra_callbacks:
        cbs.extend(extra_callbacks)
    hist = model.fit(Xfit, yfit,
                     validation_data=(Xval, yval),
                     epochs=EPOCHS_CAP, batch_size=BATCH, verbose=0, callbacks=cbs)
    return model, hist

def tune_timescale(T, weather_only, Xfit, yfit, Xval, yval):
    """Optuna study (MedianPruner + per-epoch pruning) minimising best val_loss.
    Returns best_params, best_val. Uses PATIENCE_TUNING for fail-fast trials."""
    def objective(trial):
        params = {
            "lstm_units":  trial.suggest_int("lstm_units", 16, 128, step=16),
            "dense_units": trial.suggest_int("dense_units", 8, 64, step=8),
            "dropout":     trial.suggest_float("dropout", 0.1, 0.5),
            "lr":          trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        }
        extra = [_PRUNING_CB(trial, "val_loss")] if _HAS_PRUNING_CB else None
        _, hist = fit_once(T, weather_only, Xfit, yfit, Xval, yval,
                           params, PATIENCE_TUNING, extra_callbacks=extra)
        return float(np.min(hist.history["val_loss"]))  # lowest val_loss this trial

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params, float(study.best_value)

# =============================== TRAIN ONE =====================================
def run_variant(weather_only):
    tag = "WEATHERONLY" if weather_only else "FULL"
    prune_note = "with mid-training pruning" if _HAS_PRUNING_CB else "study-level pruning only (no TFKeras callback)"
    print(f"\n########## LSTM-{tag} (Optuna, {N_TRIALS} trials/timescale, {prune_note}) ##########")
    results, metric_rows, param_rows = {}, [], []

    for ts in TIMESCALES:
        d = load_npz(ts)
        Xtr, ytr, ptr, dtr = d["X_train"], d["y_train"], d["persist_train"], d["target_date_train"]
        Xte, yte, pte, dte = d["X_test"],  d["y_test"],  d["persist_test"],  d["target_date_test"]

        # ---- Delta target: model learns the change over persistence ----
        ytr_delta = ytr - ptr
        yte_delta = yte - pte   # computed as specified; reconstruction adds pte directly

        # SPEI-6 window override (slice last 6 timesteps)
        if ts in WINDOW_OVERRIDE:
            w = WINDOW_OVERRIDE[ts]
            Xtr, Xte = Xtr[:, -w:, :], Xte[:, -w:, :]

        T = Xtr.shape[1]
        val_mask, _ = temporal_val_mask(dtr)
        fit_mask = ~val_mask

        Xin = make_inputs(Xtr, weather_only)
        if weather_only:
            Xfit, Xval = Xin[fit_mask], Xin[val_mask]
        else:
            Xfit = [Xin[0][fit_mask], Xin[1][fit_mask]]
            Xval = [Xin[0][val_mask], Xin[1][val_mask]]
        yfit, yval = ytr_delta[fit_mask], ytr_delta[val_mask]

        # ---- 1) OPTUNA SEARCH (train/val years only; test untouched; PATIENCE_TUNING) ----
        best_params, best_val = tune_timescale(T, weather_only, Xfit, yfit, Xval, yval)
        print(f"  SPEI-{ts:>2}  best_val_loss={best_val:.5f}  params={best_params}")

        # ---- 2) REBUILD + TRAIN FINAL MODEL with best params (PATIENCE_FINAL) ----
        model, hist = fit_once(T, weather_only, Xfit, yfit, Xval, yval, best_params, PATIENCE_FINAL)
        stop_epoch = int(np.argmin(hist.history["val_loss"]) + 1)

        # ---- 3) Predict delta, reconstruct absolute SPEI, score on TEST ----
        pred_delta = model.predict(make_inputs(Xte, weather_only), verbose=0).ravel()
        final_pred = pred_delta + pte
        met = compute_metrics(yte, final_pred, pte)

        model.save(MODEL_DIR / f"lstm_{tag}_spei{ts}.keras")
        names = div_names(ts)
        divs = recover_div(Xte[:, 0, STATIC_SLICE], names)
        save_predictions(OUTPUT_DIR / f"predictions_{tag}_spei{ts}.csv", dte, divs, yte, final_pred, pte)

        results[ts] = {"obs": np.asarray(yte, float), "pred": final_pred, "metrics": met}
        row = {"timescale": ts, **met, "stop_epoch": stop_epoch,
               "lstm_units": best_params["lstm_units"], "dense_units": best_params["dense_units"],
               "dropout": best_params["dropout"], "lr": best_params["lr"]}
        metric_rows.append(row)
        param_rows.append({"timescale": ts, **best_params,
                           "best_val_loss": best_val, "n_trials": N_TRIALS})
        print(f"  SPEI-{ts:>2}  T={T:>2}  stop@{stop_epoch:>3}  "
              f"RMSE={met['RMSE']:.3f}  NSE={met['NSE']:.3f}  R2={met['R2']:.3f}  "
              f"Skill={met['Skill_vs_Persist']:+.3f}")

    write_metrics(OUTPUT_DIR / f"metrics_{tag}.csv", metric_rows)
    write_best_params(OUTPUT_DIR / f"best_params_{tag}.csv", param_rows)
    scatter_2x2(results, f"LSTM-{tag} (Optuna-tuned): predicted vs observed",
                OUTPUT_DIR / f"scatter_{tag}_2x2.png")
    return results

# =============================== MAIN =========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"TensorFlow {tf.__version__}  |  Optuna {optuna.__version__}  |  "
          f"pruning_cb={'yes' if _HAS_PRUNING_CB else 'no'}  |  "
          f"devices: {[d.device_type for d in tf.config.list_physical_devices()]}")
    run_variant(weather_only=False)   # FULL
    run_variant(weather_only=True)    # WEATHER-ONLY
    print("\nDone.")
    print(f"  Figures + CSVs + best_params -> {OUTPUT_DIR}")
    print(f"  Models                       -> {MODEL_DIR}")

if __name__ == "__main__":
    main()
