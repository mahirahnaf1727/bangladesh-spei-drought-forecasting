"""
cnn_forecast.py
================================================================================
Pure 1D-CNN drought forecasting (SPEI-1/3/6/12), variants FULL & WEATHERONLY.

Strict pipeline / metric / data-handling / tuning parity with the CNN-LSTM
(cnn_lstm_forecast.py) and the Optuna-tuned Bidirectional LSTM. The ONLY thing
that differs from the CNN-LSTM is the model graph:

    CNN-LSTM :  Conv1D -> [BN] -> relu -> BiLSTM        -> [concat static] -> dense head
    CNN      :  Conv1D -> [BN] -> relu -> Conv1D -> [BN] -> relu
                -> SpatialDropout1D -> GlobalAveragePooling1D
                                                        -> [concat static] -> dense head

The recurrent block is replaced by (a) a second Conv1D block that gives the CNN
comparable depth/capacity, and (b) GlobalAveragePooling1D, which collapses the
time axis into a fixed-length vector the same way the BiLSTM did -- and does so
identically for every SPEI window (T = 1, 3, 6, 12). The tuned `lstm_units`
hyperparameter is replaced by `conv2_filters` so the Optuna search keeps exactly
6 tuned dimensions.

Everything else -- arrays, slices, split, delta targets, seeds, callbacks,
metrics, predictions/metrics/params CSVs, and the 2x2 scatter -- is unchanged.
Deviations from the raw spec are limited to ADDITIONS, each guarded by a CONFIG
flag and marked in code with:  # [ADDED] ...
================================================================================
"""

# ------------------------------------------------------------------ imports ---
import os
os.environ["PYTHONHASHSEED"] = "42"          # note: only fully effective if also
                                             # exported in the shell before launch
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                        # headless-safe figure backend
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D,
    SpatialDropout1D, Concatenate, Dense, Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================= CONFIG BLOCK ===
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402  (one relative-path config for the whole package)
ARRAY_DIR  = P.WORK / "4_tuning" / "data_prep" / "arrays"
META_DIR   = P.WORK / "4_tuning" / "data_prep" / "meta"
OUTPUT_DIR = P.WORK / "4_tuning" / "CNN" / "output"
MODEL_DIR  = P.WORK / "4_tuning" / "CNN" / "models"

SEED             = 42
VAL_START_YEAR   = 2014          # 2014-2018 -> early-stopping val; fit on < 2014
TEST_START_YEAR  = 2019          # test = 2019-2024 (untouched during tuning)
N_TRIALS         = 20            # per (timescale, variant)  [matches CNN-LSTM / try_5 LSTM]
EPOCHS           = 200
BATCH_SIZE       = 32
HUBER_DELTA      = 1.0
PATIENCE_TUNING  = 15            # fail-fast during Optuna trials   [parity with CNN-LSTM]
PATIENCE_FINAL   = 40            # generous patience for final retrain
RLR_FACTOR       = 0.5
RLR_PATIENCE     = 15
RLR_MIN_LR       = 1e-5

TIMESCALES       = [1, 3, 6, 12]
VARIANTS         = ["FULL", "WEATHERONLY"]

# Channel layout (18 ch): 0-6 climate | 7 SPEI-so-far | 8-9 month sin/cos | 10-17 div one-hots
SEQ_SLICE_FULL     = slice(0, 10)   # FULL sequence feats  -> 10 channels
SEQ_SLICE_WEATHER  = slice(0, 7)    # WEATHERONLY feats     -> 7  channels
STATIC_SLICE       = slice(10, 18)  # 8 division one-hots
N_DIV              = 8

# window == accumulation length; SPEI-6 array ships 13 steps -> slice last 6
WINDOW_OVERRIDE    = {6: 6}

# ------------------------------------------------------------ ADDED FLAGS -----
# [ADDED] BatchNorm in the CNN blocks. Part of the CNN feature extractor (applied
#         identically to both conv blocks). When False the convs revert to plain
#         relu Conv1D. Default ON = stabilises the feature extractor on this small,
#         non-stationary dataset. Kept identical to the CNN-LSTM front-end so the
#         two architectures share the same conv-normalisation regime.
USE_BATCHNORM = True

# [ADDED] Optuna MedianPruner. Kills hopeless trials early. The try_5 LSTM DID use
#         a MedianPruner(n_startup_trials=5, n_warmup_steps=10), so pruning is now ON
#         to preserve parity with that protocol (and to keep the search fast).
USE_PRUNING       = True
PRUNER_STARTUP    = 5            # trials run fully before pruning kicks in
PRUNER_WARMUP     = 10           # epochs before a trial becomes prunable
# ------------------------------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ========================================================== DETERMINISM =======
def set_all_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


set_all_seeds()


# ============================================================== HELPERS ========
def year_of(dates):
    """Extract integer year from 'YYYY-MM-DD' strings (handles bytes/np.str_)."""
    return np.array([int(str(d)[:4]) for d in dates])


def apply_window(X, ts):
    """Slice the last W timesteps if a window override exists for this timescale."""
    if ts in WINDOW_OVERRIDE:
        w = WINDOW_OVERRIDE[ts]
        return X[:, -w:, :]
    return X


def load_division_names(ts):
    """
    Recover division names from meta (feature_names_2d); fall back to div_0..div_7.
    Sorted to mirror the LSTM stage's 'argmax over sorted div columns' convention.
    NOTE: verify this mapping against the LSTM predictions_*.csv before building any
    thesis figure (see handoff caution on division-label correctness).
    """
    p = META_DIR / f"SPEI{ts}_meta.json"
    try:
        meta = json.loads(p.read_text())
        fn = meta.get("feature_names_2d", [])
        divs = [f for f in fn if "div" in str(f).lower()]
        if len(divs) == N_DIV:
            return sorted(divs)
    except Exception:
        pass
    return [f"div_{i}" for i in range(N_DIV)]


def recover_divisions(X_static_block, div_names):
    """argmax over the 8 one-hot channels at timestep 0 -> division name."""
    idx = X_static_block.argmax(axis=1)          # (samples,)
    return np.array([div_names[i] for i in idx])


def make_inputs(variant, seq, static):
    return [seq, static] if variant == "FULL" else seq


def compute_metrics(obs, pred, persist):
    """RMSE, MAE, NSE(vs test mean), WI(Willmott d), R2=(Pearson r)^2, Skill=1-MSE/MSE_persist."""
    obs = np.asarray(obs, float); pred = np.asarray(pred, float); persist = np.asarray(persist, float)
    err = pred - obs
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    obar = obs.mean()

    den_nse = np.sum((obs - obar) ** 2)
    nse = float(1 - np.sum(err ** 2) / den_nse) if den_nse > 0 else np.nan

    den_wi = np.sum((np.abs(pred - obar) + np.abs(obs - obar)) ** 2)
    wi = float(1 - np.sum(err ** 2) / den_wi) if den_wi > 0 else np.nan

    if obs.std() > 0 and pred.std() > 0:
        r = float(np.corrcoef(obs, pred)[0, 1]); r2 = r ** 2
    else:
        r2 = np.nan

    mse_model   = float(np.mean(err ** 2))
    mse_persist = float(np.mean((persist - obs) ** 2))
    skill = float(1 - mse_model / mse_persist) if mse_persist > 0 else np.nan

    return dict(RMSE=rmse, MAE=mae, NSE=nse, WI=wi, R2=r2, Skill_vs_Persist=skill)


# ============================================================= MODEL ===========
def build_model(variant, T, seq_feats, filters, kernel_size,
                conv2_filters, dense_units, dropout, lr):
    seq_in = Input(shape=(T, seq_feats), name="seq_input")

    # ---- CNN block 1: padding='same' preserves T ------------------------------
    x = Conv1D(filters=filters, kernel_size=kernel_size,
               padding="same", activation=None)(seq_in)
    if USE_BATCHNORM:                                  # [ADDED] see CONFIG flags
        x = BatchNormalization()(x)
    x = Activation("relu")(x)                          # equals conv(relu) when BN off

    # ---- CNN block 2: extra depth (replaces the CNN-LSTM recurrent block) -----
    # This second convolution gives the pure CNN capacity comparable to the
    # BiLSTM it stands in for. Kernel size is shared with block 1 (single tuned
    # `kernel_size`), keeping the search space the same size as the CNN-LSTM.
    x = Conv1D(filters=conv2_filters, kernel_size=kernel_size,
               padding="same", activation=None)(x)
    if USE_BATCHNORM:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # channel-wise dropout on the conv feature maps -- the CNN analogue of the
    # BiLSTM's input `dropout`, so regularization strength stays comparable.
    x = SpatialDropout1D(dropout)(x)

    # ---- temporal aggregation: collapse the T axis into a fixed vector --------
    # GlobalAveragePooling1D is T-independent, so it reduces every SPEI window
    # (T = 1, 3, 6, 12) to a fixed (conv2_filters,) vector -- the CNN counterpart
    # of the BiLSTM's sequence-to-vector reduction. No Flatten (which would make
    # the head size depend on T) and no manual pooling stride.
    x = GlobalAveragePooling1D()(x)

    # ---- FULL branch: concat static division one-hots -------------------------
    if variant == "FULL":
        static_in = Input(shape=(N_DIV,), name="static_input")
        x = Concatenate()([x, static_in])
        inputs = [seq_in, static_in]
    else:
        inputs = seq_in

    # ---- dense head -----------------------------------------------------------
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout)(x)
    out = Dense(1)(x)

    model = Model(inputs, out)
    model.compile(optimizer=Adam(learning_rate=lr), loss=Huber(delta=HUBER_DELTA))
    return model


# --------------------------------------------------- [ADDED] pruning callback --
class KerasPruningCallback(Callback):
    """Lightweight optuna pruning hook (avoids the fragile optuna-integration import)."""
    def __init__(self, trial, monitor="val_loss"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val = logs.get(self.monitor)
        if val is None:
            return
        self.trial.report(val, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def kernel_bounds(T):
    """kernel_size in [2, min(5, T)], guarded for very short windows."""
    hi = min(5, T)
    lo = 2
    if hi < lo:          # T < 2 (degenerate) -> force kernel_size = 1
        return 1, 1
    return lo, hi


# =========================================================== TUNE ONE ==========
def tune_and_train(ts, variant):
    tag = variant
    print(f"\n{'='*70}\n[{variant}] SPEI-{ts}\n{'='*70}", flush=True)

    # ---------------- load arrays ----------------
    npz = np.load(ARRAY_DIR / f"SPEI{ts}_3D.npz", allow_pickle=True)
    Xtr_full = apply_window(npz["X_train"], ts)
    Xte_full = apply_window(npz["X_test"],  ts)
    ytr_all  = npz["y_train"].astype(float)
    yte      = npz["y_test"].astype(float)
    ptr_all  = npz["persist_train"].astype(float)
    pte      = npz["persist_test"].astype(float)
    dtr_all  = npz["target_date_train"]
    dte      = npz["target_date_test"]

    T = Xtr_full.shape[1]

    # sequence / static slices per variant
    seq_slice = SEQ_SLICE_FULL if variant == "FULL" else SEQ_SLICE_WEATHER
    seq_feats = seq_slice.stop - seq_slice.start
    exp = 10 if variant == "FULL" else 7
    assert seq_feats == exp, f"seq_feats={seq_feats}, expected {exp}"

    Xtr_seq = Xtr_full[:, :, seq_slice]
    Xte_seq = Xte_full[:, :, seq_slice]
    Xtr_static = Xtr_full[:, 0, STATIC_SLICE]        # (n, 8)
    Xte_static = Xte_full[:, 0, STATIC_SLICE]

    # ---------------- train -> fit / val split (leak-free) ----------------
    yr_tr = year_of(dtr_all)
    fit_m = yr_tr <  VAL_START_YEAR
    val_m = (yr_tr >= VAL_START_YEAR)                 # tuning val = 2014-2018

    seq_fit, seq_val = Xtr_seq[fit_m], Xtr_seq[val_m]
    stat_fit, stat_val = Xtr_static[fit_m], Xtr_static[val_m]
    y_fit, y_val = ytr_all[fit_m], ytr_all[val_m]
    p_fit, p_val = ptr_all[fit_m], ptr_all[val_m]

    # delta targets (BOTH train and val -> early stopping stays meaningful)
    yd_fit = y_fit - p_fit
    yd_val = y_val - p_val

    in_fit = make_inputs(variant, seq_fit, stat_fit)
    in_val = make_inputs(variant, seq_val, stat_val)

    k_lo, k_hi = kernel_bounds(T)
    if T < 2:
        print(f"  [warn] T={T} < 2: degenerate sequence; kernel_size forced to 1.", flush=True)

    # ---------------- Optuna objective ----------------
    def objective(trial):
        filters       = trial.suggest_int("filters", 16, 128, step=16)
        kernel_size   = trial.suggest_int("kernel_size", k_lo, k_hi)
        conv2_filters = trial.suggest_int("conv2_filters", 16, 128, step=16)
        dense_units   = trial.suggest_int("dense_units", 8, 64, step=8)
        dropout       = trial.suggest_float("dropout", 0.1, 0.5)
        lr            = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        set_all_seeds()                              # reproducible per-trial init
        model = build_model(variant, T, seq_feats, filters, kernel_size,
                            conv2_filters, dense_units, dropout, lr)

        cbs = [
            EarlyStopping(monitor="val_loss", patience=PATIENCE_TUNING,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=RLR_FACTOR,
                              patience=RLR_PATIENCE, min_lr=RLR_MIN_LR),
        ]
        if USE_PRUNING:                              # [ADDED]
            cbs.append(KerasPruningCallback(trial, "val_loss"))

        hist = model.fit(in_fit, yd_fit,
                         validation_data=(in_val, yd_val),
                         epochs=EPOCHS, batch_size=BATCH_SIZE,
                         callbacks=cbs, verbose=0, shuffle=False)
        tf.keras.backend.clear_session()
        return float(np.min(hist.history["val_loss"]))

    pruner = (optuna.pruners.MedianPruner(n_startup_trials=PRUNER_STARTUP,
                                          n_warmup_steps=PRUNER_WARMUP)
              if USE_PRUNING else optuna.pruners.NopPruner())
    study = optuna.create_study(direction="minimize",
                                sampler=TPESampler(seed=SEED),
                                pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    bp = study.best_params
    best_val = study.best_value
    print(f"  best val_loss={best_val:.5f}  params={bp}", flush=True)

    # ---------------- retrain best config (reproducible final model) ----------
    set_all_seeds()
    model = build_model(variant, T, seq_feats,
                        bp["filters"], bp["kernel_size"], bp["conv2_filters"],
                        bp["dense_units"], bp["dropout"], bp["lr"])
    cbs = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE_FINAL,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=RLR_FACTOR,
                          patience=RLR_PATIENCE, min_lr=RLR_MIN_LR),
    ]
    hist = model.fit(in_fit, yd_fit,
                     validation_data=(in_val, yd_val),
                     epochs=EPOCHS, batch_size=BATCH_SIZE,
                     callbacks=cbs, verbose=0, shuffle=False)
    stop_epoch = len(hist.history["loss"])

    # ---------------- predict test (reconstruct from delta) -------------------
    in_te = make_inputs(variant, Xte_seq, Xte_static)
    pred_delta = model.predict(in_te, verbose=0).ravel()
    final_pred = pred_delta + pte                    # add persistence back

    metrics = compute_metrics(yte, final_pred, pte)

    # ---------------- save model + predictions --------------------------------
    model.save(MODEL_DIR / f"cnn_{tag}_spei{ts}.keras")

    div_names = load_division_names(ts)
    divisions = recover_divisions(Xte_static, div_names)
    pred_df = pd.DataFrame({
        "target_date": [str(d) for d in dte],
        "division":    divisions,
        "observed":    yte,
        "predicted":   final_pred,
        "persistence": pte,
    })
    pred_df.to_csv(OUTPUT_DIR / f"predictions_{tag}_spei{ts}.csv", index=False)

    metrics_row = dict(timescale=ts, **metrics, stop_epoch=stop_epoch, **bp)
    params_row  = dict(timescale=ts, filters=bp["filters"], kernel_size=bp["kernel_size"],
                       conv2_filters=bp["conv2_filters"], dense_units=bp["dense_units"],
                       dropout=bp["dropout"], lr=bp["lr"],
                       best_val_loss=best_val, n_trials=N_TRIALS)

    tf.keras.backend.clear_session()
    return metrics_row, params_row, (yte, final_pred, metrics)


# ============================================================= MAIN ============
def main():
    for variant in VARIANTS:
        metrics_rows, params_rows, scatter_data = [], [], {}
        for ts in TIMESCALES:
            m_row, p_row, sdata = tune_and_train(ts, variant)
            metrics_rows.append(m_row)
            params_rows.append(p_row)
            scatter_data[ts] = sdata

        # ---- per-variant CSVs ----
        pd.DataFrame(metrics_rows).to_csv(OUTPUT_DIR / f"metrics_{variant}.csv", index=False)
        pd.DataFrame(params_rows).to_csv(OUTPUT_DIR / f"best_params_{variant}.csv", index=False)

        # ---- 2x2 scatter (styled to match the LSTM figure) ----
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
        for ax, ts in zip(axes.ravel(), TIMESCALES):
            obs, pred, mets = scatter_data[ts]
            lo = float(min(obs.min(), pred.min()))
            hi = float(max(obs.max(), pred.max()))
            pad = 0.05 * (hi - lo if hi > lo else 1.0); lo -= pad; hi += pad
            ax.scatter(obs, pred, s=12, alpha=0.5, edgecolor="none", label="test points")
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="1:1 line")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Observed SPEI"); ax.set_ylabel("Predicted SPEI")
            ax.set_title(f"SPEI-{ts}")
            box = (f"RMSE = {mets['RMSE']:.3f}\nNSE  = {mets['NSE']:.3f}\n"
                   f"R2   = {mets['R2']:.3f}\nSkill= {mets['Skill_vs_Persist']:.3f}")
            ax.text(0.04, 0.96, box, transform=ax.transAxes, va="top", ha="left",
                    fontsize=9, family="monospace",
                    bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        fig.suptitle(f"CNN ({variant}): predicted vs observed", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(OUTPUT_DIR / f"scatter_{variant}_2x2.png", dpi=150)
        plt.close(fig)

        print(f"\n[{variant}] done -> {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
