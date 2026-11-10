"""
ceemdan_lstm_forecast.py
================================================================================
CEEMDAN-LSTM hybrid drought forecasting (SPEI-1/3/6/12), variants FULL & WEATHERONLY.

Strict pipeline / metric / data-handling / tuning PARITY with the Optuna-tuned
Bidirectional LSTM and CNN-LSTM (same delta target, Huber loss, temporal val,
20-trial Optuna + MedianPruner, dual patience, no recurrent_dropout, same metrics
and figure style). The ONLY genuinely new thing is the CEEMD front-end + the
architecture that consumes it.

--------------------------------------------------------------------------------
CEEMD DESIGN (the part that matters for integrity)
--------------------------------------------------------------------------------
LEAKAGE-FREE, BY CONSTRUCTION. CEEMDAN is applied to the SPEI-history channel of
*each input window on its own* -- using only the timesteps that were already the
model's input. No future values, no cross-sample information, no whole-series
decomposition. Because the decomposition of a sample never looks outside that
sample's own lookback window, train/test separation is preserved exactly.

Trade-off (stated honestly): windows are short (6-13 steps), so CEEMDAN yields
few IMFs (2-3) and the benefit is modest, not the dramatic (and usually leaky)
gains seen in whole-series CEEMD papers. If it doesn't beat the LSTM, that's a
legitimate finding.

Fixed IMF count: CEEMDAN returns a VARIABLE number of IMFs per window, which a
neural net can't ingest. We pad/truncate to N_IMFS fixed channels (zero-padding
faithfully preserves reconstruction, verified). Empty slots are zero.

Caching: decomposition is ~0.5 s/window (~20 min per timescale/variant). It is
computed ONCE per (timescale, variant) and cached to disk; all Optuna trials and
the final retrain reuse the cache. Delete the cache dir to force recompute.

--------------------------------------------------------------------------------
ARCHITECTURE (ambitious but safe)
--------------------------------------------------------------------------------
Dual-stream design that keeps the LSTM comparison logic intact:
  Stream A (RAW seq)  : the original sequence channels -> BiLSTM   (as in LSTM/CNN-LSTM)
  Stream B (IMF seq)  : the CEEMDAN IMF channels        -> BiLSTM
  -> temporal ATTENTION pooling on each stream (learns which timesteps matter,
     instead of only taking the last state)
  -> concat [A, B] (+ static one-hots for FULL) -> dense head -> delta output

Every constant we agreed to keep is kept. New pieces are guarded by CONFIG flags.

Dependencies: EMD-signal (PyEMD).  Install into the LSTM venv:
    python -m pip install EMD-signal
================================================================================
"""

# ------------------------------------------------------------------ imports ---
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Bidirectional, LSTM, Concatenate, Dense, Dropout,
    Layer, LayerNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

import optuna
from optuna.samplers import TPESampler

try:
    from PyEMD import CEEMDAN
except Exception as e:                                # fail loud & clear
    raise ImportError(
        "PyEMD not found. Install into the LSTM venv:\n"
        r"  python -m pip install EMD-signal"
    ) from e

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================= CONFIG BLOCK ===
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402  (one relative-path config for the whole package)
ARRAY_DIR  = P.WORK / "4_tuning" / "data_prep" / "arrays"
META_DIR   = P.WORK / "4_tuning" / "data_prep" / "meta"
OUTPUT_DIR = P.WORK / "4_tuning" / "CEEMDAN-BiLSTM" / "output"
MODEL_DIR  = P.WORK / "4_tuning" / "CEEMDAN-BiLSTM" / "models"
CACHE_DIR  = P.WORK / "4_tuning" / "CEEMDAN-BiLSTM" / "cache"   # decomposition cache

SEED             = 42
VAL_START_YEAR   = 2014          # 2014-2018 -> early-stopping val; fit on < 2014
TEST_START_YEAR  = 2019          # test = 2019-2024 (untouched during tuning)
N_TRIALS         = 20            # per (timescale, variant)  [parity]
EPOCHS           = 200
BATCH_SIZE       = 32
HUBER_DELTA      = 1.0
PATIENCE_TUNING  = 15            # [parity]
PATIENCE_FINAL   = 40            # [parity]
RLR_FACTOR       = 0.5
RLR_PATIENCE     = 15
RLR_MIN_LR       = 1e-5

TIMESCALES       = [1, 3, 6, 12]
VARIANTS         = ["FULL", "WEATHERONLY"]

# Channel layout (18 ch): 0-6 climate | 7 SPEI-so-far | 8-9 month sin/cos | 10-17 div one-hots
SEQ_SLICE_FULL     = slice(0, 10)
SEQ_SLICE_WEATHER  = slice(0, 7)
STATIC_SLICE       = slice(10, 18)
SPEI_CHANNEL       = 7            # the SPEI-so-far channel -> this is what we decompose
N_DIV              = 8

WINDOW_OVERRIDE    = {6: 6}

# ------------------------------------------------------------ CEEMD FLAGS -----
N_IMFS         = 4               # fixed IMF channels fed to the net (pad/truncate)
CEEMDAN_TRIALS = 50              # CEEMDAN ensemble size (noise realisations)
CEEMDAN_SEED   = 42              # reproducible decomposition
USE_CACHE      = True            # cache decompositions to disk (essential for speed)

# --------------------------------------------------------- ARCH FLAGS ---------
# [ADDED] temporal attention pooling instead of last-state only. Learns which
#         timesteps matter. When False, falls back to plain last-state BiLSTM
#         (i.e. byte-for-byte the LSTM/CNN-LSTM recurrent behaviour).
USE_ATTENTION = True
# [ADDED] LayerNorm on each stream's pooled vector (training stability on small data).
USE_LAYERNORM = True

# ------------------------------------------------------------ PRUNING ---------
USE_PRUNING       = True
PRUNER_STARTUP    = 5
PRUNER_WARMUP     = 10
# ------------------------------------------------------------------------------

for d in (OUTPUT_DIR, MODEL_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)


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
    return np.array([int(str(d)[:4]) for d in dates])


def apply_window(X, ts):
    if ts in WINDOW_OVERRIDE:
        w = WINDOW_OVERRIDE[ts]
        return X[:, -w:, :]
    return X


def load_division_names(ts):
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
    idx = X_static_block.argmax(axis=1)
    return np.array([div_names[i] for i in idx])


def compute_metrics(obs, pred, persist):
    """RMSE, MAE, NSE, WI, R2=(Pearson r)^2, Skill=1-MSE/MSE_persist. Identical to CNN-LSTM."""
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


# ==================================================== CEEMDAN DECOMPOSITION ====
def decompose_window(spei_seq, n_imfs=N_IMFS, trials=CEEMDAN_TRIALS, seed=CEEMDAN_SEED):
    """
    Leakage-free CEEMDAN on ONE window's SPEI history.
    Input:  spei_seq (T,) -- the SPEI-so-far channel of a single sample's window.
    Output: (T, n_imfs)   -- fixed IMF channels, zero-padded/truncated.

    Only this window's own past is used; nothing external enters. Zero-padding
    empty slots preserves reconstruction exactly (sum of channels == input).
    """
    T = spei_seq.shape[0]
    out = np.zeros((n_imfs, T), dtype=float)
    # constant / degenerate window -> CEEMDAN undefined; return zeros (net sees no structure)
    if T < 2 or np.allclose(spei_seq, spei_seq[0]):
        return out.T
    # parallel=False: PyEMD defaults to spawning a fresh multiprocessing pool per
    # call, and this is called once per window (thousands of times) -> pool-spawn
    # overhead dwarfs the actual (tiny, T=6-13) decomposition work. Single-threaded
    # per window is far faster here; parallelism belongs at the window-loop level
    # if ever needed.
    c = CEEMDAN(trials=trials, parallel=False)
    c.noise_seed(seed)
    try:
        imfs = c.ceemdan(spei_seq.astype(float))     # (k, T)
    except Exception:
        return out.T                                 # robust: never crash the run
    k = imfs.shape[0]
    take = min(k, n_imfs)
    out[:take] = imfs[:take]
    # if k > n_imfs, fold the extra (highest-order/residual) IMFs into the last kept slot
    # so no signal is silently dropped (keeps reconstruction faithful)
    if k > n_imfs:
        out[n_imfs - 1] += imfs[n_imfs:].sum(axis=0)
    return out.T                                     # (T, n_imfs)


def decompose_block(spei_block, tag):
    """
    Decompose every window in a (n, T) SPEI block -> (n, T, N_IMFS). Cached to disk.
    """
    cache_f = CACHE_DIR / f"imfs_{tag}.npy"
    if USE_CACHE and cache_f.exists():
        arr = np.load(cache_f)
        if arr.shape[0] == spei_block.shape[0] and arr.shape[1] == spei_block.shape[1]:
            print(f"    [cache] loaded IMFs {arr.shape} from {cache_f.name}", flush=True)
            return arr
    n, T = spei_block.shape
    out = np.zeros((n, T, N_IMFS), dtype=np.float32)
    print(f"    [ceemdan] decomposing {n} windows (T={T}) ... this is the slow step", flush=True)
    for i in range(n):
        out[i] = decompose_window(spei_block[i])
        if (i + 1) % 250 == 0:
            print(f"      {i+1}/{n}", flush=True)
    if USE_CACHE:
        np.save(cache_f, out)
        print(f"    [cache] saved IMFs -> {cache_f.name}", flush=True)
    return out


# ============================================================ ATTENTION ========
class TemporalAttention(Layer):
    """
    Additive (Bahdanau-style) temporal attention pooling over a sequence.
    Input : (batch, T, units)  ->  Output: (batch, units)
    Learns a weight per timestep and returns the weighted sum, so the model can
    emphasise informative months instead of only using the final hidden state.
    """
    def __init__(self, **kw):
        super().__init__(**kw)

    def build(self, input_shape):
        u = int(input_shape[-1])
        self.W = self.add_weight(name="att_W", shape=(u, u),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_b", shape=(u,),
                                 initializer="zeros", trainable=True)
        self.v = self.add_weight(name="att_v", shape=(u, 1),
                                 initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, x):                               # x: (B, T, U)
        score = tf.tanh(tf.tensordot(x, self.W, axes=[[2], [0]]) + self.b)  # (B,T,U)
        score = tf.tensordot(score, self.v, axes=[[2], [0]])               # (B,T,1)
        weights = tf.nn.softmax(score, axis=1)                            # (B,T,1)
        return tf.reduce_sum(x * weights, axis=1)                         # (B,U)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def pool_stream(seq_tensor, lstm_units, dropout, name):
    """BiLSTM -> (attention | last-state) pooling -> optional LayerNorm."""
    if USE_ATTENTION:
        h = Bidirectional(LSTM(lstm_units, dropout=dropout, return_sequences=True),
                          name=f"{name}_bilstm")(seq_tensor)
        pooled = TemporalAttention(name=f"{name}_attn")(h)
    else:
        # fallback == exact LSTM/CNN-LSTM behaviour (last state, no recurrent_dropout)
        pooled = Bidirectional(LSTM(lstm_units, dropout=dropout),
                               name=f"{name}_bilstm")(seq_tensor)
    if USE_LAYERNORM:
        pooled = LayerNormalization(name=f"{name}_ln")(pooled)
    return pooled


# ============================================================= MODEL ===========
def build_model(variant, T, seq_feats, lstm_units, dense_units, dropout, lr):
    """
    Dual-stream: RAW sequence stream + IMF stream, each BiLSTM + attention pooled,
    concatenated (+ static one-hots for FULL) -> dense head -> delta output.
    Keeps: BiLSTM core, no recurrent_dropout, input+head dropout, Huber, delta target.
    """
    raw_in = Input(shape=(T, seq_feats), name="seq_input")
    imf_in = Input(shape=(T, N_IMFS),    name="imf_input")

    raw_vec = pool_stream(raw_in, lstm_units, dropout, name="raw")
    imf_vec = pool_stream(imf_in, lstm_units, dropout, name="imf")

    merged = Concatenate(name="stream_concat")([raw_vec, imf_vec])

    if variant == "FULL":
        static_in = Input(shape=(N_DIV,), name="static_input")
        merged = Concatenate(name="static_concat")([merged, static_in])
        inputs = [raw_in, imf_in, static_in]
    else:
        inputs = [raw_in, imf_in]

    x = Dense(dense_units, activation="relu")(merged)
    x = Dropout(dropout)(x)
    out = Dense(1)(x)

    model = Model(inputs, out)
    model.compile(optimizer=Adam(learning_rate=lr), loss=Huber(delta=HUBER_DELTA))
    return model


def make_inputs(variant, raw_seq, imf_seq, static):
    return [raw_seq, imf_seq, static] if variant == "FULL" else [raw_seq, imf_seq]


# --------------------------------------------------- [ADDED] pruning callback --
class KerasPruningCallback(Callback):
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


# =========================================================== TUNE ONE ==========
def tune_and_train(ts, variant):
    tag = variant
    print(f"\n{'='*70}\n[CEEMDAN-LSTM][{variant}] SPEI-{ts}\n{'='*70}", flush=True)

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

    seq_slice = SEQ_SLICE_FULL if variant == "FULL" else SEQ_SLICE_WEATHER
    seq_feats = seq_slice.stop - seq_slice.start
    exp = 10 if variant == "FULL" else 7
    assert seq_feats == exp, f"seq_feats={seq_feats}, expected {exp}"

    Xtr_seq = Xtr_full[:, :, seq_slice]
    Xte_seq = Xte_full[:, :, seq_slice]
    Xtr_static = Xtr_full[:, 0, STATIC_SLICE]
    Xte_static = Xte_full[:, 0, STATIC_SLICE]

    # ---- CEEMDAN on the SPEI channel of each window (leakage-free, cached) ----
    # SPEI channel index is 7 in the full array for both variants.
    spei_tr = Xtr_full[:, :, SPEI_CHANNEL]           # (n_tr, T)
    spei_te = Xte_full[:, :, SPEI_CHANNEL]           # (n_te, T)
    imf_tr = decompose_block(spei_tr, tag=f"SPEI{ts}_{variant}_train")
    imf_te = decompose_block(spei_te, tag=f"SPEI{ts}_{variant}_test")

    # ---- fit / val split (temporal, leak-free) ----
    yr_tr = year_of(dtr_all)
    fit_m = yr_tr <  VAL_START_YEAR
    val_m = yr_tr >= VAL_START_YEAR

    seq_fit, seq_val   = Xtr_seq[fit_m],    Xtr_seq[val_m]
    imf_fit, imf_val   = imf_tr[fit_m],     imf_tr[val_m]
    stat_fit, stat_val = Xtr_static[fit_m], Xtr_static[val_m]
    y_fit, y_val = ytr_all[fit_m], ytr_all[val_m]
    p_fit, p_val = ptr_all[fit_m], ptr_all[val_m]

    yd_fit = y_fit - p_fit
    yd_val = y_val - p_val

    in_fit = make_inputs(variant, seq_fit, imf_fit, stat_fit)
    in_val = make_inputs(variant, seq_val, imf_val, stat_val)

    # ---- Optuna objective (no CNN params; attention arch is fixed) ----
    def objective(trial):
        lstm_units  = trial.suggest_int("lstm_units", 16, 128, step=16)
        dense_units = trial.suggest_int("dense_units", 8, 64, step=8)
        dropout     = trial.suggest_float("dropout", 0.1, 0.5)
        lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        set_all_seeds()
        model = build_model(variant, T, seq_feats, lstm_units, dense_units, dropout, lr)
        cbs = [
            EarlyStopping(monitor="val_loss", patience=PATIENCE_TUNING,
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=RLR_FACTOR,
                              patience=RLR_PATIENCE, min_lr=RLR_MIN_LR),
        ]
        if USE_PRUNING:
            cbs.append(KerasPruningCallback(trial, "val_loss"))
        hist = model.fit(in_fit, yd_fit, validation_data=(in_val, yd_val),
                         epochs=EPOCHS, batch_size=BATCH_SIZE,
                         callbacks=cbs, verbose=0, shuffle=False)
        tf.keras.backend.clear_session()
        return float(np.min(hist.history["val_loss"]))

    pruner = (optuna.pruners.MedianPruner(n_startup_trials=PRUNER_STARTUP,
                                          n_warmup_steps=PRUNER_WARMUP)
              if USE_PRUNING else optuna.pruners.NopPruner())
    study = optuna.create_study(direction="minimize",
                                sampler=TPESampler(seed=SEED), pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    bp = study.best_params
    best_val = study.best_value
    print(f"  best val_loss={best_val:.5f}  params={bp}", flush=True)

    # ---- retrain best config ----
    set_all_seeds()
    model = build_model(variant, T, seq_feats,
                        bp["lstm_units"], bp["dense_units"], bp["dropout"], bp["lr"])
    cbs = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE_FINAL,
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=RLR_FACTOR,
                          patience=RLR_PATIENCE, min_lr=RLR_MIN_LR),
    ]
    hist = model.fit(in_fit, yd_fit, validation_data=(in_val, yd_val),
                     epochs=EPOCHS, batch_size=BATCH_SIZE,
                     callbacks=cbs, verbose=0, shuffle=False)
    stop_epoch = len(hist.history["loss"])

    # ---- predict test (reconstruct from delta) ----
    in_te = make_inputs(variant, Xte_seq, imf_te, Xte_static)
    pred_delta = model.predict(in_te, verbose=0).ravel()
    final_pred = pred_delta + pte
    metrics = compute_metrics(yte, final_pred, pte)

    # ---- save ----
    model.save(MODEL_DIR / f"ceemdan_lstm_{tag}_spei{ts}.keras")
    div_names = load_division_names(ts)
    divisions = recover_divisions(Xte_static, div_names)
    pd.DataFrame({
        "target_date": [str(d) for d in dte],
        "division":    divisions,
        "observed":    yte,
        "predicted":   final_pred,
        "persistence": pte,
    }).to_csv(OUTPUT_DIR / f"predictions_{tag}_spei{ts}.csv", index=False)

    metrics_row = dict(timescale=ts, **metrics, stop_epoch=stop_epoch, **bp)
    params_row  = dict(timescale=ts, lstm_units=bp["lstm_units"], dense_units=bp["dense_units"],
                       dropout=bp["dropout"], lr=bp["lr"],
                       best_val_loss=best_val, n_trials=N_TRIALS,
                       n_imfs=N_IMFS, ceemdan_trials=CEEMDAN_TRIALS)

    tf.keras.backend.clear_session()
    return metrics_row, params_row, (yte, final_pred, metrics)


# ============================================================= MAIN ============
def main():
    print(f"TensorFlow {tf.__version__}  |  N_IMFS={N_IMFS}  CEEMDAN_TRIALS={CEEMDAN_TRIALS}  "
          f"attention={USE_ATTENTION}  layernorm={USE_LAYERNORM}", flush=True)
    for variant in VARIANTS:
        metrics_rows, params_rows, scatter_data = [], [], {}
        for ts in TIMESCALES:
            m_row, p_row, sdata = tune_and_train(ts, variant)
            metrics_rows.append(m_row)
            params_rows.append(p_row)
            scatter_data[ts] = sdata

        pd.DataFrame(metrics_rows).to_csv(OUTPUT_DIR / f"metrics_{variant}.csv", index=False)
        pd.DataFrame(params_rows).to_csv(OUTPUT_DIR / f"best_params_{variant}.csv", index=False)

        # ---- 2x2 scatter (identical style to LSTM / CNN-LSTM) ----
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
        fig.suptitle(f"CEEMDAN-LSTM ({variant}): predicted vs observed", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(OUTPUT_DIR / f"scatter_{variant}_2x2.png", dpi=150)
        plt.close(fig)

        print(f"\n[CEEMDAN-LSTM][{variant}] done -> {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
