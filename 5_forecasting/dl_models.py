"""
================================================================================
dl_models.py  --  the four deep-learning architectures + one training routine
================================================================================
STAGE 2 CHANGES (15 Oct 2026)
  * CEEMDAN-BiLSTM added, copied unchanged from ceemdan_lstm_forecast.py
    (two BiLSTM streams with temporal attention + LayerNorm; IMF input from
    ceemdan_features.py).
  * Batch shuffling follows common.SHUFFLE_BY_MODEL (see the note there).
  * train_and_evaluate() can save the model to a per-seed path.
  Nothing else changed: the other three builders are byte-for-byte as before.
--------------------------------------------------------------------------------
The architectures are reproduced EXACTLY from the thesis scripts. Any change
here breaks the lead-1 regression test, which is the point of that test.

WHERE EACH ARCHITECTURE COMES FROM, AND ONE TRAP
  CNN-BiLSTM  <- cnn_lstm_forecast.py   (BatchNorm on, no recurrent_dropout)
  CNN         <- cnn_forecast.py        (two conv blocks, GlobalAveragePooling)
  BiLSTM      <- prove_improvement.py   *** NOT lstm_forecast.py ***

  The trap: lstm_forecast.py builds its BiLSTM with
      Bidirectional(LSTM(units, dropout=d, recurrent_dropout=d))
  while prove_improvement.py applies the "speed fix" and drops recurrent_dropout
      Bidirectional(LSTM(units, dropout=d))
  These are different models and produce different numbers. The BiLSTM results
  reported in the thesis (SPEI-6: RMSE 0.4966, NSE 0.7109, skill 0.3359) come
  from the prove_improvement.py version at seed 42, so that is the one used here.

ALL THREE MODELS PREDICT THE DELTA AGAINST PERSISTENCE
      train on   d = y[t+LEAD] - y[t]
      predict    yhat = dhat + y[t]
  Persistence is y[t] at every lead. XGBoost is the exception -- it predicts the
  SPEI level directly (see run6).
================================================================================
"""

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, SpatialDropout1D,
    GlobalAveragePooling1D, Bidirectional, LSTM, Dense, Dropout, Concatenate,
    Layer, LayerNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import common as C
import ceemdan_features as CF


# =============================================================================
# architectures
# =============================================================================

def build_cnn_bilstm(T, seq_feats, filters, kernel_size, lstm_units,
                     dense_units, dropout, lr):
    """CNN front-end (padding='same', so T is preserved -- no pooling, no
    flatten) feeding a BiLSTM, then concat of the static division one-hots."""
    seq_in = Input(shape=(T, seq_feats), name="seq_input")

    x = Conv1D(filters=filters, kernel_size=kernel_size,
               padding="same", activation=None)(seq_in)
    if C.USE_BATCHNORM:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # no recurrent_dropout -> the fast fused LSTM kernel is used
    x = Bidirectional(LSTM(lstm_units, dropout=dropout))(x)

    static_in = Input(shape=(C.N_DIV,), name="static_input")
    x = Concatenate()([x, static_in])

    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout)(x)
    out = Dense(1)(x)

    model = Model([seq_in, static_in], out)
    model.compile(optimizer=Adam(learning_rate=lr), loss=Huber(delta=C.HUBER_DELTA))
    return model


def build_cnn(T, seq_feats, filters, kernel_size, conv2_filters,
              dense_units, dropout, lr):
    """Two conv blocks, then GlobalAveragePooling1D to collapse the time axis.

    Pooling (rather than Flatten) keeps the head size independent of T, which is
    what lets the same architecture serve every window length."""
    seq_in = Input(shape=(T, seq_feats), name="seq_input")

    x = Conv1D(filters=filters, kernel_size=kernel_size,
               padding="same", activation=None)(seq_in)
    if C.USE_BATCHNORM:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(filters=conv2_filters, kernel_size=kernel_size,
               padding="same", activation=None)(x)
    if C.USE_BATCHNORM:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # channel-wise dropout: the CNN analogue of the BiLSTM's input dropout
    x = SpatialDropout1D(dropout)(x)
    x = GlobalAveragePooling1D()(x)

    static_in = Input(shape=(C.N_DIV,), name="static_input")
    x = Concatenate()([x, static_in])

    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout)(x)
    out = Dense(1)(x)

    model = Model([seq_in, static_in], out)
    model.compile(optimizer=Adam(learning_rate=lr), loss=Huber(delta=C.HUBER_DELTA))
    return model


def build_bilstm(T, seq_feats, lstm_units, dense_units, dropout, lr):
    """BiLSTM exactly as in prove_improvement.py -- NO recurrent_dropout."""
    seq_in = Input(shape=(T, seq_feats), name="seq")
    x = Bidirectional(LSTM(lstm_units, dropout=dropout), name="bilstm")(seq_in)

    st_in = Input(shape=(C.N_DIV,), name="static")
    x = Concatenate()([x, st_in])

    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout)(x)
    out = Dense(1)(x)

    model = Model([seq_in, st_in], out)
    model.compile(optimizer=Adam(lr), loss=Huber(delta=C.HUBER_DELTA))
    return model


# ---- CEEMDAN-BiLSTM (copied unchanged from ceemdan_lstm_forecast.py) --------
@tf.keras.utils.register_keras_serializable(package="p1")
class TemporalAttention(Layer):
    """Additive (Bahdanau-style) temporal attention pooling over a sequence.
    Input (batch, T, units) -> output (batch, units)."""
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

    def call(self, x):                                   # x: (B, T, U)
        score = tf.tanh(tf.tensordot(x, self.W, axes=[[2], [0]]) + self.b)
        score = tf.tensordot(score, self.v, axes=[[2], [0]])
        weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(x * weights, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def _pool_stream(seq_tensor, lstm_units, dropout, name):
    """BiLSTM (return_sequences) -> temporal attention -> LayerNorm (as tuned)."""
    h = Bidirectional(LSTM(lstm_units, dropout=dropout, return_sequences=True),
                      name=f"{name}_bilstm")(seq_tensor)
    pooled = TemporalAttention(name=f"{name}_attn")(h)
    pooled = LayerNormalization(name=f"{name}_ln")(pooled)
    return pooled


def build_ceemdan_bilstm(T, seq_feats, lstm_units, dense_units, dropout, lr):
    """Dual stream: raw sequence + CEEMDAN IMF channels, each BiLSTM + attention,
    concatenated with the static division one-hots -> dense head -> delta."""
    raw_in = Input(shape=(T, seq_feats), name="seq_input")
    imf_in = Input(shape=(T, C.N_IMFS), name="imf_input")
    raw_vec = _pool_stream(raw_in, lstm_units, dropout, name="raw")
    imf_vec = _pool_stream(imf_in, lstm_units, dropout, name="imf")
    merged = Concatenate(name="stream_concat")([raw_vec, imf_vec])
    static_in = Input(shape=(C.N_DIV,), name="static_input")
    merged = Concatenate(name="static_concat")([merged, static_in])
    x = Dense(dense_units, activation="relu")(merged)
    x = Dropout(dropout)(x)
    out = Dense(1)(x)
    model = Model([raw_in, imf_in, static_in], out)
    model.compile(optimizer=Adam(learning_rate=lr), loss=Huber(delta=C.HUBER_DELTA))
    return model


BUILDERS = {
    "CNN-BiLSTM":     build_cnn_bilstm,
    "CNN":            build_cnn,
    "BiLSTM":         build_bilstm,
    "CEEMDAN-BiLSTM": build_ceemdan_bilstm,
}


# =============================================================================
# shared training + evaluation
# =============================================================================

def make_inputs(X, imf=None):
    """Split the 18-channel window into the sequence branch (channels 0-9) and
    the static division one-hots (channels 10-17, constant along time).
    CEEMDAN-BiLSTM additionally takes the IMF block as its second input."""
    if imf is None:
        return [X[:, :, C.SEQ_SLICE], X[:, 0, C.STATIC_SLICE]]
    return [X[:, :, C.SEQ_SLICE], imf, X[:, 0, C.STATIC_SLICE]]


def ceemdan_inputs(Xtr, Xte, ts, lead, log=print):
    """IMF blocks for the train and test windows (computed once, then cached).
    The decomposition does not depend on the model seed."""
    tag = f"SPEI{ts}_lead{lead}_{C.PIPELINE}"
    kw = dict(cache_dir=C.CEEMDAN_CACHE_DIR, n_imfs=C.N_IMFS,
              trials=C.CEEMDAN_TRIALS, seed=C.CEEMDAN_SEED, log=log)
    imf_tr = CF.decompose_block(Xtr[:, :, C.SPEI_CHANNEL], tag=f"{tag}_train", **kw)
    imf_te = CF.decompose_block(Xte[:, :, C.SPEI_CHANNEL], tag=f"{tag}_test", **kw)
    return imf_tr, imf_te


def train_and_evaluate(model_name, ts, lead, params, seed=None,
                       save_model=True, verbose=True, model_path=None):
    """Train one (model, timescale, lead) and score it on the held-out test set.

    `seed` defaults to common.SEED, so run3-run6 behave exactly as before.
    run8 varies it to measure how stable each result is across initialisations.

    Returns (metrics_dict, predictions_dataframe_fields)."""
    seed = C.SEED if seed is None else seed
    C.set_all_seeds(seed)

    d = C.load_arrays(ts, lead, "3D")
    Xtr, ytr, ptr = d["X_train"], d["y_train"], d["persist_train"]
    Xte, yte, pte = d["X_test"],  d["y_test"],  d["persist_test"]
    dtr = d["target_date_train"]
    dte, vte = d["target_date_test"], d["division_test"]

    # lag-depth sensitivity result: SPEI-6 keeps only the last 6 timesteps
    if ts in C.WINDOW_OVERRIDE:
        w = C.WINDOW_OVERRIDE[ts]
        Xtr, Xte = Xtr[:, -w:, :], Xte[:, -w:, :]

    T = Xtr.shape[1]
    seq_feats = C.SEQ_SLICE.stop - C.SEQ_SLICE.start

    # ---- inner fit / validation split, by TARGET year ----
    yr_tr   = C.year_of(dtr)
    val_m   = yr_tr >= C.VAL_START_YEAR_DL
    fit_m   = ~val_m
    if fit_m.sum() == 0 or val_m.sum() == 0:
        raise RuntimeError(
            f"{C.describe_run(model_name, ts, lead)}: empty fit/val split "
            f"(VAL_START_YEAR_DL={C.VAL_START_YEAR_DL})."
        )

    # ---- CEEMDAN-BiLSTM: IMF channels of each window's SPEI history ----
    imf_tr = imf_te = None
    if model_name == "CEEMDAN-BiLSTM":
        imf_tr, imf_te = ceemdan_inputs(Xtr, Xte, ts, lead, log=print)
        C.set_all_seeds(seed)          # decomposition must not disturb the seed

    # ---- delta targets on BOTH fit and val, so early stopping stays meaningful
    ytr_delta = ytr - ptr
    Xin  = make_inputs(Xtr, imf_tr)
    Xfit = [a[fit_m] for a in Xin]
    Xval = [a[val_m] for a in Xin]
    yfit, yval = ytr_delta[fit_m], ytr_delta[val_m]

    if verbose:
        print(f"    window T={T}  channels={seq_feats}+{C.N_DIV}  "
              f"fit={fit_m.sum()}  val={val_m.sum()}  test={len(yte)}")

    # ---- build with the lead-1 tuned hyperparameters ----
    # every builder takes (T, seq_feats, **its own tuned params)
    model = BUILDERS[model_name](T=T, seq_feats=seq_feats, **params)

    es = EarlyStopping(monitor="val_loss", patience=C.PATIENCE_FINAL,
                       restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=C.RLR_FACTOR,
                            patience=C.RLR_PATIENCE, min_lr=C.RLR_MIN_LR, verbose=0)

    hist = model.fit(Xfit, yfit, validation_data=(Xval, yval),
                     epochs=C.EPOCHS_CAP, batch_size=C.BATCH_SIZE,
                     verbose=0, callbacks=[es, rlr],
                     shuffle=C.SHUFFLE_BY_MODEL[model_name])
    stop_epoch = len(hist.history["loss"])

    # ---- predict the delta, then add persistence back ----
    pred_delta = model.predict(make_inputs(Xte, imf_te), verbose=0).ravel()
    final_pred = pred_delta + pte

    met = C.compute_metrics(yte, final_pred, pte)
    evt = C.event_metrics(yte, final_pred)

    # run8 trains many seeds per cell; saving all of them just clutters the tree
    if save_model:
        path = model_path if model_path is not None else \
            C.model_weights_path(model_name, ts, lead, "keras")
        model.save(path)

    row = dict(model=model_name, timescale=ts, lead=lead, seed=seed,
               **{k: met[k] for k in ["RMSE", "MAE", "NSE", "WI", "R2", "Skill",
                                      "RMSE_persistence"]},
               **{k: evt[k] for k in ["POD", "FAR", "CSI", "SR",
                                      "hits", "misses", "false_alarms",
                                      "n_observed_events"]},
               stop_epoch=stop_epoch, window=T,
               shuffle=C.SHUFFLE_BY_MODEL[model_name], **params)

    preds = dict(target_date=dte, division=vte,
                 obs=yte, pred=final_pred, persist=pte)
    return row, preds


def run_model_across_leads(model_name, ts, params_by_ts):
    """Train one model at one timescale across every lead in common.LEADS."""
    C.ensure_dirs()
    C.banner(f"{model_name}  --  SPEI-{ts}  --  leads {C.LEADS}")

    print(f"TensorFlow {tf.__version__} | devices: "
          f"{[dev.device_type for dev in tf.config.list_physical_devices()]}")

    rows = []
    for lead in C.LEADS:
        print(f"\n  [{C.describe_run(model_name, ts, lead)}]")
        params = dict(params_by_ts[ts])

        row, preds = train_and_evaluate(model_name, ts, lead, params)

        path = C.save_predictions(model_name, ts, lead,
                                  preds["target_date"], preds["division"],
                                  preds["obs"], preds["pred"], preds["persist"])
        C.append_metrics_row(model_name, row)
        rows.append(row)

        print(f"    RMSE={row['RMSE']:.4f}  NSE={row['NSE']:.4f}  "
              f"Skill={row['Skill']:+.4f}  POD={row['POD']:.3f}  CSI={row['CSI']:.3f}")
        print(f"    stopped at epoch {row['stop_epoch']}  ->  {path.name}")

        # lead-1 regression check, reported the moment it is known
        ref = C.REFERENCE_LEAD1.get((model_name, ts))
        if lead == 1 and ref:
            print("\n    --- LEAD-1 REGRESSION CHECK "
                  "(must reproduce the published numbers) ---")
            worst = 0.0
            for k in ["RMSE", "NSE", "Skill"]:
                diff = abs(row[k] - ref[k])
                worst = max(worst, diff)
                flag = "   OK" if diff <= C.TOL_DL else "   <-- MISMATCH"
                print(f"      {k:<6} got {row[k]:+.6f}  "
                      f"published {ref[k]:+.6f}  |diff|={diff:.2e}{flag}")
            if worst > C.TOL_DL:
                print("      *** The refactored pipeline did NOT reproduce lead 1. ***")
                print("      *** Treat leads 2 and 3 as untrustworthy until resolved. ***")
            else:
                print(f"      PASS (worst |diff| = {worst:.2e})")
            print()

    print(f"\n{model_name} done. Metrics -> {C.model_metrics_path(model_name)}")
    return rows
