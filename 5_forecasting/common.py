"""
================================================================================
common.py  --  shared configuration and helpers  (Paper 1, Stage 2 version)
================================================================================
STAGE 2 CHANGES (15 Oct 2026) -- everything else in this file is unchanged
  * Paths now come from p1_config.py (one place for all folders).
  * The "strict" pipeline (F1+F2 sensitivity) writes to its own sub-folder.
  * CEEMDAN-BiLSTM added (tuned values from its best_params_FULL.csv).
  * SEEDS, per-run file names and a per-model batch-shuffle setting added.
  * DESEASON_REF added: "all" years (main) or "train" years only (strict).
--------------------------------------------------------------------------------
Everything that more than one script needs lives here, so the LEAD change is
made in exactly ONE place instead of four.

WHAT THE EXTENSION DOES
  The published pipeline forecasts SPEI at a 1-month lead. This extension runs
  the SAME pipeline at leads 2 and 3 to measure how fast genuine skill (against
  a persistence baseline) decays.

  Only the selected model per timescale is run, plus XGBoost as the conventional
  reference:
      SPEI-3   ->  CNN-BiLSTM
      SPEI-6   ->  BiLSTM
      SPEI-12  ->  CNN
      all      ->  XGBoost

  Lead 1 is INCLUDED in the run list on purpose: re-running it through the
  refactored pipeline is the regression test. If lead-1 output does not
  reproduce the published numbers, nothing downstream is trustworthy.

THE EXPECTED RESULT  (read this before you call anything a bug)
  The lead-lag screening found predictor correlation collapses at roughly the
  accumulation length:
      SPEI-3   precip r ~0.50-0.53 at leads 1-2, collapsing to ~0.07 at lead 3
      SPEI-6   r ~0.38-0.45 holds out to lead 5
      SPEI-12  r ~0.31-0.33 holds out to lead 11
  So SPEI-3 at lead 3 SHOULD lose most or all of its skill against persistence.
  A near-zero or negative skill score there is the finding, not a failure.

WHAT CHANGES WITH LEAD, AND NOTHING ELSE
  target          :  y[t + LEAD]        (was y[t+1])
  delta target    :  d = y[t+LEAD] - y[t]      (deep-learning models only)
  reconstruction  :  yhat = dhat + y[t]
  persistence     :  still y[t]  -- carry the last OBSERVED value forward LEAD
                     months. This is DIRECT multi-step forecasting; one-step
                     forecasts are never chained.
================================================================================
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. PATHS  --  all folders are set in p1_config.py (edit them there, not here)
# =============================================================================

import p1_config as CFG

PIPELINE = CFG.PIPELINE                     # "main" or "strict"

if PIPELINE == "main":
    SPEI_CSV     = CFG.SPEI_CSV              # thesis SPEI (fitted on 1980-2024)
    OUT_ROOT     = CFG.OUT_ROOT
    DESEASON_REF = "all"                     # thesis: calendar-month means, all years
else:                                        # F1 + F2 sensitivity
    SPEI_CSV     = str(CFG.SPEI_REFIT_CSV)   # SPEI fitted on 1980-2018 only
    OUT_ROOT     = CFG.OUT_ROOT / "_strict"
    DESEASON_REF = "train"                   # calendar-month means, 1980-2018 only

VARS_CSV = CFG.VARS_CSV

# --- derived folders ---------------------------------------------------------
# Intermediates are prefixed with "_" so the four model folders sort first.
FEATURE_DIR = OUT_ROOT / "_feature_matrix"
PREP_DIR    = OUT_ROOT / "_data_prep"
ARRAY_DIR   = PREP_DIR / "arrays"
SCALER_DIR  = PREP_DIR / "scalers"
META_DIR    = PREP_DIR / "meta"
LOG_DIR     = OUT_ROOT / "_logs"
FIG_DIR     = OUT_ROOT / "_figures"
TABLE_DIR   = OUT_ROOT / "_tables"

# One folder per model, named exactly as the model is named in the manuscript.
MODEL_DIRS = {
    "CNN-BiLSTM":     OUT_ROOT / "CNN-BiLSTM",
    "BiLSTM":         OUT_ROOT / "BiLSTM",
    "CNN":            OUT_ROOT / "CNN",
    "XGBoost":        OUT_ROOT / "XGBoost",
    "CEEMDAN-BiLSTM": OUT_ROOT / "CEEMDAN-BiLSTM",
}
CEEMDAN_CACHE_DIR = OUT_ROOT / "_ceemdan_cache"   # IMF decompositions, computed once

ALL_DIRS = [OUT_ROOT, FEATURE_DIR, PREP_DIR, ARRAY_DIR, SCALER_DIR, META_DIR,
            LOG_DIR, FIG_DIR, TABLE_DIR, CEEMDAN_CACHE_DIR] + list(MODEL_DIRS.values())


def ensure_dirs():
    """Create the whole output tree. Safe to call repeatedly."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    # each model folder gets a models/ subfolder for saved weights
    for name, d in MODEL_DIRS.items():
        (d / "models").mkdir(parents=True, exist_ok=True)
        (d / "runs").mkdir(parents=True, exist_ok=True)
    return OUT_ROOT


# =============================================================================
# 2. THE LEAD SETTING  --  the one knob this whole extension turns
# =============================================================================

# Lead 1 is included so it can serve as the regression test (see run7).
LEADS = [1, 2, 3]

TIMESCALES = [3, 6, 12]          # SPEI-1 excluded: too noisy (thesis decision)

# Which model is the selected one at each timescale.
SELECTED_MODEL = {3: "CNN-BiLSTM", 6: "BiLSTM", 12: "CNN"}

# Models plotted / tabulated, in display order (brief section 5).
MODELS = ["XGBoost", "CNN", "BiLSTM", "CNN-BiLSTM", "CEEMDAN-BiLSTM"]
DL_MODELS = ["CNN", "BiLSTM", "CNN-BiLSTM", "CEEMDAN-BiLSTM"]

# Stage 2: five seeds for every reported number. 42 is the thesis seed.
SEEDS = [42, 43, 44, 45, 46]


# =============================================================================
# 3. LOCKED PIPELINE SETTINGS  (copied verbatim from the thesis scripts)
# =============================================================================

CLIMATE_DRIVERS = ["tp", "mx2t", "mn2t", "ssrd", "tcc", "wind", "d2m"]
CLIMATE_MAXLAG  = 3
CLIMATE_LAGS    = [1, 2, 3]

# Autoregressive own-SPEI lag depth per timescale (from ACF/PACF).
SPEI_LAGS = {1: 1, 3: 7, 6: 13, 12: 13}

TRAIN_END_YEAR  = 2018   # training targets: target year <= this
TEST_START_YEAR = 2019   # test targets:     target year >= this

# Inner fit/validation split, applied to TRAIN rows only, by target year.
# NOTE the two models use different values -- this is inherited from the thesis
# scripts and is kept so lead-1 reproduces exactly.
VAL_START_YEAR_DL  = 2014    # deep-learning models: val = 2014-2018
VAL_START_YEAR_XGB = 2017    # XGBoost:              val = 2017-2018

# Deep-learning training configuration (unchanged).
SEED            = 42
BATCH_SIZE      = 32
EPOCHS_CAP      = 200
PATIENCE_FINAL  = 40
RLR_FACTOR      = 0.5
RLR_PATIENCE    = 15
RLR_MIN_LR      = 1e-5
HUBER_DELTA     = 1.0
USE_BATCHNORM   = True

# 3D channel layout (18 channels):
#   0-6   climate anomalies (*_lag1)
#   7     spei_lag0  (SPEI at time t)
#   8-9   month sin / cos
#   10-17 division one-hots
SEQ_SLICE    = slice(0, 10)
STATIC_SLICE = slice(10, 18)
N_DIV        = 8

# Lag-depth sensitivity result: SPEI-6 uses only the last 6 timesteps.
WINDOW_OVERRIDE = {6: 6}

# Batch shuffling during training. The thesis scripts were NOT consistent:
#   cnn_forecast.py, cnn_lstm_forecast.py, ceemdan_lstm_forecast.py -> shuffle=False
#   prove_improvement.py (the reported BiLSTM)                    -> Keras default (True)
#   the lead-time extension (dl_models.py, all models)            -> Keras default (True)
# Default below = "as tuned": each model is trained exactly the way its
# hyperparameters were tuned. Decision recorded in the Decision Log (Stage 2).
SHUFFLE_BY_MODEL = {
    "CNN":            False,
    "CNN-BiLSTM":     False,
    "CEEMDAN-BiLSTM": False,
    "BiLSTM":         True,
}

# CEEMDAN front-end (copied unchanged from ceemdan_lstm_forecast.py)
SPEI_CHANNEL   = 7       # the SPEI-so-far channel is the one decomposed
N_IMFS         = 4       # fixed IMF channels (zero-pad / fold surplus into the last)
CEEMDAN_TRIALS = 50      # noise realisations
CEEMDAN_SEED   = 42      # fixed: the decomposition is the same for every model seed

# XGBoost
N_ESTIMATORS_CAP      = 3000
EARLY_STOPPING_ROUNDS = 50

DROUGHT_THRESHOLD = -1.0   # event detection threshold for POD / FAR / CSI


# =============================================================================
# 4. TUNED HYPERPARAMETERS FROM THE LEAD-1 SEARCH
# =============================================================================
# These are REUSED at leads 2 and 3 rather than re-searched. Input shape,
# sequence length and channel layout are identical across leads; only the target
# moves. The three-seed stability test already showed metric variation across
# seeds is comparable to the tuning gain, so a fresh 20-trial search per lead is
# not warranted. State this in the methods section.

# CNN-BiLSTM -- from best_params_FULL_CNN-BiLSTM.csv
PARAMS_CNN_BILSTM = {
    3:  dict(filters=112, kernel_size=4, lstm_units=96,  dense_units=40,
             dropout=0.4307459503277717, lr=0.00017279938772027972),
    6:  dict(filters=80,  kernel_size=2, lstm_units=16,  dense_units=64,
             dropout=0.4862528132298237, lr=0.004138040112561018),
    12: dict(filters=96,  kernel_size=4, lstm_units=112, dense_units=40,
             dropout=0.4185848262139801, lr=0.0002049120916321119),
}

# CNN -- from best_params_FULL_CNN.csv
PARAMS_CNN = {
    3:  dict(filters=16, kernel_size=5, conv2_filters=80,  dense_units=48,
             dropout=0.10823379771832098, lr=0.008706020878304856),
    6:  dict(filters=96, kernel_size=3, conv2_filters=64,  dense_units=32,
             dropout=0.2800778743737434,  lr=0.0003593924490912408),
    12: dict(filters=32, kernel_size=4, conv2_filters=112, dense_units=24,
             dropout=0.10993392039995764, lr=0.001660250153775578),
}

# BiLSTM -- the TUNED_PARAMS block from prove_improvement.py.
# IMPORTANT: the published BiLSTM numbers come from prove_improvement.py, whose
# architecture has NO recurrent_dropout (the "speed fix"), NOT from
# lstm_forecast.py, which does. Using the wrong one will not reproduce lead 1.
PARAMS_BILSTM = {
    3:  dict(lstm_units=16,  dense_units=48, dropout=0.4290, lr=0.003810),
    6:  dict(lstm_units=16,  dense_units=64, dropout=0.4948, lr=0.002713),
    12: dict(lstm_units=128, dense_units=32, dropout=0.4878, lr=0.001904),
}

# CEEMDAN-BiLSTM -- from Models/CEEMDAN_LSTM/best_params_FULL.csv (full precision)
PARAMS_CEEMDAN = {
    3:  dict(lstm_units=32, dense_units=48, dropout=0.43439799268831475,
             lr=0.005893431331850884),
    6:  dict(lstm_units=48, dense_units=40, dropout=0.2727780074568463,
             lr=0.0003823475224675188),
    12: dict(lstm_units=32, dense_units=48, dropout=0.42898562024038517,
             lr=0.0040192186371608),
}

# XGBoost -- FULL rows from best_params_Xgboost.csv
# NOTE: that file stores the tuned values rounded to 4 decimals, which is why
# lead-1 XGBoost differs slightly from the thesis run (e.g. SPEI-6 skill 0.164
# vs 0.152). The rounded values are what can be reproduced, so they are used.
PARAMS_XGB = {
    3:  dict(max_depth=3, learning_rate=0.0441, subsample=0.6138,
             colsample_bytree=0.9637, min_child_weight=3,
             reg_lambda=0.4468, reg_alpha=0.0177, gamma=2.6003),
    6:  dict(max_depth=3, learning_rate=0.1716, subsample=0.9863,
             colsample_bytree=0.9234, min_child_weight=3,
             reg_lambda=0.0025, reg_alpha=0.5457, gamma=2.2008),
    12: dict(max_depth=3, learning_rate=0.1151, subsample=0.8827,
             colsample_bytree=0.8916, min_child_weight=7,
             reg_lambda=0.0020, reg_alpha=0.0272, gamma=0.5793),
}


PARAMS_BY_MODEL = {
    "CNN-BiLSTM":     PARAMS_CNN_BILSTM,
    "CNN":            PARAMS_CNN,
    "BiLSTM":         PARAMS_BILSTM,
    "CEEMDAN-BiLSTM": None,   # filled in just below (defined after this block)
    "XGBoost":        None,
}


# =============================================================================
# 5. PUBLISHED LEAD-1 VALUES  --  the regression-test reference
# =============================================================================
# Taken from the thesis metrics CSVs. run7 compares the freshly computed lead-1
# results against these. Any mismatch means the refactor changed behaviour.

REFERENCE_LEAD1 = {
    ("CNN-BiLSTM", 3):  dict(RMSE=0.621006, NSE=0.601883, Skill=0.418833),
    ("BiLSTM",     6):  dict(RMSE=0.496600, NSE=0.710900, Skill=0.335900),
    ("CNN",       12):  dict(RMSE=0.281777, NSE=0.875270, Skill=0.169962),
    ("XGBoost",    3):  dict(RMSE=0.644700, NSE=0.571000, Skill=0.373700),
    ("XGBoost",    6):  dict(RMSE=0.561000, NSE=0.631100, Skill=0.152500),
    ("XGBoost",   12):  dict(RMSE=0.323200, NSE=0.835900, Skill=-0.092200),
}

# XGBoost is deterministic, so it should match tightly. Deep-learning runs
# involve GPU non-determinism and TensorFlow version differences, so a looser
# tolerance is realistic. These are reporting thresholds, not pass/fail gates
# you should silently ignore -- see run7's printed verdict.
TOL_XGB = 1e-3
TOL_DL  = 5e-3

PARAMS_BY_MODEL["CEEMDAN-BiLSTM"] = PARAMS_CEEMDAN
PARAMS_BY_MODEL["XGBoost"] = PARAMS_XGB

# Thesis seed-42 lead-1 values for EVERY model (from Models/Model Files/*/metrics),
# used by run10 to check that seed 42 reproduces the thesis run.
THESIS_LEAD1_SEED42 = {
    ("XGBoost", 3):  dict(RMSE=0.6447, NSE=0.5710, Skill=0.3737),
    ("XGBoost", 6):  dict(RMSE=0.5610, NSE=0.6311, Skill=0.1525),
    ("XGBoost", 12): dict(RMSE=0.3232, NSE=0.8359, Skill=-0.0922),
    ("CNN", 3):  dict(RMSE=0.705634, NSE=0.485982, Skill=0.249642),
    ("CNN", 6):  dict(RMSE=0.506858, NSE=0.698880, Skill=0.308269),
    ("CNN", 12): dict(RMSE=0.281777, NSE=0.875270, Skill=0.169962),
    ("BiLSTM", 3):  dict(RMSE=0.6457, NSE=0.5696, Skill=0.3717),
    ("BiLSTM", 6):  dict(RMSE=0.4966, NSE=0.7109, Skill=0.3359),
    ("BiLSTM", 12): dict(RMSE=0.3109, NSE=0.8482, Skill=-0.0102),
    ("CNN-BiLSTM", 3):  dict(RMSE=0.621006, NSE=0.601883, Skill=0.418833),
    ("CNN-BiLSTM", 6):  dict(RMSE=0.535428, NSE=0.663976, Skill=0.228089),
    ("CNN-BiLSTM", 12): dict(RMSE=0.291397, NSE=0.866608, Skill=0.112320),
    ("CEEMDAN-BiLSTM", 3):  dict(RMSE=0.680010, NSE=0.522636, Skill=0.303148),
    ("CEEMDAN-BiLSTM", 6):  dict(RMSE=0.569016, NSE=0.620496, Skill=0.128206),
    ("CEEMDAN-BiLSTM", 12): dict(RMSE=0.314559, NSE=0.844560, Skill=-0.034400),
}


# =============================================================================
# 6. FILE-NAMING  --  every artefact is keyed by (timescale, lead)
# =============================================================================

def feature_matrix_path(ts, lead):
    return FEATURE_DIR / f"feature_matrix_SPEI{ts}_lead{lead}.csv"

def npz_path(ts, lead, kind):
    """kind is '2D' (XGBoost) or '3D' (deep-learning)."""
    return ARRAY_DIR / f"SPEI{ts}_{kind}_lead{lead}.npz"

def meta_path(ts, lead):
    return META_DIR / f"SPEI{ts}_meta_lead{lead}.json"

def scaler_path(ts, lead):
    return SCALER_DIR / f"SPEI{ts}_scaler_lead{lead}.pkl"

def predictions_path(model, ts, lead):
    return MODEL_DIRS[model] / f"predictions_SPEI{ts}_lead{lead}.csv"

def model_weights_path(model, ts, lead, ext="keras"):
    return MODEL_DIRS[model] / "models" / f"{model}_SPEI{ts}_lead{lead}.{ext}"

def model_metrics_path(model):
    return MODEL_DIRS[model] / f"metrics_{model}_all_leads.csv"


# --- Stage 2: one file per (model, timescale, lead, seed) -------------------
def run_stem(model, ts, lead, seed):
    return f"{model}_SPEI{ts}_lead{lead}_seed{seed}"

def run_predictions_path(model, ts, lead, seed):
    return MODEL_DIRS[model] / "runs" / f"pred_{run_stem(model, ts, lead, seed)}.csv"

def run_metrics_path(model, ts, lead, seed):
    return MODEL_DIRS[model] / "runs" / f"metrics_{run_stem(model, ts, lead, seed)}.json"

def run_model_path(model, ts, lead, seed, ext="keras"):
    return MODEL_DIRS[model] / "models" / f"{run_stem(model, ts, lead, seed)}.{ext}"


# =============================================================================
# 7. METRICS  --  one definition, used everywhere
# =============================================================================
# Identical formulas to the thesis scripts. Previously this function was
# duplicated in four training scripts; it now lives here only.

def compute_metrics(obs, pred, persist):
    """RMSE, MAE, NSE (vs test mean), WI (Willmott d), R2 (Pearson r squared),
    and Skill = 1 - MSE_model / MSE_persistence."""
    obs     = np.asarray(obs, float)
    pred    = np.asarray(pred, float)
    persist = np.asarray(persist, float)

    err  = pred - obs
    obar = obs.mean()

    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obar) ** 2)
    wi_den = np.sum((np.abs(pred - obar) + np.abs(obs - obar)) ** 2)

    mse_m = float(np.mean(err ** 2))
    mse_p = float(np.mean((persist - obs) ** 2))

    return {
        "RMSE":  float(np.sqrt(mse_m)),
        "MAE":   float(np.mean(np.abs(err))),
        "NSE":   float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "WI":    float(1 - ss_res / wi_den) if wi_den > 0 else np.nan,
        "R2":    float(np.corrcoef(pred, obs)[0, 1] ** 2)
                 if (pred.std() > 0 and obs.std() > 0) else np.nan,
        "Skill": float(1 - mse_m / mse_p) if mse_p > 0 else np.nan,
        "MSE_persistence": mse_p,
        "RMSE_persistence": float(np.sqrt(mse_p)),
    }


def event_metrics(obs, pred, threshold=DROUGHT_THRESHOLD):
    """Treat the forecast as a drought DETECTOR and score the 2x2 contingency.

        POD = hits / (hits + misses)                 higher better
        FAR = false alarms / (hits + false alarms)   lower  better
        CSI = hits / (hits + misses + false alarms)  higher better
        SR  = 1 - FAR                                higher better
    """
    obs  = np.asarray(obs, float)
    pred = np.asarray(pred, float)

    o_evt = obs  <= threshold
    p_evt = pred <= threshold

    hits    = int(np.sum(o_evt & p_evt))
    misses  = int(np.sum(o_evt & ~p_evt))
    false_a = int(np.sum(~o_evt & p_evt))
    corr_n  = int(np.sum(~o_evt & ~p_evt))

    pod = hits / (hits + misses)  if (hits + misses)  > 0 else np.nan
    far = false_a / (hits + false_a) if (hits + false_a) > 0 else np.nan
    csi = hits / (hits + misses + false_a) if (hits + misses + false_a) > 0 else np.nan

    return {
        "hits": hits, "misses": misses,
        "false_alarms": false_a, "correct_negatives": corr_n,
        "n_observed_events": hits + misses,
        "POD": float(pod) if pod == pod else np.nan,
        "FAR": float(far) if far == far else np.nan,
        "CSI": float(csi) if csi == csi else np.nan,
        "SR":  float(1 - far) if far == far else np.nan,
    }


# =============================================================================
# 8. SEEDING
# =============================================================================

def set_all_seeds(seed=SEED):
    """Seed every source of randomness, and ask TensorFlow for deterministic ops.
    Call this immediately before building a model, not once at import."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ImportError:
        pass


# =============================================================================
# 9. ARRAY LOADING + PREDICTION SAVING
# =============================================================================

def load_arrays(ts, lead, kind):
    path = npz_path(ts, lead, kind)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}.\nRun run1_build_matrices.py then run2_data_prep.py first."
        )
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    z.close()
    return d


def save_predictions(model, ts, lead, target_date, division, obs, pred, persist,
                     path=None):
    """Write one tidy prediction CSV. Column names match the thesis convention so
    the existing comparison / event-scoring scripts can read these unchanged.
    Stage 2: `path` lets the seed runner write one file per seed."""
    df = pd.DataFrame({
        "target_date": target_date,
        "division":    division,
        "observed":    np.asarray(obs, float),
        "predicted":   np.asarray(pred, float),
        "persistence": np.asarray(persist, float),
    })
    path = predictions_path(model, ts, lead) if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    return path


def append_metrics_row(model, row):
    """Accumulate one metrics row per (timescale, lead) into the model's folder."""
    path = model_metrics_path(model)
    df_new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        # replace any existing row for this (timescale, lead)
        keep = ~((old["timescale"] == row["timescale"]) & (old["lead"] == row["lead"]))
        df_new = pd.concat([old[keep], df_new], ignore_index=True)
    df_new = df_new.sort_values(["timescale", "lead"]).reset_index(drop=True)
    df_new.to_csv(path, index=False, float_format="%.6f")
    return path


# =============================================================================
# 10. FIGURE STYLE  --  Times New Roman, large type, 400 dpi
# =============================================================================

DPI = 400

# Okabe-Ito colour-blind-safe palette. Identical to the model-comparison script,
# so lead-extension figures sit beside the existing ones without a colour clash.
MODEL_COLORS = {
    "XGBoost":        "#E69F00",   # orange
    "CNN":            "#009E73",   # bluish green
    "BiLSTM":         "#0072B2",   # blue
    "CNN-BiLSTM":     "#D55E00",   # vermillion
    "CEEMDAN-BiLSTM": "#4D4D4D",   # charcoal grey
}

# Lead time is drawn with line style + marker, so the figures survive greyscale
# printing and the model colour stays free to mean "model".
LEAD_STYLES = {
    1: dict(linestyle="-",  marker="o", markersize=11),
    2: dict(linestyle="--", marker="s", markersize=10),
    3: dict(linestyle=":",  marker="^", markersize=11),
}

OBS_COLOR = "#000000"
PERSIST_COLOR = "#B22222"


def check_times_new_roman():
    """Report whether a Times-compatible font is actually available. On Windows
    with Office installed it will be; matplotlib falls back to DejaVu Serif if not."""
    from matplotlib import font_manager
    try:
        fp = font_manager.FontProperties(family="Times New Roman")
        try:
            fname = font_manager.findfont(fp, fallback_to_default=False)
        except TypeError:
            fname = font_manager.findfont(fp)
        resolved = font_manager.FontProperties(fname=fname).get_name()
        if "Times" in resolved:
            print(f"[i] Times New Roman available ({resolved})")
            return True
        print(f"[!] Requested Times New Roman; matplotlib resolved to {resolved}")
        return False
    except Exception as exc:
        print(f"[!] Times New Roman not found -- using serif fallback. {exc}")
        return False


def set_style():
    """Journal figure defaults: Times New Roman, enlarged type, 400 dpi output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    check_times_new_roman()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "dejavuserif",

        # Enlarged relative to the model-comparison script, for readability at
        # journal column width.
        "font.size":        15.0,
        "axes.labelsize":   17.0,
        "axes.titlesize":   18.0,
        "xtick.labelsize":  14.0,
        "ytick.labelsize":  14.0,
        "legend.fontsize":  14.0,

        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.0,
        "axes.grid":         False,
        "grid.color":        "0.75",
        "grid.alpha":        0.30,
        "grid.linewidth":    0.7,

        "lines.linewidth":   2.4,

        "legend.frameon":    True,
        "legend.framealpha": 0.90,
        "legend.edgecolor":  "0.8",
        "legend.fancybox":   False,

        "figure.dpi":        120,
        "savefig.dpi":       DPI,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })


def light_grid(ax, axis="both"):
    ax.grid(True, axis=axis, color="0.75", alpha=0.30, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def save_fig(fig, stem, outdir=None):
    """Save to PNG (400 dpi) and PDF (vector, for the journal)."""
    import matplotlib.pyplot as plt
    outdir = Path(outdir) if outdir is not None else FIG_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{stem}.png"
    pdf = outdir / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"    saved  {png.name}  +  {pdf.name}")
    return png


# =============================================================================
# 11. SMALL SHARED UTILITIES
# =============================================================================

def year_of(date_strings):
    """Extract the year from an array of 'YYYY-MM-DD' strings."""
    return np.array([int(str(d)[:4]) for d in np.asarray(date_strings)])


def banner(text, char="="):
    line = char * 78
    print(f"\n{line}\n{text}\n{line}")


def describe_run(model, ts, lead):
    return f"{model}  SPEI-{ts}  lead {lead}"
