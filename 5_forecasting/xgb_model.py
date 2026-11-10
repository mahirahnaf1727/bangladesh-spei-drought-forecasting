"""
================================================================================
xgb_model.py  --  XGBoost training for one (timescale, lead, seed)
================================================================================
Taken from run6_xgboost.py (lead-time extension). The ONLY change: the random
seed is an argument (it was fixed at common.SEED = 42), so the seed runner can
train seeds 42-46. XGBoost uses the seed for row/column subsampling.

Unchanged from run6:
  * predicts the SPEI level directly (no delta target), scored against
    persistence like every other model;
  * early stopping on target years 2017-2018 (VAL_START_YEAR_XGB);
  * tuned FULL parameters reused at every lead; n_estimators cap 3000.
================================================================================
"""

import numpy as np
import xgboost as xgb

import common as C


def fit_with_early_stopping(params, X_fit, y_fit, X_val, y_val):
    """Unchanged from run6 (handles both XGBoost early-stopping signatures)."""
    try:
        model = xgb.XGBRegressor(**params,
                                 early_stopping_rounds=C.EARLY_STOPPING_ROUNDS,
                                 eval_metric="rmse")
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    except TypeError:
        model = xgb.XGBRegressor(**params, eval_metric="rmse")
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=C.EARLY_STOPPING_ROUNDS, verbose=False)
    return model


def train_and_evaluate(ts, lead, seed, model_path=None):
    d = C.load_arrays(ts, lead, "2D")
    Xtr, ytr, ptr = d["X_train"], d["y_train"], d["persist_train"]
    Xte, yte, pte = d["X_test"],  d["y_test"],  d["persist_test"]
    dtr = d["target_date_train"]
    dte, vte = d["target_date_test"], d["division_test"]

    yr_tr = C.year_of(dtr)
    val_m = yr_tr >= C.VAL_START_YEAR_XGB
    fit_m = ~val_m

    tuned = dict(C.PARAMS_XGB[ts])
    params = dict(tuned, n_estimators=C.N_ESTIMATORS_CAP, random_state=seed,
                  n_jobs=-1, tree_method="hist")

    model = fit_with_early_stopping(params, Xtr[fit_m], ytr[fit_m],
                                    Xtr[val_m], ytr[val_m])
    pred = model.predict(Xte)

    met = C.compute_metrics(yte, pred, pte)
    evt = C.event_metrics(yte, pred)
    best_iter = int(getattr(model, "best_iteration", -1) or -1)
    if model_path is not None:
        model.save_model(str(model_path))

    row = dict(model="XGBoost", timescale=ts, lead=lead, seed=seed,
               **{k: met[k] for k in ["RMSE", "MAE", "NSE", "WI", "R2", "Skill",
                                      "RMSE_persistence"]},
               **{k: evt[k] for k in ["POD", "FAR", "CSI", "SR",
                                      "hits", "misses", "false_alarms",
                                      "n_observed_events"]},
               best_iteration=best_iter, **tuned)
    preds = dict(target_date=dte, division=vte, obs=yte, pred=pred, persist=pte)
    return row, preds
