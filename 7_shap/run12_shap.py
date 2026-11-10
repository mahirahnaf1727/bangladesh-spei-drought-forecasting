"""
================================================================================
run12_shap.py  --  SHAP for the selected models  (Stage 2)
================================================================================
Same method as the thesis SHAP script (seed 42 only; not included):
  * shap.GradientExplainer, background = 150 random TRAINING windows (seed 42),
    explained = every test window (2019-2024);
  * per variable: signed SHAP summed over the time steps of the window, then
    mean |value| (importance) and mean signed value (direction);
  * month sin/cos dropped from the output, as before.
What changed: it explains the Stage 2 models of the models chosen by the
selection rule fixed in advance, for ALL FIVE SEEDS, and reports each variable's
importance as mean +/- SD across seeds (so Fig. 10a-c carries the same seed
uncertainty as the metrics). Raw per-sample values are saved for every seed so
the beeswarms (Fig. 10d-f) can be drawn by 8_figures. No figures are made here.

Because the deep-learning models predict the CHANGE from persistence, SHAP
explains that change: what each input adds to or takes from persistence.

RUN ORDER: after run10 --set B (needs the lead-1 models of all seeds).
OUTPUT:    OUT_ROOT/_shap/shap_importance_<model>_SPEI<ts>_seed<s>.csv
           OUT_ROOT/_shap/shap_values_<model>_SPEI<ts>_seed<s>.npz
           OUT_ROOT/_shap/shap_importance_selected.csv   (mean +/- SD over seeds)
================================================================================
"""

import gc
import json

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "5_forecasting"))
import common as C  # noqa: E402

N_BACKGROUND = 150
RANDOM_SEED = 42
DROP = {"month_sin", "month_cos"}
PRETTY = {
    "tp_lag1": "Total Precipitation", "mx2t_lag1": "Maximum Temperature",
    "mn2t_lag1": "Minimum Temperature", "ssrd_lag1": "Surface Solar Radiation Downwards",
    "tcc_lag1": "Total Cloud Cover", "wind_lag1": "Wind Speed",
    "d2m_lag1": "Dewpoint Temperature", "spei_lag0": "SPEI",
}


def _squeeze(a):
    a = np.asarray(a)
    return a[..., 0] if (a.ndim >= 3 and a.shape[-1] == 1) else a


def explain(model_name, ts, seed):
    import tensorflow as tf
    import shap
    import dl_models

    path = C.run_model_path(model_name, ts, 1, seed, "keras")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run run10 --set B first.")
    model = tf.keras.models.load_model(path, compile=False)

    d = C.load_arrays(ts, 1, "3D")
    Xtr, Xte = d["X_train"], d["X_test"]
    if ts in C.WINDOW_OVERRIDE:
        w = C.WINDOW_OVERRIDE[ts]
        Xtr, Xte = Xtr[:, -w:, :], Xte[:, -w:, :]
    imf_tr = imf_te = None
    if model_name == "CEEMDAN-BiLSTM":
        imf_tr, imf_te = dl_models.ceemdan_inputs(Xtr, Xte, ts, 1)
    tr_in = dl_models.make_inputs(Xtr.astype("float32"), imf_tr)
    te_in = dl_models.make_inputs(Xte.astype("float32"), imf_te)

    rng = np.random.default_rng(RANDOM_SEED)
    bg_idx = rng.choice(len(Xtr), size=min(N_BACKGROUND, len(Xtr)), replace=False)
    bg = [a[bg_idx] for a in tr_in]

    explainer = shap.GradientExplainer(model, bg)
    sv = explainer.shap_values(te_in)
    if isinstance(sv, list) and len(sv) == 1 and isinstance(sv[0], list):
        sv = sv[0]
    sv_seq = _squeeze(sv[0])                         # (n, T, 10)

    meta = json.loads(C.meta_path(ts, 1).read_text())
    names = meta["timestep_feature_names_3d"][:sv_seq.shape[-1]]
    shap_2d = sv_seq.sum(axis=1)                     # signed, summed over time
    feat_2d = te_in[0].mean(axis=1)
    keep = [i for i, n in enumerate(names) if n not in DROP]
    labels = [PRETTY.get(names[i], names[i]) for i in keep]
    shap_2d, feat_2d = shap_2d[:, keep], feat_2d[:, keep]

    if model_name == "CEEMDAN-BiLSTM":               # IMF stream, reported as one row
        imf_sv = _squeeze(sv[1]).sum(axis=(1, 2))[:, None]
        shap_2d = np.hstack([shap_2d, imf_sv])
        feat_2d = np.hstack([feat_2d, np.full((len(feat_2d), 1), np.nan)])
        labels = labels + ["SPEI IMFs (CEEMDAN)"]

    out = C.OUT_ROOT / "_shap"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"model": model_name, "timescale": ts, "seed": seed, "variable": labels,
                       "mean_abs_shap": np.abs(shap_2d).mean(axis=0),
                       "mean_signed_shap": shap_2d.mean(axis=0)}
                      ).sort_values("mean_abs_shap", ascending=False)
    df.to_csv(out / f"shap_importance_{model_name}_SPEI{ts}_seed{seed}.csv", index=False,
              float_format="%.6f")
    np.savez_compressed(out / f"shap_values_{model_name}_SPEI{ts}_seed{seed}.npz",
                        shap_2d=shap_2d, feat_2d=feat_2d, labels=np.array(labels))
    del explainer, model
    tf.keras.backend.clear_session()
    gc.collect()
    return df


def main():
    import p1_selection as S
    C.banner("RUN12  --  SHAP for the selected models (lead 1, all seeds)")
    selected, report, _ = S.apply_rule(S.summarise(S.collect_runs(lead=1)))
    for line in report:
        print("  " + line)
    frames = []
    for ts, model_name in selected.items():
        if model_name == "XGBoost":
            print(f"  SPEI-{ts}: XGBoost selected -- GradientExplainer does not apply; "
                  "(TreeExplainer would be used instead).")
            continue
        for seed in C.SEEDS:
            f = C.OUT_ROOT / "_shap" / f"shap_importance_{model_name}_SPEI{ts}_seed{seed}.csv"
            if f.exists():
                print(f"  {model_name} SPEI-{ts} seed {seed}: already done")
                frames.append(pd.read_csv(f))
                continue
            print(f"\n  explaining {model_name} at SPEI-{ts}, seed {seed} ...")
            df = explain(model_name, ts, seed)
            print(df.head(3).to_string(index=False))
            frames.append(df)
    if frames:
        allr = pd.concat(frames)
        summ = (allr.groupby(["model", "timescale", "variable"])
                .agg(mean_abs_shap=("mean_abs_shap", "mean"),
                     sd_abs_shap=("mean_abs_shap", lambda v: v.std(ddof=1)),
                     mean_signed_shap=("mean_signed_shap", "mean"),
                     n_seeds=("seed", "count"))
                .reset_index().sort_values(["timescale", "mean_abs_shap"],
                                           ascending=[True, False]))
        summ.to_csv(C.OUT_ROOT / "_shap" / "shap_importance_selected.csv",
                    index=False, float_format="%.6f")
        print("\n" + summ.round(4).to_string(index=False))
    print(f"\nDone. Files in {C.OUT_ROOT / '_shap'}")


if __name__ == "__main__":
    main()
