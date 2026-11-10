"""
p1_paper_tables.py  --  rebuilds every CSV behind the tables of the paper and the SI.

Sources (all in P1_Data unless noted):
  * the 135 run prediction files and their metric/metadata files   stage2_output/<model>/runs/
  * the 5 strict-pipeline runs (Table S18B)                          stage2_output/_strict/
  * the SHAP value files (Table S17)                                 stage2_output/_shap/
  * the SPEI refitted on 1980-2018 (Table S18A)                      stage2_output/_spei_refit/
  * the bootstrap and Diebold-Mariano outputs of p1_uncertainty.py   stage2_output/_tables/
  * the station-derived SPEI and Tier-1 tables (Tables 1, 4, S3-S6) P1_work/2_era5_validation/
    (written by 2_era5_validation/p1_station_zero_check.py; needs the BMD file, which is
    not distributed, so these tables are skipped when it is absent)

Every table is written to --out (default P1_work/tables) under its final number in the
paper, and compared with the archived copy in --ref (default P1_Data/tables) at the
precision stored there. Nothing is changed to make a value match: differences are
reported.

    python p1_paper_tables.py                 # rebuild and compare
    python p1_paper_tables.py --out DIR       # write elsewhere
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "5_forecasting")]
import paths as P                # noqa: E402
import common as C               # noqa: E402
import p1_selection as S         # noqa: E402
from p1_spei_refit import kappa  # noqa: E402

OUT = P.STAGE2_OUT
SEL = C.SELECTED_MODEL                     # {3: CNN-BiLSTM, 6: BiLSTM, 12: CNN}
MODELS = C.MODELS
TS = C.TIMESCALES
SEEDS = C.SEEDS
THR = C.DROUGHT_THRESHOLD
METRICS = ["RMSE", "MAE", "NSE", "WI", "R2", "Skill"]


# ------------------------------------------------------------------ loading
def load_runs(root=OUT):
    """{(model, ts, lead, seed): (prediction DataFrame, metadata dict)} for every run."""
    runs = {}
    pat = re.compile(r"pred_(.+)_SPEI(\d+)_lead(\d)_seed(\d+)\.csv")
    for m in MODELS:
        for f in sorted((root / m / "runs").glob("pred_*.csv")):
            g = pat.match(f.name)
            key = (g.group(1), int(g.group(2)), int(g.group(3)), int(g.group(4)))
            d = pd.read_csv(f).sort_values(["target_date", "division"]).reset_index(drop=True)
            meta = json.loads((f.parent / f.name.replace("pred_", "metrics_").replace(".csv", ".json"))
                              .read_text(encoding="utf-8"))
            runs[key] = (d, meta)
    return runs


def score(d, col="predicted"):
    o, p, q = d.observed.values, d[col].values, d.persistence.values
    r = C.compute_metrics(o, p, q)
    e = C.event_metrics(o, p)
    return {**{k: r[k] for k in METRICS}, "POD": e["POD"], "FAR": e["FAR"], "CSI": e["CSI"], "SR": e["SR"],
            "hits": e["hits"], "misses": e["misses"], "false_alarms": e["false_alarms"],
            "CN": e["correct_negatives"], "n_observed_events": e["n_observed_events"]}


def ms(x):
    x = pd.Series(x, dtype=float).dropna()
    return (x.mean() if len(x) else np.nan, x.std(ddof=1) if len(x) > 1 else np.nan)


# ------------------------------------------------------------------ tables from the runs
STORED = METRICS + ["POD", "FAR", "CSI", "SR", "hits", "misses", "false_alarms", "n_observed_events"]


def per_seed(runs):
    """Scores of every run, as saved by the training run itself (full precision). The same
    scores recomputed from the prediction files (stored to 6 decimals) are a cross-check."""
    rows, worst = [], 0.0
    for (m, ts, lead, sd), (d, meta) in runs.items():
        rec = score(d)
        for k in STORED:
            v = meta.get(k)
            if v is not None and rec[k] is not None and np.isfinite(rec[k]) and np.isfinite(float(v)):
                worst = max(worst, abs(float(v) - rec[k]))
        row = {k: meta.get(k, rec[k]) for k in STORED}
        row["FAR"] = np.nan if row["FAR"] is None else row["FAR"]
        row["SR"] = np.nan if row["SR"] is None else row["SR"]
        row["CN"] = len(d) - row["hits"] - row["misses"] - row["false_alarms"]
        rows.append(dict(model=m, timescale=ts, lead=lead, seed=sd, **row,
                         stop_epoch=meta.get("stop_epoch"), best_iteration=meta.get("best_iteration"),
                         shuffle=meta.get("shuffle"), elapsed_s=meta.get("elapsed_s")))
    cols = ["model", "timescale", "lead", "seed"] + METRICS + ["POD", "FAR", "CSI", "SR", "hits", "misses",
                                                               "false_alarms", "CN", "n_observed_events",
                                                               "stop_epoch", "best_iteration", "shuffle", "elapsed_s"]
    t = pd.DataFrame(rows)[cols].sort_values(["model", "timescale", "lead", "seed"]).reset_index(drop=True)
    return t, worst


def persistence_scores(runs, ts, lead):
    d = next(v[0] for k, v in runs.items() if k[1] == ts and k[2] == lead)
    return score(d, "persistence")


def table2_lead1(s14, runs):
    rows = []
    for ts in TS:
        for m in sorted(MODELS) + ["Persistence"]:
            if m == "Persistence":
                p = persistence_scores(runs, ts, 1)
                row = dict(timescale=ts, model=m, selected=False)
                for k in METRICS:
                    row[f"{k}_mean"], row[f"{k}_sd"] = p[k], np.nan
                row.update(Skill_min=np.nan, Skill_max=np.nan, Skill_n_pos=np.nan)
            else:
                g = s14[(s14.model == m) & (s14.timescale == ts) & (s14.lead == 1)]
                row = dict(timescale=ts, model=m, selected=SEL[ts] == m)
                for k in METRICS:
                    row[f"{k}_mean"], row[f"{k}_sd"] = ms(g[k])
                row.update(Skill_min=g.Skill.min(), Skill_max=g.Skill.max(), Skill_n_pos=float((g.Skill > 0).sum()))
            rows.append(row)
    return pd.DataFrame(rows)


def tableS11_contingency(s14, runs):
    rows = []
    keys = s14[["model", "timescale", "lead"]].drop_duplicates().sort_values(["model", "timescale", "lead"])
    cols = ["hits", "misses", "false_alarms", "CN", "POD", "FAR", "CSI", "SR"]
    for _, k in keys.iterrows():
        g = s14[(s14.model == k.model) & (s14.timescale == k.timescale) & (s14.lead == k.lead)]
        row = dict(model=k.model, timescale=k.timescale, lead=k.lead)
        for c in cols:
            row[f"{c}_mean"], row[f"{c}_sd"] = ms(g[c])
        row["n_seeds_with_drought_forecasts"] = float(((g.hits + g.false_alarms) > 0).sum())
        rows.append(row)
    for ts in TS:
        for lead in C.LEADS:
            p = persistence_scores(runs, ts, lead)
            row = dict(model="Persistence", timescale=ts, lead=lead)
            for c in cols:
                row[f"{c}_mean"], row[f"{c}_sd"] = p[c], np.nan
            row["n_seeds_with_drought_forecasts"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def table3_multilead(s14, runs, ci, diff, dm):
    rows = []
    for ts in TS:
        for lead in C.LEADS:
            for m in [SEL[ts], "XGBoost", "Persistence"]:
                row = dict(timescale=ts, lead=lead, model=m, selected=(m == SEL[ts]))
                if m == "Persistence":
                    p = persistence_scores(runs, ts, lead)
                    for c in ["RMSE", "NSE", "Skill", "POD", "FAR", "CSI"]:
                        row[f"{c}_mean"], row[f"{c}_sd"] = p[c], np.nan
                    for c in ["hits", "misses", "false_alarms"]:
                        row[f"{c}_mean"], row[f"{c}_sd"] = p[c], np.nan
                    row.update(n_seeds=np.nan, FAR_n_defined=np.nan, Skill_min=np.nan, Skill_max=np.nan)
                else:
                    g = s14[(s14.model == m) & (s14.timescale == ts) & (s14.lead == lead)]
                    for c in ["RMSE", "NSE", "Skill", "POD", "FAR", "CSI", "hits", "misses", "false_alarms"]:
                        row[f"{c}_mean"], row[f"{c}_sd"] = ms(g[c])
                    row.update(n_seeds=float(len(g)), FAR_n_defined=float(g.FAR.notna().sum()),
                               Skill_min=g.Skill.min(), Skill_max=g.Skill.max())
                c = ci[(ci.model == m) & (ci.timescale == ts) & (ci.lead == lead)].set_index("metric")
                for k in ["Skill", "NSE", "CSI"]:
                    ok = m != "Persistence" or k != "Skill"
                    row[f"{k}_lo"] = c.loc[k, "lo"] if ok and k in c.index else np.nan
                    row[f"{k}_hi"] = c.loc[k, "hi"] if ok and k in c.index else np.nan
                for k in ["CSI", "POD"]:
                    dd = diff[(diff.model == m) & (diff.vs == "Persistence") & (diff.timescale == ts)
                              & (diff.lead == lead) & (diff.metric == k)]
                    row[f"d{k}_vs_pers"] = dd.point.iloc[0] if len(dd) else np.nan
                    row[f"d{k}_lo"] = dd.lo.iloc[0] if len(dd) else np.nan
                    row[f"d{k}_hi"] = dd.hi.iloc[0] if len(dd) else np.nan
                t = dm[(dm.model == m) & (dm.timescale == ts) & (dm.lead == lead)]
                ens = t[t.forecast == "seed-ensemble mean"]
                row["DM_stat_ens"] = ens.dm_stat.iloc[0] if len(ens) else np.nan
                row["DM_p_ens"] = ens.p_value.iloc[0] if len(ens) else np.nan
                seeds_ = t[t.forecast.str.startswith("seed ")]
                row["DM_n_seeds_p_lt_0.05"] = float((seeds_.p_value < 0.05).sum()) if len(seeds_) else np.nan
                rows.append(row)
    cols = ["timescale", "lead", "model", "selected", "RMSE_mean", "RMSE_sd", "NSE_mean", "NSE_sd", "Skill_mean",
            "Skill_sd", "POD_mean", "POD_sd", "FAR_mean", "FAR_sd", "CSI_mean", "CSI_sd", "hits_mean", "hits_sd",
            "misses_mean", "misses_sd", "false_alarms_mean", "false_alarms_sd", "n_seeds", "FAR_n_defined",
            "Skill_min", "Skill_max", "Skill_lo", "Skill_hi", "NSE_lo", "NSE_hi", "CSI_lo", "CSI_hi",
            "dCSI_vs_pers", "dCSI_lo", "dCSI_hi", "dPOD_vs_pers", "dPOD_lo", "dPOD_hi", "DM_stat_ens", "DM_p_ens",
            "DM_n_seeds_p_lt_0.05"]
    return pd.DataFrame(rows)[cols]


def r6(x):
    return float(f"{x:.6f}")


def half_up(x, nd=4):
    """4 decimals from the 6-decimal value, as in the archived table."""
    return round(round(float(x), 6), nd)


def median_skill_seed(s14):
    """Third of the five seeds ranked by skill; values from the 6-decimal Table S14, 4 decimals."""
    rows = []
    for (m, ts, lead), g in s14.groupby(["model", "timescale", "lead"]):
        g = g.sort_values("Skill")
        r = g.iloc[2]
        rows.append(dict(model=m, timescale=ts, lead=lead, median_seed=int(r.seed), skill=half_up(r.Skill),
                         NSE=half_up(r.NSE), skill_mean=half_up(g.Skill.mean())))
    return pd.DataFrame(rows)


def selection_rule(s14):
    runs = s14[s14.lead == 1].rename(columns={})
    summ = S.summarise(runs)
    _, report, table = S.apply_rule(summ)
    return table, report


def reproduction_seed42(s14):
    rows = []
    for (m, ts), ref in C.THESIS_LEAD1_SEED42.items():
        g = s14[(s14.model == m) & (s14.timescale == ts) & (s14.lead == 1) & (s14.seed == 42)].iloc[0]
        rows.append(dict(model=m, timescale=ts, **{f"{k}_now": g[k] for k in ["RMSE", "NSE", "Skill"]},
                         **{f"{k}_thesis": ref[k] for k in ["RMSE", "NSE", "Skill"]},
                         max_abs_diff=max(abs(g[k] - ref[k]) for k in ["RMSE", "NSE", "Skill"])))
    return pd.DataFrame(rows)


def dispersion_and_taylor(runs):
    disp, tay = [], []
    for (m, ts, lead, sd), (d, _) in runs.items():
        if lead != 1:
            continue
        o, p, q = d.observed.values, d.predicted.values, d.persistence.values
        disp.append(dict(model=m, timescale=ts, seed=sd, n=len(o), SD_obs=o.std(ddof=1), SD_pred=p.std(ddof=1),
                         SD_ratio=p.std(ddof=1) / o.std(ddof=1), r_obs_pred=np.corrcoef(o, p)[0, 1],
                         slope_pred_on_obs=np.polyfit(o, p, 1)[0], r_pred_persistence=np.corrcoef(p, q)[0, 1],
                         r_dobs_dpred=np.corrcoef(o - q, p - q)[0, 1],
                         d_amplitude_ratio=(p - q).std(ddof=1) / (o - q).std(ddof=1)))
        crmse = np.sqrt(np.mean(((p - p.mean()) - (o - o.mean())) ** 2))
        tay.append(dict(model=m, timescale=ts, seed=sd, SD_obs=o.std(), SD_pred=p.std(), SD_ratio=p.std() / o.std(),
                        r=np.corrcoef(o, p)[0, 1], CRMSE=crmse, CRMSE_norm=crmse / o.std()))
    disp = pd.DataFrame(disp).sort_values(["model", "timescale", "seed"]).reset_index(drop=True)
    tay = pd.DataFrame(tay).sort_values(["model", "timescale", "seed"]).reset_index(drop=True)

    def summ(t, cols, selected=False):
        rows = []
        for (m, ts), g in t.groupby(["model", "timescale"]):
            row = dict(model=m, timescale=ts)
            for c in cols:
                row[f"{c}_mean"], row[f"{c}_sd"], row[f"{c}_n"] = g[c].mean(), g[c].std(ddof=1), len(g)
            if selected:
                row["selected"] = SEL[ts] == m
            rows.append(row)
        return pd.DataFrame(rows)
    dcols = ["SD_obs", "SD_pred", "SD_ratio", "r_obs_pred", "slope_pred_on_obs", "r_pred_persistence",
             "r_dobs_dpred", "d_amplitude_ratio"]
    return disp, summ(disp, dcols, True), tay, summ(tay, ["SD_obs", "SD_pred", "SD_ratio", "r", "CRMSE", "CRMSE_norm"])


def spatial_cases(runs):
    rows = []
    for ts, m in SEL.items():
        d0 = runs[(m, ts, 1, SEEDS[0])][0]
        nat = d0.groupby("target_date")["observed"].mean()
        cases = {"worst": nat.idxmin(), "near_normal": nat.abs().idxmin()}
        for case, date in cases.items():
            for sd in SEEDS:
                d = runs[(m, ts, 1, sd)][0]
                x = d[d.target_date == date].set_index("division").sort_index()
                o, p = x.observed, x.predicted
                rev = [div for div in x.index if np.sign(o[div]) != np.sign(p[div])]
                rows.append(dict(timescale=ts, model=m, case=case, date=date[:10], seed=sd,
                                 national_obs=o.mean(), national_pred=p.mean(), spatial_r=np.corrcoef(o, p)[0, 1],
                                 var_ratio=p.var() / o.var(), sd_ratio=p.std() / o.std(),
                                 driest_obs=o.idxmin(), driest_pred=p.idxmin(), driest_match=o.idxmin() == p.idxmin(),
                                 sign_reversals=len(rev), sign_reversal_divisions=";".join(rev) if rev else np.nan,
                                 obs_drought_divs=int((o <= THR).sum()), pred_drought_divs=int((p <= THR).sum())))
    per = pd.DataFrame(rows)
    summ = []
    for (ts, m, case, date), g in per.groupby(["timescale", "model", "case", "date"]):
        summ.append(dict(timescale=ts, model=m, case=case, date=date, spatial_r_mean=g.spatial_r.mean(),
                         spatial_r_sd=g.spatial_r.std(ddof=1), var_ratio_mean=g.var_ratio.mean(),
                         var_ratio_sd=g.var_ratio.std(ddof=1), sd_ratio_mean=g.sd_ratio.mean(),
                         national_obs=g.national_obs.iloc[0], national_pred_mean=g.national_pred.mean(),
                         driest_obs=g.driest_obs.iloc[0], driest_match_seeds=int(g.driest_match.sum()),
                         sign_reversals_mean=g.sign_reversals.mean(),
                         sign_reversal_divisions=" | ".join(s if isinstance(s, str) else "-"
                                                            for s in g.sign_reversal_divisions)))
    return per, pd.DataFrame(summ)


def shap_tables():
    """Per seed: the importances written by run12_shap.py (6 decimals); each is checked
    against the full SHAP values in the matching .npz file."""
    rows, worst = [], 0.0
    for ts, m in SEL.items():
        for sd in SEEDS:
            imp = pd.read_csv(OUT / "_shap" / f"shap_importance_{m}_SPEI{ts}_seed{sd}.csv")
            z = np.load(OUT / "_shap" / f"shap_values_{m}_SPEI{ts}_seed{sd}.npz", allow_pickle=True)
            full = dict(zip([str(x) for x in z["labels"]], np.abs(z["shap_2d"]).mean(axis=0)))
            worst = max(worst, max(abs(full[v] - a) for v, a in zip(imp.variable, imp.mean_abs_shap)))
            labs, mabs, msig = list(imp.variable), imp.mean_abs_shap.values, imp.mean_signed_shap.values
            met = sum(a for a, lab in zip(mabs, labs) if lab != "SPEI")
            rank = pd.Series(mabs).rank(ascending=False, method="min").astype(int).values
            for lab, a, s_, rk in zip(labs, mabs, msig, rank):
                rows.append(dict(model=m, timescale=ts, seed=sd, variable=lab, mean_abs_shap=a, mean_signed_shap=s_,
                                 met_share_pct=100 * a / met if lab != "SPEI" else np.nan, rank=rk))
    print(f"  SHAP importances equal the .npz SHAP values (max |diff| {worst:.1e})")
    per = pd.DataFrame(rows)
    summ = []
    for (m, ts, var), g in per.groupby(["model", "timescale", "variable"], sort=False):
        row = dict(model=m, timescale=ts, variable=var)
        for c in ["mean_abs_shap", "mean_signed_shap", "met_share_pct", "rank"]:
            x = g[c].dropna()
            row[f"{c}_mean"], row[f"{c}_sd"], row[f"{c}_n"] = x.mean(), x.std(ddof=1), len(x)
        summ.append(row)
    return per, pd.DataFrame(summ)


def refit_agreement():
    a = pd.read_csv(P.SPEI_CSV).sort_values(["division", "date"]).reset_index(drop=True)
    b = pd.read_csv(OUT / "_spei_refit" / "divisional_spei_1980_2024_refit1980-2018.csv")
    b = b.sort_values(["division", "date"]).reset_index(drop=True)
    test = a.year >= C.TEST_START_YEAR
    rows = []
    for k in [1, 3, 6, 12]:
        x, y = a.loc[test, f"SPEI_{k}"].values, b.loc[test, f"SPEI_{k}"].values
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        dx, dy = x <= THR, y <= THR
        rows.append(dict(timescale=k, n=int(m.sum()), r=np.corrcoef(x, y)[0, 1], mean_diff_refit_minus_thesis=np.mean(y - x),
                         max_abs_diff=np.max(np.abs(y - x)), drought_months_thesis=int(dx.sum()),
                         drought_months_refit=int(dy.sum()), drought_agreement_pct=100 * np.mean(dx == dy),
                         kappa=kappa(dx, dy)))
    return pd.DataFrame(rows)


def strict_vs_main(s14):
    def runs_of(root):
        rows = [json.loads(f.read_text(encoding="utf-8"))
                for f in sorted((root / "BiLSTM" / "runs").glob("metrics_BiLSTM_SPEI6_lead1_seed*.json"))]
        return pd.DataFrame(rows)
    rows = []
    for name, root in [("main", OUT), ("strict", OUT / "_strict")]:
        df = runs_of(root)                     # the scores each run saved, as run11_sensitivity.py
        k = ["Skill", "NSE", "RMSE", "CSI", "POD"]
        rows.append(dict(pipeline=name, n=len(df), **{f"{c}_mean": df[c].mean() for c in k},
                         **{f"{c}_sd": df[c].std(ddof=1) for c in k}))
    return pd.DataFrame(rows)


def derived_numbers(t2, s14, runs):
    rows = []
    lead1 = t2[t2.model != "Persistence"]
    best_hi = {"RMSE": False, "MAE": False, "NSE": True, "WI": True, "R2": True, "Skill": True, "CSI": True, "POD": True}
    cont = tableS11_contingency(s14, runs)
    for ts in TS:
        g = lead1[lead1.timescale == ts].set_index("model")
        dl = g.drop(index="XGBoost")
        best_dl = dl.RMSE_mean.idxmin()
        rows.append(dict(item="XGBoost RMSE above best DL model (%)", timescale=ts,
                         value=100 * (g.loc["XGBoost", "RMSE_mean"] / dl.loc[best_dl, "RMSE_mean"] - 1),
                         note=f"best DL by mean RMSE = {best_dl}", lead=np.nan))
        for k, hi in best_hi.items():
            if k in ("CSI", "POD"):
                c = cont[(cont.timescale == ts) & (cont.lead == 1) & (cont.model != "Persistence")].set_index("model")
                v = c[f"{k}_mean"]
            else:
                v = g[f"{k}_mean"]
            b, w = (v.idxmax(), v.idxmin()) if hi else (v.idxmin(), v.idxmax())
            rows.append(dict(item=f"best {k} (5-seed mean)", timescale=ts, value=v[b], note=b, lead=np.nan))
            rows.append(dict(item=f"worst {k} (5-seed mean)", timescale=ts, value=v[w], note=w, lead=np.nan))
    for ts in TS:
        m = SEL[ts]
        for lead in C.LEADS:
            g = s14[(s14.model == m) & (s14.timescale == ts) & (s14.lead == lead)]
            rows.append(dict(item="skill SD / NSE SD (selected model)", timescale=ts,
                             value=g.Skill.std(ddof=1) / g.NSE.std(ddof=1), note=m, lead=float(lead)))
        r1, r3 = persistence_scores(runs, ts, 1)["RMSE"], persistence_scores(runs, ts, 3)["RMSE"]
        rows.append(dict(item="persistence RMSE increase lead 1 -> 3 (%)", timescale=ts, value=100 * (r3 / r1 - 1),
                         note=np.nan, lead=np.nan))
    return pd.DataFrame(rows)


def hyperparameters(runs):
    """Tables S8A and S9: the tuned values stored by every run; they must be identical in every seed and lead."""
    dl_keys = ["window", "filters", "kernel_size", "conv2_filters", "lstm_units", "dense_units", "dropout", "lr", "shuffle"]
    xgb_keys = ["max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "gamma",
                "reg_lambda", "reg_alpha"]
    dl, xgb = {}, {}
    for (m, ts, lead, sd), (_, meta) in runs.items():
        keys, store = (xgb_keys, xgb) if m == "XGBoost" else (dl_keys, dl)
        vals = {k: meta.get(k) for k in keys}
        if (m, ts) in store and store[(m, ts)] != vals:
            raise AssertionError(f"{m} SPEI-{ts}: tuned values differ between runs")
        store[(m, ts)] = vals
    s8 = pd.DataFrame([dict(model=m, timescale=ts, **v) for (m, ts), v in sorted(dl.items())])
    for c in ["window", "filters", "kernel_size", "conv2_filters", "lstm_units", "dense_units"]:
        s8[c] = s8[c].astype("Int64")
    s9 = pd.DataFrame([dict(model=m, timescale=ts, **v) for (m, ts), v in sorted(xgb.items())])
    return s8, s9


# ------------------------------------------------------------------ station-derived tables (need the BMD file)
def station_tables(runs, tier1):
    ser = pd.read_csv(tier1 / "spei_series_bmd_vs_era5.csv")
    rows = []
    for ts, m in SEL.items():
        st = ser[["division", "year", "month", f"SPEI{ts}_b"]].rename(columns={f"SPEI{ts}_b": "station"})
        for sd in SEEDS:
            d = runs[(m, ts, 1, sd)][0].copy()
            dt = pd.to_datetime(d.target_date)
            d["year"], d["month"] = dt.dt.year, dt.dt.month
            d = d[d.year <= 2022].merge(st, on=["division", "year", "month"])
            for ref, col in [("ERA5-derived", "observed"), ("Station-derived", "station")]:
                for who, pc in [(m, "predicted"), ("Persistence", "persistence")]:
                    if who == "Persistence" and sd != SEEDS[0]:
                        continue
                    o, p, q = d[col].values, d[pc].values, d.persistence.values
                    mse, msep = np.mean((p - o) ** 2), np.mean((q - o) ** 2)
                    O, F = o <= THR, p <= THR
                    h, mi, fa = (O & F).sum(), (O & ~F).sum(), (~O & F).sum()
                    r = np.corrcoef(o, p)[0, 1]
                    rows.append(dict(n=len(o), RMSE=np.sqrt(mse), MAE=np.mean(np.abs(p - o)), r=r, R2=r * r,
                                     NSE=1 - mse / np.var(o), Skill=1 - mse / msep,
                                     POD=h / (h + mi) if h + mi else np.nan, FAR=fa / (h + fa) if h + fa else np.nan,
                                     events_obs=int(O.sum()), timescale=ts, model=who, reference=ref,
                                     seed=sd if who != "Persistence" else -1))
    per = pd.DataFrame(rows)
    summ = []
    for (ts, m, ref), g in per.groupby(["timescale", "model", "reference"], sort=False):
        row = dict(timescale=ts, model=m, reference=ref)
        for c in ["r", "RMSE", "MAE", "NSE", "Skill", "POD", "FAR", "events_obs", "n"]:
            row[f"{c}_mean"], row[f"{c}_sd"], row[f"{c}_n"] = g[c].mean(), g[c].std(ddof=1) if len(g) > 1 else np.nan, len(g)
        summ.append(row)
    return per, pd.DataFrame(summ)


def tier1_tables(tier1):
    out = {}
    out["Table1A_S3_precipitation_validation.csv"] = pd.read_csv(tier1 / "table1_precipitation_validation.csv")
    out["Table1B_S5_pet_sensitivity.csv"] = pd.read_csv(tier1 / "table2_pet_sensitivity.csv")
    a = pd.read_csv(tier1 / "table3_spei_agreement.csv")
    e = pd.read_csv(tier1 / "table4_drought_event_detection.csv")
    out["Table1C_S6_spei_agreement.csv"] = a.merge(e, on=["scale", "n"])
    out["TableS4_station_month_shifts.csv"] = pd.read_csv(tier1 / "table5_per_station_month_shifts.csv")
    return out


# ------------------------------------------------------------------ comparison
def decimals(s):
    s = str(s)
    return len(s.split(".")[1]) if "." in s and re.fullmatch(r"[-+]?\d+\.\d+", s) else 0


def compare(new, ref_path, keys):
    """Every cell of the archived table must equal the rebuilt one at the stored precision."""
    ref = pd.read_csv(ref_path, dtype=str, keep_default_na=False)
    n = new.copy()
    for k in keys:
        n[k] = n[k].astype(str)
        ref[k] = ref[k].astype(str).str.replace(r"\.0+$", "", regex=True)
        n[k] = n[k].str.replace(r"\.0+$", "", regex=True)
    n = n.set_index(keys)
    ref = ref.set_index(keys)
    bad, worst, cells = [], 0.0, 0
    missing = [i for i in ref.index if i not in n.index]
    for idx, row in ref.iterrows():
        if idx not in n.index:
            continue
        for c in ref.columns:
            if c not in n.columns:
                bad.append((idx, c, row[c], "column missing"))
                continue
            rv, nv = row[c], n.loc[idx, c]
            cells += 1
            if rv == "":
                if not (nv is None or (isinstance(nv, float) and np.isnan(nv)) or str(nv) in ("", "nan", "None")):
                    bad.append((idx, c, rv, nv))
                continue
            try:
                r_ = float(rv)
                n_ = float(nv)
            except (TypeError, ValueError):
                if str(rv) != str(nv):
                    bad.append((idx, c, rv, nv))
                continue
            tol = 0.5 * 10 ** -decimals(rv) + 1e-9
            d = abs(r_ - n_)
            worst = max(worst, d if d <= tol else 0)
            if not (d <= tol or (np.isnan(r_) and np.isnan(n_))):
                bad.append((idx, c, rv, f"{n_:.6f}"))
    return bad, missing, cells


FLOAT = "%.6f"
SPEC = [  # (final file name, builder key, key columns for the comparison, float format)
    ("Table2_lead1_all_models.csv", "t2", ["timescale", "model"], FLOAT),
    ("Table3_multilead.csv", "t3", ["timescale", "lead", "model"], FLOAT),
    ("TableS10B_selection_rule.csv", "sel", ["timescale", "model"], FLOAT),
    ("TableS11_contingency.csv", "s11", ["model", "timescale", "lead"], FLOAT),
    ("TableS12_reproduction_seed42.csv", "s12", ["model", "timescale"], FLOAT),
    ("TableS13A_bootstrap_ci.csv", "s13a", ["model", "timescale", "lead", "metric"], "%.4f"),
    ("TableS13B_bootstrap_differences.csv", "s13b", ["model", "vs", "timescale", "lead", "metric"], "%.4f"),
    ("TableS13C_diebold_mariano.csv", "s13c", ["model", "timescale", "lead", "forecast"], "%.4f"),
    ("TableS14_per_seed_runs.csv", "s14", ["model", "timescale", "lead", "seed"], FLOAT),
    ("TableS15_dispersion_per_seed.csv", "s15p", ["model", "timescale", "seed"], FLOAT),
    ("TableS15_dispersion_summary.csv", "s15s", ["model", "timescale"], FLOAT),
    ("TableS16_spatial_cases_per_seed.csv", "s16p", ["timescale", "case", "seed"], FLOAT),
    ("TableS16_spatial_cases_summary.csv", "s16s", ["timescale", "case"], FLOAT),
    ("TableS17_shap_per_seed.csv", "s17p", ["timescale", "seed", "variable"], FLOAT),
    ("TableS17_shap_summary.csv", "s17s", ["timescale", "variable"], FLOAT),
    ("TableS18A_spei_refit_agreement.csv", "s18a", ["timescale"], "%.4f"),
    ("TableS18B_strict_vs_main.csv", "s18b", ["pipeline"], FLOAT),
    ("TextS5_median_skill_seed.csv", "med", ["model", "timescale", "lead"], None),
    ("TableS8A_dl_hyperparameters.csv", "s8", ["model", "timescale"], None),
    ("TableS9_xgboost_hyperparameters.csv", "s9", ["model", "timescale"], None),
    ("Fig05_taylor_stats_per_seed.csv", "tayp", ["model", "timescale", "seed"], FLOAT),
    ("Fig05_taylor_stats_summary.csv", "tays", ["model", "timescale"], FLOAT),
    ("Derived_numbers.csv", "der", ["item", "timescale", "lead"], FLOAT),
    ("Table4_station_target_per_seed.csv", "t4p", ["timescale", "model", "reference", "seed"], FLOAT),
    ("Table4_station_target_summary.csv", "t4s", ["timescale", "model", "reference"], FLOAT),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(P.WORK / "tables"))
    ap.add_argument("--ref", default=str(P.TABLES))
    ap.add_argument("--tier1", default=str(P.WORK / "2_era5_validation"),
                    help="folder with the outputs of p1_station_zero_check.py")
    a = ap.parse_args()
    out, ref, tier1 = Path(a.out), Path(a.ref), Path(a.tier1)
    out.mkdir(parents=True, exist_ok=True)

    print("loading the run files ...")
    runs = load_runs()
    n_main = sum(1 for k in runs)
    print(f"  {n_main} runs (expected 135)")
    T = {}
    T["s14"], worst = per_seed(runs)
    print(f"  scores recomputed from the prediction files equal the scores saved by each run "
          f"(max |diff| {worst:.1e}; the files store 6 decimals)")
    T["t2"] = table2_lead1(T["s14"], runs)
    T["s11"] = tableS11_contingency(T["s14"], runs)
    ci = pd.read_csv(OUT / "_tables" / "stage2_bootstrap_ci.csv")
    diff = pd.read_csv(OUT / "_tables" / "stage2_bootstrap_diff.csv")
    dm = pd.read_csv(OUT / "_tables" / "stage2_dm_tests.csv")
    T["s13a"], T["s13b"], T["s13c"] = ci, diff, dm
    T["t3"] = table3_multilead(T["s14"], runs, ci, diff, dm)
    T["med"] = median_skill_seed(T["s14"])
    T["sel"], report = selection_rule(T["s14"])
    T["s12"] = reproduction_seed42(T["s14"])
    T["s15p"], T["s15s"], T["tayp"], T["tays"] = dispersion_and_taylor(runs)
    T["s16p"], T["s16s"] = spatial_cases(runs)
    T["s17p"], T["s17s"] = shap_tables()
    T["s18a"] = refit_agreement()
    T["s18b"] = strict_vs_main(T["s14"])
    T["der"] = derived_numbers(T["t2"], T["s14"], runs)
    T["s8"], T["s9"] = hyperparameters(runs)
    print("  tuned hyperparameters are identical in every seed and lead of each model and timescale")
    extra = {}
    if (tier1 / "spei_series_bmd_vs_era5.csv").exists():
        T["t4p"], T["t4s"] = station_tables(runs, tier1)
        extra = tier1_tables(tier1)
    else:
        print(f"  station-derived tables skipped: run 2_era5_validation/p1_station_zero_check.py first "
              f"(needs the BMD file, see the README)")
    vif = P.WORK / "3_predictor_screening" / "vif" / "vif_results.csv"
    if vif.exists():
        extra["TableS7_vif.csv"] = pd.read_csv(vif).rename(columns={"Unnamed: 0": "variable"})
    else:
        print("  Table S7 skipped: run 3_predictor_screening/vif_test.py first")
    print("\nselection rule:\n  " + "\n  ".join(report))

    print(f"\nwriting to {out} and comparing with {ref}")
    total_bad = 0
    for fname, key, keys, ff in SPEC:
        if key not in T:
            continue
        df = T[key]
        df.to_csv(out / fname, index=False, float_format=ff) if ff else df.to_csv(out / fname, index=False)
        rp = ref / fname
        if not rp.exists():
            print(f"  NEW   {fname:<40} {len(df):>4} rows (no archived copy to compare)")
            continue
        bad, missing, cells = compare(pd.read_csv(out / fname, dtype=str, keep_default_na=False), rp, keys)
        total_bad += len(bad) + len(missing)
        print(f"  {'OK  ' if not bad and not missing else 'DIFF'}  {fname:<40} {len(df):>4} rows, {cells:>5} cells"
              + (f"  -> {len(bad)} differing cell(s), {len(missing)} missing row(s)" if bad or missing else ""))
        for b in bad[:8]:
            print(f"        {b}")
    for fname, df in extra.items():
        df.to_csv(out / fname, index=False, float_format=FLOAT)
        rp = ref / fname
        if rp.exists():
            bad, missing, cells = compare(pd.read_csv(out / fname, dtype=str, keep_default_na=False), rp,
                                          [df.columns[0]])
            total_bad += len(bad) + len(missing)
            print(f"  {'OK  ' if not bad and not missing else 'DIFF'}  {fname:<40} {len(df):>4} rows, {cells:>5} cells")
        else:
            print(f"  NEW   {fname:<40} {len(df):>4} rows")
    print(f"\n{'Every rebuilt table equals the archived copy.' if not total_bad else f'{total_bad} difference(s) found (listed above).'}")
    return total_bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
