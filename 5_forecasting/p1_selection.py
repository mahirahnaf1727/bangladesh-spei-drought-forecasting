"""
================================================================================
p1_selection.py  --  the model-selection rule, fixed in advance  (Stage 2)
================================================================================
Fixed in Stage 1 (15 Oct 2026), BEFORE any Stage 2 result was seen:

  1. Score = 5-seed mean MSE skill against persistence at lead 1, every run
     through the same pipeline. (Within a timescale, skill, NSE and RMSE rank
     models identically; event detection is reported, not used to select.)
  2. The incumbent (CNN-BiLSTM at SPEI-3, BiLSTM at SPEI-6, CNN at SPEI-12)
     stays unless a challenger's mean skill beats it by MORE than one seed SD
     (the larger of the two models' SDs).
  3. No challenger clears the bar -> the models are reported as
     indistinguishable and the incumbent is carried forward as representative.
  4. A challenger clears it -> it becomes the selected model (and leads 2-3,
     SHAP, station target and SI maps are rerun for that timescale only).

Used by run10 (to choose the models for Set C) and by the Stage 2 tables, so
the rule is applied in exactly one place.
================================================================================
"""

import json

import numpy as np
import pandas as pd

import common as C

INCUMBENT = {3: "CNN-BiLSTM", 6: "BiLSTM", 12: "CNN"}


def collect_runs(lead=None):
    """Every finished run's metrics (one row per model x timescale x lead x seed)."""
    rows = []
    for model in C.MODELS:
        for f in sorted((C.MODEL_DIRS[model] / "runs").glob("metrics_*.json")):
            rows.append(json.loads(f.read_text(encoding="utf-8")))
    df = pd.DataFrame(rows)
    if len(df) and lead is not None:
        df = df[df["lead"] == lead]
    return df


def summarise(df):
    """Mean, SD (ddof=1), min, max and n across seeds for each model x timescale x lead."""
    g = df.groupby(["model", "timescale", "lead"])
    out = g["Skill"].agg(skill_mean="mean", skill_sd=lambda v: np.std(v, ddof=1),
                         skill_min="min", skill_max="max", n_seeds="count").reset_index()
    for m in ["RMSE", "NSE", "CSI", "POD"]:
        out[f"{m}_mean"] = g[m].mean().values
        out[f"{m}_sd"] = g[m].std(ddof=1).values
    return out


def apply_rule(lead1_summary, n_required=5):
    """Return (selected: {ts: model}, report: list of text lines, table: DataFrame)."""
    selected, report, rows = {}, [], []
    for ts, inc in INCUMBENT.items():
        cell = lead1_summary[(lead1_summary.timescale == ts) & (lead1_summary.lead == 1)]
        # Guard (added 16 Oct 2026): the rule compares 5-seed means of ALL five
        # models, so it must never run on an unfinished Set B. Before, a
        # challenger with missing seeds (or no runs at all) was silently skipped.
        short = [f"{m} {int(cell.loc[cell.model == m, 'n_seeds'].sum())}/{n_required}"
                 for m in C.MODELS
                 if cell.loc[cell.model == m, "n_seeds"].sum() < n_required]
        if short:
            raise RuntimeError(
                f"Set B is not finished at SPEI-{ts} (seeds done: {', '.join(short)}). "
                "Run S4_setB_lead1_seeds.py again until it prints 'Set B finished'.")
        if inc not in set(cell.model):
            raise RuntimeError(f"SPEI-{ts}: incumbent {inc} has no lead-1 runs yet.")
        i = cell[cell.model == inc].iloc[0]
        if i.n_seeds < n_required:
            raise RuntimeError(f"SPEI-{ts}: incumbent {inc} has {i.n_seeds} seeds, "
                               f"needs {n_required}.")
        winners = []
        for _, c in cell[cell.model != inc].iterrows():
            if c.n_seeds < n_required:
                rows.append(dict(timescale=ts, model=c.model, verdict="not enough seeds"))
                continue
            bar = max(i.skill_sd, c.skill_sd)
            margin = c.skill_mean - i.skill_mean
            beats = margin > bar
            rows.append(dict(timescale=ts, model=c.model,
                             skill_mean=c.skill_mean, skill_sd=c.skill_sd,
                             incumbent=inc, incumbent_mean=i.skill_mean,
                             incumbent_sd=i.skill_sd, margin=margin, bar=bar,
                             verdict="BEATS incumbent" if beats else "does not beat"))
            if beats:
                winners.append((c.skill_mean, c.model))
        if winners:
            new = max(winners)[1]
            selected[ts] = new
            report.append(f"SPEI-{ts}: {new} beats the incumbent {inc} by more than one "
                          f"seed SD -> SELECTION CHANGES to {new}.")
        else:
            selected[ts] = inc
            close = [r["model"] for r in rows
                     if r["timescale"] == ts and r.get("skill_mean") is not None
                     and abs(r["skill_mean"] - i.skill_mean) <= r["bar"]]
            report.append(f"SPEI-{ts}: no challenger beats {inc} by more than one seed SD "
                          f"-> {inc} stays. Indistinguishable from it: "
                          f"{', '.join(close) if close else 'none'}.")
    return selected, report, pd.DataFrame(rows)
