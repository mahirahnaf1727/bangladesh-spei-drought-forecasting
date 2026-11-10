"""
fig_data.py  --  loads the Stage 2 tables and prediction files for the figures,
and checks that what is plotted equals the reported numbers.

No number is computed here that is not already in the Stage 2 tables, except
where a figure needs it (each such case is written to the check log and named
in the figure log). Every check that fails raises an error, so a figure can
never be drawn from data that disagree with the paper's tables.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import p1_fig_config as C

# file-name model keys (Stage 2) in the fixed display order of the brief
MODELS = ["XGBoost", "CNN", "BiLSTM", "CNN-BiLSTM", "CEEMDAN-BiLSTM"]
DISPLAY = {"XGBoost": "XGBoost", "CNN": "CNN", "BiLSTM": "Bi-LSTM",
           "CNN-BiLSTM": "CNN-BiLSTM", "CEEMDAN-BiLSTM": "CEEMDAN-BiLSTM",
           "Persistence": "Persistence"}
SELECTED = {3: "CNN-BiLSTM", 6: "BiLSTM", 12: "CNN"}
TIMESCALES = [3, 6, 12]
SEEDS = [42, 43, 44, 45, 46]
N_DROUGHT = {3: 119, 6: 156, 12: 204}       # observed drought months, test period
TOL = 2e-6                                   # tables are stored to 6 decimals


# ---------------------------------------------------------------------------
# check log
# ---------------------------------------------------------------------------
class CheckLog:
    def __init__(self):
        self.lines = []

    def ok(self, msg):
        self.lines.append(f"- OK: {msg}")

    def note(self, msg):
        self.lines.append(f"- NOTE: {msg}")

    def section(self, title):
        self.lines.append(f"\n### {title}\n")

    def fail(self, msg):
        self.lines.append(f"- FAIL: {msg}")
        raise AssertionError(msg)

    def write(self, path):
        Path(path).write_text("\n".join(self.lines).lstrip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def table(name):
    p = C.TABLES_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Stage 2 table not found: {p}  (check TABLES_DIR in p1_fig_config.py)")
    return pd.read_csv(p)


def pred_path(model, ts, lead, seed):
    return C.OUT_ROOT / model / "runs" / f"pred_{model}_SPEI{ts}_lead{lead}_seed{seed}.csv"


def pred(model, ts, lead, seed):
    p = pred_path(model, ts, lead, seed)
    if not p.exists():
        raise FileNotFoundError(f"prediction file not found: {p}  (check OUT_ROOT)")
    d = pd.read_csv(p, parse_dates=["target_date"])
    return d.sort_values(["target_date", "division"]).reset_index(drop=True)


def median_seed(model, ts, lead=1):
    ms = table("TextS5_median_skill_seed.csv")
    r = ms[(ms.model == model) & (ms.timescale == ts) & (ms.lead == lead)]
    if len(r) != 1:
        raise KeyError(f"no median-skill seed for {model} SPEI-{ts} lead {lead}")
    return int(r.median_seed.iloc[0])


# ---------------------------------------------------------------------------
# metric helpers (same definitions as Stage 2 common.py)
# ---------------------------------------------------------------------------
def skill_from_file(d):
    mse_m = np.mean((d.predicted - d.observed) ** 2)
    mse_p = np.mean((d.persistence - d.observed) ** 2)
    return 1.0 - mse_m / mse_p


# ---------------------------------------------------------------------------
# checks shared by several figures
# ---------------------------------------------------------------------------
def check_per_seed_against(log, s15, summary, keys, metric_pairs, label):
    """Mean and sample SD of the per-seed runs (Table S14) must equal the summary table."""
    g = s15.groupby(keys)
    worst = 0.0
    for (k, row) in summary.set_index(keys).iterrows():
        if k not in g.groups:
            continue
        runs = g.get_group(k)
        if len(runs) != 5:
            log.fail(f"{label}: {k} has {len(runs)} seeds, expected 5")
        for col_runs, col_mean, col_sd in metric_pairs:
            m, s = runs[col_runs].mean(), runs[col_runs].std(ddof=1)
            dm = abs(m - row[col_mean])
            ds = abs(s - row[col_sd]) if col_sd else 0.0
            worst = max(worst, dm, ds)
            if dm > TOL or ds > TOL:
                log.fail(f"{label}: {k} {col_runs} mean/SD {m:.6f}/{s:.6f} "
                         f"vs table {row[col_mean]:.6f}/{row[col_sd] if col_sd else float('nan'):.6f}")
    log.ok(f"{label}: mean and SD of the per-seed values equal the table (max |diff| {worst:.1e})")


def check_median_seed_files(log, models, lead=1):
    """For every single-seed panel: the seed is the median-skill seed, and the skill
    recomputed from its prediction file equals Table S14."""
    s15 = table("TableS14_per_seed_runs.csv")
    worst = 0.0
    for ts in TIMESCALES:
        for m in models:
            sd = median_seed(m, ts, lead)
            runs = s15[(s15.model == m) & (s15.timescale == ts) & (s15.lead == lead)].sort_values("Skill")
            if int(runs.iloc[2].seed) != sd:
                log.fail(f"{m} SPEI-{ts} lead {lead}: median seed {sd} is not the 3rd-ranked seed")
            sk_file = skill_from_file(pred(m, ts, lead, sd))
            sk_tab = float(runs[runs.seed == sd].Skill.iloc[0])
            worst = max(worst, abs(sk_file - sk_tab))
            if abs(sk_file - sk_tab) > 1e-5:
                log.fail(f"{m} SPEI-{ts} seed {sd}: skill from file {sk_file:.6f} vs S15 {sk_tab:.6f}")
    log.ok(f"single-seed panels use the median-skill seed (3rd of 5 by skill) and the skill "
           f"recomputed from each prediction file equals Table S14 (max |diff| {worst:.1e})")


def persistence_series(log, ts, lead=1):
    """The persistence column is the same data in every model's file; check this and return it."""
    ref = None
    n = 0
    for m in MODELS:
        for s in SEEDS:
            p = pred_path(m, ts, lead, s)
            if not p.exists():
                continue
            d = pred(m, ts, lead, s)
            if ref is None:
                ref = d[["target_date", "division", "observed", "persistence"]].copy()
            else:
                if not (np.allclose(d.observed.values, ref.observed.values, atol=1e-9)
                        and np.allclose(d.persistence.values, ref.persistence.values, atol=1e-9)):
                    log.fail(f"observed/persistence differ between prediction files (SPEI-{ts}, {m} seed {s})")
            n += 1
    log.ok(f"SPEI-{ts} lead {lead}: observed and persistence columns identical in all {n} prediction files")
    return ref
