"""
================================================================================
run11_sensitivity.py  --  F1 + F2 sensitivity test  (Stage 2)
================================================================================
Builds the "strict" pipeline and retrains ONE model in it:
    * SPEI fitted on 1980-2018 only            (fix F1; file from p1_spei_refit.py)
    * climate anomalies from 1980-2018 only    (fix F2)
    * Bi-LSTM, SPEI-6, lead 1, seeds 42-46     (same tuned settings as the main runs)

Then it applies the rule fixed in the Stage 2 plan, BEFORE seeing the result:
    if |mean skill (strict) - mean skill (main)| < one seed SD of the main runs
        -> the main results stand; report this test in the SI
    otherwise
        -> switch to the strict pipeline for all runs

Everything is written under OUT_ROOT/_strict/, so the main runs are untouched.

RUN ORDER:  p1_spei_refit.py  ->  run10 --set B (needs the main Bi-LSTM runs)
            ->  this script
================================================================================
"""

import os
os.environ["P1_PIPELINE"] = "strict"          # must be set before importing common

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import p1_config as CFG
import common as C

MODEL, TS, LEAD = "BiLSTM", 6, 1


def load_runs(root):
    rows = []
    for s in C.SEEDS:
        f = Path(root) / MODEL / "runs" / f"metrics_{C.run_stem(MODEL, TS, LEAD, s)}.json"
        if f.exists():
            rows.append(json.loads(f.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def main():
    if not Path(C.SPEI_CSV).exists():
        sys.exit(f"Missing {C.SPEI_CSV}\nRun p1_spei_refit.py first.")

    C.banner("RUN11  --  F1 + F2 SENSITIVITY  (strict pipeline)")
    import run1_build_matrices as R1
    import run2_data_prep as R2
    R1.main()
    R2.main()

    import run10_seed_runs as R10
    for s in C.SEEDS:
        if R10.done(MODEL, TS, LEAD, s):
            print(f"  seed {s}: already done")
            continue
        row = R10.train_one(MODEL, TS, LEAD, s)
        print(f"  seed {s}: skill {row['Skill']:+.4f}  NSE {row['NSE']:.4f}  ({row['elapsed_s']} s)")

    strict = load_runs(C.OUT_ROOT)
    main_ = load_runs(CFG.OUT_ROOT)
    if len(main_) < len(C.SEEDS):
        sys.exit("The main-pipeline Bi-LSTM SPEI-6 runs are not finished yet "
                 "(run10 --set B). Run this script again afterwards for the verdict.")

    out = []
    for name, df in [("main", main_), ("strict", strict)]:
        out.append(dict(pipeline=name, n=len(df),
                        **{f"{m}_mean": df[m].mean() for m in ["Skill", "NSE", "RMSE", "CSI", "POD"]},
                        **{f"{m}_sd": df[m].std(ddof=1) for m in ["Skill", "NSE", "RMSE", "CSI", "POD"]}))
    tab = pd.DataFrame(out)
    delta = tab.loc[1, "Skill_mean"] - tab.loc[0, "Skill_mean"]
    bar = tab.loc[0, "Skill_sd"]
    stands = abs(delta) < bar
    verdict = (f"|delta skill| = {abs(delta):.4f} vs one seed SD of the main runs = {bar:.4f} -> "
               + ("MAIN RESULTS STAND (option B)." if stands
                  else "SWITCH TO THE STRICT PIPELINE (option A)."))

    CFG.OUT_ROOT.joinpath("_tables").mkdir(parents=True, exist_ok=True)
    tab.to_csv(CFG.OUT_ROOT / "_tables" / "stage2_sensitivity_F1F2.csv", index=False,
               float_format="%.6f")
    (CFG.OUT_ROOT / "_tables" / "stage2_sensitivity_F1F2.txt").write_text(
        f"{datetime.now():%Y-%m-%d %H:%M}\n{verdict}\n", encoding="utf-8")
    print("\n" + tab.round(4).to_string(index=False))
    print("\nVERDICT (rule fixed in advance): " + verdict)


if __name__ == "__main__":
    main()
