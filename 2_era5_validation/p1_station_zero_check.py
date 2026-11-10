"""
p1_station_zero_check.py  --  checks the station zero-value rule against the paper.

Runs the ERA5-vs-BMD validation of Validation_1.py (with the zero-value rule of
1_data_and_index/station_zero_rule.py, Text S2) and prints every corrected number
next to the value printed in the paper:
    Table 1A (pooled precipitation), Table S3 (eight divisions),
    Table S4 (Kutubdia precipitation r), Table 1C / S6 (SPEI agreement),
    Fig. S2 (full-record SPEI-3 r for Dhaka and Barisal).
With --no-rule it reproduces the numbers published before the correction.

It also writes the Tier-1 tables to P1_work/2_era5_validation (same names as
Validation_1.py), which 8_figures/figS01_validation.py and
6_evaluation/p1_paper_tables.py read.

Needs the BMD compilation in P1_Data/external (not distributed; see the README).

    python p1_station_zero_check.py            # corrected numbers (the paper)
    python p1_station_zero_check.py --no-rule  # numbers before the correction
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import Validation_1 as V  # noqa: E402

PAPER = {   # values printed in the paper (corrected, zero-value rule applied)
    "Table 1A": {"ALL": (0.891, 0.696, 3.4, 94.6, 0.790, 0.875)},
    "Table S3": {"Barisal": (0.934, 0.726, -7.2, 75.2, 0.865, 0.844),
                 "Chittagong": (0.939, 0.733, -24.1, 131.3, 0.766, 0.597),
                 "Dhaka": (0.904, 0.699, 7.0, 68.8, 0.797, 0.878),
                 "Khulna": (0.933, 0.756, 5.5, 54.8, 0.861, 0.913),
                 "Mymensingh": (0.851, 0.594, 17.5, 116.9, 0.658, 0.765),
                 "Rajshahi": (0.898, 0.663, 18.0, 69.7, 0.733, 0.767),
                 "Rangpur": (0.869, 0.564, 18.0, 109.7, 0.698, 0.773),
                 "Sylhet": (0.926, 0.707, 7.2, 102.1, 0.839, 0.889)},
    "Table 1C": {1: (4128, 0.809, 0.635, 0.622, 0.713, 0.270, 0.660),
                 3: (4112, 0.766, 0.689, 0.534, 0.652, 0.345, 0.579),
                 6: (4088, 0.712, 0.755, 0.427, 0.557, 0.436, 0.468),
                 12: (4040, 0.647, 0.825, 0.294, 0.522, 0.482, 0.422)},
    "Table S4 Kutubdia P r": 0.971,
    "Fig. S2": {"Dhaka": 0.786, "Barisal": 0.792},
    "counts": (101, 1505, 5),
}
BEFORE = {  # numbers published before the correction (thesis), reproduced with --no-rule
    "Table 1A": {"ALL": (0.887, 0.694, 9.2, 95.1, 0.769, 0.854)},
    "Table 1C": {1: (4128, 0.805, 0.641, 0.616, 0.711, 0.270, 0.658),
                 3: (4112, 0.757, 0.704, 0.518, 0.615, 0.379, 0.537),
                 6: (4088, 0.693, 0.781, 0.390, 0.532, 0.450, 0.443),
                 12: (4040, 0.602, 0.877, 0.207, 0.484, 0.508, 0.381)},
    "Table S4 Kutubdia P r": 0.970,
    "Fig. S2": {"Dhaka": 0.814, "Barisal": 0.792},
}


def same(a, b, dec):
    return round(float(a), dec) == round(float(b), dec) or abs(float(a) - float(b)) < 0.5 * 10 ** -dec + 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rule", action="store_true", help="leave the zero values in (thesis numbers)")
    a = ap.parse_args()
    V.ZERO_RULE = not a.no_rule
    ref = BEFORE if a.no_rule else PAPER
    R = V.run_analysis()
    fails = 0

    def show(label, got, exp, decs):
        nonlocal fails
        ok = all(same(g, e, d) for g, e, d in zip(got, exp, decs))
        fails += not ok
        g = "  ".join(f"{x:.{d}f}" for x, d in zip(got, decs))
        e = "  ".join(f"{x:.{d}f}" for x, d in zip(exp, decs))
        print(f"  {'OK  ' if ok else 'DIFF'} {label:<24} got {g}   paper {e}")

    pr = R["precip"].set_index("division")
    cols, decs = ["r", "r_anom", "pbias", "RMSE", "NSE", "KGE"], [3, 3, 1, 1, 3, 3]
    print("\nTable 1A (pooled) and Table S3 (divisions): r, ra, PBIAS (%), RMSE (mm), NSE, KGE")
    for key, exp in ref["Table 1A"].items():
        show("pooled", [pr.loc[key, c] for c in cols], exp, decs)
    for key, exp in ref.get("Table S3", {}).items():
        show(key, [pr.loc[key, c] for c in cols], exp, decs)

    ag = R["agree"].merge(R["events"][["scale", "POD", "FAR", "kappa"]], on="scale").set_index("scale")
    print("\nTable 1C / S6: n, r, RMSE, NSE, POD, FAR, kappa")
    for sc, exp in ref["Table 1C"].items():
        show(f"SPEI-{sc}", [ag.loc[sc, c] for c in ["n", "r", "RMSE", "NSE", "POD", "FAR", "kappa"]], exp,
             [0, 3, 3, 3, 3, 3, 3])

    sh = R["shift_tab"]
    print("\nTable S4")
    show("Kutubdia P r", [sh[sh.Station.str.startswith("Kutubdia")].P_r.iloc[0]], [ref["Table S4 Kutubdia P r"]], [3])

    print("\nFig. S2: full-record SPEI-3 correlation, station vs ERA5 precipitation")
    s = R["series"]
    for div, exp in ref["Fig. S2"].items():
        d = s[s.division == div]
        k = d.SPEI3_b.notna() & d.SPEI3_e.notna()
        show(f"{div} (n = {int(k.sum())})", [np.corrcoef(d.SPEI3_b[k], d.SPEI3_e[k])[0, 1]], [exp], [3])

    out = V.OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    R["precip"].to_csv(out / "table1_precipitation_validation.csv", index=False)
    R["pet_sens"].to_csv(out / "table2_pet_sensitivity.csv", index=False)
    R["agree"].to_csv(out / "table3_spei_agreement.csv", index=False)
    R["events"].to_csv(out / "table4_drought_event_detection.csv", index=False)
    R["shift_tab"].to_csv(out / "table5_per_station_month_shifts.csv", index=False)
    R["series"].to_csv(out / "spei_series_bmd_vs_era5.csv", index=False)
    print(f"\nTier-1 tables written to {out}")
    print(f"{'All values match the paper.' if not fails else f'{fails} value(s) differ.'}"
          f"{'  (numbers before the zero-value correction)' if a.no_rule else ''}")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
