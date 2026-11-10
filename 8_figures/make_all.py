"""
make_all.py  --  builds every figure of the paper made from code, and writes the check log.
(Fig. 1, Fig. 2 and Fig. S3 are not made from code: see the README.)

    python make_all.py                 # all figures
    python make_all.py --only Fig04    # one figure (comma-separated list allowed)

Outputs go to FIG_OUT (p1_fig_config.py):
    FigXX.pdf                 vector PDF, fonts embedded (submission)
    FigXX.tif                 600 dpi LZW TIFF (submission)
    FigXX.png                 600 dpi PNG (paste into the Word draft; also fine for submission)
    figures_check_log.md      every data check that ran
"""

import argparse
import importlib
import sys
import time

import p1_fig_config as C
import paths as P
import fig_data as D

# figure id -> module (added batch by batch)
FIGURES = {
    "Fig03": "fig03_screening",      # writes Fig03 and the split pair Fig03a / Fig03b
    "Fig04": "fig04_lead1_metrics",
    "Fig05": "fig05_fidelity",       # also writes Fig05a and Fig05b (split pair)
    "Fig06": "fig06_detection",
    "Fig07": "fig07_multilead",
    "Fig08": "fig08_spei12_seeds",
    "Fig09": "fig09_maps",           # writes Fig09 (worst month) and FigS08 (near-normal month); needs geopandas
    "Fig10": "fig10_shap",           # SHAP bars (a-c) and beeswarms (d-f)
    "FigS01": "figS01_validation",   # writes FigS01 and FigS02 (needs the BMD file; run Validation_1.py first)
    "FigS04": "figS04_acf",          # needs statsmodels
    "FigS05": "figS05_architectures",  # writes FigS05 (a-e combined) and FigS05a ... FigS05e
    "FigS06": "figS06_series",       # writes FigS06 and FigS07
    "FigS09": "figS09_shap_shares",
    "FigS10": "figS10_refit",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated figure ids, e.g. Fig04,Fig06")
    a = ap.parse_args()
    todo = [f.strip() for f in a.only.split(",") if f.strip()] or list(FIGURES)
    unknown = [f for f in todo if f not in FIGURES]
    if unknown:
        sys.exit(f"unknown figure id(s): {unknown}; choose from {list(FIGURES)}")

    print("Paths:")
    if not C.check_paths(["OUT_ROOT", "TABLES_DIR", "SPEI_CSV", "VARS_CSV", "PETWB_CSV"]):
        sys.exit("Fix the MISSING path(s) (paths.py), then run again.")
    if a.only == "":                       # skip figures whose external inputs are absent
        if not C.SHAPEFILE.exists():
            print("  shapefile not found: Fig09/FigS08 skipped (see the README)")
            todo = [f for f in todo if f != "Fig09"]
        if not P.BMD_CSV.exists():
            print("  BMD file not found: FigS01/FigS02 skipped (see the README)")
            todo = [f for f in todo if f != "FigS01"]
    C.FIG_OUT.mkdir(parents=True, exist_ok=True)

    log = D.CheckLog()
    log.lines.append("# Figure check log\n")
    log.lines.append(f"Run: {time.strftime('%Y-%m-%d %H:%M')}  |  figures: {', '.join(todo)}")
    total = 0
    for fid in todo:
        print(f"\n{fid}:")
        mod = importlib.import_module(FIGURES[fid])
        problems = mod.make(C.FIG_OUT, log) or []
        total += len(problems)
        for p in problems:
            log.lines.append(f"- LAYOUT: {fid}: {p}")
    log.write(C.FIG_OUT / "figures_check_log.md")
    print(f"\nDone. {total} layout problem(s). Check log: {C.FIG_OUT / 'figures_check_log.md'}")
    print("Next: python check_figs.py  (page sizes, embedded fonts, TIFF resolution)")


if __name__ == "__main__":
    main()
