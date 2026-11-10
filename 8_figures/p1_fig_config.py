"""
================================================================================
p1_fig_config.py  --  folder paths of the figure code (from paths.py)
================================================================================
Every figure script reads its paths from here. If a folder on your PC is
different, change it in this file and nowhere else.

The defaults are the folders already used by the thesis scripts and by Stage 2
(p1_config.py). Figures are written to a NEW folder (FIG_OUT), so nothing from
the thesis or Stage 2 is overwritten.

Any path can also be overridden with an environment variable named P1_<NAME>
(for example P1_OUT_ROOT). You do not need this on your PC; it is how the same
code runs unchanged on a second machine.
================================================================================
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults come from paths.py (all relative to the package); any can be
# overridden with an environment variable P1_<NAME>.
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "OUT_ROOT":        str(P.STAGE2_OUT),        # archived runs, predictions, SHAP values
    "TABLES_DIR":      str(P.TABLES),            # the CSVs behind the paper tables (final numbers)
    "SPEI_CSV":        str(P.SPEI_CSV),
    "VARS_CSV":        str(P.VARS_CSV),
    "PETWB_CSV":       str(P.PETWB_CSV),
    "LEADLAG_SUMMARY": str(P.WORK / "3_predictor_screening" / "leadlag" / "leadlag_summary.csv"),  # optional check
    "SHAPEFILE":       str(P.SHAPEFILE),         # not distributed (Figs 9 and S8)
    "VALIDATION_DIR":  str(P.WORK / "2_era5_validation"),   # outputs of Validation_1.py (Figs S1-S2 check)
    "FIG_OUT":         str(P.WORK / "figures"),
}
# ---------------------------------------------------------------------------


def _get(name):
    return os.environ.get("P1_" + name, _DEFAULTS[name])


OUT_ROOT       = Path(_get("OUT_ROOT"))
TABLES_DIR     = Path(_get("TABLES_DIR"))
SPEI_CSV       = Path(_get("SPEI_CSV"))
VARS_CSV       = Path(_get("VARS_CSV"))
PETWB_CSV      = Path(_get("PETWB_CSV"))
LEADLAG_SUMMARY = Path(_get("LEADLAG_SUMMARY"))
SHAPEFILE      = Path(_get("SHAPEFILE"))
VALIDATION_DIR = Path(_get("VALIDATION_DIR"))
FIG_OUT        = Path(_get("FIG_OUT"))


def check_paths(names):
    """Print True/False for each named path; return True only if all exist."""
    ok = True
    for n in names:
        p = globals()[n]
        e = p.exists()
        ok &= e
        print(f"  {'OK     ' if e else 'MISSING'} {n:<15} {p}")
    return ok


if __name__ == "__main__":
    check_paths(["OUT_ROOT", "TABLES_DIR", "SPEI_CSV", "VARS_CSV", "PETWB_CSV",
                 "SHAPEFILE", "VALIDATION_DIR", "LEADLAG_SUMMARY"])
    print("  (LEADLAG_SUMMARY is optional: without it the Fig 3 lead-lag check is skipped)")
    print("  figures will be written to:", FIG_OUT)
