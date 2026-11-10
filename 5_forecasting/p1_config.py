"""
================================================================================
p1_config.py  --  THE ONLY FILE WITH FOLDER PATHS  (Paper 1, Stage 2)
================================================================================
Every Stage 2 script reads its paths from here. If a folder on your PC is
different, change it in this file and nowhere else.

The defaults are the folders your thesis scripts already used. Results from
Stage 2 go to a NEW folder (OUT_ROOT below), so nothing from the thesis or the
lead-time extension is overwritten.

Any path can also be overridden with an environment variable named
P1_<NAME> (for example P1_OUT_ROOT). You do not need this on your PC; it is
how the same code runs unchanged on a second machine.
================================================================================
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402

# ---------------------------------------------------------------------------
# EDIT HERE if a folder on your PC is different
# ---------------------------------------------------------------------------
_DEFAULTS = {
    # inputs (P1_Data/inputs)
    "SPEI_CSV":  str(P.SPEI_CSV),
    "VARS_CSV":  str(P.VARS_CSV),
    "PETWB_CSV": str(P.PETWB_CSV),

    # the archived runs (P1_Data/stage2_output). The run scripts skip finished runs,
    # so to retrain from scratch set P1_OUT_ROOT to an empty folder.
    "OUT_ROOT":  str(P.STAGE2_OUT),
}
# ---------------------------------------------------------------------------


def _get(name):
    return os.environ.get("P1_" + name, _DEFAULTS[name])


SPEI_CSV  = _get("SPEI_CSV")
VARS_CSV  = _get("VARS_CSV")
PETWB_CSV = _get("PETWB_CSV")
OUT_ROOT  = Path(_get("OUT_ROOT"))

# SPEI refitted on 1980-2018 only (written by p1_spei_refit.py, used by the
# F1+F2 sensitivity run). Lives inside OUT_ROOT so it can never overwrite the
# thesis SPEI file.
SPEI_REFIT_CSV = OUT_ROOT / "_spei_refit" / "divisional_spei_1980_2024_refit1980-2018.csv"

# Which pipeline the scripts build and train.
#   "main"   : the thesis pipeline (SPEI fitted on 1980-2024; climate anomalies
#              from all years). This is the pipeline the paper reports.
#   "strict" : the F1+F2 sensitivity pipeline (SPEI fitted on 1980-2018 only;
#              climate anomalies from training years only). run11 sets this.
PIPELINE = os.environ.get("P1_PIPELINE", "main").strip().lower()
if PIPELINE not in ("main", "strict"):
    raise ValueError(f"P1_PIPELINE must be 'main' or 'strict', got {PIPELINE!r}")
