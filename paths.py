"""
paths.py  --  the one place that says where everything is (Paper 1 code package).

All paths are relative to this package. The expected layout is the code folder
(this repository, any name) with the Zenodo data archive unzipped next to it:

    <any folder>/
        <code folder>/  this code (paths.py lives here)
        P1_Data/        inputs, archived runs and the tables behind the paper
        P1_work/        created on first use: everything the scripts rebuild

Override any of the three roots with an environment variable if your layout differs:
    P1_DATA=/path/to/P1_Data     P1_WORK=/path/to/scratch
(the stage-2 code also honours P1_OUT_ROOT, P1_SPEI_CSV, ... as before).
"""

import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("P1_DATA", CODE_ROOT.parent / "P1_Data"))
WORK = Path(os.environ.get("P1_WORK", CODE_ROOT.parent / "P1_work"))

# ---- archived inputs (ERA5-derived divisional series, shareable) -------------
INPUTS = DATA_ROOT / "inputs"
VARS_CSV = INPUTS / "divisional_averaged_1980_2024.csv"          # 9 ERA5 variables, 8 divisions
PETWB_CSV = INPUTS / "divisional_pet_waterbalance_1980_2024.csv"  # FAO-56 PET and P - PET
SPEI_CSV = INPUTS / "divisional_spei_1980_2024.csv"               # SPEI-1/3/6/12

# ---- archived results ---------------------------------------------------------
STAGE2_OUT = DATA_ROOT / "stage2_output"   # 135 runs (+5 strict), lead-1 models, SHAP, refit SPEI
TABLES = DATA_ROOT / "tables"              # CSVs behind the paper and SI tables (final numbers)

# ---- external inputs: NOT distributed (see the P1_Data README) ----------------
EXTERNAL = DATA_ROOT / "external"
ERA5_NC = EXTERNAL / "dataset_ninevar_1980_2024_bd_clipped.nc"     # raw ERA5 monthly means, clipped
BMD_CSV = EXTERNAL / "Final-Rainfall_Report_BMD_MENDELAY.csv"       # BMD 35-station compilation
SHAPEFILE = EXTERNAL / "bd_div_wgs.shp"                             # 8 division polygons
DEM_DIR = EXTERNAL / "dem"                                          # elevation rasters (*.tif)


def pick(*candidates):
    """First path that exists (used to prefer a rebuilt file over the archived one)."""
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(candidates[-1])


if __name__ == "__main__":
    for name in ["CODE_ROOT", "DATA_ROOT", "WORK", "VARS_CSV", "PETWB_CSV", "SPEI_CSV", "STAGE2_OUT",
                 "TABLES", "ERA5_NC", "BMD_CSV", "SHAPEFILE", "DEM_DIR"]:
        p = globals()[name]
        print(f"  {'OK     ' if Path(p).exists() else 'missing'} {name:<11} {p}")
    print("  (ERA5_NC, BMD_CSV, SHAPEFILE and DEM_DIR are external inputs and are not distributed.)")
