# =====================================================================
#  01_divisional_average.py
#  SPEI PIPELINE - STAGE 1 of 3
# ---------------------------------------------------------------------
#  Purpose : Collapse the 274 ERA5 land cells (0.25 deg) into 8 Bangladesh
#            administrative divisions by AREA-AVERAGING the 9 variables
#            per division, per month, using point-in-polygon assignment.
#
#  Input   : P1_Data/external/dataset_ninevar_1980_2024_bd_clipped.nc   (540 months, 9 vars; see P1_Data README)
#            P1_Data/external/bd_div_wgs.shp                            (8 division polygons; not distributed)
#
#  Output  : divisional_averaged_1980_2024.csv   (8 div x 540 months, wide format)
#            cell_to_division_mapping.csv         (audit: which cell -> which division)
#
#  NOTE    : This is STAGE 1. It does NOT compute PET or SPEI and does NOT
#            need the DEM/elevation. It only produces the averaged inputs
#            that STAGE 2 (PET) will consume.
#
#  Averaging philosophy: average the INPUT variables first, then (in stage 2)
#            compute PET -> D -> SPEI on the divisional series.
#            Per-cell SPEI (for maps) is a separate layer built later.
# =====================================================================

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

# ---------------------------------------------------------------------
#  CONFIG  - edit paths here only
# ---------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402  (one relative-path config for the whole package)
NC_PATH      = str(P.ERA5_NC)
SHAPE_DIR    = str(P.SHAPEFILE.parent)
OUT_DIR      = str(P.WORK / "1_data_and_index")   # rebuilt intermediates live here
OUT_CSV      = os.path.join(OUT_DIR, "divisional_averaged_1980_2024.csv")
OUT_MAP_CSV  = os.path.join(OUT_DIR, "cell_to_division_mapping.csv")

VARS = ['t2m', 'u10', 'v10', 'd2m', 'ssrd', 'tp', 'tcc', 'mx2t', 'mn2t']  # the 9 ERA5 variables

# Official 8 divisions (with common spelling variants) - used only to
# auto-detect the correct name field in the shapefile attribute table.
DIVISION_ALIASES = {
    'dhaka', 'chattogram', 'chittagong', 'rajshahi', 'khulna',
    'barishal', 'barisal', 'sylhet', 'rangpur', 'mymensingh'
}

EXPECTED_CELLS  = 274
EXPECTED_MONTHS = 540
EXPECTED_DIVS   = 8

os.makedirs(OUT_DIR, exist_ok=True)


def hr(title=""):
    print("=" * 64)
    if title:
        print("  " + title)
        print("=" * 64)


# =====================================================================
#  1. LOAD + VALIDATE THE DIVISION SHAPEFILE
# =====================================================================
hr("STAGE 1  |  DIVISIONAL AREA-AVERAGING")

shp_candidates = glob.glob(os.path.join(SHAPE_DIR, "*.shp"))
if not shp_candidates:
    raise FileNotFoundError(f"No .shp file found in {SHAPE_DIR}")
SHP_PATH = shp_candidates[0]
if len(shp_candidates) > 1:
    print(f"  WARNING: multiple .shp files found; using the first:\n    {SHP_PATH}")

gdf = gpd.read_file(SHP_PATH)

hr("SHAPEFILE - CRS CHECK")
print(f"  File            : {SHP_PATH}")
print(f"  Original CRS    : {gdf.crs}")

# Reproject to EPSG:4326 (lat/lon) to match ERA5 if needed.
if gdf.crs is None:
    raise ValueError("Shapefile has NO CRS (.prj missing?). Cannot safely reproject. "
                     "Provide the .prj or tell me the known CRS.")
if gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)
    print(f"  Reprojected to  : EPSG:4326  (was not 4326)")
else:
    print(f"  Already EPSG:4326 - no reprojection needed  [OK]")

# --- feature count gate (reveals shapefile vintage/level) -------------
hr("SHAPEFILE - FEATURE COUNT")
n_feat = len(gdf)
print(f"  Polygons found  : {n_feat}")
if n_feat == EXPECTED_DIVS:
    print(f"  [OK] 8 polygons - matches the 8-division plan.")
else:
    print(f"  [WARNING] Expected {EXPECTED_DIVS} divisions, got {n_feat}.")
    print(f"            7 => older (pre-Mymensingh 2015); 64 => districts (wrong file).")

# --- auto-detect the division NAME field ------------------------------
hr("SHAPEFILE - NAME FIELD DETECTION")
name_field = None
best_hits  = -1
for col in gdf.columns:
    if col.lower() == 'geometry':
        continue
    try:
        vals = gdf[col].astype(str).str.strip().str.lower()
    except Exception:
        continue
    hits = vals.isin(DIVISION_ALIASES).sum()
    if hits > best_hits:
        best_hits, name_field = hits, col

print(f"  Available fields: {[c for c in gdf.columns if c != 'geometry']}")
if name_field is None or best_hits == 0:
    print("  [WARNING] Could not auto-detect a division-name field.")
    print("            Inspect the fields above and set name_field manually.")
else:
    print(f"  Detected field  : '{name_field}'  ({best_hits}/{n_feat} names matched known divisions)")
    print(f"  Division names  : {sorted(gdf[name_field].astype(str).unique())}")


# =====================================================================
#  2. LOAD THE ERA5 CUBE + EXTRACT VALID LAND CELLS
# =====================================================================
ds = xr.open_dataset(NC_PATH)

# Auto-detect coordinate names (ERA5 files vary: time vs valid_time).
time_name = 'valid_time' if 'valid_time' in ds.coords else 'time'
lat_name  = 'latitude'   if 'latitude'   in ds.coords else 'lat'
lon_name  = 'longitude'  if 'longitude'  in ds.coords else 'lon'

hr("ERA5 CUBE - OVERVIEW")
print(f"  File        : {NC_PATH}")
print(f"  Time coord  : '{time_name}'  ({ds.sizes[time_name]} steps)")
print(f"  Lat coord   : '{lat_name}'   ({ds.sizes[lat_name]})")
print(f"  Lon coord   : '{lon_name}'   ({ds.sizes[lon_name]})")
print(f"  Variables   : {list(ds.data_vars)}")

# confirm all 9 expected variables are present
missing = [v for v in VARS if v not in ds.data_vars]
if missing:
    raise KeyError(f"Missing expected variables in the .nc file: {missing}")
print(f"  [OK] all 9 expected variables present.")

if ds.sizes[time_name] != EXPECTED_MONTHS:
    print(f"  [WARNING] expected {EXPECTED_MONTHS} months, got {ds.sizes[time_name]}.")

# --- flatten spatial grid to a 'cell' axis, keep only valid land cells -
# A cell is 'land' if it holds valid (non-NaN) data. Clipped cells outside
# Bangladesh are NaN and get dropped here.
stacked = ds.stack(cell=(lat_name, lon_name))
valid_mask = stacked['t2m'].isel({time_name: 0}).notnull().values
stacked = stacked.isel(cell=valid_mask)

n_cells = stacked.sizes['cell']
cell_lats = stacked[lat_name].values      # aligned to 'cell' order
cell_lons = stacked[lon_name].values

hr("ERA5 CUBE - VALID LAND CELLS")
print(f"  Valid land cells: {n_cells}")
if n_cells == EXPECTED_CELLS:
    print(f"  [OK] matches expected {EXPECTED_CELLS} cells.")
else:
    print(f"  [WARNING] expected {EXPECTED_CELLS}, got {n_cells}. Investigate before trusting output.")


# =====================================================================
#  3. POINT-IN-POLYGON  - assign each cell to a division
# =====================================================================
# Build cell centroids as points IN THE SAME ORDER as 'stacked.cell',
# so the resulting labels line up exactly when we assign them back.
cells_gdf = gpd.GeoDataFrame(
    {'cell_idx': np.arange(n_cells),
     'lat': cell_lats,
     'lon': cell_lons},
    geometry=[Point(lo, la) for lo, la in zip(cell_lons, cell_lats)],
    crs="EPSG:4326"
)

div_gdf = gdf[[name_field, 'geometry']].rename(columns={name_field: 'division'})

# primary assignment: centroid strictly within a division polygon
joined = gpd.sjoin(cells_gdf, div_gdf, how='left', predicate='within')
# sjoin can duplicate a cell if polygons overlap on a boundary - keep first.
joined = joined[~joined['cell_idx'].duplicated(keep='first')].sort_values('cell_idx')

# fallback: any cell not caught by 'within' (edge/coastal mismatch between
# the national clip and the division polygons) -> nearest division polygon.
n_unassigned = joined['division'].isna().sum()
if n_unassigned > 0:
    miss = cells_gdf[joined['division'].isna().values].copy()
    nearest = gpd.sjoin_nearest(miss, div_gdf, how='left')
    nearest = nearest[~nearest['cell_idx'].duplicated(keep='first')]
    fill = dict(zip(nearest['cell_idx'], nearest['division']))
    joined['division'] = [
        fill.get(ci, dv) for ci, dv in zip(joined['cell_idx'], joined['division'])
    ]

division_labels = joined.sort_values('cell_idx')['division'].astype(str).to_numpy(dtype=object)

hr("POINT-IN-POLYGON - CELL ASSIGNMENT")
counts = pd.Series(division_labels).value_counts().sort_index()
for dname, c in counts.items():
    print(f"    {str(dname):14s} : {c:3d} cells")
print(f"  {'-'*30}")
print(f"    {'TOTAL':14s} : {int(counts.sum()):3d} cells")
if int(counts.sum()) == n_cells and counts.min() >= 1:
    print(f"  [OK] every cell assigned; every division has >=1 cell.")
else:
    print(f"  [WARNING] assignment problem - check counts above.")
if n_unassigned > 0:
    print(f"  NOTE: {n_unassigned} edge cell(s) used nearest-division fallback (not strict 'within').")


# =====================================================================
#  4. AREA-AVERAGE THE 9 VARIABLES PER DIVISION, PER MONTH
# =====================================================================
# Attach division label to the 'cell' axis, then group + mean over cells.
stacked = stacked.assign_coords(division=('cell', division_labels))
div_mean = stacked[VARS].groupby('division').mean('cell')     # dims: (time, division)

# reshape to a tidy WIDE table: one row per (division, month)
df = div_mean.to_dataframe().reset_index()                    # cols: time, division, <9 vars>
df = df.rename(columns={time_name: 'date'})
df['date']  = pd.to_datetime(df['date'])
df['year']  = df['date'].dt.year
df['month'] = df['date'].dt.month
df = df[['division', 'date', 'year', 'month'] + VARS]
df = df.sort_values(['division', 'date']).reset_index(drop=True)


# =====================================================================
#  5. VERIFICATION  - NaN audit + physical-range sanity
# =====================================================================
hr("OUTPUT - NaN AUDIT")
nan_total = int(df[VARS].isnull().sum().sum())
if nan_total == 0:
    print(f"  [OK] zero NaNs across all divisional-averaged variables.")
else:
    print(f"  [WARNING] {nan_total} NaN value(s) in the averaged output:")
    print(df[VARS].isnull().sum())

hr("OUTPUT - PHYSICAL SANITY RANGES  (post-conversion units)")
sanity = {
    't2m':  ("mean air temp  degC", 8,  35),
    'mx2t': ("mean daily max degC", 12, 42),
    'mn2t': ("mean daily min degC", 4,  32),
    'd2m':  ("dewpoint       degC", 2,  30),
    'tp':   ("precip      mm/day",  0,  40),
    'ssrd': ("solar   MJ/m2/day",   3,  32),
    'tcc':  ("cloud cover  0..1",   0,  1),
    'u10':  ("wind-U        m/s",  -12, 12),
    'v10':  ("wind-V        m/s",  -12, 12),
}
for v, (label, lo, hi) in sanity.items():
    vmin, vmax = float(df[v].min()), float(df[v].max())
    flag = "[OK]" if (vmin >= lo and vmax <= hi) else "[CHECK]"
    print(f"    {v:5s} {label:20s}: {vmin:8.3f} -> {vmax:8.3f}   expect [{lo}, {hi}]  {flag}")

# structural check
hr("OUTPUT - STRUCTURE")
n_div_out = df['division'].nunique()
print(f"  Rows            : {len(df)}   (expect {EXPECTED_DIVS} x {EXPECTED_MONTHS} = {EXPECTED_DIVS*EXPECTED_MONTHS})")
print(f"  Divisions       : {n_div_out}  -> {sorted(df['division'].unique())}")
print(f"  Date range      : {df['date'].min().date()}  ->  {df['date'].max().date()}")


# =====================================================================
#  6. SAVE OUTPUTS
# =====================================================================
df.to_csv(OUT_CSV, index=False)

map_df = pd.DataFrame({'lat': cell_lats, 'lon': cell_lons, 'division': division_labels})
map_df.to_csv(OUT_MAP_CSV, index=False)

hr("SAVED")
print(f"  Divisional averages : {OUT_CSV}")
print(f"  Cell->division map  : {OUT_MAP_CSV}")
print("=" * 64)
print("  STAGE 1 COMPLETE.  Inspect the CSV, then proceed to 02_compute_pet.py")
print("=" * 64)
