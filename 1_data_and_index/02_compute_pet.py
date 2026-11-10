# =====================================================================
#  02_compute_pet.py
#  SPEI PIPELINE - STAGE 2 of 3
# ---------------------------------------------------------------------
#  Purpose : Compute reference evapotranspiration ET0 (PET) via the full
#            FAO-56 Penman-Monteith method for each of the 8 divisions,
#            then form the water balance  D = P - PET.
#
#  Input   : divisional_averaged_1980_2024.csv   (from stage 1)
#            cell_to_division_mapping.csv        (for division centroid latitude)
#            P1_Data/external/dem/*.tif      (elevation -> pressure; not distributed)
#            P1_Data/external/bd_div_wgs.shp (polygons for zonal elevation; not distributed)
#  The archived output keeps elev_m, pressure_kPa and every FAO-56 term, so PET
#  can be audited without the DEM.
#
#  Output  : divisional_pet_waterbalance_1980_2024.csv
#            division_elevation_summary.csv
#
#  METHOD DECISIONS:
#    * G (soil heat flux)  : monthly form  G = 0.14 * (T_i - T_i-1),
#                            first month of each division seeded to 0.
#    * Atmospheric pressure: DEM-derived mean elevation per division.
#    * Net radiation       : FULL FAO-56 longwave step (Rn = Rns - Rnl),
#                            NOT the Rn ~ Rns shortcut.
#    * Wind                : u10/v10 -> speed -> u2 via factor 0.748.
#    * Units discipline    : P and PET both kept in mm/day, differenced to
#                            form D, and ONLY THEN scaled to a monthly total.
#
#  ET0 = [0.408*D*(Rn-G) + g*(900/(T+273))*u2*(es-ea)] / [D + g*(1+0.34*u2)]
#         where D = slope of vapour pressure curve, g = psychrometric const.
#
#  NOTE: intermediate FAO-56 terms are kept as columns on purpose, so every
#        number in ET0 is auditable/defensible rather than a black box.
# =====================================================================

import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd

# rasterio is used for the zonal (per-division) mean elevation
import rasterio
from rasterio.mask import mask as rio_mask

# ---------------------------------------------------------------------
#  CONFIG - edit paths here only
# ---------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402  (one relative-path config for the whole package)
IN_DIR      = str(P.WORK / "1_data_and_index")          # output of 01_divisional_average.py
IN_CSV      = os.path.join(IN_DIR, "divisional_averaged_1980_2024.csv")
MAP_CSV     = os.path.join(IN_DIR, "cell_to_division_mapping.csv")
DEM_DIR     = str(P.DEM_DIR)
SHAPE_DIR   = str(P.SHAPEFILE.parent)

OUT_DIR     = str(P.WORK / "1_data_and_index")

OUT_CSV     = os.path.join(OUT_DIR, "divisional_pet_waterbalance_1980_2024.csv")
OUT_ELEV    = os.path.join(OUT_DIR, "division_elevation_summary.csv")

# ---- physical constants (FAO-56) ------------------------------------
G_SC        = 0.0820          # solar constant,        MJ m-2 min-1
SIGMA       = 4.903e-9        # Stefan-Boltzmann,      MJ K-4 m-2 day-1
ALBEDO      = 0.23            # reference crop albedo  (-)
WIND_FACTOR = 0.748           # 10 m -> 2 m wind adjustment (4.87/ln(67.8*10-5.42))

os.makedirs(OUT_DIR, exist_ok=True)


def hr(t=""):
    print("=" * 68)
    if t:
        print("  " + t)
        print("=" * 68)


hr("STAGE 2  |  FAO-56 PENMAN-MONTEITH PET  +  WATER BALANCE")

# =====================================================================
#  1. LOAD STAGE-1 OUTPUT
# =====================================================================
df = pd.read_csv(IN_CSV, parse_dates=['date'])
df = df.sort_values(['division', 'date']).reset_index(drop=True)

hr("INPUT - STAGE 1 TABLE")
print(f"  File       : {IN_CSV}")
print(f"  Rows       : {len(df)}")
print(f"  Divisions  : {df['division'].nunique()} -> {sorted(df['division'].unique())}")
print(f"  Date range : {df['date'].min().date()} -> {df['date'].max().date()}")

req = ['t2m', 'mx2t', 'mn2t', 'd2m', 'ssrd', 'tp', 'u10', 'v10']
missing = [c for c in req if c not in df.columns]
if missing:
    raise KeyError(f"Stage-1 CSV missing required columns: {missing}")
print(f"  [OK] all required variables present.")


# =====================================================================
#  2. DIVISION LATITUDE  (needed for extraterrestrial radiation Ra)
# =====================================================================
# Use the MEAN LATITUDE OF THE ASSIGNED CELLS, not the polygon centroid:
# the climate series is the average of those cells, so their mean latitude
# is the physically consistent latitude for the radiation calculation.
mapping = pd.read_csv(MAP_CSV)
lat_by_div = mapping.groupby('division')['lat'].mean().to_dict()

hr("DIVISION LATITUDE  (mean of assigned ERA5 cells)")
for d in sorted(lat_by_div):
    print(f"    {d:14s} : {lat_by_div[d]:7.3f} deg N")


# =====================================================================
#  3. DEM  ->  MEAN ELEVATION PER DIVISION  ->  ATMOSPHERIC PRESSURE
# =====================================================================
hr("DEM - ZONAL MEAN ELEVATION PER DIVISION")

dem_files = (glob.glob(os.path.join(DEM_DIR, "*.tif"))
             + glob.glob(os.path.join(DEM_DIR, "*.TIF"))
             + glob.glob(os.path.join(DEM_DIR, "*.img"))
             + glob.glob(os.path.join(DEM_DIR, "*.vrt")))
if not dem_files:
    raise FileNotFoundError(
        f"No raster (.tif/.img/.vrt) found in {DEM_DIR}. "
        f"Folder contents: {os.listdir(DEM_DIR) if os.path.isdir(DEM_DIR) else 'PATH NOT FOUND'}"
    )
DEM_PATH = dem_files[0]
if len(dem_files) > 1:
    print(f"  WARNING: {len(dem_files)} rasters found; using the first:\n    {DEM_PATH}")
print(f"  DEM file   : {DEM_PATH}")

# division polygons, reprojected to the DEM's CRS for correct masking
shp = glob.glob(os.path.join(SHAPE_DIR, "*.shp"))[0]
gdf = gpd.read_file(shp)

aliases = {'dhaka', 'chattogram', 'chittagong', 'rajshahi', 'khulna',
           'barishal', 'barisal', 'sylhet', 'rangpur', 'mymensingh'}
name_field, best = None, -1
for c in gdf.columns:
    if c == 'geometry':
        continue
    try:
        hits = gdf[c].astype(str).str.strip().str.lower().isin(aliases).sum()
    except Exception:
        continue
    if hits > best:
        best, name_field = hits, c
if name_field is None or best == 0:
    raise ValueError(f"Could not detect division-name field. Fields: {list(gdf.columns)}")

elev_rows = []
with rasterio.open(DEM_PATH) as src:
    print(f"  DEM CRS    : {src.crs}")
    print(f"  DEM size   : {src.width} x {src.height}   nodata={src.nodata}")

    # reproject polygons to match the raster (avoids the metres/degrees trap)
    gdf_dem = gdf.to_crs(src.crs) if (gdf.crs is not None and src.crs is not None
                                      and gdf.crs != src.crs) else gdf

    for _, row in gdf_dem.iterrows():
        dname = str(row[name_field]).strip()
        try:
            arr, _ = rio_mask(src, [row.geometry], crop=True, filled=True,
                              nodata=(src.nodata if src.nodata is not None else -9999))
            a = arr[0].astype('float64')
            nod = src.nodata if src.nodata is not None else -9999
            a = np.where(a == nod, np.nan, a)
            # guard against un-flagged fill values (e.g. -32768 / -9999 leaking in)
            a = np.where(a < -500, np.nan, a)
            mean_z = float(np.nanmean(a))
            min_z, max_z = float(np.nanmin(a)), float(np.nanmax(a))
            npix = int(np.sum(~np.isnan(a)))
        except Exception as e:
            print(f"  [WARNING] zonal stats failed for {dname}: {e}  -> using 0 m")
            mean_z, min_z, max_z, npix = 0.0, np.nan, np.nan, 0
        elev_rows.append({'division': dname, 'mean_elev_m': mean_z,
                          'min_elev_m': min_z, 'max_elev_m': max_z, 'dem_pixels': npix})

elev_df = pd.DataFrame(elev_rows)
# pressure from elevation (FAO-56 eq. 7)
elev_df['pressure_kPa'] = 101.3 * (((293.0 - 0.0065 * elev_df['mean_elev_m']) / 293.0) ** 5.26)

print()
print(f"  {'division':14s} {'mean_z(m)':>10s} {'min':>8s} {'max':>8s} {'P(kPa)':>9s} {'pixels':>10s}")
for _, r in elev_df.sort_values('division').iterrows():
    print(f"    {r['division']:14s} {r['mean_elev_m']:9.1f} {r['min_elev_m']:8.1f} "
          f"{r['max_elev_m']:8.1f} {r['pressure_kPa']:9.3f} {r['dem_pixels']:10d}")

# match elevation names to the CSV's division names (tolerate spelling variants)
csv_divs = set(df['division'].unique())
elev_map = {}
for _, r in elev_df.iterrows():
    nm = r['division']
    if nm in csv_divs:
        elev_map[nm] = r
    else:
        cand = [d for d in csv_divs if d.strip().lower() == nm.strip().lower()]
        if cand:
            elev_map[cand[0]] = r
unmatched = csv_divs - set(elev_map)
if unmatched:
    print(f"\n  [WARNING] no DEM elevation matched for: {unmatched}  -> defaulting to 0 m / 101.3 kPa")

df['elev_m']       = df['division'].map(lambda d: float(elev_map[d]['mean_elev_m']) if d in elev_map else 0.0)
df['pressure_kPa'] = df['division'].map(lambda d: float(elev_map[d]['pressure_kPa']) if d in elev_map else 101.3)
df['lat_deg']      = df['division'].map(lambda d: lat_by_div.get(d, np.nan))
if df['lat_deg'].isna().any():
    raise ValueError("Some divisions have no latitude from the cell mapping - name mismatch?")


# =====================================================================
#  4. FAO-56 SUB-TERMS
# =====================================================================
hr("FAO-56 - DERIVING SUB-TERMS")

T    = df['t2m'].values          # mean air temperature, degC
Tmax = df['mx2t'].values         # mean daily maximum,   degC
Tmin = df['mn2t'].values         # mean daily minimum,   degC
Tdew = df['d2m'].values          # dewpoint temperature, degC
Rs   = df['ssrd'].values         # incoming solar,       MJ m-2 day-1
P_mm = df['tp'].values           # precipitation,        mm/day


def sat_vp(t):
    """Saturation vapour pressure e0(T) in kPa - FAO-56 eq. 11."""
    return 0.6108 * np.exp((17.27 * t) / (t + 237.3))


# --- vapour pressures -------------------------------------------------
es = (sat_vp(Tmax) + sat_vp(Tmin)) / 2.0        # FAO-56 eq. 12
ea = sat_vp(Tdew)                                # FAO-56 eq. 14 (from dewpoint)
# ea should not exceed es; tiny exceedances can occur from monthly averaging
ea = np.minimum(ea, es)
vpd = np.maximum(es - ea, 0.0)                   # vapour pressure deficit

# --- slope of the saturation vapour pressure curve (FAO-56 eq. 13) ----
delta = (4098.0 * (0.6108 * np.exp((17.27 * T) / (T + 237.3)))) / ((T + 237.3) ** 2)

# --- psychrometric constant (FAO-56 eq. 8) ----------------------------
gamma = 0.665e-3 * df['pressure_kPa'].values

# --- wind speed at 2 m (FAO-56 eq. 47) --------------------------------
u10 = np.sqrt(df['u10'].values ** 2 + df['v10'].values ** 2)
u2  = u10 * WIND_FACTOR
u2  = np.maximum(u2, 0.5)   # FAO-56: lower-bound u2 at 0.5 m/s for stability

print("  es, ea, VPD, slope(delta), gamma, u2   [OK]")

# --- extraterrestrial radiation Ra (FAO-56 eqs. 21-25) ----------------
# FAO-56 uses the middle day of the month: J = INT(30.4*M - 15)
Jday = (30.4 * df['month'].values - 15).astype(int)
phi  = np.deg2rad(df['lat_deg'].values)                  # latitude, radians
dr   = 1.0 + 0.033 * np.cos(2.0 * np.pi * Jday / 365.0)  # inverse rel. earth-sun distance
decl = 0.409 * np.sin(2.0 * np.pi * Jday / 365.0 - 1.39) # solar declination

ws_arg = np.clip(-np.tan(phi) * np.tan(decl), -1.0, 1.0)
ws     = np.arccos(ws_arg)                                # sunset hour angle

Ra = (24.0 * 60.0 / np.pi) * G_SC * dr * (
        ws * np.sin(phi) * np.sin(decl) +
        np.cos(phi) * np.cos(decl) * np.sin(ws))

# --- clear-sky radiation Rso (FAO-56 eq. 37) --------------------------
Rso = (0.75 + 2e-5 * df['elev_m'].values) * Ra

# --- net shortwave / net longwave / net radiation ---------------------
Rns = (1.0 - ALBEDO) * Rs                                 # FAO-56 eq. 38

ratio = np.divide(Rs, Rso, out=np.zeros_like(Rs), where=(Rso > 0))
ratio = np.clip(ratio, 0.0, 1.0)                          # FAO-56: Rs/Rso <= 1

TmaxK4 = (Tmax + 273.16) ** 4
TminK4 = (Tmin + 273.16) ** 4
Rnl = (SIGMA * ((TmaxK4 + TminK4) / 2.0)
       * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0.0)))
       * (1.35 * ratio - 0.35))                           # FAO-56 eq. 39
Rnl = np.maximum(Rnl, 0.0)                                # net longwave loss is >= 0

Rn = Rns - Rnl                                            # FAO-56 eq. 40
print("  Ra, Rso, Rns, Rnl, Rn                  [OK]  (full longwave step applied)")

# --- soil heat flux G, monthly form (FAO-56 eq. 43) -------------------
# G = 0.14 * (T_month - T_previous_month), computed WITHIN each division.
df['_T'] = T
G = df.groupby('division')['_T'].diff().values * 0.14
G = np.where(np.isnan(G), 0.0, G)     # seed the first month of each division to 0
df.drop(columns='_T', inplace=True)
print("  G = 0.14*(Ti - Ti-1)                   [OK]  (first month seeded to 0)")


# =====================================================================
#  5. ET0  (FAO-56 PENMAN-MONTEITH, eq. 6)
# =====================================================================
numerator   = 0.408 * delta * (Rn - G) + gamma * (900.0 / (T + 273.0)) * u2 * vpd
denominator = delta + gamma * (1.0 + 0.34 * u2)
ET0 = numerator / denominator
ET0 = np.maximum(ET0, 0.0)            # ET0 cannot be negative

hr("ET0 COMPUTED")
print(f"  ET0 range  : {ET0.min():.3f} -> {ET0.max():.3f} mm/day   (mean {ET0.mean():.3f})")


# =====================================================================
#  6. WATER BALANCE   D = P - PET
# =====================================================================
# CRITICAL UNITS DISCIPLINE: P and PET are BOTH mm/day here. Difference
# them FIRST, and only then scale to a monthly total.
D_daily = P_mm - ET0                                   # mm/day
days_in_month = df['date'].dt.days_in_month.values      # per-month day count
D_month = D_daily * days_in_month                       # mm/month  -> feeds SPEI

# also store P and PET as monthly totals for reporting/interpretation
P_month   = P_mm * days_in_month
PET_month = ET0 * days_in_month


# =====================================================================
#  7. ASSEMBLE OUTPUT TABLE  (intermediates kept for auditability)
# =====================================================================
out = df[['division', 'date', 'year', 'month',
          't2m', 'mx2t', 'mn2t', 'd2m', 'ssrd', 'tp', 'tcc', 'u10', 'v10',
          'lat_deg', 'elev_m', 'pressure_kPa']].copy()

out['es_kPa']        = es
out['ea_kPa']        = ea
out['vpd_kPa']       = vpd
out['delta_kPa_C']   = delta
out['gamma_kPa_C']   = gamma
out['wind10_ms']     = u10
out['u2_ms']         = u2
out['Ra_MJ']         = Ra
out['Rso_MJ']        = Rso
out['Rns_MJ']        = Rns
out['Rnl_MJ']        = Rnl
out['Rn_MJ']         = Rn
out['G_MJ']          = G
out['PET_mm_day']    = ET0
out['P_mm_day']      = P_mm
out['D_mm_day']      = D_daily
out['days_in_month'] = days_in_month
out['PET_mm_month']  = PET_month
out['P_mm_month']    = P_month
out['D_mm_month']    = D_month          # <-- the series stage 3 (SPEI) consumes


# =====================================================================
#  8. VERIFICATION
# =====================================================================
hr("VERIFY - NaN AUDIT")
nan_cols = out.isnull().sum()
nan_cols = nan_cols[nan_cols > 0]
if len(nan_cols) == 0:
    print("  [OK] zero NaNs across all columns.")
else:
    print("  [WARNING] NaNs present:")
    print(nan_cols)

hr("VERIFY - PHYSICAL SANITY RANGES")
checks = {
    'es_kPa':      ("sat. vapour press. kPa",  1.0,  6.0),
    'ea_kPa':      ("act. vapour press. kPa",  0.5,  5.0),
    'vpd_kPa':     ("vapour press. deficit",   0.0,  3.0),
    'u2_ms':       ("wind at 2 m       m/s",   0.3,  8.0),
    'Ra_MJ':       ("extraterr. rad  MJ/m2/d", 25.0, 42.0),
    'Rso_MJ':      ("clear-sky rad   MJ/m2/d", 18.0, 33.0),
    'Rns_MJ':      ("net shortwave   MJ/m2/d", 2.0,  25.0),
    'Rnl_MJ':      ("net longwave    MJ/m2/d", 0.0,  9.0),
    'Rn_MJ':       ("net radiation   MJ/m2/d", 0.0,  20.0),
    'G_MJ':        ("soil heat flux  MJ/m2/d", -1.5, 1.5),
    'PET_mm_day':  ("PET (ET0)         mm/d",  0.5,  9.0),
}
for c, (lab, lo, hi) in checks.items():
    v = out[c]
    flag = "[OK]" if (v.min() >= lo and v.max() <= hi) else "[CHECK]"
    print(f"    {c:14s} {lab:24s}: {v.min():9.3f} -> {v.max():9.3f}  expect [{lo}, {hi}]  {flag}")

hr("VERIFY - ANNUAL PET TOTALS PER DIVISION  (expect ~1000-1700 mm/yr for BD)")
ann = out.groupby(['division', 'year'])['PET_mm_month'].sum().groupby('division').mean()
for d, v in ann.sort_index().items():
    flag = "[OK]" if 900 <= v <= 1800 else "[CHECK]"
    print(f"    {d:14s} : {v:8.1f} mm/yr   {flag}")

hr("VERIFY - SEASONAL WATER BALANCE  (D should be << 0 in the dry season, >> 0 in monsoon)")
seas = out.groupby('month')[['P_mm_month', 'PET_mm_month', 'D_mm_month']].mean()
print(f"    {'month':>6s} {'P(mm)':>10s} {'PET(mm)':>10s} {'D(mm)':>11s}")
for m, r in seas.iterrows():
    print(f"    {m:6d} {r['P_mm_month']:10.1f} {r['PET_mm_month']:10.1f} {r['D_mm_month']:11.1f}")
print("\n  Expect: Dec-Mar strongly NEGATIVE (dry season), Jun-Sep strongly POSITIVE (monsoon).")

hr("VERIFY - LONG-TERM ANNUAL WATER BALANCE PER DIVISION")
annD = out.groupby(['division', 'year'])['D_mm_month'].sum().groupby('division').mean()
for d, v in annD.sort_index().items():
    print(f"    {d:14s} : {v:9.1f} mm/yr")
print("\n  Expect POSITIVE for all divisions (BD is water-surplus annually);")
print("  the driest (Rajshahi) should be the LOWEST, the wettest (Sylhet) the HIGHEST.")


# =====================================================================
#  9. SAVE
# =====================================================================
out.to_csv(OUT_CSV, index=False)
elev_df.to_csv(OUT_ELEV, index=False)

hr("SAVED")
print(f"  PET + water balance : {OUT_CSV}")
print(f"  Elevation summary   : {OUT_ELEV}")
print(f"  Rows                : {len(out)}")
print("=" * 68)
print("  STAGE 2 COMPLETE.  Column 'D_mm_month' is the input for stage 3 (SPEI).")
print("=" * 68)
