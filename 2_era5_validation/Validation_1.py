"""
TIER 1 -- ERA5 vs BMD VALIDATION FIGURE SUITE
=============================================
Journal-ready (Q1) figures supporting the "Validation of ERA5 against BMD
observations" section of the thesis.

Self-contained: recomputes the full validation (unit harmonisation, Thornthwaite
PET, log-logistic SPEI, agreement metrics, event-detection skill) directly from
the two source CSVs, then writes every figure as a SEPARATE PNG.

No titles are drawn inside the images -- the title is the FILE NAME, so figures
can be captioned in Word/LaTeX without a duplicated heading.

Reads   : divisional_averaged_1980_2024.csv   (ERA5 divisional means)
          Final-Rainfall_Report.csv           (BMD, 35 stations, Mendeley)
Writes  : PNG figures @ 400 dpi + the supporting metric CSVs into OUT_DIR

Author  : Tier-1 validation suite
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import norm
from scipy.special import gamma as GAMMA

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG
# ==============================================================================
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402  (one relative-path config for the whole package)
_sys.path.insert(0, str(P.CODE_ROOT / "1_data_and_index"))
import station_zero_rule as ZR  # noqa: E402
OUT_DIR  = P.WORK / "2_era5_validation"

ERA5_CSV = "divisional_averaged_1980_2024.csv"
BMD_CSV  = "Final-Rainfall_Report_BMD_MENDELAY.csv"

YEAR_START, YEAR_END = 1980, 2022     # common overlap period
SPEI_SCALES          = [1, 3, 6, 12]
DROUGHT_THRESHOLD    = -1.0           # SPEI < -1.0 = moderate drought or worse
DPI                  = 400
ZERO_RULE            = True     # station zero-value rule (Text S2); False reproduces the thesis numbers

# Stations shown in the single-station panel (Fig 1d). Chosen to span the full
# range of phase errors found in the BMD file.
SINGLE_STATIONS = ["Dhaka (11111)", "Rangpur (10208)",
                   "Barisal (11704)", "Cox's Bazar (11927)"]

# SPEI time-series panels (Fig 3a) and the window they display
TS_DIVISIONS  = ["Dhaka", "Barisal"]
TS_SCALE      = 3
TS_WINDOW     = (2005, 2023)



def find_source(fname):
    """Locate a source CSV (paths.py: ERA5 divisional means in P1_Data/inputs,
    the BMD compilation in P1_Data/external, which is not distributed)."""
    cands = [{ERA5_CSV: P.VARS_CSV, BMD_CSV: P.BMD_CSV}.get(fname, P.EXTERNAL / fname)]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find '{fname}'. Looked in:\n  " +
        "\n  ".join(str(c) for c in cands) +
        "\nSee paths.py and the README of P1_Data.")


# ==============================================================================
# STYLE  (Q1 journal defaults -- matched to the model-comparison figure suite)
# ==============================================================================
def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.grid": False,
        "grid.color": "0.75",
        "grid.alpha": 0.30,
        "grid.linewidth": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.8",
        "legend.fancybox": False,
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


# Okabe-Ito colourblind-safe palette (yellow #F0E442 excluded: unreadable on white)
C_BMD    = "#0072B2"   # blue        -- BMD station observations
C_ERA5   = "#D55E00"   # vermillion  -- ERA5 reanalysis
C_POS    = "#D55E00"   # positive bias
C_NEG    = "#0072B2"   # negative bias
C_ACCENT = "#009E73"   # bluish green
C_GREY   = "#4D4D4D"   # charcoal
C_BAND   = "#009E73"   # acceptable-bias band

SCALE_COLORS = {1: "#E69F00", 3: "#009E73", 6: "#0072B2", 12: "#D55E00"}


def light_grid(ax, axis="both"):
    ax.grid(True, axis=axis, color="0.75", alpha=0.30, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  saved  {p.name}")


# ==============================================================================
# STATION -> DIVISION MAPPING  (35 BMD stations, 8 BBS divisions)
# ==============================================================================
ST2DIV = {
    'Barisal (11704)': 'Barisal', 'Bhola (11706)': 'Barisal',
    'Khepupara (12110)': 'Barisal', 'Patuakhali (12103)': 'Barisal',
    'Ambagan (41977)': 'Chittagong', 'Chandpur (11316)': 'Chittagong',
    'Chittagong(Air-port) (11921)': 'Chittagong', 'Comilla (11313)': 'Chittagong',
    "Cox's Bazar (11927)": 'Chittagong', 'Feni (11805)': 'Chittagong',
    'Hatiya (11814)': 'Chittagong', 'Kutubdia (11925)': 'Chittagong',
    'M. Court (11809)': 'Chittagong', 'Rangamati (12007)': 'Chittagong',
    'Sandwip (11916)': 'Chittagong', 'Sitakunda (11912)': 'Chittagong',
    'Teknaf (11929)': 'Chittagong',
    'Dhaka (11111)': 'Dhaka', 'Faridpur (11505)': 'Dhaka',
    'Madaripur (11513)': 'Dhaka', 'Tangail (41909)': 'Dhaka',
    'Chuadanga (41926)': 'Khulna', 'Jessore (11407)': 'Khulna',
    'Khulna (11604)': 'Khulna', 'Mongla (41958)': 'Khulna',
    'Satkhira (11610)': 'Khulna',
    'Mymensingh (10609)': 'Mymensingh',
    'Bogra (10408)': 'Rajshahi', 'Ishurdi (10910)': 'Rajshahi',
    'Rajshahi (10320)': 'Rajshahi',
    'Dinajpur (10120)': 'Rangpur', 'Rangpur (10208)': 'Rangpur',
    'Syedpur (41858)': 'Rangpur',
    'Srimongal (10724)': 'Sylhet', 'Sylhet (10705)': 'Sylhet',
}

# Representative divisional latitude, used for the Thornthwaite daylight term
DIVLAT = {'Barisal': 22.70, 'Chittagong': 22.60, 'Dhaka': 23.85, 'Khulna': 22.85,
          'Mymensingh': 24.75, 'Rajshahi': 24.40, 'Rangpur': 25.70, 'Sylhet': 24.85}
DIVS = list(DIVLAT)

MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# ==============================================================================
# DATA LOADING + UNIT HARMONISATION
# ==============================================================================
def load_bmd():
    b = pd.read_csv(find_source(BMD_CSV), low_memory=False).dropna(subset=['Station'])
    b['Station'] = b.Station.astype(str)
    b = b[(b.Year >= YEAR_START) & (b.Year <= YEAR_END)].copy()
    b['division'] = b.Station.map(ST2DIV)
    ren = {'Total_Rainfall_(mm)': 'P', 'Cloud Coverage (Octs)': 'CLD',
           'Humidity': 'RH', 'Maximum_Temperature': 'TMAX',
           'Minimum_Temperature': 'TMIN', 'Sunshine': 'SUN', 'Windspeed': 'WS'}
    for k, v in ren.items():
        b[v] = pd.to_numeric(b[k], errors='coerce')
    b['year'] = b.Year.astype(int)
    b['month'] = b.Month.astype(int)
    b = b[['Station', 'division', 'year', 'month',
           'P', 'CLD', 'RH', 'TMAX', 'TMIN', 'SUN', 'WS']]
    if ZERO_RULE:                       # missing data stored as 0 mm (Text S2)
        b, rep = ZR.apply(b)
        print(rep)
    return b


def load_era5():
    e = pd.read_csv(find_source(ERA5_CSV))
    e = e[(e.year >= YEAR_START) & (e.year <= YEAR_END)].copy()
    dim = e.apply(lambda r: pd.Period(f'{int(r.year)}-{int(r.month)}', 'M').days_in_month,
                  axis=1)
    out = pd.DataFrame({
        'division': e.division, 'year': e.year.astype(int), 'month': e.month.astype(int),
        'P': e.tp * dim,                 # mm/day -> mm/month total
        'TMAX': e.mx2t, 'TMIN': e.mn2t, 'TMEAN': e.t2m,
        'CLD': e.tcc * 8.0,              # cloud fraction -> octas
        'WS': np.hypot(e.u10, e.v10),    # 10 m scalar wind speed
        'D2M': e.d2m, 'SSRD': e.ssrd,
    })
    es = 6.112 * np.exp(17.67 * out.TMEAN / (out.TMEAN + 243.5))
    ea = 6.112 * np.exp(17.67 * out.D2M / (out.D2M + 243.5))
    out['RH'] = 100 * ea / es
    return out


# ==============================================================================
# THORNTHWAITE PET  +  SPEI (Vicente-Serrano et al. 2010)
# ==============================================================================
def daylight_factor(lat_deg, months):
    lat = np.radians(lat_deg)
    mid = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    dec = 0.409 * np.sin(2 * np.pi * mid / 365 - 1.39)
    x = np.clip(-np.tan(lat) * np.tan(dec), -1, 1)
    N = 24 / np.pi * np.arccos(x)
    dim = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    return ((N / 12) * (dim / 30))[np.asarray(months) - 1]


def thornthwaite(tmean, month, lat):
    t = np.maximum(np.asarray(tmean, float), 0.0)
    df = pd.DataFrame({'t': t, 'm': np.asarray(month)})
    hm = df.groupby('m').t.mean()
    I = float(((hm / 5) ** 1.514).sum())
    a = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239
    return 16 * (10 * t / I) ** a * daylight_factor(lat, df.m.values)


def _fit_loglogistic(x):
    """3-parameter log-logistic via probability weighted moments."""
    x = np.sort(np.asarray(x, float)); n = len(x)
    F = (np.arange(1, n + 1) - 0.35) / n
    w0 = x.mean(); w1 = ((1 - F) * x).sum() / n; w2 = (((1 - F) ** 2) * x).sum() / n
    den = 6 * w1 - w0 - 6 * w2
    if abs(den) < 1e-12:
        return None
    beta = (2 * w1 - w0) / den
    if not (0.05 < beta < 50):
        return None
    gg = GAMMA(1 + 1 / beta) * GAMMA(1 - 1 / beta)
    if not np.isfinite(gg) or gg <= 0:
        return None
    alpha = (w0 - 2 * w1) * beta / gg
    if alpha <= 0:
        return None
    return alpha, beta, w0 - alpha * gg


def spei(D, month, scale):
    """Standardised index of the accumulated series D, fitted per calendar month."""
    A = pd.Series(np.asarray(D, float)).rolling(scale).sum().values
    m = np.asarray(month)
    out = np.full(len(A), np.nan)
    for k in range(1, 13):
        idx = np.where((m == k) & np.isfinite(A))[0]
        if len(idx) < 20:
            continue
        x = A[idx]
        fit = _fit_loglogistic(x)
        if fit is None:                                   # empirical fallback
            p = (pd.Series(x).rank().values - 0.44) / (len(x) + 0.12)
        else:
            a, b, g = fit
            p = np.clip(1.0 / (1.0 + (a / np.maximum(x - g, 1e-9)) ** b), 1e-6, 1 - 1e-6)
        out[idx] = norm.ppf(p)
    return np.clip(out, -3.5, 3.5)


# ==============================================================================
# METRICS
# ==============================================================================
def metrics(o, p):
    """o = reference (BMD station), p = test (ERA5)."""
    o = np.asarray(o, float); p = np.asarray(p, float)
    k = np.isfinite(o) & np.isfinite(p); o, p = o[k], p[k]
    if len(o) < 10:
        return dict(n=len(o))
    r = np.corrcoef(o, p)[0, 1]
    return dict(
        n=len(o), r=r,
        bias=(p - o).mean(),
        pbias=100 * (p - o).sum() / o.sum() if o.sum() != 0 else np.nan,
        MAE=np.abs(p - o).mean(),
        RMSE=np.sqrt(((p - o) ** 2).mean()),
        NSE=1 - ((p - o) ** 2).sum() / ((o - o.mean()) ** 2).sum(),
        KGE=1 - np.sqrt((r - 1) ** 2 + (p.std() / o.std() - 1) ** 2
                        + (p.mean() / o.mean() - 1) ** 2),
    )


def best_month_shift(bmd_clim, era_clim):
    """Circular month-shift k that best aligns a BMD climatology with ERA5."""
    cc = [(np.corrcoef(np.roll(bmd_clim, k), era_clim)[0, 1], k) for k in range(12)]
    return max(cc)[1], max(cc)[0]


# ==============================================================================
# ANALYSIS PIPELINE
# ==============================================================================
def run_analysis():
    print("Loading and harmonising source data ...")
    bmd, era = load_bmd(), load_era5()

    bd = (bmd.groupby(['division', 'year', 'month'])[['P', 'TMAX', 'TMIN', 'CLD', 'RH', 'WS']]
             .mean().reset_index())
    bd['TMEAN'] = (bd.TMAX + bd.TMIN) / 2
    mg = (bd.merge(era, on=['division', 'year', 'month'], suffixes=('_b', '_e'))
            .sort_values(['division', 'year', 'month']).reset_index(drop=True))
    print(f"  matched {len(mg)} division-months ({YEAR_START}-{YEAR_END})")

    # ---- national climatologies (Fig 1a-c) ----------------------------------
    clim = pd.DataFrame({
        'month': np.arange(1, 13),
        'P_bmd': bd.groupby('month').P.mean().values,
        'P_era': era.groupby('month').P.mean().values,
        'TMAX_bmd': bd.groupby('month').TMAX.mean().values,
        'TMAX_era': era.groupby('month').TMAX.mean().values,
        'TMIN_bmd': bd.groupby('month').TMIN.mean().values,
        'TMIN_era': era.groupby('month').TMIN.mean().values,
    })

    # ---- single-station climatologies (Fig 1d) -----------------------------
    era_clim = era.groupby(['division', 'month'])[['TMAX', 'TMIN', 'P']].mean()
    stat = {}
    for st in SINGLE_STATIONS:
        dv = ST2DIV[st]
        s = bmd[bmd.Station == st]
        stat[st] = dict(
            division=dv,
            tmax_b=s.groupby('month').TMAX.mean().values,
            tmax_e=era_clim.loc[dv, 'TMAX'].values,
            p_b=s.groupby('month').P.mean().values,
            p_e=era_clim.loc[dv, 'P'].values,
        )
        k, r = best_month_shift(stat[st]['tmax_b'], stat[st]['tmax_e'])
        stat[st].update(shift=k, shift_r=r)

    # ---- full per-station shift table (evidence for the report) ------------
    rows = []
    for st, dv in ST2DIV.items():
        s = bmd[bmd.Station == st]
        rec = {'Station': st, 'Division': dv}
        for v in ['TMAX', 'TMIN', 'P']:
            k, r = best_month_shift(s.groupby('month')[v].mean().values,
                                    era_clim.loc[dv, v].values)
            rec[f'{v}_shift'] = k
            rec[f'{v}_r'] = round(r, 3)
        rows.append(rec)
    shift_tab = pd.DataFrame(rows).sort_values(['Division', 'Station'])

    # ---- precipitation validation (Fig 2) ----------------------------------
    prows = []
    for d in DIVS + ['ALL']:
        s = mg if d == 'ALL' else mg[mg.division == d]
        m = metrics(s.P_b, s.P_e)
        m['r_anom'] = metrics(s.groupby('month').P_b.transform(lambda x: x - x.mean()),
                              s.groupby('month').P_e.transform(lambda x: x - x.mean()))['r']
        m['r_annual'] = np.corrcoef(s.groupby('year').P_b.sum(),
                                    s.groupby('year').P_e.sum())[0, 1]
        m['division'] = d
        prows.append(m)
    precip = pd.DataFrame(prows)[['division', 'n', 'r', 'r_anom', 'r_annual',
                                  'bias', 'pbias', 'MAE', 'RMSE', 'NSE', 'KGE']]

    # ---- PET sensitivity + SPEI agreement (Fig 3) --------------------------
    pet_rows, series = [], []
    for d in DIVS:
        s = mg[mg.division == d]
        mo, lat = s.month.values, DIVLAT[d]
        pet = thornthwaite(s.TMEAN_e, mo, lat)            # ERA5 PET for BOTH sources
        petc = pd.Series(pet).groupby(mo).transform('mean').values
        rec = {'division': d}
        r = pd.DataFrame({'division': d, 'year': s.year.values, 'month': mo})
        for sc in SPEI_SCALES:
            a = spei(s.P_e.values - pet, mo, sc)
            b = spei(s.P_e.values - petc, mo, sc)
            k = np.isfinite(a) & np.isfinite(b)
            rec[f'SPEI{sc}'] = np.corrcoef(a[k], b[k])[0, 1]
            r[f'SPEI{sc}_b'] = spei(s.P_b.values - pet, mo, sc)
            r[f'SPEI{sc}_e'] = spei(s.P_e.values - pet, mo, sc)
        pet_rows.append(rec)
        series.append(r)
    pet_sens = pd.DataFrame(pet_rows)
    series = pd.concat(series, ignore_index=True)

    irows, erows = [], []
    for sc in SPEI_SCALES:
        m = metrics(series[f'SPEI{sc}_b'], series[f'SPEI{sc}_e'])
        m['scale'] = sc
        irows.append(m)

        o = series[f'SPEI{sc}_b'].values; p = series[f'SPEI{sc}_e'].values
        k = np.isfinite(o) & np.isfinite(p); o, p = o[k], p[k]
        O, P = o < DROUGHT_THRESHOLD, p < DROUGHT_THRESHOLD
        H, M = int((O & P).sum()), int((O & ~P).sum())
        F, C = int((~O & P).sum()), int((~O & ~P).sum())
        n = len(o); acc = (H + C) / n
        pe = ((H + M) * (H + F) + (M + C) * (F + C)) / n ** 2
        erows.append(dict(scale=sc, n=n, events_obs=int(O.sum()),
                          events_era5=int(P.sum()), hits=H,
                          POD=H / (H + M) if H + M else np.nan,
                          FAR=F / (H + F) if H + F else np.nan,
                          CSI=H / (H + M + F) if H + M + F else np.nan,
                          ACC=acc, kappa=(acc - pe) / (1 - pe),
                          HSS=2 * (H * C - M * F) / ((H + M) * (M + C) + (H + F) * (F + C))))
    agree = pd.DataFrame(irows)[['scale', 'n', 'r', 'bias', 'MAE', 'RMSE', 'NSE', 'KGE']]
    events = pd.DataFrame(erows)

    return dict(mg=mg, clim=clim, stat=stat, shift_tab=shift_tab, precip=precip,
                pet_sens=pet_sens, series=series, agree=agree, events=events)


# ==============================================================================
# FIGURE 1 -- DATA-QUALITY SCREENING
# ==============================================================================
def fig1a_precipitation(clim):
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    light_grid(ax)
    ax.plot(clim.month, clim.P_bmd, 'o-', color=C_BMD, lw=2.0, ms=7,
            label='BMD stations (35-station mean)', zorder=3)
    ax.plot(clim.month, clim.P_era, 's--', color=C_ERA5, lw=2.0, ms=7,
            label='ERA5 reanalysis (8-division mean)', zorder=3)
    ax.set_xlabel('Month')
    ax.set_ylabel('Precipitation (mm month$^{-1}$)')
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTH_ABBR)
    ax.set_ylim(0, max(clim.P_bmd.max(), clim.P_era.max()) * 1.30)
    ax.legend(loc='upper left', borderpad=0.6)
    save(fig, 'fig1_a_precipitation_consistent')


def _fig1_temp(clim, var, label, fname):
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    light_grid(ax)
    b, e = clim[f'{var}_bmd'], clim[f'{var}_era']
    ax.plot(clim.month, b, 'o-', color=C_BMD, lw=2.0, ms=7,
            label=f'BMD stations  (annual range {b.max() - b.min():.2f} $^\\circ$C)', zorder=3)
    ax.plot(clim.month, e, 's--', color=C_ERA5, lw=2.0, ms=7,
            label=f'ERA5 reanalysis  (annual range {e.max() - e.min():.2f} $^\\circ$C)', zorder=3)
    ax.set_xlabel('Month')
    ax.set_ylabel(f'{label} ($^\\circ$C)')
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTH_ABBR)
    lo = min(b.min(), e.min()); hi = max(b.max(), e.max()); rng = hi - lo
    ax.set_ylim(lo - 0.32 * rng, hi + 0.10 * rng)   # headroom below for the legend
    ax.legend(loc='lower center', borderpad=0.6)
    save(fig, fname)


def fig1b_tmax(clim):
    _fig1_temp(clim, 'TMAX', 'Mean daily maximum temperature',
               'fig1_b_Tmax_BMD_file_has_no_seasonal_cycle')


def fig1c_tmin(clim):
    _fig1_temp(clim, 'TMIN', 'Mean daily minimum temperature',
               'fig1_c_Tmin_BMD_file_has_no_seasonal_cycle')


def fig1d_single_stations(stat):
    """Combined panel: the defect exists BEFORE any averaging."""
    n = len(SINGLE_STATIONS)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8), constrained_layout=True)
    for ax, st in zip(axes.ravel(), SINGLE_STATIONS):
        d = stat[st]
        light_grid(ax)
        ax.plot(range(1, 13), d['tmax_b'], 'o-', color=C_BMD, lw=1.9, ms=6, zorder=3)
        ax.plot(range(1, 13), d['tmax_e'], 's--', color=C_ERA5, lw=1.9, ms=6, zorder=3)
        pk_b = MONTH_ABBR[int(np.argmax(d['tmax_b']))]
        pk_e = MONTH_ABBR[int(np.argmax(d['tmax_e']))]
        amp_b = d['tmax_b'].max() - d['tmax_b'].min()
        amp_e = d['tmax_e'].max() - d['tmax_e'].min()
        ax.set_title(f"{st.split(' (')[0]}  ({d['division']} Division)",
                     fontsize=12.5, pad=8)
        ax.text(0.028, 0.965,
                f"BMD peak: {pk_b}   range {amp_b:.1f} $^\\circ$C\n"
                f"ERA5 peak: {pk_e}   range {amp_e:.1f} $^\\circ$C\n"
                f"required month-shift: k = {d['shift']}",
                transform=ax.transAxes, va='top', ha='left', fontsize=10.2,
                bbox=dict(boxstyle='round,pad=0.42', fc='white', ec='0.8', alpha=0.92))
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_ABBR, fontsize=9.5)
        lo = min(d['tmax_b'].min(), d['tmax_e'].min())
        hi = max(d['tmax_b'].max(), d['tmax_e'].max())
        ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.62 * (hi - lo))
        ax.set_xlabel('Month')
        ax.set_ylabel('Mean daily maximum temperature ($^\\circ$C)')
    fig.legend(handles=[
        Line2D([], [], marker='o', ls='-', color=C_BMD, lw=1.9, ms=6,
               label='BMD station (single station, no averaging)'),
        Line2D([], [], marker='s', ls='--', color=C_ERA5, lw=1.9, ms=6,
               label='ERA5 reanalysis (containing division)')],
        loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.045), borderpad=0.6)
    save(fig, 'fig1_d_single_station_temperature_cycles_before_averaging')


# ==============================================================================
# FIGURE 2 -- PRECIPITATION VALIDATION
# ==============================================================================
def fig2a_scatter(mg):
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    light_grid(ax)
    r = np.corrcoef(mg.P_b, mg.P_e)[0, 1]
    n = len(mg)
    hi = max(mg.P_b.max(), mg.P_e.max()) * 1.05
    ax.scatter(mg.P_b, mg.P_e, s=17, alpha=0.32, color=C_BMD, lw=0, zorder=3)
    ax.plot([0, hi], [0, hi], '--', color='0.25', lw=1.4, zorder=4)
    sl, ic = np.polyfit(mg.P_b, mg.P_e, 1)
    ax.plot([0, hi], [ic, ic + sl * hi], '-', color=C_ERA5, lw=1.9, zorder=5)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('BMD Station Monthly Precipitation (mm)')
    ax.set_ylabel('ERA5 Monthly Precipitation (mm)')
    ax.legend(handles=[
        Line2D([], [], marker='o', ls='', color=C_BMD, ms=7, alpha=0.6,
               label=f'Division-month  ($r$ = {r:.3f},  $n$ = {n:,})'),
        Line2D([], [], ls='--', color='0.25', lw=1.4, label='1:1 line'),
        Line2D([], [], ls='-', color=C_ERA5, lw=1.9, label='Least-squares fit')],
        loc='upper left', borderpad=0.6)
    save(fig, 'fig2_a_ERA5_vs_BMD_monthly_precipitation')


def fig2b_skill(precip):
    p = precip[precip.division != 'ALL'].reset_index(drop=True)
    x = np.arange(len(p)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    light_grid(ax, axis='y')
    ax.bar(x - w / 2, p.r, w, color=C_BMD, edgecolor='white',
           label='Raw monthly series', zorder=3)
    ax.bar(x + w / 2, p.r_anom, w, color=C_ERA5, edgecolor='white',
           label='Anomaly series (seasonal cycle removed)', zorder=3)
    for xi, (a, b) in enumerate(zip(p.r, p.r_anom)):
        ax.text(xi - w / 2, a + 0.014, f'{a:.2f}', ha='center', fontsize=9.6)
        ax.text(xi + w / 2, b + 0.014, f'{b:.2f}', ha='center', fontsize=9.6)
    ax.set_xticks(x); ax.set_xticklabels(p.division, rotation=30, ha='right')
    ax.set_ylim(0, 1.20)                       # headroom keeps bars clear of legend
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel('Division')
    ax.set_ylabel('Pearson correlation coefficient, $r$')
    ax.legend(loc='upper center', ncol=2, borderpad=0.6, bbox_to_anchor=(0.5, 1.01))
    save(fig, 'fig2_b_precipitation_skill_by_division')


def fig2c_bias(precip):
    p = precip[precip.division != 'ALL'].reset_index(drop=True)
    x = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    light_grid(ax, axis='y')
    ax.axhspan(-15, 15, color=C_BAND, alpha=0.13, zorder=1)
    ax.bar(x, p.pbias, 0.6, zorder=3, edgecolor='white',
           color=[C_POS if v > 0 else C_NEG for v in p.pbias])
    for xi, v in zip(x, p.pbias):
        ax.text(xi, v + (1.6 if v > 0 else -3.4), f'{v:+.1f}', ha='center', fontsize=9.8)
    ax.axhline(0, color='black', lw=1.0, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(p.division, rotation=30, ha='right')
    ax.set_ylim(min(p.pbias.min() - 12, -25), p.pbias.max() + 20)
    ax.set_xlabel('Division')
    ax.set_ylabel('Percent bias, PBIAS (%)')
    ax.legend(handles=[
        Patch(fc=C_POS, ec='white', label='ERA5 over-estimates rainfall'),
        Patch(fc=C_NEG, ec='white', label='ERA5 under-estimates rainfall'),
        Patch(fc=C_BAND, alpha=0.30, ec='none',
              label='$\\pm$15% : "very good" (Moriasi et al., 2007)')],
        loc='upper left', borderpad=0.6)
    save(fig, 'fig2_c_precipitation_percent_bias_by_division')


# ==============================================================================
# FIGURE 3 -- DROUGHT-INDEX LEVEL VALIDATION
# ==============================================================================
def fig3a_spei_series(series):
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.4), constrained_layout=True)
    y0, y1 = TS_WINDOW
    for ax, dv in zip(axes, TS_DIVISIONS):
        full = series[series.division == dv]
        r = full[[f'SPEI{TS_SCALE}_b', f'SPEI{TS_SCALE}_e']].corr().iloc[0, 1]
        s = full[(full.year >= y0) & (full.year < y1)].sort_values(['year', 'month'])
        t = s.year + (s.month - 1) / 12
        light_grid(ax)
        ax.axhline(0, color='0.55', lw=0.8, zorder=2)
        ax.axhline(DROUGHT_THRESHOLD, ls=':', color=C_GREY, lw=1.2, zorder=2)
        ax.fill_between(t, s[f'SPEI{TS_SCALE}_b'], 0,
                        where=s[f'SPEI{TS_SCALE}_b'] < 0,
                        color=C_ERA5, alpha=0.16, lw=0, zorder=2)
        ax.plot(t, s[f'SPEI{TS_SCALE}_b'], '-', color=C_BMD, lw=1.5, zorder=4)
        ax.plot(t, s[f'SPEI{TS_SCALE}_e'], '--', color=C_ERA5, lw=1.5, zorder=4)
        ax.set_xlim(y0, y1)
        ax.set_ylim(-3.9, 3.15)
        ax.set_ylabel(f'SPEI-{TS_SCALE}')
        ax.set_xlabel('Year')
        ax.text(0.011, 0.955, f'{dv} Division    full-period $r$ = {r:.3f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=12.5,
                bbox=dict(boxstyle='round,pad=0.42', fc='white', ec='0.8', alpha=0.92))
    fig.legend(handles=[
        Line2D([], [], ls='-', color=C_BMD, lw=1.6, label='SPEI from BMD station precipitation'),
        Line2D([], [], ls='--', color=C_ERA5, lw=1.6, label='SPEI from ERA5 precipitation'),
        Line2D([], [], ls=':', color=C_GREY, lw=1.2,
               label=f'Drought threshold (SPEI = {DROUGHT_THRESHOLD:.1f})')],
        loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.085), borderpad=0.6)
    save(fig, 'fig3_a_SPEI3_time_series_Dhaka_and_Barisal')


def fig3b_agreement(agree, events):
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    light_grid(ax)
    x = np.arange(len(SPEI_SCALES))
    ax.plot(x, agree.r, 'o-', color=C_BMD, lw=2.0, ms=8,
            label='Pearson correlation, $r$', zorder=4)
    ax.plot(x, events.POD, '^-.', color=C_ACCENT, lw=2.0, ms=8,
            label=f'Probability of detection (SPEI < {DROUGHT_THRESHOLD:.1f})', zorder=4)
    ax.plot(x, events.kappa, 's--', color=C_ERA5, lw=2.0, ms=8,
            label="Cohen's $\\kappa$", zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([f'SPEI-{s}' for s in SPEI_SCALES])
    ax.set_ylim(0, 1.14); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel('SPEI accumulation timescale')
    ax.set_ylabel('Agreement statistic')
    ax.legend(loc='upper center', ncol=1, borderpad=0.6, bbox_to_anchor=(0.5, 1.015))
    save(fig, 'fig3_b_agreement_versus_SPEI_timescale')


def fig3c_pet(pet_sens):
    x = np.arange(len(pet_sens)); w = 0.20
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    light_grid(ax, axis='y')
    for i, sc in enumerate(SPEI_SCALES):
        ax.bar(x + (i - 1.5) * w, pet_sens[f'SPEI{sc}'], w, zorder=3,
               color=SCALE_COLORS[sc], edgecolor='white', label=f'SPEI-{sc}')
    ax.set_xticks(x); ax.set_xticklabels(pet_sens.division, rotation=30, ha='right')
    ax.set_ylim(0.80, 1.038); ax.set_yticks(np.arange(0.80, 1.001, 0.05))
    ax.set_xlabel('Division')
    ax.set_ylabel('Correlation: full PET vs climatological PET')
    ax.legend(loc='upper center', ncol=4, borderpad=0.6, bbox_to_anchor=(0.5, 1.015))
    save(fig, 'fig3_c_PET_sensitivity_of_SPEI_by_division')


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("TIER 1 -- ERA5 vs BMD VALIDATION FIGURE SUITE")
    print("=" * 74)

    R = run_analysis()

    print("\nWriting supporting tables ...")
    R['precip'].to_csv(OUT_DIR / 'table1_precipitation_validation.csv', index=False)
    R['pet_sens'].to_csv(OUT_DIR / 'table2_pet_sensitivity.csv', index=False)
    R['agree'].to_csv(OUT_DIR / 'table3_spei_agreement.csv', index=False)
    R['events'].to_csv(OUT_DIR / 'table4_drought_event_detection.csv', index=False)
    R['shift_tab'].to_csv(OUT_DIR / 'table5_per_station_month_shifts.csv', index=False)
    R['series'].to_csv(OUT_DIR / 'spei_series_bmd_vs_era5.csv', index=False)
    for f in ['table1_precipitation_validation', 'table2_pet_sensitivity',
              'table3_spei_agreement', 'table4_drought_event_detection',
              'table5_per_station_month_shifts']:
        print(f"  saved  {f}.csv")

    print("\nFigure 1 -- data-quality screening")
    fig1a_precipitation(R['clim'])
    fig1b_tmax(R['clim'])
    fig1c_tmin(R['clim'])
    fig1d_single_stations(R['stat'])

    print("\nFigure 2 -- precipitation validation")
    fig2a_scatter(R['mg'])
    fig2b_skill(R['precip'])
    fig2c_bias(R['precip'])

    print("\nFigure 3 -- drought-index level validation")
    fig3a_spei_series(R['series'])
    fig3b_agreement(R['agree'], R['events'])
    fig3c_pet(R['pet_sens'])

    # ---- console summary ---------------------------------------------------
    a = R['precip'][R['precip'].division == 'ALL'].iloc[0]
    print("\n" + "=" * 74)
    print("KEY RESULTS")
    print("=" * 74)
    print(f"  Precipitation (pooled) : r = {a.r:.3f}   NSE = {a.NSE:.3f}   "
          f"KGE = {a.KGE:.3f}   PBIAS = {a.pbias:+.1f}%")
    ps = R['pet_sens']
    print("  PET sensitivity        : " + "   ".join(
        f"SPEI-{sc} r = {ps[f'SPEI{sc}'].mean():.3f}" for sc in SPEI_SCALES))
    print("  PET share of variance  : " + "   ".join(
        f"SPEI-{sc} {100 * (1 - ps[f'SPEI{sc}'].mean() ** 2):.1f}%" for sc in SPEI_SCALES))
    e1 = R['events'][R['events'].scale == 1].iloc[0]
    g1 = R['agree'][R['agree'].scale == 1].iloc[0]
    print(f"  SPEI-1 agreement       : r = {g1.r:.3f}   POD = {e1.POD:.3f}   "
          f"FAR = {e1.FAR:.3f}   kappa = {e1.kappa:.3f}")
    print(f"\nAll output written to: {OUT_DIR}")
    print("=" * 74)


if __name__ == "__main__":
    main()
