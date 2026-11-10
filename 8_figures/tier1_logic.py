"""
tier1_logic.py  --  the ERA5-vs-BMD (Tier 1) validation calculations, copied from the thesis script  Validation/Validation 1/Tire_1/tier1_validation_figures.py
(station-division map, unit harmonisation, Thornthwaite PET, log-logistic SPEI,
metrics, month-shift search, run_analysis). Changes: the file lookup (find_source) uses
paths.py, and the station zero-value rule (1_data_and_index/station_zero_rule.py, Text S2)
is applied when the BMD file is loaded. The plotting part of the original is not used.
Used by figS01_validation.py (Figs S1 and S2).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import gamma as GAMMA

import p1_fig_config as C
import paths as P
import sys as _sys
_sys.path.insert(0, str(P.CODE_ROOT / "1_data_and_index"))
import station_zero_rule as ZR  # noqa: E402

ZERO_RULE = True   # False reproduces the thesis (pre-correction) numbers

ERA5_CSV = "divisional_averaged_1980_2024.csv"
BMD_CSV  = "Final-Rainfall_Report_BMD_MENDELAY.csv"

YEAR_START, YEAR_END = 1980, 2022     # common overlap period
SPEI_SCALES          = [1, 3, 6, 12]
DROUGHT_THRESHOLD    = -1.0           # SPEI < -1.0 = moderate drought or worse

# Stations shown in the single-station panel (Fig 1d). Chosen to span the full
# range of phase errors found in the BMD file.
SINGLE_STATIONS = ["Dhaka (11111)", "Rangpur (10208)",
                   "Barisal (11704)", "Cox's Bazar (11927)"]

# SPEI time-series panels (Fig 3a) and the window they display
TS_DIVISIONS  = ["Dhaka", "Barisal"]
TS_SCALE      = 3
TS_WINDOW     = (2005, 2023)


def find_source(fname):
    """ERA5 divisional means from P1_Data/inputs; the BMD compilation from
    P1_Data/external (not distributed, see the P1_Data README)."""
    p = {ERA5_CSV: C.VARS_CSV, BMD_CSV: P.BMD_CSV}[fname]
    if p.exists():
        return p
    raise FileNotFoundError(f"{fname} not found: {p} (see paths.py)")


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
    if ZERO_RULE:
        b, _ = ZR.apply(b)
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
