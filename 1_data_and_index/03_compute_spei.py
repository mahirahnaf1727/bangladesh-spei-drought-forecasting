# =====================================================================
#  03_compute_spei.py
#  SPEI PIPELINE - STAGE 3 of 3
# ---------------------------------------------------------------------
#  Purpose : Convert the divisional water-balance series D into SPEI at
#            four timescales (k = 1, 3, 6, 12 months), per division.
#
#  Input   : divisional_pet_waterbalance_1980_2024.csv   (stage 2)
#            -> uses the column 'D_mm_month'
#
#  Output  : divisional_spei_1980_2024.csv        (SPEI_1/3/6/12 + classes)
#            spei_fitted_parameters.csv           (parameter audit table)
#            spei_summary_statistics.csv          (mean/sd/min/max per series)
#
#  METHOD DECISIONS:
#    * Aggregation   : D_k(n) = sum_{i=0}^{k-1} D(n-i),  n >= k
#    * Fitting basis : PER CALENDAR MONTH (each January fitted against the
#                      other Januaries). ESSENTIAL for Bangladesh: pooling all
#                      months would make SPEI track the monsoon season rather
#                      than the anomaly.
#    * Reference     : FULL RECORD 1980-2024 (standard climatological practice;
#                      note as a limitation for the pseudo-future test).
#    * Standardize   : Abramowitz & Stegun 26.2.23 rational approximation,
#                      d3 = 0.001308.
#
# ---------------------------------------------------------------------
#  IMPORTANT NOTE ON THE DISTRIBUTION  (read before writing the methods)
# ---------------------------------------------------------------------
#  The closed form printed in the source papers,
#        F(x) = [1 + (alpha/(x - gamma))^beta]^-1 ,
#  is the 3-parameter LOG-LOGISTIC. Its L-skewness is tau3 = 1/beta, which is
#  ALWAYS POSITIVE - so that form can only represent RIGHT-SKEWED samples.
#  Left-skewed calendar-month samples (common in the dry season, where
#  D ~ -PET) yield beta < 0 and cannot be fitted by it at all.
#
#  The reference R "SPEI" package (Vicente-Serrano et al. 2010) therefore fits
#  the GENERALIZED LOGISTIC (GLO) via L-moments. GLO covers both skew
#  directions, and for k < 0 it is MATHEMATICALLY IDENTICAL to the log-logistic
#  above, with the exact correspondence
#        beta = -1/k ,   alpha_LL = beta * a ,   gamma_LL = xi - beta * a
#  (verified numerically to ~1e-21). This script therefore fits GLO, and where
#  k < 0 it ALSO reports the equivalent log-logistic (alpha, beta, gamma) so the
#  parameters can be quoted directly against the papers' equation.
#
#  VALIDATION: the fitting and standardization routines were verified against
#  (a) synthetic parameter recovery and (b) scipy's exact inverse normal CDF
#  (thesis-stage check; the check script is not included).
# =====================================================================

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
#  CONFIG - edit paths here only
# ---------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402  (one relative-path config for the whole package)
# reads the PET file rebuilt by 02_compute_pet.py if present, otherwise the archived one
IN_DIR   = str(P.pick(P.WORK / "1_data_and_index" / "divisional_pet_waterbalance_1980_2024.csv",
                      P.PETWB_CSV).parent)
IN_CSV   = os.path.join(IN_DIR, "divisional_pet_waterbalance_1980_2024.csv")

OUT_DIR  = str(P.WORK / "1_data_and_index")
OUT_CSV  = os.path.join(OUT_DIR, "divisional_spei_1980_2024.csv")
OUT_PARS = os.path.join(OUT_DIR, "spei_fitted_parameters.csv")
OUT_STAT = os.path.join(OUT_DIR, "spei_summary_statistics.csv")

TIMESCALES = [1, 3, 6, 12]

# Abramowitz & Stegun 26.2.23 constants (d3 LOCKED at 0.001308)
C0, C1, C2 = 2.515517, 0.802853, 0.010328
D1, D2, D3 = 1.432788, 0.189269, 0.001308

os.makedirs(OUT_DIR, exist_ok=True)


def hr(t=""):
    print("=" * 72)
    if t:
        print("  " + t)
        print("=" * 72)


# =====================================================================
#  CORE FUNCTIONS  (validated - see validate_spei.py)
# =====================================================================
def lmoments_ub(x):
    """
    First three L-moments from Hosking's UNBIASED probability-weighted moments.
        b_r = (1/n) * sum_j [ (j-1)...(j-r) / ((n-1)...(n-r)) ] * x_(j)
        l1 = b0 ;  l2 = 2*b1 - b0 ;  l3 = 6*b2 - 6*b1 + b0
    """
    x = np.sort(np.asarray(x, dtype=float))          # ascending
    n = len(x)
    j = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((j - 1) / (n - 1) * x) / n
    b2 = np.sum((j - 1) * (j - 2) / ((n - 1) * (n - 2)) * x) / n
    return b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0


def fit_glo(x):
    """
    Generalized logistic via L-moments (Hosking):
        k  = -tau3
        a  = l2 * sin(k*pi) / (k*pi)
        xi = l1 - a * (1/k - pi/sin(k*pi))
    Returns (xi, a, k) or None if the sample cannot be fitted.
    """
    if len(x) < 10:
        return None
    l1, l2, l3 = lmoments_ub(x)
    if not np.isfinite(l2) or l2 <= 0:
        return None
    k = -(l3 / l2)
    if not np.isfinite(k) or abs(k) >= 0.98:      # outside the usable range
        return None
    if abs(k) < 1e-6:                              # logistic limit
        return float(l1), float(l2), 0.0
    kp = k * np.pi
    a = l2 * np.sin(kp) / kp
    if not np.isfinite(a) or a <= 0:
        return None
    xi = l1 - a * (1.0 / k - np.pi / np.sin(kp))
    if not np.isfinite(xi):
        return None
    return float(xi), float(a), float(k)


def glo_cdf(x, xi, a, k):
    """GLO CDF: F(x) = 1 / (1 + exp(-y)),  y = -ln(1 - k(x-xi)/a) / k."""
    x = np.asarray(x, dtype=float)
    if abs(k) < 1e-6:
        y = (x - xi) / a
    else:
        arg = 1.0 - k * (x - xi) / a
        arg = np.maximum(arg, 1e-12)               # guard the support boundary
        y = -np.log(arg) / k
    return 1.0 / (1.0 + np.exp(-y))


def glo_to_loglogistic(xi, a, k):
    """
    Exact reparameterisation to the papers' log-logistic form
    F(x) = [1 + (alpha/(x-gamma))^beta]^-1, valid only when k < 0.
    """
    if k >= 0:
        return np.nan, np.nan, np.nan
    beta = -1.0 / k
    alpha = beta * a
    gamma = xi - beta * a
    return alpha, beta, gamma


def gringorten_cdf(x):
    """Empirical last-resort CDF: F = (rank - 0.44) / (n + 0.12)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    ranks = pd.Series(x).rank(method='average').values
    return (ranks - 0.44) / (n + 0.12)


def standardize(P):
    """
    P = probability of EXCEEDING D. Converts to SPEI via the Abramowitz &
    Stegun rational approximation to the inverse normal CDF.
    Dry (D low) -> F low -> P high -> negative SPEI.
    """
    P = np.asarray(P, dtype=float)
    P = np.clip(P, 1e-8, 1 - 1e-8)
    out = np.empty_like(P)
    low = P <= 0.5
    Pw = np.where(low, P, 1.0 - P)
    W = np.sqrt(-2.0 * np.log(Pw))
    val = W - (C0 + C1 * W + C2 * W ** 2) / (1.0 + D1 * W + D2 * W ** 2 + D3 * W ** 3)
    out[low] = val[low]
    out[~low] = -val[~low]
    return out


def classify(s):
    """Five-class scheme: None (SPEI >= 0), Mild, Moderate, Severe, Extreme."""
    if not np.isfinite(s):
        return np.nan
    if s >= 0:
        return "None"
    if s > -1.0:
        return "Mild"
    if s > -1.5:
        return "Moderate"
    if s > -2.0:
        return "Severe"
    return "Extreme"


# =====================================================================
#  1. LOAD STAGE-2 OUTPUT
# =====================================================================
hr("STAGE 3  |  SPEI COMPUTATION  (k = 1, 3, 6, 12)")

df = pd.read_csv(IN_CSV, parse_dates=['date'])
df = df.sort_values(['division', 'date']).reset_index(drop=True)

if 'D_mm_month' not in df.columns:
    raise KeyError("Stage-2 CSV has no 'D_mm_month' column.")

divisions = sorted(df['division'].unique())
print(f"  Input      : {IN_CSV}")
print(f"  Rows       : {len(df)}")
print(f"  Divisions  : {len(divisions)} -> {divisions}")
print(f"  Date range : {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"  D range    : {df['D_mm_month'].min():.1f} -> {df['D_mm_month'].max():.1f} mm/month")


# =====================================================================
#  2. AGGREGATE D, FIT PER CALENDAR MONTH, STANDARDIZE
# =====================================================================
hr("FITTING  |  generalized logistic, unbiased L-moments, per calendar month")

param_rows = []
n_fallback = 0
n_fits = 0

for k_scale in TIMESCALES:
    dcol = f"D{k_scale}_mm"
    scol = f"SPEI_{k_scale}"
    df[dcol] = np.nan
    df[scol] = np.nan

    for div in divisions:
        mask = (df['division'] == div)
        sub = df.loc[mask].sort_values('date')
        idx = sub.index

        # --- aggregate: rolling k-month sum of D (NaN for the first k-1) ---
        Dk = sub['D_mm_month'].rolling(window=k_scale, min_periods=k_scale).sum().values
        df.loc[idx, dcol] = Dk

        months = sub['month'].values

        # --- fit and standardize SEPARATELY for each calendar month ---
        for mo in range(1, 13):
            sel = (months == mo) & np.isfinite(Dk)
            if sel.sum() < 10:
                continue
            vals = Dk[sel]
            target_idx = idx[sel]
            n_fits += 1

            fit = fit_glo(vals)
            if fit is None:
                F = gringorten_cdf(vals)
                xi = a = kk = np.nan
                al = be = ga = np.nan
                method = "gringorten_fallback"
                n_fallback += 1
            else:
                xi, a, kk = fit
                F = glo_cdf(vals, xi, a, kk)
                al, be, ga = glo_to_loglogistic(xi, a, kk)
                method = "glo_lmoments"

            P = 1.0 - F                                  # probability of exceeding
            df.loc[target_idx, scol] = standardize(P)

            param_rows.append({
                'timescale': k_scale, 'division': div, 'calendar_month': mo,
                'n_years': int(sel.sum()), 'method': method,
                'glo_xi': xi, 'glo_alpha': a, 'glo_k': kk,
                'loglogistic_alpha': al, 'loglogistic_beta': be, 'loglogistic_gamma': ga,
                'D_mean': float(np.mean(vals)), 'D_sd': float(np.std(vals, ddof=1)),
                'D_min': float(np.min(vals)), 'D_max': float(np.max(vals)),
            })

    print(f"  SPEI_{k_scale:<2d} done  ->  {len(divisions)} divisions x 12 calendar months")

pars = pd.DataFrame(param_rows)
print(f"\n  Total distribution fits : {n_fits}   "
      f"(expect {len(TIMESCALES) * len(divisions) * 12} = 4 x 8 x 12)")
if n_fallback == 0:
    print("  [OK] every calendar-month sample was fitted with the GLO; no empirical fallback.")
else:
    print(f"  [WARNING] {n_fallback} sample(s) fell back to the Gringorten empirical CDF.")
    print(f"            Locate them via the 'method' column in {os.path.basename(OUT_PARS)}.")

ok = pars[pars['method'] == 'glo_lmoments']
if len(ok):
    n_neg = int((ok['glo_k'] < 0).sum())
    print(f"\n  Shape parameter k : {n_neg}/{len(ok)} samples have k < 0")
    print( "                      -> those are EXACTLY the classical log-logistic")
    print( "                         (equivalent alpha/beta/gamma reported in the CSV).")
    print(f"                      {len(ok) - n_neg} sample(s) have k >= 0 (left-skewed),")
    print( "                         which the closed log-logistic form cannot represent.")
    print(f"  k range           : {ok['glo_k'].min():+.3f} -> {ok['glo_k'].max():+.3f}")


# =====================================================================
#  3. CLASSIFY
# =====================================================================
for k_scale in TIMESCALES:
    df[f"class_{k_scale}"] = df[f"SPEI_{k_scale}"].apply(classify)


# =====================================================================
#  4. VERIFICATION
# =====================================================================
hr("VERIFY  |  NaN structure  (k-1 leading NaNs per division, by construction)")
print(f"    {'timescale':>10s} {'NaN found':>11s} {'NaN expected':>13s}   status")
for k_scale in TIMESCALES:
    got = int(df[f"SPEI_{k_scale}"].isna().sum())
    exp = (k_scale - 1) * len(divisions)
    print(f"    {('SPEI_' + str(k_scale)):>10s} {got:11d} {exp:13d}   {'[OK]' if got == exp else '[CHECK]'}")

hr("VERIFY  |  mean ~ 0 and sd ~ 1  (built-in proof the fit + standardization worked)")
stat_rows = []
print(f"    {'division':14s} " + "".join([f"{('SPEI_' + str(k)):>18s}" for k in TIMESCALES]))
print(f"    {'':14s} " + "".join([f"{'mean / sd':>18s}" for _ in TIMESCALES]))
for div in divisions:
    line = f"    {div:14s} "
    for k_scale in TIMESCALES:
        s = df.loc[df['division'] == div, f"SPEI_{k_scale}"].dropna()
        line += f"{s.mean():+8.3f} /{s.std(ddof=0):7.3f}"
        stat_rows.append({'division': div, 'timescale': k_scale, 'n': len(s),
                          'mean': s.mean(), 'sd': s.std(ddof=0),
                          'min': s.min(), 'max': s.max()})
    print(line)

allok = all(abs(r['mean']) < 0.05 and abs(r['sd'] - 1.0) < 0.10 for r in stat_rows)
print(f"\n  {'[OK] all series are ~N(0,1).' if allok else '[CHECK] some series deviate from N(0,1) - inspect above.'}")
print("  NOTE: with ~45 years per calendar-month fit, |SPEI| saturates near 2.3;")
print("        that is expected, not a bug (1-in-45 years ~ the 2nd percentile).")

hr("VERIFY  |  drought class frequency  (% of valid months)")
print(f"    {'timescale':>10s} {'None':>9s} {'Mild':>9s} {'Moderate':>10s} {'Severe':>9s} {'Extreme':>9s}")
for k_scale in TIMESCALES:
    vc = df[f"class_{k_scale}"].value_counts(normalize=True) * 100
    print(f"    {('SPEI_' + str(k_scale)):>10s} {vc.get('None', 0):8.2f}% {vc.get('Mild', 0):8.2f}% "
          f"{vc.get('Moderate', 0):9.2f}% {vc.get('Severe', 0):8.2f}% {vc.get('Extreme', 0):8.2f}%")
print("\n  Theoretical N(0,1) expectation: None 50.0 | Mild 34.1 | Moderate 9.2 |")
print("                                  Severe 4.4 | Extreme 2.3")

hr("VERIFY  |  10 driest division-months on SPEI_12")
top = df.nsmallest(10, 'SPEI_12')[['division', 'date', 'SPEI_12', 'class_12']].copy()
top['date'] = top['date'].dt.strftime('%Y-%m')
print(top.to_string(index=False))
print("\n  Bangladesh drought years commonly cited in the literature include 1979,")
print("  1981, 1989, 1992, 1994-95 and 2010. Overlap here is supportive evidence;")
print("  exact agreement is not expected (different index, data source and region).")

hr("VERIFY  |  worst drought month per division on SPEI_12")
for div in divisions:
    s = df[df['division'] == div].dropna(subset=['SPEI_12'])
    r = s.loc[s['SPEI_12'].idxmin()]
    print(f"    {div:14s} : {r['date'].strftime('%Y-%m')}   SPEI_12 = {r['SPEI_12']:+.3f}  ({r['class_12']})")


# =====================================================================
#  5. SAVE
# =====================================================================
keep = (['division', 'date', 'year', 'month', 'P_mm_month', 'PET_mm_month', 'D_mm_month']
        + [f"D{k}_mm" for k in TIMESCALES]
        + [f"SPEI_{k}" for k in TIMESCALES]
        + [f"class_{k}" for k in TIMESCALES])
keep = [c for c in keep if c in df.columns]
df[keep].to_csv(OUT_CSV, index=False)
pars.to_csv(OUT_PARS, index=False)
pd.DataFrame(stat_rows).to_csv(OUT_STAT, index=False)

hr("SAVED")
print(f"  SPEI series       : {OUT_CSV}")
print(f"  Fitted parameters : {OUT_PARS}")
print(f"  Summary stats     : {OUT_STAT}")
print(f"  Rows              : {len(df)}")
print("=" * 72)
print("  STAGE 3 COMPLETE.  SPEI_1 / SPEI_3 / SPEI_6 / SPEI_12 are the")
print("  forecasting targets for the deep-learning stage.")
print("=" * 72)
