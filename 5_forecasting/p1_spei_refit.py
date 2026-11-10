"""
================================================================================
p1_spei_refit.py  --  F1: refit SPEI on 1980-2018 only and compare  (Stage 2)
================================================================================
WHY
  The thesis fits the SPEI distribution on the full 1980-2024 record, which
  includes the 2019-2024 test years (brief, fix F1). This script refits the
  SAME distribution on 1980-2018 only and applies it to every year, then shows
  how much the test-period SPEI moves.

HOW (logic copied unchanged from 03_compute_spei.py)
  rolling k-month sum of D  ->  per calendar month: GLO fit by unbiased
  L-moments  ->  exceedance probability  ->  Abramowitz & Stegun inverse normal.
  The ONLY change: the fit uses years <= REF_END_YEAR; the fitted CDF is then
  applied to all years of that calendar month.

SELF-CHECK
  With REF_END_YEAR = 2024 the script must reproduce the thesis SPEI file
  exactly. It does this first and stops if it does not.

OUTPUT  (inside OUT_ROOT/_spei_refit/)
  divisional_spei_1980_2024_refit1980-2018.csv   same columns as the thesis file
  spei_refit_comparison.csv                       test-period agreement table
================================================================================
"""

import sys

import numpy as np
import pandas as pd

import p1_config as CFG

TIMESCALES   = [1, 3, 6, 12]
REF_END_YEAR = 2018
TEST_START   = 2019
THRESHOLD    = -1.0

C0, C1, C2 = 2.515517, 0.802853, 0.010328
D1, D2, D3 = 1.432788, 0.189269, 0.001308


# ---- core functions: copied unchanged from 03_compute_spei.py -------------------
def lmoments_ub(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    j = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((j - 1) / (n - 1) * x) / n
    b2 = np.sum((j - 1) * (j - 2) / ((n - 1) * (n - 2)) * x) / n
    return b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0


def fit_glo(x):
    if len(x) < 10:
        return None
    l1, l2, l3 = lmoments_ub(x)
    if not np.isfinite(l2) or l2 <= 0:
        return None
    k = -(l3 / l2)
    if not np.isfinite(k) or abs(k) >= 0.98:
        return None
    if abs(k) < 1e-6:
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
    x = np.asarray(x, dtype=float)
    if abs(k) < 1e-6:
        y = (x - xi) / a
    else:
        arg = 1.0 - k * (x - xi) / a
        arg = np.maximum(arg, 1e-12)
        y = -np.log(arg) / k
    return 1.0 / (1.0 + np.exp(-y))


def standardize(P):
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
# ---------------------------------------------------------------------------------


def compute_spei(df, ref_end_year):
    """SPEI for every timescale, fitting each calendar month on years <= ref_end_year."""
    df = df.copy()
    n_fallback = 0
    for k_scale in TIMESCALES:
        dcol, scol = f"D{k_scale}_mm", f"SPEI_{k_scale}"
        df[dcol] = np.nan
        df[scol] = np.nan
        for div in sorted(df["division"].unique()):
            sub = df.loc[df["division"] == div].sort_values("date")
            idx = sub.index
            Dk = sub["D_mm_month"].rolling(window=k_scale, min_periods=k_scale).sum().values
            df.loc[idx, dcol] = Dk
            months, years = sub["month"].values, sub["year"].values
            for mo in range(1, 13):
                sel_all = (months == mo) & np.isfinite(Dk)
                sel_ref = sel_all & (years <= ref_end_year)
                if sel_ref.sum() < 10:
                    continue
                fit = fit_glo(Dk[sel_ref])
                if fit is None:
                    n_fallback += 1
                    # empirical fallback on the reference sample (none expected)
                    ref = np.sort(Dk[sel_ref])
                    n = len(ref)
                    pos = (np.arange(1, n + 1) - 0.44) / (n + 0.12)
                    F = np.interp(Dk[sel_all], ref, pos)
                else:
                    F = glo_cdf(Dk[sel_all], *fit)
                df.loc[idx[sel_all], scol] = standardize(1.0 - F)
    for k_scale in TIMESCALES:
        df[f"class_{k_scale}"] = df[f"SPEI_{k_scale}"].apply(classify)
    return df, n_fallback


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def main():
    out_dir = CFG.SPEI_REFIT_CSV.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    src = pd.read_csv(CFG.PETWB_CSV, parse_dates=["date"])
    src = src.sort_values(["division", "date"]).reset_index(drop=True)
    thesis = pd.read_csv(CFG.SPEI_CSV, parse_dates=["date"])
    thesis = thesis.sort_values(["division", "date"]).reset_index(drop=True)

    # ---- 1. self-check: full reference period must reproduce the thesis file ----
    full, _ = compute_spei(src, ref_end_year=2024)
    worst = 0.0
    for k in TIMESCALES:
        m = full[f"SPEI_{k}"].notna()
        worst = max(worst, float(np.max(np.abs(full.loc[m, f"SPEI_{k}"].values
                                               - thesis.loc[m, f"SPEI_{k}"].values))))
    print(f"[self-check] rebuild of the thesis SPEI (fit 1980-2024): max |diff| = {worst:.2e}")
    if worst > 1e-6:
        sys.exit("STOP: the copied SPEI code does not reproduce the thesis file.")
    print("[self-check] PASSED -- the refit below uses identical code.")

    # ---- 2. refit on 1980-2018 only ----------------------------------------------
    refit, n_fb = compute_spei(src, ref_end_year=REF_END_YEAR)
    print(f"[refit] fitted on {REF_END_YEAR} and earlier; empirical fallbacks: {n_fb}")
    keep = (["division", "date", "year", "month", "P_mm_month", "PET_mm_month", "D_mm_month"]
            + [f"D{k}_mm" for k in TIMESCALES] + [f"SPEI_{k}" for k in TIMESCALES]
            + [f"class_{k}" for k in TIMESCALES])
    keep = [c for c in keep if c in refit.columns]
    refit[keep].to_csv(CFG.SPEI_REFIT_CSV, index=False)
    print(f"[refit] saved {CFG.SPEI_REFIT_CSV}")

    # ---- 3. compare over the test period -------------------------------------------
    rows = []
    test = thesis["year"] >= TEST_START
    for k in TIMESCALES:
        a = thesis.loc[test, f"SPEI_{k}"].values
        b = refit.loc[test, f"SPEI_{k}"].values
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        da, db = a <= THRESHOLD, b <= THRESHOLD
        rows.append(dict(
            timescale=k, n=int(m.sum()),
            r=float(np.corrcoef(a, b)[0, 1]),
            mean_diff_refit_minus_thesis=float(np.mean(b - a)),
            max_abs_diff=float(np.max(np.abs(b - a))),
            drought_months_thesis=int(da.sum()),
            drought_months_refit=int(db.sum()),
            drought_agreement_pct=float(100 * np.mean(da == db)),
            kappa=float(kappa(da, db)),
        ))
    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "spei_refit_comparison.csv", index=False, float_format="%.4f")
    print("\nTest period 2019-2024: thesis SPEI (fit 1980-2024) vs refit (fit 1980-2018)")
    print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
