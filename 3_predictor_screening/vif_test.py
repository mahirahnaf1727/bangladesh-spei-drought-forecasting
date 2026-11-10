"""
VIF (Variance Inflation Factor) multicollinearity check on the predictor set.

PURPOSE
  Cross-check the PHYSICS-based collinearity decision (drop t2m; keep mx2t+mn2t; exclude D/PET;
  keep d2m de-emphasized) against a numerical test. VIF measures how well each predictor is
  explained by the OTHERS - high VIF = redundant.

WHAT IT RUNS  (VIF is a feature-vs-feature test: no target, no lags)
  1. FULL set (9 base vars, incl. t2m and both wind components' speed) - shows the raw problem.
  2. REDUCED set (your 7-var decomposed choice: tp, mx2t, mn2t, ssrd, tcc, wind, d2m) - confirms
     whether the physics choice actually resolved it.
  3. Iterative drop: greedily removes the highest-VIF variable until all remain below THRESHOLD,
     as an independent "what would a purely numerical method drop?" comparison.

  All on DESEASONALIZED (anomaly) predictors, pooled across divisions - matching how they enter
  the model. (Raw variables would show inflated VIF from shared seasonal cycles.)

READING VIF
  >=10 : serious multicollinearity (redundant)   |   5-10 : moderate   |   <5 : acceptable
  If the numerical drop-list matches the physics drop-list (t2m; and tp/D overlap) -> corroboration.
  If it disagrees -> worth knowing before committing.

Reads only; writes a CSV + console table.
"""

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import paths as P  # noqa: E402  (one relative-path config for the whole package)
SPEI_CSV = str(P.SPEI_CSV)
VARS_CSV = str(P.VARS_CSV)
PET_CSV  = str(P.PETWB_CSV)
OUT_DIR  = str(P.WORK / "3_predictor_screening" / "vif")

THRESHOLD = 10.0   # iterative-drop target

# ----------------------------------------------------------------------------
# Column detection (verified against the real files)
# ----------------------------------------------------------------------------
def _norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())

def find_division_col(df):
    for c in df.columns:
        if _norm(c) in {"division", "div", "divname", "name", "region", "zone"}:
            return c
    for c in df.columns:
        if df[c].dtype == object and 2 <= df[c].nunique() <= 20:
            return c
    raise ValueError("No division column: %s" % list(df.columns))

def build_datetime(df):
    cols = {_norm(c): c for c in df.columns}
    for k in ("date", "time", "datetime", "yearmonth", "ym"):
        if k in cols:
            return pd.to_datetime(df[cols[k]], errors="coerce")
    y = next((cols[k] for k in cols if k in ("year", "yr", "yyyy")), None)
    m = next((cols[k] for k in cols if k in ("month", "mon", "mm")), None)
    if y and m:
        return pd.to_datetime(dict(year=df[y].astype(int), month=df[m].astype(int), day=1), errors="coerce")
    for c in df.columns:
        p = pd.to_datetime(df[c], errors="coerce")
        if p.notna().mean() > 0.9:
            return p
    raise ValueError("No date column: %s" % list(df.columns))

def pick(df, *aliases):
    cols = {_norm(c): c for c in df.columns}
    for a in aliases:
        if _norm(a) in cols:
            return cols[_norm(a)]
    return None

def standardize_keys(df):
    out = df.copy()
    out["_division"] = out[find_division_col(df)].astype(str).str.strip()
    out["_date"] = build_datetime(df)
    return out

def deseasonalize(df, cols):
    out = df.copy()
    out["_month"] = out["_date"].dt.month
    for c in cols:
        out[c] = out.groupby(["_division", "_month"])[c].transform(lambda s: s - s.mean())
    return out

# ----------------------------------------------------------------------------
# VIF: VIF_j = 1 / (1 - R^2_j), R^2 from regressing var j on all others
# ----------------------------------------------------------------------------
def compute_vif(X):
    """X: DataFrame of predictors (no NaN). Returns Series of VIF per column."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xc = X.copy()
    Xc.insert(0, "_const", 1.0)                 # intercept required for correct VIF
    vals = {}
    for i, col in enumerate(Xc.columns):
        if col == "_const":
            continue
        vals[col] = variance_inflation_factor(Xc.values, i)
    return pd.Series(vals).sort_values(ascending=False)

def iterative_drop(X, threshold):
    """Greedily drop the highest-VIF column until all < threshold. Returns (kept, drop_order)."""
    cols = list(X.columns)
    dropped = []
    while len(cols) > 1:
        vif = compute_vif(X[cols])
        top, val = vif.index[0], vif.iloc[0]
        if val < threshold:
            break
        dropped.append((top, round(float(val), 2)))
        cols.remove(top)
    return cols, dropped

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        import statsmodels  # noqa
    except Exception:
        sys.exit("Needs statsmodels. Run: pip install statsmodels")

    vars_ = standardize_keys(pd.read_csv(VARS_CSV))
    v = {k: pick(vars_, *al) for k, al in {
        "tp": ("tp", "precip"), "t2m": ("t2m", "tmean"), "mx2t": ("mx2t", "tmax"),
        "mn2t": ("mn2t", "tmin"), "d2m": ("d2m", "dewpoint"), "ssrd": ("ssrd", "solar"),
        "tcc": ("tcc", "cloud"), "u10": ("u10",), "v10": ("v10",),
    }.items()}
    keep = {k: c for k, c in v.items() if c is not None}
    print("Detected variables:", keep)

    df = vars_[["_division", "_date"] + list(keep.values())].rename(
        columns={c: k for k, c in keep.items()})
    if {"u10", "v10"}.issubset(df.columns):
        df["wind"] = np.hypot(df["u10"], df["v10"])

    # optional: pull PET/D to show what including them does to tp's VIF
    try:
        pet = standardize_keys(pd.read_csv(PET_CSV))
        pet_col = pick(pet, "pet_mm_month", "pet", "et0")
        d_col = pick(pet, "d_mm_month", "d_month", "d")
        addcols = ["_division", "_date"] + [c for c in (pet_col, d_col) if c]
        petsub = pet[addcols].rename(columns={**({pet_col: "PET"} if pet_col else {}),
                                              **({d_col: "D"} if d_col else {})})
        df = df.merge(petsub, on=["_division", "_date"], how="inner")
    except Exception as e:
        print("(PET/D not merged: %s)" % e)

    base9 = [c for c in ["tp", "t2m", "mx2t", "mn2t", "d2m", "ssrd", "tcc", "wind"] if c in df.columns]
    reduced7 = [c for c in ["tp", "mx2t", "mn2t", "ssrd", "tcc", "wind", "d2m"] if c in df.columns]
    withDPET = base9 + [c for c in ["PET", "D"] if c in df.columns]

    df = deseasonalize(df, list(set(base9 + withDPET)))

    def run(label, cols):
        sub = df[cols].dropna()
        vif = compute_vif(sub)
        print("\n=== %s  (n=%d) ===" % (label, len(sub)))
        for name, val in vif.items():
            tag = "SERIOUS" if val >= 10 else ("moderate" if val >= 5 else "ok")
            print("  %-6s VIF = %8.2f   [%s]" % (name, val, tag))
        return vif

    print("\n" + "#" * 60)
    print("# VIF on DESEASONALIZED (anomaly) predictors, pooled")
    print("#" * 60)

    vif_full = run("FULL 8-var set (incl. t2m)", base9)
    vif_dpet = run("FULL + PET + D (shows tp/D overlap)", withDPET) if "D" in df.columns else None
    vif_red  = run("REDUCED 7-var decomposed set (physics choice)", reduced7)

    print("\n--- Iterative numerical drop (target VIF < %.0f), starting from full+PET+D ---" % THRESHOLD)
    start = withDPET if "D" in df.columns else base9
    kept, order = iterative_drop(df[start].dropna(), THRESHOLD)
    for name, val in order:
        print("  dropped %-6s (VIF was %.2f)" % (name, val))
    print("  numerical method keeps:", kept)
    print("\n  PHYSICS keeps:", reduced7)
    print("  MATCH?  physics-dropped =", sorted(set(start) - set(reduced7)),
          "| numeric-dropped =", sorted(set(start) - set(kept)))

    # save
    out = pd.DataFrame({"full8": vif_full})
    if vif_dpet is not None:
        out = out.join(vif_dpet.rename("full_plus_DPET"), how="outer")
    out = out.join(vif_red.rename("reduced7"), how="outer")
    out.to_csv(os.path.join(OUT_DIR, "vif_results.csv"))
    print("\nSaved:", os.path.join(OUT_DIR, "vif_results.csv"))
    print("\nInterpret: if numeric-dropped matches physics-dropped, VIF corroborates the physics choice.")

if __name__ == "__main__":
    main()
