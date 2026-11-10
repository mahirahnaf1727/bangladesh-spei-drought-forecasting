"""
station_zero_rule.py  --  the BMD station zero-value rule (Text S2 of the paper).

The BMD compilation (Diptho et al., 2026) stores missing data as 0 mm. Before the
stations of a division are averaged, two kinds of value are set to missing:
  1. every station-year whose annual precipitation total is zero
     (101 of the 1,505 station-years in 1980-2022), and
  2. zero monthly totals in the May-September rainy season
     (5 station-months: Hatiya May-Aug 1993, Kutubdia Sep 1980).
Each divisional mean then uses only the stations that reported in that month.

Used by 2_era5_validation/Validation_1.py and 8_figures/tier1_logic.py.
"""

import numpy as np

RAINY_MONTHS = (5, 9)          # May-September, inclusive


def apply(b, value="P"):
    """b: station-month table with columns Station, year, month and `value` (mm).
    Returns (copy with the zero values set to NaN, one-line report)."""
    b = b.copy()
    annual = b.groupby(["Station", "year"])[value].transform("sum")
    zero_year = (annual == 0) & b[value].notna()
    n_sy = b.loc[zero_year, ["Station", "year"]].drop_duplicates().shape[0]
    n_all = b[["Station", "year"]].drop_duplicates().shape[0]
    b.loc[zero_year, value] = np.nan
    zero_month = (b[value] == 0) & b["month"].between(*RAINY_MONTHS)
    n_sm = int(zero_month.sum())
    b.loc[zero_month, value] = np.nan
    report = (f"  zero-value rule: {n_sy} of {n_all} station-years with a zero annual total and "
              f"{n_sm} zero station-months in May-September set to missing")
    return b, report
