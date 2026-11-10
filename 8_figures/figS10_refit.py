"""
Fig S10  --  SPEI reference-period sensitivity (F1; Table S18A; no thesis figure)

1 x 4 scatter, SPEI-1/3/6/12, over the test period (2019-2024, 576 division-months):
x = SPEI fitted on the full 1980-2024 record (the thesis SPEI, used by every model);
y = SPEI refitted on 1980-2018 only. 1:1 line; drought threshold (-1) on both axes,
so the off-diagonal quadrants hold the months whose drought class changes. Each panel
prints r, the mean shift, the drought-month counts and the agreement / kappa, all
recomputed from the two files and checked against Table S18A.

Sources: divisional_spei_1980_2024.csv (thesis), _spei_refit/
divisional_spei_1980_2024_refit1980-2018.csv (Stage 2), TableS18a_spei_refit_agreement.csv.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import journal_style as js
import fig_data as D
import p1_fig_config as C

SCALES = [1, 3, 6, 12]


def load():
    refit = C.OUT_ROOT / "_spei_refit" / "divisional_spei_1980_2024_refit1980-2018.csv"
    if not refit.exists():
        raise FileNotFoundError(f"{refit} not found")
    a = pd.read_csv(C.SPEI_CSV)
    b = pd.read_csv(refit)
    for d in (a, b):
        d["_date"] = pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp()
    m = a.merge(b, on=["division", "_date"], suffixes=("_full", "_refit"))
    return m[(m._date >= "2019-01-01") & (m._date <= "2024-12-01")].reset_index(drop=True)


def stats(x, y):
    dx, dy = x <= -1, y <= -1
    agree = (dx == dy).mean()
    pe = dx.mean() * dy.mean() + (1 - dx.mean()) * (1 - dy.mean())
    return dict(n=len(x), r=np.corrcoef(x, y)[0, 1], shift=(y - x).mean(),
                maxdiff=np.abs(y - x).max(), dm_full=int(dx.sum()), dm_refit=int(dy.sum()),
                agree=100 * agree, kappa=(agree - pe) / (1 - pe))


def make(outdir, log):
    log.section("Fig S10 - SPEI reference-period sensitivity")
    m = load()
    tab = D.table("TableS18A_spei_refit_agreement.csv").set_index("timescale")
    st = {}
    for k in SCALES:
        s = stats(m[f"SPEI_{k}_full"].values, m[f"SPEI_{k}_refit"].values)
        t = tab.loc[k]
        ok = (s["n"] == t.n and abs(s["r"] - t.r) < 5e-5 and abs(s["shift"] - t.mean_diff_refit_minus_thesis) < 5e-5
              and abs(s["maxdiff"] - t.max_abs_diff) < 5e-5 and s["dm_full"] == t.drought_months_thesis
              and s["dm_refit"] == t.drought_months_refit and abs(s["agree"] - t.drought_agreement_pct) < 5e-4
              and abs(s["kappa"] - t.kappa) < 5e-5)
        if not ok:
            log.fail(f"SPEI-{k}: recomputed agreement differs from Table S18A: {s}")
        st[k] = s
    log.ok("r, mean shift, max |diff|, drought months, agreement and kappa recomputed from the "
           "two SPEI files equal Table S18A for SPEI-1/3/6/12 (576 division-months each)")
    allv = np.concatenate([m[f"SPEI_{k}_{v}"].values for k in SCALES for v in ("full", "refit")])
    lim = np.ceil(np.abs(allv).max() * 2) / 2
    js.apply()
    fig, axes = plt.subplots(1, 4, figsize=(js.WIDTH["double"], 60 * js.MM), sharex=True, sharey=True)
    for ax, k in zip(axes, SCALES):
        x, y, s = m[f"SPEI_{k}_full"], m[f"SPEI_{k}_refit"], st[k]
        ax.axvspan(-lim, -1, ymin=0, ymax=1, color="#FBEDED", lw=0, zorder=0)
        ax.axhspan(-lim, -1, color="#FBEDED", lw=0, zorder=0)
        ax.axvline(-1, color=js.GREY, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        ax.axhline(-1, color=js.GREY, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.8, ls=(0, (3, 1.6)), zorder=2)
        ax.scatter(x, y, s=2.2, color="#3C3C3C", alpha=0.45, lw=0, zorder=3)
        ax.text(0.97, 0.03,
                f"r = {s['r']:.3f}\nshift {s['shift']:+.3f}".replace("-", "−") +
                f"\ndrought {s['dm_full']} → {s['dm_refit']}\nagreement {s['agree']:.1f}%\nκ = {s['kappa']:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, linespacing=1.15)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"SPEI-{k}", fontweight="bold", pad=3)
        ax.set_xlabel("SPEI, fitted 1980–2024")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
    axes[0].set_ylabel("SPEI, fitted 1980–2018")
    js.panel_labels(list(axes), dx=-4, dy=10)
    return js.save(fig, "FigS10", outdir)


if __name__ == "__main__":
    log = D.CheckLog()
    make(C.FIG_OUT, log)
    print("\n".join(log.lines))
