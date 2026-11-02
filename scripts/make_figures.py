#!/usr/bin/env python3
"""
Figures for the README, from the two synthetic runs:
  results/                      run with a planted effect (make_synthetic_data.py default)
  results/control_no_effect/    control run, --effect 0

Left: value-weighted next-month return by Disruption Score quintile.
Right: cumulative long-short return (top minus bottom quintile, gross).
Both use the primary score, ds_level_raw. Output: results/figures/synthetic_validation.png
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [("Planted effect (-1.0 %/mo)", os.path.join(ROOT, "results"), "#2a78d6"),
        ("Control (no effect)", os.path.join(ROOT, "results/control_no_effect"), "#eb6834")]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def quintile_returns(panel_csv):
    d = pd.read_csv(panel_csv, parse_dates=["month"])
    d = d[np.isfinite(d.score) & d.ret_fwd_1m.notna() & (d.mktcap > 0)]
    d["q"] = d.groupby("month").score.transform(
        lambda v: pd.qcut(v.rank(method="first"), 5, labels=False)) + 1
    vw = d.groupby(["month", "q"]).apply(
        lambda g: np.average(g.ret_fwd_1m, weights=g.mktcap), include_groups=False)
    return vw.groupby("q").mean() * 100


fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150, facecolor=SURF)
w = 0.38
for k, (lab, path, col) in enumerate(RUNS):
    q = quintile_returns(os.path.join(path, "tables/panel_for_R.csv"))
    x = np.arange(1, 6) + (k - 0.5) * w
    a1.bar(x, q.values, width=w - 0.04, color=col, label=lab, zorder=3)
    ls = pd.read_csv(os.path.join(path, "tables/ls_series.csv"), parse_dates=["month"])
    cum = (1 + ls.ls).cumprod() - 1
    a2.plot(ls.month, cum * 100, color=col, lw=2, label=lab, zorder=3)
    a2.annotate(f"{cum.iloc[-1]*100:+.0f}%", (ls.month.iloc[-1], cum.iloc[-1] * 100),
                xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)

a1.set_xticks(range(1, 6), ["Q1\nlow DS", "Q2", "Q3", "Q4", "Q5\nhigh DS"])
a1.set_ylabel("Next-month return, % (value-weighted)", color=INK2)
a1.set_title("Return by Disruption Score quintile", loc="left", fontsize=11, color=INK)
a2.set_ylabel("Cumulative return, %", color=INK2)
a2.set_title("Long-short: high minus low DS quintile", loc="left", fontsize=11, color=INK)
for a in (a1, a2):
    a.set_facecolor(SURF)
    a.axhline(0, color=INK2, lw=0.8, zorder=2)
    a.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(GRID)
    a.tick_params(colors=INK2, labelsize=9)
a2.margins(x=0.08)
a1.legend(frameon=False, fontsize=9, loc="lower left")
fig.text(0.01, 0.01, "Synthetic data only (scripts/make_synthetic_data.py, seed 453). "
         "Not a result on real markets.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
os.makedirs(os.path.join(ROOT, "results/figures"), exist_ok=True)
out = os.path.join(ROOT, "results/figures/synthetic_validation.png")
fig.savefig(out, facecolor=SURF)
print("wrote", out)
