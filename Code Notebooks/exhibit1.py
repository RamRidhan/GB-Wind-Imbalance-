"""Exhibit 1: describe the day-ahead wind forecast error (sign: error = actual - forecast; negative = over-forecast)."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

track = sys.argv[1] if len(sys.argv) > 1 else "main"
path = {"main": "data/clean/panel.parquet", "neso": "data/clean/panel_neso.parquet"}[track]
df = pd.read_parquet(path)
t = pd.to_datetime(df.start_time_utc, utc=True).dt.tz_convert("Europe/London")
df["hour"] = t.dt.hour
df["month"] = t.dt.month
df["season"] = df.month.map(lambda m: "Winter" if m in (12, 1, 2) else "Spring" if m in (3, 4, 5)
                            else "Summer" if m in (6, 7, 8) else "Autumn")
# quintile bounds from the TRAIN split only (no look-ahead into the test period)
_b = df.loc[df.split == "train", "forecast_mw"].quantile([0, .2, .4, .6, .8, 1]).to_numpy()
_b[0], _b[-1] = -np.inf, np.inf
df["quintile"] = pd.cut(df.forecast_mw, _b, labels=[f"Q{i}" for i in range(1, 6)])


def cluster_mean(x, g):
    """Mean, day-clustered SE (CR0 with G/(G-1) correction), t, p (normal approx)."""
    x = np.asarray(x, float); n = len(x); m = x.mean()
    s = pd.Series(x - m).groupby(np.asarray(g)).sum()
    G = len(s)
    se = np.sqrt(G / (G - 1) * (s ** 2).sum()) / n
    from math import erf, sqrt
    t_ = m / se
    p = 2 * (1 - 0.5 * (1 + erf(abs(t_) / sqrt(2))))
    return m, se, t_, p


def stats(d):
    e, c = d.error_mw, d.error_pct
    return pd.Series({"n": len(d), "bias_mw": e.mean(), "mae_mw": e.abs().mean(),
                      "rmse_mw": np.sqrt((e ** 2).mean()), "bias_pct": c.mean(),
                      "mae_pct": c.abs().mean(), "rmse_pct": np.sqrt((c ** 2).mean())})


print(f"=== track: {track}  {df.settlement_date.min()} to {df.settlement_date.max()} ===")
print("Overall:\n", stats(df).round(2).to_string())
m, se, t_, p = cluster_mean(df.error_mw, df.settlement_date)
print(f"Bias test (MW): mean {m:.1f}, day-clustered SE {se:.1f}, t = {t_:.2f}, p = {p:.3g}, "
      f"days = {df.settlement_date.nunique()}")
for col in ("season", "quintile"):
    print(f"\nBy {col}:\n", df.groupby(col, observed=True).apply(stats).round(2).to_string())
print("\nForecast quintile bounds (MW):", df.forecast_mw.quantile([0, .2, .4, .6, .8, 1]).round(0).tolist())
print("\nBy month:\n", df.groupby("month").apply(stats)[["n", "bias_mw", "bias_pct"]].round(2).to_string())

q = []
for k, d in df.groupby("quintile", observed=True):
    m, se, t_, p = cluster_mean(d.error_pct, d.settlement_date)
    q.append((k, m, 1.96 * se, p))
print("\nQuintile bias test (error_pct, clustered):")
for k, m, ci, p in q:
    print(f"  {k}: {m:.2f}% ± {ci:.2f}  p={p:.3g}")

h = []
for k, d in df.groupby("hour"):
    m, se, *_ = cluster_mean(d.error_pct, d.settlement_date)
    h.append((k, m, 1.96 * se))
h = pd.DataFrame(h, columns=["hour", "m", "ci"])
print("\nBy hour (error_pct, clustered 95% CI):\n", h.round(2).T.to_string(header=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
ax[0].axhline(0, color="#888", lw=.8)
ax[0].errorbar(h.hour, h.m, yerr=h.ci, fmt="o-", color="#1f5f99", capsize=2, ms=4)
ax[0].set(xlabel="Hour of day (London)", ylabel="Mean error, % of capacity", title="By hour")
ax[0].set_xticks(range(0, 24, 3))
qq = pd.DataFrame(q, columns=["q", "m", "ci", "p"])
ax[1].axhline(0, color="#888", lw=.8)
ax[1].errorbar(range(5), qq.m, yerr=qq.ci, fmt="o", color="#b5451b", capsize=3, ms=6)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(["Q1\nlow", "Q2", "Q3", "Q4", "Q5\nhigh"])
ax[1].set(xlabel="Forecast quintile", title="By forecast level")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Exhibit 1: day-ahead wind forecast error (actual − forecast; negative = over-forecast), "
             f"{track} track, 95% day-clustered CIs", fontsize=9)
fig.tight_layout()
out = f"data/clean/exhibit1_{track}.png"
fig.savefig(out, dpi=160)
print("saved", out)
