"""Exhibit 2: price impact. spread = system_price - mid_price (GBP/MWh); shortfall_gw > 0 = wind below forecast.
Association ("moved with"), not causal. Usage: python exhibit2.py [main|neso]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

track = sys.argv[1] if len(sys.argv) > 1 else "main"
df = pd.read_parquet({"main": "data/clean/panel.parquet", "neso": "data/clean/panel_neso.parquet"}[track])
t = pd.to_datetime(df.start_time_utc, utc=True).dt.tz_convert("Europe/London")
df["hour"], df["month"] = t.dt.hour, t.dt.month
df["demand_gw"] = df.demand_mw / 1000
FORM = "spread ~ shortfall_gw + demand_gw + mid_price + C(hour) + C(month)"
m = smf.ols(FORM, data=df).fit()
print(f"=== track {track}: {len(df)} half-hours, {df.settlement_date.nunique()} days ===")
b = m.params["shortfall_gw"]
print(f"shortfall coef: {b:.3f} GBP/MWh per 1 GW  (R2 {m.rsquared:.3f})")
print(f"  iid SE {m.bse['shortfall_gw']:.3f}")

# HAC (Newey-West, 48 lags = one day) and day-clustered
hac = smf.ols(FORM, data=df).fit(cov_type="HAC", cov_kwds={"maxlags": 48})
cl = smf.ols(FORM, data=df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.settlement_date)[0]})
print(f"  HAC(48) SE {hac.bse['shortfall_gw']:.3f}  CI [{hac.conf_int().loc['shortfall_gw', 0]:.2f}, "
      f"{hac.conf_int().loc['shortfall_gw', 1]:.2f}]")
print(f"  day-clustered SE {cl.bse['shortfall_gw']:.3f}")

# Day-block bootstrap (resample whole days, 1000x) via per-day X'X and X'y
y, X = m.model.endog, m.model.exog
col = m.model.exog_names.index("shortfall_gw")
day = pd.factorize(df.settlement_date)[0]
G = day.max() + 1
XtX = np.zeros((G, X.shape[1], X.shape[1])); Xty = np.zeros((G, X.shape[1]))
for g in range(G):
    xi = X[day == g]; XtX[g] = xi.T @ xi; Xty[g] = xi.T @ y[day == g]
rng = np.random.default_rng(0)
boots = []
for _ in range(1000):
    w = np.bincount(rng.integers(0, G, G), minlength=G)
    boots.append(np.linalg.pinv(np.tensordot(w, XtX, 1)) @ np.tensordot(w, Xty, 1))
bs = np.array(boots)[:, col]
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"  day-block bootstrap (1000): 95% CI [{lo:.2f}, {hi:.2f}], SD {bs.std():.3f}")

# Non-linearity: bucketed shortfall
edges = [-np.inf, -1, 1, 2, np.inf]
labels = ["Surplus (<-1 GW)", "+/-1 GW (base)", "1-2 GW shortfall", ">2 GW shortfall"]
df["bucket"] = pd.cut(df.shortfall_gw, edges, labels=labels)
mb = smf.ols("spread ~ C(bucket, Treatment('+/-1 GW (base)')) + demand_gw + mid_price + C(hour) + C(month)",
             data=df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.settlement_date)[0]})
tab = df.groupby("bucket", observed=True).agg(n=("spread", "size"), mean_spread=("spread", "mean"),
                                              median_spread=("spread", "median"),
                                              p95_spread=("spread", lambda s: s.quantile(.95)))
adj = {}
for l in labels:
    k = f"C(bucket, Treatment('+/-1 GW (base)'))[T.{l}]"
    adj[l] = (mb.params[k], mb.bse[k]) if k in mb.params else (0.0, np.nan)
tab["adj_vs_base"] = [adj[l][0] for l in tab.index]; tab["adj_se"] = [adj[l][1] for l in tab.index]
pd.set_option("display.width", 200)
print("\nBy shortfall bucket (adj_vs_base: controls for demand, MID, hour, month; day-clustered SE):")
print(tab.round(2).to_string())
tab.round(3).to_csv(f"data/clean/exhibit2_buckets_{track}.csv")

# Sub-sample stability
for s, d in df.groupby(df.split):
    r = smf.ols(FORM, data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 48})
    print(f"  {s}: coef {r.params['shortfall_gw']:.2f} (HAC SE {r.bse['shortfall_gw']:.2f}, n={len(d)})")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].bar(range(4), tab.mean_spread, color="#b5451b", width=.6)
ax[0].errorbar(range(4), tab.mean_spread, yerr=1.96 * df.groupby("bucket", observed=True).spread.sem(),
               fmt="none", color="k", capsize=3, lw=1)
ax[0].axhline(0, color="#888", lw=.8)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["Surplus\n(<-1 GW)", "+/-1 GW", "1-2 GW\nshortfall", ">2 GW\nshortfall"])
ax[0].set(ylabel="Mean spread, GBP/MWh (system - MID)", title="Spread by wind shortfall bucket")
ax[1].hist(bs, bins=40, color="#1f5f99")
ax[1].axvline(b, color="k", lw=1); ax[1].axvline(lo, color="#888", ls="--"); ax[1].axvline(hi, color="#888", ls="--")
ax[1].set(xlabel="GBP/MWh per 1 GW shortfall", title="Day-block bootstrap of the coefficient")
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Exhibit 2: spread moved with wind shortfall ({track} track); association, not causal effect", fontsize=9)
fig.tight_layout(); fig.savefig(f"data/clean/exhibit2_{track}.png", dpi=160)
