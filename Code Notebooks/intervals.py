"""Prediction intervals for actual wind from the day-ahead forecast, tested out of sample.

Interval = forecast + q * capacity, with q a quantile of error_pct/100 (error = actual - forecast).
  80% interval = [q10, q90]; 95% interval = [q05, q95].
Method A: empirical quantiles of error_pct by hour x season (train only).
Method B: statsmodels QuantReg of error_pct on forecast % of capacity (linear + squared),
          hour dummies and season dummies, one fit per quantile (train only).
Usage: python intervals.py [main|neso]   (uses the panel's own train/test split)
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

track = sys.argv[1] if len(sys.argv) > 1 else "main"
df = pd.read_parquet({"main": "data/clean/panel.parquet", "neso": "data/clean/panel_neso.parquet"}[track])
t = pd.to_datetime(df.start_time_utc, utc=True).dt.tz_convert("Europe/London")
df["hour"] = t.dt.hour
df["season"] = t.dt.month.map(lambda m: "Winter" if m in (12, 1, 2) else "Spring" if m in (3, 4, 5)
                              else "Summer" if m in (6, 7, 8) else "Autumn")
df["fc_pct"] = df.forecast_mw / df.capacity_mw * 100
QS = [0.05, 0.10, 0.90, 0.95]
train, test = df[df.split == "train"].copy(), df[df.split == "test"].copy()
print(f"=== track {track}: train {len(train)} ({train.settlement_date.min()}..{train.settlement_date.max()}), "
      f"test {len(test)} ({test.settlement_date.min()}..{test.settlement_date.max()}) ===")

# ---- Method A
qa = train.groupby(["hour", "season"]).error_pct.quantile(QS).unstack()
qa.columns = [f"q{int(q * 100):02d}" for q in QS]
A = test[["hour", "season"]].join(qa, on=["hour", "season"])


# ---- Method B
def design(d):
    X = pd.DataFrame({"fc": d.fc_pct, "fc2": d.fc_pct ** 2}, index=d.index)
    X = X.join(pd.get_dummies(d.hour, prefix="h", drop_first=True).astype(float))
    X = X.join(pd.get_dummies(d.season, prefix="s").drop(columns="s_Autumn").astype(float))
    return sm.add_constant(X, has_constant="add")


Xtr = design(train)
Xte = design(test).reindex(columns=Xtr.columns, fill_value=0.0)
B = pd.DataFrame(index=test.index)
for q in QS:
    res = sm.QuantReg(train.error_pct, Xtr).fit(q=q, max_iter=5000)
    B[f"q{int(q * 100):02d}"] = Xte @ res.params
    print(f"  QuantReg q={q}: fc coef {res.params['fc']:.3f}, fc2 {res.params['fc2']:.4f}")
crossed = (B.q05 > B.q10) | (B.q10 > B.q90) | (B.q90 > B.q95)
print(f"  quantile crossings in test: {crossed.sum()} rows ({crossed.mean():.2%}); sorted to repair")
B = pd.DataFrame(np.sort(B.values, axis=1), index=B.index, columns=B.columns)


def evaluate(Q, label):
    out = []
    for nominal, lo, hi in ((80, "q10", "q90"), (95, "q05", "q95")):
        lo_mw = test.forecast_mw + Q[lo] / 100 * test.capacity_mw
        hi_mw = test.forecast_mw + Q[hi] / 100 * test.capacity_mw
        inside = (test.actual_mw >= lo_mw) & (test.actual_mw <= hi_mw)
        below = test.actual_mw < lo_mw
        above = test.actual_mw > hi_mw
        d = pd.DataFrame({"inside": inside, "below": below, "above": above,
                          "width_mw": hi_mw - lo_mw, "width_pct": (hi_mw - lo_mw) / test.capacity_mw * 100,
                          "season": test.season, "day": test.settlement_date})
        for scope, g in [("All", d)] + [(s, x) for s, x in d.groupby("season")]:
            # day-clustered SE of coverage
            c = g.groupby("day").inside.agg(["sum", "count"])
            p = g.inside.mean(); n = len(g); G = len(c)
            se = np.nan if G < 2 else np.sqrt(G / (G - 1) *((c["sum"] - p * c["count"]) ** 2).sum()) / n
            out.append({"method": label, "nominal": nominal, "scope": scope, "n": n,
                        "coverage_%": 100 * p, "se_%": 100 * se, "miss_below_%": 100 * g.below.mean(),
                        "miss_above_%": 100 * g.above.mean(), "avg_width_MW": g.width_mw.mean(),
                        "avg_width_%cap": g.width_pct.mean()})
    return pd.DataFrame(out)


res = pd.concat([evaluate(A, "A empirical hour x season"), evaluate(B, "B QuantReg")])
pd.set_option("display.width", 200)
for nominal in (80, 95):
    print(f"\n--- nominal {nominal}% ---")
    print(res[res.nominal == nominal].drop(columns="nominal").round(1).to_string(index=False))
res.to_csv(f"data/clean/interval_coverage_{track}.csv", index=False)
a = res[(res.nominal == 80) & (res.scope == "All")].set_index("method")
for m in a.index:
    print(f'\nHeadline ({m}): 80% intervals captured {a.loc[m, "coverage_%"]:.1f}% of {track}-test outcomes '
          f'(avg width {a.loc[m, "avg_width_MW"]:.0f} MW, {a.loc[m, "avg_width_%cap"]:.1f}% of capacity)')
