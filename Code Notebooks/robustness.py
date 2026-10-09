"""Stress tests (main track). Parameters fitted on 2024-25 (train), applied to 2026 (test).

1. Out-of-sample price impact: OLS fitted on train, predict test spread; incremental OOS R2 of shortfall.
2. Robustness variants: baseline | drop clock-change days | winsorised spread (0.5/99.5 pct, train bounds)
   | (alt MID provider: N2EXMIDP is 99.7% zero-price rows, so it cannot be used - see output).
   For each: price-impact coefficient (train, test, full; HAC) and Exhibit-3 Strategy B saving on 2026
   with tau re-chosen on train.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

CLOCK = ["2024-03-31", "2024-10-27", "2025-03-30", "2025-10-26", "2026-03-29"]
df0 = pd.read_parquet("data/clean/panel.parquet")
t = pd.to_datetime(df0.start_time_utc, utc=True).dt.tz_convert("Europe/London")
df0["hour"], df0["month"] = t.dt.hour, t.dt.month
df0["season"] = df0.month.map(lambda m: "Winter" if m in (12, 1, 2) else "Spring" if m in (3, 4, 5)
                              else "Summer" if m in (6, 7, 8) else "Autumn")
df0["demand_gw"] = df0.demand_mw / 1000
df0["fc_p"] = df0.forecast_mw / df0.capacity_mw * 1000
df0["act_p"] = df0.actual_mw / df0.capacity_mw * 1000
FORM = "spread ~ shortfall_gw + demand_gw + mid_price + C(hour) + C(month)"
FORM0 = "spread ~ demand_gw + mid_price + C(hour) + C(month)"

# ---- 1. out-of-sample price impact
tr, te = df0[df0.split == "train"], df0[df0.split == "test"]
m1, m0 = smf.ols(FORM, tr).fit(), smf.ols(FORM0, tr).fit()
# 2026 has months 1-9 only; train covers all months so predict() works
sse = lambda m: ((te.spread - m.predict(te)) ** 2).sum()
sst = ((te.spread - tr.spread.mean()) ** 2).sum()
print("=== 1. OOS price impact (fit 2024-25, predict 2026) ===")
print(f"OOS R2 full model {1 - sse(m1) / sst:.4f}; without shortfall {1 - sse(m0) / sst:.4f}; "
      f"incremental from shortfall {(sse(m0) - sse(m1)) / sst:.4f}")
print(f"train coef {m1.params['shortfall_gw']:.2f}; refit on test "
      f"{smf.ols(FORM, te).fit().params['shortfall_gw']:.2f}")


# ---- 2. variants
def coef(d):
    r = smf.ols(FORM, d).fit(cov_type="HAC", cov_kwds={"maxlags": 48})
    return f"{r.params['shortfall_gw']:.2f} ({r.bse['shortfall_gw']:.2f})"


def saving(d):
    """Exhibit 3: Strategy B (tau picked on train) vs A on test, using d['spread'] for prices."""
    d = d.copy()
    d["sys"] = d.mid_price + d.spread                      # system price consistent with spread used
    cost = lambda x, sold: (x.act_p - sold) * (x.mid_price - x.sys) * 0.5
    trn, tst = d[d.split == "train"], d[d.split == "test"]
    err = (d.act_p - d.fc_p) / 1000 * 100                   # error_pct (capacity scaling cancels)
    q = lambda tau: trn.assign(e=err[trn.index]).groupby(["hour", "season"]).e.quantile(tau).rename("q")
    sold = lambda x, tau: (x.fc_p + x[["hour", "season"]].join(q(tau), on=["hour", "season"]).q * 10).clip(lower=0)
    taus = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    best = min(taus, key=lambda x: cost(trn, sold(trn, x)).sum())
    A, B = cost(tst, tst.fc_p).sum(), cost(tst, sold(tst, best)).sum()
    return best, A, B, (1 - B / A) * 100


lo, hi = tr.spread.quantile([0.005, 0.995])
variants = {
    "baseline": df0,
    "drop clock-change days": df0[~df0.settlement_date.astype(str).isin(CLOCK)],
    "winsorised spread (0.5/99.5, train bounds)": df0.assign(spread=df0.spread.clip(lo, hi)),
}
print(f"\n=== 2. Robustness (winsor bounds {lo:.0f}/{hi:.0f} GBP/MWh) ===")
rows = []
for name, d in variants.items():
    best, A, B, s = saving(d)
    rows.append({"variant": name, "n": len(d), "coef_train": coef(d[d.split == 'train']),
                 "coef_test": coef(d[d.split == 'test']), "coef_full": coef(d),
                 "tau": best, "A_2026_GBP": round(A), "B_2026_GBP": round(B), "B_saving_%": round(s, 1)})
out = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", 20)
print(out.to_string(index=False))
out.to_csv("data/clean/robustness.csv", index=False)

# spike share: how much of the spread variation / cost comes from the top 0.5% of periods
s = df0.spread.abs()
top = s >= s.quantile(0.995)
print(f"\ntop 0.5% |spread| periods: {top.sum()}; share of sum of squared spread {((df0.spread[top]) ** 2).sum() / (df0.spread ** 2).sum():.1%}")
