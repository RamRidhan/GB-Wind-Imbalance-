"""Extra findings for the note: materiality, cost concentration, regimes, safer sizing rules, value ceiling."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
pd.set_option("display.width", 200)
df = pd.read_parquet("data/clean/panel.parquet")
t = pd.to_datetime(df.start_time_utc, utc=True).dt.tz_convert("Europe/London")
df["hour"], df["month"] = t.dt.hour, t.dt.month
df["season"] = df.month.map(lambda m: "Winter" if m in (12,1,2) else "Spring" if m in (3,4,5) else "Summer" if m in (6,7,8) else "Autumn")
df["demand_gw"] = df.demand_mw / 1000
df["fc_p"] = df.forecast_mw / df.capacity_mw * 1000; df["act_p"] = df.actual_mw / df.capacity_mw * 1000
df["costA"] = (df.act_p - df.fc_p) * (df.mid_price - df.system_price) * .5
df["gen"] = df.act_p * .5
df["year"] = pd.to_datetime(df.settlement_date).dt.year
te, tr = df[df.split == "test"], df[df.split == "train"]

print("== a. materiality (1 GW, strategy A)")
for lab, d in [("2025", df[df.year == 2025]), ("2026 Jan-Sep", te)]:
    rev = (d.gen * d.mid_price).sum()
    print(f"{lab}: MID-valued revenue £{rev/1e6:.1f}m; imbalance cost £{d.costA.sum()/1e3:.0f}k = {d.costA.sum()/rev*100:.2f}% of revenue; "
          f"mean MID £{(d.gen*d.mid_price).sum()/d.gen.sum():.1f}/MWh; cost £{d.costA.sum()/d.gen.sum():.2f}/MWh")
print("== b. concentration (all years, A)")
c = df.costA; gross = c[c > 0].sum()
for p in (0.01, 0.05):
    thr = c.quantile(1 - p); print(f"top {p:.0%} of half-hours = {c[c>=thr].sum()/gross:.0%} of gross loss; net cost share {c[c>=thr].sum()/c.sum():.0%}")
d_ = df.groupby("settlement_date").costA.sum().sort_values(ascending=False)
print(f"top 10 days = {d_.head(10).sum()/d_[d_>0].sum():.0%} of gross daily loss; worst day £{d_.iloc[0]:,.0f} ({d_.index[0]})")
print("worst 5 days:", [(str(i), round(v)) for i, v in d_.head(5).items()])
print("gains periods share", (c < 0).mean().round(3), "total gross gain", round(c[c<0].sum()), "gross loss", round(gross))
print("== c. A cost by shortfall bucket (all)")
df["bk"] = pd.cut(df.shortfall_gw, [-np.inf, -1, 1, 2, np.inf], labels=["surplus", "+-1", "1-2 short", ">2 short"])
print(df.groupby("bk", observed=True).agg(n=("costA","size"), cost=("costA","sum"), per_mwh=("costA", lambda s: s.sum()/df.loc[s.index,"gen"].sum())).round(2))
print("== d. A cost by hour block and season (all)")
df["blk"] = pd.cut(df.hour, [-1,5,9,15,20,23], labels=["night 0-5","morning 6-9","midday 10-15","evening 16-20","late 21-23"])
for k in ("blk","season"):
    print(df.groupby(k, observed=True).apply(lambda g: pd.Series({"cost": g.costA.sum(), "per_mwh": g.costA.sum()/g.gen.sum(), "sd_spread": g.spread.std()})).round(2))
print("== e. slope heterogeneity (HAC48; spread ~ shortfall + controls within subsample)")
F = "spread ~ shortfall_gw + demand_gw + mid_price + C(hour) + C(month)"
def slope(d):
    r = smf.ols(F, d).fit(cov_type="HAC", cov_kwds={"maxlags": 48}); return f"{r.params['shortfall_gw']:.2f} ({r.bse['shortfall_gw']:.2f}) n={len(d)}"
df["dterc"] = pd.qcut(df.demand_gw, 3, labels=["low","mid","high"]); df["mterc"] = pd.qcut(df.mid_price, 3, labels=["low","mid","high"])
for k, d in df.groupby("dterc", observed=True): print("demand", k, slope(d))
for k, d in df.groupby("mterc", observed=True): print("MID", k, slope(d))
for k, d in df.groupby(df.hour.between(16,19)): print("evening peak" if k else "other hours", slope(d))
for k, d in df.groupby("season"): print(k, slope(d.assign(month=d.month)) if d.month.nunique()>1 else "")
print("== f. spike probability by bucket (|spread|>50)")
df["sp_hi"] = df.spread > 50; df["sp_lo"] = df.spread < -50
print(df.groupby("bk", observed=True)[["sp_hi","sp_lo"]].mean().round(3))
print("overall P(spread>50)", df.sp_hi.mean().round(3), "P(<-50)", df.sp_lo.mean().round(3))
print("== g. moderate shading, test 2026 (tau fixed from train grid)")
def sold(d, tau):
    q = tr.groupby(["hour","season"]).error_pct.quantile(tau).rename("q")
    return (d.fc_p + d[["hour","season"]].join(q, on=["hour","season"]).q * 10).clip(lower=0)
lo, hi = tr.spread.quantile([.005, .995])
for tau in (0.01, 0.1, 0.2, 0.3, 0.5):
    s_ = sold(te, tau); out = []
    for nm, spread in (("raw", te.spread), ("winsor", te.spread.clip(lo, hi))):
        sys_ = te.mid_price + spread
        A = ((te.act_p - te.fc_p) * (te.mid_price - sys_) * .5).sum(); B = ((te.act_p - s_) * (te.mid_price - sys_) * .5).sum()
        out.append(f"{nm}: A {A/1e3:.0f}k B {B/1e3:.0f}k saving {100*(1-B/A):.0f}%")
    print(f"tau {tau}: fwd given up {((te.fc_p-s_)*.5).sum()/te.gen.sum()*100:.0f}% | " + " | ".join(out))
print("== h. sizing: error quantiles (train) in MW per 1 GW, and 2026 realised")
for nm, d in (("train", tr), ("test", te)):
    q = (d.error_pct / 100 * 1000).quantile([.05,.1,.5,.9,.95]).round(0); print(nm, dict(q))
print("== i. value ceiling: A cost is upper bound on what a perfect forecast saves (1 GW)")
for y in (2025, 2026): d = df[df.year == y]; print(y, f"£{d.costA.sum()/1e3:.0f}k period; gross loss £{d.costA[d.costA>0].sum()/1e3:.0f}k; gains £{d.costA[d.costA<0].sum()/1e3:.0f}k")
print("== j. error vs system: corr(error, demand outturn change?) skip; share of days where |mean error|>1GW:", (df.groupby('settlement_date').error_mw.mean().abs() > 1000).mean().round(3))
print("rolling 30d coverage-like: RMSE by quarter"); df["q_"] = pd.to_datetime(df.settlement_date).dt.to_period("Q")
print(df.groupby("q_").apply(lambda g: np.sqrt((g.error_mw**2).mean())).round(0).to_string())
