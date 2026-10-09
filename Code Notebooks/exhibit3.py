"""Exhibit 3: imbalance cost for a hypothetical 1 GW portfolio.

ASSUMPTION: the portfolio's error profile equals the national one (portfolio_forecast =
forecast_pct_of_capacity * 1000 MW, portfolio error % = national error %). Real portfolios carry
extra idiosyncratic error, so this UNDERSTATES real-world imbalance cost.

cost = (actual - sold) * (MID - system_price) * 0.5   [GBP per half-hour]; POSITIVE = loss.
Identity: revenue = sold*MID + (actual-sold)*system_price = actual*MID - cost, so cost is exactly
what is lost versus selling actual output at MID. Selling less forward (Strategy B) is therefore
captured: when long, the surplus is valued at system_price instead of MID (cost > 0 if sys < MID).
Strategy A: sold = forecast.  Strategy B: sold = max(0, forecast + q_tau(hour, season)*capacity).
tau is chosen by minimising 2024-25 cost; evaluation on 2026 (Jan-Sep).
"""
import numpy as np
import pandas as pd

df = pd.read_parquet("data/clean/panel.parquet")
t = pd.to_datetime(df.start_time_utc, utc=True).dt.tz_convert("Europe/London")
df["hour"] = t.dt.hour
df["season"] = t.dt.month.map(lambda m: "Winter" if m in (12, 1, 2) else "Spring" if m in (3, 4, 5)
                              else "Summer" if m in (6, 7, 8) else "Autumn")
df["year"] = pd.to_datetime(df.settlement_date).dt.year
PORT = 1000.0
df["fc_p"] = df.forecast_mw / df.capacity_mw * PORT                 # portfolio forecast, MW
df["act_p"] = df.fc_p + df.error_pct / 100 * PORT                   # forecast + national error %
assert np.allclose(df.act_p, df.actual_mw / df.capacity_mw * PORT)
df["gen_mwh"] = df.act_p * 0.5
train = df[df.split == "train"]


def cost(d, sold):
    return (d.act_p - sold) * (d.mid_price - d.system_price) * 0.5


def sold_B(d, tau):
    q = train.groupby(["hour", "season"]).error_pct.quantile(tau).rename("q")
    q = d[["hour", "season"]].join(q, on=["hour", "season"]).q
    return (d.fc_p + q / 100 * PORT).clip(lower=0)


taus = np.round(np.r_[0.01, 0.02, np.arange(0.05, 0.501, 0.05)], 2)
grid = pd.DataFrame({"tau": taus, "train_cost_gbp": [cost(train, sold_B(train, x)).sum() for x in taus]})
base = cost(train, train.fc_p).sum()
grid["vs_A_%"] = (1 - grid.train_cost_gbp / base) * 100
best = grid.loc[grid.train_cost_gbp.idxmin(), "tau"]
print("tau grid on 2024-25 (Strategy A train cost GBP %.0f):" % base)
print(grid.round({"tau": 2, "train_cost_gbp": 0, "vs_A_%": 1}).to_string(index=False))
print(f"chosen tau = {best}")

df["costA"] = cost(df, df.fc_p)
df["soldB"] = sold_B(df, best)
df["costB"] = cost(df, df.soldB)
df["sold_gap_mwh"] = (df.fc_p - df.soldB) * 0.5     # forward energy given up by B


def summary(d, label):
    gen = d.gen_mwh.sum()
    days = d.settlement_date.nunique()
    ann = 365 / days
    A, B = d.costA.sum(), d.costB.sum()
    return {"period": label, "days": days, "gen_MWh": gen,
            "A_cost_GBP": A, "B_cost_GBP": B, "A_annualised_GBP": A * ann, "B_annualised_GBP": B * ann,
            "A_GBP/MWh": A / gen, "B_GBP/MWh": B / gen, "saving_%": (1 - B / A) * 100,
            "fwd_given_up_%gen": d.sold_gap_mwh.sum() / gen * 100,
            "B_long_share_%": (d.act_p > d.soldB).mean() * 100}


rows = [summary(df[df.year == y], str(y) + (" (in-sample)" if y < 2026 else " (test, Jan-Sep)"))
        for y in (2024, 2025, 2026)]
rows.append(summary(df[df.split == "test"], "2026 test period"))
out = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("\n", out.round(2).T.to_string(header=False))
out.round(3).to_csv("data/clean/exhibit3_costs.csv", index=False)

# day-block bootstrap of the 2026 saving
te = df[df.split == "test"]
d = te.groupby("settlement_date")[["costA", "costB"]].sum().to_numpy()
rng = np.random.default_rng(0)
s = [(1 - d[i, 1].sum() / d[i, 0].sum()) * 100 for i in (rng.integers(0, len(d), len(d)) for _ in range(1000))]
print(f"\n2026 saving, day-block bootstrap 95% CI: [{np.percentile(s, 2.5):.1f}%, {np.percentile(s, 97.5):.1f}%]")

# what B gives up when it sells less: forward sales at MID forgone, and where the cost comes from
long_ = te.act_p > te.soldB
print("2026 test, Strategy B: share of periods long %.1f%%; mean (MID - sys) when long %.2f, when short %.2f"
      % (long_.mean() * 100, (te.mid_price - te.system_price)[long_].mean(),
         (te.mid_price - te.system_price)[~long_].mean()))
print("2026 test mean (MID - system_price): %.2f GBP/MWh" % (te.mid_price - te.system_price).mean())
for lab, c in (("A", "costA"), ("B", "costB")):
    print(f"  {lab}: cost from short periods {te.loc[te.act_p < te['fc_p' if lab=='A' else 'soldB'], c].sum():.0f}, "
          f"from long periods {te.loc[te.act_p >= te['fc_p' if lab=='A' else 'soldB'], c].sum():.0f}")
