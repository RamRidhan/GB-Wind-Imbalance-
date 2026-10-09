"""Note-ready figures (no embedded titles; captions live in the document)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, RUST, GREY = "#1f5f99", "#b5451b", "#888888"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
OUT = "data/clean/note/"
df = pd.read_parquet("data/clean/panel.parquet")
t = pd.to_datetime(df.start_time_utc, utc=True).dt.tz_convert("Europe/London")
df["hour"] = t.dt.hour


def cmean(x, g):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    s = pd.Series(x - m).groupby(np.asarray(g)).sum(); G = len(s)
    return m, 1.96 * np.sqrt(G / (G - 1) * (s ** 2).sum()) / n


# ---- Exhibit 1
b = df.loc[df.split == "train", "forecast_mw"].quantile([0, .2, .4, .6, .8, 1]).to_numpy(); b[0], b[-1] = -np.inf, np.inf
df["q"] = pd.cut(df.forecast_mw, b, labels=False)
h = pd.DataFrame([(k, *cmean(d.error_pct, d.settlement_date)) for k, d in df.groupby("hour")], columns=["x", "m", "ci"])
q = pd.DataFrame([(k, *cmean(d.error_pct, d.settlement_date)) for k, d in df.groupby("q")], columns=["x", "m", "ci"])
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for a in ax: a.axhline(0, color=GREY, lw=.8)
ax[0].errorbar(h.x, h.m, yerr=h.ci, fmt="o-", color=BLUE, capsize=2, ms=3.5, lw=1.2)
ax[0].set(xlabel="Hour of day (London time)", ylabel="Mean forecast error, % of capacity", title="By hour of day")
ax[0].set_xticks(range(0, 24, 3))
ax[1].errorbar(q.x, q.m, yerr=q.ci, fmt="o", color=RUST, capsize=3, ms=5)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"])
ax[1].set(xlabel="Forecast quintile (bounds from 2024-25)", title="By forecast level")
fig.tight_layout(); fig.savefig(OUT + "ex1.png", dpi=220); plt.close(fig)

# ---- Exhibit 2
bk = pd.read_csv("data/clean/exhibit2_buckets_main.csv", index_col=0)
rb = pd.read_csv("data/clean/robustness.csv")
import re
def parse(s): m_, se = re.match(r"(-?[\d.]+) \(([\d.]+)\)", s).groups(); return float(m_), 1.96 * float(se)
rows = [("Full sample", *parse(rb.coef_full[0])), ("2024-25 (train)", *parse(rb.coef_train[0])),
        ("2026 (test)", *parse(rb.coef_test[0])), ("Drop clock-change days", *parse(rb.coef_full[1])),
        ("Spread winsorised", *parse(rb.coef_full[2])), ("NESO forecast\n(to Dec 2025)", -0.487, 1.96 * 0.658)]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
labs = ["Surplus\n<-1 GW", "+/-1\nGW", "1-2 GW\nshort", ">2 GW\nshort"]
se = df.assign(bk=pd.cut(df.shortfall_gw, [-np.inf, -1, 1, 2, np.inf], labels=False)).groupby("bk").spread.sem()
ax[0].bar(range(4), bk.mean_spread, color=RUST, width=.6)
ax[0].errorbar(range(4), bk.mean_spread, yerr=1.96 * se.to_numpy(), fmt="none", color="k", capsize=3, lw=1)
ax[0].axhline(0, color=GREY, lw=.8); ax[0].set_xticks(range(4)); ax[0].set_xticklabels(labs)
ax[0].set(ylabel="Mean spread, GBP/MWh (system - MID)", title="Raw spread by shortfall bucket")
y = np.arange(len(rows))[::-1]
for yi, (n, m_, ci) in zip(y, rows):
    ax[1].errorbar(m_, yi, xerr=ci, fmt="o", color=BLUE, capsize=3, ms=4.5)
ax[1].axvline(0, color=GREY, lw=.8); ax[1].set_yticks(y); ax[1].set_yticklabels([r[0] for r in rows])
ax[1].set(xlabel="GBP/MWh per GW (HAC 95% CI)", title="Adjusted slope, by specification")
fig.tight_layout(); fig.savefig(OUT + "ex2.png", dpi=220); plt.close(fig)

# ---- Exhibit 3
c = pd.read_csv("data/clean/exhibit3_costs.csv").iloc[:3]
lab = ["2024\n(141 days)", "2025", "2026\n(Jan-Sep, test)"]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.1, 1]})
x = np.arange(3); w = .36
ax[0].bar(x - w / 2, c["A_GBP/MWh"], w, color=BLUE, label="A: sell the forecast")
ax[0].bar(x + w / 2, c["B_GBP/MWh"], w, color=RUST, label="B: sell 1st-percentile volume")
ax[0].axhline(0, color=GREY, lw=.8); ax[0].set_xticks(x); ax[0].set_xticklabels(lab)
ax[0].set(ylabel="Imbalance cost, GBP per MWh generated", title="Cost by period (positive = loss)")
ax[0].set_ylim(-0.1, 0.5); ax[0].legend(frameon=False, fontsize=7.5, loc="upper right", ncol=1)
v = rb[["variant", "B_saving_%"]]
names = ["Baseline", "Drop\nclock days", "Winsorised\nspread"]
ax[1].bar(range(3), v["B_saving_%"], color=[RUST, RUST, RUST], width=.6)
for i, val in enumerate(v["B_saving_%"]): ax[1].text(i, val + (6 if val >= 0 else -14), f"{val:.0f}%", ha="center", fontsize=8)
ax[1].axhline(0, color=GREY, lw=.8); ax[1].set_xticks(range(3)); ax[1].set_xticklabels(names)
ax[1].set(ylabel="B's saving vs A, 2026 (%)", title="2026 saving is fragile"); ax[1].set_ylim(-30, 140)
fig.tight_layout(); fig.savefig(OUT + "ex3.png", dpi=220); plt.close(fig)
print("ok")

# ---- Exhibit 4: where the cost sits, and do big misses cause spikes?
df["gen"] = df.actual_mw / df.capacity_mw * 500
df["costA"] = (df.actual_mw - df.forecast_mw) / df.capacity_mw * 1000 * (df.mid_price - df.system_price) * .5
df["bk"] = pd.cut(df.shortfall_gw, [-np.inf, -1, 1, 2, np.inf], labels=False)
g = df.groupby("bk")
share_n = g.size() / len(df) * 100
share_c = g.costA.sum() / df.costA.sum() * 100
hi = g.apply(lambda d: (d.spread > 50).mean() * 100); lo = g.apply(lambda d: (d.spread < -50).mean() * 100)
labs = ["Surplus\n<-1 GW", "+/-1\nGW", "1-2 GW\nshort", ">2 GW\nshort"]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
x = np.arange(4); w = .38
ax[0].bar(x - w / 2, share_n, w, color=GREY, label="Share of half-hours")
ax[0].bar(x + w / 2, share_c, w, color=RUST, label="Share of net imbalance cost")
ax[0].set_xticks(x); ax[0].set_xticklabels(labs); ax[0].set(ylabel="%", title="Big misses cost more per period")
ax[0].legend(frameon=False, fontsize=7.5, loc="upper right"); ax[0].set_ylim(0, 62)
ax[1].bar(x - w / 2, hi, w, color=BLUE, label="Spread above +GBP 50")
ax[1].bar(x + w / 2, lo, w, color="#7fa6c9", label="Spread below -GBP 50")
ax[1].set_xticks(x); ax[1].set_xticklabels(labs); ax[1].set(ylabel="% of half-hours", title="...but do not raise upward-spike risk")
ax[1].legend(frameon=False, fontsize=7.5, loc="upper left"); ax[1].set_ylim(0, 6)
fig.tight_layout(); fig.savefig(OUT + "ex4.png", dpi=220); plt.close(fig)
print(share_n.round(1).tolist(), share_c.round(1).tolist(), hi.round(1).tolist(), lo.round(1).tolist())
