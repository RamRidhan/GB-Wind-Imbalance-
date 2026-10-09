"""Clean and align Elexon raw files into data/clean/panel.parquet (key: settlement_date, settlement_period).

SIGN CONVENTION (defined once, used everywhere):
  error_mw     = actual_mw - forecast_mw        (negative = wind came in BELOW forecast)
  shortfall_gw = -error_mw / 1000               (positive = wind below forecast)
  spread       = system_price - mid_price       (positive = imbalance price above day-ahead-ish index)

Track 1 (main, panel.parquet): Elexon WINDFOR, leak-safe vintage, Jan 2024-Sep 2026;
  train <= 2025-12-31, test 2026.
Track 2 (cross-check, panel_neso.parquet): NESO CSV, rows stamped <= 09:00 D-1, through
  2025-12-10; train <= 2025-06-30, test after.
Actual (both tracks): Elexon AGWS total wind (incl. embedded), screened for faulty days. FUELHH WIND is
  BM-metered only (~20% low) and is kept as fuelhh_mw for reference.
Capacity (both tracks): NESO Capacity column, forward-filled where missing.
"""
import pathlib

import pandas as pd

RAW = pathlib.Path("data/raw")
OUT = pathlib.Path("data/clean")
LONDON = "Europe/London"
NESO_CSV = "NESO_DayWindForecast.csv"
START, END = "2024-01-01", "2026-09-30"
AGWS_MIN_RATIO = 0.8
NESO_END = "2025-12-10"  # last date with compliant NESO publish stamps (Track 2)
TRAIN_END = {"main": "2025-12-31", "neso": "2025-06-30"}  # train <= this date, test after
CLOCK_CHANGES = {  # date -> expected periods
    "2024-03-31": 46, "2024-10-27": 50, "2025-03-30": 46, "2025-10-26": 50, "2026-03-29": 46,
}


def to_sp(ts_utc):
    ts = pd.to_datetime(ts_utc, utc=True)
    local = ts.dt.tz_convert(LONDON)
    date = local.dt.normalize()
    midnight_utc = date.dt.tz_convert("UTC")  # local midnight, expressed in UTC
    sp = ((ts - midnight_utc).dt.total_seconds() // 1800 + 1).astype(int)
    return date.dt.date, sp


def load(name):
    return pd.concat([pd.read_parquet(p) for p in sorted(RAW.glob(f"{name}_2[0-9][0-9][0-9]-*.parquet"))],
                     ignore_index=True)


def key(df):
    df = df.copy()
    df["settlement_date"] = pd.to_datetime(df["settlementDate"]).dt.date
    df["settlement_period"] = df["settlementPeriod"].astype(int)
    return df


def check_periods(df, label):
    n = df.groupby("settlement_date").size()
    exp = pd.Series({pd.Timestamp(d).date(): v for d, v in CLOCK_CHANGES.items()})
    for d, v in exp.items():
        if d in n.index:
            assert n[d] == v, f"{label}: {d} has {n[d]} periods, expected {v}"
    normal = n[~n.index.isin(exp.index)]
    bad = normal[normal != 48]
    if len(bad):
        print(f"  {label}: {len(bad)} non-clock-change days without 48 periods, e.g. {bad.head(3).to_dict()}")
    assert not df.duplicated(["settlement_date", "settlement_period"]).any(), f"{label}: duplicate keys"


WINDFOR_OFFSET_MIN = -30  # hourly value treated as centred at startTime+offset; picked on TRAIN RMSE


def build_windfor(grid):
    """Track 1: Elexon WINDFOR hourly forecast, linearly interpolated to half-hour midpoints.

    `grid` has one row per half-hour (startTime UTC, settlement_date, settlement_period).
    Each half-hour interpolates between the two neighbouring hourly knots, and each knot uses the
    latest vintage published <= 09:00 London on D-1 where D is the HALF-HOUR's settlement date
    (so no post-cutoff vintage can enter, even for a knot that falls in the next settlement day).
    Alignment: the hourly value sits at startTime + WINDFOR_OFFSET_MIN. Versus copying the hour to
    both half-hours this cut train RMSE ~2.4% (offsets -30..-45 tied; -30 used).
    """
    f = load("windfor")
    f["publish"] = pd.to_datetime(f["publishTime"], utc=True)
    f["knot"] = pd.to_datetime(f["startTime"], utc=True)
    n_vintages = f["publish"].nunique()
    f = f[["publish", "knot", "generation"]].assign(pub_time=f["publish"]).sort_values("publish")

    hh = grid[["startTime", "settlement_date", "settlement_period"]].copy()
    hh["start"] = pd.to_datetime(hh["startTime"], utc=True)
    hh["cutoff"] = (pd.to_datetime(hh["settlement_date"]).dt.tz_localize(LONDON)
                    - pd.Timedelta(days=1) + pd.Timedelta(hours=9)).dt.tz_convert("UTC")
    x = hh["start"] + pd.Timedelta(minutes=15) - pd.Timedelta(minutes=WINDFOR_OFFSET_MIN)
    t0 = x.dt.floor("h")
    w = ((x - t0) / pd.Timedelta(hours=1)).to_numpy()
    for k in (0, 1):
        q = pd.DataFrame({"i": hh.index, "knot": t0 + pd.Timedelta(hours=k), "cutoff": hh["cutoff"]})
        q = pd.merge_asof(q.sort_values("cutoff"), f, left_on="cutoff", right_on="publish",
                          by="knot", direction="backward").set_index("i").sort_index()
        hh[f"g{k}"], hh[f"p{k}"] = q["generation"], q["pub_time"]
    hh["forecast_mw"] = (1 - w) * hh["g0"] + w * hh["g1"]
    hh["forecast_publish_time"] = hh[["p0", "p1"]].max(axis=1)
    hh = hh.dropna(subset=["forecast_mw"])
    # leakage assertions: every vintage used was published <= 09:00 London on D-1, before delivery
    assert (hh["p0"] <= hh["cutoff"]).all() and (hh["p1"] <= hh["cutoff"]).all()
    assert (hh["forecast_publish_time"] < hh["start"]).all()
    print(f"[main] WINDFOR: {n_vintages} vintages; hourly knots interpolated to half-hours "
          f"(offset {WINDFOR_OFFSET_MIN} min), vintage <= 09:00 London on D-1")
    return hh[["settlement_date", "settlement_period", "forecast_mw", "forecast_publish_time"]]


def read_neso():
    f = pd.read_csv(NESO_CSV, parse_dates=["Forecast_Timestamp"])
    f = f[(f["Date"] >= START) & (f["Date"] <= END)].copy()
    f["settlement_date"] = pd.to_datetime(f["Date"]).dt.date
    f["settlement_period"] = f["Settlement_period"].astype(int)
    return f


def build_neso():
    """Track 2: NESO Day Ahead Wind Forecast (one vintage per row).

    Forecast_Timestamp is London local clock (~08:49 year-round). Rows stamped after 09:00 on
    D-1 are dropped; from ~Dec 2025 almost all fail (same-day stamps, June 2026 backfills).
    """
    f = read_neso()
    cutoff = pd.to_datetime(f["Date"]) - pd.Timedelta(hours=15)  # D-1 09:00 local clock
    late = f["Forecast_Timestamp"] > cutoff
    print(f"[neso] {late.sum()} of {len(f)} rows stamped after D-1 09:00 dropped (leakage)")
    f = f[~late & (f["Date"] <= NESO_END)].sort_values("Forecast_Timestamp").drop_duplicates(
        ["settlement_date", "settlement_period"], keep="last")
    assert (f["Forecast_Timestamp"] <= pd.to_datetime(f["Date"]) - pd.Timedelta(hours=15)).all()
    return f.rename(columns={"Incentive_forecast": "forecast_mw", "Forecast_Timestamp": "forecast_publish_time"})[
        ["settlement_date", "settlement_period", "forecast_mw", "forecast_publish_time"]]


def neso_capacity():
    """NESO Capacity per period (slow-moving installed capacity; not a forecast, so rows
    stamped late are still usable). Latest stamp wins on duplicates."""
    f = read_neso().sort_values("Forecast_Timestamp").drop_duplicates(
        ["settlement_date", "settlement_period"], keep="last")
    return f[["settlement_date", "settlement_period", "Capacity"]].rename(columns={"Capacity": "capacity_mw"})


def assemble(track, fc, base, cap):
    k = ["settlement_date", "settlement_period"]
    srcs = {"forecast": fc, **base}
    for name, df in srcs.items():
        check_periods(df, f"{track}/{name}")
    panel = base["wind"]
    print(f"[{track}] {'wind (base)':14s} {len(panel):>7d} rows")
    for name in ("forecast", "system_price", "mid", "demand"):
        before = len(panel)
        panel = panel.merge(srcs[name], on=k, how="inner")
        total = len(srcs[name])
        print(f"[{track}] {name:14s} {total:>7d} rows; panel {before} -> {len(panel)} "
              f"(panel lost {before - len(panel)}, {name} lost {total - len(panel)})")
    panel = panel.sort_values(k).reset_index(drop=True)
    check_periods(panel, f"{track}/panel")

    # capacity: NESO Capacity where present, else forward-filled from the last known value
    panel = panel.merge(cap, on=k, how="left")
    n_neso = panel["capacity_mw"].notna().sum()
    panel["capacity_mw"] = panel["capacity_mw"].ffill().bfill()
    print(f"[{track}] capacity: NESO column for {n_neso}/{len(panel)} rows, "
          f"{len(panel) - n_neso} forward/back-filled")

    panel["forecast_mw"] = panel["forecast_mw"].astype(float)
    panel["error_mw"] = panel.actual_mw - panel.forecast_mw
    panel["error_pct"] = panel.error_mw / panel.capacity_mw * 100
    panel["shortfall_gw"] = -panel.error_mw / 1000  # positive = wind came in below forecast
    panel["spread"] = panel.system_price - panel.mid_price
    panel["split"] = (pd.to_datetime(panel.settlement_date) <= TRAIN_END[track]).map(
        {True: "train", False: "test"})
    panel["track"] = track
    panel = panel.rename(columns={"startTime": "start_time_utc"})
    print(f"[{track}] {panel.settlement_date.min()} to {panel.settlement_date.max()}; "
          f"train {(panel.split == 'train').sum()}, test {(panel.split == 'test').sum()}")
    print(panel[["error_mw", "error_pct", "shortfall_gw", "spread"]].describe().round(2))
    return panel


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # actual = AGWS total wind (incl. estimated embedded). FUELHH WIND is BM-metered only (~20% lower)
    # and made the forecast look ~14% too high; it is kept only as fuelhh_mw for reference.
    wind = key(load("agws")).rename(columns={"agws_mw": "actual_mw"})
    fuelhh = key(load("fuelhh")).rename(columns={"generation": "fuelhh_mw"})
    wind = wind.merge(fuelhh[["settlement_date", "settlement_period", "fuelhh_mw"]],
                      on=["settlement_date", "settlement_period"], how="left")
    sysp = key(load("sysprice"))
    mid = key(load("mid")).rename(columns={"price": "mid_price", "volume": "mid_volume"})
    dem = key(load("demand")).rename(columns={"initialDemandOutturn": "demand_mw"})

    # AGWS data-quality screen: Apr-Nov 2024 (and a few later days) report ~0.3-0.5x of FUELHH, a
    # source fault (normal days are 1.1-1.3x). Drop days whose AGWS/FUELHH ratio is < 0.8.
    ratio = wind.groupby("settlement_date").apply(lambda x: x.actual_mw.sum() / x.fuelhh_mw.sum())
    bad_days = ratio[ratio < AGWS_MIN_RATIO].index
    print(f"AGWS screen: dropping {len(bad_days)} days with AGWS/FUELHH < {AGWS_MIN_RATIO} "
          f"({pd.Series(pd.to_datetime(bad_days)).dt.to_period('M').value_counts().sort_index().to_dict()})")
    wind = wind[~wind.settlement_date.isin(bad_days)]

    # sanity: Elexon's settlementDate/Period agree with the NESO-style conversion
    d, sp = to_sp(wind["startTime"])
    assert (d == wind["settlement_date"]).all() and (sp == wind["settlement_period"]).all(), "to_sp mismatch"
    assert (sysp["systemSellPrice"] == sysp["systemBuyPrice"]).all(), "dual pricing present"
    sysp = sysp.rename(columns={"systemSellPrice": "system_price"})

    k = ["settlement_date", "settlement_period"]
    base = {
        "wind": wind[k + ["startTime", "actual_mw", "fuelhh_mw"]],
        "system_price": sysp[k + ["system_price"]],
        "mid": mid[k + ["mid_price", "mid_volume"]],
        "demand": dem[k + ["demand_mw"]],
    }
    cap = neso_capacity()
    main_p = assemble("main", build_windfor(wind), base, cap)
    neso_p = assemble("neso", build_neso(), base, cap)
    main_p.to_parquet(OUT / "panel.parquet", index=False)
    neso_p.to_parquet(OUT / "panel_neso.parquet", index=False)
    print(f"saved {OUT / 'panel.parquet'} {main_p.shape} and {OUT / 'panel_neso.parquet'} {neso_p.shape}")


if __name__ == "__main__":
    main()
