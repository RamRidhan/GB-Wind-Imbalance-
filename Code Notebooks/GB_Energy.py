"""Pull GB energy data from Elexon Insights (no API key) as monthly Parquet files.

Datasets (one file per month each, in data/raw/):
  fuelhh_{YYYY-MM}.parquet  actual wind generation (FUELHH, fuelType=WIND), half-hourly MW
  sysprice_{YYYY-MM}.parquet system prices (one call per day); systemSellPrice == systemBuyPrice
                             under single pricing
  mid_{YYYY-MM}.parquet     Market Index Price, provider APXMIDP only (N2EXMIDP is mostly
                            zero volume / zero price)
  demand_{YYYY-MM}.parquet  initial demand outturn (INDO) via demand/outturn
  agws_{YYYY-MM}.parquet    TOTAL actual wind (AGWS: onshore + offshore, incl. estimated embedded wind),
                            latest revision. This is the right "actual" for forecast-error work:
                            FUELHH WIND is BM-metered only and runs ~20% below total wind.
  windfor_{YYYY-MM}.parquet day-ahead wind forecast (WINDFOR, NESO-sourced), ALL vintages,
                            hourly; month = month of publishTime

Re-runnable: months whose file already exists are skipped. The in-progress current
month is never saved, so it gets pulled once it is complete.
"""
import pathlib
import time

import pandas as pd
import requests

BASE = "https://data.elexon.co.uk/bmrs/api/v1"
RAW = pathlib.Path("data/raw")
SLEEP = 0.2
MID_PROVIDER = "APXMIDP"

session = requests.Session()


def month_ranges(start="2024-01-01", end="2026-09-30"):
    return pd.period_range(start, end, freq="M")


def get_json(path, params=None, retries=5):
    """GET with exponential backoff on network errors, 429 and 5xx."""
    for attempt in range(retries):
        try:
            r = session.get(f"{BASE}/{path}", params=params, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} {r.text[:200]}")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def rows(payload):
    """Stream endpoints return a list; others wrap it in {'data': [...]}."""
    return payload["data"] if isinstance(payload, dict) else payload


def windows(period, step_days, pad_days=0):
    """Split a month into (from, to) date windows of step_days (each optionally padded)."""
    first, last = period.start_time.normalize(), period.end_time.normalize()
    d = first
    while d <= last:
        end = min(d + pd.Timedelta(days=step_days - 1), last)
        yield d - pd.Timedelta(days=pad_days), end + pd.Timedelta(days=pad_days)
        d = end + pd.Timedelta(days=1)


def run(name, period, fetch):
    """Skip if saved; otherwise fetch the month and write Parquet."""
    p = RAW / f"{name}_{period}.parquet"
    if p.exists():
        return
    if period.end_time.normalize() >= pd.Timestamp.now().normalize():
        print(f"{name} {period}: month not complete, skipping")
        return
    df = fetch(period)
    df = df[df["settlementDate"].between(
        period.start_time.date().isoformat(), period.end_time.date().isoformat())]
    df = df.drop_duplicates(["settlementDate", "settlementPeriod"]).sort_values(
        ["settlementDate", "settlementPeriod"])
    df.to_parquet(p, index=False)
    print(f"{name} {period}: {len(df)} rows")


def fetch_fuelhh(period):
    data = get_json("datasets/FUELHH/stream", {
        "settlementDateFrom": period.start_time.date().isoformat(),
        "settlementDateTo": period.end_time.date().isoformat(),
        "fuelType": "WIND", "format": "json"})
    return pd.DataFrame(rows(data))


def fetch_sysprice(period):
    out = []
    for day in pd.date_range(period.start_time, period.end_time.normalize()):
        out += rows(get_json(f"balancing/settlement/system-prices/{day.date().isoformat()}"))
        time.sleep(SLEEP)
    return pd.DataFrame(out)


def fetch_mid(period, provider=MID_PROVIDER):
    # API limits ranges to 7 days (filtered on UTC startTime); pad each window by a day
    # so BST settlement days are complete, then run() trims to the month and dedupes.
    out = []
    for a, b in windows(period, step_days=4, pad_days=1):
        data = get_json("balancing/pricing/market-index", {
            "from": a.strftime("%Y-%m-%dT00:00Z"),
            "to": (b + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00Z"),
            "dataProviders": provider, "format": "json"})
        out += [r for r in rows(data) if r["dataProvider"] == provider]
        time.sleep(SLEEP)
    return pd.DataFrame(out)


def fetch_agws(period):
    out = []
    for a, b in windows(period, step_days=7):
        data = get_json("generation/actual/per-type/wind-and-solar", {
            "from": a.date().isoformat(), "to": (b + pd.Timedelta(days=1)).date().isoformat(),
            "format": "json"})
        out += [r for r in rows(data) if r["psrType"].startswith("Wind")]
        time.sleep(SLEEP)
    df = pd.DataFrame(out).sort_values("publishTime").drop_duplicates(
        ["settlementDate", "settlementPeriod", "psrType"], keep="last")
    g = df.groupby(["settlementDate", "settlementPeriod"]).agg(
        startTime=("startTime", "first"), agws_mw=("quantity", "sum"), n_psr=("psrType", "size"))
    return g.reset_index()


def fetch_demand(period):
    # API limits ranges to 28 days
    out = []
    for a, b in windows(period, step_days=14):
        data = get_json("demand/outturn", {
            "settlementDateFrom": a.date().isoformat(),
            "settlementDateTo": b.date().isoformat(), "format": "json"})
        out += rows(data)
        time.sleep(SLEEP)
    return pd.DataFrame(out)


def fetch_windfor(period):
    out = []
    for a, b in windows(period, step_days=3):
        data = get_json("datasets/WINDFOR/stream", {
            "publishDateTimeFrom": a.strftime("%Y-%m-%dT00:00Z"),
            "publishDateTimeTo": (b + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00Z"),
            "format": "json"})
        out += rows(data)
        time.sleep(SLEEP)
    df = pd.DataFrame(out)
    # keep only vintages published inside this month so files do not overlap
    pt = pd.to_datetime(df["publishTime"], utc=True)
    return df[(pt >= period.start_time.tz_localize("UTC"))
              & (pt < (period.end_time.normalize() + pd.Timedelta(days=1)).tz_localize("UTC"))]


def run_windfor(period):
    p = RAW / f"windfor_{period}.parquet"
    if p.exists():
        return
    if period.end_time.normalize() >= pd.Timestamp.now().normalize():
        print(f"windfor {period}: month not complete, skipping")
        return
    df = fetch_windfor(period).drop_duplicates(["publishTime", "startTime"])
    df.sort_values(["publishTime", "startTime"]).to_parquet(p, index=False)
    print(f"windfor {period}: {len(df)} rows")


DATASETS = {
    "fuelhh": fetch_fuelhh,
    "sysprice": fetch_sysprice,
    "mid": fetch_mid,
    "agws": fetch_agws,
    "mid_n2ex": lambda p: fetch_mid(p, "N2EXMIDP"),  # robustness only
    "demand": fetch_demand,
}

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for m in month_ranges():
        for name, fetch in DATASETS.items():
            run(name, m, fetch)
        run_windfor(m)
