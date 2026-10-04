"""
Step 1: download all settled Kalshi contracts on New York's daily high temperature
(series KXHIGHNY) together with their outcomes.

Run:     python src/step1_fetch_markets.py
Output:  data/raw/markets.json   raw API response for every settled contract
         data/markets.csv        the same, as a table; later steps read this

Terminology:
- series: a family of contracts, e.g. KXHIGHNY = Central Park daily high temperature.
- event:  one day within the series, e.g. 1 July 2025.
- market: one contract within that day, e.g. "Will the high be 84-85°F?".
  A day normally has about six range contracts that together cover every temperature.
- Kalshi moves older settled contracts to the /historical/ endpoints, so both are queried.
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXHIGHNY"
OUT_RAW = Path("data/raw")
OUT_RAW.mkdir(parents=True, exist_ok=True)


def get(path, params=None):
    """GET request with a simple back-off when rate-limited (HTTP 429)."""
    for attempt in range(5):
        r = requests.get(BASE + path, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Rate-limited repeatedly: {path}")


def fetch_all(path, params):
    """Results come in pages; follow the cursor until there are no more pages."""
    out, cursor = [], None
    while True:
        p = dict(params, limit=1000)
        if cursor:
            p["cursor"] = cursor
        data = get(path, p)
        out += data.get("markets", [])
        cursor = data.get("cursor")
        print(f"  {path}: {len(out)} fetched")
        if not cursor:
            return out
        time.sleep(0.2)


if __name__ == "__main__":
    # 1) Boundary between live and historical data
    try:
        print("Historical cutoff:", get("/historical/cutoff"))
    except Exception as e:
        print("Could not read cutoff (not needed later):", e)

    # 2) Query both endpoints, then merge and de-duplicate
    live = fetch_all("/markets", {"series_ticker": SERIES, "status": "settled"})
    try:
        hist = fetch_all("/historical/markets", {"series_ticker": SERIES})
    except Exception as e:
        print("Historical endpoint failed:", e)
        hist = []
    markets = {m["ticker"]: m for m in hist + live}
    markets = list(markets.values())
    (OUT_RAW / "markets.json").write_text(json.dumps(markets, indent=1))

    # 3) Print one example to see which fields a contract has
    print("\nExample contract:")
    print(json.dumps(markets[0], indent=1)[:2500])

    # 4) Tabulate. Field names can change between API versions; missing ones are left blank
    keep = ["ticker", "event_ticker", "title", "subtitle", "yes_sub_title",
            "strike_type", "floor_strike", "cap_strike",
            "open_time", "close_time", "result", "volume", "status"]
    df = pd.DataFrame([{k: m.get(k) for k in keep} for m in markets])
    df.to_csv("data/markets.csv", index=False)

    print(f"\n{len(df)} contracts over {df['event_ticker'].nunique()} days")
    print("Outcomes:\n", df["result"].value_counts(dropna=False))
    print("First / last close:", df["close_time"].min(), "/", df["close_time"].max())
