"""
Step 2: download the hourly price history ("candlesticks") for every contract.

Run:     python src/step2_fetch_prices.py
Output:  data/raw/candles/<ticker>.json   one file per contract, raw API response

Why only 2023 onwards:
  Step 1 shows that in 2021-2022 the number of contracts per day varied and many were
  one-sided ("above X°"). From 2023 every day has six range contracts, so days are comparable.
  Tickers use two prefixes, HIGHNY (older) and KXHIGHNY (newer); both are included.

What a candlestick is:
  A summary of one period (here, one hour): open, high, low and close traded price, plus the
  best bid and ask at the end of the hour and the volume. Later steps take the quote at a
  fixed moment and compare it with the forecast available at that moment.

The script can be stopped and restarted: contracts already downloaded are skipped.
About 8,200 contracts; a full run takes a few hours.
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
OUT = Path("data/raw/candles")
OUT.mkdir(parents=True, exist_ok=True)


def to_ts(iso):
    """Convert '2026-08-01T14:00:00Z' to a Unix timestamp (seconds), as the API expects."""
    return int(pd.Timestamp(iso).timestamp())


def get(path, params):
    """As in step 1: back off when rate-limited. Return None if not found (404)."""
    for attempt in range(5):
        r = requests.get(BASE + path, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Rate-limited repeatedly: {path}")


def fetch_candles(m):
    params = {
        "start_ts": to_ts(m["open_time"]),
        "end_ts": to_ts(m["close_time"]),
        "period_interval": 60,  # minutes: one candle per hour
    }
    # Recent contracts are on the live endpoint, older ones on /historical: try live first
    series = m["ticker"].split("-")[0]  # HIGHNY or KXHIGHNY
    data = get(f"/series/{series}/markets/{m['ticker']}/candlesticks", params)
    if not data or not data.get("candlesticks"):
        data = get(f"/historical/markets/{m['ticker']}/candlesticks", params)
    return data


if __name__ == "__main__":
    markets = pd.read_csv("data/markets.csv")
    markets = markets[markets["close_time"] >= "2023-01-01"]
    print(f"Downloading price history for {len(markets)} contracts")

    failed = []
    for i, m in enumerate(markets.to_dict("records")):
        f = OUT / f"{m['ticker']}.json"
        if f.exists():
            continue
        try:
            data = fetch_candles(m)
            f.write_text(json.dumps(data))
        except Exception as e:
            failed.append((m["ticker"], str(e)))
        if i % 200 == 0:
            print(f"  progress {i}/{len(markets)}")
        time.sleep(0.07)  # stay under the rate limit

    print(f"Done. {len(failed)} failed")
    for t, e in failed[:10]:
        print("  ", t, e)

    # Print one example; the parsing code in step 5 is based on its fields
    sample = next(OUT.glob("*.json"))
    d = json.loads(sample.read_text())
    print("\nExample:", sample.name)
    print(json.dumps(d, indent=1)[:2000])
