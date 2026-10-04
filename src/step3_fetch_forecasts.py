"""
Step 3: download archived weather forecasts *as they were issued* (hourly temperature,
Central Park, New York).

Run:     python src/step3_fetch_forecasts.py
Output:  data/forecasts_<model>.csv

Source: Open-Meteo Previous Runs API (free, no key).
  For every past hour it gives not only the final value but also what the forecast
  issued 1 day, 2 days, ... earlier predicted for that hour. This is what makes it possible
  to avoid look-ahead bias.

Columns (for the hour 2025-07-01 15:00, say):
  temperature_2m               shortest-lead forecast, close to the observed value (reference only)
  temperature_2m_previous_day1 forecast issued about 24 h earlier for this hour
  temperature_2m_previous_day2 forecast issued about 48 h earlier for this hour

Why previous_day2 and not just day1 (a concrete case of look-ahead bias):
  Market prices are taken at 4 pm on the previous day. The daily high can occur at any hour.
  For 11 pm on the target day, the 24-hour-ahead forecast was issued at 11 pm the previous
  day, after 4 pm, so it was not yet available. Using it would be cheating.
  Every 48-hour-ahead forecast was issued before 4 pm the previous day, so it is safe.
  Both are downloaded: if day1 looks much better, that is look-ahead bias at work.

Two models:
  GFS (US): available from 2021, covers the full sample
  ECMWF IFS (Europe): usually more accurate, but only available here from 2024
"""

import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LAT, LON = 40.7790, -73.9692          # Central Park station, which Kalshi settles on
MODELS = {"gfs": "gfs_seamless", "ecmwf": "ecmwf_ifs025"}
START, END = "2023-01-01", "2026-10-01"
Path("data").mkdir(exist_ok=True)


def fetch_year(model, start, end):
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m,temperature_2m_previous_day1,temperature_2m_previous_day2",
        "models": model,
        "start_date": start, "end_date": end,
        "temperature_unit": "fahrenheit",   # Kalshi uses Fahrenheit
        "timezone": "America/New_York",     # local time, to align with contract dates
    }
    r = requests.get(URL, params=params, timeout=60)
    if r.status_code != 200:
        print("  error:", r.status_code, r.text[:300])
        return None
    return pd.DataFrame(r.json()["hourly"])


if __name__ == "__main__":
    for name, model in MODELS.items():
        print(f"Downloading {name} ({model})...")
        parts = []
        # Very long requests are rejected, so download one year at a time
        for y in range(int(START[:4]), int(END[:4]) + 1):
            s, e = max(START, f"{y}-01-01"), min(END, f"{y}-12-31")
            df = fetch_year(model, s, e)
            if df is not None:
                parts.append(df)
                print(f"  {y}: {len(df)} hours, {df.iloc[:, -1].notna().sum()} non-missing day2 forecasts")
            time.sleep(1)
        if parts:
            out = pd.concat(parts, ignore_index=True)
            out.to_csv(f"data/forecasts_{name}.csv", index=False)
            print(f"  saved data/forecasts_{name}.csv; first rows:")
            print(out.dropna().head(3).to_string())
