"""
Step 4: turn weather forecasts into a probability for each temperature range, and check
whether those probabilities are accurate. (No market prices needed yet.)

Run:     python src/step4_forecast_model.py
Output:  data/model_probs.csv          per contract: model probability, climatology, outcome
         figures/step4_errors.png      forecast error distributions
         figures/step4_calibration.png calibration curves

Four parts:
  A. Actual daily high: the settlement temperature in the Kalshi data (expiration_value)
  B. Forecast daily high: the maximum of the hourly forecasts over the day
  C. Error model: distribution of actual - forecast, estimated from *past* days only
  D. Forecast + error distribution -> probability for each range, compared with the
     simplest baseline (climatology)
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Path("figures").mkdir(exist_ok=True)
WINDOW = 30   # number of past days used to estimate the error model
# Why 30: GFS bias is strongly seasonal (about 3.5°F too warm in summer, 2.5°F too cold in winter),
# so a long window lags behind the seasons. 30, 45 and 90 days were tried; 30 worked best.
# Caveat: this was chosen after seeing all the data, so it is mildly overfitted. A cleaner
# design would choose it on 2023 data and evaluate on 2024 onwards.


# ---------- A. Actual daily high ----------
def load_markets():
    m = pd.read_csv("data/markets.csv")
    raw = pd.read_json("data/raw/markets.json")[["ticker", "expiration_value"]]
    m = m.merge(raw, on="ticker")
    m = m[m["close_time"] >= "2023-01-01"].copy()
    # event ticker KXHIGHNY-26AUG02 -> date 2026-08-02
    m["date"] = pd.to_datetime(m["event_ticker"].str.split("-").str[1], format="%y%b%d")
    m["actual"] = pd.to_numeric(m["expiration_value"], errors="coerce")
    m["y"] = (m["result"] == "yes").astype(int)
    return m


# ---------- B. Forecast daily high ----------
def daily_forecast_max(path):
    """
    An easy detail to miss: the National Weather Service defines a day in local *standard*
    time, so during daylight saving it runs from 1 am to 1 am. Kalshi settles on NWS data,
    so days are split the same way here; otherwise summer days would be off by an hour.
    """
    f = pd.read_csv(path)
    t = pd.to_datetime(f["time"]).dt.tz_localize("America/New_York",
                                                  ambiguous="NaT", nonexistent="NaT")
    f["date"] = t.dt.tz_convert("Etc/GMT+5").dt.tz_localize(None).dt.normalize()  # fixed UTC-5 = New York standard time
    cols = {"temperature_2m_previous_day1": "fc_d1", "temperature_2m_previous_day2": "fc_d2"}
    g = f.rename(columns=cols).groupby("date")[list(cols.values())]
    out = g.max()
    out[g.count().min(axis=1) < 22] = np.nan   # drop days with too many missing hours
    return out


# ---------- C + D. Error model -> range probabilities ----------
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bucket_bounds(row):
    """
    Map a contract to an interval on a continuous temperature scale. NWS reports whole
    degrees, so:
      "82-83°" -> [81.5, 83.5)
      ">87°"   -> 88° or above -> [87.5, inf)
      "<80°"   -> 79° or below -> (-inf, 79.5)
    """
    if row["strike_type"] == "between":
        return row["floor_strike"] - 0.5, row["cap_strike"] + 0.5
    if row["strike_type"] == "greater":
        return row["floor_strike"] + 0.5, np.inf
    if row["strike_type"] == "less":
        return -np.inf, row["cap_strike"] - 0.5
    return np.nan, np.nan


def prob_normal(lo, hi, mu, sd):
    a = 0.0 if lo == -np.inf else norm_cdf((lo - mu) / sd)
    b = 1.0 if hi == np.inf else norm_cdf((hi - mu) / sd)
    return b - a


def brier(p, y):
    return float(np.mean((p - y) ** 2))


if __name__ == "__main__":
    m = load_markets()
    days = m.groupby("date")["actual"].first().to_frame()

    for name in ["gfs", "ecmwf"]:
        fc = daily_forecast_max(f"data/forecasts_{name}.csv").add_suffix(f"_{name}")
        days = days.join(fc)

    # error = actual - forecast
    for col in [c for c in days.columns if c.startswith("fc_")]:
        days["err" + col[2:]] = days["actual"] - days[col]

    print("Forecast errors (actual - forecast, °F):")
    print(days.filter(like="err").describe().round(2).to_string())

    # Error model: for each day, mean and s.d. of the errors over the previous WINDOW days
    # (shift(1) makes sure the day itself is excluded)
    main = "d2_gfs"   # main analysis: GFS, 48 h ahead (covers 2023 onwards, no look-ahead)
    e = days[f"err_{main}"]
    days["bias"] = e.shift(1).rolling(WINDOW, min_periods=20).mean()
    days["sd"] = e.shift(1).rolling(WINDOW, min_periods=20).std()
    days["mu"] = days[f"fc_{main}"] + days["bias"]

    # Climatology baseline: past observed highs in the same calendar month only
    days["month"] = days.index.month

    def clim_samples(d):
        past = days[(days.index < d) & (days["month"] == d.month)]["actual"].dropna()
        return past.values

    # probability for each contract
    m = m.join(days[["mu", "sd"]], on="date")
    lo_hi = m.apply(bucket_bounds, axis=1, result_type="expand")
    m["lo"], m["hi"] = lo_hi[0], lo_hi[1]
    m = m.dropna(subset=["mu", "sd", "lo"])
    m["p_model"] = [prob_normal(r.lo, r.hi, r.mu, r.sd) for r in m.itertuples()]

    cache = {}
    def p_clim(r):
        if r.date not in cache:
            cache[r.date] = clim_samples(r.date)
        s = cache[r.date]
        if len(s) < 20:
            return np.nan
        return float(np.mean((s >= r.lo) & (s < r.hi)))
    m["p_clim"] = [p_clim(r) for r in m.itertuples()]
    m = m.dropna(subset=["p_clim"])

    print(f"\nContracts evaluated: {len(m)} over {m['date'].nunique()} days")
    print(f"Brier score (lower is better): model {brier(m.p_model, m.y):.4f}   climatology {brier(m.p_clim, m.y):.4f}")
    m[["ticker", "date", "strike_type", "lo", "hi", "p_model", "p_clim", "y", "actual"]].to_csv(
        "data/model_probs.csv", index=False)

    # Figure 1: error distributions
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, lab in [("err_d2_gfs", "GFS, 48h ahead"), ("err_d1_gfs", "GFS, 24h ahead"),
                     ("err_d2_ecmwf", "ECMWF, 48h ahead")]:
        ax.hist(days[col].dropna(), bins=np.arange(-12, 12.5, 1), histtype="step", lw=1.6, label=lab)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Actual high − forecast high (°F)")
    ax.set_ylabel("Days")
    ax.set_title("Forecast errors for NYC daily high")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("figures/step4_errors.png", dpi=150)

    # Figure 2: calibration (bin probabilities into deciles, compare with observed frequency)
    fig, ax = plt.subplots(figsize=(5, 5))
    bins = np.linspace(0, 1, 11)
    for col, lab in [("p_model", "Forecast model"), ("p_clim", "Climatology")]:
        g = m.groupby(pd.cut(m[col], bins, include_lowest=True), observed=True)
        ax.plot(g[col].mean(), g["y"].mean(), "o-", label=lab)
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (2023–2026)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("figures/step4_calibration.png", dpi=150)
    print("Figures saved to figures/")
