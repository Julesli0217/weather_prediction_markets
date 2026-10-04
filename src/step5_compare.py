"""
Step 5: market vs. weather model -- whose probabilities are more accurate?

Run:     python src/step5_compare.py
Output:  data/prices_hourly.csv        hourly bid/ask for every contract (parsed candles)
         data/compare.csv              per contract: market, model, recalibrated model, outcome
         figures/step5_calibration.png market and model calibration curves

Three parts:
  A. Market probability: the bid and ask at 4 pm New York time on the previous day,
     averaged to a mid-price.
     - Wide spreads (ask - bid) mean little trading at that moment, so those quotes are dropped.
     - The six mids on a day usually sum to slightly more than 1 (the market maker's margin),
       so they are divided by their sum.
  B. Recalibrate the model with logistic regression (step 4 showed it is overconfident).
     - Fitted only on earlier months and applied to the current month, so nothing from the
       future is used.
  C. Compare Brier scores and calibration curves on the same set of contracts.
"""

import glob
import json
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SNAP_HOUR = 16       # 4 pm on the previous day
MAX_SPREAD = 0.10    # ignore quotes with a spread above 10 cents


# ---------- A. Parse candles, take snapshot quotes ----------
def pick(d, *keys):
    """The live and historical endpoints name fields differently (close_dollars / close)."""
    for k in keys:
        if d and d.get(k) is not None:
            return float(d[k])
    return np.nan


def load_prices():
    rows = []
    for f in glob.glob("data/raw/candles/*.json"):
        d = json.load(open(f))
        if not d:
            continue
        ticker = d.get("ticker") or f.split("/")[-1][:-5]
        for c in d.get("candlesticks") or []:
            rows.append({
                "ticker": ticker,
                "ts": c["end_period_ts"],
                "bid": pick(c.get("yes_bid"), "close_dollars", "close"),
                "ask": pick(c.get("yes_ask"), "close_dollars", "close"),
                "volume": pick(c, "volume_fp", "volume"),
            })
    p = pd.DataFrame(rows).sort_values(["ticker", "ts"])
    p.to_csv("data/prices_hourly.csv", index=False)
    return p


def snapshot(prices, markets):
    """For each contract, the bid/ask from the last candle at or before the snapshot time."""
    snap = (markets["date"] - pd.Timedelta(days=1) + pd.Timedelta(hours=SNAP_HOUR))
    markets = markets.assign(snap_ts=snap.dt.tz_localize("America/New_York")
                             .dt.tz_convert("UTC").astype("int64") // 10**9)
    out = pd.merge_asof(markets.sort_values("snap_ts"), prices.sort_values("ts"),
                        left_on="snap_ts", right_on="ts", by="ticker", direction="backward")
    out["spread"] = out["ask"] - out["bid"]
    out["mid"] = (out["bid"] + out["ask"]) / 2
    return out


# ---------- B. Logistic-regression recalibration ----------
def logit(p):
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1 - p))


def fit_logistic(x, y, iters=25):
    """
    Fit P(y=1) = 1 / (1 + exp(-(a + b*x))) with x = logit(model probability),
    by maximum likelihood using Newton's method. With two parameters a few lines suffice.
    Intuition: b < 1 means the model is overconfident and its probabilities should be pulled
    towards 50%; a shifts them up or down overall.
    """
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (y - p)
        hess = X.T @ (X * (p * (1 - p))[:, None])
        w += np.linalg.solve(hess, grad)
    return w


def recalibrate(df):
    """Expanding window by month: fit on all earlier months, apply to the current month."""
    df = df.sort_values("date").copy()
    df["p_recal"] = np.nan
    months = df["date"].dt.to_period("M")
    for mth in months.unique():
        train = df[months < mth]
        if len(train) < 1500:          # skip early months with too little history
            continue
        a, b = fit_logistic(logit(train["p_model"].values), train["y"].values)
        idx = months == mth
        df.loc[idx, "p_recal"] = 1 / (1 + np.exp(-(a + b * logit(df.loc[idx, "p_model"].values))))
    # after recalibration the six ranges no longer sum to 1, so renormalise
    df["p_recal"] = df["p_recal"] / df.groupby("date")["p_recal"].transform("sum")
    print(f"Latest fit: a = {a:.2f}, b = {b:.2f}")
    return df


def brier(p, y):
    return float(np.mean((p - y) ** 2))


if __name__ == "__main__":
    model = pd.read_csv("data/model_probs.csv", parse_dates=["date"])
    model = recalibrate(model)

    prices = load_prices()
    print(f"Contracts with price data: {prices['ticker'].nunique()}")

    df = snapshot(prices, model)
    n0 = len(df)
    df = df.dropna(subset=["mid"])
    # keep only days where all six ranges have reliable quotes, so they can be normalised
    ok = df.groupby("date")["spread"].transform(lambda s: (s <= MAX_SPREAD).all() and len(s) == 6)
    df = df[ok.astype(bool)].copy()
    df["p_market"] = df["mid"] / df.groupby("date")["mid"].transform("sum")
    df = df.dropna(subset=["p_recal"])
    print(f"Contracts compared: {len(df)} over {df['date'].nunique()} days (of {n0})")

    print("\nBrier score (lower is better), same contracts:")
    for col, lab in [("p_market", "Market"), ("p_recal", "Model (recalibrated)"),
                     ("p_model", "Model (raw)"), ("p_clim", "Climatology")]:
        print(f"  {lab:22s} {brier(df[col], df['y']):.4f}")

    df[["ticker", "date", "lo", "hi", "y", "actual", "bid", "ask", "spread",
        "p_market", "p_model", "p_recal", "p_clim"]].to_csv("data/compare.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 5))
    bins = np.linspace(0, 1, 11)
    for col, lab in [("p_market", "Market (D−1, 4pm)"), ("p_recal", "Forecast model, recalibrated"),
                     ("p_model", "Forecast model, raw")]:
        g = df.groupby(pd.cut(df[col], bins, include_lowest=True), observed=True)
        ax.plot(g[col].mean(), g["y"].mean(), "o-", label=lab)
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title(f"Market vs model ({df['date'].nunique()} days)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig("figures/step5_calibration.png", dpi=150)
    print("Figure saved to figures/step5_calibration.png")
