"""
Step 6: does the model contain information the market has not priced in? If so, can it
be traded profitably after fees?

Run:     python src/step6_edge.py
Input:   data/compare.csv (output of step 5)
Output:  figures/step6_pnl.png   cumulative P&L of each strategy

Three parts:

A. Incremental-information regression (full sample, descriptive)
   y = 1 / (1 + exp(-(a + b1*logit(market) + b2*logit(model))))
   - If the market already contains everything in the model, b2 should be about 0.
   - Standard errors are clustered by day: exactly one of a day's six contracts resolves
     "yes", so they are not independent. Treating them as independent would inflate t-stats.

B. Out-of-sample combination (the actual test)
   Each month, fit the regression in A on earlier data only and predict the current month,
   giving a "combined" probability. Only if it beats the market's Brier score is the
   model's information useful in practice.

C. Backtest after fees (one contract per trade)
   Kalshi taker fee ~ 0.07 x P x (1 - P) per contract (rounded up to the cent per order,
   so single-contract orders cost more; a large-order approximation is used here).
   Trades are assumed to fill at the quoted ask (buy) or bid (sell) at the snapshot.
   Strategy 1  sell longshots: for contracts with ask <= 5 cents, buy NO (i.e. sell YES at the bid)
   Strategy 2  follow the model: buy YES if the combined probability exceeds the ask by more
               than THRESH; buy NO if it is below the bid by more than THRESH
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESH = 0.05
FEE = 0.07


def logit(p):
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1 - p))


def fit_logistic(X, y, iters=30):
    """Multivariate Newton's method, as in step 5. Returns coefficients and the Hessian (for SEs)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None])
        w += np.linalg.solve(H, X.T @ (y - p))
    p = 1 / (1 + np.exp(-X @ w))
    H = X.T @ (X * (p * (1 - p))[:, None])
    return w, H, p


def clustered_se(X, y, p, H, groups):
    """
    Day-clustered sandwich standard errors: Var = H^-1 (sum_d g_d g_d^T) H^-1,
    where g_d is the sum of the score (gradient) contributions of all contracts on day d.
    """
    scores = X * (y - p)[:, None]
    G = pd.DataFrame(scores).groupby(groups.values).sum().values
    Hinv = np.linalg.inv(H)
    V = Hinv @ (G.T @ G) @ Hinv
    return np.sqrt(np.diag(V))


def design(df):
    return np.column_stack([np.ones(len(df)), logit(df["p_market"]), logit(df["p_model"])])


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def fee(price):
    return FEE * price * (1 - price)


if __name__ == "__main__":
    df = pd.read_csv("data/compare.csv", parse_dates=["date"]).sort_values("date")
    y = df["y"].values.astype(float)

    # ---------- A ----------
    X = design(df)
    w, H, p = fit_logistic(X, y)
    se = clustered_se(X, y, p, H, df["date"])
    print("A. Incremental-information regression (full sample, day-clustered SEs)")
    for name, b, s in zip(["const a", "market b1", "model b2"], w, se):
        print(f"   {name:10s} {b:+.3f}   (se {s:.3f}, t = {b / s:+.1f})")

    # ---------- B ----------
    df["p_comb"] = np.nan
    months = df["date"].dt.to_period("M")
    for mth in months.unique():
        train = df[months < mth]
        if train["date"].nunique() < 120:      # require at least 120 days of history
            continue
        wm, _, _ = fit_logistic(design(train), train["y"].values.astype(float))
        idx = (months == mth).values
        df.loc[idx, "p_comb"] = 1 / (1 + np.exp(-design(df[idx]) @ wm))
    df["p_comb"] = df["p_comb"] / df.groupby("date")["p_comb"].transform("sum")
    oos = df.dropna(subset=["p_comb"])
    print(f"\nB. Out-of-sample Brier score ({oos['date'].nunique()} days):")
    for col, lab in [("p_market", "Market"), ("p_comb", "Market + model"), ("p_recal", "Model (recalibrated)")]:
        print(f"   {lab:22s} {brier(oos[col], oos['y']):.4f}")

    # ---------- C ----------
    t = oos.copy()
    # Strategy 1: sell longshots (price of NO = 1 - bid)
    s1 = t["ask"] <= 0.05
    pnl1 = np.where(s1, t["bid"] - t["y"] - fee(1 - t["bid"]), 0.0)
    # Strategy 2: follow the combined model
    buy_yes = t["p_comb"] - t["ask"] > THRESH
    buy_no = t["bid"] - t["p_comb"] > THRESH
    pnl2 = np.where(buy_yes, t["y"] - t["ask"] - fee(t["ask"]), 0.0) \
         + np.where(buy_no, t["bid"] - t["y"] - fee(1 - t["bid"]), 0.0)
    t["pnl1"], t["pnl2"] = pnl1, pnl2

    print("\nC. Backtest (one contract per trade, USD)")
    for col, mask, lab in [("pnl1", s1, "Sell longshots"), ("pnl2", buy_yes | buy_no, "Follow model")]:
        n = int(mask.sum())
        daily = t.groupby("date")[col].sum()
        tstat = daily.mean() / daily.std() * np.sqrt(len(daily)) if daily.std() > 0 else np.nan
        print(f"   {lab}: {n} trades, total P&L {t[col].sum():+.2f}, per trade {t[col].sum() / max(n, 1):+.4f}, "
              f"t = {tstat:+.2f}")
        print("     by year:", t.groupby(t["date"].dt.year)[col].sum().round(2).to_dict())

    fig, ax = plt.subplots(figsize=(7, 4))
    for col, lab in [("pnl1", "Sell longshots (ask ≤ 5¢)"), ("pnl2", f"Follow model (edge > {THRESH:.0%})")]:
        ax.plot(t.groupby("date")[col].sum().cumsum(), label=lab)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("Cumulative P&L ($, 1 contract per trade)")
    ax.set_title("Backtest after fees, out-of-sample")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("figures/step6_pnl.png", dpi=150)
    print("\nFigure saved to figures/step6_pnl.png")
