"""
第 5 步：市场 vs 天气模型 —— 谁的概率更准？

运行：  python src/step5_compare.py
输出：  data/prices_hourly.csv        所有合约的逐小时买价 / 卖价（从 K 线整理出来）
        data/compare.csv              每个合约：市场概率、模型概率、校准后模型概率、结果
        figures/step5_calibration.png 市场和模型的校准曲线放在一起

（第 2 步没下载完也可以先跑，只会用已经下载好的那部分合约。）

三块内容：
  A. 市场概率：取“前一天下午 4 点（纽约时间）”那一刻的买价和卖价，取中间值（mid）。
     - 价差（ask − bid）太大的，说明那一刻没什么人交易，价格不可靠，过滤掉。
     - 同一天 6 个区间的 mid 加起来往往略大于 1（做市商要赚钱），所以除以总和，归一化。
  B. 用 logistic regression 重新校准模型（第 4 步发现模型在高概率端过度自信）。
     - 只用【之前月份】的数据来拟合，再用到当月 —— 和误差模型一样，不偷看未来。
  C. 在同一批合约上比较 Brier score 和校准曲线。
"""

import glob
import json
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SNAP_HOUR = 16       # 前一天下午 4 点
MAX_SPREAD = 0.10    # 价差超过 10 美分就不用


# ---------- A. 整理 K 线，取快照价格 ----------
def pick(d, *keys):
    """新旧两个接口的字段名不一样（close_dollars / close），哪个有就用哪个。"""
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
    """对每个合约，找“快照时刻或之前最后一根 K 线”的买卖价。"""
    snap = (markets["date"] - pd.Timedelta(days=1) + pd.Timedelta(hours=SNAP_HOUR))
    markets = markets.assign(snap_ts=snap.dt.tz_localize("America/New_York")
                             .dt.tz_convert("UTC").astype("int64") // 10**9)
    out = pd.merge_asof(markets.sort_values("snap_ts"), prices.sort_values("ts"),
                        left_on="snap_ts", right_on="ts", by="ticker", direction="backward")
    out["spread"] = out["ask"] - out["bid"]
    out["mid"] = (out["bid"] + out["ask"]) / 2
    return out


# ---------- B. logistic regression 校准 ----------
def logit(p):
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1 - p))


def fit_logistic(x, y, iters=25):
    """
    拟合 P(y=1) = 1 / (1 + exp(−(a + b·x)))，x = logit(模型概率)。
    用牛顿法求最大似然 —— 只有两个参数，自己写几行就够，不需要 sklearn。
    直觉：b < 1 表示模型太自信，需要把概率往 50% 方向“压”；a 调整整体偏高或偏低。
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
    """按月滚动：每个月用之前所有月份的数据拟合，再用到这个月。"""
    df = df.sort_values("date").copy()
    df["p_recal"] = np.nan
    months = df["date"].dt.to_period("M")
    for mth in months.unique():
        train = df[months < mth]
        if len(train) < 1500:          # 数据太少的早期月份不校准
            continue
        a, b = fit_logistic(logit(train["p_model"].values), train["y"].values)
        idx = months == mth
        df.loc[idx, "p_recal"] = 1 / (1 + np.exp(-(a + b * logit(df.loc[idx, "p_model"].values))))
    # 校准后同一天 6 个区间的概率之和不再是 1，重新归一化
    df["p_recal"] = df["p_recal"] / df.groupby("date")["p_recal"].transform("sum")
    print(f"最近一次拟合：a = {a:.2f}, b = {b:.2f}")
    return df


def brier(p, y):
    return float(np.mean((p - y) ** 2))


if __name__ == "__main__":
    model = pd.read_csv("data/model_probs.csv", parse_dates=["date"])
    model = recalibrate(model)

    prices = load_prices()
    print(f"已下载价格的合约：{prices['ticker'].nunique()} 个")

    df = snapshot(prices, model)
    n0 = len(df)
    df = df.dropna(subset=["mid"])
    # 只保留 6 个区间都有可靠报价的日子，否则没法归一化
    ok = df.groupby("date")["spread"].transform(lambda s: (s <= MAX_SPREAD).all() and len(s) == 6)
    df = df[ok.astype(bool)].copy()
    df["p_market"] = df["mid"] / df.groupby("date")["mid"].transform("sum")
    df = df.dropna(subset=["p_recal"])
    print(f"可比较的合约：{len(df)} 个，{df['date'].nunique()} 天（原有 {n0} 个）")

    print("\nBrier score（越小越好），同一批合约：")
    for col, lab in [("p_market", "市场"), ("p_recal", "模型（校准后）"),
                     ("p_model", "模型（原始）"), ("p_clim", "气候基线")]:
        print(f"  {lab:10s} {brier(df[col], df['y']):.4f}")

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
    print("图已保存到 figures/step5_calibration.png")
