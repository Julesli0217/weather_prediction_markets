"""
第 6 步：模型有没有市场还没用上的信息？如果有，扣掉手续费后能不能赚钱？

运行：  python src/step6_edge.py
输入：  data/compare.csv（第 5 步的输出）
输出：  figures/step6_pnl.png   各策略的累计盈亏

三块内容：

A. 增量信息回归（全样本，用来“描述”）
   y = 1 / (1 + exp(−(a + b1·logit(市场概率) + b2·logit(模型概率))))
   - 如果市场已经包含了模型的全部信息，b2 应该 ≈ 0。
   - 标准误要按“天”聚类：同一天的 6 个合约里恰好一个是 yes，它们不是独立的。
     如果当作 6 个独立样本，会高估证据强度（t 值虚高）。

B. 样本外组合（用来“检验”）
   每个月只用之前的数据拟合 A 里的回归，再预测当月 → 得到“组合概率”。
   如果组合概率的 Brier score 比市场好，才说明模型的信息在实际中有用。

C. 计入手续费的回测（每笔 1 份合约）
   Kalshi 吃单手续费 ≈ 0.07 × P × (1 − P) 每份（官方按每笔订单向上取整到 1 美分，
   单笔只买 1 份会更贵；这里按大单近似）。我们假设在快照时刻以卖价买入 / 以买价卖出。
   策略 1  卖“彩票”：市场价 ≤ 5 美分的合约，买 NO（等价于以 bid 卖出 YES）
   策略 2  跟模型：组合概率比卖价高 THRESH 以上就买 YES；比买价低 THRESH 以上就买 NO
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
    """多元版本的牛顿法，和第 5 步一样。返回系数 w 和 Hessian（算标准误要用）。"""
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
    按天聚类的“三明治”标准误：Var = H⁻¹ · (Σ_天 g_d g_dᵀ) · H⁻¹，
    其中 g_d 是这一天所有合约的得分（梯度）之和。
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
    print("A. 增量信息回归（全样本，标准误按天聚类）")
    for name, b, s in zip(["常数 a", "市场 b1", "模型 b2"], w, se):
        print(f"   {name:8s} {b:+.3f}   (se {s:.3f}, t = {b / s:+.1f})")

    # ---------- B ----------
    df["p_comb"] = np.nan
    months = df["date"].dt.to_period("M")
    for mth in months.unique():
        train = df[months < mth]
        if train["date"].nunique() < 120:      # 至少用 120 天的历史来拟合
            continue
        wm, _, _ = fit_logistic(design(train), train["y"].values.astype(float))
        idx = (months == mth).values
        df.loc[idx, "p_comb"] = 1 / (1 + np.exp(-design(df[idx]) @ wm))
    df["p_comb"] = df["p_comb"] / df.groupby("date")["p_comb"].transform("sum")
    oos = df.dropna(subset=["p_comb"])
    print(f"\nB. 样本外（{oos['date'].nunique()} 天）Brier score：")
    for col, lab in [("p_market", "市场"), ("p_comb", "市场 + 模型组合"), ("p_recal", "模型（校准后）")]:
        print(f"   {lab:12s} {brier(oos[col], oos['y']):.4f}")

    # ---------- C ----------
    t = oos.copy()
    # 策略 1：卖彩票（买 NO 的价格 = 1 − bid）
    s1 = t["ask"] <= 0.05
    pnl1 = np.where(s1, t["bid"] - t["y"] - fee(1 - t["bid"]), 0.0)
    # 策略 2：跟组合模型
    buy_yes = t["p_comb"] - t["ask"] > THRESH
    buy_no = t["bid"] - t["p_comb"] > THRESH
    pnl2 = np.where(buy_yes, t["y"] - t["ask"] - fee(t["ask"]), 0.0) \
         + np.where(buy_no, t["bid"] - t["y"] - fee(1 - t["bid"]), 0.0)
    t["pnl1"], t["pnl2"] = pnl1, pnl2

    print("\nC. 回测（每笔 1 份合约，单位：美元）")
    for col, mask, lab in [("pnl1", s1, "卖彩票"), ("pnl2", buy_yes | buy_no, "跟模型")]:
        n = int(mask.sum())
        daily = t.groupby("date")[col].sum()
        tstat = daily.mean() / daily.std() * np.sqrt(len(daily)) if daily.std() > 0 else np.nan
        print(f"   {lab}：{n} 笔，总盈亏 {t[col].sum():+.2f}，每笔平均 {t[col].sum() / max(n, 1):+.4f}，"
              f"t = {tstat:+.2f}")
        print("     按年：", t.groupby(t["date"].dt.year)[col].sum().round(2).to_dict())

    fig, ax = plt.subplots(figsize=(7, 4))
    for col, lab in [("pnl1", "Sell longshots (ask ≤ 5¢)"), ("pnl2", f"Follow model (edge > {THRESH:.0%})")]:
        ax.plot(t.groupby("date")[col].sum().cumsum(), label=lab)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("Cumulative P&L ($, 1 contract per trade)")
    ax.set_title("Backtest after fees, out-of-sample")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("figures/step6_pnl.png", dpi=150)
    print("\n图已保存到 figures/step6_pnl.png")
