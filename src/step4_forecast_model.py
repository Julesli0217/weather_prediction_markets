"""
第 4 步：把天气预报变成“每个温度区间的概率”，并检验这个概率准不准。
（这一步还不用市场价格，所以不用等第 2 步下载完。）

运行：  python src/step4_forecast_model.py
输出：  data/model_probs.csv          每个合约：模型概率、气候基线概率、实际结果
        figures/step4_errors.png      预报误差分布
        figures/step4_calibration.png 模型的校准曲线

思路分四块：
  A. 每天的实际最高温：直接用 Kalshi 合约里的结算温度（expiration_value）
  B. 每天的预报最高温：把逐小时预报在“这一天”里取最大值
  C. 误差模型：实际 − 预报 的分布，只用【过去】的日子来估计（避免前视偏差）
  D. 把“预报 + 误差分布”变成每个区间的概率，再和最笨的基线（气候统计）比
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Path("figures").mkdir(exist_ok=True)
WINDOW = 30   # 误差模型用过去多少天来估计
# 为什么是 30：GFS 的偏差随季节变化很大（夏天预报偏高约 3.5°F，冬天偏低约 2.5°F），
# 窗口太长会跟不上季节。试过 30 / 45 / 90 天，30 天最好。
# 注意：这是看了全部数据后才选的，严格来说有轻微“过拟合”。更规范的做法是只用 2023 年选参数、
# 用 2024 年以后评估 —— 之后写报告时要这么做。


# ---------- A. 实际最高温 ----------
def load_markets():
    m = pd.read_csv("data/markets.csv")
    raw = pd.read_json("data/raw/markets.json")[["ticker", "expiration_value"]]
    m = m.merge(raw, on="ticker")
    m = m[m["close_time"] >= "2023-01-01"].copy()
    # 事件代码 KXHIGHNY-26AUG02 → 日期 2026-08-02
    m["date"] = pd.to_datetime(m["event_ticker"].str.split("-").str[1], format="%y%b%d")
    m["actual"] = pd.to_numeric(m["expiration_value"], errors="coerce")
    m["y"] = (m["result"] == "yes").astype(int)
    return m


# ---------- B. 预报最高温 ----------
def daily_forecast_max(path):
    """
    一个容易忽略的细节：美国气象局（NWS）的“一天”按当地【标准时间】算，
    夏令时期间实际上是凌晨 1 点到第二天凌晨 1 点。Kalshi 按 NWS 的数据结算，
    所以我们也要这么切分一天，否则夏天会错位一小时。
    """
    f = pd.read_csv(path)
    t = pd.to_datetime(f["time"]).dt.tz_localize("America/New_York",
                                                  ambiguous="NaT", nonexistent="NaT")
    f["date"] = t.dt.tz_convert("Etc/GMT+5").dt.tz_localize(None).dt.normalize()  # 固定 UTC−5 = 纽约标准时间
    cols = {"temperature_2m_previous_day1": "fc_d1", "temperature_2m_previous_day2": "fc_d2"}
    g = f.rename(columns=cols).groupby("date")[list(cols.values())]
    out = g.max()
    out[g.count().min(axis=1) < 22] = np.nan   # 一天缺太多小时就不要了
    return out


# ---------- C + D. 误差模型 → 区间概率 ----------
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bucket_bounds(row):
    """
    把合约变成连续温度上的区间。NWS 报告的是整数温度，所以：
      “82–83°”  → 实际值在 [81.5, 83.5)
      “>87°”    → 即 88° 及以上 → [87.5, ∞)
      “<80°”    → 即 79° 及以下 → (−∞, 79.5)
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

    # 误差 = 实际 − 预报
    for col in [c for c in days.columns if c.startswith("fc_")]:
        days["err" + col[2:]] = days["actual"] - days[col]

    print("预报误差（实际 − 预报，°F）：")
    print(days.filter(like="err").describe().round(2).to_string())

    # 误差模型：每一天只用它【之前】WINDOW 天的误差估计均值和标准差（shift(1) 保证不包含当天）
    main = "d2_gfs"   # 主分析：GFS、提前 48 小时（覆盖 2023 年至今、无前视偏差）
    e = days[f"err_{main}"]
    days["bias"] = e.shift(1).rolling(WINDOW, min_periods=20).mean()
    days["sd"] = e.shift(1).rolling(WINDOW, min_periods=20).std()
    days["mu"] = days[f"fc_{main}"] + days["bias"]

    # 气候基线：只用过去的、同月份的实际温度（同样不偷看未来）
    days["month"] = days.index.month

    def clim_samples(d):
        past = days[(days.index < d) & (days["month"] == d.month)]["actual"].dropna()
        return past.values

    # 每个合约的概率
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

    print(f"\n可评估的合约：{len(m)} 个，{m['date'].nunique()} 天")
    print(f"Brier score（越小越好）：模型 {brier(m.p_model, m.y):.4f}   气候基线 {brier(m.p_clim, m.y):.4f}")
    m[["ticker", "date", "strike_type", "lo", "hi", "p_model", "p_clim", "y", "actual"]].to_csv(
        "data/model_probs.csv", index=False)

    # 图 1：误差分布
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

    # 图 2：校准曲线（把概率分成 10 组，看每组的实际发生频率）
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
    print("图已保存到 figures/")
