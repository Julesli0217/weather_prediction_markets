"""
第 1 步：下载 Kalshi 纽约每日最高气温合约（系列代码 KXHIGHNY）的列表和结算结果。

运行：  python src/step1_fetch_markets.py
输出：  data/raw/markets.json      所有已结算合约的原始数据
        data/markets.csv           整理成表格的版本，之后每一步都从这里读

几个概念：
- series（系列）：一类合约，比如 KXHIGHNY = 纽约中央公园每日最高气温。
- event（事件）：系列里的某一天，比如 2025 年 7 月 1 日。
- market（合约）：某一天里的一个具体问题，比如“最高温在 84–85°F 之间吗？”
  同一天通常有 6 个左右的区间合约，它们加起来覆盖所有可能的温度。
- Kalshi 把较早结算的合约挪到了 /historical/ 接口，所以要两边都查。
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
    """发一个 GET 请求；遇到限速（429）就等一下再试。"""
    for attempt in range(5):
        r = requests.get(BASE + path, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"多次被限速：{path}")


def fetch_all(path, params):
    """Kalshi 每页最多返回一部分结果，用 cursor 一页一页往后翻，直到翻完。"""
    out, cursor = [], None
    while True:
        p = dict(params, limit=1000)
        if cursor:
            p["cursor"] = cursor
        data = get(path, p)
        out += data.get("markets", [])
        cursor = data.get("cursor")
        print(f"  {path}: 已取 {len(out)} 个")
        if not cursor:
            return out
        time.sleep(0.2)


if __name__ == "__main__":
    # 1) 新旧数据的分界时间
    try:
        print("历史数据分界：", get("/historical/cutoff"))
    except Exception as e:
        print("查不到分界时间（不影响后面）：", e)

    # 2) 两个接口都查，再合并去重
    live = fetch_all("/markets", {"series_ticker": SERIES, "status": "settled"})
    try:
        hist = fetch_all("/historical/markets", {"series_ticker": SERIES})
    except Exception as e:
        print("历史接口出错：", e)
        hist = []
    markets = {m["ticker"]: m for m in hist + live}
    markets = list(markets.values())
    (OUT_RAW / "markets.json").write_text(json.dumps(markets, indent=1))

    # 3) 先打印一个样例，看看每个合约都有哪些字段
    print("\n样例合约：")
    print(json.dumps(markets[0], indent=1)[:2500])

    # 4) 整理成表格。字段名可能随 API 版本变化，缺的就留空
    keep = ["ticker", "event_ticker", "title", "subtitle", "yes_sub_title",
            "strike_type", "floor_strike", "cap_strike",
            "open_time", "close_time", "result", "volume", "status"]
    df = pd.DataFrame([{k: m.get(k) for k in keep} for m in markets])
    df.to_csv("data/markets.csv", index=False)

    print(f"\n共 {len(df)} 个合约，{df['event_ticker'].nunique()} 天")
    print("结算结果分布：\n", df["result"].value_counts(dropna=False))
    print("最早 / 最晚收盘：", df["close_time"].min(), "/", df["close_time"].max())
