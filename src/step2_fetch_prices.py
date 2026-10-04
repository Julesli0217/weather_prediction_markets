"""
第 2 步：下载每个合约的价格历史（每小时一根“K 线”）。

运行：  python src/step2_fetch_prices.py
输出：  data/raw/candles/<合约代码>.json   每个合约一个文件，原样保存服务器返回的数据

为什么只要 2023 年以后：
  第 1 步的数据显示，2021–2022 年的合约每天数量不固定，常常只有“高于 X 度”这种问题。
  2023 年起每天固定 6 个区间合约，格式统一，这样分析才可比。
  注意：合约代码前缀有两种，旧的叫 HIGHNY，后来改叫 KXHIGHNY，两种都要。

K 线（candlestick）是什么：
  把一段时间（这里是 1 小时）内的成交浓缩成几个数：开盘价、最高价、最低价、收盘价，
  再加上这一小时结束时的买价（bid）、卖价（ask）和成交量。之后我们会取“某个固定时刻”
  的价格，和那个时刻能拿到的天气预报对比。

这个脚本可以中断后重跑：已经下载过的合约会自动跳过。
大约 8200 个合约，全部下完可能要半小时以上，可以开着终端去做别的事。
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
    """把 '2026-08-01T14:00:00Z' 这样的时间转成 Unix 时间戳（从 1970 年起的秒数），API 要这种格式。"""
    return int(pd.Timestamp(iso).timestamp())


def get(path, params):
    """和第 1 步一样：发请求，被限速就等一下再试。找不到（404）就返回 None。"""
    for attempt in range(5):
        r = requests.get(BASE + path, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"多次被限速：{path}")


def fetch_candles(m):
    params = {
        "start_ts": to_ts(m["open_time"]),
        "end_ts": to_ts(m["close_time"]),
        "period_interval": 60,  # 单位是分钟：每小时一根
    }
    # 新合约在普通接口，旧合约被挪到 historical 接口：先试新的，找不到再试旧的
    series = m["ticker"].split("-")[0]  # HIGHNY 或 KXHIGHNY
    data = get(f"/series/{series}/markets/{m['ticker']}/candlesticks", params)
    if not data or not data.get("candlesticks"):
        data = get(f"/historical/markets/{m['ticker']}/candlesticks", params)
    return data


if __name__ == "__main__":
    markets = pd.read_csv("data/markets.csv")
    markets = markets[markets["close_time"] >= "2023-01-01"]
    print(f"需要下载 {len(markets)} 个合约的价格历史")

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
            print(f"  进度 {i}/{len(markets)}")
        time.sleep(0.07)  # 放慢一点，免得被限速

    print(f"完成。失败 {len(failed)} 个")
    for t, e in failed[:10]:
        print("  ", t, e)

    # 打印一个样例，下一步要根据它的字段来写解析代码
    sample = next(OUT.glob("*.json"))
    d = json.loads(sample.read_text())
    print("\n样例：", sample.name)
    print(json.dumps(d, indent=1)[:2000])
