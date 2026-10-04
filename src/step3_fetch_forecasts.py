"""
第 3 步：下载“当时发布的”历史天气预报（纽约中央公园，逐小时气温）。

运行：  python src/step3_fetch_forecasts.py
输出：  data/forecasts_<模型>.csv

数据来源：Open-Meteo Previous Runs API（免费，不用注册）
  它的特别之处：对过去每一个小时，不仅给出最终的气温，还给出“提前 1 天、提前 2 天……
  时预报的气温”。这正是避免前视偏差需要的东西。

字段含义（以 2025-07-01 15:00 这个小时为例）：
  temperature_2m               最短时效的预报，几乎等于实际值（只作参考，不能用来预测）
  temperature_2m_previous_day1 大约 24 小时前发布的预报对这一小时的预测
  temperature_2m_previous_day2 大约 48 小时前发布的预报对这一小时的预测

为什么要 previous_day2，不只用 day1（前视偏差的具体例子）：
  假设我们用“前一天下午 4 点”的市场价格。当天最高温可能出现在任何一个小时。
  对当天晚上 11 点来说，“24 小时前的预报”是前一天晚上 11 点才发布的，
  比下午 4 点晚 —— 那时的我们还看不到。用它就是作弊。
  48 小时前的预报则全部早于前一天下午 4 点，是安全的。
  我们两个都下载，之后可以比较：如果 day1 显得特别准，就是前视偏差在起作用的证据。

两个模型：
  GFS（美国）：2021 年起就有，覆盖我们全部数据
  ECMWF IFS（欧洲）：通常更准，但这个接口只有 2024 年起的数据
"""

import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LAT, LON = 40.7790, -73.9692          # 中央公园气象站（Kalshi 用它结算）
MODELS = {"gfs": "gfs_seamless", "ecmwf": "ecmwf_ifs025"}
START, END = "2023-01-01", "2026-10-01"
Path("data").mkdir(exist_ok=True)


def fetch_year(model, start, end):
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m,temperature_2m_previous_day1,temperature_2m_previous_day2",
        "models": model,
        "start_date": start, "end_date": end,
        "temperature_unit": "fahrenheit",   # Kalshi 用华氏度
        "timezone": "America/New_York",     # 用纽约当地时间，方便和合约日期对齐
    }
    r = requests.get(URL, params=params, timeout=60)
    if r.status_code != 200:
        print("  出错：", r.status_code, r.text[:300])
        return None
    return pd.DataFrame(r.json()["hourly"])


if __name__ == "__main__":
    for name, model in MODELS.items():
        print(f"下载 {name}（{model}）…")
        parts = []
        # 一次请求太长会被拒，所以按年分段
        for y in range(int(START[:4]), int(END[:4]) + 1):
            s, e = max(START, f"{y}-01-01"), min(END, f"{y}-12-31")
            df = fetch_year(model, s, e)
            if df is not None:
                parts.append(df)
                print(f"  {y}: {len(df)} 小时，非空 day2 预报 {df.iloc[:, -1].notna().sum()} 个")
            time.sleep(1)
        if parts:
            out = pd.concat(parts, ignore_index=True)
            out.to_csv(f"data/forecasts_{name}.csv", index=False)
            print(f"  已保存 data/forecasts_{name}.csv，前几行：")
            print(out.dropna().head(3).to_string())
