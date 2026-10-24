"""FinMind 資料下載與快取。

免費註冊 https://finmindtrade.com 可拿 token（提高每小時請求上限），
設環境變數 FINMIND_TOKEN 即可。資料會快取在 data/cache/，重跑不會重抓。
"""

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://api.finmindtrade.com/api/v4/data"
CACHE = Path("data/cache")


def _request(dataset: str, data_id: str, start: str, end: str, token: str | None) -> pd.DataFrame:
    params = {"dataset": dataset, "data_id": data_id, "start_date": start, "end_date": end}
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.load(resp)
            break
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
    if payload.get("status") == 402:
        raise RuntimeError("FinMind 已達每小時請求上限；已下載的部分有快取，等一小時後重跑即可接續（設 FINMIND_TOKEN 可提高上限）")
    if payload.get("status") != 200:
        raise RuntimeError(f"FinMind {dataset} {data_id}: {payload.get('msg')}")
    return pd.DataFrame(payload["data"])


def fetch(dataset: str, data_id: str, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    path = CACHE / dataset / f"{data_id}_{start}_{end}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype={"stock_id": str})
    df = _request(dataset, data_id, start, end, os.environ.get("FINMIND_TOKEN"))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def adjust_for_dividends(px: pd.DataFrame, div: pd.DataFrame) -> pd.DataFrame:
    """向後還原權息：除權息日之前的價格乘上 after_price / before_price。

    免費帳號拿不到 TaiwanStockPriceAdj，不還原的話除息日會出現假跌幅
    （高殖利率股一年可達 5% 以上），污染標籤與技術指標。
    """
    px = px.sort_values("date").reset_index(drop=True)
    if div.empty:
        return px
    div = div[(div["before_price"] > 0) & (div["after_price"] > 0)]
    factor = pd.Series(1.0, index=px.index)
    dates = pd.to_datetime(px["date"])
    for ex_date, ratio in zip(pd.to_datetime(div["date"]), div["after_price"] / div["before_price"]):
        factor[dates < ex_date] *= ratio
    for col in ("open", "high", "low", "close"):
        px[col] = px[col] * factor
    return px


def _prices(stock_id: str, start: str, end: str, refresh: bool) -> pd.DataFrame:
    df = fetch("TaiwanStockPrice", stock_id, start, end, refresh)
    df = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})[
        ["date", "stock_id", "open", "high", "low", "close", "volume"]
    ]
    div = fetch("TaiwanStockDividendResult", stock_id, start, end, refresh)
    return adjust_for_dividends(df, div)


def _institutional(stock_id: str, start: str, end: str, refresh: bool) -> pd.DataFrame:
    df = fetch("TaiwanStockInstitutionalInvestorsBuySell", stock_id, start, end, refresh)
    if df.empty:
        return df
    df["net"] = df["buy"] - df["sell"]
    group = df["name"].map(
        lambda n: "foreign_net" if n.startswith("Foreign")
        else "trust_net" if n == "Investment_Trust"
        else "dealer_net"
    )
    out = df.assign(group=group).pivot_table(index=["date", "stock_id"], columns="group", values="net", aggfunc="sum")
    return out.reset_index()


def _margin(stock_id: str, start: str, end: str, refresh: bool) -> pd.DataFrame:
    df = fetch("TaiwanStockMarginPurchaseShortSale", stock_id, start, end, refresh)
    if df.empty:
        return df
    return df.rename(
        columns={"MarginPurchaseTodayBalance": "margin_balance", "ShortSaleTodayBalance": "short_balance"}
    )[["date", "stock_id", "margin_balance", "short_balance"]]


def _revenue(stock_id: str, start: str, end: str, refresh: bool) -> pd.DataFrame:
    # 往前多抓一年，才算得出 YoY。
    early = str(int(start[:4]) - 1) + start[4:]
    df = fetch("TaiwanStockMonthRevenue", stock_id, early, end, refresh)
    if df.empty:
        return df
    return df[["stock_id", "revenue_year", "revenue_month", "revenue"]]


def load_universe(stock_ids: list[str], start: str, end: str, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """回傳 {'prices', 'institutional', 'margin', 'revenue', 'market'} 五張長表。"""
    parts: dict[str, list[pd.DataFrame]] = {k: [] for k in ("prices", "institutional", "margin", "revenue")}
    for i, sid in enumerate(stock_ids, 1):
        print(f"[{i}/{len(stock_ids)}] {sid}", flush=True)
        parts["prices"].append(_prices(sid, start, end, refresh))
        parts["institutional"].append(_institutional(sid, start, end, refresh))
        parts["margin"].append(_margin(sid, start, end, refresh))
        parts["revenue"].append(_revenue(sid, start, end, refresh))
    out = {k: pd.concat([d for d in v if not d.empty], ignore_index=True) for k, v in parts.items()}
    market = fetch("TaiwanStockPrice", "TAIEX", start, end, refresh)
    out["market"] = market[["date", "close"]]
    for k in ("prices", "institutional", "margin", "market"):
        out[k]["date"] = pd.to_datetime(out[k]["date"])
    for k in ("prices", "institutional", "margin", "revenue"):
        out[k]["stock_id"] = out[k]["stock_id"].astype(str)
    return out
