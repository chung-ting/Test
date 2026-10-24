"""特徵與標籤。

原則：t 日的每個特徵只能用 t 日收盤（含）以前「已公開」的資訊。
- 價量、法人、融資券：當日盤後即公布，可用。
- 月營收：法規為次月 10 日前公告，保守起見視為次月 11 日才可用。
標籤用 t+1 開盤到 t+horizon 收盤的報酬，扣掉來回交易成本。
"""

import numpy as np
import pandas as pd

from twpred.config import Config

FEATURES = [
    # 技術面
    "ret_1", "ret_5", "ret_20", "ret_60", "bias_5", "bias_20", "bias_60",
    "rsi_14", "k_9", "d_9", "macd_hist", "atr_14", "bb_width", "vol_ratio",
    "vol_ratio_5", "dist_high_60", "dist_low_60", "gap", "limit_up",
    # 籌碼 / 風向
    "foreign_ratio", "foreign_ratio_5", "foreign_ratio_20", "trust_ratio", "trust_ratio_5",
    "dealer_ratio_5", "foreign_streak", "trust_streak", "margin_chg_5", "margin_chg_20",
    "short_margin_ratio",
    # 月營收
    "rev_yoy", "rev_mom", "rev_yoy_3m", "rev_new_high_12",
    # 大盤
    "mkt_ret_1", "mkt_ret_5", "mkt_ret_20", "mkt_bias_20",
    # 季節（投信季底作帳）
    "days_to_quarter_end",
    # 橫斷面排名（同一天內和其他股票比）
    "ret_5_rank", "ret_20_rank", "foreign_ratio_5_rank", "trust_ratio_5_rank", "vol_ratio_rank", "rev_yoy_rank",
]


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def _streak(net: pd.Series) -> pd.Series:
    """連續買超天數（正）或連續賣超天數（負）。"""
    sign = np.sign(net.fillna(0))
    block = (sign != sign.shift()).cumsum()
    return sign * sign.groupby(block).cumcount().add(1)


def _per_stock(g: pd.DataFrame) -> pd.DataFrame:
    c, h, l, o, v = g["close"], g["high"], g["low"], g["open"], g["volume"]
    f = pd.DataFrame(index=g.index)
    for n in (1, 5, 20, 60):
        f[f"ret_{n}"] = c.pct_change(n)
    for n in (5, 20, 60):
        f[f"bias_{n}"] = c / c.rolling(n).mean() - 1
    f["rsi_14"] = _rsi(c, 14)

    low9, high9 = l.rolling(9).min(), h.rolling(9).max()
    rsv = (c - low9) / (high9 - low9).replace(0, np.nan) * 100
    f["k_9"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    f["d_9"] = f["k_9"].ewm(alpha=1 / 3, adjust=False).mean()

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    prev_c = c.shift()
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    f["atr_14"] = tr.rolling(14).mean() / c
    f["bb_width"] = c.rolling(20).std() * 4 / c.rolling(20).mean()

    vma20 = v.rolling(20).mean()
    f["vol_ratio"] = v / vma20
    f["vol_ratio_5"] = v.rolling(5).mean() / vma20
    f["dist_high_60"] = c / h.rolling(60).max() - 1
    f["dist_low_60"] = c / l.rolling(60).min() - 1
    f["gap"] = o / prev_c - 1
    f["limit_up"] = (f["ret_1"] >= 0.095).astype(float)
    f["turnover_20"] = (c * v).rolling(20).mean()

    for col, name in (("foreign_net", "foreign"), ("trust_net", "trust"), ("dealer_net", "dealer")):
        net = g[col].fillna(0)
        f[f"{name}_ratio"] = net / v.replace(0, np.nan)
        f[f"{name}_ratio_5"] = net.rolling(5).sum() / v.rolling(5).sum().replace(0, np.nan)
        f[f"{name}_ratio_20"] = net.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    f["foreign_streak"] = _streak(g["foreign_net"])
    f["trust_streak"] = _streak(g["trust_net"])

    mb = g["margin_balance"].ffill()
    f["margin_chg_5"] = mb.pct_change(5, fill_method=None)
    f["margin_chg_20"] = mb.pct_change(20, fill_method=None)
    f["short_margin_ratio"] = g["short_balance"].ffill() / mb.replace(0, np.nan)
    return f


def revenue_features(revenue: pd.DataFrame) -> pd.DataFrame:
    """每檔每月一列，附上可使用日 available_date（次月 11 日）。"""
    r = revenue.copy()
    r["period"] = pd.to_datetime(dict(year=r["revenue_year"], month=r["revenue_month"], day=1))
    r = r.sort_values(["stock_id", "period"]).drop_duplicates(["stock_id", "period"], keep="last")
    out = []
    for sid, g in r.groupby("stock_id"):
        # 以完整月份序列計算，避免缺月時 shift(12) 對錯月份。
        s = g.set_index("period")["revenue"].asfreq("MS")
        feat = pd.DataFrame({
            "rev_yoy": s / s.shift(12) - 1,
            "rev_mom": s / s.shift(1) - 1,
            "rev_yoy_3m": s.rolling(3).sum() / s.shift(12).rolling(3).sum() - 1,
            "rev_new_high_12": (s >= s.rolling(12).max()).astype(float).where(s.shift(11).notna()),
        })
        feat = feat[s.notna()].reset_index()
        feat["stock_id"] = sid
        out.append(feat)
    feat = pd.concat(out, ignore_index=True)
    feat["available_date"] = feat["period"] + pd.offsets.MonthBegin(1) + pd.Timedelta(days=10)
    return feat


def market_features(market: pd.DataFrame) -> pd.DataFrame:
    m = market.sort_values("date").set_index("date")["close"]
    return pd.DataFrame({
        "mkt_ret_1": m.pct_change(),
        "mkt_ret_5": m.pct_change(5),
        "mkt_ret_20": m.pct_change(20),
        "mkt_bias_20": m / m.rolling(20).mean() - 1,
    }).reset_index()


def add_labels(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """t+1 開盤買、t+horizon 收盤賣的淨報酬。t+1 開盤一字漲停買不到 → 不可交易。"""
    g = df.groupby("stock_id", sort=False)
    entry = g["open"].shift(-1)
    exit_ = g["close"].shift(-cfg.horizon)
    next_high, next_low = g["high"].shift(-1), g["low"].shift(-1)
    locked_limit_up = (entry >= df["close"] * 1.095) & (next_high == next_low)
    df["fwd_ret"] = exit_ / entry - 1
    df["net_ret"] = df["fwd_ret"] - cfg.round_trip_cost
    df["tradable"] = df["fwd_ret"].notna() & ~locked_limit_up
    df["label"] = (df["net_ret"] > cfg.target_return).astype(float).where(df["fwd_ret"].notna())
    return df


def build_dataset(data: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    px = data["prices"].sort_values(["stock_id", "date"])
    px = px[(px["volume"] > 0) & (px["close"] > 0)]
    df = px.merge(data["institutional"], on=["date", "stock_id"], how="left")
    df = df.merge(data["margin"], on=["date", "stock_id"], how="left")
    for col in ("foreign_net", "trust_net", "dealer_net", "margin_balance", "short_balance"):
        if col not in df:
            df[col] = np.nan
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    feats = pd.concat([_per_stock(g) for _, g in df.groupby("stock_id", sort=False)])
    df = df.join(feats)

    rev = revenue_features(data["revenue"]).sort_values("available_date")
    df = pd.merge_asof(
        df.sort_values("date"), rev.drop(columns="period"),
        left_on="date", right_on="available_date", by="stock_id", direction="backward",
    ).drop(columns="available_date")
    df = df.merge(market_features(data["market"]), on="date", how="left")

    quarter_end = df["date"] + pd.offsets.QuarterEnd(0)
    df["days_to_quarter_end"] = (quarter_end - df["date"]).dt.days

    for col in ("ret_5", "ret_20", "foreign_ratio_5", "trust_ratio_5", "vol_ratio", "rev_yoy"):
        df[f"{col}_rank"] = df.groupby("date")[col].rank(pct=True)

    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    df = add_labels(df, cfg)
    df["liquid"] = df["turnover_20"] >= cfg.min_turnover
    return df.sort_values(["date", "stock_id"]).reset_index(drop=True)
