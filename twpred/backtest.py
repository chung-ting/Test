"""回測與績效統計。

勝率定義：扣掉來回交易成本後報酬 > 0。
所有門檻都是事前可知的（每日排名前 K 名、固定機率門檻），不會用整段樣本外分布決定門檻。
"""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from twpred.config import Config


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """勝率 95% 信賴區間。樣本少時區間很寬，看到高勝率先看這個。"""
    if n == 0:
        return (math.nan, math.nan)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def tradable_rows(preds: pd.DataFrame) -> pd.DataFrame:
    return preds[preds["tradable"] & preds["liquid"] & preds["net_ret"].notna()]


def _stats(t: pd.DataFrame) -> dict:
    n = len(t)
    wins = int((t["net_ret"] > 0).sum())
    lo, hi = wilson(wins, n)
    return {
        "trades": n,
        "days_with_trade": t["date"].nunique(),
        "win_rate": wins / n if n else math.nan,
        "win_ci_low": lo,
        "win_ci_high": hi,
        "avg_net_ret": t["net_ret"].mean() if n else math.nan,
        "median_net_ret": t["net_ret"].median() if n else math.nan,
    }


def daily_top_k(preds: pd.DataFrame, k: int) -> pd.DataFrame:
    t = tradable_rows(preds)
    return t[t.groupby("date")["prob"].rank(ascending=False, method="first") <= k]


def threshold_curve(preds: pd.DataFrame, top_ks=(1, 3, 5, 10), probs=(0.3, 0.4, 0.5, 0.6, 0.7)) -> pd.DataFrame:
    """不同出手門檻下的交易次數與勝率 —— 「門檻越嚴、勝率越高、機會越少」那條線。"""
    t = tradable_rows(preds)
    rows = [{"rule": "全部（基準）", **_stats(t)}]
    for k in top_ks:
        rows.append({"rule": f"每日前 {k} 名", **_stats(daily_top_k(preds, k))})
    for p in probs:
        rows.append({"rule": f"機率 ≥ {p:.2f}", **_stats(t[t["prob"] >= p])})
    for k in (1, 3):
        for p in (0.5, 0.6):
            top = daily_top_k(preds, k)
            rows.append({"rule": f"每日前 {k} 名且機率 ≥ {p:.2f}", **_stats(top[top["prob"] >= p])})
    return pd.DataFrame(rows)


def yearly(trades: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in sorted(base["date"].dt.year.unique()):
        t = trades[trades["date"].dt.year == year]
        b = base[base["date"].dt.year == year]
        s = _stats(t)
        rows.append({
            "year": year,
            "trades": s["trades"],
            "win_rate": s["win_rate"],
            "base_win_rate": (b["net_ret"] > 0).mean(),
            "avg_net_ret": s["avg_net_ret"],
        })
    return pd.DataFrame(rows)


def auc_by_year(preds: pd.DataFrame) -> pd.DataFrame:
    p = preds[preds["label"].notna() & preds["liquid"]]
    rows = []
    for year, g in [("全部", p), *p.groupby(p["date"].dt.year)]:
        auc = roc_auc_score(g["label"], g["prob"]) if g["label"].nunique() == 2 else math.nan
        rows.append({"year": year, "auc": auc, "n": len(g)})
    return pd.DataFrame(rows)


def portfolio(preds: pd.DataFrame, market: pd.DataFrame, cfg: Config, k: int, min_prob: float) -> tuple[pd.DataFrame, dict]:
    """每 horizon 天換股一次（不重疊），等權買進當天前 k 名且機率 ≥ min_prob；沒有就空手。"""
    dates = np.sort(preds["date"].unique())
    rebal = dates[:: cfg.horizon]
    t = tradable_rows(preds)
    universe = t.groupby("date")["fwd_ret"].mean()  # 基準：等權買全部股票，不扣成本
    picks = daily_top_k(preds, k)
    picks = picks[picks["prob"] >= min_prob].groupby("date")["net_ret"].mean()

    m = market.set_index("date")["close"].sort_index()
    rows = []
    for d in rebal:
        idx = m.index.searchsorted(d)
        exit_idx = idx + cfg.horizon
        ok = exit_idx < len(m) and m.index[idx] == d
        mkt = m.iloc[exit_idx] / m.iloc[idx] - 1 if ok else math.nan
        rows.append({
            "date": d,
            "invested": d in picks.index,
            "strategy": picks.get(d, 0.0),
            "equal_weight": universe.get(d, math.nan),
            "taiex": mkt,
        })
    curve = pd.DataFrame(rows).dropna(subset=["equal_weight"]).set_index("date")

    periods_per_year = 252 / cfg.horizon
    summary = {}
    for col in ("strategy", "equal_weight", "taiex"):
        r = curve[col].fillna(0)
        equity = (1 + r).cumprod()
        years = len(r) / periods_per_year
        summary[col] = {
            "total_return": equity.iloc[-1] - 1,
            "cagr": equity.iloc[-1] ** (1 / years) - 1 if years > 0 else math.nan,
            "max_drawdown": (equity / equity.cummax() - 1).min(),
            "sharpe": r.mean() / r.std() * math.sqrt(periods_per_year) if r.std() > 0 else math.nan,
        }
        curve[f"{col}_equity"] = equity
    summary["strategy"]["exposure"] = curve["invested"].mean()
    return curve, summary
