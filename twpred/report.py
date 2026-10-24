"""Markdown 回測報告。"""

import math

import pandas as pd

from twpred.config import Config


def _pct(x: float, digits: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{digits}f}%"


def _table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def build(
    cfg: Config,
    preds: pd.DataFrame,
    curve_table: pd.DataFrame,
    auc: pd.DataFrame,
    yearly_table: pd.DataFrame,
    port_summary: dict,
    port_rule: str,
    importance: pd.DataFrame,
) -> str:
    first, last = preds["date"].min().date(), preds["date"].max().date()
    out = [
        "# 台股短期趨勢預測：回測報告",
        "",
        f"- 樣本外期間：{first} ～ {last}（walk-forward，每 {cfg.retrain_every} 個交易日重訓）",
        f"- 交易規則：t 日收盤出訊號 → t+1 開盤買 → t+{cfg.horizon} 收盤賣",
        f"- 來回成本：{_pct(cfg.round_trip_cost, 3)}（已從所有報酬扣除）",
        f"- 勝率定義：扣成本後報酬 > 0",
        "",
        "## 1. 門檻 vs. 勝率（最重要的一張表）",
        "",
        "看「勝率下限」而不是只看勝率：交易次數少時，95% 信賴區間會很寬。",
        "和第一列「全部（基準）」比，才知道模型有沒有真的挑出比較好的股票。",
        "",
    ]
    t = curve_table.copy()
    t = pd.DataFrame({
        "出手規則": t["rule"],
        "交易次數": t["trades"],
        "有交易的天數": t["days_with_trade"],
        "勝率": t["win_rate"].map(_pct),
        "95% 信賴區間": [f"{_pct(a)} ～ {_pct(b)}" for a, b in zip(t["win_ci_low"], t["win_ci_high"])],
        "平均淨報酬": t["avg_net_ret"].map(lambda x: _pct(x, 2)),
        "中位數淨報酬": t["median_net_ret"].map(lambda x: _pct(x, 2)),
    })
    out += [_table(t), ""]

    out += [f"## 2. 模型鑑別力（AUC，0.5 = 瞎猜）", ""]
    a = auc.assign(auc=auc["auc"].map(lambda x: "—" if math.isnan(x) else f"{x:.3f}"))
    out += [_table(a.rename(columns={"year": "年度", "auc": "AUC", "n": "樣本數"})), ""]

    out += [f"## 3. 逐年表現（{port_rule}）", "", "只有某一年特別好、其他年普通，多半是運氣。", ""]
    y = yearly_table
    y = pd.DataFrame({
        "年度": y["year"],
        "交易次數": y["trades"],
        "勝率": y["win_rate"].map(_pct),
        "基準勝率": y["base_win_rate"].map(_pct),
        "平均淨報酬": y["avg_net_ret"].map(lambda x: _pct(x, 2)),
    })
    out += [_table(y), ""]

    out += [f"## 4. 投資組合模擬（{port_rule}，每 {cfg.horizon} 天換股）", ""]
    names = {"strategy": "策略（扣成本）", "equal_weight": "等權買全部（未扣成本）", "taiex": "加權指數"}
    rows = [
        {
            "": names[k],
            "總報酬": _pct(v["total_return"]),
            "年化報酬": _pct(v["cagr"]),
            "最大回撤": _pct(v["max_drawdown"]),
            "夏普值": "—" if math.isnan(v["sharpe"]) else f"{v['sharpe']:.2f}",
        }
        for k, v in port_summary.items()
    ]
    out += [_table(pd.DataFrame(rows)), "", f"策略持股時間比例：{_pct(port_summary['strategy']['exposure'])}", ""]

    out += ["## 5. 特徵重要性（前 15）", ""]
    imp = importance.head(15).reset_index()
    imp.columns = ["特徵", "重要性"]
    imp["重要性"] = imp["重要性"].map(_pct)
    out += [_table(imp), ""]

    out += [
        "## 注意",
        "",
        "- 股票池是「現在的」台灣 50 成分股，有倖存者偏差，實際結果會比回測差。",
        "- 沒有計入滑價與漲跌停排隊買不到的情況（一字漲停已排除）。",
        "- 回測好不代表未來會好；實盤前請先模擬交易至少一到兩季。",
        "- 本工具僅供研究，不構成投資建議。",
    ]
    return "\n".join(out)
