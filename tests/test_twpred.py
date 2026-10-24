"""用合成資料測試流程正確性（尤其是「沒有偷看未來」）。合成資料的績效沒有任何意義。"""

import unittest

import numpy as np
import pandas as pd

from twpred import backtest
from twpred.config import Config
from twpred.features import FEATURES, add_labels, build_dataset, revenue_features
from twpred.model import walk_forward

CFG = Config(min_train_days=300, retrain_every=60, min_turnover=0)


def synthetic(n_stocks=12, n_days=700, signal=0.0, seed=0) -> dict[str, pd.DataFrame]:
    """signal > 0 時，外資買超會預告未來 5 日報酬（埋一個已知訊號讓模型找）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    prices, inst, margin, rev = [], [], [], []
    for s in range(n_stocks):
        sid = str(1000 + s)
        flow = rng.normal(0, 1, n_days)
        drift = np.zeros(n_days)
        # 外資在 t 日的買超影響 t+1..t+5 的報酬
        for lag in range(1, 6):
            drift[lag:] += signal * flow[:-lag] / 5
        ret = rng.normal(0, 0.02, n_days) + drift
        close = 100 * np.exp(np.cumsum(ret))
        open_ = close * np.exp(rng.normal(0, 0.005, n_days))
        high = np.maximum(open_, close) * 1.01
        low = np.minimum(open_, close) * 0.99
        volume = rng.integers(5_000_000, 10_000_000, n_days).astype(float)
        prices.append(pd.DataFrame({"date": dates, "stock_id": sid, "open": open_, "high": high,
                                    "low": low, "close": close, "volume": volume}))
        inst.append(pd.DataFrame({"date": dates, "stock_id": sid, "foreign_net": flow * 500_000,
                                  "trust_net": rng.normal(0, 1e5, n_days), "dealer_net": rng.normal(0, 1e5, n_days)}))
        margin.append(pd.DataFrame({"date": dates, "stock_id": sid,
                                    "margin_balance": 1e6 + rng.normal(0, 1e4, n_days).cumsum(),
                                    "short_balance": 1e5 + rng.normal(0, 1e3, n_days).cumsum()}))
        months = pd.date_range("2019-01-01", dates[-1], freq="MS")
        rev.append(pd.DataFrame({"stock_id": sid, "revenue_year": months.year, "revenue_month": months.month,
                                 "revenue": rng.uniform(1e9, 2e9, len(months))}))
    market = pd.concat(prices).groupby("date")["close"].mean().reset_index()
    return {"prices": pd.concat(prices, ignore_index=True), "institutional": pd.concat(inst, ignore_index=True),
            "margin": pd.concat(margin, ignore_index=True), "revenue": pd.concat(rev, ignore_index=True),
            "market": market}


class NoLookaheadTests(unittest.TestCase):
    def test_features_unchanged_by_future_data(self):
        data = synthetic(n_stocks=4, n_days=300)
        cutoff = data["prices"]["date"].unique()[200]
        truncated = {
            k: v[v["date"] <= cutoff] if "date" in v else v for k, v in data.items()
        }
        full = build_dataset(data, CFG)
        part = build_dataset(truncated, CFG)
        a = full[full["date"] == cutoff].set_index("stock_id")[FEATURES]
        b = part[part["date"] == cutoff].set_index("stock_id")[FEATURES]
        pd.testing.assert_frame_equal(a, b.loc[a.index])

    def test_revenue_available_on_11th_of_next_month(self):
        rev = pd.DataFrame({"stock_id": "1", "revenue_year": [2024] * 14, "revenue_month": list(range(1, 13)) + [1, 2],
                            "revenue": range(14)})
        rev.loc[12:, "revenue_year"] = 2025
        feat = revenue_features(rev)
        jan = feat[feat["period"] == "2024-01-01"].iloc[0]
        self.assertEqual(jan["available_date"], pd.Timestamp("2024-02-11"))

    def test_walk_forward_purges_label_overlap(self):
        df = build_dataset(synthetic(n_stocks=4, n_days=450), CFG)
        preds, _ = walk_forward(df, CFG)
        dates = np.sort(df["date"].unique())
        for _, g in preds.groupby("train_end"):
            test_start = g["date"].min()
            gap = np.searchsorted(dates, test_start) - np.searchsorted(dates, g["train_end"].iloc[0])
            self.assertGreater(gap, CFG.horizon - 1)

    def test_noise_has_no_skill(self):
        """純雜訊資料上 AUC 必須接近 0.5，否則代表某處偷看了未來。"""
        df = build_dataset(synthetic(signal=0.0, seed=1), CFG)
        preds, _ = walk_forward(df, CFG)
        auc = backtest.auc_by_year(preds).iloc[0]["auc"]
        self.assertLess(abs(auc - 0.5), 0.04)

    def test_planted_signal_is_found(self):
        df = build_dataset(synthetic(signal=0.03, seed=2), CFG)
        preds, importance = walk_forward(df, CFG)
        self.assertGreater(backtest.auc_by_year(preds).iloc[0]["auc"], 0.6)
        self.assertTrue(importance.index[0].startswith("foreign"))


class LabelTests(unittest.TestCase):
    def test_net_return_and_locked_limit_up(self):
        df = pd.DataFrame({
            "stock_id": "1",
            "open":  [100, 110, 100, 100, 100, 100, 100],
            "high":  [100, 110, 100, 100, 100, 100, 100],
            "low":   [100, 110, 100, 100, 100, 100, 100],
            "close": [100, 110, 100, 100, 100, 120, 100],
        })
        out = add_labels(df, Config(horizon=5))
        self.assertFalse(out.loc[0, "tradable"])  # 隔天一字漲停買不到
        self.assertAlmostEqual(out.loc[1, "fwd_ret"], 100 / 100 - 1)
        self.assertAlmostEqual(out.loc[0, "net_ret"], 120 / 110 - 1 - Config().round_trip_cost)
        self.assertTrue(np.isnan(out.loc[2, "label"]))  # 未來不足 5 天

    def test_wilson_interval(self):
        lo, hi = backtest.wilson(8, 10)
        self.assertLess(lo, 0.5)  # 10 次中 8 次，統計上仍不能排除 50%
        self.assertGreater(hi, 0.9)


class PortfolioTests(unittest.TestCase):
    def test_threshold_curve_and_portfolio_run(self):
        data = synthetic(signal=0.01, seed=3)
        df = build_dataset(data, CFG)
        preds, _ = walk_forward(df, CFG)
        curve = backtest.threshold_curve(preds)
        base, top1 = curve.iloc[0], curve[curve["rule"] == "每日前 1 名"].iloc[0]
        self.assertGreater(top1["win_rate"], base["win_rate"])
        _, summary = backtest.portfolio(preds, data["market"], CFG, k=3, min_prob=0.0)
        self.assertEqual(summary["strategy"]["exposure"], 1.0)


if __name__ == "__main__":
    unittest.main()
