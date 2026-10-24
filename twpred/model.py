"""LightGBM + walk-forward（滾動前進）驗證。

每 retrain_every 個交易日重新訓練一次，只用測試區間開始「之前」的資料；
訓練資料的標籤會用到未來 horizon 天，所以訓練集最後 horizon 天要剔除（purge），
否則標籤會偷看到測試期的價格。超參數固定不調，避免對歷史過度擬合。
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from twpred.config import Config
from twpred.features import FEATURES


def make_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        random_state=seed,
        verbose=-1,
    )


def _trainable(df: pd.DataFrame) -> pd.Series:
    return df["label"].notna() & df["liquid"]


def walk_forward(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """回傳 (樣本外預測, 平均特徵重要性)。預測只涵蓋 min_train_days 之後的日期。"""
    dates = np.sort(df["date"].unique())
    if len(dates) <= cfg.min_train_days + cfg.horizon:
        raise ValueError(f"資料只有 {len(dates)} 個交易日，至少需要 {cfg.min_train_days + cfg.horizon + 1}")

    preds, importances = [], []
    for start in range(cfg.min_train_days, len(dates), cfg.retrain_every):
        test_dates = dates[start : start + cfg.retrain_every]
        train_end = start - cfg.horizon  # purge：標籤視窗不可跨入測試期
        train_start = 0 if cfg.train_window is None else max(0, train_end - cfg.train_window)
        train_mask = df["date"].isin(dates[train_start:train_end]) & _trainable(df)
        test_mask = df["date"].isin(test_dates)

        model = make_model(cfg.seed)
        train = df[train_mask]
        model.fit(train[FEATURES], train["label"].astype(int))
        test = df.loc[test_mask, ["date", "stock_id", "fwd_ret", "net_ret", "label", "tradable", "liquid"]].copy()
        test["prob"] = model.predict_proba(df.loc[test_mask, FEATURES])[:, 1]
        test["train_end"] = dates[train_end - 1]
        preds.append(test)
        importances.append(pd.Series(model.booster_.feature_importance("gain"), index=FEATURES))

    imp = pd.concat(importances, axis=1).mean(axis=1)
    imp = (imp / imp.sum()).sort_values(ascending=False).rename("importance").to_frame()
    return pd.concat(preds, ignore_index=True), imp


def fit_latest(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """用全部已知標籤訓練，對最新一個交易日出訊號。"""
    train = df[_trainable(df)]
    if cfg.train_window is not None:
        keep = np.sort(train["date"].unique())[-cfg.train_window :]
        train = train[train["date"].isin(keep)]
    model = make_model(cfg.seed)
    model.fit(train[FEATURES], train["label"].astype(int))
    latest = df[(df["date"] == df["date"].max()) & df["liquid"]].copy()
    latest["prob"] = model.predict_proba(latest[FEATURES])[:, 1]
    return latest.sort_values("prob", ascending=False)[["date", "stock_id", "close", "prob"]]
