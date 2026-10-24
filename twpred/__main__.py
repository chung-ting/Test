"""CLI。

  python -m twpred backtest --start 2017-01-01 --end 2026-10-23
  python -m twpred signal   --start 2021-01-01 --end 2026-10-23 --top 10
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from twpred import backtest, report
from twpred.config import TW50, Config
from twpred.data import load_universe
from twpred.features import build_dataset
from twpred.model import fit_latest, walk_forward


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="twpred", description="台股短期趨勢預測 MVP")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("backtest", "signal"):
        p = sub.add_parser(name)
        p.add_argument("--start", required=True, help="資料起日 YYYY-MM-DD")
        p.add_argument("--end", required=True, help="資料迄日 YYYY-MM-DD")
        p.add_argument("--stocks", help="逗號分隔的股票代號，預設台灣 50")
        p.add_argument("--horizon", type=int, default=5)
        p.add_argument("--target", type=float, default=0.02, help="訓練標籤門檻（扣成本後報酬）")
        p.add_argument("--fee-discount", type=float, default=1.0, help="手續費折數，例如 0.28 = 2.8 折")
        p.add_argument("--refresh", action="store_true", help="忽略快取重新下載")
    bt = sub.choices["backtest"]
    bt.add_argument("--top", type=int, default=3, help="投資組合模擬：每期買前幾名")
    bt.add_argument("--min-prob", type=float, default=0.5, help="投資組合模擬：機率門檻")
    bt.add_argument("--out", default="reports", help="報告輸出資料夾")
    sub.choices["signal"].add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    cfg = Config(horizon=args.horizon, target_return=args.target)
    cfg = replace(cfg, fee_rate=cfg.fee_rate * args.fee_discount)
    stocks = args.stocks.split(",") if args.stocks else TW50

    data = load_universe(stocks, args.start, args.end, args.refresh)
    df = build_dataset(data, cfg)

    if args.cmd == "signal":
        latest = fit_latest(df, cfg).head(args.top)
        print(f"\n{latest['date'].iloc[0].date()} 收盤後訊號（下一個交易日開盤進場，持有 {cfg.horizon} 天）\n")
        print(latest[["stock_id", "close", "prob"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        return 0

    preds, importance = walk_forward(df, cfg)
    rule = f"每日前 {args.top} 名且機率 ≥ {args.min_prob:.2f}"
    picks = backtest.daily_top_k(preds, args.top)
    picks = picks[picks["prob"] >= args.min_prob]
    curve, summary = backtest.portfolio(preds, data["market"], cfg, args.top, args.min_prob)
    text = report.build(
        cfg,
        preds,
        backtest.threshold_curve(preds),
        backtest.auc_by_year(preds),
        backtest.yearly(picks, backtest.tradable_rows(preds)),
        summary,
        rule,
        importance,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text + "\n")
    preds.to_csv(out / "predictions.csv", index=False)
    curve.to_csv(out / "equity_curve.csv")
    print(text)
    print(f"\n報告已存到 {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
