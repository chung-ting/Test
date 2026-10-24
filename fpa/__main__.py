"""CLI: python -m fpa --budget data/budget.csv --actuals data/actuals.csv"""

import argparse
import sys
from decimal import Decimal

from fpa.ledger import load_lines
from fpa.report import build_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fpa", description="Budget vs. actual FP&A report")
    parser.add_argument("--budget", required=True, help="budget CSV")
    parser.add_argument("--actuals", required=True, help="actuals CSV")
    parser.add_argument("--abs-threshold", type=Decimal, default=Decimal(10000), help="materiality, in currency")
    parser.add_argument("--pct-threshold", type=Decimal, default=Decimal(10), help="materiality, in percent")
    parser.add_argument("--cash", type=Decimal, help="current cash balance, enables runway")
    parser.add_argument("--forecast-months", type=int, default=6)
    parser.add_argument("-o", "--output", help="write report here instead of stdout")
    args = parser.parse_args(argv)

    try:
        report = build_report(
            load_lines(args.budget),
            load_lines(args.actuals),
            abs_threshold=args.abs_threshold,
            pct_threshold=args.pct_threshold,
            cash=args.cash,
            forecast_months=args.forecast_months,
        )
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.output:
        with open(args.output, "w") as f:
            f.write(report + "\n")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
