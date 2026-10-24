# FP&A toolkit

Dependency-free Python (3.10+) tools for monthly financial planning & analysis:

- **Budget vs. actual variance** per account and period, with favorable/unfavorable
  flagging by category (revenue over budget is good; costs over budget are bad) and
  materiality thresholds (absolute **or** percentage).
- **P&L summary**: revenue, COGS, gross profit and margin, opex, operating income.
- **Run-rate forecast**: projects revenue and costs forward using the average
  month-over-month growth of the last 3 months.
- **Cash runway** from a cash balance and the latest month's burn.

## Usage

```sh
python3 -m fpa --budget data/budget.csv --actuals data/actuals.csv --cash 2500000
python3 -m fpa --budget data/budget.csv --actuals data/actuals.csv \
  --abs-threshold 25000 --pct-threshold 5 --forecast-months 12 -o report.md
```

## Input format

CSV with columns `period,account,category,amount`:

| column   | notes                                   |
|----------|-----------------------------------------|
| period   | `YYYY-MM`                               |
| account  | any name; must keep one category        |
| category | `revenue`, `cogs`, or `opex`            |
| amount   | positive number; commas allowed         |

`data/` has a sample quarter (Jul–Sep 2026) for a SaaS business.

## Tests

```sh
python3 -m unittest -v
```
