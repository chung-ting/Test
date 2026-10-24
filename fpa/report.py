"""Markdown report rendering."""

from decimal import Decimal

from fpa.forecast import forecast, runway_months
from fpa.ledger import Line
from fpa.variance import compute_variances, pnl_summary


def _money(x: Decimal) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.0f}"


def _pct(x: Decimal | None) -> str:
    return "n/a" if x is None else f"{x:+.1f}%"


def build_report(
    budget: list[Line],
    actuals: list[Line],
    abs_threshold: Decimal = Decimal(10000),
    pct_threshold: Decimal = Decimal(10),
    cash: Decimal | None = None,
    forecast_months: int = 6,
) -> str:
    out = ["# FP&A Report", ""]

    b_pnl, a_pnl = pnl_summary(budget), pnl_summary(actuals)
    out += ["## P&L summary (actual vs. budget)", ""]
    out += ["| Period | Revenue | Budget | Gross margin | Opex | Budget | Operating income | Budget |"]
    out += ["|---|---:|---:|---:|---:|---:|---:|---:|"]
    zero = {"revenue": Decimal(0), "opex": Decimal(0), "operating_income": Decimal(0), "gross_margin_pct": Decimal(0)}
    for period in sorted(set(b_pnl) | set(a_pnl)):
        a, b = a_pnl.get(period, zero), b_pnl.get(period, zero)
        out.append(
            f"| {period} | {_money(a['revenue'])} | {_money(b['revenue'])} | "
            f"{a['gross_margin_pct']:.1f}% | {_money(a['opex'])} | {_money(b['opex'])} | "
            f"{_money(a['operating_income'])} | {_money(b['operating_income'])} |"
        )
    out.append("")

    variances = compute_variances(budget, actuals)
    material = [v for v in variances if v.is_material(abs_threshold, pct_threshold)]
    material.sort(key=lambda v: (v.period, -abs(v.variance)))
    out += [
        f"## Material variances (>= {_money(abs_threshold)} or >= {pct_threshold}%)",
        "",
    ]
    if material:
        out += ["| Period | Account | Budget | Actual | Variance | % | |", "|---|---|---:|---:|---:|---:|---|"]
        for v in material:
            flag = "Favorable" if v.favorable else "**Unfavorable**"
            out.append(
                f"| {v.period} | {v.account} | {_money(v.budget)} | {_money(v.actual)} | "
                f"{_money(v.variance)} | {_pct(v.variance_pct)} | {flag} |"
            )
    else:
        out.append("No material variances.")
    out.append("")

    if a_pnl:
        revenue_fc = forecast({p: v["revenue"] for p, v in a_pnl.items()}, forecast_months)
        costs_fc = forecast({p: v["cogs"] + v["opex"] for p, v in a_pnl.items()}, forecast_months)
        out += [f"## Run-rate forecast (next {forecast_months} months)", ""]
        out += ["| Period | Revenue | Total costs | Net |", "|---|---:|---:|---:|"]
        for period in revenue_fc:
            net = revenue_fc[period] - costs_fc[period]
            out.append(f"| {period} | {_money(revenue_fc[period])} | {_money(costs_fc[period])} | {_money(net)} |")
        out.append("")

        if cash is not None:
            last = a_pnl[max(a_pnl)]
            burn = -last["operating_income"]
            months = runway_months(cash, burn)
            out += ["## Cash runway", ""]
            if months is None:
                out.append(f"Cash {_money(cash)}; latest month is operating-income positive, so no burn.")
            else:
                out.append(f"Cash {_money(cash)} at latest monthly burn of {_money(burn)}: **{months} months** of runway.")
            out.append("")

    return "\n".join(out)
