"""Run-rate forecasting and cash runway."""

from decimal import Decimal


def next_period(period: str) -> str:
    year, month = map(int, period.split("-"))
    year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return f"{year:04d}-{month:02d}"


def forecast(history: dict[str, Decimal], months: int, lookback: int = 3) -> dict[str, Decimal]:
    """Project a monthly series forward using its average month-over-month growth.

    Growth is averaged over the last `lookback` changes. Periods with a zero
    base are skipped when computing growth; with no usable growth the series
    is held flat at its last value.
    """
    if not history:
        raise ValueError("history is empty")
    periods = sorted(history)
    values = [history[p] for p in periods]
    window = values[-(lookback + 1) :]
    rates = [(cur - prev) / prev for prev, cur in zip(window, window[1:]) if prev != 0]
    growth = sum(rates, Decimal(0)) / len(rates) if rates else Decimal(0)

    result = {}
    period, value = periods[-1], values[-1]
    for _ in range(months):
        period = next_period(period)
        value = value * (1 + growth)
        result[period] = value.quantize(Decimal("0.01"))
    return result


def runway_months(cash: Decimal, monthly_net_burn: Decimal) -> Decimal | None:
    """Months of cash left at the given net burn; None if not burning cash."""
    if monthly_net_burn <= 0:
        return None
    return (cash / monthly_net_burn).quantize(Decimal("0.1"))
