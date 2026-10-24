"""Budget vs. actual variance analysis."""

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal

from fpa.ledger import Line


@dataclass(frozen=True)
class Variance:
    period: str
    account: str
    category: str
    budget: Decimal
    actual: Decimal

    @property
    def variance(self) -> Decimal:
        return self.actual - self.budget

    @property
    def variance_pct(self) -> Decimal | None:
        if self.budget == 0:
            return None
        return self.variance / self.budget * 100

    @property
    def favorable(self) -> bool:
        # Over budget is good for revenue, bad for costs.
        if self.category == "revenue":
            return self.variance >= 0
        return self.variance <= 0

    def is_material(self, abs_threshold: Decimal, pct_threshold: Decimal) -> bool:
        """Material if it breaches either the absolute or the percentage threshold."""
        if abs(self.variance) >= abs_threshold:
            return True
        pct = self.variance_pct
        return pct is not None and abs(pct) >= pct_threshold


def _totals(lines: list[Line]) -> dict[tuple[str, str, str], Decimal]:
    totals: dict[tuple[str, str, str], Decimal] = defaultdict(Decimal)
    for line in lines:
        totals[(line.period, line.account, line.category)] += line.amount
    return totals


def compute_variances(budget: list[Line], actuals: list[Line]) -> list[Variance]:
    """One Variance per (period, account), including accounts present on only one side."""
    b, a = _totals(budget), _totals(actuals)
    keys = sorted(set(b) | set(a))
    seen: dict[tuple[str, str], str] = {}
    for period, account, category in keys:
        prev = seen.setdefault((period, account), category)
        if prev != category:
            raise ValueError(
                f"account {account!r} in {period} has conflicting categories {prev!r} and {category!r}"
            )
    return [Variance(*key, b.get(key, Decimal(0)), a.get(key, Decimal(0))) for key in keys]


def pnl_summary(lines: list[Line]) -> dict[str, dict[str, Decimal]]:
    """Per-period P&L: revenue, cogs, gross_profit, gross_margin_pct, opex, operating_income."""
    summary: dict[str, dict[str, Decimal]] = {}
    for line in lines:
        p = summary.setdefault(line.period, {"revenue": Decimal(0), "cogs": Decimal(0), "opex": Decimal(0)})
        p[line.category] += line.amount
    for p in summary.values():
        p["gross_profit"] = p["revenue"] - p["cogs"]
        p["gross_margin_pct"] = p["gross_profit"] / p["revenue"] * 100 if p["revenue"] else Decimal(0)
        p["operating_income"] = p["gross_profit"] - p["opex"]
    return dict(sorted(summary.items()))
