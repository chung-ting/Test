"""Loading of budget / actuals CSVs.

Expected columns: period (YYYY-MM), account, category, amount.
Category is one of: revenue, cogs, opex. Amounts are positive numbers;
sign conventions are applied by category, not by the input.
"""

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

CATEGORIES = ("revenue", "cogs", "opex")


@dataclass(frozen=True)
class Line:
    period: str
    account: str
    category: str
    amount: Decimal


def _validate_period(period: str) -> str:
    parts = period.split("-")
    if len(parts) != 2 or len(parts[0]) != 4 or not all(p.isdigit() for p in parts):
        raise ValueError(f"period must be YYYY-MM, got {period!r}")
    if not 1 <= int(parts[1]) <= 12:
        raise ValueError(f"invalid month in period {period!r}")
    return period


def load_lines(path: str | Path) -> list[Line]:
    lines = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = {"period", "account", "category", "amount"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row_num, row in enumerate(reader, start=2):
            category = row["category"].strip().lower()
            if category not in CATEGORIES:
                raise ValueError(f"{path}:{row_num}: unknown category {row['category']!r}")
            try:
                amount = Decimal(row["amount"].strip().replace(",", ""))
            except InvalidOperation:
                raise ValueError(f"{path}:{row_num}: invalid amount {row['amount']!r}") from None
            lines.append(
                Line(
                    period=_validate_period(row["period"].strip()),
                    account=row["account"].strip(),
                    category=category,
                    amount=amount,
                )
            )
    return lines
