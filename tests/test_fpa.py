import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from fpa import Line, compute_variances, forecast, load_lines, pnl_summary, runway_months
from fpa.__main__ import main
from fpa.forecast import next_period

D = Decimal
DATA = Path(__file__).resolve().parent.parent / "data"


class VarianceTests(unittest.TestCase):
    def test_favorability_depends_on_category(self):
        budget = [Line("2026-01", "Rev", "revenue", D(100)), Line("2026-01", "Rent", "opex", D(50))]
        actuals = [Line("2026-01", "Rev", "revenue", D(120)), Line("2026-01", "Rent", "opex", D(60))]
        by_account = {v.account: v for v in compute_variances(budget, actuals)}
        self.assertTrue(by_account["Rev"].favorable)
        self.assertEqual(by_account["Rev"].variance, D(20))
        self.assertFalse(by_account["Rent"].favorable)
        self.assertEqual(by_account["Rent"].variance_pct, D(20))

    def test_unbudgeted_account_is_included(self):
        variances = compute_variances([], [Line("2026-01", "Legal", "opex", D(5000))])
        self.assertEqual(len(variances), 1)
        self.assertEqual(variances[0].budget, 0)
        self.assertIsNone(variances[0].variance_pct)
        self.assertTrue(variances[0].is_material(D(1000), D(10)))

    def test_materiality_either_threshold(self):
        (v,) = compute_variances(
            [Line("2026-01", "A", "opex", D(100))], [Line("2026-01", "A", "opex", D(115))]
        )
        self.assertTrue(v.is_material(D(10000), D(10)))  # pct breach
        self.assertFalse(v.is_material(D(10000), D(20)))

    def test_conflicting_categories_rejected(self):
        with self.assertRaises(ValueError):
            compute_variances(
                [Line("2026-01", "A", "opex", D(1))], [Line("2026-01", "A", "cogs", D(1))]
            )

    def test_pnl_summary(self):
        lines = [
            Line("2026-01", "Rev", "revenue", D(1000)),
            Line("2026-01", "Host", "cogs", D(200)),
            Line("2026-01", "Pay", "opex", D(500)),
        ]
        p = pnl_summary(lines)["2026-01"]
        self.assertEqual(p["gross_profit"], D(800))
        self.assertEqual(p["gross_margin_pct"], D(80))
        self.assertEqual(p["operating_income"], D(300))


class ForecastTests(unittest.TestCase):
    def test_next_period_rolls_year(self):
        self.assertEqual(next_period("2026-12"), "2027-01")
        self.assertEqual(next_period("2026-03"), "2026-04")

    def test_growth_applied(self):
        fc = forecast({"2026-01": D(100), "2026-02": D(110), "2026-03": D(121)}, 2)
        self.assertEqual(fc, {"2026-04": D("133.10"), "2026-05": D("146.41")})

    def test_lookback_limits_window(self):
        history = {"2026-01": D(100), "2026-02": D(200), "2026-03": D(200)}
        self.assertEqual(forecast(history, 1, lookback=1), {"2026-04": D("200.00")})

    def test_single_point_held_flat(self):
        self.assertEqual(forecast({"2026-01": D(50)}, 1), {"2026-02": D("50.00")})

    def test_runway(self):
        self.assertEqual(runway_months(D(1000), D(300)), D("3.3"))
        self.assertIsNone(runway_months(D(1000), D(-5)))


class LoadAndCliTests(unittest.TestCase):
    def test_bad_category(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("period,account,category,amount\n2026-01,X,equity,1\n")
        with self.assertRaisesRegex(ValueError, "unknown category"):
            load_lines(f.name)

    def test_bad_period(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("period,account,category,amount\n2026-13,X,opex,1\n")
        with self.assertRaisesRegex(ValueError, "invalid month"):
            load_lines(f.name)

    def test_cli_on_sample_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.md"
            rc = main(["--budget", str(DATA / "budget.csv"), "--actuals", str(DATA / "actuals.csv"),
                       "--cash", "2500000", "-o", str(out)])
            self.assertEqual(rc, 0)
            text = out.read_text()
        for heading in ("P&L summary", "Material variances", "Run-rate forecast", "Cash runway"):
            self.assertIn(heading, text)


if __name__ == "__main__":
    unittest.main()
