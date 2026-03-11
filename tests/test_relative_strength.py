import unittest

import pandas as pd

from etf_relative_strength import analyze_relative_strength


class FakeFetcher:
    def __init__(self, mapping):
        self.mapping = mapping

    def get_historical_data(self, code, days=180):
        return self.mapping.get(code)


class RelativeStrengthTests(unittest.TestCase):
    def _frame(self, closes):
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
            "close": closes,
        })

    def test_analyze_relative_strength_detects_outperformance_and_rotation(self):
        target = self._frame([100 + i * 1.2 for i in range(120)])
        hs300 = self._frame([100 + i * 0.6 for i in range(120)])
        zz500 = self._frame([100 + i * 0.5 for i in range(120)])
        zz1000 = self._frame([100 + i * 0.4 for i in range(120)])
        fetcher = FakeFetcher({
            "512880": target,
            "510300": hs300,
            "510500": zz500,
            "512100": zz1000,
        })

        result = analyze_relative_strength(fetcher, "512880", etf_name="证券ETF", days=120)

        self.assertTrue(result["cyclical"])
        self.assertEqual(result["rotation"]["phase"], "持续走强")
        self.assertEqual(result["rotation_advice"]["action"], "持有")
        self.assertIn("20", result["window_summary"])
        self.assertTrue(result["history"]["dates"])
        self.assertTrue(result["history"]["series"])
        self.assertTrue(any(item["status"] == "强于指数" for item in result["benchmarks"]))

    def test_analyze_relative_strength_marks_self_benchmark(self):
        target = self._frame([100 + i * 0.5 for i in range(120)])
        zz500 = self._frame([100 + i * 0.4 for i in range(120)])
        zz1000 = self._frame([100 + i * 0.3 for i in range(120)])
        fetcher = FakeFetcher({
            "510300": target,
            "510500": zz500,
            "512100": zz1000,
        })

        result = analyze_relative_strength(fetcher, "510300", etf_name="沪深300ETF", days=120)
        self_benchmark = next(item for item in result["benchmarks"] if item["code"] == "510300")

        self.assertEqual(self_benchmark["status"], "基准")
        self.assertEqual(self_benchmark["color"], "blue")


if __name__ == "__main__":
    unittest.main()
