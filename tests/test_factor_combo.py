import unittest
import tempfile
from unittest.mock import patch

import pandas as pd

from etf_data import ETFDataFetcher, load_history_csv
from etf_factors import ETFFactorCalculator
from etf_relative_strength import analyze_relative_strength
from factor_combo import run_combo_analysis
from stock_data import StockCodeMapper, StockDataFetcher


class FakeStorage:
    def __init__(self):
        self.df = None

    def load_prices(self, code, start_date=None, end_date=None):
        if self.df is None:
            return None
        out = self.df.copy()
        if start_date is not None:
            out = out[out["date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            out = out[out["date"] <= pd.Timestamp(end_date)]
        return out.reset_index(drop=True)

    def save_prices(self, code, df):
        self.df = df.copy()
        return len(df)

    def upsert_meta(self, code, name=""):
        return None


class FakeFetcher:
    def __init__(self, mapping):
        self.mapping = mapping

    def get_historical_data(self, code, days=320):
        return self.mapping.get(code)


class FactorComboTests(unittest.TestCase):
    def _frame(self, rows=300, start=100.0, step=1.0):
        closes = [start + i * step for i in range(rows)]
        return pd.DataFrame(
            {
                "date": pd.date_range(end=pd.Timestamp.today().normalize(), periods=rows, freq="D"),
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [1000 + i for i in range(rows)],
            }
        )

    def test_etf_history_returns_requested_trading_rows_without_cache(self):
        fetcher = ETFDataFetcher()
        df = self._frame(rows=260)
        with patch.object(fetcher, "_fetch_history_network", return_value=df):
            result = fetcher.get_historical_data("510300", days=120, use_cache=False)
        self.assertEqual(len(result), 120)
        self.assertEqual(result["date"].iloc[0], df["date"].iloc[-120])

    def test_stock_history_returns_requested_trading_rows(self):
        fetcher = StockDataFetcher()
        fetcher.storage = FakeStorage()
        df = self._frame(rows=260)
        with patch.object(fetcher, "_fetch_from_akshare", return_value=df):
            result = fetcher.get_historical_data("000001", days=100)
        self.assertEqual(len(result), 100)
        self.assertEqual(result["date"].iloc[0], df["date"].iloc[-100])

    def test_relative_strength_includes_250_day_window(self):
        target = self._frame(rows=320, start=100.0, step=1.5)[["date", "close"]]
        benchmark = self._frame(rows=320, start=100.0, step=0.8)[["date", "close"]]
        fetcher = FakeFetcher(
            {
                "512880": target,
                "510300": benchmark,
                "510500": benchmark,
                "512100": benchmark,
            }
        )
        result = analyze_relative_strength(fetcher, "512880", etf_name="证券ETF", days=320)
        self.assertIn("250", result["window_summary"])
        self.assertIsNotNone(result["window_summary"]["250"]["avg_excess"])
        self.assertIn("excess_return_250", result["benchmarks"][0])

    def test_stock_static_fallback_codes_are_available(self):
        self.assertEqual(StockCodeMapper.get_stock_name("000001"), "平安银行")
        self.assertEqual(StockCodeMapper.get_stock_industry("000001"), "银行")

    def test_etf_combo_analysis_returns_score_and_backtest(self):
        df = self._frame(rows=260)
        benchmark = self._frame(rows=260, start=100.0, step=0.6)
        result = run_combo_analysis(
            "etf",
            "510300",
            df,
            metadata={"fund_size": 8_000_000_000},
            combo_name="etf_trend_size",
            benchmark_df=benchmark,
        )
        self.assertIn("composite_score", result)
        self.assertIn("backtest", result)
        self.assertIn("fund_size_percentile", result["factor_values"])
        self.assertIn("market_fear", result["factor_values"])

    def test_fresh_52w_breakout_scores_recent_breakout_higher_than_stale_one(self):
        calc = ETFFactorCalculator()
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=260, freq="D")
        stale_close = [100.0] * 220 + [110.0] * 40
        fresh_close = [100.0] * 250 + [100.0, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.5, 101.8, 102.0]
        stale_df = pd.DataFrame(
            {"date": dates, "open": stale_close, "high": stale_close, "low": stale_close, "close": stale_close, "volume": 1000}
        )
        fresh_df = pd.DataFrame(
            {"date": dates, "open": fresh_close, "high": fresh_close, "low": fresh_close, "close": fresh_close, "volume": 1000}
        )

        stale_score = calc.calculate(stale_df, "fresh_52w_breakout").iloc[-1]
        fresh_score = calc.calculate(fresh_df, "fresh_52w_breakout").iloc[-1]

        self.assertEqual(stale_score, 0.0)
        self.assertGreater(fresh_score, 0.0)

    def test_etf_fresh_high_combo_exposes_breakout_factor(self):
        base = [100.0] * 250
        closes = base + [100.0, 100.2, 100.4, 100.6, 100.8, 101.0, 101.2, 101.5, 101.8, 102.0]
        df = pd.DataFrame(
            {
                "date": pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(closes), freq="D"),
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [1000 + i for i in range(len(closes))],
            }
        )
        result = run_combo_analysis(
            "etf",
            "510300",
            df,
            metadata={"fund_size": 8_000_000_000},
            combo_name="etf_fresh_high",
        )
        self.assertIn("fresh_52w_breakout", result["factor_values"])
        self.assertGreater(result["factor_values"]["fresh_52w_breakout"], 0.0)

    def test_load_history_csv_supports_common_cn_columns(self):
        df = pd.DataFrame(
            {
                "日期": ["2026-03-10", "2026-03-11"],
                "开盘": [1.0, 1.1],
                "最高": [1.2, 1.3],
                "最低": [0.9, 1.0],
                "收盘": [1.1, 1.25],
                "成交量": [1000, 1200],
            }
        )
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as fp:
            path = fp.name
        try:
            df.to_csv(path, index=False)
            parsed = load_history_csv(path)
        finally:
            import os
            os.unlink(path)

        self.assertEqual(list(parsed.columns), ["date", "open", "high", "low", "close", "volume"])
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed["close"].iloc[-1], 1.25)

    def test_stock_combo_analysis_uses_fundamentals(self):
        df = self._frame(rows=260)
        benchmark = self._frame(rows=260, start=100.0, step=0.5)
        fundamentals = {
            "market_cap": 300_000_000_000,
            "roe": 18.0,
            "industry": "银行",
        }
        result = run_combo_analysis(
            "stock",
            "000001",
            df,
            metadata=fundamentals,
            combo_name="stock_quality_industry",
            benchmark_df=benchmark,
            fundamentals=fundamentals,
        )
        self.assertIn("market_cap_bucket", result["factor_values"])
        self.assertEqual(result["factor_values"]["industry"], "银行")
        self.assertGreater(result["composite_score"], -1.0)


if __name__ == "__main__":
    unittest.main()
