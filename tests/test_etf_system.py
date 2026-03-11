import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from etf_system import ETFAnalysisSystem, format_relative_strength, format_rotation_ranking
from etf_data import ETFCodeMapper


class ETFSystemTests(unittest.TestCase):
    def _price_frame(self, rows=80):
        closes = list(range(100, 100 + rows))
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000] * rows,
        })

    def test_get_investment_advice_keeps_partial_factors_when_one_fails(self):
        system = ETFAnalysisSystem()
        df = self._price_frame()
        backtest_result = MagicMock(total_return=0.1, sharpe_ratio=1.0, max_drawdown=-0.1, win_rate=0.5)

        values = {
            "rsi": pd.Series([55.0]),
            "trend_strength": pd.Series([0.03]),
            "ma_alignment": pd.Series([0.8]),
            "supertrend": pd.DataFrame({"trend": [1]}),
        }

        def fake_calculate(_df, factor_name, **params):
            if factor_name == "adx":
                raise ValueError("boom")
            return values[factor_name]

        with patch.object(system.data_fetcher, "get_realtime_quote", return_value={"latest_price": 100.0}), \
             patch.object(system.data_fetcher, "get_historical_data", return_value=df), \
             patch.object(system.factor_calculator, "calculate", side_effect=fake_calculate), \
             patch("etf_system.get_investment_advice") as mock_advice:
            system.get_investment_advice("510300", "macd", backtest_result)

        self.assertTrue(mock_advice.called)
        factors = mock_advice.call_args[0][4]
        self.assertEqual(factors["rsi"], 55.0)
        self.assertEqual(factors["trend"], 0.03)
        self.assertEqual(factors["ma_alignment"], 0.8)
        self.assertEqual(factors["supertrend_trend"], 1)
        self.assertNotIn("adx", factors)

    def test_format_relative_strength_contains_rotation_and_benchmark_lines(self):
        text = format_relative_strength({
            "rotation": {"phase": "弱转强", "reason": "短期超额回升"},
            "rotation_advice": {"action": "试错", "reason": "适合小仓位观察"},
            "window_summary": {"120": {"label": "偏强", "avg_excess": 3.2}},
            "benchmarks": [
                {"name": "沪深300ETF", "code": "510300", "status": "强于指数", "excess_return_20": 2.5, "excess_return_60": 4.0, "excess_return_120": 6.0},
            ],
        }, rs_window=120)

        self.assertIn("轮动阶段: 弱转强", text)
        self.assertIn("建议: 试错", text)
        self.assertIn("120日窗口: 偏强", text)
        self.assertIn("沪深300ETF(510300): 强于指数", text)
        self.assertIn("120日超额 +6.00%", text)

    def test_print_rotation_only_uses_selected_window(self):
        system = ETFAnalysisSystem()
        relative = {
            "rotation": {"phase": "持续走强", "reason": "test"},
            "rotation_advice": {"action": "持有", "reason": "test"},
            "window_summary": {"250": {"label": "偏强", "avg_excess": 5.0}},
            "benchmarks": [],
        }

        with patch("etf_system.ETFCodeMapper.get_etf_name", return_value="测试ETF"), \
             patch("etf_system.analyze_relative_strength", return_value=relative), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            system.print_rotation_only("510300", rs_window=250)

        output = mock_stdout.getvalue()
        self.assertIn("相对强弱窗口: 250 日", output)
        self.assertIn("250日窗口: 偏强", output)

    def test_format_rotation_ranking_orders_items_for_cli_output(self):
        text = format_rotation_ranking([
            {"code": "512880", "name": "证券ETF", "phase": "持续走强", "action": "持有", "avg_excess": 4.2},
            {"code": "515220", "name": "煤炭ETF", "phase": "弱转强", "action": "试错", "avg_excess": 2.0},
        ], rs_window=60, category="industry")

        self.assertIn("行业ETF轮动排序 (60日窗口)", text)
        self.assertIn("512880 证券ETF | 持续走强 | 持有 | 60日均超额 +4.20%", text)

    def test_export_metadata_table_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "etf_metadata.csv")
            rows = ETFCodeMapper.export_metadata_table(path, category="industry", refresh=False)

            self.assertGreater(rows, 0)
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as fp:
                content = fp.read()
            self.assertIn("code,name,category,category_label,sector,tags", content)
            self.assertIn("512880", content)
            self.assertIn("industry", content)


if __name__ == "__main__":
    unittest.main()
