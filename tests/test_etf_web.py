import json
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from etf_advisor import InvestmentAdvice, Signal
from etf_backtest import BacktestResult
from etf_web import _build_market_regime, _build_strategy_fit, _parse_strategy_params, app


class ETFWebTests(unittest.TestCase):
    def _price_frame(self, rows=40):
        closes = list(range(100, 100 + rows))
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000] * rows,
        })

    def test_parse_strategy_params_filters_unknown_and_invalid_values(self):
        payload = {
            "fast": "12",
            "slow": "26",
            "signal": "bad",
            "ignored": 999,
        }
        with app.test_request_context(
            "/api/analysis/510300",
            query_string={"strategy": "macd", "params": json.dumps(payload)},
        ):
            params = _parse_strategy_params("macd")

        self.assertEqual(params, {"fast": 12, "slow": 26})

    def test_build_market_regime_classifies_trend(self):
        regime = _build_market_regime({
            "adx": 30.0,
            "ma_alignment": 0.8,
            "roc20": 5.0,
            "supertrend_trend": 1,
        })
        self.assertEqual(regime["label"], "趋势")
        self.assertEqual(regime["color"], "green")
        self.assertGreater(regime["confidence"], 0.5)

    def test_build_strategy_fit_matches_regime_and_strategy(self):
        fit = _build_strategy_fit("bollinger", {"label": "趋势"})
        self.assertEqual(fit["label"], "不适合")
        self.assertEqual(fit["color"], "red")

    def test_api_analysis_passes_sanitized_params_to_engine(self):
        df_main = self._price_frame(60)
        df_factor = self._price_frame(40)
        result = BacktestResult(
            total_return=0.1,
            annualized_return=0.08,
            sharpe_ratio=1.2,
            max_drawdown=-0.05,
            win_rate=0.5,
            profit_factor=1.8,
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            equity_curve=pd.DataFrame({"date": list(range(60)), "equity": [100000 + i for i in range(60)]}),
            trades=[{"date": 10, "action": "BUY", "price": 101}, {"date": 20, "action": "SELL", "price": 105, "profit": 0.03, "holding_days": 10}],
        )
        advice = InvestmentAdvice(
            signal=Signal.HOLD,
            score=50.0,
            reasons=["test"],
            risks=["risk"],
            confidence=0.5,
        )

        with app.test_client() as client, \
             patch("etf_web._fetcher.get_historical_data", side_effect=[df_main, df_factor]), \
             patch("etf_web._fetcher.get_realtime_quote", return_value={"latest_price": 100.0, "source": "test"}), \
             patch("etf_web._engine.run", return_value=result) as mock_run, \
             patch("etf_web._advisor.generate_advice", return_value=advice), \
             patch("etf_web.analyze_relative_strength", return_value={"benchmarks": [], "rotation": {"phase": "未知", "color": "yellow", "reason": "test"}, "rotation_advice": {"action": "观察", "color": "yellow", "reason": "test"}, "cyclical": False}), \
             patch("etf_web.ETFCodeMapper.get_etf_name", return_value="测试ETF"):
            response = client.get(
                "/api/analysis/510300",
                query_string={
                    "strategy": "trend_filter_macd",
                    "days": "500",
                    "params": json.dumps({
                        "fast": "8",
                        "slow": "21",
                        "signal": "5",
                        "adx_period": "14",
                        "adx_min": "22.5",
                        "ma_alignment_min": "0.4",
                        "drop_me": "x",
                    }),
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(mock_run.called)
        _, strategy_name, params = mock_run.call_args[0]
        self.assertEqual(strategy_name, "trend_filter_macd")
        self.assertEqual(params, {
            "fast": 8,
            "slow": 21,
            "signal": 5,
            "adx_period": 14,
            "adx_min": 22.5,
            "ma_alignment_min": 0.4,
        })

        data = response.get_json()
        self.assertEqual(data["params"]["fast"], 8)
        self.assertNotIn("drop_me", data["params"])
        self.assertEqual(data["metadata"]["category"], "wide_basis")
        self.assertEqual(data["metadata"]["sector"], "broad_market")
        self.assertIn("market_regime", data)
        self.assertIn("strategy_fit", data)
        self.assertIn("relative_strength", data)
        self.assertIn("rotation_advice", data["relative_strength"])

    def test_api_analysis_returns_explicit_infinite_profit_factor_flag(self):
        df_main = self._price_frame(60)
        df_factor = self._price_frame(40)
        result = BacktestResult(
            total_return=0.1,
            annualized_return=0.08,
            sharpe_ratio=1.2,
            max_drawdown=-0.05,
            win_rate=1.0,
            profit_factor=float("inf"),
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            equity_curve=pd.DataFrame({"date": list(range(60)), "equity": [100000 + i for i in range(60)]}),
            trades=[],
        )
        advice = InvestmentAdvice(
            signal=Signal.HOLD,
            score=50.0,
            reasons=["test"],
            risks=["risk"],
            confidence=0.5,
        )

        with app.test_client() as client, \
             patch("etf_web._fetcher.get_historical_data", side_effect=[df_main, df_factor]), \
             patch("etf_web._fetcher.get_realtime_quote", return_value={"latest_price": 100.0, "source": "test"}), \
             patch("etf_web._engine.run", return_value=result), \
             patch("etf_web._advisor.generate_advice", return_value=advice), \
             patch("etf_web.analyze_relative_strength", return_value={"benchmarks": [], "rotation": {"phase": "未知", "color": "yellow", "reason": "test"}, "rotation_advice": {"action": "观察", "color": "yellow", "reason": "test"}, "cyclical": False}), \
             patch("etf_web.ETFCodeMapper.get_etf_name", return_value="测试ETF"):
            response = client.get("/api/analysis/510300", query_string={"strategy": "macd", "days": "500"})

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsNone(data["backtest"]["profit_factor"])
        self.assertTrue(data["backtest"]["profit_factor_infinite"])

    def test_api_analysis_supports_etf_combo_mode(self):
        df = self._price_frame(120)
        combo_result = {
            "combo": "etf_trend_size",
            "description": "test",
            "buy_threshold": 0.3,
            "sell_threshold": -0.1,
            "composite_score": 0.45,
            "factor_values": {"ma_alignment": 0.8},
            "factor_contributions": {"ma_alignment": 0.2},
            "backtest": {
                "total_return": 0.12,
                "annualized_return": 0.08,
                "sharpe_ratio": 1.1,
                "max_drawdown": -0.06,
                "win_rate": 0.5,
                "profit_factor": 1.5,
                "total_trades": 2,
                "winning_trades": 1,
                "losing_trades": 1,
                "equity_curve": pd.DataFrame({"date": df["date"], "equity": [100000 + i for i in range(len(df))]}),
                "trades": [],
            },
        }
        with app.test_client() as client, \
             patch("etf_web._fetcher.get_historical_data", side_effect=[df, df]), \
             patch("etf_web._fetcher.get_fund_profile", return_value={"fund_size": 1e10}), \
             patch("etf_web._fetcher.get_realtime_quote", return_value={"latest_price": 100.0}), \
             patch("etf_web.run_combo_analysis", return_value=combo_result), \
             patch("etf_web.analyze_relative_strength", return_value={"benchmarks": []}), \
             patch("etf_web.ETFCodeMapper.get_etf_name", return_value="测试ETF"):
            response = client.get("/api/analysis/510300", query_string={"combo": "etf_trend_size", "days": "120"})

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["combo_result"]["combo"], "etf_trend_size")
        self.assertEqual(data["combo_result"]["factor_values"]["ma_alignment"], 0.8)

    def test_api_etfs_returns_metadata_fields(self):
        with app.test_client() as client:
            response = client.get("/api/etfs", query_string={"q": "证券", "category": "all"})

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["etfs"])
        item = next(etf for etf in data["etfs"] if etf["code"] == "512880")
        self.assertEqual(item["category"], "industry")
        self.assertEqual(item["category_label"], "行业")
        self.assertEqual(item["sector"], "financials")
        self.assertIn("broker", item["tags"])

    def test_api_metadata_supports_refresh_and_category_filter(self):
        rows = [
            {
                "code": "510300",
                "name": "沪深300ETF",
                "category": "wide_basis",
                "category_label": "宽基",
                "sector": "broad_market",
                "tags": ["hs300"],
            }
        ]
        with app.test_client() as client, \
             patch("etf_web.ETFCodeMapper.get_metadata_table", return_value=rows) as mock_table:
            response = client.get(
                "/api/metadata",
                query_string={"category": "wide_basis", "refresh": "true", "q": "300"},
            )

        self.assertEqual(response.status_code, 200)
        mock_table.assert_called_once_with(category="wide_basis", refresh=True)
        data = response.get_json()
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["items"][0]["code"], "510300")

    def test_api_stocks_prefers_live_refresh_when_live_cache_empty(self):
        with app.test_client() as client, \
             patch("etf_web.StockCodeMapper._live", {}), \
             patch("etf_web.StockCodeMapper.load_from_akshare", return_value=2) as mock_load, \
             patch("etf_web.StockCodeMapper.get_all_codes", return_value=["000001", "300059"]), \
             patch("etf_web.StockCodeMapper.get_stock_name", side_effect=["平安银行", "东方财富"]), \
             patch("etf_web.StockCodeMapper.get_stock_industry", side_effect=["银行", "证券"]):
            response = client.get("/api/stocks")

        self.assertEqual(response.status_code, 200)
        mock_load.assert_called_once()
        data = response.get_json()
        self.assertEqual(data["total"], 2)

    def test_api_stock_analysis_returns_backtest_in_non_combo_mode(self):
        df = self._price_frame(80)
        stock_backtest = MagicMock(
            total_return=0.15,
            annualized_return=0.10,
            sharpe_ratio=1.3,
            max_drawdown=-0.07,
            win_rate=0.6,
            profit_factor=1.9,
            total_trades=3,
            winning_trades=2,
            losing_trades=1,
        )
        factors = {
            "rsi": pd.Series([55.0]),
            "adx": pd.DataFrame({"adx": [23.0]}),
            "ma_alignment": pd.Series([0.6]),
        }
        with app.test_client() as client, \
             patch("etf_web._stock_fetcher.get_historical_data", side_effect=[df, self._price_frame(80)]), \
             patch("etf_web._stock_fetcher.get_realtime_quote", return_value={"latest_price": 12.3}), \
             patch("etf_web._stock_fetcher.get_stock_fundamentals", return_value={"name": "平安银行", "industry": "银行", "market_cap": 3e11, "roe": 12.5}), \
             patch("etf_web.StockBacktestEngine.run", return_value=stock_backtest), \
             patch("etf_web._stock_calculator.calculate_all_factors", return_value=factors), \
             patch("etf_web._stock_calculator.calculate", side_effect=[pd.Series([3e11]), pd.Series([2]), pd.Series([12.5]), pd.Series([18.0])]):
            response = client.get("/api/stock/analysis/000001", query_string={"strategy": "stock_macd", "days": "80"})

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["strategy"], "stock_macd")
        self.assertIn("backtest", data)
        self.assertEqual(data["backtest"]["total_trades"], 3)
        self.assertEqual(data["factors"]["industry"], "银行")

    def test_api_rotation_rank_sanitizes_inputs_and_returns_items(self):
        ranking = [
            {
                "code": "512880",
                "name": "证券ETF",
                "phase": "持续走强",
                "phase_color": "green",
                "action": "持有",
                "action_color": "green",
                "avg_excess": 3.25,
                "window_label": "强于指数",
                "phase_rank": 5,
            }
        ]

        with app.test_client() as client, \
             patch("etf_web._build_rotation_ranking", return_value=ranking) as mock_build:
            response = client.get(
                "/api/rotation-rank",
                query_string={
                    "category": "bad-category",
                    "window": "999",
                    "top": "500",
                },
            )

        self.assertEqual(response.status_code, 200)
        mock_build.assert_called_once_with(category="industry", rs_window=60, top=100)
        data = response.get_json()
        self.assertEqual(data["category"], "industry")
        self.assertEqual(data["window"], 60)
        self.assertEqual(data["top"], 100)
        self.assertEqual(data["items"][0]["code"], "512880")
        self.assertEqual(data["items"][0]["phase"], "持续走强")


if __name__ == "__main__":
    unittest.main()
