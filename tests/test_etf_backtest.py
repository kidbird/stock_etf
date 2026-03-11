import unittest
from unittest.mock import patch

import pandas as pd

from etf_backtest import ETFBacktestEngine
from strategies.base import Strategy, StrategyContext, StrategyDecision


class ETFBacktestEngineTests(unittest.TestCase):
    def _price_frame(self, closes):
        return pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000] * len(closes),
        })

    def test_signals_execute_on_next_bar(self):
        closes = list(range(100, 125))
        df = self._price_frame(closes)
        engine = ETFBacktestEngine()

        class StubStrategy(Strategy):
            name = "stub"

            def decide(self, ctx):
                signal = pd.Series(0, index=ctx.df.index)
                signal.iloc[20] = 1
                signal.iloc[22] = -1
                return StrategyDecision(
                    regime=pd.Series(1, index=ctx.df.index, dtype=int),
                    entry=(signal == 1).astype(int),
                    exit=(signal == -1).astype(int),
                )

        engine.strategies["stub"] = StubStrategy()
        result = engine.run(df, "stub", {})

        self.assertEqual(len(result.trades), 2)
        self.assertEqual(result.trades[0]["action"], "BUY")
        self.assertEqual(result.trades[0]["date"], 21)
        self.assertEqual(result.trades[0]["price"], closes[21])
        self.assertEqual(result.trades[1]["action"], "SELL")
        self.assertEqual(result.trades[1]["date"], 23)
        self.assertEqual(result.trades[1]["price"], closes[23])

    def test_open_position_is_force_closed_at_end(self):
        closes = list(range(100, 125))
        df = self._price_frame(closes)
        engine = ETFBacktestEngine()

        class HoldStrategy(Strategy):
            name = "hold"

            def decide(self, ctx):
                signal = pd.Series(0, index=ctx.df.index)
                signal.iloc[20] = 1
                return StrategyDecision(
                    regime=pd.Series(1, index=ctx.df.index, dtype=int),
                    entry=(signal == 1).astype(int),
                    exit=(signal == -1).astype(int),
                )

        engine.strategies["hold"] = HoldStrategy()
        result = engine.run(df, "hold", {})

        self.assertEqual(len(result.trades), 2)
        buy_trade, sell_trade = result.trades
        self.assertEqual(buy_trade["action"], "BUY")
        self.assertEqual(buy_trade["date"], 21)
        self.assertEqual(sell_trade["action"], "SELL")
        self.assertEqual(sell_trade["date"], len(closes) - 1)
        self.assertTrue(sell_trade.get("forced_exit"))
        self.assertGreater(result.total_trades, 0)
        self.assertEqual(result.total_trades, 1)
        self.assertEqual(result.winning_trades, 1)

    def test_strategy_decision_interface_runs(self):
        closes = [100] * 20 + [101, 103, 104, 102, 105]
        df = self._price_frame(closes)
        engine = ETFBacktestEngine()

        class DecisionStrategy(Strategy):
            name = "decision_strategy"

            def prepare(self, df, params):
                return StrategyContext(df=df, params=params)

            def decide(self, ctx):
                regime = pd.Series(0, index=ctx.df.index, dtype=int)
                regime.iloc[20:] = 1
                entry = pd.Series(0, index=ctx.df.index, dtype=int)
                exit_ = pd.Series(0, index=ctx.df.index, dtype=int)
                entry.iloc[20] = 1
                exit_.iloc[23] = 1
                return StrategyDecision(regime=regime, entry=entry, exit=exit_)

        engine.strategies["decision_strategy"] = DecisionStrategy()
        result = engine.run(df, "decision_strategy", {})

        self.assertEqual(len(result.trades), 2)
        self.assertEqual(result.trades[0]["date"], 21)
        self.assertEqual(result.trades[1]["date"], 24)
        self.assertEqual(result.total_trades, 1)

    def test_trend_filter_macd_opens_when_trend_filter_passes(self):
        closes = [100] * 20 + [101, 103, 105, 106, 104]
        df = self._price_frame(closes)
        engine = ETFBacktestEngine()

        macd_df = pd.DataFrame({
            "macd": [0] * 20 + [1, 1, 1, -1, -1],
            "signal": [0] * 20 + [0, 0, 0, 0, 0],
        })
        adx_df = pd.DataFrame({
            "adx": [25] * len(df),
            "di_plus": [30] * len(df),
            "di_minus": [10] * len(df),
        })
        ma_align = pd.Series([0.8] * len(df), index=df.index)

        with patch("strategies.trend.calculate_factor", side_effect=[macd_df, adx_df, ma_align]):
            result = engine.run(df, "trend_filter_macd", {})

        self.assertEqual(len(result.trades), 2)
        self.assertEqual(result.trades[0]["action"], "BUY")
        self.assertEqual(result.trades[1]["action"], "SELL")

    def test_supertrend_follow_stays_flat_when_momentum_filter_fails(self):
        closes = [100] * 20 + [101, 103, 105, 106, 104]
        df = self._price_frame(closes)
        engine = ETFBacktestEngine()

        st_df = pd.DataFrame({
            "supertrend": [99] * len(df),
            "trend": [0] * 20 + [1, 1, 1, -1, -1],
            "upper_band": [101] * len(df),
            "lower_band": [99] * len(df),
        })
        roc = pd.Series([-1] * len(df), index=df.index)

        with patch("strategies.trend.calculate_factor", side_effect=[st_df, roc]):
            result = engine.run(df, "supertrend_follow", {})

        self.assertEqual(len(result.trades), 0)
        self.assertEqual(result.total_trades, 0)

    def test_donchian_breakout_requires_breakout_and_trend(self):
        closes = [100] * 20 + [101, 102, 106, 107, 103]
        df = self._price_frame(closes)
        engine = ETFBacktestEngine()

        dc_df = pd.DataFrame({
            "upper": [102] * len(df),
            "lower": [98] * len(df),
            "middle": [100] * len(df),
            "position": [0] * len(df),
        })
        adx_df = pd.DataFrame({
            "adx": [25] * len(df),
            "di_plus": [30] * len(df),
            "di_minus": [10] * len(df),
        })

        with patch("strategies.breakout.calculate_factor", side_effect=[dc_df, adx_df]):
            result = engine.run(df, "donchian_breakout", {})

        self.assertGreaterEqual(len(result.trades), 2)
        self.assertEqual(result.trades[0]["action"], "BUY")


if __name__ == "__main__":
    unittest.main()
