from typing import Dict

import pandas as pd

from etf_factors import calculate_factor
from strategies.base import Strategy, StrategyContext, StrategyDecision


class SignalGenerator:
    def generate(self, df: pd.DataFrame, params: Dict) -> pd.Series:
        raise NotImplementedError


class RSISignal(SignalGenerator):
    def generate(self, df: pd.DataFrame, params: Dict) -> pd.Series:
        period = params.get("period", 14)
        oversold = params.get("oversold", 30)
        overbought = params.get("overbought", 70)
        rsi = calculate_factor(df, "rsi", period=period)
        signal = pd.Series(0, index=df.index)
        signal[rsi < oversold] = 1
        signal[rsi > overbought] = -1
        return signal


class MACDSignal(SignalGenerator):
    def generate(self, df: pd.DataFrame, params: Dict) -> pd.Series:
        macd_df = calculate_factor(
            df,
            "macd",
            fast=params.get("fast", 12),
            slow=params.get("slow", 26),
            signal=params.get("signal", 9),
        )
        signal = pd.Series(0, index=df.index)
        signal[macd_df["macd"] > macd_df["signal"]] = 1
        signal[macd_df["macd"] < macd_df["signal"]] = -1
        return signal


class MASignal(SignalGenerator):
    def generate(self, df: pd.DataFrame, params: Dict) -> pd.Series:
        return calculate_factor(
            df,
            "ma_cross",
            short_period=params.get("short_period", 5),
            long_period=params.get("long_period", 20),
        )


class BollingerSignal(SignalGenerator):
    def generate(self, df: pd.DataFrame, params: Dict) -> pd.Series:
        bb_df = calculate_factor(
            df,
            "bollinger_bands",
            window=params.get("window", 20),
            num_std=params.get("num_std", 2),
        )
        signal = pd.Series(0, index=df.index)
        signal[bb_df["position"] < 0.2] = 1
        signal[bb_df["position"] > 0.8] = -1
        return signal


class LegacySignalStrategy(Strategy):
    def __init__(self, name: str, signal_generator: SignalGenerator):
        self.name = name
        self.signal_generator = signal_generator

    def decide(self, ctx: StrategyContext) -> StrategyDecision:
        signal = self.signal_generator.generate(ctx.df, ctx.params).fillna(0).astype(int)
        return StrategyDecision(
            regime=pd.Series(1, index=ctx.df.index, dtype=int),
            entry=(signal == 1).astype(int),
            exit=(signal == -1).astype(int),
        )
