"""
Stock Strategies Module
股票专用交易策略
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


@dataclass
class StrategyDecision:
    signals: pd.Series
    positions: pd.Series
    metadata: Dict[str, Any]


class BaseStockStrategy:
    def prepare(self, df: pd.DataFrame, params: Dict) -> Any:
        raise NotImplementedError

    def decide(self, context: Any) -> StrategyDecision:
        raise NotImplementedError


class StockRSIStrategy(BaseStockStrategy):
    def prepare(self, df: pd.DataFrame, params: Dict) -> Dict:
        period = params.get("period", 14)
        oversold = params.get("oversold", 30)
        overbought = params.get("overbought", 70)

        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return {"rsi": rsi, "oversold": oversold, "overbought": overbought}

    def decide(self, ctx: Dict) -> StrategyDecision:
        rsi = ctx["rsi"]
        oversold = ctx["oversold"]
        overbought = ctx["overbought"]

        signals = pd.Series(0, index=rsi.index)
        signals[rsi < oversold] = 1
        signals[rsi > overbought] = -1

        positions = signals.copy()

        return StrategyDecision(
            signals=signals,
            positions=positions,
            metadata={"strategy": "stock_rsi", "rsi": rsi},
        )


class StockMACDStrategy(BaseStockStrategy):
    def prepare(self, df: pd.DataFrame, params: Dict) -> Dict:
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        signal = params.get("signal", 9)

        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()

        return {"macd": macd, "signal": signal_line}

    def decide(self, ctx: Dict) -> StrategyDecision:
        macd = ctx["macd"]
        signal = ctx["signal"]

        crossover = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        crossunder = (macd < signal) & (macd.shift(1) >= signal.shift(1))

        signals = pd.Series(0, index=macd.index)
        signals[crossover] = 1
        signals[crossunder] = -1

        positions = signals.copy()

        return StrategyDecision(
            signals=signals,
            positions=positions,
            metadata={"strategy": "stock_macd", "macd": macd, "signal": signal},
        )


class StockSupertrendStrategy(BaseStockStrategy):
    def prepare(self, df: pd.DataFrame, params: Dict) -> Dict:
        period = params.get("period", 10)
        multiplier = params.get("multiplier", 3.0)

        hl_avg = (df["high"] + df["low"]) / 2

        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift())
        tr3 = abs(df["low"] - df["close"].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        upper_band = hl_avg + multiplier * atr
        lower_band = hl_avg - multiplier * atr

        trend = pd.Series(1, index=df.index)
        for i in range(1, len(df)):
            if df["close"].iloc[i] > upper_band.iloc[i - 1]:
                trend.iloc[i] = 1
            elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i - 1]

        return {"trend": trend, "atr": atr}

    def decide(self, ctx: Dict) -> StrategyDecision:
        trend = ctx["trend"]

        signals = pd.Series(0, index=trend.index)
        signal_change = trend.diff()
        signals[signal_change > 0] = 1
        signals[signal_change < 0] = -1

        positions = trend.copy()

        return StrategyDecision(
            signals=signals,
            positions=positions,
            metadata={"strategy": "stock_supertrend", "trend": trend},
        )


class StockBollingerBreakoutStrategy(BaseStockStrategy):
    def prepare(self, df: pd.DataFrame, params: Dict) -> Dict:
        window = params.get("window", 20)
        num_std = params.get("num_std", 2.0)

        ma = df["close"].rolling(window=window).mean()
        std = df["close"].rolling(window=window).std()
        upper = ma + num_std * std
        lower = ma - num_std * std

        return {"ma": ma, "upper": upper, "lower": lower, "close": df["close"]}

    def decide(self, ctx: Dict) -> StrategyDecision:
        close = ctx["close"]
        upper = ctx["upper"]
        lower = ctx["lower"]

        signals = pd.Series(0, index=close.index)
        signals[close < lower] = 1
        signals[close > upper] = -1

        positions = signals.copy()

        return StrategyDecision(
            signals=signals,
            positions=positions,
            metadata={"strategy": "stock_bollinger_breakout"},
        )


class StockVolumeBreakoutStrategy(BaseStockStrategy):
    def prepare(self, df: pd.DataFrame, params: Dict) -> Dict:
        volume_ma_period = params.get("volume_ma_period", 20)
        volume_threshold = params.get("volume_threshold", 2.0)

        volume_ma = df["volume"].rolling(window=volume_ma_period).mean()

        return {
            "volume": df["volume"],
            "volume_ma": volume_ma,
            "close": df["close"],
            "threshold": volume_threshold,
        }

    def decide(self, ctx: Dict) -> StrategyDecision:
        volume = ctx["volume"]
        volume_ma = ctx["volume_ma"]
        close = ctx["close"]
        threshold = ctx["threshold"]

        volume_ratio = volume / volume_ma
        price_change = close.pct_change()

        signals = pd.Series(0, index=close.index)
        signals[(volume_ratio > threshold) & (price_change > 0.02)] = 1
        signals[(volume_ratio > threshold) & (price_change < -0.02)] = -1

        positions = signals.copy()

        return StrategyDecision(
            signals=signals,
            positions=positions,
            metadata={
                "strategy": "stock_volume_breakout",
                "volume_ratio": volume_ratio,
            },
        )


class StockDualThrustStrategy(BaseStockStrategy):
    def prepare(self, df: pd.DataFrame, params: Dict) -> Dict:
        period = params.get("period", 20)
        k = params.get("k", 0.5)

        hh = df["high"].rolling(window=period).max()
        lc = df["close"].rolling(window=period).max()
        hc = df["low"].rolling(window=period).min()
        ll = df["low"].rolling(window=period).min()

        range_val = pd.concat([hh - lc, hc - ll], axis=1).max(axis=1)

        open_price = df["open"]
        upper = open_price + k * range_val
        lower = open_price - k * range_val

        return {"upper": upper, "lower": lower, "close": df["close"]}

    def decide(self, ctx: Dict) -> StrategyDecision:
        close = ctx["close"]
        upper = ctx["upper"]
        lower = ctx["lower"]

        signals = pd.Series(0, index=close.index)
        signals[close > upper] = 1
        signals[close < lower] = -1

        positions = signals.copy()

        return StrategyDecision(
            signals=signals,
            positions=positions,
            metadata={"strategy": "stock_dual_thrust"},
        )


def build_stock_strategy_registry() -> Dict[str, BaseStockStrategy]:
    return {
        "stock_rsi": StockRSIStrategy(),
        "stock_macd": StockMACDStrategy(),
        "stock_supertrend": StockSupertrendStrategy(),
        "stock_bollinger_breakout": StockBollingerBreakoutStrategy(),
        "stock_volume_breakout": StockVolumeBreakoutStrategy(),
        "stock_dual_thrust": StockDualThrustStrategy(),
    }
