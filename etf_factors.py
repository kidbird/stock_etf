"""
ETF Factors Module - 回测因子库
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class FactorRegistry:
    _factors: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, func: Callable):
        cls._factors[name] = func

    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        return cls._factors.get(name)

    @classmethod
    def list_factors(cls) -> List[str]:
        return list(cls._factors.keys())


class ReturnFactors:
    @staticmethod
    def cumulative_return(df: pd.DataFrame, periods: int = 1) -> pd.Series:
        return df['close'].pct_change(periods)

    @staticmethod
    def annualized_return(df: pd.DataFrame, periods: int = 252) -> pd.Series:
        daily_return = df['close'].pct_change()
        return daily_return.rolling(window=periods).mean() * 252

    @staticmethod
    def monthly_return(df: pd.DataFrame) -> pd.Series:
        return df['close'].pct_change(21)


class RiskFactors:
    @staticmethod
    def volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        daily_return = df['close'].pct_change()
        return daily_return.rolling(window=window).std() * np.sqrt(252)

    @staticmethod
    def max_drawdown(df: pd.DataFrame, window: int = 252) -> pd.Series:
        close = df['close']
        rolling_max = close.rolling(window=window, min_periods=1).max()
        drawdown = (close - rolling_max) / rolling_max
        return drawdown.rolling(window=window).min()

    @staticmethod
    def sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.03, window: int = 252) -> pd.Series:
        daily_return = df['close'].pct_change()
        excess_return = daily_return.rolling(window=window).mean() * 252 - risk_free_rate
        volatility = daily_return.rolling(window=window).std() * np.sqrt(252)
        return excess_return / volatility.replace(0, np.nan)


class MomentumFactors:
    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        close = df['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        close = df['close']
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return pd.DataFrame({'macd': macd_line, 'signal': signal_line, 'histogram': macd_line - signal_line})

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        close = df['close']
        middle = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        bb_position = (close - lower) / (upper - lower).replace(0, np.nan)
        return pd.DataFrame({'middle': middle, 'upper': upper, 'lower': lower, 'position': bb_position})


class TrendFactors:
    @staticmethod
    def ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        return df['close'].rolling(window=period).mean()

    @staticmethod
    def ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
        return df['close'].ewm(span=period, adjust=False).mean()

    @staticmethod
    def ma_cross(df: pd.DataFrame, short_period: int = 5, long_period: int = 20) -> pd.Series:
        short_ma = TrendFactors.ma(df, short_period)
        long_ma = TrendFactors.ma(df, long_period)
        signal = pd.Series(0, index=df.index)
        signal[short_ma > long_ma] = 1
        signal[short_ma < long_ma] = -1
        return signal

    @staticmethod
    def trend_strength(df: pd.DataFrame, window: int = 20) -> pd.Series:
        close = df['close']
        ma = close.rolling(window=window).mean()
        return (close - ma) / ma


class ETFFactorCalculator:
    def __init__(self):
        self._register_all_factors()

    def _register_all_factors(self):
        factors = {
            'cumulative_return': ReturnFactors.cumulative_return,
            'annualized_return': ReturnFactors.annualized_return,
            'monthly_return': ReturnFactors.monthly_return,
            'volatility': RiskFactors.volatility,
            'max_drawdown': RiskFactors.max_drawdown,
            'sharpe_ratio': RiskFactors.sharpe_ratio,
            'rsi': MomentumFactors.rsi,
            'macd': MomentumFactors.macd,
            'bollinger_bands': MomentumFactors.bollinger_bands,
            'ma': TrendFactors.ma,
            'ema': TrendFactors.ema,
            'ma_cross': TrendFactors.ma_cross,
            'trend_strength': TrendFactors.trend_strength,
        }
        for name, func in factors.items():
            FactorRegistry.register(name, func)

    def calculate(self, df: pd.DataFrame, factor_name: str, **params):
        func = FactorRegistry.get(factor_name)
        if func is None:
            raise ValueError(f"因子 '{factor_name}' 不存在")
        return func(df, **params)

    def get_available_factors(self) -> List[str]:
        return FactorRegistry.list_factors()


def calculate_factor(df: pd.DataFrame, factor_name: str, **params):
    return ETFFactorCalculator().calculate(df, factor_name, **params)
