"""
Stock Factor Calculator Module
股票专用因子计算器，包含基本面因子和技术因子
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union


class StockFactorCalculator:
    def __init__(self):
        self._factor_methods = {
            "rsi": self._rsi,
            "macd": self._macd,
            "bollinger_bands": self._bollinger_bands,
            "adx": self._adx,
            "supertrend": self._supertrend,
            "ma": self._ma,
            "ema": self._ema,
            "ma_cross": self._ma_cross,
            "ma_alignment": self._ma_alignment,
            "atr": self._atr,
            "stochastic": self._stochastic,
            "cci": self._cci,
            "williams_r": self._williams_r,
            "obv": self._obv,
            "volume_ratio": self._volume_ratio,
            "price_volume_correlation": self._price_volume_correlation,
            "high_low_range": self._high_low_range,
            "gap": self._gap,
            "pivot_points": self._pivot_points,
            "fibonacci_retracement": self._fibonacci_retracement,
            "roc": self._roc,
            "market_cap": self._market_cap,
            "market_cap_bucket": self._market_cap_bucket,
            "roe": self._roe,
            "industry": self._industry,
            "market_fear": self._market_fear,
        }

    def get_available_factors(self) -> List[str]:
        return list(self._factor_methods.keys())

    def calculate(
        self, df: pd.DataFrame, factor: str, **params
    ) -> Union[pd.DataFrame, pd.Series, float]:
        if factor not in self._factor_methods:
            raise ValueError(f"未知因子: {factor}")
        return self._factor_methods[factor](df, **params)

    def _rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _macd(
        self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return pd.DataFrame(
            {
                "macd": macd_line,
                "signal": signal_line,
                "histogram": macd_line - signal_line,
            }
        )

    def _bollinger_bands(
        self, df: pd.DataFrame, window: int = 20, num_std: float = 2.0
    ) -> pd.DataFrame:
        ma = df["close"].rolling(window=window).mean()
        std = df["close"].rolling(window=window).std()
        upper = ma + num_std * std
        lower = ma - num_std * std
        return pd.DataFrame(
            {
                "middle": ma,
                "upper": upper,
                "lower": lower,
                "position": (df["close"] - lower) / (upper - lower),
            }
        )

    def _adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return pd.DataFrame({"adx": adx, "di_plus": plus_di, "di_minus": minus_di})

    def _supertrend(
        self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
    ) -> pd.DataFrame:
        hl_avg = (df["high"] + df["low"]) / 2
        tr = self._atr(df, period)

        upper_band = hl_avg + multiplier * tr
        lower_band = hl_avg - multiplier * tr

        trend = pd.Series(1, index=df.index)
        supertrend = pd.Series(upper_band.values, index=df.index)

        for i in range(1, len(df)):
            if df["close"].iloc[i] > upper_band.iloc[i - 1]:
                trend.iloc[i] = 1
            elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i - 1]

            if trend.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]

        return pd.DataFrame({"supertrend": supertrend, "trend": trend})

    def _ma(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        return df["close"].rolling(window=period).mean()

    def _ema(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        return df["close"].ewm(span=period, adjust=False).mean()

    def _roc(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        return (df["close"] - df["close"].shift(period)) / df["close"].shift(period) * 100

    def _ma_cross(
        self, df: pd.DataFrame, short_period: int = 5, long_period: int = 20
    ) -> pd.DataFrame:
        short_ma = df["close"].rolling(window=short_period).mean()
        long_ma = df["close"].rolling(window=long_period).mean()
        return pd.DataFrame(
            {"short_ma": short_ma, "long_ma": long_ma, "cross": short_ma - long_ma}
        )

    def _ma_alignment(self, df: pd.DataFrame, periods: List[int] = None) -> pd.Series:
        if periods is None:
            periods = [5, 10, 20, 60]

        mas = {}
        for p in periods:
            mas[p] = df["close"].rolling(window=p).mean()

        scores = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)):
            aligned = 0
            for j, p in enumerate(periods[:-1]):
                if mas[p].iloc[i] > mas[periods[j + 1]].iloc[i]:
                    aligned += 1
            scores.iloc[i] = (aligned / (len(periods) - 1)) * 2 - 1

        return scores

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.rolling(window=period).mean()

    def _stochastic(
        self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> pd.DataFrame:
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()

        k = 100 * (df["close"] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()

        return pd.DataFrame({"k": k, "d": d})

    def _cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad)
        return cci

    def _williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_max = df["high"].rolling(window=period).max()
        low_min = df["low"].rolling(window=period).min()
        wr = -100 * (high_max - df["close"]) / (high_max - low_min)
        return wr

    def _obv(self, df: pd.DataFrame) -> pd.Series:
        obv = pd.Series(0, index=df.index)
        obv.iloc[1:] = np.where(
            df["close"].iloc[1:] > df["close"].iloc[:-1].values,
            df["volume"].iloc[1:],
            np.where(
                df["close"].iloc[1:] < df["close"].iloc[:-1].values,
                -df["volume"].iloc[1:],
                0,
            ),
        ).cumsum()
        return obv

    def _volume_ratio(self, df: pd.DataFrame, period: int = 5) -> pd.Series:
        return (
            df["volume"].rolling(window=period).mean()
            / df["volume"].rolling(window=period * 2).mean()
        )

    def _price_volume_correlation(
        self, df: pd.DataFrame, period: int = 20
    ) -> pd.Series:
        return df["close"].rolling(window=period).corr(df["volume"])

    def _high_low_range(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        return (df["high"] - df["low"]).rolling(window=period).mean() / df["close"]

    def _gap(self, df: pd.DataFrame) -> pd.DataFrame:
        gap = df["open"] - df["close"].shift(1)
        gap_pct = gap / df["close"].shift(1)
        return pd.DataFrame({"gap": gap, "gap_pct": gap_pct})

    def _pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        pp = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3
        r1 = 2 * pp - df["low"].shift(1)
        s1 = 2 * pp - df["high"].shift(1)
        r2 = pp + (df["high"].shift(1) - df["low"].shift(1))
        s2 = pp - (df["high"].shift(1) - df["low"].shift(1))

        return pd.DataFrame({"pivot": pp, "r1": r1, "r2": r2, "s1": s1, "s2": s2})

    def _fibonacci_retracement(
        self, df: pd.DataFrame, lookback: int = 100
    ) -> Dict[str, float]:
        high = df["high"].iloc[-lookback:].max()
        low = df["low"].iloc[-lookback:].min()
        diff = high - low

        return {
            "level_0": high,
            "level_236": high - 0.236 * diff,
            "level_382": high - 0.382 * diff,
            "level_500": high - 0.500 * diff,
            "level_618": high - 0.618 * diff,
            "level_786": high - 0.786 * diff,
            "level_100": low,
        }

    def _constant_series(self, df: pd.DataFrame, value: Optional[Any]) -> pd.Series:
        if value in ("", None):
            return pd.Series(np.nan, index=df.index, dtype=float)
        try:
            constant = float(value)
        except (TypeError, ValueError):
            return pd.Series([value] * len(df), index=df.index)
        return pd.Series(constant, index=df.index, dtype=float)

    def _market_cap(self, df: pd.DataFrame, fundamentals: Optional[Dict[str, Any]] = None) -> pd.Series:
        fundamentals = fundamentals or {}
        return self._constant_series(df, fundamentals.get("market_cap"))

    def _market_cap_bucket(self, df: pd.DataFrame, fundamentals: Optional[Dict[str, Any]] = None) -> pd.Series:
        fundamentals = fundamentals or {}
        market_cap = fundamentals.get("market_cap")
        if market_cap in ("", None):
            bucket = np.nan
        elif market_cap >= 1_000_000_000_000:
            bucket = 3
        elif market_cap >= 200_000_000_000:
            bucket = 2
        elif market_cap >= 50_000_000_000:
            bucket = 1
        else:
            bucket = 0
        return pd.Series(bucket, index=df.index, dtype=float)

    def _roe(self, df: pd.DataFrame, fundamentals: Optional[Dict[str, Any]] = None) -> pd.Series:
        fundamentals = fundamentals or {}
        return self._constant_series(df, fundamentals.get("roe"))

    def _industry(self, df: pd.DataFrame, fundamentals: Optional[Dict[str, Any]] = None) -> pd.Series:
        fundamentals = fundamentals or {}
        industry = fundamentals.get("industry")
        return pd.Series([industry] * len(df), index=df.index)

    def _market_fear(
        self, df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None, window: int = 20
    ) -> pd.Series:
        base = benchmark_df if benchmark_df is not None and len(benchmark_df) > 0 else df
        close = base["close"]
        returns = close.pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252) * 100
        drawdown = ((close / close.cummax()) - 1).rolling(window=window, min_periods=1).min().abs() * 100
        downside = returns.where(returns < 0, 0).rolling(window=window).mean().abs() * 1000
        fear = volatility.fillna(0) * 0.5 + drawdown.fillna(0) * 0.3 + downside.fillna(0) * 0.2
        if benchmark_df is not None and len(benchmark_df) > 0:
            aligned = base[["date"]].copy()
            aligned["fear"] = fear.values
            merged = df[["date"]].merge(aligned, on="date", how="left")
            return merged["fear"]
        return fear

    def calculate_all_factors(self, df: pd.DataFrame) -> Dict[str, Any]:
        factors = {}

        factors["rsi"] = self.calculate(df, "rsi", period=14)
        factors["macd"] = self.calculate(df, "macd")
        factors["bollinger"] = self.calculate(df, "bollinger_bands")
        factors["adx"] = self.calculate(df, "adx", period=14)
        factors["supertrend"] = self.calculate(
            df, "supertrend", period=10, multiplier=3.0
        )
        factors["ma_alignment"] = self.calculate(df, "ma_alignment")
        factors["atr"] = self.calculate(df, "atr", period=14)
        factors["stochastic"] = self.calculate(df, "stochastic")
        factors["cci"] = self.calculate(df, "cci", period=20)
        factors["volume_ratio"] = self.calculate(df, "volume_ratio")

        return factors
