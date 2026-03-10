"""
ETF Factors Module - 回测因子库
适配中线策略（持股周期 3 周～3 个月），以趋势类因子为核心
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


# ── 收益类因子 ────────────────────────────────────────────────────────────────

class ReturnFactors:
    @staticmethod
    def cumulative_return(df: pd.DataFrame, periods: int = 1) -> pd.Series:
        """N 期累计收益率"""
        return df['close'].pct_change(periods)

    @staticmethod
    def annualized_return(df: pd.DataFrame, periods: int = 252) -> pd.Series:
        """滚动年化收益率"""
        daily_return = df['close'].pct_change()
        return daily_return.rolling(window=periods).mean() * 252

    @staticmethod
    def monthly_return(df: pd.DataFrame) -> pd.Series:
        """月收益率（21 交易日）"""
        return df['close'].pct_change(21)

    @staticmethod
    def roc(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        ROC（变化率）= (close - close_N期前) / close_N期前 * 100
        中线常用 period=20（月线动量）或 period=60（季线动量）
        """
        close = df['close']
        return (close - close.shift(period)) / close.shift(period) * 100


# ── 风险类因子 ────────────────────────────────────────────────────────────────

class RiskFactors:
    @staticmethod
    def volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """年化波动率"""
        daily_return = df['close'].pct_change()
        return daily_return.rolling(window=window).std() * np.sqrt(252)

    @staticmethod
    def max_drawdown(df: pd.DataFrame, window: int = 252) -> pd.Series:
        """滚动最大回撤"""
        close = df['close']
        rolling_max = close.rolling(window=window, min_periods=1).max()
        drawdown = (close - rolling_max) / rolling_max
        return drawdown.rolling(window=window).min()

    @staticmethod
    def sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.03, window: int = 252) -> pd.Series:
        """夏普比率"""
        daily_return = df['close'].pct_change()
        excess_return = daily_return.rolling(window=window).mean() * 252 - risk_free_rate
        volatility = daily_return.rolling(window=window).std() * np.sqrt(252)
        return excess_return / volatility.replace(0, np.nan)

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        ATR（Average True Range，平均真实波幅）
        True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        中线用途：动态止损位计算、Supertrend 基础、仓位管理
        """
        high, low, close = df['high'], df['low'], df['close']
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()


# ── 动量类因子 ────────────────────────────────────────────────────────────────

class MomentumFactors:
    @staticmethod
    def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        RSI（相对强弱指数）
        中线建议 period=21 或 42，降低短线噪音
        """
        close = df['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD（异同移动平均线）"""
        close = df['close']
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line,
        })

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """布林带（中线建议 window=60）"""
        close = df['close']
        middle = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        upper = middle + std * num_std
        lower = middle - std * num_std
        bb_position = (close - lower) / (upper - lower).replace(0, np.nan)
        return pd.DataFrame({
            'middle': middle,
            'upper': upper,
            'lower': lower,
            'position': bb_position,
        })


# ── 趋势类因子（中线核心） ────────────────────────────────────────────────────

class TrendFactors:
    @staticmethod
    def ma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """简单移动平均"""
        return df['close'].rolling(window=period).mean()

    @staticmethod
    def ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """指数移动平均"""
        return df['close'].ewm(span=period, adjust=False).mean()

    @staticmethod
    def dema(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        DEMA（双重指数移动平均）= 2*EMA - EMA(EMA)
        减少 EMA 滞后，中线趋势更灵敏
        """
        ema1 = df['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        return 2 * ema1 - ema2

    @staticmethod
    def ma_cross(df: pd.DataFrame, short_period: int = 9, long_period: int = 21) -> pd.Series:
        """均线交叉信号（+1 多头 / -1 空头）"""
        short_ma = TrendFactors.ma(df, short_period)
        long_ma = TrendFactors.ma(df, long_period)
        signal = pd.Series(0, index=df.index)
        signal[short_ma > long_ma] = 1
        signal[short_ma < long_ma] = -1
        return signal

    @staticmethod
    def ma_alignment(df: pd.DataFrame,
                     periods: List[int] = None) -> pd.Series:
        """
        均线多空排列得分（中线核心信号）
        检查多条均线是否形成多头/空头排列
        periods 默认 [9, 21, 50]（短/中/长期均线）
        返回 -1.0 ~ +1.0：
          +1.0  完全多头排列（短均线全部在长均线上方）
          -1.0  完全空头排列
           0    均线交叉混乱，无明显趋势
        """
        if periods is None:
            periods = [9, 21, 50]
        mas = [df['close'].rolling(p).mean() for p in periods]
        n = len(mas)
        pairs = n * (n - 1) // 2
        score = pd.Series(0.0, index=df.index)
        for i in range(n):
            for j in range(i + 1, n):
                score += (mas[i] > mas[j]).astype(float)
        return (score / pairs) * 2 - 1  # 归一化到 [-1, 1]

    @staticmethod
    def trend_strength(df: pd.DataFrame, window: int = 21) -> pd.Series:
        """趋势强度（价格偏离均线幅度）"""
        close = df['close']
        ma = close.rolling(window=window).mean()
        return (close - ma) / ma

    @staticmethod
    def linear_regression_slope(df: pd.DataFrame, window: int = 21) -> pd.Series:
        """
        线性回归斜率（趋势角度）
        对收盘价做 N 日滚动线性回归，取斜率再除以均价标准化
        > 0 向上趋势，< 0 向下趋势，绝对值越大趋势越陡
        中线建议 window=20（月线）或 window=60（季线）
        """
        close = df['close']
        slopes = pd.Series(np.nan, index=close.index)
        x = np.arange(window)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        for i in range(window - 1, len(close)):
            y = close.iloc[i - window + 1: i + 1].values
            y_mean = y.mean()
            slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
            slopes.iloc[i] = slope / y_mean  # 标准化：斜率/均价
        return slopes

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        ADX（平均趋向指数）+ DI+、DI-
        ADX > 25：趋势成立，值越大趋势越强
        ADX < 20：震荡行情，趋势策略不适用
        中线建议 period=14 或 21
        返回 DataFrame：adx、di_plus、di_minus
        """
        high, low, close = df['high'], df['low'], df['close']
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        # True Range
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        # 方向移动
        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)

        atr_s = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx_val = dx.ewm(span=period, adjust=False).mean()

        return pd.DataFrame({
            'adx': adx_val,
            'di_plus': plus_di,
            'di_minus': minus_di,
        })

    @staticmethod
    def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """
        Supertrend（ATR 趋势追踪线）
        中线最实用的趋势跟踪指标之一：
          trend = +1  价格在上轨上方 → 多头持有
          trend = -1  价格在下轨下方 → 空头/离场
        中线建议 period=10, multiplier=3（或 period=14, multiplier=2）
        返回 DataFrame：supertrend、trend（+1/-1）、upper_band、lower_band
        """
        high, low, close = df['high'], df['low'], df['close']
        hl2 = (high + low) / 2

        # ATR
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        supertrend = pd.Series(np.nan, index=close.index)
        trend = pd.Series(0, index=close.index)
        final_upper = upper_band.copy()
        final_lower = lower_band.copy()

        for i in range(1, len(close)):
            # 上轨
            if upper_band.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
                final_upper.iloc[i] = upper_band.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i - 1]
            # 下轨
            if lower_band.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
                final_lower.iloc[i] = lower_band.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i - 1]
            # 趋势方向
            if close.iloc[i] > final_upper.iloc[i - 1]:
                trend.iloc[i] = 1
            elif close.iloc[i] < final_lower.iloc[i - 1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i - 1]
            supertrend.iloc[i] = final_lower.iloc[i] if trend.iloc[i] == 1 else final_upper.iloc[i]

        return pd.DataFrame({
            'supertrend': supertrend,
            'trend': trend,
            'upper_band': final_upper,
            'lower_band': final_lower,
        })

    @staticmethod
    def donchian_channel(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Donchian 通道（N 日最高价/最低价通道）
        突破上轨 → 中线做多信号；跌破下轨 → 离场/做空信号
        中线建议 window=20（月线通道）或 window=60（季线通道）
        返回 DataFrame：upper、lower、middle、position(-1~1)
        """
        high, low, close = df['high'], df['low'], df['close']
        upper = high.rolling(window=window).max()
        lower = low.rolling(window=window).min()
        middle = (upper + lower) / 2
        channel_range = (upper - lower).replace(0, np.nan)
        position = (close - lower) / channel_range * 2 - 1  # -1 ~ +1
        return pd.DataFrame({
            'upper': upper,
            'lower': lower,
            'middle': middle,
            'position': position,
        })

    @staticmethod
    def keltner_channel(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10,
                        multiplier: float = 2.0) -> pd.DataFrame:
        """
        Keltner 通道（EMA ± ATR 倍数）
        比布林带更平滑，适合中线趋势跟踪
        突破上轨：强势信号；在中轨上方：多头趋势
        返回 DataFrame：upper、middle、lower、position(-1~1)
        """
        close = df['close']
        high, low = df['high'], df['low']
        middle = close.ewm(span=ema_period, adjust=False).mean()
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=atr_period, adjust=False).mean()
        upper = middle + multiplier * atr
        lower = middle - multiplier * atr
        channel_range = (upper - lower).replace(0, np.nan)
        position = (close - lower) / channel_range * 2 - 1
        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'position': position,
        })

    @staticmethod
    def aroon(df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
        """
        Aroon 指标（趋势年龄）
        Aroon Up   = (period - 距上一个最高点的天数) / period * 100
        Aroon Down = (period - 距上一个最低点的天数) / period * 100
        Aroon Up > 70 & Down < 30：上升趋势成立
        Aroon Oscillator = Up - Down（> 0 多头，< 0 空头）
        中线建议 period=25（约 5 周）
        """
        high, low = df['high'], df['low']
        aroon_up = pd.Series(np.nan, index=df.index)
        aroon_down = pd.Series(np.nan, index=df.index)

        for i in range(period, len(df)):
            window_high = high.iloc[i - period: i + 1]
            window_low = low.iloc[i - period: i + 1]
            days_since_high = period - window_high.values[::-1].argmax()
            days_since_low = period - window_low.values[::-1].argmin()
            aroon_up.iloc[i] = (period - days_since_high) / period * 100
            aroon_down.iloc[i] = (period - days_since_low) / period * 100

        return pd.DataFrame({
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'oscillator': aroon_up - aroon_down,
        })

    @staticmethod
    def ichimoku(df: pd.DataFrame,
                 tenkan: int = 9, kijun: int = 26,
                 senkou_b: int = 52, displacement: int = 26) -> pd.DataFrame:
        """
        一目均衡表（Ichimoku Cloud）
        中线最全面的趋势系统：
          tenkan_sen（转换线，9日）：短期趋势
          kijun_sen（基准线，26日）：中期趋势支撑/阻力
          senkou_span_a（先行A，向前位移26日）：云层上沿
          senkou_span_b（先行B，52日，向前位移26日）：云层下沿
          chikou_span（迟行线，收盘价向后位移26日）：确认信号
        价格在云层上方 = 强势多头区间
        tenkan 上穿 kijun = 金叉（买入）
        中线建议使用默认参数
        """
        high, low, close = df['high'], df['low'], df['close']

        def mid(h, l, p): return (h.rolling(p).max() + l.rolling(p).min()) / 2

        tenkan_sen = mid(high, low, tenkan)
        kijun_sen = mid(high, low, kijun)
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        senkou_b = mid(high, low, senkou_b).shift(displacement)
        chikou = close.shift(-displacement)

        # 云层厚度（正 = 多头云，负 = 空头云）
        cloud_thickness = senkou_a - senkou_b
        # 价格相对云层位置（+1 在云上，-1 在云下，0 在云中）
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
        price_vs_cloud = pd.Series(0, index=close.index)
        price_vs_cloud[close > cloud_top] = 1
        price_vs_cloud[close < cloud_bot] = -1

        return pd.DataFrame({
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_a,
            'senkou_span_b': senkou_b,
            'chikou_span': chikou,
            'cloud_thickness': cloud_thickness,
            'price_vs_cloud': price_vs_cloud,
        })


# ── 因子计算器 ────────────────────────────────────────────────────────────────

class ETFFactorCalculator:
    def __init__(self):
        self._register_all_factors()

    def _register_all_factors(self):
        factors = {
            # 收益类
            'cumulative_return':        ReturnFactors.cumulative_return,
            'annualized_return':        ReturnFactors.annualized_return,
            'monthly_return':           ReturnFactors.monthly_return,
            'roc':                      ReturnFactors.roc,
            # 风险类
            'volatility':               RiskFactors.volatility,
            'max_drawdown':             RiskFactors.max_drawdown,
            'sharpe_ratio':             RiskFactors.sharpe_ratio,
            'atr':                      RiskFactors.atr,
            # 动量类
            'rsi':                      MomentumFactors.rsi,
            'macd':                     MomentumFactors.macd,
            'bollinger_bands':          MomentumFactors.bollinger_bands,
            # 趋势类（中线核心）
            'ma':                       TrendFactors.ma,
            'ema':                      TrendFactors.ema,
            'dema':                     TrendFactors.dema,
            'ma_cross':                 TrendFactors.ma_cross,
            'ma_alignment':             TrendFactors.ma_alignment,
            'trend_strength':           TrendFactors.trend_strength,
            'linear_regression_slope':  TrendFactors.linear_regression_slope,
            'adx':                      TrendFactors.adx,
            'supertrend':               TrendFactors.supertrend,
            'donchian_channel':         TrendFactors.donchian_channel,
            'keltner_channel':          TrendFactors.keltner_channel,
            'aroon':                    TrendFactors.aroon,
            'ichimoku':                 TrendFactors.ichimoku,
        }
        for name, func in factors.items():
            FactorRegistry.register(name, func)

    def calculate(self, df: pd.DataFrame, factor_name: str, **params):
        func = FactorRegistry.get(factor_name)
        if func is None:
            raise ValueError(f"因子 '{factor_name}' 不存在，可用: {self.get_available_factors()}")
        return func(df, **params)

    def get_available_factors(self) -> List[str]:
        return FactorRegistry.list_factors()


def calculate_factor(df: pd.DataFrame, factor_name: str, **params):
    return ETFFactorCalculator().calculate(df, factor_name, **params)
