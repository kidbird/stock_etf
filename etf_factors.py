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

    @staticmethod
    def relative_strength(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None,
                          period: int = 60) -> pd.Series:
        if benchmark_df is None or len(benchmark_df) == 0:
            return pd.Series(np.nan, index=df.index)
        merged = df[["date", "close"]].merge(
            benchmark_df[["date", "close"]].rename(columns={"close": "benchmark_close"}),
            on="date",
            how="left",
        )
        ratio = merged["close"] / merged["benchmark_close"].replace(0, np.nan)
        return ratio.pct_change(period) * 100


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

    @staticmethod
    def market_fear(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None,
                    window: int = 20) -> pd.Series:
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

    @staticmethod
    def fund_size(df: pd.DataFrame, fund_size: Optional[float] = None,
                  metadata: Optional[Dict] = None) -> pd.Series:
        size = fund_size
        if size is None and metadata:
            size = metadata.get("fund_size")
        value = np.nan if size in (None, "") else float(size)
        return pd.Series(value, index=df.index, dtype=float)

    @staticmethod
    def fund_size_percentile(df: pd.DataFrame, fund_size: Optional[float] = None,
                             metadata: Optional[Dict] = None) -> pd.Series:
        size = fund_size
        if size is None and metadata:
            size = metadata.get("fund_size")
        if size in (None, ""):
            value = np.nan
        else:
            value = max(0.0, min(100.0, np.log10(max(float(size), 1.0)) / 2 * 100))
        return pd.Series(value, index=df.index, dtype=float)

    @staticmethod
    def turnover_liquidity(df: pd.DataFrame, window: int = 20) -> pd.Series:
        amount = df["close"] * df["volume"]
        return amount.rolling(window=window, min_periods=5).mean()

    @staticmethod
    def fresh_52w_breakout(
        df: pd.DataFrame,
        lookback: int = 252,
        freshness_window: int = 20,
        near_high_threshold: float = 0.03,
        min_periods: int = 120,
    ) -> pd.Series:
        """
        近一年刚创新高的“新鲜度”得分。

        逻辑：
        1. 用前 lookback 个交易日（不含当日）的最高收盘价定义“前高”。
        2. 当日收盘价突破前高，视为一次新高突破事件。
        3. 仅当最近 freshness_window 日内发生过突破，且当前价格仍接近近一年高点，
           才给出正分；突破太久未再创新高则分数衰减为 0。
        """
        close = pd.to_numeric(df["close"], errors="coerce")
        prior_high = close.shift(1).rolling(window=lookback, min_periods=min_periods).max()
        rolling_high = close.rolling(window=lookback, min_periods=min_periods).max()
        breakout = close > prior_high

        last_breakout_index = pd.Series(np.nan, index=df.index, dtype=float)
        last_idx = np.nan
        for i, flag in enumerate(breakout.fillna(False)):
            if flag:
                last_idx = float(i)
            last_breakout_index.iloc[i] = last_idx

        age = pd.Series(df.index, index=df.index, dtype=float) - last_breakout_index
        freshness = 1 - (age / max(freshness_window, 1))
        freshness = freshness.clip(lower=0, upper=1).fillna(0)

        distance_from_high = (rolling_high - close) / rolling_high.replace(0, np.nan)
        still_near_high = (distance_from_high <= near_high_threshold).astype(float).fillna(0)

        return (freshness * still_near_high).fillna(0)


# ── 突破类因子 ────────────────────────────────────────────────────────────────

class BreakoutFactors:
    @staticmethod
    def new_high_breakout(
        df: pd.DataFrame,
        lookback: int = 252,
        min_periods: int = 120,
    ) -> pd.DataFrame:
        """
        52周(一年)新高突破因子 - 首次突破一年最高点

        检测价格是否首次突破过去N个交易日的最高点。
        只有当收盘价创下过去252个交易日的最高价时，才算真正的新高突破。

        参数:
            lookback: 回溯期（交易日），默认252（一年）
            min_periods: 最小计算期，避免数据不足时误差

        返回:
            DataFrame 包含:
            - breakout: 0-1 信号，1表示当日首次突破一年新高
            - distance_pct: 距离一年高点的百分比
            - at_high: 是否处于一年最高点
        """
        close = pd.to_numeric(df["close"], errors="coerce")

        n = len(close)
        result = pd.DataFrame(index=df.index)
        result["breakout"] = 0
        result["distance_pct"] = 0.0
        result["at_high"] = 0

        # 计算滚动最高价（过去lookback天的最高收盘价）
        rolling_max = close.rolling(window=lookback, min_periods=min_periods).max()

        # 当前位置是否处于一年最高点（收盘价 = 过去252天的最高价）
        at_high = (close >= rolling_max) & rolling_max.notna()

        # 首次突破信号：当日是一年新高，且前一天不是
        prev_at_high = at_high.shift(1).fillna(False)
        first_breakout = at_high & (~prev_at_high)

        # 计算距离一年高点的百分比
        distance = (rolling_max - close) / rolling_max.replace(0, np.nan) * 100

        # 填充结果
        result["breakout"] = first_breakout.astype(int)
        result["distance_pct"] = distance.fillna(0)
        result["at_high"] = at_high.astype(int)

        return result


# ── 形态类因子 ────────────────────────────────────────────────────────────────

class PatternFactors:
    @staticmethod
    def cup_and_handle(
        df: pd.DataFrame,
        cup_window: int = 30,
        handle_window: int = 8,
        min_cup_depth: float = 0.05,
        max_cup_depth: float = 0.50,
        handle_retrace: float = 0.30,
    ) -> pd.DataFrame:
        """
        杯柄形态检测 (Cup and Handle Pattern) - 支持日线/周线

        经典技术分析形态，由William O'Neil提出：
        - 杯部：U形或圆弧底，左侧下跌后右侧上涨
        - 柄部：杯部右侧的小幅回调，通常向下倾斜5-15%

        参数:
            cup_window: 杯部检测窗口 (根数，建议日线30/周线15)
            handle_window: 柄部检测窗口 (建议日线8/周线4)
            min_cup_depth: 杯部最小深度 (相对杯口跌幅比例)
            max_cup_depth: 杯部最大深度
            handle_retrace: 柄部最大回撤幅度

        返回:
            DataFrame 包含:
            - cup_and_handle: 0-1 之间的形态匹配度分数
            - status: 状态码 0=无形态 1=杯部形成中 2=柄部形成中 3=突破前夕 4=刚刚突破
            - phase: 状态描述
        """
        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df.get("high", close), errors="coerce")
        low = pd.to_numeric(df.get("low", close), errors="coerce")
        volume = pd.to_numeric(df.get("volume", pd.Series(1, index=df.index)), errors="coerce")

        n = len(close)
        result = pd.DataFrame(index=df.index)
        result["cup_and_handle"] = 0.0
        result["status"] = 0
        result["phase"] = "无形态"

        # 移动窗口检测
        for i in range(cup_window + handle_window + 5, n):
            cup_start = i - cup_window - handle_window
            cup_end = i - handle_window
            handle_start = cup_end

            cup_prices = close.iloc[cup_start:cup_end].values
            handle_prices = close.iloc[handle_start:i].values

            if len(cup_prices) < 10 or len(handle_prices) < 3:
                continue

            # 找到杯部的最低点
            cup_min_idx = np.argmin(cup_prices)
            cup_min = cup_prices[cup_min_idx]

            # 左杯口: 最低点之前的最高价
            left_cup = cup_prices[:cup_min_idx] if cup_min_idx > 0 else []
            left_peak = np.max(left_cup) if len(left_cup) > 1 else cup_prices[0]

            # 右杯口: 最低点之后的最高价
            right_cup = cup_prices[cup_min_idx:] if cup_min_idx < len(cup_prices) else []
            right_peak = np.max(right_cup) if len(right_cup) > 1 else cup_prices[-1]

            # 杯部深度
            cup_depth = (left_peak - cup_min) / left_peak if left_peak > 0 else 0

            # 杯部形态条件
            if not (min_cup_depth <= cup_depth <= max_cup_depth):
                continue

            # 右杯口不应该明显低于左杯口
            if right_peak < left_peak * 0.88:
                continue

            # 柄部分析
            handle_high = np.max(handle_prices)
            handle_low = np.min(handle_prices)
            handle_retrace_actual = (handle_high - handle_low) / handle_high if handle_high > 0 else 0

            # 柄部条件: 小幅回调，在杯口下方运行
            if not (handle_retrace_actual <= handle_retrace and handle_low > cup_min):
                continue

            # 成交量萎缩（可选）
            cup_vol = np.mean(volume.iloc[cup_start:cup_end].values)
            handle_vol = np.mean(volume.iloc[handle_start:i].values) if len(volume.iloc[handle_start:i]) > 0 else cup_vol
            volume_ok = handle_vol < cup_vol * 1.5 if cup_vol > 0 else True

            # 计算形态匹配分数
            # 1. 杯部深度合理性 (0-0.4)
            ideal_depth = 0.20
            depth_score = max(0, 1 - abs(cup_depth - ideal_depth) / ideal_depth) * 0.4

            # 2. 右杯口高度 (0-0.3)
            right_cup_score = min(0.3, (right_peak / left_peak) * 0.3) if left_peak > 0 else 0

            # 3. 柄部强度 (0-0.2)
            handle_score = max(0, (1 - handle_retrace_actual / handle_retrace)) * 0.2

            # 4. 成交量萎缩 (0-0.1)
            vol_score = max(0, (1 - handle_vol / cup_vol)) * 0.1 if cup_vol > 0 and volume_ok else 0

            total_score = min(1.0, depth_score + right_cup_score + handle_score + vol_score)

            # 当前状态判断
            current_price = close.iloc[i - 1]

            # 状态判断逻辑
            if current_price < right_peak * 0.90:
                # 还在杯部
                status = 1
                phase = "杯部形成"
            elif current_price < right_peak * 0.97:
                # 接近右杯口，柄部形成中
                status = 2
                phase = "柄部形成"
            elif current_price < handle_high:
                # 接近或略低于柄高点，突破前夕
                status = 3
                phase = "突破前夕"
            else:
                # 已经突破柄部高点
                status = 4
                phase = "刚刚突破"

            result.iloc[i, result.columns.get_loc("cup_and_handle")] = total_score
            result.iloc[i, result.columns.get_loc("status")] = status
            result.iloc[i, result.columns.get_loc("phase")] = phase

        return result

    @staticmethod
    def cup_and_handle_weekly(
        df: pd.DataFrame,
        cup_weeks: int = 20,
        handle_weeks: int = 5,
        min_cup_depth: float = 0.08,
        max_cup_depth: float = 0.50,
        handle_retrace: float = 0.35,
    ) -> pd.DataFrame:
        """
        杯柄形态检测 - 周线专用版本

        参数:
            cup_weeks: 杯部检测周数
            handle_weeks: 柄部检测周数
            min_cup_depth: 杯部最小深度
            max_cup_depth: 杯部最大深度
            handle_retrace: 柄部最大回撤幅度

        返回:
            DataFrame 包含:
            - cup_and_handle: 形态匹配度分数
            - status: 状态码
            - phase: 状态描述
        """
        # 转换为周线
        df_weekly = PatternFactors._convert_to_weekly(df)
        if df_weekly is None or len(df_weekly) < cup_weeks + handle_weeks + 5:
            return pd.DataFrame({
                "cup_and_handle": [],
                "status": [],
                "phase": []
            })

        # 调用主函数
        return PatternFactors.cup_and_handle(
            df_weekly,
            cup_window=cup_weeks,
            handle_window=handle_weeks,
            min_cup_depth=min_cup_depth,
            max_cup_depth=max_cup_depth,
            handle_retrace=handle_retrace,
        )

    @staticmethod
    def _convert_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
        """将日线数据转换为周线数据"""
        if "date" not in df.columns:
            return None

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # 周线聚合
        weekly = df.resample("W").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })

        # 删除全NaN行
        weekly = weekly.dropna(how="all")
        weekly = weekly.reset_index()
        weekly = weekly.rename(columns={"date": "date"})

        return weekly


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
            'relative_strength':        ReturnFactors.relative_strength,
            # 风险类
            'volatility':               RiskFactors.volatility,
            'max_drawdown':             RiskFactors.max_drawdown,
            'sharpe_ratio':             RiskFactors.sharpe_ratio,
            'atr':                      RiskFactors.atr,
            'market_fear':              RiskFactors.market_fear,
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
            'fund_size':                TrendFactors.fund_size,
            'fund_size_percentile':     TrendFactors.fund_size_percentile,
            'turnover_liquidity':       TrendFactors.turnover_liquidity,
            'fresh_52w_breakout':       TrendFactors.fresh_52w_breakout,
            # 突破类
            'new_high_breakout':        BreakoutFactors.new_high_breakout,
            # 形态类
            'cup_and_handle':           PatternFactors.cup_and_handle,
            'cup_and_handle_weekly':    PatternFactors.cup_and_handle_weekly,
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
