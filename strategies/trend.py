import pandas as pd

from etf_factors import calculate_factor
from strategies.base import Strategy, StrategyContext, StrategyDecision


class TrendFilterMACDStrategy(Strategy):
    name = "trend_filter_macd"

    def prepare(self, df: pd.DataFrame, params: dict) -> StrategyContext:
        ctx = StrategyContext(df=df, params=params)
        ctx.indicators["macd"] = calculate_factor(
            df,
            "macd",
            fast=params.get("fast", 12),
            slow=params.get("slow", 26),
            signal=params.get("signal", 9),
        )
        ctx.indicators["adx"] = calculate_factor(df, "adx", period=params.get("adx_period", 14))
        ctx.indicators["ma_alignment"] = calculate_factor(
            df, "ma_alignment", periods=params.get("ma_periods", [9, 21, 50])
        )
        return ctx

    def decide(self, ctx: StrategyContext) -> StrategyDecision:
        macd_df = ctx.indicators["macd"]
        adx = ctx.indicators["adx"]["adx"]
        ma_alignment = ctx.indicators["ma_alignment"]

        regime = ((adx > ctx.params.get("adx_min", 20)) &
                  (ma_alignment >= ctx.params.get("ma_alignment_min", 0.3))).astype(int)
        entry = ((macd_df["macd"] > macd_df["signal"]) & (regime == 1)).astype(int)
        exit_ = ((macd_df["macd"] < macd_df["signal"]) | (regime == 0)).astype(int)
        return StrategyDecision(regime=regime, entry=entry, exit=exit_)


class SupertrendFollowStrategy(Strategy):
    name = "supertrend_follow"

    def prepare(self, df: pd.DataFrame, params: dict) -> StrategyContext:
        ctx = StrategyContext(df=df, params=params)
        ctx.indicators["supertrend"] = calculate_factor(
            df,
            "supertrend",
            period=params.get("period", 10),
            multiplier=params.get("multiplier", 3.0),
        )
        ctx.indicators["roc"] = calculate_factor(df, "roc", period=params.get("roc_period", 20))
        return ctx

    def decide(self, ctx: StrategyContext) -> StrategyDecision:
        st_df = ctx.indicators["supertrend"]
        roc = ctx.indicators["roc"]
        trend = st_df["trend"].fillna(0).astype(int)
        regime = (roc > ctx.params.get("roc_min", 0)).astype(int)
        entry = ((trend == 1) & (trend.shift(1).fillna(0) != 1) & (regime == 1)).astype(int)
        exit_ = ((trend == -1) | (regime == 0)).astype(int)
        return StrategyDecision(regime=regime, entry=entry, exit=exit_)
