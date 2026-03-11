import pandas as pd

from etf_factors import calculate_factor
from strategies.base import Strategy, StrategyContext, StrategyDecision


class DonchianBreakoutStrategy(Strategy):
    name = "donchian_breakout"

    def prepare(self, df: pd.DataFrame, params: dict) -> StrategyContext:
        ctx = StrategyContext(df=df, params=params)
        ctx.indicators["donchian"] = calculate_factor(
            df, "donchian_channel", window=params.get("window", 20)
        )
        ctx.indicators["adx"] = calculate_factor(df, "adx", period=params.get("adx_period", 14))
        return ctx

    def decide(self, ctx: StrategyContext) -> StrategyDecision:
        dc_df = ctx.indicators["donchian"]
        adx = ctx.indicators["adx"]["adx"]
        close = ctx.df["close"]
        upper = dc_df["upper"].shift(1)
        middle = dc_df["middle"].shift(1)
        lower = dc_df["lower"].shift(1)

        regime = (adx > ctx.params.get("adx_min", 20)).astype(int)
        entry = ((close > upper) & (regime == 1)).astype(int)
        exit_ = ((close < middle) | (close < lower) | (regime == 0)).astype(int)
        return StrategyDecision(regime=regime, entry=entry, exit=exit_)
