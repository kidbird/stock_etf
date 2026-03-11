from strategies.breakout import DonchianBreakoutStrategy
from strategies.legacy import (
    BollingerSignal,
    LegacySignalStrategy,
    MACDSignal,
    MASignal,
    RSISignal,
)
from strategies.trend import SupertrendFollowStrategy, TrendFilterMACDStrategy


def build_strategy_registry():
    return {
        "rsi": LegacySignalStrategy("rsi", RSISignal()),
        "macd": LegacySignalStrategy("macd", MACDSignal()),
        "ma_cross": LegacySignalStrategy("ma_cross", MASignal()),
        "bollinger": LegacySignalStrategy("bollinger", BollingerSignal()),
        "trend_filter_macd": TrendFilterMACDStrategy(),
        "supertrend_follow": SupertrendFollowStrategy(),
        "donchian_breakout": DonchianBreakoutStrategy(),
    }
