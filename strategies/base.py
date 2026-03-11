from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import pandas as pd


IndicatorValue = Union[pd.Series, pd.DataFrame]


@dataclass
class StrategyContext:
    df: pd.DataFrame
    params: Dict
    indicators: Dict[str, IndicatorValue] = field(default_factory=dict)


@dataclass
class StrategyDecision:
    regime: pd.Series
    entry: pd.Series
    exit: pd.Series
    stop_loss: Optional[pd.Series] = None
    take_profit: Optional[pd.Series] = None


class Strategy:
    name = "base"

    def prepare(self, df: pd.DataFrame, params: Dict) -> StrategyContext:
        return StrategyContext(df=df, params=params)

    def decide(self, ctx: StrategyContext) -> StrategyDecision:
        raise NotImplementedError
