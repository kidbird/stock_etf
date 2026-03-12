"""
Stock Backtest Engine Module
股票回测引擎
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from stock_strategies import build_stock_strategy_registry, StrategyDecision

logger = logging.getLogger(__name__)


@dataclass
class StockBacktestConfig:
    initial_capital: float = 100000.0
    commission_rate: float = 0.0003
    slippage: float = 0.0001
    position_size: float = 1.0
    risk_free_rate: float = 0.03


@dataclass
class StockBacktestResult:
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: List[Dict] = field(default_factory=list)


class StockBacktestEngine:
    def __init__(self, config: Optional[StockBacktestConfig] = None):
        self.config = config or StockBacktestConfig()
        self.strategies = build_stock_strategy_registry()

    def run(self, df: pd.DataFrame, strategy: str, params: Dict) -> StockBacktestResult:
        if len(df) < 20:
            return StockBacktestResult()

        strategy_obj = self.strategies.get(strategy)
        if strategy_obj is None:
            raise ValueError(f"未知的策略: {strategy}")

        ctx = strategy_obj.prepare(df, params)
        decision = strategy_obj.decide(ctx)
        return self._simulate(df, decision)

    def _simulate(
        self, df: pd.DataFrame, decision: StrategyDecision
    ) -> StockBacktestResult:
        cash = self.config.initial_capital
        position = 0.0
        equity = [cash]

        signals = decision.signals
        close_prices = df["close"].values

        trades = []
        entry_price = 0.0
        entry_date = None

        for i in range(1, len(df)):
            signal = signals.iloc[i]
            price = close_prices[i]
            date = df["date"].iloc[i]

            if signal == 1 and position == 0:
                buy_price = price * (1 + self.config.slippage)
                cost = cash * self.config.position_size
                shares = cost / buy_price
                commission = cost * self.config.commission_rate
                position = shares
                cash = cash - cost - commission
                entry_price = buy_price
                entry_date = date

            elif signal == -1 and position > 0:
                sell_price = price * (1 - self.config.slippage)
                proceeds = position * sell_price
                commission = proceeds * self.config.commission_rate
                net_proceeds = proceeds - commission

                trade_pnl = (sell_price - entry_price) / entry_price
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": sell_price,
                        "return": trade_pnl,
                        "shares": position,
                    }
                )

                cash = cash + net_proceeds
                position = 0.0
                entry_price = 0.0

            current_equity = cash + position * price
            equity.append(current_equity)

        if position > 0:
            last_price = close_prices[-1]
            proceeds = position * last_price
            trade_pnl = (last_price - entry_price) / entry_price
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": df["date"].iloc[-1],
                    "entry_price": entry_price,
                    "exit_price": last_price,
                    "return": trade_pnl,
                    "shares": position,
                }
            )
            cash = cash + proceeds

        return self._calculate_metrics(df, equity, trades)

    def _calculate_metrics(
        self, df: pd.DataFrame, equity: List[float], trades: List[Dict]
    ) -> StockBacktestResult:
        if not equity or equity[-1] == 0:
            return StockBacktestResult()

        equity_array = np.array(equity)
        total_return = (equity_array[-1] - equity_array[0]) / equity_array[0]

        days = len(df)
        years = days / 252
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0.0

        returns = np.diff(equity_array) / equity_array[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (
                (returns.mean() - self.config.risk_free_rate / 252)
                / returns.std()
                * np.sqrt(252)
            )
        else:
            sharpe_ratio = 0.0

        cummax = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - cummax) / cummax
        max_drawdown = drawdowns.min()

        winning_trades = [t for t in trades if t["return"] > 0]
        losing_trades = [t for t in trades if t["return"] <= 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0.0

        total_profit = sum(
            [
                t["return"] * t["shares"] * self.config.initial_capital
                for t in winning_trades
            ]
        )
        total_loss = abs(
            sum(
                [
                    t["return"] * t["shares"] * self.config.initial_capital
                    for t in losing_trades
                ]
            )
        )
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0

        equity_curve = pd.DataFrame(
            {"date": df["date"].values[: len(equity)], "equity": equity}
        )

        return StockBacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            equity_curve=equity_curve,
            trades=trades,
        )
