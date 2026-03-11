"""
ETF Backtest Engine Module
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategies import build_strategy_registry
from strategies.base import StrategyDecision

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    commission_rate: float = 0.0003
    slippage: float = 0.0001
    position_size: float = 1.0
    risk_free_rate: float = 0.03


@dataclass
class BacktestResult:
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


class ETFBacktestEngine:
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.strategies = build_strategy_registry()

    def run(self, df: pd.DataFrame, strategy: str, params: Dict) -> BacktestResult:
        if len(df) < 20:
            return BacktestResult()
        strategy_obj = self.strategies.get(strategy)
        if strategy_obj is None:
            raise ValueError(f"未知的策略: {strategy}")
        ctx = strategy_obj.prepare(df, params)
        decision = strategy_obj.decide(ctx)
        return self._simulate(df, decision)

    def _simulate(self, df: pd.DataFrame, decision: StrategyDecision) -> BacktestResult:
        cash = self.config.initial_capital
        position = 0.0
        equity = [cash]
        trades = []
        entry_price = 0.0
        entry_idx = 0

        entry = decision.entry.shift(1).fillna(0).astype(int)
        exit_ = decision.exit.shift(1).fillna(0).astype(int)
        regime = decision.regime.shift(1).fillna(0).astype(int)

        for i in range(1, len(df)):
            current_price = df["close"].iloc[i]
            can_enter = int(regime.iloc[i]) == 1
            entry_signal = int(entry.iloc[i]) == 1
            exit_signal = int(exit_.iloc[i]) == 1 or int(regime.iloc[i]) <= 0

            if entry_signal and can_enter and position == 0:
                shares = (cash * self.config.position_size) / (
                    current_price * (1 + self.config.commission_rate + self.config.slippage)
                )
                cost = shares * current_price * (1 + self.config.commission_rate + self.config.slippage)
                if cost <= cash * 1.001:
                    cash -= cost
                    position = shares
                    entry_price = current_price
                    entry_idx = i
                    trades.append({
                        "date": i,
                        "action": "BUY",
                        "price": current_price,
                        "shares": shares,
                        "cost": cost,
                    })

            elif position > 0 and exit_signal:
                proceeds = position * current_price * (1 - self.config.commission_rate - self.config.slippage)
                profit = (current_price - entry_price) / entry_price
                trades.append({
                    "date": i,
                    "action": "SELL",
                    "price": current_price,
                    "profit": profit,
                    "holding_days": i - entry_idx,
                })
                cash += proceeds
                position = 0

            equity.append(cash + position * current_price)

        if position > 0:
            last_idx = len(df) - 1
            last_price = df["close"].iloc[last_idx]
            proceeds = position * last_price * (1 - self.config.commission_rate - self.config.slippage)
            profit = (last_price - entry_price) / entry_price
            trades.append({
                "date": last_idx,
                "action": "SELL",
                "price": last_price,
                "profit": profit,
                "holding_days": last_idx - entry_idx,
                "forced_exit": True,
            })
            cash += proceeds
            position = 0
            equity[-1] = cash

        equity_curve = pd.DataFrame({"date": list(range(len(equity))), "equity": equity})
        return self._calculate_metrics(equity_curve, trades)

    def _calculate_metrics(self, equity_curve: pd.DataFrame, trades: List[Dict]) -> BacktestResult:
        if len(equity_curve) == 0:
            return BacktestResult()

        equity = equity_curve["equity"]
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        n_days = len(equity)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0.0
        daily_returns = equity.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (
            (daily_returns.mean() * 252 - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        )

        rolling_max = equity.expanding().max()
        max_drawdown = ((equity - rolling_max) / rolling_max).min()

        winning_trades = [t for t in trades if t.get("action") == "SELL" and t.get("profit", 0) > 0]
        losing_trades = [t for t in trades if t.get("action") == "SELL" and t.get("profit", 0) <= 0]
        total_trades = len(winning_trades) + len(losing_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        total_profit = sum(t["profit"] for t in winning_trades)
        total_loss = abs(sum(t["profit"] for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf") if total_profit > 0 else 0.0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            equity_curve=equity_curve,
            trades=trades,
        )


def run_backtest(
    etf_code: str,
    strategy: str,
    params: Dict,
    days: int = 500,
    config: Optional[BacktestConfig] = None,
) -> BacktestResult:
    from etf_data import ETFDataFetcher

    df = ETFDataFetcher().get_historical_data(etf_code, days)
    if df is None or len(df) < 20:
        return BacktestResult()
    return ETFBacktestEngine(config).run(df, strategy, params)
