"""
ETF Backtest Engine Module
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

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



class SignalGenerator:
    def generate(self, df: pd.DataFrame, params: Dict) -> pd.Series:
        raise NotImplementedError


class RSISignal(SignalGenerator):
    def generate(self, df: pd.DataFrame, params: Dict) -> pd.Series:
        from etf_factors import calculate_factor
        period = params.get('period', 14)
        oversold = params.get('oversold', 30)
        overbought = params.get('overbought', 70)
        rsi = calculate_factor(df, 'rsi', period=period)
        signal = pd.Series(0, index=df.index)
        signal[rsi < oversold] = 1
        signal[rsi > overbought] = -1
        return signal


class MACDSignal(SignalGenerator):
    def generate(self, df: pd.DataFrame, params: Dict) -> pd.Series:
        from etf_factors import calculate_factor
        macd_df = calculate_factor(df, 'macd', fast=params.get('fast', 12), slow=params.get('slow', 26), signal=params.get('signal', 9))
        signal = pd.Series(0, index=df.index)
        signal[macd_df['macd'] > macd_df['signal']] = 1
        signal[macd_df['macd'] < macd_df['signal']] = -1
        return signal


class MASignal(SignalGenerator):
    def generate(self, df: pd.DataFrame, params: Dict) -> pd.Series:
        from etf_factors import calculate_factor
        return calculate_factor(df, 'ma_cross', short_period=params.get('short_period', 5), long_period=params.get('long_period', 20))


class BollingerSignal(SignalGenerator):
    def generate(self, df: pd.DataFrame, params: Dict) -> pd.Series:
        from etf_factors import calculate_factor
        bb_df = calculate_factor(df, 'bollinger_bands', window=params.get('window', 20), num_std=params.get('num_std', 2))
        signal = pd.Series(0, index=df.index)
        signal[bb_df['position'] < 0.2] = 1
        signal[bb_df['position'] > 0.8] = -1
        return signal


class ETFBacktestEngine:
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.signal_generators = {
            'rsi': RSISignal(),
            'macd': MACDSignal(),
            'ma_cross': MASignal(),
            'bollinger': BollingerSignal(),
        }

    def run(self, df: pd.DataFrame, strategy: str, params: Dict) -> BacktestResult:
        if len(df) < 20:
            return BacktestResult()
        signal_generator = self.signal_generators.get(strategy)
        if signal_generator is None:
            raise ValueError(f"未知的策略: {strategy}")
        signals = signal_generator.generate(df, params)
        return self._simulate(df, signals)

    def _simulate(self, df: pd.DataFrame, signals: pd.Series) -> BacktestResult:
        cash = self.config.initial_capital
        position = 0
        equity = [cash]
        trades = []
        entry_price = 0
        entry_idx = 0

        signals = signals.fillna(0).astype(int)

        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            signal = int(signals.iloc[i])

            if signal == 1 and position == 0:
                shares = (cash * self.config.position_size) / (current_price * (1 + self.config.commission_rate + self.config.slippage))
                cost = shares * current_price * (1 + self.config.commission_rate + self.config.slippage)
                if cost <= cash * 1.001:
                    cash -= cost
                    position = shares
                    entry_price = current_price
                    entry_idx = i
                    trades.append({'date': i, 'action': 'BUY', 'price': current_price, 'shares': shares, 'cost': cost})

            elif signal == -1 and position > 0:
                proceeds = position * current_price * (1 - self.config.commission_rate - self.config.slippage)
                profit = (current_price - entry_price) / entry_price
                trades.append({'date': i, 'action': 'SELL', 'price': current_price, 'profit': profit, 'holding_days': i - entry_idx})
                cash += proceeds
                position = 0

            equity.append(cash + position * current_price)

        equity_curve = pd.DataFrame({'date': list(range(len(equity))), 'equity': equity})
        return self._calculate_metrics(equity_curve, trades)

    def _calculate_metrics(self, equity_curve: pd.DataFrame, trades: List[Dict]) -> BacktestResult:
        if len(equity_curve) == 0:
            return BacktestResult()

        equity = equity_curve['equity']
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        daily_returns = equity.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (total_return * 252 - self.config.risk_free_rate) / volatility if volatility > 0 else 0

        rolling_max = equity.expanding().max()
        max_drawdown = ((equity - rolling_max) / rolling_max).min()


        winning_trades = [t for t in trades if t.get('action') == 'SELL' and t.get('profit', 0) > 0]
        losing_trades = [t for t in trades if t.get('action') == 'SELL' and t.get('profit', 0) <= 0]
        total_trades = len(winning_trades) + len(losing_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        return BacktestResult(
            total_return=total_return,
            annualized_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            equity_curve=equity_curve,
            trades=trades
        )


def run_backtest(etf_code: str, strategy: str, params: Dict, days: int = 500, config: Optional[BacktestConfig] = None) -> BacktestResult:
    from etf_data import ETFDataFetcher
    df = ETFDataFetcher().get_historical_data(etf_code, days)
    if df is None or len(df) < 20:
        return BacktestResult()
    return ETFBacktestEngine(config).run(df, strategy, params)
