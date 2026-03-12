"""
Multi-factor combo analysis and backtest helpers.
"""

from dataclasses import dataclass, field
import inspect
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from etf_factors import ETFFactorCalculator, FactorRegistry
from stock_factors import StockFactorCalculator


@dataclass
class FactorSpec:
    name: str
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)
    scorer: str = "identity"


@dataclass
class ComboTemplate:
    name: str
    universe_type: str
    description: str
    buy_threshold: float
    sell_threshold: float
    factors: List[FactorSpec]


ETF_COMBOS: Dict[str, ComboTemplate] = {
    "etf_trend": ComboTemplate(
        name="etf_trend",
        universe_type="etf",
        description="趋势动量组合",
        buy_threshold=0.30,
        sell_threshold=-0.10,
        factors=[
            FactorSpec("ma_alignment", 0.35, scorer="alignment"),
            FactorSpec("roc", 0.30, {"period": 20}, scorer="roc"),
            FactorSpec("adx", 0.20, {"period": 14}, scorer="adx"),
            FactorSpec("relative_strength", 0.15, {"period": 60}, scorer="roc"),
        ],
    ),
    "etf_trend_size": ComboTemplate(
        name="etf_trend_size",
        universe_type="etf",
        description="趋势 + 规模 + 恐慌过滤",
        buy_threshold=0.35,
        sell_threshold=-0.05,
        factors=[
            FactorSpec("ma_alignment", 0.30, scorer="alignment"),
            FactorSpec("roc", 0.20, {"period": 20}, scorer="roc"),
            FactorSpec("fund_size_percentile", 0.20, scorer="percentile"),
            FactorSpec("turnover_liquidity", 0.15, scorer="zscore"),
            FactorSpec("market_fear", 0.15, scorer="fear_inverse"),
        ],
    ),
}


STOCK_COMBOS: Dict[str, ComboTemplate] = {
    "stock_quality": ComboTemplate(
        name="stock_quality",
        universe_type="stock",
        description="技术趋势 + 市值 + ROE",
        buy_threshold=0.30,
        sell_threshold=-0.10,
        factors=[
            FactorSpec("ma_alignment", 0.30, scorer="alignment"),
            FactorSpec("roc", 0.20, {"period": 20}, scorer="roc"),
            FactorSpec("market_cap_bucket", 0.20, scorer="bucket"),
            FactorSpec("roe", 0.20, scorer="roe"),
            FactorSpec("market_fear", 0.10, scorer="fear_inverse"),
        ],
    ),
    "stock_quality_industry": ComboTemplate(
        name="stock_quality_industry",
        universe_type="stock",
        description="技术趋势 + 市值 + ROE + 行业偏好",
        buy_threshold=0.35,
        sell_threshold=-0.05,
        factors=[
            FactorSpec("ma_alignment", 0.25, scorer="alignment"),
            FactorSpec("roc", 0.20, {"period": 20}, scorer="roc"),
            FactorSpec("market_cap_bucket", 0.20, scorer="bucket"),
            FactorSpec("roe", 0.20, scorer="roe"),
            FactorSpec("industry", 0.15, scorer="industry"),
        ],
    ),
}


INDUSTRY_SCORE_MAP = {
    "银行": 0.40,
    "保险": 0.30,
    "白酒": 0.20,
    "电力": 0.15,
    "家电": 0.15,
    "电力设备": 0.10,
    "有色金属": 0.05,
    "证券": 0.00,
    "汽车": 0.00,
    "电子": 0.00,
    "化工": -0.05,
    "未知": 0.00,
}


def list_combo_templates(universe_type: str) -> List[str]:
    if universe_type == "etf":
        return sorted(ETF_COMBOS.keys())
    if universe_type == "stock":
        return sorted(STOCK_COMBOS.keys())
    return []


def get_combo_template(universe_type: str, combo_name: str) -> ComboTemplate:
    mapping = ETF_COMBOS if universe_type == "etf" else STOCK_COMBOS
    if combo_name not in mapping:
        raise ValueError(f"未知{universe_type}组合: {combo_name}")
    return mapping[combo_name]


def _ensure_series(df: pd.DataFrame, value: Any) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if "adx" in value.columns:
            return value["adx"]
        if "trend" in value.columns:
            return value["trend"]
        return value.iloc[:, 0]
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, (int, float, np.number)):
        return pd.Series(float(value), index=df.index, dtype=float)
    return pd.Series([value] * len(df), index=df.index)


def _score_series(series: pd.Series, scorer: str) -> pd.Series:
    if scorer == "alignment":
        return series.clip(-1, 1).fillna(0)
    if scorer == "roc":
        return (series / 10.0).clip(-1, 1).fillna(0)
    if scorer == "adx":
        return ((series - 20) / 20.0).clip(-1, 1).fillna(0)
    if scorer == "percentile":
        return ((series - 50) / 50.0).clip(-1, 1).fillna(0)
    if scorer == "zscore":
        mean = series.rolling(window=60, min_periods=10).mean()
        std = series.rolling(window=60, min_periods=10).std().replace(0, np.nan)
        return (((series - mean) / std) / 3.0).clip(-1, 1).fillna(0)
    if scorer == "fear_inverse":
        return (-(series - 20) / 20.0).clip(-1, 1).fillna(0)
    if scorer == "bucket":
        return ((series - 1.5) / 1.5).clip(-1, 1).fillna(0)
    if scorer == "roe":
        return ((series - 10) / 20.0).clip(-1, 1).fillna(0)
    if scorer == "industry":
        mapped = series.map(lambda value: INDUSTRY_SCORE_MAP.get(value, 0.0))
        return mapped.astype(float).clip(-1, 1).fillna(0)
    return pd.to_numeric(series, errors="coerce").clip(-1, 1).fillna(0)


def _simulate_signals(df: pd.DataFrame, score: pd.Series, buy_threshold: float, sell_threshold: float) -> Dict[str, Any]:
    cash = 100000.0
    position = 0.0
    equity = [cash]
    trades: List[Dict[str, Any]] = []
    entry_price = 0.0
    entry_index = 0

    entry_signal = ((score >= buy_threshold) & (score.shift(1).fillna(-10) < buy_threshold)).astype(int)
    exit_signal = ((score <= sell_threshold) & (score.shift(1).fillna(10) > sell_threshold)).astype(int)

    for i in range(1, len(df)):
        price = float(df["close"].iloc[i])
        if entry_signal.iloc[i - 1] == 1 and position == 0:
            position = cash / price
            cash = 0.0
            entry_price = price
            entry_index = i
            trades.append({"date": i, "action": "BUY", "price": price})
        elif exit_signal.iloc[i - 1] == 1 and position > 0:
            cash = position * price
            trades.append(
                {
                    "date": i,
                    "action": "SELL",
                    "price": price,
                    "profit": (price - entry_price) / entry_price if entry_price else 0.0,
                    "holding_days": i - entry_index,
                }
            )
            position = 0.0
        equity.append(cash + position * price)

    if position > 0:
        last_price = float(df["close"].iloc[-1])
        cash = position * last_price
        trades.append(
            {
                "date": len(df) - 1,
                "action": "SELL",
                "price": last_price,
                "profit": (last_price - entry_price) / entry_price if entry_price else 0.0,
                "holding_days": len(df) - 1 - entry_index,
                "forced_exit": True,
            }
        )
        equity[-1] = cash

    equity_curve = pd.DataFrame({"date": df["date"], "equity": equity})
    total_return = (equity_curve["equity"].iloc[-1] - equity_curve["equity"].iloc[0]) / equity_curve["equity"].iloc[0]
    returns = equity_curve["equity"].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    sharpe = ((returns.mean() * 252) / volatility) if volatility and volatility > 0 else 0.0
    drawdown = ((equity_curve["equity"] / equity_curve["equity"].cummax()) - 1).min()
    sells = [item for item in trades if item["action"] == "SELL"]
    wins = [item for item in sells if item.get("profit", 0) > 0]
    losses = [item for item in sells if item.get("profit", 0) <= 0]
    total_loss = abs(sum(item.get("profit", 0) for item in losses))
    total_profit = sum(item.get("profit", 0) for item in wins)
    profit_factor = total_profit / total_loss if total_loss > 0 else (float("inf") if total_profit > 0 else 0.0)
    annualized = (1 + total_return) ** (252 / len(equity_curve)) - 1 if len(equity_curve) > 0 else 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized,
        "sharpe_ratio": sharpe,
        "max_drawdown": drawdown,
        "win_rate": len(wins) / len(sells) if sells else 0.0,
        "profit_factor": profit_factor,
        "total_trades": len(sells),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "equity_curve": equity_curve,
        "trades": trades,
    }


def run_combo_analysis(
    universe_type: str,
    code: str,
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    combo_name: str,
    benchmark_df: Optional[pd.DataFrame] = None,
    fundamentals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    template = get_combo_template(universe_type, combo_name)
    calculator = ETFFactorCalculator() if universe_type == "etf" else StockFactorCalculator()

    factor_series: Dict[str, pd.Series] = {}
    factor_values: Dict[str, Any] = {}
    factor_contributions: Dict[str, float] = {}
    score = pd.Series(0.0, index=df.index)
    total_weight = sum(spec.weight for spec in template.factors) or 1.0

    for spec in template.factors:
        params = dict(spec.params)
        params.setdefault("metadata", metadata)
        params.setdefault("fundamentals", fundamentals or {})
        params.setdefault("benchmark_df", benchmark_df)
        signature_target = (
            FactorRegistry.get(spec.name)
            if universe_type == "etf"
            else calculator._factor_methods[spec.name]
        )
        accepted = inspect.signature(signature_target).parameters
        filtered_params = {key: value for key, value in params.items() if key in accepted}
        raw = calculator.calculate(df, spec.name, **filtered_params)
        series = _ensure_series(df, raw)
        factor_series[spec.name] = series
        latest_value = series.iloc[-1] if len(series) > 0 else None
        if pd.isna(latest_value):
            latest_value = None
        factor_values[spec.name] = latest_value
        scored = _score_series(series, spec.scorer)
        score += scored * spec.weight
        factor_contributions[spec.name] = float(scored.iloc[-1]) * spec.weight if len(scored) > 0 else 0.0

    score = score / total_weight
    backtest = _simulate_signals(df, score, template.buy_threshold, template.sell_threshold)
    latest_score = float(score.iloc[-1]) if len(score) > 0 and pd.notna(score.iloc[-1]) else 0.0

    return {
        "combo": template.name,
        "description": template.description,
        "buy_threshold": template.buy_threshold,
        "sell_threshold": template.sell_threshold,
        "factor_values": factor_values,
        "factor_contributions": factor_contributions,
        "composite_score": latest_score,
        "score_series": score,
        "backtest": backtest,
    }
