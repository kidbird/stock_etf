"""
ETF Investment Advisor Module
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    STRONG_BUY = "强烈买入"
    BUY = "买入"
    HOLD = "持有"
    SELL = "卖出"
    STRONG_SELL = "强烈卖出"


@dataclass
class InvestmentAdvice:
    signal: Signal
    score: float
    reasons: List[str]
    risks: List[str]
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    confidence: float = 0.0


class ETFInvestmentAdvisor:
    def generate_advice(self, etf_code: str, etf_name: str, backtest_result, current_price: float, factors: Dict) -> InvestmentAdvice:
        score = 50.0

        if backtest_result.total_return > 0:
            score += min(backtest_result.total_return * 50, 20)
        else:
            score += max(backtest_result.total_return * 30, -20)


        if backtest_result.sharpe_ratio > 1:
            score += 10
        elif backtest_result.sharpe_ratio < 0:
            score -= 10

        if abs(backtest_result.max_drawdown) < 0.1:
            score += 10
        elif abs(backtest_result.max_drawdown) > 0.25:
            score -= 10

        if backtest_result.win_rate > 0.5:
            score += 5
        elif backtest_result.win_rate < 0.4:
            score -= 5

        score = max(0, min(100, score))

        if score >= 75:
            signal = Signal.STRONG_BUY
            confidence = min(0.9, 0.5 + score / 200)
        elif score >= 60:
            signal = Signal.BUY
            confidence = 0.7
        elif score >= 40:
            signal = Signal.HOLD
            confidence = 0.5
        elif score >= 25:
            signal = Signal.SELL
            confidence = 0.6
        else:
            signal = Signal.STRONG_SELL
            confidence = 0.7

        reasons = []
        if backtest_result.total_return > 0:
            reasons.append(f"策略历史收益表现良好，总收益率为 {backtest_result.total_return*100:.2f}%")
        if backtest_result.win_rate > 0.5:
            reasons.append(f"策略胜率 {backtest_result.win_rate*100:.1f}%超过50%")
        if not reasons:
            reasons.append("当前市场环境下，建议谨慎观望")

        risks = ["市场有风险，投资需谨慎", "过往业绩不代表未来表现"]


        target_price = round(current_price * 1.1, 2) if backtest_result.total_return > 0 else None
        stop_loss = round(current_price * 0.9, 2) if backtest_result.max_drawdown < 0 else None

        return InvestmentAdvice(signal=signal, score=score, reasons=reasons, risks=risks,
                            target_price=target_price, stop_loss=stop_loss, confidence=confidence)


def get_investment_advice(etf_code: str, etf_name: str, backtest_result, current_price: float, factors: Dict) -> InvestmentAdvice:
    return ETFInvestmentAdvisor().generate_advice(etf_code, etf_name, backtest_result, current_price, factors)


def format_advice(advice: InvestmentAdvice) -> str:
    emoji = {Signal.STRONG_BUY: "🚀", Signal.BUY: "✅", Signal.HOLD: "⏸️", Signal.SELL: "⚠️", Signal.STRONG_SELL: "🛑"}
    lines = [f"## {emoji[advice.signal]} {advice.signal.value}", f"**综合评分**: {advice.score:.1f}/100", f"**置信度**: {advice.confidence*100:.1f}%"]
    if advice.target_price:
        lines.append(f"**目标价**: {advice.target_price}元")
    if advice.stop_loss:
        lines.append(f"**止损价**: {advice.stop_loss}元")
    lines.append("### 建议理由")
    lines.extend([f"- {r}" for r in advice.reasons])
    lines.append("### 风险提示")
    lines.extend([f"- {r}" for r in advice.risks])
    return "\n".join(lines)
