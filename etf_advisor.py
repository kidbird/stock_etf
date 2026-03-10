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

        # RSI因子：超买扣分，超卖加分
        rsi = factors.get('rsi')
        if rsi is not None:
            if rsi < 30:
                score += 8
            elif rsi < 45:
                score += 3
            elif rsi > 70:
                score -= 8
            elif rsi > 55:
                score -= 3

        # 趋势强度因子：正趋势加分，负趋势扣分
        trend = factors.get('trend')
        if trend is not None:
            if trend > 0.05:
                score += 5
            elif trend > 0:
                score += 2
            elif trend < -0.05:
                score -= 5
            elif trend < 0:
                score -= 2

        # 均线排列：多头排列加分，空头排列扣分（中线核心信号）
        ma_align = factors.get('ma_alignment')
        if ma_align is not None:
            if ma_align >= 0.8:
                score += 10   # 完全多头排列
            elif ma_align >= 0.3:
                score += 5
            elif ma_align <= -0.8:
                score -= 10  # 完全空头排列
            elif ma_align <= -0.3:
                score -= 5

        # ADX：趋势成立时放大信号，震荡时降低置信度
        adx = factors.get('adx')
        if adx is not None:
            if adx > 30:
                score += 5   # 强趋势，信号更可靠
            elif adx < 20:
                score -= 5   # 震荡行情，趋势信号失效

        # Supertrend 方向：多头+8，空头-8
        st_trend = factors.get('supertrend_trend')
        if st_trend is not None:
            if st_trend == 1:
                score += 8
            elif st_trend == -1:
                score -= 8

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
        if rsi is not None and rsi < 30:
            reasons.append(f"RSI(21)={rsi:.1f} 处于超卖区间，存在反弹机会")
        elif rsi is not None and rsi > 70:
            reasons.append(f"RSI(21)={rsi:.1f} 处于超买区间，注意回调风险")
        if ma_align is not None and ma_align >= 0.8:
            reasons.append("9/21/50均线完全多头排列，中线趋势明确向上")
        elif ma_align is not None and ma_align <= -0.8:
            reasons.append("9/21/50均线完全空头排列，中线趋势明确向下")
        if adx is not None and adx > 25:
            reasons.append(f"ADX={adx:.1f} 趋势强度充分，中线信号可信度高")
        elif adx is not None and adx < 20:
            reasons.append(f"ADX={adx:.1f} 当前处于震荡格局，趋势信号仅供参考")
        if st_trend == 1:
            reasons.append("Supertrend 指向多头，建议持有或做多")
        elif st_trend == -1:
            reasons.append("Supertrend 指向空头，建议观望或止损")
        if trend is not None and trend > 0.05:
            reasons.append(f"价格高于21日均线 {trend*100:.1f}%，短期趋势向上")
        elif trend is not None and trend < -0.05:
            reasons.append(f"价格低于21日均线 {abs(trend)*100:.1f}%，短期趋势向下")
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
