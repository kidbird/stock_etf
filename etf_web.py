"""
ETF Web Dashboard - Flask 后端
提供 REST API，供前端调用 ETF 分析数据
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from etf_data import ETFCodeMapper, ETFDataFetcher, _get_storage
from etf_factors import ETFFactorCalculator
from etf_backtest import ETFBacktestEngine, BacktestConfig
from etf_advisor import ETFInvestmentAdvisor
from etf_relative_strength import analyze_relative_strength

app = Flask(__name__)

_PARAM_SPECS = {
    "rsi": {"period": int, "oversold": float, "overbought": float},
    "macd": {"fast": int, "slow": int, "signal": int},
    "trend_filter_macd": {
        "fast": int, "slow": int, "signal": int, "adx_period": int,
        "adx_min": float, "ma_alignment_min": float,
    },
    "supertrend_follow": {
        "period": int, "multiplier": float, "roc_period": int, "roc_min": float,
    },
    "donchian_breakout": {"window": int, "adx_period": int, "adx_min": float},
    "ma_cross": {"short_period": int, "long_period": int},
    "bollinger": {"window": int, "num_std": float},
}

# ── JSON 序列化（处理 numpy 类型）────────────────────────────────────────────

class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Timestamp): return obj.strftime('%Y-%m-%d')
        return super().default(obj)

app.json_encoder = _Encoder

def _safe(v):
    """将 numpy 标量转为 Python 原生类型，NaN → None。"""
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    return v


def _parse_strategy_params(strategy: str) -> dict:
    raw = request.args.get("params", "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}

    spec = _PARAM_SPECS.get(strategy, {})
    params = {}
    for key, caster in spec.items():
        if key not in payload:
            continue
        value = payload[key]
        if value in ("", None):
            continue
        try:
            params[key] = caster(value)
        except (TypeError, ValueError):
            continue
    return params


def _build_market_regime(factors: dict) -> dict:
    adx = factors.get("adx")
    ma_alignment = factors.get("ma_alignment")
    roc20 = factors.get("roc20")
    supertrend_trend = factors.get("supertrend_trend")

    score = 0
    reasons = []

    if adx is not None:
        if adx >= 25:
            score += 2
            reasons.append(f"ADX={adx:.1f}，趋势强度较高")
        elif adx < 18:
            score -= 2
            reasons.append(f"ADX={adx:.1f}，趋势较弱")

    if ma_alignment is not None:
        if ma_alignment >= 0.5:
            score += 2
            reasons.append("均线多头排列")
        elif ma_alignment <= -0.5:
            score -= 2
            reasons.append("均线空头排列")
        else:
            reasons.append("均线排列分化")

    if roc20 is not None:
        if roc20 > 2:
            score += 1
            reasons.append(f"20日动量为正 ({roc20:.2f}%)")
        elif roc20 < -2:
            score -= 1
            reasons.append(f"20日动量为负 ({roc20:.2f}%)")

    if supertrend_trend == 1:
        score += 1
        reasons.append("Supertrend 指向多头")
    elif supertrend_trend == -1:
        score -= 1
        reasons.append("Supertrend 指向空头")

    if score >= 3:
        label = "趋势"
        color = "green"
    elif score <= -3:
        label = "风险"
        color = "red"
    else:
        label = "震荡"
        color = "yellow"

    confidence = min(1.0, 0.4 + abs(score) * 0.12)
    return {
        "label": label,
        "color": color,
        "score": score,
        "confidence": confidence,
        "reasons": reasons[:4],
    }


def _build_strategy_fit(strategy: str, market_regime: dict) -> dict:
    regime = market_regime.get("label", "未知")
    fit_map = {
        "趋势": {
            "trend_filter_macd": ("适配", "green", "趋势过滤后再入场，适合当前环境"),
            "supertrend_follow": ("适配", "green", "趋势跟随策略与当前状态一致"),
            "donchian_breakout": ("适配", "green", "突破策略更容易在趋势环境中兑现"),
            "macd": ("适配", "green", "MACD 更适合已有方向的市场"),
            "ma_cross": ("适配", "green", "均线交叉在趋势行情中更稳定"),
            "rsi": ("一般", "yellow", "可用于回调买入，但不宜单独逆势判断"),
            "bollinger": ("不适合", "red", "布林带逆势交易在趋势中容易失效"),
        },
        "震荡": {
            "rsi": ("适配", "green", "RSI 更适合区间震荡与回归交易"),
            "bollinger": ("适配", "green", "布林带在箱体区间更有效"),
            "macd": ("一般", "yellow", "MACD 在震荡区间容易来回打脸"),
            "ma_cross": ("不适合", "red", "均线交叉在震荡中会反复交易"),
            "trend_filter_macd": ("一般", "yellow", "趋势过滤会减少误判，但信号可能偏少"),
            "supertrend_follow": ("不适合", "red", "趋势跟随在震荡区间不占优"),
            "donchian_breakout": ("不适合", "red", "突破策略容易遇到假突破"),
        },
        "风险": {
            "rsi": ("一般", "yellow", "仅适合做超跌观察，不宜激进开仓"),
            "bollinger": ("一般", "yellow", "可观察极端偏离，但需严格控制风险"),
            "macd": ("不适合", "red", "趋势向下时 MACD 多头信号可靠性低"),
            "ma_cross": ("不适合", "red", "下行风险阶段应优先防守"),
            "trend_filter_macd": ("不适合", "red", "当前环境不支持趋势做多"),
            "supertrend_follow": ("不适合", "red", "趋势跟随更容易持续空仓"),
            "donchian_breakout": ("不适合", "red", "风险阶段突破成功率较低"),
        },
    }
    label, color, reason = fit_map.get(regime, {}).get(
        strategy, ("一般", "yellow", "当前状态下暂无明确适配结论")
    )
    return {"label": label, "color": color, "reason": reason}


def _rotation_phase_rank(phase: str) -> int:
    order = {
        "持续走强": 5,
        "弱转强": 4,
        "胶着震荡": 3,
        "强转弱": 2,
        "持续走弱": 1,
        "未知": 0,
    }
    return order.get(phase, 0)


def _build_rotation_ranking(category: str, rs_window: int, top: int) -> list:
    mapper = ETFCodeMapper()
    if category == "industry":
        codes = mapper.get_industry_codes()
    elif category == "wide_basis":
        codes = mapper.get_wide_basis_codes()
    else:
        codes = mapper.get_all_codes()

    if not codes:
        ETFCodeMapper.load_from_akshare()
        mapper = ETFCodeMapper()
        if category == "industry":
            codes = mapper.get_industry_codes()
        elif category == "wide_basis":
            codes = mapper.get_wide_basis_codes()
        else:
            codes = mapper.get_all_codes()

    ranked = []
    for code in codes:
        name = mapper.get_etf_name(code) or ""
        relative = analyze_relative_strength(_fetcher, code, etf_name=name, days=320)
        window = relative.get("window_summary", {}).get(str(rs_window), {})
        ranked.append({
            "code": code,
            "name": name,
            "phase": relative.get("rotation", {}).get("phase", "未知"),
            "phase_color": relative.get("rotation", {}).get("color", "yellow"),
            "action": relative.get("rotation_advice", {}).get("action", "观察"),
            "action_color": relative.get("rotation_advice", {}).get("color", "yellow"),
            "avg_excess": window.get("avg_excess"),
            "window_label": window.get("label", "数据不足"),
            "phase_rank": _rotation_phase_rank(relative.get("rotation", {}).get("phase", "未知")),
        })

    ranked.sort(key=lambda item: (
        item["phase_rank"],
        -9999 if item["avg_excess"] is None else item["avg_excess"],
    ), reverse=True)
    return ranked[:top]

# ── 全局单例 ──────────────────────────────────────────────────────────────────

_fetcher = ETFDataFetcher()
_calculator = ETFFactorCalculator()
_engine = ETFBacktestEngine()
_advisor = ETFInvestmentAdvisor()


# ── API: ETF 列表 / 搜索 ──────────────────────────────────────────────────────

@app.route('/api/etfs')
def api_etfs():
    q = request.args.get('q', '').strip()
    category = request.args.get('category', 'all')  # all | wide | industry

    # 确保动态列表已加载
    if not ETFCodeMapper._live:
        ETFCodeMapper.load_from_akshare()

    mapper = ETFCodeMapper()
    if category == 'wide':
        codes = mapper.get_wide_basis_codes()
    elif category == 'industry':
        codes = mapper.get_industry_codes()
    else:
        codes = mapper.get_all_codes()

    etfs = []
    for code in codes:
        meta = mapper.get_etf_metadata(code)
        etfs.append({
            'code': code,
            'name': meta['name'] or '',
            'wide': meta['category'] == 'wide_basis',
            'category': meta['category'],
            'category_label': meta['category_label'],
            'sector': meta['sector'],
            'tags': meta['tags'],
        })

    if q:
        q_lower = q.lower()
        etfs = [e for e in etfs if q_lower in e['code'] or q_lower in e['name'].lower()]

    return jsonify({'total': len(etfs), 'etfs': etfs[:200]})  # 最多返回 200 条


@app.route('/api/metadata')
def api_metadata():
    category = request.args.get('category', 'all').strip()
    refresh = request.args.get('refresh', '').lower() in {'1', 'true', 'yes'}
    q = request.args.get('q', '').strip().lower()

    export_category = None if category == 'all' else category
    rows = ETFCodeMapper.get_metadata_table(category=export_category, refresh=refresh)
    if q:
        rows = [
            row for row in rows
            if q in row['code'].lower()
            or q in row['name'].lower()
            or q in row['category'].lower()
            or q in row['sector'].lower()
            or any(q in tag.lower() for tag in row['tags'])
        ]

    return jsonify({
        'total': len(rows),
        'category': category,
        'refresh': refresh,
        'items': rows[:500],
    })


# ── API: 实时行情 ──────────────────────────────────────────────────────────────

@app.route('/api/quote/<code>')
def api_quote(code):
    q = _fetcher.get_realtime_quote(code)
    if not q:
        return jsonify({'error': '获取行情失败'}), 404
    return jsonify(q)


# ── API: 历史价格（用于 K 线 / 折线图）────────────────────────────────────────

@app.route('/api/history/<code>')
def api_history(code):
    days = int(request.args.get('days', 250))
    df = _fetcher.get_historical_data(code, days=days)
    if df is None or len(df) == 0:
        return jsonify({'error': '获取历史数据失败'}), 404

    return jsonify({
        'code': code,
        'dates': df['date'].dt.strftime('%Y-%m-%d').tolist(),
        'open':  [_safe(v) for v in df['open']],
        'high':  [_safe(v) for v in df['high']],
        'low':   [_safe(v) for v in df['low']],
        'close': [_safe(v) for v in df['close']],
        'volume':[_safe(v) for v in df['volume']],
    })


# ── API: 完整分析（回测 + 因子 + 建议）────────────────────────────────────────

@app.route('/api/analysis/<code>')
def api_analysis(code):
    strategy = request.args.get('strategy', 'macd')
    days     = int(request.args.get('days', 500))
    params   = _parse_strategy_params(strategy)

    mapper = ETFCodeMapper()
    metadata = mapper.get_etf_metadata(code)
    etf_name = metadata.get('name') or '未知ETF'

    # ── 历史数据 ──────────────────────────────────────────────
    df = _fetcher.get_historical_data(code, days=days)
    if df is None or len(df) < 30:
        return jsonify({'error': '历史数据不足，请先运行 --download'}), 400

    dates = df['date'].dt.strftime('%Y-%m-%d').tolist()

    # ── 回测 ──────────────────────────────────────────────────
    _engine.config = BacktestConfig()
    result = _engine.run(df, strategy, params)

    equity = [_safe(v) for v in result.equity_curve['equity'].tolist()]
    # 对齐日期（equity_curve 长度 = len(df)，索引从 0 开始）
    eq_dates = dates[:len(equity)]

    trades_out = []
    for t in result.trades:
        trades_out.append({
            'action':       t.get('action'),
            'date_idx':     int(t.get('date', 0)),
            'price':        _safe(t.get('price', 0)),
            'profit':       _safe(t.get('profit', 0)),
            'holding_days': int(t.get('holding_days', 0)),
        })

    backtest = {
        'total_return':      _safe(result.total_return),
        'annualized_return': _safe(result.annualized_return),
        'sharpe_ratio':      _safe(result.sharpe_ratio),
        'max_drawdown':      _safe(result.max_drawdown),
        'win_rate':          _safe(result.win_rate),
        'profit_factor':     None if result.profit_factor == float('inf') else _safe(result.profit_factor),
        'profit_factor_infinite': result.profit_factor == float('inf'),
        'total_trades':      result.total_trades,
        'winning_trades':    result.winning_trades,
        'losing_trades':     result.losing_trades,
        'equity':            equity,
        'eq_dates':          eq_dates,
        'trades':            trades_out,
    }

    # ── 因子计算（用近 200 日数据）────────────────────────────
    df_factor = _fetcher.get_historical_data(code, days=200)
    factors = {}
    if df_factor is not None and len(df_factor) > 60:
        try:
            factors['rsi']   = _safe(_calculator.calculate(df_factor, 'rsi', period=21).iloc[-1])
            factors['trend'] = _safe(_calculator.calculate(df_factor, 'trend_strength', window=21).iloc[-1])
            factors['roc20'] = _safe(_calculator.calculate(df_factor, 'roc', period=20).iloc[-1])
            factors['roc60'] = _safe(_calculator.calculate(df_factor, 'roc', period=60).iloc[-1])

            ma_align = _calculator.calculate(df_factor, 'ma_alignment', periods=[9, 21, 50])
            factors['ma_alignment'] = _safe(ma_align.iloc[-1])

            adx_df = _calculator.calculate(df_factor, 'adx', period=14)
            factors['adx']      = _safe(adx_df['adx'].iloc[-1])
            factors['di_plus']  = _safe(adx_df['di_plus'].iloc[-1])
            factors['di_minus'] = _safe(adx_df['di_minus'].iloc[-1])

            st_df = _calculator.calculate(df_factor, 'supertrend', period=10, multiplier=3.0)
            factors['supertrend_trend'] = int(st_df['trend'].iloc[-1])
            factors['supertrend_line']  = _safe(st_df['supertrend'].iloc[-1])

            aroon_df = _calculator.calculate(df_factor, 'aroon', period=25)
            factors['aroon_up']   = _safe(aroon_df['aroon_up'].iloc[-1])
            factors['aroon_down'] = _safe(aroon_df['aroon_down'].iloc[-1])
            factors['aroon_osc']  = _safe(aroon_df['oscillator'].iloc[-1])

            factors['lr_slope'] = _safe(_calculator.calculate(
                df_factor, 'linear_regression_slope', window=21).iloc[-1])

            dc_df = _calculator.calculate(df_factor, 'donchian_channel', window=20)
            factors['dc_position'] = _safe(dc_df['position'].iloc[-1])

            ich = _calculator.calculate(df_factor, 'ichimoku')
            factors['ichimoku_cloud'] = int(ich['price_vs_cloud'].iloc[-1])
        except Exception as e:
            app.logger.warning(f'因子计算异常 {code}: {e}')

    # ── 投资建议 ──────────────────────────────────────────────
    quote = _fetcher.get_realtime_quote(code)
    current_price = quote['latest_price'] if quote else 0
    adv = _advisor.generate_advice(code, etf_name, result, current_price, {
        'rsi':              factors.get('rsi'),
        'trend':            factors.get('trend'),
        'ma_alignment':     factors.get('ma_alignment'),
        'adx':              factors.get('adx'),
        'supertrend_trend': factors.get('supertrend_trend'),
    })

    advice = {
        'signal':       adv.signal.value,
        'score':        _safe(adv.score),
        'confidence':   _safe(adv.confidence),
        'reasons':      adv.reasons,
        'risks':        adv.risks,
        'target_price': _safe(adv.target_price) if adv.target_price else None,
        'stop_loss':    _safe(adv.stop_loss)    if adv.stop_loss    else None,
    }
    market_regime = _build_market_regime(factors) if factors else {
        "label": "未知",
        "color": "yellow",
        "score": 0,
        "confidence": 0.0,
        "reasons": ["因子数据不足，无法判断市场状态"],
    }
    strategy_fit = _build_strategy_fit(strategy, market_regime)
    relative_strength = analyze_relative_strength(_fetcher, code, etf_name=etf_name, days=320)

    return jsonify({
        'code':     code,
        'name':     etf_name,
        'metadata': metadata,
        'quote':    quote,
        'backtest': backtest,
        'factors':  factors,
        'market_regime': market_regime,
        'strategy_fit': strategy_fit,
        'relative_strength': relative_strength,
        'advice':   advice,
        'strategy': strategy,
        'params':   params,
        'days':     days,
    })


# ── API: 缓存统计 ──────────────────────────────────────────────────────────────

@app.route('/api/cache/stats')
def api_cache_stats():
    storage = _get_storage()
    df = storage.get_cache_stats()
    return jsonify({
        'etf_count': len(df),
        'total_rows': int(df['rows'].sum()) if not df.empty else 0,
        'items': df.to_dict(orient='records') if not df.empty else [],
    })


@app.route('/api/rotation-rank')
def api_rotation_rank():
    category = request.args.get('category', 'industry')
    if category not in {'industry', 'wide_basis', 'all'}:
        category = 'industry'
    rs_window = request.args.get('window', '60')
    try:
        rs_window = int(rs_window)
    except ValueError:
        rs_window = 60
    if rs_window not in {20, 60, 120, 250}:
        rs_window = 60

    try:
        top = int(request.args.get('top', 20))
    except ValueError:
        top = 20
    top = max(1, min(100, top))

    items = _build_rotation_ranking(category=category, rs_window=rs_window, top=top)
    return jsonify({
        'category': category,
        'window': rs_window,
        'top': top,
        'items': items,
    })


# ── 主页 ──────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    print("ETF 分析系统 Web 服务启动中...")
    print("访问 http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
