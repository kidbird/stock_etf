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

app = Flask(__name__)

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

    etfs = [{'code': c, 'name': mapper.get_etf_name(c) or '', 'wide': mapper.is_wide_basis(c)}
            for c in codes]

    if q:
        q_lower = q.lower()
        etfs = [e for e in etfs if q_lower in e['code'] or q_lower in e['name'].lower()]

    return jsonify({'total': len(etfs), 'etfs': etfs[:200]})  # 最多返回 200 条


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

    mapper = ETFCodeMapper()
    etf_name = mapper.get_etf_name(code) or '未知ETF'

    # ── 历史数据 ──────────────────────────────────────────────
    df = _fetcher.get_historical_data(code, days=days)
    if df is None or len(df) < 30:
        return jsonify({'error': '历史数据不足，请先运行 --download'}), 400

    dates = df['date'].dt.strftime('%Y-%m-%d').tolist()

    # ── 回测 ──────────────────────────────────────────────────
    _engine.config = BacktestConfig()
    result = _engine.run(df, strategy, {})

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
        'profit_factor':     _safe(result.profit_factor) if result.profit_factor != float('inf') else 999,
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

    return jsonify({
        'code':     code,
        'name':     etf_name,
        'quote':    quote,
        'backtest': backtest,
        'factors':  factors,
        'advice':   advice,
        'strategy': strategy,
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


# ── 主页 ──────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    print("ETF 分析系统 Web 服务启动中...")
    print("访问 http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
