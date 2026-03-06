"""
ETF Analysis and Backtest System - Main Entry
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from etf_data import ETFDataFetcher, ETFCodeMapper
from etf_factors import ETFFactorCalculator
from etf_backtest import ETFBacktestEngine, BacktestConfig
from etf_advisor import get_investment_advice, format_advice


class ETFAnalysisSystem:
    def __init__(self):
        self.data_fetcher = ETFDataFetcher()
        self.factor_calculator = ETFFactorCalculator()
        self.backtest_engine = ETFBacktestEngine()

    def list_all_etfs(self, category: str = "all"):
        mapper = ETFCodeMapper()
        if category == "wide_basis":
            codes = mapper.get_wide_basis_codes()
        elif category == "industry":
            codes = mapper.get_industry_codes()
        else:
            codes = mapper.get_all_codes()
        return [{"code": c, "name": mapper.get_etf_name(c), "type": "宽基" if mapper.is_wide_basis(c) else "行业"} for c in codes]

    def run_backtest(self, etf_code: str, strategy: str, params: dict = None, days: int = 500):
        params = params or {}
        self.backtest_engine.config = BacktestConfig()
        df = self.data_fetcher.get_historical_data(etf_code, days)
        if df is None:
            return None
        return self.backtest_engine.run(df, strategy, params)

    def get_investment_advice(self, etf_code: str, strategy: str, backtest_result, days: int = 60):
        etf_name = ETFCodeMapper.get_etf_name(etf_code) or "未知ETF"
        current_quote = self.data_fetcher.get_realtime_quote(etf_code)
        current_price = current_quote['latest_price'] if current_quote else 0
        df = self.data_fetcher.get_historical_data(etf_code, days)
        factors = {}
        if df is not None and len(df) > 20:
            try:
                factors['rsi'] = self.factor_calculator.calculate(df, 'rsi', period=14).iloc[-1]
                factors['trend'] = self.factor_calculator.calculate(df, 'trend_strength', window=20).iloc[-1]
            except: pass
        return get_investment_advice(etf_code, etf_name, backtest_result, current_price, factors)


    def full_analysis(self, etf_code: str, strategy: str = "rsi", params: dict = None):
        params = params or {}
        print(f"\n{'='*60}")
        print(f"ETF: {etf_code} - {ETFCodeMapper.get_etf_name(etf_code)}")
        print(f"策略: {strategy}")
        print(f"{'='*60}\n")

        quote = self.data_fetcher.get_realtime_quote(etf_code)
        if quote:
            print(f"当前价格: {quote['latest_price']}, 涨跌幅: {quote['change_pct']}%")

        result = self.run_backtest(etf_code, strategy, params)
        if result:
            print(f"总收益率: {result.total_return*100:.2f}%, 夏普比率: {result.sharpe_ratio:.2f}, 最大回撤: {result.max_drawdown*100:.2f}%")

        advice = self.get_investment_advice(etf_code, strategy, result)
        print(f"\n{format_advice(advice)}")
        return result


def main():
    parser = argparse.ArgumentParser(description="A股ETF分析和回测系统")
    parser.add_argument('--code', '-c', type=str, help='ETF代码')
    parser.add_argument('--strategy', '-s', type=str, default='rsi', choices=['rsi', 'macd', 'ma_cross', 'bollinger'])
    parser.add_argument('--list', '-l', action='store_true', help='列出所有ETF')
    parser.add_argument('--quote', '-q', action='store_true', help='获取实时行情')
    parser.add_argument('--days', type=int, default=500, help='回测天数')
    args = parser.parse_args()

    system = ETFAnalysisSystem()

    if args.list:
        etfs = system.list_all_etfs()
        print(f"\nETF列表 ({len(etfs)}只)")
        for e in etfs:
            print(f"  {e['code']} - {e['name']} [{e['type']}]")

    elif args.quote and args.code:
        quote = system.data_fetcher.get_realtime_quote(args.code)
        if quote:
            print(f"{quote['name']} ({quote['code']}) - 最新价: {quote['latest_price']}, 涨跌幅: {quote['change_pct']}%")

    elif args.code:
        system.full_analysis(args.code, args.strategy)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
