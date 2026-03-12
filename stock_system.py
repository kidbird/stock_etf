"""
Stock Analysis System - Main Entry
A股股票分析回测系统主入口
"""

import os
import sys
import argparse
import logging
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_data import StockDataFetcher, StockCodeMapper
from stock_factors import StockFactorCalculator
from stock_backtest import StockBacktestEngine, StockBacktestConfig
from etf_data import ETFDataFetcher
from factor_combo import list_combo_templates, run_combo_analysis

logger = logging.getLogger(__name__)


class StockAnalysisSystem:
    def __init__(self):
        self.data_fetcher = StockDataFetcher()
        self.factor_calculator = StockFactorCalculator()
        self.backtest_engine = StockBacktestEngine()
        self.market_fetcher = ETFDataFetcher()

    def list_all_stocks(self):
        return StockCodeMapper.get_all_codes()

    def get_quote(self, stock_code: str) -> Optional[Dict]:
        return self.data_fetcher.get_realtime_quote(stock_code)

    def get_historical_data(self, stock_code: str, days: int = 250):
        return self.data_fetcher.get_historical_data(stock_code, days)

    def run_backtest(
        self, stock_code: str, strategy: str, params: Dict = None, days: int = 500
    ):
        df = self.data_fetcher.get_historical_data(stock_code, days)
        if df is None or len(df) < 50:
            raise ValueError(f"股票 {stock_code} 数据不足，无法回测")

        params = params or {}
        return self.backtest_engine.run(df, strategy, params)

    def calculate_factors(self, stock_code: str, days: int = 250):
        df = self.data_fetcher.get_historical_data(stock_code, days)
        if df is None:
            return None
        return self.factor_calculator.calculate_all_factors(df)

    def analyze_factor_combo(self, stock_code: str, combo_name: str, days: int = 500):
        df = self.data_fetcher.get_historical_data(stock_code, days)
        if df is None or len(df) < 30:
            return None
        fundamentals = self.data_fetcher.get_stock_fundamentals(stock_code)
        benchmark_df = self.market_fetcher.get_historical_data("510300", max(days, 320))
        return run_combo_analysis(
            "stock",
            stock_code,
            df,
            metadata=fundamentals,
            combo_name=combo_name,
            benchmark_df=benchmark_df,
            fundamentals=fundamentals,
        )

    def full_analysis(
        self, stock_code: str, strategy: str = "stock_rsi", params: Dict = None, days: int = 500
    ):
        quote = self.get_quote(stock_code)
        if quote is None:
            raise ValueError(f"无法获取股票 {stock_code} 的行情数据")

        params = params or {}
        backtest_result = self.run_backtest(stock_code, strategy, params, days=days)
        factors = self.calculate_factors(stock_code, days=60)

        return {
            "quote": quote,
            "backtest": backtest_result,
            "factors": factors,
        }


def main():
    parser = argparse.ArgumentParser(description="A股股票分析回测系统")
    parser.add_argument("--code", type=str, help="股票代码，如 000001")
    parser.add_argument(
        "--strategy",
        type=str,
        default="stock_rsi",
        help="策略名称: stock_rsi, stock_macd, stock_supertrend, stock_bollinger_breakout, stock_volume_breakout, stock_dual_thrust",
    )
    parser.add_argument("--combo", type=str, choices=list_combo_templates("stock"), help="股票多因子组合名称")
    parser.add_argument("--days", type=int, default=500, help="回测天数")
    parser.add_argument("--list", action="store_true", help="列出所有股票")
    parser.add_argument("--list-combos", action="store_true", help="列出股票多因子组合")
    parser.add_argument("--list-factors", action="store_true", help="列出股票可用因子")
    parser.add_argument("--quote", action="store_true", help="查询实时行情")
    parser.add_argument("--factors", action="store_true", help="计算技术因子")
    parser.add_argument("--download", action="store_true", help="批量下载历史数据")

    args = parser.parse_args()

    if args.list:
        mapper = StockCodeMapper()
        mapper.load_from_akshare()
        codes = mapper.get_all_codes()
        print(f"股票总数: {len(codes)} 只")
        return

    if args.list_combos:
        print("股票多因子组合模板:")
        for combo in list_combo_templates("stock"):
            print(f"  {combo}")
        return

    if args.list_factors:
        system = StockAnalysisSystem()
        print("股票可用因子:")
        for factor in system.factor_calculator.get_available_factors():
            print(f"  {factor}")
        return

    if args.download:
        system = StockAnalysisSystem()
        mapper = StockCodeMapper()
        mapper.load_from_akshare()
        codes = mapper.get_all_codes()[:100]
        print(f"开始下载 {len(codes)} 只股票的历史数据...")
        results = system.data_fetcher.download_batch(codes, days=args.days)
        print(
            f"下载完成，成功 {sum(1 for v in results.values() if v > 0)} / {len(codes)}"
        )
        return

    if args.code:
        system = StockAnalysisSystem()

        if args.combo:
            result = system.analyze_factor_combo(args.code, args.combo, days=args.days)
            if not result:
                raise ValueError(f"股票 {args.code} 数据不足，无法运行多因子组合")
            print(f"股票: {args.code}")
            print(f"组合: {result['combo']} - {result['description']}")
            print(f"综合得分: {result['composite_score']:.2f}")
            print(f"总收益率: {result['backtest']['total_return'] * 100:.2f}%")
            print(f"年化收益率: {result['backtest']['annualized_return'] * 100:.2f}%")
            print("因子值:")
            for key, value in result["factor_values"].items():
                print(f"  {key}: {value}")
            return

        if args.quote:
            quote = system.get_quote(args.code)
            if quote:
                print(
                    f"{quote['name']} ({quote['code']}) - 最新价: {quote['latest_price']}, 涨跌幅: {quote['change_pct']:.2f}%"
                )
            return

        if args.factors:
            factors = system.calculate_factors(args.code)
            if factors:
                rsi = factors["rsi"].iloc[-1]
                macd = factors["macd"]
                adx = factors["adx"]["adx"].iloc[-1]
                print(f"RSI(14): {rsi:.1f}")
                print(f"ADX(14): {adx:.1f}")
                print(f"MACD: {macd['macd'].iloc[-1]:.2f}")
            return

        result = system.full_analysis(args.code, strategy=args.strategy, days=args.days)
        print(f"股票: {result['quote']['name']} ({result['quote']['code']})")
        print(f"最新价: {result['quote']['latest_price']}")
        print(f"回测结果:")
        print(f"  总收益率: {result['backtest'].total_return * 100:.2f}%")
        print(f"  年化收益率: {result['backtest'].annualized_return * 100:.2f}%")
        print(f"  夏普比率: {result['backtest'].sharpe_ratio:.2f}")
        print(f"  最大回撤: {result['backtest'].max_drawdown * 100:.2f}%")
        print(f"  胜率: {result['backtest'].win_rate * 100:.1f}%")
        print(f"  交易次数: {result['backtest'].total_trades}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
