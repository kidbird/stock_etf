"""
ETF Analysis and Backtest System - Main Entry
"""

import os, sys, argparse
import logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from concurrent.futures import ThreadPoolExecutor, as_completed

from etf_data import ETFDataFetcher, ETFCodeMapper, _get_storage
from etf_factors import ETFFactorCalculator
from etf_backtest import ETFBacktestEngine, BacktestConfig
from etf_advisor import get_investment_advice, format_advice
from etf_relative_strength import analyze_relative_strength

logger = logging.getLogger(__name__)


def _fmt_pct(v):
    if v is None:
        return "—"
    return f"{v:+.2f}%"


def format_relative_strength(relative: dict, rs_window: int = 60) -> str:
    if not relative:
        return "## 相对强弱\n暂无数据"

    lines = ["## 相对强弱"]
    rotation = relative.get("rotation", {})
    rotation_advice = relative.get("rotation_advice", {})
    window_summary = relative.get("window_summary", {})
    lines.append(
        f"轮动阶段: {rotation.get('phase', '未知')} | 建议: {rotation_advice.get('action', '观察')}"
    )
    if rotation.get("reason"):
        lines.append(f"说明: {rotation['reason']}")
    if rotation_advice.get("reason"):
        lines.append(f"操作: {rotation_advice['reason']}")
    selected_window = window_summary.get(str(rs_window))
    if selected_window:
        lines.append(
            f"{rs_window}日窗口: {selected_window.get('label', '数据不足')} | "
            f"平均超额 {_fmt_pct(selected_window.get('avg_excess'))}"
        )

    benchmarks = relative.get("benchmarks", [])
    if benchmarks:
        lines.append("对比:")
        for item in benchmarks:
            window_value = item.get(f"excess_return_{rs_window}")
            lines.append(
                f"- {item['name']}({item['code']}): {item['status']} | "
                f"{rs_window}日超额 {_fmt_pct(window_value)}"
            )
    return "\n".join(lines)


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


def format_rotation_ranking(items: list, rs_window: int, category: str) -> str:
    title_map = {"industry": "行业ETF", "wide_basis": "宽基ETF", "all": "全部ETF"}
    lines = [f"## {title_map.get(category, category)}轮动排序 ({rs_window}日窗口)"]
    if not items:
        lines.append("暂无可排序数据")
        return "\n".join(lines)

    for idx, item in enumerate(items, start=1):
        lines.append(
            f"{idx:>2}. {item['code']} {item['name']} | "
            f"{item['phase']} | {item['action']} | "
            f"{rs_window}日均超额 {_fmt_pct(item['avg_excess'])}"
        )
    return "\n".join(lines)


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
        items = []
        for code in codes:
            meta = mapper.get_etf_metadata(code)
            items.append({
                "code": code,
                "name": meta["name"],
                "type": meta["category_label"],
                "category": meta["category"],
                "sector": meta["sector"],
                "tags": meta["tags"],
            })
        return items

    def run_backtest(self, etf_code: str, strategy: str, params: dict = None, days: int = 500):
        params = params or {}
        self.backtest_engine.config = BacktestConfig()
        df = self.data_fetcher.get_historical_data(etf_code, days)
        if df is None:
            return None
        return self.backtest_engine.run(df, strategy, params)

    def get_investment_advice(self, etf_code: str, strategy: str, backtest_result, days: int = 60):
        etf_name = self.data_fetcher.mapper.get_etf_name(etf_code) or "未知ETF"
        current_quote = self.data_fetcher.get_realtime_quote(etf_code)
        current_price = current_quote['latest_price'] if current_quote else 0
        df = self.data_fetcher.get_historical_data(etf_code, days)
        factors = {}
        if df is not None and len(df) > 50:
            factor_jobs = {
                'rsi': lambda: self.factor_calculator.calculate(df, 'rsi', period=21).iloc[-1],
                'trend': lambda: self.factor_calculator.calculate(df, 'trend_strength', window=21).iloc[-1],
                'ma_alignment': lambda: self.factor_calculator.calculate(
                    df, 'ma_alignment', periods=[9, 21, 50]
                ).iloc[-1],
                'adx': lambda: self.factor_calculator.calculate(df, 'adx', period=14)['adx'].iloc[-1],
                'supertrend_trend': lambda: self.factor_calculator.calculate(
                    df, 'supertrend', period=10, multiplier=3.0
                )['trend'].iloc[-1],
            }
            for name, compute in factor_jobs.items():
                try:
                    factors[name] = compute()
                except Exception as exc:
                    logger.warning("因子计算失败 %s %s: %s", etf_code, name, exc)
        return get_investment_advice(etf_code, etf_name, backtest_result, current_price, factors)

    def full_analysis(self, etf_code: str, strategy: str = "rsi", params: dict = None, rs_window: int = 60):
        params = params or {}
        meta = ETFCodeMapper.get_etf_metadata(etf_code)
        etf_name = meta["name"]
        print(f"\n{'='*60}")
        print(f"ETF: {etf_code} - {etf_name}")
        print(
            f"分类: {meta['category_label']} | 行业: {meta['sector']} | "
            f"标签: {', '.join(meta['tags']) if meta['tags'] else '—'}"
        )
        print(f"策略: {strategy}")
        print(f"{'='*60}\n")

        quote = self.data_fetcher.get_realtime_quote(etf_code)
        if quote:
            print(f"当前价格: {quote['latest_price']}, 涨跌幅: {quote['change_pct']}%")

        result = self.run_backtest(etf_code, strategy, params)
        if result:
            print(f"总收益率: {result.total_return*100:.2f}%, 夏普比率: {result.sharpe_ratio:.2f}, 最大回撤: {result.max_drawdown*100:.2f}%")

        relative = analyze_relative_strength(self.data_fetcher, etf_code, etf_name=etf_name or "")
        print(f"\n{format_relative_strength(relative, rs_window=rs_window)}")

        advice = self.get_investment_advice(etf_code, strategy, result)
        print(f"\n{format_advice(advice)}")
        return result

    def print_rotation_only(self, etf_code: str, rs_window: int = 60):
        meta = ETFCodeMapper.get_etf_metadata(etf_code)
        etf_name = meta["name"]
        print(f"\n{'='*60}")
        print(f"ETF: {etf_code} - {etf_name}")
        print(
            f"分类: {meta['category_label']} | 行业: {meta['sector']} | "
            f"标签: {', '.join(meta['tags']) if meta['tags'] else '—'}"
        )
        print(f"相对强弱窗口: {rs_window} 日")
        print(f"{'='*60}\n")
        relative = analyze_relative_strength(self.data_fetcher, etf_code, etf_name=etf_name or "")
        print(format_relative_strength(relative, rs_window=rs_window))

    def rank_rotation(self, category: str = "industry", rs_window: int = 60, top: int = 20):
        mapper = ETFCodeMapper()
        if category == "industry":
            codes = mapper.get_industry_codes()
        elif category == "wide_basis":
            codes = mapper.get_wide_basis_codes()
        else:
            codes = mapper.get_all_codes()

        if not codes and category == "industry":
            ETFCodeMapper.load_from_akshare()
            codes = mapper.get_industry_codes()
        elif not codes and category == "wide_basis":
            ETFCodeMapper.load_from_akshare()
            codes = mapper.get_wide_basis_codes()

        ranked = []
        for code in codes:
            name = mapper.get_etf_name(code) or ""
            relative = analyze_relative_strength(self.data_fetcher, code, etf_name=name)
            window = relative.get("window_summary", {}).get(str(rs_window), {})
            ranked.append({
                "code": code,
                "name": name,
                "phase": relative.get("rotation", {}).get("phase", "未知"),
                "action": relative.get("rotation_advice", {}).get("action", "观察"),
                "avg_excess": window.get("avg_excess"),
                "phase_rank": _rotation_phase_rank(relative.get("rotation", {}).get("phase", "未知")),
            })

        ranked.sort(key=lambda item: (
            item["phase_rank"],
            -9999 if item["avg_excess"] is None else item["avg_excess"],
        ), reverse=True)
        return ranked[:top]


# ── 批量下载 / 增量更新 ───────────────────────────────────────────────────────

def _download_one(code: str, days: int, fetcher: ETFDataFetcher) -> tuple:
    """单只 ETF 下载任务，返回 (code, ok, rows)。"""
    try:
        df, meta = fetcher.get_historical_data(code, days=days, use_cache=True, return_metadata=True)
        rows = len(df) if df is not None else 0
        return code, True, {"rows": rows, "new_rows": int(meta.get("new_rows", 0))}
    except Exception as e:
        return code, False, str(e)


def cmd_download(days: int, workers: int):
    """批量下载所有宽基 ETF 历史数据到本地 SQLite。"""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    print("正在从 akshare 获取全量 ETF 列表...")
    ETFCodeMapper.load_from_akshare()
    codes = ETFCodeMapper.get_wide_basis_codes()
    print(f"宽基 ETF 共 {len(codes)} 只，开始下载 {days} 日历史数据（并发 {workers}）...\n")

    fetcher = ETFDataFetcher()
    ok_count = fail_count = total_rows = 0
    failed_codes = []

    bar = tqdm(total=len(codes), unit="只") if tqdm else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, c, days, fetcher): c for c in codes}
        for future in as_completed(futures):
            code, ok, detail = future.result()
            if ok:
                ok_count += 1
                total_rows += detail["new_rows"]
            else:
                fail_count += 1
                failed_codes.append(code)
            if bar:
                bar.set_postfix(ok=ok_count, fail=fail_count)
                bar.update(1)
            else:
                done = ok_count + fail_count
                if done % 50 == 0 or done == len(codes):
                    print(f"  进度 {done}/{len(codes)}  成功 {ok_count}  失败 {fail_count}")

    if bar:
        bar.close()

    print(f"\n下载完成：成功 {ok_count} 只，失败 {fail_count} 只，共写入 {total_rows:,} 条数据")
    if failed_codes:
        print(f"失败代码：{', '.join(failed_codes)}")


def cmd_update(workers: int):
    """增量更新本地已缓存 ETF 的最新数据（只补 last_date+1 到今天的缺口）。"""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    storage = _get_storage()
    codes = storage.get_cached_codes()
    if not codes:
        print("本地无缓存数据，请先运行 --download 下载历史数据。")
        return

    print(f"本地已缓存 {len(codes)} 只 ETF，开始增量更新（并发 {workers}）...\n")

    fetcher = ETFDataFetcher()
    ok_count = fail_count = new_rows = 0
    failed_codes = []

    # 更新时只需拉近 30 日，缓存逻辑会自动计算实际缺口
    bar = tqdm(total=len(codes), unit="只") if tqdm else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, c, 30, fetcher): c for c in codes}
        for future in as_completed(futures):
            code, ok, detail = future.result()
            if ok:
                ok_count += 1
                new_rows += detail["new_rows"]
            else:
                fail_count += 1
                failed_codes.append(code)
            if bar:
                bar.set_postfix(ok=ok_count, fail=fail_count)
                bar.update(1)
            else:
                done = ok_count + fail_count
                if done % 50 == 0 or done == len(codes):
                    print(f"  进度 {done}/{len(codes)}  成功 {ok_count}  失败 {fail_count}")

    if bar:
        bar.close()

    print(f"\n增量更新完成：成功 {ok_count} 只，失败 {fail_count} 只")
    if failed_codes:
        print(f"失败代码：{', '.join(failed_codes)}")


def cmd_cache_stats():
    """显示本地缓存统计信息。"""
    storage = _get_storage()
    df = storage.get_cache_stats()
    if df.empty:
        print("本地暂无缓存数据，请先运行 --download。")
        return

    total_rows = df["rows"].sum()
    print(f"\n本地缓存统计（共 {len(df)} 只 ETF，{total_rows:,} 条记录）")
    print(f"{'代码':<10}{'名称':<20}{'条数':>6}  {'最早日期':<12}{'最新日期':<12}{'最后更新'}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['code']:<10}{(row['name'] or ''):<20}{int(row['rows'] or 0):>6}  "
              f"{str(row['first_date'] or ''):<12}{str(row['last_date'] or ''):<12}"
              f"{str(row['updated_at'] or '')[:16]}")


def cmd_export_metadata(output: str, category: str = "all", refresh: bool = False):
    export_category = None if category == "all" else category
    rows = ETFCodeMapper.export_metadata_table(output, category=export_category, refresh=refresh)
    print(f"已导出 {rows} 条ETF分类记录到 {output}")


# ── CLI 入口 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="A股ETF分析和回测系统")
    parser.add_argument('--code', '-c', type=str, help='ETF代码')
    parser.add_argument('--strategy', '-s', type=str, default='rsi',
                        choices=['rsi', 'macd', 'ma_cross', 'bollinger',
                                 'trend_filter_macd', 'supertrend_follow', 'donchian_breakout'])
    parser.add_argument('--list', '-l', action='store_true', help='列出所有ETF')
    parser.add_argument('--quote', '-q', action='store_true', help='获取实时行情')
    parser.add_argument('--days', type=int, default=500,
                        help='回测/下载天数（默认500，推荐1500）')
    parser.add_argument('--refresh', action='store_true',
                        help='从akshare刷新ETF列表并显示统计')
    parser.add_argument('--download', action='store_true',
                        help='批量下载所有宽基ETF历史数据到本地SQLite')
    parser.add_argument('--update', action='store_true',
                        help='增量更新本地已缓存ETF的最新数据')
    parser.add_argument('--cache-stats', action='store_true',
                        help='查看本地缓存统计信息')
    parser.add_argument('--workers', type=int, default=5,
                        help='批量下载/更新并发数（默认5，建议不超过8）')
    parser.add_argument('--rs-window', type=int, default=60, choices=[20, 60, 120, 250],
                        help='相对强弱观察窗口（20/60/120/250，默认60）')
    parser.add_argument('--rotation-only', action='store_true',
                        help='只输出相对强弱与轮动建议，不跑完整分析')
    parser.add_argument('--rotation-rank', action='store_true',
                        help='批量输出ETF轮动排序')
    parser.add_argument('--rotation-category', type=str, default='industry',
                        choices=['industry', 'wide_basis', 'all'],
                        help='轮动排序分类（industry/wide_basis/all，默认industry）')
    parser.add_argument('--top', type=int, default=20,
                        help='轮动排序输出数量（默认20）')
    parser.add_argument('--metadata-table', action='store_true',
                        help='导出ETF分类表（CSV）')
    parser.add_argument('--metadata-output', type=str, default='etf_metadata.csv',
                        help='ETF分类表导出路径（默认 etf_metadata.csv）')
    parser.add_argument('--metadata-category', type=str, default='all',
                        choices=['all', 'wide_basis', 'industry', 'theme', 'commodity', 'bond', 'cross_border', 'other'],
                        help='ETF分类表导出筛选（默认all）')
    args = parser.parse_args()

    # --download 和 --update 不能同时使用
    if args.download and args.update:
        print("错误：--download 和 --update 不能同时使用")
        sys.exit(1)

    system = ETFAnalysisSystem()

    if args.download:
        cmd_download(days=args.days, workers=args.workers)

    elif args.update:
        cmd_update(workers=args.workers)

    elif args.cache_stats:
        cmd_cache_stats()

    elif args.metadata_table:
        cmd_export_metadata(
            output=args.metadata_output,
            category=args.metadata_category,
            refresh=args.refresh,
        )

    elif args.refresh:
        n = ETFCodeMapper.load_from_akshare()
        wide = ETFCodeMapper.get_wide_basis_codes()
        industry = ETFCodeMapper.get_industry_codes()
        print(f"\nakshare ETF总量: {n} 只")
        print(f"  宽基: {len(wide)} 只")
        print(f"  行业: {len(industry)} 只")

    elif args.list:
        etfs = system.list_all_etfs()
        print(f"\nETF列表 ({len(etfs)}只)")
        for e in etfs:
            tags = ",".join(e["tags"]) if e["tags"] else "—"
            print(f"  {e['code']} - {e['name']} [{e['type']}] 行业={e['sector']} 标签={tags}")

    elif args.quote and args.code:
        quote = system.data_fetcher.get_realtime_quote(args.code)
        if quote:
            print(f"{quote['name']} ({quote['code']}) - 最新价: {quote['latest_price']}, 涨跌幅: {quote['change_pct']}%")

    elif args.code and args.rotation_only:
        system.print_rotation_only(args.code, rs_window=args.rs_window)

    elif args.rotation_rank:
        ranked = system.rank_rotation(
            category=args.rotation_category,
            rs_window=args.rs_window,
            top=args.top,
        )
        print(format_rotation_ranking(ranked, rs_window=args.rs_window, category=args.rotation_category))

    elif args.code:
        system.full_analysis(args.code, args.strategy, rs_window=args.rs_window)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
