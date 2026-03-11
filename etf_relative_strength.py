"""
ETF Relative Strength Analysis
相对沪深300/中证500/中证1000的强弱比较，以及强弱切换阶段判断。
"""

from typing import Dict, List, Optional

import pandas as pd


BENCHMARK_ETFS = [
    {"key": "hs300", "code": "510300", "name": "沪深300ETF"},
    {"key": "zz500", "code": "510500", "name": "中证500ETF"},
    {"key": "zz1000", "code": "512100", "name": "中证1000ETF"},
]

CYCLICAL_KEYWORDS = (
    "煤炭", "钢铁", "有色", "证券", "银行", "地产", "房地产",
    "建材", "化工", "能源", "油气", "资源", "周期", "机械",
)


def _pct_change(series: pd.Series, periods: int) -> Optional[float]:
    if len(series) <= periods:
        return None
    value = series.pct_change(periods=periods).iloc[-1]
    return None if pd.isna(value) else float(value * 100)


def _calc_pair_strength(target_df: pd.DataFrame, bench_df: pd.DataFrame,
                        benchmark: Dict, is_self: bool) -> Dict:
    merged = target_df[["date", "close"]].rename(columns={"close": "target_close"}).merge(
        bench_df[["date", "close"]].rename(columns={"close": "bench_close"}),
        on="date",
        how="inner",
    )
    if len(merged) < 61:
        return {
            "code": benchmark["code"],
            "name": benchmark["name"],
            "status": "数据不足",
            "color": "yellow",
            "excess_return_20": None,
            "excess_return_60": None,
            "relative_momentum_20": None,
            "relative_momentum_60": None,
            "reason": "历史重叠样本不足",
        }

    target_close = merged["target_close"]
    bench_close = merged["bench_close"]
    ratio = target_close / bench_close.replace(0, pd.NA)

    target_ret_20 = _pct_change(target_close, 20)
    target_ret_60 = _pct_change(target_close, 60)
    target_ret_120 = _pct_change(target_close, 120)
    bench_ret_20 = _pct_change(bench_close, 20)
    bench_ret_60 = _pct_change(bench_close, 60)
    bench_ret_120 = _pct_change(bench_close, 120)
    rel_mom_20 = _pct_change(ratio, 20)
    rel_mom_60 = _pct_change(ratio, 60)

    excess_20 = None if target_ret_20 is None or bench_ret_20 is None else target_ret_20 - bench_ret_20
    excess_60 = None if target_ret_60 is None or bench_ret_60 is None else target_ret_60 - bench_ret_60
    excess_120 = None if target_ret_120 is None or bench_ret_120 is None else target_ret_120 - bench_ret_120

    if is_self:
        return {
            "code": benchmark["code"],
            "name": benchmark["name"],
            "status": "基准",
            "color": "blue",
            "excess_return_20": 0.0,
            "excess_return_60": 0.0,
            "excess_return_120": 0.0,
            "relative_momentum_20": 0.0,
            "relative_momentum_60": 0.0,
            "reason": "当前 ETF 即该基准代理",
        }

    score = 0
    if excess_20 is not None:
        if excess_20 > 1.0:
            score += 1
        elif excess_20 < -1.0:
            score -= 1
    if excess_60 is not None:
        if excess_60 > 2.0:
            score += 1
        elif excess_60 < -2.0:
            score -= 1
    if rel_mom_20 is not None:
        if rel_mom_20 > 1.0:
            score += 1
        elif rel_mom_20 < -1.0:
            score -= 1

    if score >= 2:
        status, color = "强于指数", "green"
    elif score <= -2:
        status, color = "弱于指数", "red"
    else:
        status, color = "接近指数", "yellow"

    reasons: List[str] = []
    if excess_20 is not None:
        reasons.append(f"20日超额 {excess_20:+.2f}%")
    if excess_60 is not None:
        reasons.append(f"60日超额 {excess_60:+.2f}%")
    if excess_120 is not None:
        reasons.append(f"120日超额 {excess_120:+.2f}%")
    if rel_mom_20 is not None:
        reasons.append(f"相对动量 {rel_mom_20:+.2f}%")

    return {
        "code": benchmark["code"],
        "name": benchmark["name"],
        "status": status,
        "color": color,
        "excess_return_20": excess_20,
        "excess_return_60": excess_60,
        "excess_return_120": excess_120,
        "relative_momentum_20": rel_mom_20,
        "relative_momentum_60": rel_mom_60,
        "reason": "，".join(reasons[:3]) if reasons else "暂无明显超额特征",
    }


def _is_cyclical(name: str) -> bool:
    return any(keyword in (name or "") for keyword in CYCLICAL_KEYWORDS)


def _build_rotation_summary(comparisons: List[Dict], cyclical: bool) -> Dict:
    valid = [c for c in comparisons if c["status"] != "基准" and c["excess_return_20"] is not None]
    if not valid:
        return {
            "phase": "未知",
            "color": "yellow",
            "reason": "基准对比数据不足，暂无法判断强弱切换",
        }

    avg_20 = sum(c["excess_return_20"] for c in valid if c["excess_return_20"] is not None) / len(valid)
    avg_60 = sum(c["excess_return_60"] for c in valid if c["excess_return_60"] is not None) / len(valid)

    if avg_20 > 1.0 and avg_60 > 1.0:
        phase, color = "持续走强", "green"
        reason = "短中期均跑赢主流基准，强势延续"
    elif avg_20 > 1.0 and avg_60 <= 0:
        phase, color = "弱转强", "green"
        reason = "短期超额回升，正在从弱势转向强势"
    elif avg_20 < -1.0 and avg_60 > 0:
        phase, color = "强转弱", "yellow"
        reason = "中期仍强，但短期超额回落，需警惕见顶钝化"
    elif avg_20 < -1.0 and avg_60 < -1.0:
        phase, color = "持续走弱", "red"
        reason = "短中期均弱于基准，当前应偏防守"
    else:
        phase, color = "胶着震荡", "yellow"
        reason = "相对强弱没有形成一致方向"

    if cyclical and phase in ("弱转强", "强转弱", "持续走强"):
        reason += "；行业/周期风格可能处于轮动切换阶段"

    return {"phase": phase, "color": color, "reason": reason}


def _build_rotation_advice(rotation: Dict, cyclical: bool) -> Dict:
    phase = rotation.get("phase", "未知")
    if phase == "持续走强":
        action, color = "持有", "green"
        reason = "相对强势已形成，可继续跟踪强势延续"
    elif phase == "弱转强":
        action, color = "试错", "green"
        reason = "强弱切换刚启动，适合小仓位观察或分批试错"
    elif phase == "强转弱":
        action, color = "减仓", "yellow"
        reason = "短期相对强度回落，适合降低仓位并等待再确认"
    elif phase == "持续走弱":
        action, color = "回避", "red"
        reason = "持续弱于主流基准，应优先防守或等待企稳"
    elif phase == "胶着震荡":
        action, color = "观察", "yellow"
        reason = "尚未形成稳定超额方向，耐心等待切换信号"
    else:
        action, color = "观察", "yellow"
        reason = "数据不足，先观察相对强弱变化"

    if cyclical:
        reason += "；周期/行业 ETF 更适合结合轮动节奏分段操作"

    return {"action": action, "color": color, "reason": reason}


def _build_relative_history(target_df: pd.DataFrame, benchmark_frames: List[Dict], lookback: int = 250) -> Dict:
    series_out = []
    base_dates = None
    for item in benchmark_frames:
        benchmark = item["benchmark"]
        bench_df = item["df"]
        merged = target_df[["date", "close"]].rename(columns={"close": "target_close"}).merge(
            bench_df[["date", "close"]].rename(columns={"close": "bench_close"}),
            on="date",
            how="inner",
        ).tail(lookback)
        if len(merged) < 20:
            continue

        if benchmark["code"] == item["target_code"]:
            relative_index = pd.Series([100.0] * len(merged))
        else:
            ratio = merged["target_close"] / merged["bench_close"].replace(0, pd.NA)
            first = ratio.iloc[0]
            if pd.isna(first) or first == 0:
                continue
            relative_index = (ratio / first) * 100

        dates = merged["date"].dt.strftime("%Y-%m-%d").tolist()
        if base_dates is None or len(dates) < len(base_dates):
            base_dates = dates
        series_out.append({
            "code": benchmark["code"],
            "name": benchmark["name"],
            "values": [None if pd.isna(v) else round(float(v), 2) for v in relative_index.tolist()],
        })

    if not series_out:
        return {"dates": [], "series": []}

    min_len = min(len(item["values"]) for item in series_out)
    return {
        "dates": base_dates[-min_len:] if base_dates else [],
        "series": [
            {"code": item["code"], "name": item["name"], "values": item["values"][-min_len:]}
            for item in series_out
        ],
    }


def _build_window_summary(comparisons: List[Dict]) -> Dict:
    valid = [c for c in comparisons if c["status"] != "基准"]
    windows = {}
    for period in (20, 60, 120):
        key = f"excess_return_{period}"
        values = [c.get(key) for c in valid if c.get(key) is not None]
        avg = sum(values) / len(values) if values else None
        if avg is None:
            label, color = "数据不足", "yellow"
        elif avg > 1.0:
            label, color = "偏强", "green"
        elif avg < -1.0:
            label, color = "偏弱", "red"
        else:
            label, color = "胶着", "yellow"
        windows[str(period)] = {"avg_excess": avg, "label": label, "color": color}
    return windows


def analyze_relative_strength(fetcher, etf_code: str, etf_name: str = "", days: int = 320) -> Dict:
    target_df = fetcher.get_historical_data(etf_code, days=days)
    if target_df is None or len(target_df) < 61:
        rotation = {"phase": "未知", "color": "yellow", "reason": "目标 ETF 历史数据不足"}
        return {
            "benchmarks": [],
            "rotation": rotation,
            "rotation_advice": _build_rotation_advice(rotation, _is_cyclical(etf_name)),
            "window_summary": {},
            "history": {"dates": [], "series": []},
            "cyclical": _is_cyclical(etf_name),
        }

    comparisons = []
    benchmark_frames = []
    for benchmark in BENCHMARK_ETFS:
        bench_df = target_df if benchmark["code"] == etf_code else fetcher.get_historical_data(benchmark["code"], days=days)
        if bench_df is None or len(bench_df) < 61:
            comparisons.append({
                "code": benchmark["code"],
                "name": benchmark["name"],
                "status": "数据不足",
                "color": "yellow",
                "excess_return_20": None,
                "excess_return_60": None,
                "excess_return_120": None,
                "relative_momentum_20": None,
                "relative_momentum_60": None,
                "reason": "基准数据不足",
            })
            continue
        benchmark_frames.append({"benchmark": benchmark, "df": bench_df, "target_code": etf_code})
        comparisons.append(_calc_pair_strength(target_df, bench_df, benchmark, benchmark["code"] == etf_code))

    cyclical = _is_cyclical(etf_name)
    rotation = _build_rotation_summary(comparisons, cyclical)
    return {
        "benchmarks": comparisons,
        "rotation": rotation,
        "rotation_advice": _build_rotation_advice(rotation, cyclical),
        "window_summary": _build_window_summary(comparisons),
        "history": _build_relative_history(target_df, benchmark_frames),
        "cyclical": cyclical,
    }
