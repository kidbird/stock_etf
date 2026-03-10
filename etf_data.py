"""
ETF Data Module
A股ETF数据获取模块，支持上交所(510/511/512/515/516)和深交所(159)开头的ETF基金
数据源优先级：akshare -> Yahoo Finance -> 东方财富 -> 腾讯
"""

import requests
import pandas as pd
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from all_etf_codes import ETF_CODES
from etf_storage import ETFLocalStorage

# 模块级懒加载单例，所有 ETFDataFetcher 实例共用同一个 storage
_storage: Optional[ETFLocalStorage] = None

def _get_storage() -> ETFLocalStorage:
    global _storage
    if _storage is None:
        _storage = ETFLocalStorage()
    return _storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 宽基ETF分类 ───────────────────────────────────────────────────────────────
# 凡代码以下列前缀开头，均视为宽基ETF
_WIDE_BASIS_PREFIXES = ("510", "512", "56", "588", "515", "516", "159")


# ── 内存缓存 ──────────────────────────────────────────────────────────────────

class ETFCache:
    def __init__(self, ttl: int = 60):
        self._cache: Dict[str, tuple] = {}
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        self._cache.clear()


# ── ETF代码映射 ───────────────────────────────────────────────────────────────

class ETFCodeMapper:
    # 从 akshare 拉取的全量 ETF 字典（code -> name），运行时填充
    _live: Dict[str, str] = {}
    # 静态兜底
    _static: Dict[str, str] = ETF_CODES

    @classmethod
    def load_from_akshare(cls) -> int:
        """从 akshare 拉取全量 ETF 列表，返回 ETF 数量。"""
        try:
            import akshare as ak
            df = ak.fund_etf_spot_em()
            cls._live = dict(zip(df["代码"].astype(str), df["名称"].astype(str)))
            logger.info(f"akshare ETF列表加载完成，共 {len(cls._live)} 只")
            return len(cls._live)
        except Exception as e:
            logger.warning(f"akshare ETF列表加载失败，回退静态列表: {e}")
            return 0

    @classmethod
    def _all(cls) -> Dict[str, str]:
        """优先用动态列表，否则用静态列表。"""
        return cls._live if cls._live else cls._static

    @classmethod
    def get_etf_name(cls, code: str) -> Optional[str]:
        return cls._all().get(code)

    @classmethod
    def get_all_codes(cls) -> List[str]:
        return list(cls._all().keys())

    @classmethod
    def get_wide_basis_codes(cls) -> List[str]:
        return [c for c in cls._all() if cls.is_wide_basis(c)]

    @classmethod
    def get_industry_codes(cls) -> List[str]:
        return [c for c in cls._all() if cls.is_industry(c)]

    @classmethod
    def is_wide_basis(cls, code: str) -> bool:
        return code.startswith(_WIDE_BASIS_PREFIXES)

    @classmethod
    def is_industry(cls, code: str) -> bool:
        return not cls.is_wide_basis(code)


# ── 数据获取 ──────────────────────────────────────────────────────────────────

class ETFDataFetcher:
    # akshare 批量行情的进程级缓存（避免同一批次重复拉取）
    _akshare_spot_cache: Optional[Dict[str, Dict]] = None
    _akshare_spot_ts: float = 0.0
    _akshare_spot_ttl: float = 60.0

    def __init__(self):
        self.cache = ETFCache(ttl=60)
        self.mapper = ETFCodeMapper()
        self.yahoo_base = "https://query1.finance.yahoo.com/v8/finance/chart/"

    def _get_yahoo_symbol(self, etf_code: str) -> str:
        if etf_code.startswith("159") or etf_code.startswith("588"):
            return f"{etf_code}.SZ"
        return f"{etf_code}.SS"

    # ── akshare 批量行情（一次拉全量，按代码查询） ────────────────

    def _get_akshare_spot_batch(self) -> Dict[str, Dict]:
        """拉取全量 ETF 实时行情，TTL 60s 内复用缓存。"""
        now = time.time()
        if (ETFDataFetcher._akshare_spot_cache is not None
                and now - ETFDataFetcher._akshare_spot_ts < ETFDataFetcher._akshare_spot_ttl):
            return ETFDataFetcher._akshare_spot_cache

        try:
            import akshare as ak
            df = ak.fund_etf_spot_em()
            batch: Dict[str, Dict] = {}
            for _, row in df.iterrows():
                code = str(row["代码"])
                prev = row.get("昨收") or row.get("最新价", 0)
                latest = row.get("最新价", 0) or 0
                change = round(float(latest) - float(prev), 4) if prev else 0
                change_pct = round(change / float(prev) * 100, 2) if prev else 0
                batch[code] = {
                    "code": code,
                    "name": str(row.get("名称", "")),
                    "latest_price": float(latest) if latest else 0,
                    "change": change,
                    "change_pct": float(row.get("涨跌幅", change_pct) or change_pct),
                    "open": float(row.get("开盘价", 0) or 0),
                    "high": float(row.get("最高价", 0) or 0),
                    "low": float(row.get("最低价", 0) or 0),
                    "volume": float(row.get("成交量", 0) or 0),
                    "source": "akshare",
                }
            ETFDataFetcher._akshare_spot_cache = batch
            ETFDataFetcher._akshare_spot_ts = now
            logger.info(f"akshare 批量行情加载完成，共 {len(batch)} 只")
            return batch
        except Exception as e:
            logger.debug(f"akshare 批量行情失败: {e}")
            return {}

    def _quote_from_akshare(self, etf_code: str) -> Optional[Dict]:
        batch = self._get_akshare_spot_batch()
        return batch.get(etf_code)

    # ── 实时行情（含 fallback 链） ────────────────────────────────

    def get_realtime_quote(self, etf_code: str) -> Optional[Dict]:
        cache_key = f"realtime_{etf_code}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        result = (self._quote_from_akshare(etf_code)
                  or self._quote_from_yahoo(etf_code)
                  or self._quote_from_eastmoney(etf_code)
                  or self._quote_from_tencent(etf_code))
        if result:
            self.cache.set(cache_key, result)
        return result

    def _quote_from_yahoo(self, etf_code: str) -> Optional[Dict]:
        try:
            symbol = self._get_yahoo_symbol(etf_code)
            url = f"{self.yahoo_base}{symbol}"
            params = {"interval": "1d", "range": "1d", "events": "div,split"}
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            if "chart" in data and data["chart"].get("result"):
                result = data["chart"]["result"][0]
                meta = result.get("meta", {})
                quote = result.get("indicators", {}).get("quote", [{}])[0]
                return {
                    "code": etf_code,
                    "name": self.mapper.get_etf_name(etf_code) or "未知ETF",
                    "latest_price": meta.get("regularMarketPrice", 0),
                    "change": meta.get("regularMarketChange", 0),
                    "change_pct": meta.get("regularMarketChangePercent", 0),
                    "open": quote.get("open", [0])[0] if quote.get("open") else 0,
                    "high": quote.get("high", [0])[0] if quote.get("high") else 0,
                    "low": quote.get("low", [0])[0] if quote.get("low") else 0,
                    "volume": meta.get("regularMarketVolume", 0),
                    "source": "Yahoo Finance",
                }
        except Exception as e:
            logger.debug(f"Yahoo Finance 获取 {etf_code} 失败: {e}")
        return None

    def _quote_from_eastmoney(self, etf_code: str) -> Optional[Dict]:
        try:
            market = "0" if etf_code.startswith("159") or etf_code.startswith("588") else "1"
            url = "https://push2.eastmoney.com/api/qt/stock/get"
            params = {
                "secid": f"{market}.{etf_code}",
                "fields": "f43,f44,f45,f46,f47,f58,f60,f170",
                "ut": "fa5fd1943c7b386f172d6893dbfba10b",
            }
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            d = data.get("data", {})
            if not d or d.get("f43") is None:
                return None
            price = d.get("f43", 0) / 100
            prev_close = d.get("f60", d.get("f43", 100)) / 100
            change = round(price - prev_close, 4)
            change_pct = round(change / prev_close * 100, 2) if prev_close else 0
            return {
                "code": etf_code,
                "name": d.get("f58", self.mapper.get_etf_name(etf_code) or "未知ETF"),
                "latest_price": price,
                "change": change,
                "change_pct": change_pct,
                "open": d.get("f46", 0) / 100,
                "high": d.get("f44", 0) / 100,
                "low": d.get("f45", 0) / 100,
                "volume": d.get("f47", 0),
                "source": "东方财富",
            }
        except Exception as e:
            logger.debug(f"东方财富 获取 {etf_code} 失败: {e}")
        return None

    def _quote_from_tencent(self, etf_code: str) -> Optional[Dict]:
        try:
            prefix = "sz" if etf_code.startswith("159") or etf_code.startswith("588") else "sh"
            url = f"https://qt.gtimg.cn/q={prefix}{etf_code}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            text = response.text
            if "~" not in text:
                return None
            parts = text.split("~")
            if len(parts) < 10:
                return None
            price = float(parts[3]) if parts[3] else 0
            prev_close = float(parts[4]) if parts[4] else price
            change = round(price - prev_close, 4)
            change_pct = round(change / prev_close * 100, 2) if prev_close else 0
            return {
                "code": etf_code,
                "name": parts[1] if len(parts) > 1 else (self.mapper.get_etf_name(etf_code) or "未知ETF"),
                "latest_price": price,
                "change": change,
                "change_pct": change_pct,
                "open": float(parts[5]) if parts[5] else 0,
                "high": float(parts[33]) if len(parts) > 33 and parts[33] else 0,
                "low": float(parts[34]) if len(parts) > 34 and parts[34] else 0,
                "volume": int(parts[6]) if parts[6] else 0,
                "source": "腾讯",
            }
        except Exception as e:
            logger.debug(f"腾讯 获取 {etf_code} 失败: {e}")
        return None

    # ── 历史数据（含本地缓存 + fallback 链） ─────────────────────

    def _fetch_history_network(self, etf_code: str, days: int,
                                start_dt=None, end_dt=None) -> Optional[pd.DataFrame]:
        """按顺序尝试各数据源，返回第一个成功的结果（None 表示全部失败）。"""
        for df in [
            self._history_from_akshare(etf_code, days, start_dt=start_dt, end_dt=end_dt),
            self._history_from_yahoo(etf_code, days),
            self._history_from_eastmoney(etf_code, days),
        ]:
            if df is not None and len(df) > 0:
                return df
        return None

    def get_historical_data(self, etf_code: str, days: int = 30,
                            use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        获取历史行情数据。
        use_cache=True（默认）：优先读本地 SQLite，自动增量补全缺口后返回。
        use_cache=False：直接走网络拉取，不读写本地缓存。
        """
        if not use_cache:
            return self._fetch_history_network(etf_code, days)

        storage = _get_storage()
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=days)

        last_date = storage.get_last_date(etf_code)
        first_date = storage.get_first_date(etf_code)

        # ── 判断本地数据是否覆盖请求区间 ──────────────────────────
        # 允许 7 天的容差（日历日 vs 交易日偏差、节假日等）
        has_local = (last_date is not None and first_date is not None
                     and first_date <= start_dt + timedelta(days=7))

        if has_local:
            # 本地数据已是最新，直接返回
            if last_date >= end_dt - timedelta(days=1):
                return storage.load_prices(etf_code, start_dt, end_dt)

            # 只拉 last_date+1 到 end_dt 的缺口
            gap_start = last_date + timedelta(days=1)
            new_df = self._fetch_history_network(etf_code, days,
                                                  start_dt=gap_start, end_dt=end_dt)
            if new_df is not None and len(new_df) > 0:
                storage.save_prices(etf_code, new_df)
                storage.upsert_meta(etf_code, self.mapper.get_etf_name(etf_code) or "")
            return storage.load_prices(etf_code, start_dt, end_dt)

        # ── 本地无完整数据，全量拉取 ──────────────────────────────
        full_df = self._fetch_history_network(etf_code, days)
        if full_df is not None and len(full_df) > 0:
            storage.save_prices(etf_code, full_df)
            storage.upsert_meta(etf_code, self.mapper.get_etf_name(etf_code) or "")
        return full_df

    def _history_from_akshare(self, etf_code: str, days: int,
                               start_dt: Optional[date] = None,
                               end_dt: Optional[date] = None) -> Optional[pd.DataFrame]:
        try:
            import akshare as ak
            end = (end_dt or date.today()).strftime("%Y%m%d")
            start = (start_dt or (date.today() - timedelta(days=days))).strftime("%Y%m%d")
            df = ak.fund_etf_hist_em(
                symbol=etf_code, period="daily",
                start_date=start, end_date=end, adjust="qfq"
            )
            if df is None or len(df) == 0:
                return None
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
            return df
        except Exception as e:
            logger.debug(f"akshare 历史 {etf_code} 失败: {e}")
        return None

    def _history_from_yahoo(self, etf_code: str, days: int) -> Optional[pd.DataFrame]:
        try:
            symbol = self._get_yahoo_symbol(etf_code)
            url = f"{self.yahoo_base}{symbol}"
            params = {"interval": "1d", "range": f"{days}d", "events": "div,split"}
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            if "chart" in data and data["chart"].get("result"):
                result = data["chart"]["result"][0]
                timestamps = result.get("timestamp", [])
                quote = result.get("indicators", {}).get("quote", [{}])[0]
                df = pd.DataFrame({
                    "date": pd.to_datetime(timestamps, unit="s"),
                    "open": quote.get("open"),
                    "high": quote.get("high"),
                    "low": quote.get("low"),
                    "close": quote.get("close"),
                    "volume": quote.get("volume"),
                })
                df = df.dropna().sort_values("date").reset_index(drop=True)
                if len(df) > 0:
                    return df
        except Exception as e:
            logger.debug(f"Yahoo Finance 历史 {etf_code} 失败: {e}")
        return None

    def _history_from_eastmoney(self, etf_code: str, days: int) -> Optional[pd.DataFrame]:
        try:
            market = "0" if etf_code.startswith("159") or etf_code.startswith("588") else "1"
            url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
            params = {
                "secid": f"{market}.{etf_code}",
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56",
                "klt": "101",
                "fqt": "1",
                "lmt": days,
                "end": "20500101",
                "ut": "fa5fd1943c7b386f172d6893dbfba10b",
            }
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, params=params, headers=headers, timeout=10)
            klines = response.json().get("data", {}).get("klines", [])
            if not klines:
                return None
            rows = []
            for k in klines:
                parts = k.split(",")
                if len(parts) >= 6:
                    rows.append({
                        "date": pd.to_datetime(parts[0]),
                        "open": float(parts[1]),
                        "close": float(parts[2]),
                        "high": float(parts[3]),
                        "low": float(parts[4]),
                        "volume": float(parts[5]),
                    })
            df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            return df if len(df) > 0 else None
        except Exception as e:
            logger.debug(f"东方财富 历史 {etf_code} 失败: {e}")
        return None

    # ── 批量行情 ──────────────────────────────────────────────────

    def get_batch_realtime_quotes(self, etf_codes: List[str], max_workers: int = 10) -> List[Dict]:
        """批量获取实时行情。akshare 路径一次性拉全量，其他路径并发请求。"""
        # 先尝试从 akshare 批量缓存中查
        batch = self._get_akshare_spot_batch()
        if batch:
            results = [batch[c] for c in etf_codes if c in batch]
            missing = [c for c in etf_codes if c not in batch]
        else:
            results, missing = [], etf_codes

        # 对 akshare 未覆盖的代码，并发 fallback
        if missing:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.get_realtime_quote, c): c for c in missing}
                for future in as_completed(futures):
                    q = future.result()
                    if q:
                        results.append(q)
        return results


# ── 便捷函数 ──────────────────────────────────────────────────────────────────

def get_etf_quote(etf_code: str) -> Optional[Dict]:
    return ETFDataFetcher().get_realtime_quote(etf_code)


def get_etf_history(etf_code: str, days: int = 30) -> Optional[pd.DataFrame]:
    return ETFDataFetcher().get_historical_data(etf_code, days)
