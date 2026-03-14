"""
ETF Data Module
A股ETF数据获取模块，支持上交所(510/511/512/515/516)和深交所(159)开头的ETF基金
数据源优先级：akshare -> Yahoo Finance -> 东方财富 -> 腾讯
"""

import requests
import pandas as pd
import time
import csv
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from all_etf_codes import ETF_CODES
from etf_metadata import infer_etf_metadata
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

# ── 兼容旧逻辑的常量 ──────────────────────────────────────────────────────────
_WIDE_BASIS_PREFIXES = ("510", "588", "159")


def load_history_csv(path: str, days: Optional[int] = None) -> pd.DataFrame:
    """
    读取本地历史交易 CSV，并标准化为统一字段：
    date/open/high/low/close/volume
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV 文件不存在: {path}")

    df = pd.read_csv(path)
    if df is None or len(df) == 0:
        raise ValueError("CSV 文件为空")

    rename_map = {
        "日期": "date",
        "交易日期": "date",
        "date": "date",
        "Date": "date",
        "开盘": "open",
        "开盘价": "open",
        "open": "open",
        "Open": "open",
        "最高": "high",
        "最高价": "high",
        "high": "high",
        "High": "high",
        "最低": "low",
        "最低价": "low",
        "low": "low",
        "Low": "low",
        "收盘": "close",
        "收盘价": "close",
        "close": "close",
        "Close": "close",
        "成交量": "volume",
        "成交量(手)": "volume",
        "volume": "volume",
        "Volume": "volume",
    }
    df = df.rename(columns={col: rename_map[col] for col in df.columns if col in rename_map})

    required = ["date", "open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {', '.join(missing)}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    out = df[["date", "open", "high", "low", "close", "volume"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    if len(out) == 0:
        raise ValueError("CSV 标准化后无有效数据")

    if days is not None and days > 0:
        out = out.tail(days).reset_index(drop=True)
    return out


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
    def get_etf_metadata(cls, code: str) -> Dict[str, Any]:
        name = cls.get_etf_name(code) or ""
        return infer_etf_metadata(code, name)

    @classmethod
    def get_etf_category(cls, code: str) -> str:
        return cls.get_etf_metadata(code)["category"]

    @classmethod
    def get_etf_category_label(cls, code: str) -> str:
        return cls.get_etf_metadata(code)["category_label"]

    @classmethod
    def get_etf_sector(cls, code: str) -> str:
        return cls.get_etf_metadata(code)["sector"]

    @classmethod
    def get_etf_tags(cls, code: str) -> List[str]:
        return cls.get_etf_metadata(code)["tags"]

    @classmethod
    def get_all_codes(cls) -> List[str]:
        return list(cls._all().keys())

    @classmethod
    def get_metadata_table(cls, category: Optional[str] = None, refresh: bool = False) -> List[Dict[str, Any]]:
        if refresh:
            cls.load_from_akshare()

        rows: List[Dict[str, Any]] = []
        for code, name in cls._all().items():
            meta = cls.get_etf_metadata(code)
            if category and meta["category"] != category:
                continue
            rows.append(meta)

        rows.sort(key=lambda item: (item["category"], item["sector"], item["code"]))
        return rows

    @classmethod
    def export_metadata_table(cls, path: str, category: Optional[str] = None, refresh: bool = False) -> int:
        rows = cls.get_metadata_table(category=category, refresh=refresh)
        with open(path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=["code", "name", "category", "category_label", "sector", "tags"],
            )
            writer.writeheader()
            for row in rows:
                out = dict(row)
                out["tags"] = ",".join(out.get("tags", []))
                writer.writerow(out)
        return len(rows)

    @classmethod
    def get_wide_basis_codes(cls) -> List[str]:
        return [c for c in cls._all() if cls.get_etf_category(c) == "wide_basis"]

    @classmethod
    def get_industry_codes(cls) -> List[str]:
        return [c for c in cls._all() if cls.get_etf_category(c) == "industry"]

    @classmethod
    def is_wide_basis(cls, code: str) -> bool:
        return cls.get_etf_category(code) == "wide_basis"

    @classmethod
    def is_industry(cls, code: str) -> bool:
        return cls.get_etf_category(code) == "industry"


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

    def _calendar_lookback(self, days: int) -> int:
        return max(int(days * 2.2), days + 30)

    def _tail_trading_days(
        self, df: Optional[pd.DataFrame], days: int
    ) -> Optional[pd.DataFrame]:
        if df is None or len(df) == 0:
            return None
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
        return out.tail(days).reset_index(drop=True)

    def _pick_numeric(self, row: pd.Series, candidates: List[str]) -> Optional[float]:
        for key in candidates:
            if key not in row:
                continue
            value = row.get(key)
            if value in ("", None):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

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
                    "fund_size": self._pick_numeric(row, ["基金规模", "规模", "流通规模"]),
                    "turnover_rate": self._pick_numeric(row, ["换手率"]),
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

    def get_fund_profile(self, etf_code: str) -> Dict[str, Any]:
        quote = self.get_realtime_quote(etf_code) or {}
        return {
            "code": etf_code,
            "name": self.mapper.get_etf_name(etf_code) or etf_code,
            "fund_size": quote.get("fund_size"),
            "turnover_rate": quote.get("turnover_rate"),
            "source": quote.get("source", "static"),
        }

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
            self._history_from_yahoo(etf_code, days, start_dt=start_dt, end_dt=end_dt),
            self._history_from_eastmoney(etf_code, days, start_dt=start_dt, end_dt=end_dt),
        ]:
            if df is not None and len(df) > 0:
                return df
        return None

    def _normalize_history_range(self, df: Optional[pd.DataFrame],
                                 start_dt: Optional[date] = None,
                                 end_dt: Optional[date] = None) -> Optional[pd.DataFrame]:
        if df is None or len(df) == 0:
            return None
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        if start_dt is not None:
            out = out[out["date"] >= pd.Timestamp(start_dt)]
        if end_dt is not None:
            out = out[out["date"] <= pd.Timestamp(end_dt)]
        out = out.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
        return out if len(out) > 0 else None

    def get_historical_data(self, etf_code: str, days: int = 30,
                            use_cache: bool = True,
                            return_metadata: bool = False):
        """
        获取历史行情数据。
        use_cache=True（默认）：优先读本地 SQLite，自动增量补全缺口后返回。
        use_cache=False：直接走网络拉取，不读写本地缓存。
        """
        requested_days = days
        calendar_days = self._calendar_lookback(days)
        metadata = {
            "new_rows": 0,
            "source": "network" if not use_cache else "cache",
            "requested_rows": requested_days,
        }
        if not use_cache:
            df = self._fetch_history_network(etf_code, calendar_days)
            df = self._tail_trading_days(df, requested_days)
            return (df, metadata) if return_metadata else df

        storage = _get_storage()
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=calendar_days)
        local_df = storage.load_prices(etf_code, start_dt, end_dt)
        last_date = local_df["date"].max().date() if local_df is not None and len(local_df) > 0 else None

        if local_df is not None and len(local_df) >= requested_days and last_date is not None:
            if last_date >= end_dt - timedelta(days=1):
                local_df = self._tail_trading_days(local_df, requested_days)
                return (local_df, metadata) if return_metadata else local_df

        gap_start = last_date + timedelta(days=1) if last_date is not None else start_dt
        new_df = self._fetch_history_network(
            etf_code,
            calendar_days,
            start_dt=gap_start,
            end_dt=end_dt,
        )
        if new_df is not None and len(new_df) > 0:
            new_df = self._normalize_history_range(new_df, gap_start, end_dt)
        if new_df is not None and len(new_df) > 0:
            storage.save_prices(etf_code, new_df)
            storage.upsert_meta(etf_code, self.mapper.get_etf_name(etf_code) or "")
            metadata["new_rows"] = len(new_df)

        merged_df = storage.load_prices(etf_code, start_dt, end_dt)
        if merged_df is None or len(merged_df) < requested_days:
            full_df = self._fetch_history_network(etf_code, calendar_days, start_dt=start_dt, end_dt=end_dt)
            if full_df is not None and len(full_df) > 0:
                full_df = self._normalize_history_range(full_df, start_dt, end_dt)
                storage.save_prices(etf_code, full_df)
                storage.upsert_meta(etf_code, self.mapper.get_etf_name(etf_code) or "")
                metadata["new_rows"] = max(metadata["new_rows"], len(full_df))
                merged_df = storage.load_prices(etf_code, start_dt, end_dt)

        merged_df = self._tail_trading_days(merged_df, requested_days)
        return (merged_df, metadata) if return_metadata else merged_df

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

    def _history_from_yahoo(self, etf_code: str, days: int,
                            start_dt: Optional[date] = None,
                            end_dt: Optional[date] = None) -> Optional[pd.DataFrame]:
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
                    return self._normalize_history_range(df, start_dt, end_dt)
        except Exception as e:
            logger.debug(f"Yahoo Finance 历史 {etf_code} 失败: {e}")
        return None

    def _history_from_eastmoney(self, etf_code: str, days: int,
                                start_dt: Optional[date] = None,
                                end_dt: Optional[date] = None) -> Optional[pd.DataFrame]:
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
            return self._normalize_history_range(df, start_dt, end_dt)
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
