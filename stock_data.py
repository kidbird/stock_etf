"""
Stock Data Module
A股股票数据获取模块，支持沪深两市A股
数据源优先级：akshare -> 东方财富 -> 腾讯
"""

import requests
import pandas as pd
import time
import csv
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os

from all_stock_codes import STOCK_CODES, STOCK_INDUSTRIES
from stock_storage import StockLocalStorage

_storage: Optional[StockLocalStorage] = None


def _get_storage() -> StockLocalStorage:
    global _storage
    if _storage is None:
        _storage = StockLocalStorage()
    return _storage


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_WIDE_BASIS_PREFIXES = ("510", "588", "159")


class StockCache:
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


class StockCodeMapper:
    _live: Dict[str, str] = {}
    _live_industry: Dict[str, str] = {}
    _static: Dict[str, str] = STOCK_CODES
    _static_industry: Dict[str, str] = STOCK_INDUSTRIES

    @classmethod
    def _load_industries_from_akshare(cls, ak, codes: List[str]) -> Dict[str, str]:
        industries = {code: cls._static_industry.get(code, "未知") for code in codes}
        try:
            boards = ak.stock_board_industry_name_em()
        except Exception as exc:
            logger.debug(f"akshare 行业板块列表加载失败: {exc}")
            return industries

        if boards is None or len(boards) == 0:
            return industries

        board_name_col = next((col for col in ("板块名称", "名称", "industry_name") if col in boards.columns), None)
        if board_name_col is None:
            return industries

        for board_name in boards[board_name_col].dropna().astype(str).tolist():
            try:
                cons_df = ak.stock_board_industry_cons_em(symbol=board_name)
            except Exception as exc:
                logger.debug("akshare 行业成分加载失败 %s: %s", board_name, exc)
                continue
            if cons_df is None or len(cons_df) == 0:
                continue
            code_col = next((col for col in ("代码", "code", "股票代码") if col in cons_df.columns), None)
            if code_col is None:
                continue
            for code in cons_df[code_col].dropna().astype(str):
                if code in industries:
                    industries[code] = board_name
        return industries

    @classmethod
    def load_from_akshare(cls) -> int:
        try:
            import akshare as ak

            df = ak.stock_info_a_code_name()
            cls._live = dict(zip(df["code"].astype(str), df["name"].astype(str)))
            cls._live_industry = cls._load_industries_from_akshare(ak, list(cls._live))

            logger.info(f"akshare 股票列表加载完成，共 {len(cls._live)} 只")
            return len(cls._live)
        except Exception as e:
            logger.warning(f"akshare 股票列表加载失败: {e}")
            return 0

    @classmethod
    def _all(cls) -> Dict[str, str]:
        return cls._live if cls._live else cls._static

    @classmethod
    def get_stock_name(cls, code: str) -> Optional[str]:
        return cls._all().get(code)

    @classmethod
    def get_all_codes(cls) -> List[str]:
        return list(cls._all().keys())

    @classmethod
    def get_stock_industry(cls, code: str) -> str:
        return cls._live_industry.get(code) or cls._static_industry.get(code, "未知")


class StockDataFetcher:
    _akshare_spot_cache: Optional[Dict[str, Dict]] = None
    _akshare_spot_ts: float = 0.0
    _akshare_spot_ttl: float = 60.0

    def __init__(self):
        self.cache = StockCache(ttl=60)
        self.mapper = StockCodeMapper()
        self.storage = _get_storage()

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

    def _get_akshare_spot_batch(self) -> Dict[str, Dict]:
        now = time.time()
        if (
            StockDataFetcher._akshare_spot_cache is not None
            and now - StockDataFetcher._akshare_spot_ts
            < StockDataFetcher._akshare_spot_ttl
        ):
            return StockDataFetcher._akshare_spot_cache

        try:
            import akshare as ak

            df = ak.stock_zh_a_spot_em()
            batch: Dict[str, Dict] = {}
            for _, row in df.iterrows():
                code = str(row["代码"])
                prev = row.get("昨收", row.get("最新价", 0)) or 0
                latest = row.get("最新价", 0) or 0
                change = round(float(latest) - float(prev), 4) if prev else 0
                change_pct = round(change / float(prev) * 100, 2) if prev else 0
                batch[code] = {
                    "code": code,
                    "name": str(row.get("名称", "")),
                    "latest_price": float(latest) if latest else 0,
                    "change": change,
                    "change_pct": float(row.get("涨跌幅", change_pct) or change_pct),
                    "open": float(row.get("开盘", 0) or 0),
                    "high": float(row.get("最高", 0) or 0),
                    "low": float(row.get("最低", 0) or 0),
                    "volume": float(row.get("成交量", 0) or 0),
                    "amount": float(row.get("成交额", 0) or 0),
                    "market_cap": self._pick_numeric(row, ["总市值", "总市值(元)"]),
                    "float_market_cap": self._pick_numeric(row, ["流通市值", "流通市值(元)"]),
                    "pe_ttm": self._pick_numeric(row, ["市盈率-动态", "市盈率"]),
                    "source": "akshare",
                }
            StockDataFetcher._akshare_spot_cache = batch
            StockDataFetcher._akshare_spot_ts = now
            logger.info(f"akshare 批量股票行情加载完成，共 {len(batch)} 只")
            return batch
        except Exception as e:
            logger.debug(f"akshare 批量行情失败: {e}")
            return {}

    def _quote_from_akshare(self, stock_code: str) -> Optional[Dict]:
        batch = self._get_akshare_spot_batch()
        return batch.get(stock_code)

    def get_realtime_quote(self, stock_code: str) -> Optional[Dict]:
        cache_key = f"realtime_{stock_code}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        result = self._quote_from_akshare(stock_code)
        if result:
            self.cache.set(cache_key, result)
        return result

    def get_stock_fundamentals(self, stock_code: str) -> Dict[str, Any]:
        cache_key = f"fund_{stock_code}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        quote = self.get_realtime_quote(stock_code) or {}
        fundamentals = {
            "code": stock_code,
            "name": self.mapper.get_stock_name(stock_code) or stock_code,
            "industry": self.mapper.get_stock_industry(stock_code),
            "market_cap": quote.get("market_cap"),
            "float_market_cap": quote.get("float_market_cap"),
            "roe": None,
            "source": quote.get("source", "static"),
        }

        try:
            import akshare as ak

            df = ak.stock_financial_analysis_indicator(symbol=stock_code)
            if df is not None and len(df) > 0:
                latest = df.iloc[0]
                for key in ("净资产收益率(%)", "净资产收益率", "ROE"):
                    if key in latest and latest.get(key) not in ("", None):
                        try:
                            fundamentals["roe"] = float(latest[key])
                            break
                        except (TypeError, ValueError):
                            pass
        except Exception as exc:
            logger.debug("股票ROE获取失败 %s: %s", stock_code, exc)

        self.cache.set(cache_key, fundamentals)
        return fundamentals

    def get_historical_data(
        self, stock_code: str, days: int = 250
    ) -> Optional[pd.DataFrame]:
        cache_key = f"hist_{stock_code}_{days}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self._calendar_lookback(days))

        local_df = self.storage.load_prices(stock_code, start_date.date(), end_date.date())
        local_last_date = local_df["date"].max().date() if local_df is not None and len(local_df) > 0 else None
        if local_df is not None and len(local_df) >= days and local_last_date is not None:
            if local_last_date >= end_date.date() - timedelta(days=1):
                local_df = self._tail_trading_days(local_df, days)
                self.cache.set(cache_key, local_df)
                return local_df

        fetch_start = local_last_date + timedelta(days=1) if local_last_date else start_date.date()
        df = self._fetch_from_akshare(
            stock_code, datetime.combine(fetch_start, datetime.min.time()), end_date
        )
        if df is not None and len(df) > 0:
            self.storage.save_prices(stock_code, df)
            self.storage.upsert_meta(stock_code, self.mapper.get_stock_name(stock_code) or "")

        merged_df = self.storage.load_prices(stock_code, start_date.date(), end_date.date())
        if merged_df is None or len(merged_df) < days:
            full_df = self._fetch_from_akshare(stock_code, start_date, end_date)
            if full_df is not None and len(full_df) > 0:
                self.storage.save_prices(stock_code, full_df)
                self.storage.upsert_meta(stock_code, self.mapper.get_stock_name(stock_code) or "")
                merged_df = self.storage.load_prices(stock_code, start_date.date(), end_date.date())

        merged_df = self._tail_trading_days(merged_df, days)
        if merged_df is not None:
            self.cache.set(cache_key, merged_df)
        return merged_df

    def _fetch_from_akshare(
        self, stock_code: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        try:
            import akshare as ak

            if stock_code.startswith("6"):
                symbol = f"sh{stock_code}"
            else:
                symbol = f"sz{stock_code}"

            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust="qfq",
            )

            if df is not None and len(df) > 0:
                df = df.rename(
                    columns={
                        "日期": "date",
                        "开盘": "open",
                        "收盘": "close",
                        "最高": "high",
                        "最低": "low",
                        "成交量": "volume",
                        "成交额": "amount",
                        "振幅": "amplitude",
                        "涨跌幅": "change_pct",
                        "涨跌额": "change",
                        "换手率": "turnover",
                    }
                )
                df = df[["date", "open", "high", "low", "close", "volume", "amount"]]
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                return df
        except Exception as e:
            logger.warning(f"akshare 获取 {stock_code} 历史数据失败: {e}")
        return None

    def download_batch(
        self, codes: List[str], days: int = 250, max_workers: int = 5
    ) -> Dict[str, int]:
        results = {}

        def fetch_one(code: str) -> tuple:
            df = self.get_historical_data(code, days)
            return code, len(df) if df is not None else 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_one, code): code for code in codes}
            for future in as_completed(futures):
                code, count = future.result()
                results[code] = count
                logger.info(f"已下载 {code}: {count} 条记录")
        return results
