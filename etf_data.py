"""
ETF Data Module
A股ETF数据获取模块，支持上交所(510/511/512/515/516)和深交所(159)开头的ETF基金
"""

import requests
import pandas as pd
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from functools import lru_cache
import logging

from all_etf_codes import ETF_CODES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETFCache:
    def __init__(self, ttl: int = 60):
        self._cache: Dict[str, tuple] = {}
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        self._cache.clear()


class ETFCodeMapper:
    ALL_ETF = ETF_CODES

    @classmethod
    def get_etf_name(cls, code: str) -> Optional[str]:
        return cls.ALL_ETF.get(code)

    @classmethod
    def get_all_codes(cls) -> List[str]:
        return list(cls.ALL_ETF.keys())

    @classmethod
    def get_wide_basis_codes(cls) -> List[str]:
        wide_basis = []
        for code in cls.ALL_ETF.keys():
            if code.startswith("510") and int(code[3:6]) < 200:
                wide_basis.append(code)
        return wide_basis

    @classmethod
    def get_industry_codes(cls) -> List[str]:
        industry = []
        for code in cls.ALL_ETF.keys():
            if code.startswith("512") or code.startswith("515") or code.startswith("516"):
                industry.append(code)
        return industry

    @classmethod
    def is_wide_basis(cls, code: str) -> bool:
        return code.startswith("510") and int(code[3:6]) < 200

    @classmethod
    def is_industry(cls, code: str) -> bool:
        return code.startswith("512") or code.startswith("515") or code.startswith("516")


class ETFDataFetcher:
    def __init__(self):
        self.cache = ETFCache(ttl=60)
        self.mapper = ETFCodeMapper()
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"

    def _get_yahoo_symbol(self, etf_code: str) -> str:
        if etf_code.startswith("159") or etf_code.startswith("588"):
            return f"{etf_code}.SZ"
        return f"{etf_code}.SS"

    def get_realtime_quote(self, etf_code: str) -> Optional[Dict]:
        cache_key = f"realtime_{etf_code}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            symbol = self._get_yahoo_symbol(etf_code)
            url = f"{self.base_url}{symbol}"
            params = {"interval": "1d", "range": "1d", "events": "div,split"}
            headers = {"User-Agent": "Mozilla/5.0"}

            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()

            if "chart" in data and data["chart"].get("result"):
                result = data["chart"]["result"][0]
                meta = result.get("meta", {})
                quote = result.get("indicators", {}).get("quote", [{}])[0]

                result_data = {
                    "code": etf_code,
                    "name": self.mapper.get_etf_name(etf_code) or "未知ETF",
                    "latest_price": meta.get("regularMarketPrice", 0),
                    "change": meta.get("regularMarketChange", 0),
                    "change_pct": meta.get("regularMarketChangePercent", 0),
                    "open": quote.get("open", [0])[0] if quote.get("open") else 0,
                    "high": quote.get("high", [0])[0] if quote.get("high") else 0,
                    "low": quote.get("low", [0])[0] if quote.get("low") else 0,
                    "volume": meta.get("regularMarketVolume", 0),
                    "source": "Yahoo Finance"
                }

                self.cache.set(cache_key, result_data)
                return result_data
        except Exception as e:
            logger.error(f"获取ETF {etf_code} 失败: {e}")
        return None

    def get_historical_data(self, etf_code: str, days: int = 30) -> Optional[pd.DataFrame]:
        try:
            symbol = self._get_yahoo_symbol(etf_code)
            url = f"{self.base_url}{symbol}"
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
                df = df.dropna().sort_values("date")
                return df
        except Exception as e:
            logger.error(f"获取ETF {etf_code}历史失败: {e}")
        return None

    def get_batch_realtime_quotes(self, etf_codes: List[str]) -> List[Dict]:
        results = []
        for code in etf_codes:
            quote = self.get_realtime_quote(code)
            if quote:
                results.append(quote)
            time.sleep(0.1)
        return results


def get_etf_quote(etf_code: str) -> Optional[Dict]:
    return ETFDataFetcher().get_realtime_quote(etf_code)


def get_etf_history(etf_code: str, days: int = 30) -> Optional[pd.DataFrame]:
    return ETFDataFetcher().get_historical_data(etf_code, days)
