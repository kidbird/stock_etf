"""
Stock Local Storage Module
SQLite 本地缓存，支持增量更新。每线程独立连接，兼容 ThreadPoolExecutor 并发写。
"""

import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

_DB_PATH = Path(__file__).parent / "stock_cache.db"


class StockLocalStorage:
    def __init__(self, db_path: Path = _DB_PATH):
        self._db_path = str(db_path)
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                code    TEXT NOT NULL,
                date    TEXT NOT NULL,
                open    REAL,
                high    REAL,
                low     REAL,
                close   REAL,
                volume  REAL,
                amount  REAL,
                PRIMARY KEY (code, date)
            );
            CREATE TABLE IF NOT EXISTS stock_meta (
                code        TEXT PRIMARY KEY,
                name        TEXT,
                industry    TEXT,
                market      TEXT,
                updated_at  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_stock_prices_code_date ON stock_prices(code, date);
        """)
        conn.commit()

    def get_last_date(self, code: str) -> Optional[date]:
        row = (
            self._conn()
            .execute("SELECT MAX(date) FROM stock_prices WHERE code = ?", (code,))
            .fetchone()
        )
        if row and row[0]:
            return date.fromisoformat(row[0])
        return None

    def get_first_date(self, code: str) -> Optional[date]:
        row = (
            self._conn()
            .execute("SELECT MIN(date) FROM stock_prices WHERE code = ?", (code,))
            .fetchone()
        )
        if row and row[0]:
            return date.fromisoformat(row[0])
        return None

    def load_prices(
        self,
        code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Optional[pd.DataFrame]:
        sql = "SELECT date, open, high, low, close, volume FROM stock_prices WHERE code = ?"
        params: list = [code]
        if start_date:
            sql += " AND date >= ?"
            params.append(start_date.isoformat())
        if end_date:
            sql += " AND date <= ?"
            params.append(end_date.isoformat())
        sql += " ORDER BY date"

        rows = self._conn().execute(sql, params).fetchall()
        if not rows:
            return None
        df = pd.DataFrame(
            rows, columns=["date", "open", "high", "low", "close", "volume"]
        )
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_cached_codes(self) -> List[str]:
        rows = (
            self._conn().execute("SELECT code FROM stock_meta ORDER BY code").fetchall()
        )
        return [r[0] for r in rows]

    def get_cache_stats(self) -> pd.DataFrame:
        sql = """
            SELECT m.code, m.name, m.industry, m.market, m.updated_at,
                   COUNT(p.date) AS rows,
                   MIN(p.date) AS first_date,
                   MAX(p.date) AS last_date
            FROM stock_meta m
            LEFT JOIN stock_prices p ON m.code = p.code
            GROUP BY m.code
            ORDER BY m.code
        """
        rows = self._conn().execute(sql).fetchall()
        return pd.DataFrame(
            rows,
            columns=[
                "code",
                "name",
                "industry",
                "market",
                "updated_at",
                "rows",
                "first_date",
                "last_date",
            ],
        )

    def save_prices(self, code: str, df: pd.DataFrame) -> int:
        if df is None or len(df) == 0:
            return 0
        records = [
            (
                code,
                row["date"].strftime("%Y-%m-%d")
                if hasattr(row["date"], "strftime")
                else str(row["date"])[:10],
                float(row["open"]) if pd.notna(row["open"]) else None,
                float(row["high"]) if pd.notna(row["high"]) else None,
                float(row["low"]) if pd.notna(row["low"]) else None,
                float(row["close"]) if pd.notna(row["close"]) else None,
                float(row["volume"]) if pd.notna(row["volume"]) else None,
                float(row["amount"])
                if "amount" in row and pd.notna(row.get("amount"))
                else None,
            )
            for _, row in df.iterrows()
        ]
        conn = self._conn()
        conn.executemany(
            "INSERT OR REPLACE INTO stock_prices(code,date,open,high,low,close,volume,amount) VALUES(?,?,?,?,?,?,?,?)",
            records,
        )
        conn.commit()
        return len(records)

    def upsert_meta(
        self, code: str, name: str = "", industry: str = "", market: str = ""
    ) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._conn().execute(
            "INSERT OR REPLACE INTO stock_meta(code, name, industry, market, updated_at) VALUES(?, ?, ?, ?, ?)",
            (code, name, industry, market, now),
        )
        self._conn().commit()

    def delete_code(self, code: str) -> None:
        conn = self._conn()
        conn.execute("DELETE FROM stock_prices WHERE code = ?", (code,))
        conn.execute("DELETE FROM stock_meta WHERE code = ?", (code,))
        conn.commit()
