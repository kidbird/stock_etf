"""
ETF Local Storage Module
SQLite 本地缓存，支持增量更新。每线程独立连接，兼容 ThreadPoolExecutor 并发写。
"""

import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

_DB_PATH = Path(__file__).parent / "etf_cache.db"


class ETFLocalStorage:
    def __init__(self, db_path: Path = _DB_PATH):
        self._db_path = str(db_path)
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        """每线程独立连接，避免跨线程共享 sqlite3.Connection。"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")   # 支持并发读写
            conn.execute("PRAGMA synchronous=NORMAL") # 写入速度与安全的平衡
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS prices (
                code    TEXT NOT NULL,
                date    TEXT NOT NULL,
                open    REAL,
                high    REAL,
                low     REAL,
                close   REAL,
                volume  REAL,
                PRIMARY KEY (code, date)
            );
            CREATE TABLE IF NOT EXISTS etf_meta (
                code        TEXT PRIMARY KEY,
                name        TEXT,
                updated_at  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_prices_code_date ON prices(code, date);
        """)
        conn.commit()

    # ── 查询 ────────────────────────────────────────────────────

    def get_last_date(self, code: str) -> Optional[date]:
        """返回本地该 ETF 最新一条数据的日期，无数据返回 None。"""
        row = self._conn().execute(
            "SELECT MAX(date) FROM prices WHERE code = ?", (code,)
        ).fetchone()
        if row and row[0]:
            return date.fromisoformat(row[0])
        return None

    def get_first_date(self, code: str) -> Optional[date]:
        """返回本地该 ETF 最早一条数据的日期，无数据返回 None。"""
        row = self._conn().execute(
            "SELECT MIN(date) FROM prices WHERE code = ?", (code,)
        ).fetchone()
        if row and row[0]:
            return date.fromisoformat(row[0])
        return None

    def load_prices(self, code: str,
                    start_date: Optional[date] = None,
                    end_date: Optional[date] = None) -> Optional[pd.DataFrame]:
        """加载指定日期区间的行情数据，返回 DataFrame 或 None。"""
        sql = "SELECT date, open, high, low, close, volume FROM prices WHERE code = ?"
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
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_cached_codes(self) -> List[str]:
        """返回 etf_meta 中所有已缓存的 ETF 代码列表（供 --update 使用）。"""
        rows = self._conn().execute("SELECT code FROM etf_meta ORDER BY code").fetchall()
        return [r[0] for r in rows]

    def get_cache_stats(self) -> pd.DataFrame:
        """返回各 ETF 本地缓存统计（代码/名称/数据量/最早/最新日期）。"""
        sql = """
            SELECT m.code, m.name, m.updated_at,
                   COUNT(p.date) AS rows,
                   MIN(p.date) AS first_date,
                   MAX(p.date) AS last_date
            FROM etf_meta m
            LEFT JOIN prices p ON m.code = p.code
            GROUP BY m.code
            ORDER BY m.code
        """
        rows = self._conn().execute(sql).fetchall()
        return pd.DataFrame(rows, columns=["code", "name", "updated_at", "rows", "first_date", "last_date"])

    # ── 写入 ────────────────────────────────────────────────────

    def save_prices(self, code: str, df: pd.DataFrame) -> int:
        """
        批量写入行情数据（INSERT OR REPLACE，幂等）。
        df 必须包含列：date, open, high, low, close, volume
        返回写入行数。
        """
        if df is None or len(df) == 0:
            return 0
        records = [
            (
                code,
                row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])[:10],
                float(row["open"])   if pd.notna(row["open"])   else None,
                float(row["high"])   if pd.notna(row["high"])   else None,
                float(row["low"])    if pd.notna(row["low"])    else None,
                float(row["close"])  if pd.notna(row["close"])  else None,
                float(row["volume"]) if pd.notna(row["volume"]) else None,
            )
            for _, row in df.iterrows()
        ]
        conn = self._conn()
        conn.executemany(
            "INSERT OR REPLACE INTO prices(code,date,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)",
            records,
        )
        conn.commit()
        return len(records)

    def upsert_meta(self, code: str, name: str = "") -> None:
        """更新 etf_meta，记录最后更新时间。"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._conn().execute(
            "INSERT OR REPLACE INTO etf_meta(code, name, updated_at) VALUES(?, ?, ?)",
            (code, name, now),
        )
        self._conn().commit()

    def delete_code(self, code: str) -> None:
        """删除某只 ETF 的全部本地数据（含 prices 和 meta）。"""
        conn = self._conn()
        conn.execute("DELETE FROM prices WHERE code = ?", (code,))
        conn.execute("DELETE FROM etf_meta WHERE code = ?", (code,))
        conn.commit()
