"""
price_history.py
----------------
실거래가 히스토리 저장 / 조회 (SQLite - 별도 설치 불필요)

DB 파일: backend/recheck_prices.db
"""

import os
import sqlite3
from datetime import date, timedelta
from typing import Optional
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "recheck_prices.db")


DDL = """
CREATE TABLE IF NOT EXISTS price_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    brand       TEXT,
    model_name  TEXT,
    price       INTEGER,
    source      TEXT,
    sold_at     TEXT,
    recorded_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_ph_brand ON price_history (brand);
CREATE INDEX IF NOT EXISTS idx_ph_sold_at ON price_history (sold_at);
"""


def _init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(DDL)


_init_db()


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


async def save_prices(records: list[dict]) -> int:
    """
    records: [{"brand", "model_name", "price", "source", "sold_at"(date)}, ...]
    반환: 저장된 건수
    """
    if not records:
        return 0

    rows = [
        (r["brand"], r["model_name"], r["price"], r["source"],
         r["sold_at"].isoformat() if hasattr(r["sold_at"], "isoformat") else str(r["sold_at"]))
        for r in records
    ]

    with _conn() as con:
        con.executemany(
            "INSERT INTO price_history (brand, model_name, price, source, sold_at) VALUES (?,?,?,?,?)",
            rows,
        )
    return len(records)


async def get_history(brand: str, model_name: str, months: int = 6, min_price: int = 0) -> list[dict]:
    """
    최근 N개월간 월별 평균 실거래가 반환
    모델명 매칭 실패 시 브랜드 전체 평균으로 fallback
    [{"month": "3월", "price": 1200000}, ...]
    """
    cutoff = date.today().replace(day=1)
    for _ in range(months):
        cutoff = (cutoff - timedelta(days=1)).replace(day=1)

    keyword = model_name.split()[0] if model_name else ""

    def _median_by_month(con, where_extra, params):
        """월별 가격 목록 가져와서 Python에서 중앙값 계산"""
        rows = con.execute(
            f"""
            SELECT
                CAST(strftime('%m', sold_at) AS INTEGER) AS mon,
                price
            FROM price_history
            WHERE brand = ? {where_extra}
              AND price >= ?
              AND sold_at >= ?
            ORDER BY mon, price
            """,
            params,
        ).fetchall()
        from collections import defaultdict
        buckets = defaultdict(list)
        for r in rows:
            buckets[r["mon"]].append(r["price"])
        result = []
        for mon in sorted(buckets):
            vals = sorted(buckets[mon])
            median = vals[len(vals) // 2]
            result.append({"mon": mon, "median": median})
        return result

    with _conn() as con:
        # 1차: 모델명 정확 매칭
        rows = _median_by_month(con, "AND model_name = ?",
                                (brand, model_name, min_price, cutoff.isoformat()))
        # 2차: 키워드 부분 매칭
        if not rows:
            rows = _median_by_month(con, "AND model_name LIKE ?",
                                    (brand, f"%{keyword}%", min_price, cutoff.isoformat()))

    return [{"month": f"{r['mon']}월", "price": r["median"]} for r in rows]


async def get_latest_price(brand: str, model_name: str) -> Optional[int]:
    """브랜드·모델의 가장 최근 수집 가격 반환 (없으면 브랜드 중앙값)"""
    keyword = model_name.split()[0] if model_name else ""

    with _conn() as con:
        # 1차: 모델명 정확 매칭 (온디맨드 크롤 데이터)
        row = con.execute(
            "SELECT price FROM price_history WHERE brand = ? AND model_name = ? ORDER BY sold_at DESC LIMIT 1",
            (brand, model_name),
        ).fetchone()
        if row:
            return row["price"]

        # 2차: 키워드 부분 매칭
        row = con.execute(
            "SELECT price FROM price_history WHERE brand = ? AND model_name LIKE ? ORDER BY sold_at DESC LIMIT 1",
            (brand, f"%{keyword}%"),
        ).fetchone()
        if row:
            return row["price"]

    return None
