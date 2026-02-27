"""
database.py
-----------
PostgreSQL 연결 및 학습 데이터 저장/조회

테이블: recheck_training_data
  id            SERIAL PRIMARY KEY
  image_b64     TEXT              -- bbox 크롭 이미지 (base64)
  bbox_x1       FLOAT
  bbox_y1       FLOAT
  bbox_x2       FLOAT
  bbox_y2       FLOAT
  brand         VARCHAR(100)
  model_name    VARCHAR(200)
  confirmed_by  VARCHAR(10)       -- 'ai' | 'user'
  created_at    TIMESTAMP DEFAULT NOW()
"""

import os
import json
from typing import Optional

# ── asyncpg (PostgreSQL async driver) ──────────────────────────
# 미설치 시 in-memory mock으로 폴백
_pool = None
_MOCK_STORE: list[dict] = []
_mock_counter = 0

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://recheck:recheck_pw@localhost:5432/recheck_db"
)

DDL = """
CREATE TABLE IF NOT EXISTS recheck_training_data (
    id           SERIAL PRIMARY KEY,
    image_b64    TEXT,
    bbox_x1      FLOAT,
    bbox_y1      FLOAT,
    bbox_x2      FLOAT,
    bbox_y2      FLOAT,
    brand        VARCHAR(100),
    model_name   VARCHAR(200),
    confirmed_by VARCHAR(10),
    created_at   TIMESTAMP DEFAULT NOW()
);
"""


async def _get_pool():
    global _pool
    if _pool is not None:
        return _pool
    try:
        import asyncpg
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
        async with _pool.acquire() as conn:
            await conn.execute(DDL)
        print("[ReCheck DB] PostgreSQL 연결 완료")
    except Exception as e:
        print(f"[ReCheck DB] PostgreSQL 연결 실패, in-memory mock 사용: {e}")
        _pool = None
    return _pool


async def save_training_data(
    image_b64: str,
    bbox: list[float],
    brand: str,
    model_name: str,
    confirmed_by: str,
) -> int:
    """
    사용자 확인 데이터를 DB에 저장합니다.
    PostgreSQL 미연결 시 in-memory로 폴백.
    """
    global _mock_counter

    pool = await _get_pool()

    if pool:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO recheck_training_data
                  (image_b64, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                   brand, model_name, confirmed_by)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                RETURNING id
                """,
                image_b64,
                float(bbox[0]), float(bbox[1]),
                float(bbox[2]), float(bbox[3]),
                brand, model_name, confirmed_by,
            )
            return row["id"]
    else:
        # in-memory fallback
        _mock_counter += 1
        _MOCK_STORE.append({
            "id": _mock_counter,
            "brand": brand,
            "model_name": model_name,
            "confirmed_by": confirmed_by,
        })
        return _mock_counter


async def get_db_stats() -> dict:
    """학습 데이터 누적 통계를 반환합니다."""
    pool = await _get_pool()

    if pool:
        async with pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM recheck_training_data")
            ai_cnt = await conn.fetchval(
                "SELECT COUNT(*) FROM recheck_training_data WHERE confirmed_by='ai'"
            )
            user_cnt = await conn.fetchval(
                "SELECT COUNT(*) FROM recheck_training_data WHERE confirmed_by='user'"
            )
            return {
                "total_samples": total,
                "ai_confirmed": ai_cnt,
                "user_corrected": user_cnt,
            }
    else:
        ai_cnt = sum(1 for r in _MOCK_STORE if r["confirmed_by"] == "ai")
        return {
            "total_samples": len(_MOCK_STORE),
            "ai_confirmed": ai_cnt,
            "user_corrected": len(_MOCK_STORE) - ai_cnt,
        }