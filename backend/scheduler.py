"""
scheduler.py
------------
APScheduler 기반 Kream 일일 가격 수집
매일 새벽 3시에 주요 브랜드 현재가를 price_history에 저장
"""

import re
import json
from datetime import date

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from price_history import save_prices

# 매일 수집할 브랜드·모델 목록
WATCH_LIST = [
    ("Chanel",         "Classic Flap"),
    ("Chanel",         "Boy Bag"),
    ("Chanel",         "Mini"),
    ("Louis Vuitton",  "Neverfull"),
    ("Louis Vuitton",  "Speedy"),
    ("Louis Vuitton",  "Alma"),
    ("Gucci",          "Marmont"),
    ("Gucci",          "Dionysus"),
    ("Hermès",         "Birkin"),
    ("Hermès",         "Kelly"),
    ("Prada",          "Re-Edition"),
    ("Dior",           "Lady Dior"),
    ("Bottega Veneta", "Jodie"),
    ("Balenciaga",     "City"),
    ("Saint Laurent",  "Loulou"),
    ("Celine",         "Box Bag"),
    ("Loewe",          "Puzzle"),
    ("Fendi",          "Baguette"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Referer": "https://kream.co.kr/",
}


async def fetch_kream_price(client: httpx.AsyncClient, brand: str, model_name: str) -> int | None:
    """Kream 검색 결과에서 중앙값 가격 추출"""
    try:
        resp = await client.get(
            "https://kream.co.kr/search",
            params={"keyword": f"{brand} {model_name}"},
            headers=HEADERS,
            timeout=8.0,
            follow_redirects=True,
        )
        m = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            resp.text, re.DOTALL
        )
        if not m:
            return None
        txt = json.dumps(json.loads(m.group(1)), ensure_ascii=False)
        prices_raw = re.findall(
            r'"(?:price|lowestPrice|recentPrice|releasePrice)"\s*:\s*(\d+)', txt
        )
        vals = [int(p) for p in prices_raw if 100_000 <= int(p) <= 100_000_000]
        if vals:
            return sorted(vals)[len(vals) // 2]
    except Exception as e:
        print(f"[Scheduler] Kream 오류 {brand} {model_name}: {e}")
    return None


async def daily_kream_crawl():
    """매일 실행되는 Kream 가격 수집 작업"""
    print("[Scheduler] Kream 일일 수집 시작")
    records = []
    today = date.today()

    async with httpx.AsyncClient() as client:
        for brand, model_name in WATCH_LIST:
            price = await fetch_kream_price(client, brand, model_name)
            if price:
                records.append({
                    "brand":      brand,
                    "model_name": model_name,
                    "price":      price,
                    "source":     "kream",
                    "sold_at":    today,
                })
                print(f"  {brand} {model_name}: ₩{price:,}")

    if records:
        saved = await save_prices(records)
        print(f"[Scheduler] Kream 수집 완료: {saved}건 저장")
    else:
        print("[Scheduler] Kream 수집 결과 없음")


def start_scheduler() -> AsyncIOScheduler:
    """FastAPI 시작 시 호출 — 스케줄러 등록 및 시작"""
    scheduler = AsyncIOScheduler(timezone="Asia/Seoul")
    # 매일 새벽 3시 실행
    scheduler.add_job(daily_kream_crawl, "cron", hour=3, minute=0)
    scheduler.start()
    print("[Scheduler] 일일 Kream 수집 등록 완료 (매일 03:00)")
    return scheduler
