"""
bunjang.py
----------
번개장터 실거래가 수집기

사용법:
  python bunjang.py                         # 기본 브랜드 전체 수집
  python bunjang.py --brand Chanel --pages 10
"""

import asyncio
import sys
import os
import argparse
from datetime import date, datetime

import httpx

# backend 경로 추가 (price_history 임포트용)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))
from price_history import save_prices

# ── 브랜드 한영 매핑 ──────────────────────────────────────────
BRAND_MAP = {
    "Chanel":          ["샤넬", "chanel"],
    "Louis Vuitton":   ["루이비통", "louis vuitton", "lv"],
    "Gucci":           ["구찌", "gucci"],
    "Hermès":          ["에르메스", "hermes"],
    "Prada":           ["프라다", "prada"],
    "Fendi":           ["펜디", "fendi"],
    "Bottega Veneta":  ["보테가베네타", "bottega"],
    "Balenciaga":      ["발렌시아가", "balenciaga"],
    "Saint Laurent":   ["생로랑", "saint laurent", "ysl"],
    "Dior":            ["디올", "dior"],
    "Celine":          ["셀린느", "celine"],
    "Loewe":           ["로에베", "loewe"],
    "Burberry":        ["버버리", "burberry"],
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 10; SM-G973F) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Mobile Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://m.bunjang.co.kr/",
}

MIN_PRICE = 100_000    # 10만원 미만 제외 (악세서리 등)
MAX_PRICE = 100_000_000  # 1억 초과 제외


async def fetch_page(client: httpx.AsyncClient, query: str, page: int) -> list[dict]:
    """번개장터 검색 API 1페이지 호출"""
    try:
        resp = await client.get(
            "https://api.bunjang.co.kr/api/1/find_v2.json",
            params={
                "q": query,
                "order": "date",   # 최신순
                "n": 100,          # 페이지당 100개
                "page": page,
            },
            headers=HEADERS,
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json().get("list", [])
    except Exception as e:
        print(f"  [Bunjang] 페이지 {page} 오류: {e}")
        return []


def parse_item(item: dict, brand_en: str) -> dict | None:
    """번개장터 아이템 → price_history 레코드 변환"""
    try:
        price = int(item.get("price", 0))
        if not (MIN_PRICE <= price <= MAX_PRICE):
            return None

        name = item.get("name", "")
        if not name:
            return None

        # update_time: Unix timestamp
        ts = item.get("update_time", 0)
        sold_at = datetime.fromtimestamp(int(ts)).date() if ts else date.today()

        return {
            "brand":      brand_en,
            "model_name": name[:200],
            "price":      price,
            "source":     "bunjang",
            "sold_at":    sold_at,
        }
    except Exception:
        return None


async def crawl_brand(brand_en: str, keywords: list[str], pages: int = 5) -> list[dict]:
    """브랜드 키워드로 번개장터 수집"""
    records = []
    async with httpx.AsyncClient() as client:
        for kw in keywords[:2]:  # 키워드 2개만 (중복 방지)
            print(f"  검색어: '{kw}' ({pages}페이지)")
            for page in range(pages):
                items = await fetch_page(client, kw, page)
                if not items:
                    break
                for item in items:
                    rec = parse_item(item, brand_en)
                    if rec:
                        records.append(rec)
                await asyncio.sleep(0.3)  # 서버 부담 줄이기
    return records


async def run(brands: list[str] | None = None, pages: int = 5):
    """
    메인 수집 실행
    brands: None이면 BRAND_MAP 전체
    """
    targets = {k: v for k, v in BRAND_MAP.items()
               if brands is None or k in brands}

    total_saved = 0
    for brand_en, keywords in targets.items():
        print(f"\n[{brand_en}] 수집 시작...")
        records = await crawl_brand(brand_en, keywords, pages)
        if records:
            saved = await save_prices(records)
            total_saved += saved
            print(f"  → {saved}건 저장")
        else:
            print(f"  → 수집 결과 없음")

    print(f"\n✅ 번개장터 수집 완료: 총 {total_saved}건 저장")
    return total_saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="번개장터 실거래가 수집기")
    parser.add_argument("--brand",  help="특정 브랜드만 (예: Chanel)")
    parser.add_argument("--pages",  type=int, default=5, help="페이지 수 (기본 5)")
    args = parser.parse_args()

    brands = [args.brand] if args.brand else None
    asyncio.run(run(brands=brands, pages=args.pages))
