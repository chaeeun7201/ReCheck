# ReCheck Backend v2
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import hashlib
import math
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AI'))

from detector import detect_and_classify, BRAND_KO_TO_EN, _translate_model_name, add_embedding

from database import save_training_data, get_db_stats
from price_history import get_history, get_latest_price

# 번개장터 크롤러 (선택적 임포트)
try:
    from bunjang import run as bunjang_run
    _bunjang_available = True
except ImportError:
    _bunjang_available = False

app = FastAPI(title="ReCheck API", version="1.0.0")

# 스케줄러 시작 (APScheduler 설치 시에만)
try:
    from scheduler import start_scheduler
    _scheduler = start_scheduler()
except ImportError:
    print("[ReCheck] APScheduler 미설치 - 일일 수집 비활성화 (pip install apscheduler)")
    _scheduler = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 요청/응답 모델 ──────────────────────────────────────────────
class ConfirmPayload(BaseModel):
    image_b64: str          # 원본 이미지 (base64)
    bbox: list[float]       # [x1, y1, x2, y2] normalized 0~1
    brand: str
    model_name: str
    confirmed_by: str       # "ai" | "user"


class StatsResponse(BaseModel):
    total_samples: int
    ai_confirmed: int
    user_corrected: int


# ── 엔드포인트 ──────────────────────────────────────────────────
@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    """
    1단계: 이미지 업로드 → bbox 탐지 + 브랜드/모델 추측
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10MB 제한
        raise HTTPException(status_code=400, detail="파일 크기는 10MB 이하여야 합니다.")

    result = await detect_and_classify(contents)
    return result


@app.post("/api/confirm")
async def confirm(payload: ConfirmPayload):
    """
    3단계: 사용자가 확인/수정한 라벨을 DB에 저장 → 재학습 데이터 자산화
    """
    try:
        # 한국어 입력 시 영어로 변환 후 저장
        brand = BRAND_KO_TO_EN.get(payload.brand, payload.brand)
        model_name = _translate_model_name(payload.model_name)

        record_id = await save_training_data(
            image_b64=payload.image_b64,
            bbox=payload.bbox,
            brand=brand,
            model_name=model_name,
            confirmed_by=payload.confirmed_by,
        )

        # 임베딩 DB에도 즉시 반영 → 같은 사진 재입력 시 올바르게 인식
        await add_embedding(payload.image_b64, brand, model_name)

        return {
            "success": True,
            "record_id": record_id,
            "message": f"데이터가 저장되었습니다. (ID: {record_id})"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=StatsResponse)
async def stats():
    """학습 데이터 누적 현황"""
    return await get_db_stats()


@app.get("/api/price-chart")
async def price_chart(brand: str = Query(...), model_name: str = Query(...)):
    """브랜드·모델명 기반 시세 예측 차트 데이터 반환"""
    from datetime import datetime, timedelta
    import httpx, re, json as _json

    # ── 0) 인식된 모델명으로 번개장터 온디맨드 크롤 ──────────────
    if _bunjang_available and model_name:
        try:
            from bunjang import crawl_model
            await crawl_model(brand, model_name, pages=3)
        except Exception as e:
            print(f"[ReCheck] 온디맨드 크롤 실패: {e}")

    today = datetime.now()
    KR_MONTHS = ["1","2","3","4","5","6","7","8","9","10","11","12"]

    def month_label(dt, pred=False):
        suffix = "(E)" if pred else ""
        return KR_MONTHS[dt.month - 1] + "\uc6d4" + suffix

    # ── 1) DB 실거래 히스토리 조회 (현재가 확정 후 필터링하므로 일단 최신가만 먼저)
    db_latest = await get_latest_price(brand, model_name)

    # ── 2) Kream 실시간 가격 조회 ────────────────────────────────
    live_price = None
    data_source = "AI 추정"
    try:
        query = f"{brand} {model_name}"
        async with httpx.AsyncClient(timeout=6.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://kream.co.kr/search",
                params={"keyword": query},
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
                    "Referer": "https://kream.co.kr/",
                },
            )
            m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', resp.text, re.DOTALL)
            if m:
                nd = _json.loads(m.group(1))
                txt = _json.dumps(nd, ensure_ascii=False)
                prices_raw = re.findall(r'"(?:price|lowestPrice|recentPrice|releasePrice)"\s*:\s*(\d+)', txt)
                vals = [int(p) for p in prices_raw if 100000 <= int(p) <= 100000000]
                if vals:
                    live_price = int(sorted(vals)[len(vals) // 2])
                    data_source = "Kream 실시간"
    except Exception as e:
        print(f"[ReCheck] Kream 조회 실패: {e}")

    # ── 3) 현재가 결정: Kream > DB최신 > fallback ────────────────
    BASE = {
        "Chanel": 8500000, "Louis Vuitton": 2200000, "Gucci": 1600000,
        "Hermes": 32000000, "Hermès": 32000000,
        "Prada": 1300000, "Fendi": 1400000, "Bottega Veneta": 2100000,
        "Balenciaga": 1200000, "Saint Laurent": 1450000, "Dior": 3500000,
        "Celine": 1600000, "Loewe": 1800000, "Burberry": 1100000,
        "Tiffany": 2000000, "Cartier": 5000000,
    }
    seed = int(hashlib.md5(f"{brand}{model_name}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    if not live_price:
        if db_latest:
            live_price = db_latest
            data_source = "번개장터 실거래"
        else:
            base = BASE.get(brand, 1500000)
            ml = model_name.lower()
            if any(k in ml for k in ["classic", "kelly", "birkin", "2.55", "flap"]):
                base = int(base * 1.35)
            elif any(k in ml for k in ["mini", "nano", "micro"]):
                base = int(base * 0.72)
            elif any(k in ml for k in ["large", "jumbo", "gm", "xl"]):
                base = int(base * 1.15)
            live_price = int(base * (0.92 + rng.random() * 0.16))

    # ── 4) 월별 계절성 지수 ───────────────────────────────────────
    SEASON = {1:0.97, 2:0.96, 3:0.98, 4:1.00, 5:1.01, 6:0.99,
              7:0.98, 8:0.99, 9:1.02, 10:1.04, 11:1.03, 12:1.02}

    # live_price 확정 후 히스토리 조회 (현재가의 30% 미만은 다른 품목으로 간주 제외)
    real_history = await get_history(brand, model_name, months=6,
                                     min_price=int(live_price * 0.3))

    # ── 5) 과거 6개월: 실데이터 있으면 사용, 없으면 시뮬레이션 ───
    if len(real_history) >= 3:
        # 실거래 데이터로 6개월 채우기 (부족한 달은 보간)
        real_map = {r["month"]: r["price"] for r in real_history}
        history = []
        for i in range(5, -1, -1):
            dt = today - timedelta(days=30 * i)
            label = month_label(dt)
            if label in real_map:
                history.append({"month": label, "price": real_map[label]})
            else:
                # 실데이터 없는 달은 인접 데이터 보간
                noise = (rng.random() - 0.48) * 0.03
                p = live_price * (1 - 0.02 * i) * (1 + noise) * (SEASON[dt.month] / SEASON[today.month])
                history.append({"month": label, "price": max(int(p), 100000)})
        data_source = "번개장터+Kream 실거래" if live_price and data_source == "Kream 실시간" else data_source
    else:
        # 실데이터 부족 → 시뮬레이션
        history = []
        for i in range(5, -1, -1):
            dt = today - timedelta(days=30 * i)
            noise = (rng.random() - 0.48) * 0.04
            p = live_price * (1 - 0.025 * i) * (1 + noise) * (SEASON[dt.month] / SEASON[today.month])
            history.append({"month": month_label(dt), "price": max(int(p), 100000)})

    # ── 6) 미래 6개월 예측 ────────────────────────────────────────
    brand_trend = {
        "Chanel": 0.015, "Hermes": 0.012, "Hermès": 0.012, "Louis Vuitton": 0.008,
        "Dior": 0.010, "Gucci": 0.005, "Balenciaga": -0.003,
    }.get(brand, 0.007)

    prediction = []
    prev_price = live_price
    for i in range(1, 7):
        dt = today + timedelta(days=30 * i)
        seasonal_delta = SEASON[dt.month] / SEASON[today.month] - 1
        noise = (rng.random() - 0.5) * 0.015
        p = prev_price * (1 + brand_trend + seasonal_delta * 0.5 + noise)
        prev_price = int(p)
        prediction.append({"month": month_label(dt, pred=True), "price": max(int(p), 100000)})

    optimal_sell = max(prediction, key=lambda x: x["price"])
    optimal_buy_future = min(prediction, key=lambda x: x["price"])
    if live_price <= optimal_buy_future["price"]:
        optimal_buy = {"month": f"{today.month}월", "price": live_price}
    else:
        optimal_buy = optimal_buy_future

    change_1m = round((history[-1]["price"] - history[-2]["price"]) / history[-2]["price"] * 100, 1)
    change_6m = round((history[-1]["price"] - history[0]["price"])  / history[0]["price"]  * 100, 1)

    return {
        "brand": brand,
        "model_name": model_name,
        "current_price": live_price,
        "change_1m": change_1m,
        "change_6m": change_6m,
        "history": history,
        "prediction": prediction,
        "optimal_sell": optimal_sell,
        "optimal_buy":  optimal_buy,
        "data_source": data_source,
    }


@app.post("/api/crawl/bunjang")
async def crawl_bunjang(brand: str = Query(None), pages: int = Query(5)):
    """번개장터 실거래가 수동 수집 트리거 (brand=None 이면 전체 브랜드)"""
    if not _bunjang_available:
        raise HTTPException(status_code=503, detail="번개장터 크롤러를 로드할 수 없습니다.")
    try:
        brands = [brand] if brand else None
        saved = await bunjang_run(brands=brands, pages=pages)
        return {"success": True, "saved": saved}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}