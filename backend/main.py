# ReCheck Backend v2
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import hashlib
import math
import random

# Windows 콘솔(cp949)이 이모지/특수문자(em-dash 등) 출력 시 죽는 것 방지
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AI'))

from detector import detect_and_classify, BRAND_KO_TO_EN, _translate_model_name, add_embedding, assess_condition, verify_authenticity

from database import save_training_data, get_db_stats
from price_history import get_history, get_latest_price, save_prices

async def _fetch_bunjang_price(brand: str, model_name: str) -> tuple[int | None, str]:
    """
    번개장터 API에서 브랜드+모델 실시간 시세 조회
    반환: (중앙값 가격 or None, 데이터출처)
    """
    import httpx
    from datetime import date

    if not _bunjang_available:
        return None, "AI 추정"

    try:
        from bunjang import BRAND_KO, MODEL_KO, fetch_page, parse_item
    except ImportError:
        return None, "AI 추정"

    brand_ko  = BRAND_KO.get(brand, brand)
    model_ko  = MODEL_KO.get(model_name, model_name)
    queries   = list(dict.fromkeys([
        f"{brand_ko} {model_ko}",
        f"{brand_ko} {model_name}" if model_ko != model_name else None,
    ]))
    queries = [q for q in queries if q]

    prices = []
    try:
        async with httpx.AsyncClient() as client:
            for q in queries:
                items = await fetch_page(client, q, page=0)
                for item in items:
                    rec = parse_item(item, brand)
                    if rec:
                        prices.append(rec["price"])
                if prices:
                    break
    except Exception as e:
        print(f"[Bunjang] 시세 조회 실패: {e}")
        return None, "AI 추정"

    if not prices:
        return None, "AI 추정"

    prices.sort()
    median = prices[len(prices) // 2]

    # DB에 저장 (히스토리 축적)
    records = [{"brand": brand, "model_name": model_name,
                "price": p, "source": "bunjang", "sold_at": date.today()}
               for p in prices]
    try:
        await save_prices(records)
    except Exception:
        pass

    return median, "번개장터 실시간"

# 번개장터 크롤러 (선택적 임포트)
try:
    from bunjang import run as bunjang_run
    _bunjang_available = True
except ImportError:
    _bunjang_available = False

# 영→한 브랜드 매핑 (KREAM 검색용)
_BRAND_EN_TO_KO = {
    "Nike": "나이키", "Adidas": "아디다스", "New Balance": "뉴발란스",
    "The North Face": "노스페이스", "Converse": "컨버스", "Vans": "반스",
    "Puma": "퓨마", "Reebok": "리복", "ASICS": "아식스", "Salomon": "살로몬",
    "On Running": "온러닝", "Patagonia": "파타고니아", "Columbia": "콜롬비아",
    "Arc'teryx": "아크테릭스", "Moncler": "몽클레어", "Stone Island": "스톤아일랜드",
    "Supreme": "슈프림", "BAPE": "베이프", "Stussy": "스투시", "Uniqlo": "유니클로",
    "Carhartt": "카하트", "Chanel": "샤넬", "Louis Vuitton": "루이비통",
    "Gucci": "구찌", "Hermes": "에르메스", "Prada": "프라다", "Dior": "디올",
    "Fendi": "펜디", "Balenciaga": "발렌시아가", "Bottega Veneta": "보테가베네타",
    "Saint Laurent": "생로랑", "Celine": "셀린느", "Loewe": "로에베",
    "Burberry": "버버리",
}

async def _fetch_kream_price_live(brand: str, model_name: str) -> tuple[int | None, str]:
    """KREAM 실시간 시세 조회 — 1순위 가격 소스"""
    import httpx, re as _re, json as _json

    brand_ko = _BRAND_EN_TO_KO.get(brand, brand)
    # 모델명에 이미 한글 브랜드가 포함돼 있으면 중복 방지
    if brand_ko and brand_ko in model_name:
        query = model_name
    else:
        query = f"{brand_ko} {model_name}".strip()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9",
        "Referer": "https://kream.co.kr/",
    }
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(
                "https://kream.co.kr/search",
                params={"keyword": query},
                headers=headers,
                timeout=10.0,
            )
        m = _re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            resp.text, _re.DOTALL,
        )
        if not m:
            return None, "AI 추정"
        txt = _json.dumps(_json.loads(m.group(1)), ensure_ascii=False)
        prices_raw = _re.findall(
            r'"(?:price|lowestPrice|recentPrice|tradePrice)"\s*:\s*(\d+)', txt
        )
        vals = [int(p) for p in prices_raw if 1_000 <= int(p) <= 100_000_000]
        if vals:
            return sorted(vals)[len(vals) // 2], "KREAM 실시간"
    except Exception as e:
        print(f"[KREAM] 시세 조회 실패 ({query}): {e}")
    return None, "AI 추정"

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(_):
    """서버 시작 시 CLIP 모델 + 텍스트 임베딩 캐시 미리 계산 → 첫 분석 속도 개선"""
    from detector import _load_clip, _classify_brand_zeroshot, verify_authenticity
    import numpy as np

    print("[ReCheck] 워밍업 시작...")
    ok = await _load_clip()
    if ok:
        dummy_emb = np.zeros(512, dtype=np.float32)
        dummy_emb[0] = 1.0
        await _classify_brand_zeroshot(dummy_emb)

        from PIL import Image as PILImage
        import io
        img = PILImage.new("RGB", (224, 224), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        await verify_authenticity(buf.getvalue())
        print("[ReCheck] 워밍업 완료 — 이제 첫 분석도 빠릅니다")
    else:
        print("[ReCheck] CLIP 로드 실패 — 워밍업 스킵")
    yield

app = FastAPI(title="ReCheck API", version="1.0.0", lifespan=lifespan)

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


@app.post("/api/assess-condition")
async def assess_condition_api(file: UploadFile = File(...)):
    """이미지 → 상태 등급(S/A/B/C) + 세부 분석 반환"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    contents = await file.read()
    result = await assess_condition(contents)
    return result


@app.post("/api/verify-authenticity")
async def verify_authenticity_api(file: UploadFile = File(...)):
    """이미지 → 로고·재질·봉제·폰트 4항목 진위 분석"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    contents = await file.read()
    result = await verify_authenticity(contents)
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

    today = datetime.now()
    KR_MONTHS = ["1","2","3","4","5","6","7","8","9","10","11","12"]

    def month_label(dt, pred=False):
        suffix = "(E)" if pred else ""
        return KR_MONTHS[dt.month - 1] + "\uc6d4" + suffix

    # ── 1) KREAM 실시간 시세 (1순위) ─────────────────────────────
    live_price, data_source = await _fetch_kream_price_live(brand, model_name)

    # ── 2) 번개장터 실시간 (2순위) ────────────────────────────────
    if not live_price:
        live_price, data_source = await _fetch_bunjang_price(brand, model_name)

    # ── 3) DB 최신가 (3순위) ─────────────────────────────────────
    if not live_price:
        live_price = await get_latest_price(brand, model_name)
        if live_price:
            data_source = "번개장터 DB"

    # ── 4) 현재가 결정: KREAM > 번개장터 > DB > fallback ─────────
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
        base = BASE.get(brand, 1500000)
        ml = model_name.lower()
        if any(k in ml for k in ["classic", "kelly", "birkin", "2.55", "flap"]):
            base = int(base * 1.35)
        elif any(k in ml for k in ["mini", "nano", "micro"]):
            base = int(base * 0.72)
        elif any(k in ml for k in ["large", "jumbo", "gm", "xl"]):
            base = int(base * 1.15)
        live_price = int(base * (0.92 + rng.random() * 0.16))
        data_source = "AI 추정"

    # ── 5) 월별 계절성 지수 ───────────────────────────────────────
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
        data_source = "Kream+실거래" if data_source == "Kream 실시간" else data_source
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


# ── 가격 이상 탐지 + 최적가 가이드 ─────────────────────────────
class PriceCheckPayload(BaseModel):
    brand: str
    model_name: str
    asking_price: int       # 판매자가 제시한 가격 (원)
    condition: str = "A"    # S / A / B / C

@app.post("/api/price-check")
async def price_check(payload: PriceCheckPayload):
    """
    판매가 입력 → 시세 대비 이상 탐지 + 최적가 가이드
    """
    from datetime import datetime, timedelta
    brand = payload.brand
    model_name = payload.model_name
    asking = payload.asking_price
    condition = payload.condition.upper()

    # ── 1) KREAM 실시간 시세 (1순위) ────────────────────────────────
    live_price, data_source = await _fetch_kream_price_live(brand, model_name)

    # ── 2) 번개장터 실시간 (2순위) ───────────────────────────────────
    if not live_price:
        live_price, data_source = await _fetch_bunjang_price(brand, model_name)

    # ── 3) DB 저장값 (3순위) ────────────────────────────────────────
    if not live_price:
        live_price = await get_latest_price(brand, model_name)
        if live_price:
            data_source = "번개장터 DB"

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
        base = BASE.get(brand, 1500000)
        ml = model_name.lower()
        if any(k in ml for k in ["classic", "kelly", "birkin", "2.55", "flap"]):
            base = int(base * 1.35)
        elif any(k in ml for k in ["mini", "nano", "micro"]):
            base = int(base * 0.72)
        elif any(k in ml for k in ["large", "jumbo", "gm", "xl"]):
            base = int(base * 1.15)
        live_price = int(base * (0.92 + rng.random() * 0.16))
        data_source = "AI 추정"

    # ── 2) 상태 등급별 적정가 범위 산출 ─────────────────────────
    # 새상품 시세 대비 중고 상태별 기대 가격 비율
    CONDITION_RANGE = {
        "S": (0.88, 0.95),  # 거의 새것
        "A": (0.72, 0.85),  # 사용감 적음
        "B": (0.55, 0.70),  # 사용감 있음
        "C": (0.38, 0.52),  # 사용감 많음
    }
    cond = CONDITION_RANGE.get(condition, CONDITION_RANGE["A"])
    fair_min = int(live_price * cond[0])
    fair_max = int(live_price * cond[1])
    fair_mid = int((fair_min + fair_max) / 2)

    # ── 3) 이상 탐지 ─────────────────────────────────────────────
    ratio = asking / fair_mid  # 적정가 대비 비율

    if ratio < 0.5:
        level = "danger"
        signal = f"시세 대비 {round((1-ratio)*100)}% 저렴 — 사기 가능성이 높습니다"
        score = 90
    elif ratio < 0.75:
        level = "caution"
        signal = f"시세 대비 {round((1-ratio)*100)}% 저렴 — 상태 확인이 필요합니다"
        score = 45
    elif ratio <= 1.20:
        level = "safe"
        signal = f"시세 범위 내 적정가입니다"
        score = 10
    elif ratio <= 1.50:
        level = "caution"
        signal = f"시세 대비 {round((ratio-1)*100)}% 비쌈 — 흥정 여지가 있습니다"
        score = 35
    else:
        level = "caution"
        signal = f"시세 대비 {round((ratio-1)*100)}% 초과 — 과도하게 높은 가격입니다"
        score = 50

    # ── 4) 미래 6개월 예측으로 최적 타이밍 가이드 ───────────────
    today = datetime.now()
    SEASON = {1:0.97, 2:0.96, 3:0.98, 4:1.00, 5:1.01, 6:0.99,
              7:0.98, 8:0.99, 9:1.02, 10:1.04, 11:1.03, 12:1.02}
    brand_trend = {
        "Chanel": 0.015, "Hermes": 0.012, "Hermès": 0.012, "Louis Vuitton": 0.008,
        "Dior": 0.010, "Gucci": 0.005, "Balenciaga": -0.003,
    }.get(brand, 0.007)

    prediction = []
    prev = live_price
    for i in range(1, 7):
        dt = today + timedelta(days=30 * i)
        seasonal_delta = SEASON[dt.month] / SEASON[today.month] - 1
        noise = (rng.random() - 0.5) * 0.015
        p = int(prev * (1 + brand_trend + seasonal_delta * 0.5 + noise))
        prev = p
        prediction.append({"month": f"{dt.month}월(E)", "price": max(p, 100000)})

    optimal_sell = max(prediction, key=lambda x: x["price"])
    optimal_buy_future = min(prediction, key=lambda x: x["price"])
    optimal_buy = {"month": f"{today.month}월", "price": live_price} \
        if live_price <= optimal_buy_future["price"] else optimal_buy_future

    # 판매자 관점 가이드
    sell_gain = round((optimal_sell["price"] - live_price) / live_price * 100, 1)
    buy_save  = round((live_price - optimal_buy["price"]) / live_price * 100, 1)

    # ── 5) 가격 제안 메시지 ──────────────────────────────────────
    if level == "danger":
        guide = f"이 가격({asking:,}원)은 {brand} {model_name} {condition}등급 적정가({fair_min:,}~{fair_max:,}원)보다 지나치게 낮습니다. 사기를 의심하세요."
    elif asking < fair_min:
        guide = f"적정가보다 저렴합니다. 상품 상태를 꼼꼼히 확인하세요. 적정가: {fair_min:,}~{fair_max:,}원"
    elif asking > fair_max:
        guide = f"적정가보다 높습니다. {fair_max:,}원 이하로 흥정해보세요."
    else:
        guide = f"적정 가격 범위 내입니다. {brand} {model_name} {condition}등급 기준 {fair_min:,}~{fair_max:,}원"

    return {
        "level": level,
        "score": score,
        "signal": signal,
        "guide": guide,
        "market_price": live_price,
        "fair_min": fair_min,
        "fair_max": fair_max,
        "fair_mid": fair_mid,
        "asking_price": asking,
        "condition": condition,
        "ratio": round(ratio, 2),
        "data_source": data_source,
        "optimal_sell": optimal_sell,
        "optimal_buy": optimal_buy,
        "sell_gain": sell_gain,
        "buy_save": buy_save,
        "prediction": prediction,
    }


# ── 판매 링크 분석 ────────────────────────────────────────────────
class ListingPayload(BaseModel):
    url: str

_BRAND_KO_EN = {
    "나이키": "Nike", "아디다스": "Adidas", "뉴발란스": "New Balance",
    "노스페이스": "The North Face", "컨버스": "Converse", "반스": "Vans",
    "퓨마": "Puma", "리복": "Reebok", "아식스": "ASICS", "살로몬": "Salomon",
    "온러닝": "On Running", "파타고니아": "Patagonia", "콜롬비아": "Columbia",
    "아크테릭스": "Arc'teryx", "몽클레어": "Moncler", "몽클렌": "Moncler",
    "스톤아일랜드": "Stone Island", "슈프림": "Supreme", "베이프": "BAPE",
    "스투시": "Stussy", "유니클로": "Uniqlo", "카하트": "Carhartt",
    "샤넬": "Chanel", "루이비통": "Louis Vuitton", "구찌": "Gucci",
    "에르메스": "Hermes", "프라다": "Prada", "디올": "Dior",
    "펜디": "Fendi", "발렌시아가": "Balenciaga", "보테가베네타": "Bottega Veneta",
    "생로랑": "Saint Laurent", "셀린느": "Celine", "로에베": "Loewe",
    "버버리": "Burberry",
}
_SALE_PHRASES = [
    "팝니다", "팔아요", "판매합니다", "판매해요", "판매중", "판매 중",
    "처분합니다", "처분해요", "처분중", "처분 중", "급처", "급매",
    "나눔합니다", "나눔해요", "삽니다", "구합니다", "구매합니다",
    "삼니다", "교환합니다", "교환해요",
]

def _clean_title(title: str) -> str:
    import re
    # "상품명 | 사이트명" 패턴 제거
    title = title.split(" | ")[0]
    for phrase in _SALE_PHRASES:
        title = title.replace(phrase, "")
    return re.sub(r"\s+", " ", title).strip(" -|/[]()【】")

def _infer_brand_model_from_text(text: str, explicit_brand: str = "") -> tuple[str, str]:
    """제목에서 영문 브랜드 + 정제된 모델명 추출"""
    model_name = _clean_title(text)

    # API에서 브랜드가 직접 제공된 경우
    if explicit_brand:
        brand_en = _BRAND_KO_EN.get(explicit_brand, explicit_brand)
        return brand_en, model_name

    text_lower = text.lower()
    # 한국어 브랜드명 매핑
    for ko, en in _BRAND_KO_EN.items():
        if ko in text_lower:
            return en, model_name

    # 영문 브랜드명 직접 매칭
    try:
        from bunjang import BRAND_MAP
        for brand_en, keywords in BRAND_MAP.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    return brand_en, model_name
    except ImportError:
        pass

    return "", model_name

@app.post("/api/analyze-listing")
async def analyze_listing(payload: ListingPayload):
    """번개장터·당근마켓 판매 링크 → 상품 정보 추출"""
    import re, html as html_lib, httpx, json as json_lib

    url = payload.url.strip()
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    def parse_og(html_text: str, prop: str) -> str:
        escaped = re.escape(prop)
        for pat in [
            r'<meta\b[^>]+property=["\']' + escaped + r'["\'][^>]+content=["\']([^"\']*)["\']',
            r'<meta\b[^>]+content=["\']([^"\']*)["\'][^>]+property=["\']' + escaped + r'["\']',
        ]:
            hit = re.search(pat, html_text, re.IGNORECASE | re.DOTALL)
            if hit:
                return html_lib.unescape(hit.group(1))
        return ""

    def extract_price(html_text: str, parse_og_fn) -> int:
        # 1) OG price 태그
        for prop in ["product:price:amount", "og:price:amount"]:
            raw = parse_og_fn(prop)
            if raw:
                digits = re.sub(r"[^\d]", "", raw)
                return int(digits) if digits else 0
        # 2) __NEXT_DATA__ JSON에서 price 필드
        m = re.search(r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', html_text, re.DOTALL)
        if m:
            try:
                nd = json_lib.loads(m.group(1))
                nd_str = json_lib.dumps(nd)
                hit = re.search(r'"price"\s*:\s*(\d+)', nd_str)
                if hit:
                    candidate = int(hit.group(1))
                    if 1000 <= candidate <= 100_000_000:
                        return candidate
            except Exception:
                pass
        # 3) 페이지 내 "price": 숫자 패턴
        hit = re.search(r'"price"\s*:\s*(\d{4,9})', html_text)
        if hit:
            candidate = int(hit.group(1))
            if 1000 <= candidate <= 100_000_000:
                return candidate
        # 4) 한국 원화 텍스트 패턴 (예: 100,000원)
        hit = re.search(r'([\d,]{4,})\s*원', html_text)
        if hit:
            candidate = int(hit.group(1).replace(",", ""))
            if 1000 <= candidate <= 100_000_000:
                return candidate
        return 0

    # ── 번개장터 ────────────────────────────────────────────────────
    if "bunjang.co.kr" in url:
        m = re.search(r'/products/(\d+)', url)
        if not m:
            raise HTTPException(status_code=400, detail="번개장터 상품 링크를 확인해 주세요. (/products/숫자 형식)")
        pid = m.group(1)

        title = description = image_url = explicit_brand = ""
        price = 0

        # Googlebot UA로 데스크톱 URL 요청 → SSR 활성화
        GOOGLEBOT_UA = "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                r = await client.get(
                    f"https://www.bunjang.co.kr/products/{pid}",
                    headers={**HEADERS, "User-Agent": GOOGLEBOT_UA},
                    timeout=15.0,
                )
            html_text = r.text
        except Exception as e:
            html_text = ""

        if not title and html_text:
            og_title = parse_og(html_text, "og:title")
            if og_title:
                # "상품명 | 브랜드 중고거래 플랫폼, 번개장터" → "상품명"
                title = og_title.split(" | ")[0].strip()
                if title == "번개장터":
                    title = ""
            if title:
                description = parse_og(html_text, "og:description")
                image_url = parse_og(html_text, "og:image") or None
                price = extract_price(html_text, lambda p: parse_og(html_text, p))

        # 모든 전략 실패 시 — pid만이라도 반환해서 사용자가 수동 입력 가능하도록
        if not title:
            raise HTTPException(
                status_code=404,
                detail="상품 정보를 가져올 수 없습니다. 판매 종료, 삭제, 또는 비공개 상품일 수 있습니다."
            )

        brand, model_name = _infer_brand_model_from_text(title, explicit_brand)
        return {"platform": "bunjang", "title": title, "price": price,
                "description": description, "image_url": image_url or None,
                "brand": brand, "model_name": model_name}

    # ── 당근마켓 ────────────────────────────────────────────────────
    elif "daangn.com" in url:
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(url, headers=HEADERS, timeout=15.0)
                html_text = resp.text
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"당근마켓 응답 오류: {e}")

        def og(prop):
            return parse_og(html_text, prop)

        title = og("og:title")
        if not title:
            raise HTTPException(status_code=422, detail="당근마켓 상품 정보를 가져올 수 없습니다. 링크를 확인해 주세요.")

        description = og("og:description")
        image_url = og("og:image") or None

        price = extract_price(html_text, og)

        brand, model_name = _infer_brand_model_from_text(title + " " + description)
        return {"platform": "daangn", "title": title, "price": price,
                "description": description, "image_url": image_url,
                "brand": brand, "model_name": model_name}

    else:
        raise HTTPException(
            status_code=400,
            detail="지원하지 않는 플랫폼입니다. 번개장터(bunjang.co.kr) 또는 당근마켓(daangn.com) 링크를 입력해 주세요."
        )


# ── URL 사기 탐지 ────────────────────────────────────────────────
class UrlCheckPayload(BaseModel):
    url: str

@app.post("/api/check-url")
async def check_url(payload: UrlCheckPayload):
    """URL 입력 → 안전/주의/위험 3단계 위험도 분석"""
    from urllib.parse import urlparse
    import re

    raw = payload.url.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="URL을 입력해 주세요.")

    # scheme 없으면 붙여서 파싱
    try:
        parsed = urlparse(raw if "://" in raw else "https://" + raw)
    except Exception:
        return {"level": "danger", "score": 95, "reasons": ["URL 형식이 올바르지 않습니다."], "seller_trust": {}}

    domain = parsed.netloc.lower().split(":")[0]  # 포트 제거
    scheme = parsed.scheme.lower()

    # ── 점수 설계 원칙 ───────────────────────────────────────────
    # 0~20: 안전 — 공식 플랫폼 + 정상 게시글
    # 21~50: 주의 — 불확실한 요소 있음, 거래 전 확인 필요
    # 51~100: 위험 — 사기 강신호 하나 이상 존재

    score = 0
    reasons: list[str] = []

    # ── 공식 플랫폼 여부 ────────────────────────────────────────
    SAFE_DOMAINS = {
        "bunjang.com": "번개장터",
        "www.bunjang.com": "번개장터",
        "m.bunjang.com": "번개장터",
        "bunjang.co.kr": "번개장터",
        "www.bunjang.co.kr": "번개장터",
        "m.bunjang.co.kr": "번개장터",
        "joonggonara.co.kr": "중고나라",
        "www.joonggonara.co.kr": "중고나라",
        "joongna.com": "중고나라",
        "www.joongna.com": "중고나라",
        "danggeun.com": "당근마켓",
        "www.danggeun.com": "당근마켓",
        "m.danggeun.com": "당근마켓",
        "karrot.market": "당근마켓",
        "www.karrot.market": "당근마켓",
        "kream.co.kr": "Kream",
        "www.kream.co.kr": "Kream",
        "cafe.naver.com": "네이버 카페",
        "m.cafe.naver.com": "네이버 카페",
    }
    if domain in SAFE_DOMAINS:
        # 공식 플랫폼: 도메인 자체는 안전, 게시글 내용만 분석
        score += 0
        reasons.append(f"공식 플랫폼({SAFE_DOMAINS[domain]}) 링크입니다.")
    else:
        # 알 수 없는 도메인: 자체로 주의 수준
        score += 30
        reasons.append(f"공식 중고거래 플랫폼이 아닌 외부 도메인입니다: {domain}")

    # ── IP 주소 직접 사용 (+65) — 정상 사이트는 IP 사용 안 함 ──
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain):
        score += 65
        reasons.append("IP 주소를 직접 사용하는 링크 — 피싱 사이트일 가능성이 매우 높습니다.")

    # ── 사칭 도메인 (+60) — 공식 플랫폼 이름 포함한 가짜 도메인 ─
    LOOKALIKE = [
        (r"bunjang", "번개장터"),
        (r"danggeun|karrot", "당근마켓"),
        (r"joongna|joonggonara", "중고나라"),
    ]
    if domain not in SAFE_DOMAINS:
        for pattern, name in LOOKALIKE:
            if re.search(pattern, domain):
                score += 60
                reasons.append(f"'{name}'을 사칭한 가짜 도메인으로 의심됩니다.")
                break

    # ── 단축 URL (+20) — 목적지 숨김, 피싱 링크에 자주 사용 ────
    SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "me2.do", "han.gl"]
    if any(s in domain for s in SHORTENERS):
        score += 20
        reasons.append("단축 URL — 실제 목적지를 숨길 수 있습니다.")

    # ── 의심 TLD (+20) — 피싱에 압도적으로 많이 사용되는 확장자 ─
    BAD_TLDS = [".xyz", ".top", ".click", ".online", ".site", ".fun", ".icu", ".vip", ".bid", ".work", ".pw"]
    for tld in BAD_TLDS:
        if domain.endswith(tld):
            score += 20
            reasons.append(f"피싱에 자주 쓰이는 도메인 확장자({tld})입니다.")
            break

    # ── HTTP (+10) — 암호화 없음, 단독으론 낮은 위험 ───────────
    if scheme == "http" and domain not in SAFE_DOMAINS:
        score += 10
        reasons.append("암호화되지 않은 HTTP 연결입니다.")

    score = max(0, min(100, score))

    if score <= 25:
        level = "safe"
    elif score <= 55:
        level = "caution"
    else:
        level = "danger"

    # ── 실제 연결 + 페이지 내용 분석 ────────────────────────────
    import httpx
    connect_status = None
    http_code = None

    # 게시글 내용 사기 키워드 (텍스트에서 탐지)
    CONTENT_SCAM = [
        ("선입금", "선입금 요구 문구 감지 — 사기 가능성이 높습니다.", 35),
        ("입금 먼저", "입금 선요구 문구 감지 — 사기 가능성이 높습니다.", 35),
        ("계좌로 먼저", "계좌 선입금 요구 감지.", 30),
        ("카카오페이", "카카오페이 선결제 요구 패턴 감지.", 15),
        ("직거래 사절", "직거래를 거부하는 패턴 — 주의 필요.", 20),
        ("해외직구", "해외발송 사기 패턴 주의.", 10),
        ("정품보장", "과장된 정품 보장 문구.", 8),
        ("급처", "급처 문구 — 판단을 서두르게 하는 패턴.", 8),
        ("떨이", "가격 급락 유도 문구.", 5),
    ]

    try:
        target = raw if "://" in raw else "https://" + raw
        async with httpx.AsyncClient(follow_redirects=True, timeout=8.0) as client:
            resp = await client.get(target, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "ko-KR,ko;q=0.9",
            })
            http_code = resp.status_code
            if http_code < 400:
                connect_status = "ok"
                reasons.append(f"링크 연결 성공 (HTTP {http_code})")

                # 페이지 텍스트에서 사기 키워드 탐지
                page_text = resp.text
                for kw, msg, pts in CONTENT_SCAM:
                    if kw in page_text:
                        score += pts
                        reasons.append(f"⚠ 게시글 내용: {msg}")
            elif http_code == 404:
                connect_status = "error"
                score += 10
                reasons.append("페이지가 존재하지 않습니다 (HTTP 404). 삭제된 게시글일 수 있습니다.")
            else:
                connect_status = "blocked"
                reasons.append(f"서버 응답: HTTP {http_code}")
    except httpx.TimeoutException:
        connect_status = "timeout"
        score += 15
        reasons.append("연결 시간 초과 — 서버가 응답하지 않습니다.")
    except Exception as e:
        connect_status = "error"
        score += 20
        reasons.append(f"링크에 접속할 수 없습니다 — {type(e).__name__}")

    score = max(0, min(100, score))

    if score <= 25:
        level = "safe"
    elif score <= 55:
        level = "caution"
    else:
        level = "danger"

    seller_trust = {
        "platform_verified": domain in SAFE_DOMAINS,
        "https_secure": scheme == "https",
        "known_domain": domain in SAFE_DOMAINS,
        "reachable": connect_status == "ok",
        "http_code": http_code,
    }

    return {
        "level": level,
        "score": score,
        "reasons": reasons,
        "seller_trust": seller_trust,
        "domain": domain,
        "connect_status": connect_status,
    }