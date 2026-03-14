from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AI'))

from detector import detect_and_classify, BRAND_KO_TO_EN, _translate_model_name, add_embedding

from backend.database import save_training_data, get_db_stats

app = FastAPI(title="ReCheck API", version="1.0.0")

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

        # CLIP 임베딩 실시간 업데이트 (다음 요청부터 즉시 반영)
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


@app.get("/health")
async def health():
    return {"status": "ok"}