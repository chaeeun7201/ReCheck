"""
detector.py (CLIP Few-shot 버전)
---------------------------------
YOLOv8 학습 없이 CLIP 임베딩 유사도로 브랜드/모델 분류

동작 방식:
  1. build_embeddings.py 로 브랜드별 임베딩 DB 미리 생성 (1회)
  2. 새 이미지 입력 → CLIP 임베딩 추출
  3. 저장된 임베딩 DB와 코사인 유사도 비교
  4. 가장 유사한 브랜드/모델 반환
"""

import asyncio
import io
import os
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image as PILImage

# ── 경로 설정 ────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
OUTPUT_DIR     = BASE_DIR / "output"
EMBEDDING_PATH = OUTPUT_DIR / "clip_embeddings.npz"
LABEL_MAP_PATH = OUTPUT_DIR / "label_map.json"

_USE_MOCK = os.getenv("RECHECK_MOCK", "false").lower() == "true"

# ── 전역 모델/임베딩 캐시 ────────────────────────────────────────
_clip_model     = None
_clip_processor = None
_embeddings     = None   # shape: (N, 512)
_labels         = None   # list of {"brand": ..., "model": ...}


# ── 브랜드 카탈로그 (Mock용) ────────────────────────────────────
BRAND_CATALOG = {
    "Chanel": ["Classic Flap", "Boy Bag", "19 Bag", "Gabrielle", "Coco Handle"],
    "Louis Vuitton": ["Neverfull", "Speedy", "Alma", "Pochette Métis", "OnTheGo"],
    "Gucci": ["Ophidia", "Marmont", "Dionysus", "Bamboo", "Jackie"],
    "Hermès": ["Birkin", "Kelly", "Constance", "Evelyne", "Picotin"],
    "Prada": ["Re-Edition 2005", "Galleria", "Cleo", "Triangle", "Symbole"],
    "Fendi": ["Baguette", "Peekaboo", "First", "By The Way", "Mon Trésor"],
    "Bottega Veneta": ["Jodie", "Cassette", "Sardine", "Intrecciato", "Andiamo"],
    "Balenciaga": ["City", "Hourglass", "Le Cagole", "Neo Classic", "Rodeo"],
    "Saint Laurent": ["Lou Camera", "Loulou", "Sunset", "Solferino", "Le 5 à 7"],
    "Dior": ["Lady Dior", "Saddle", "Book Tote", "30 Montaigne", "Bobby"],
}


# ── CLIP 모델 로드 ───────────────────────────────────────────────
async def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return True
    try:
        from transformers import CLIPProcessor, CLIPModel
        loop = asyncio.get_event_loop()

        def _load():
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            return model, processor

        _clip_model, _clip_processor = await loop.run_in_executor(None, _load)
        print("[ReCheck] CLIP 모델 로드 완료")
        return True
    except Exception as e:
        print(f"[ReCheck] CLIP 로드 실패: {e}")
        return False


# ── 임베딩 DB 로드 ───────────────────────────────────────────────
def _load_embeddings() -> bool:
    global _embeddings, _labels
    if not EMBEDDING_PATH.exists():
        print(f"[ReCheck] 임베딩 DB 없음 → build_embeddings.py 먼저 실행하세요")
        return False
    try:
        data = np.load(EMBEDDING_PATH, allow_pickle=True)
        _embeddings = data["embeddings"]   # (N, 512)
        _labels     = data["labels"].tolist()
        print(f"[ReCheck] 임베딩 DB 로드 완료: {len(_labels)}개")
        return True
    except Exception as e:
        print(f"[ReCheck] 임베딩 DB 로드 실패: {e}")
        return False


# ── 이미지 → CLIP 임베딩 추출 ───────────────────────────────────
async def _get_image_embedding(image: PILImage.Image) -> Optional[np.ndarray]:
    if _clip_model is None or _clip_processor is None:
        return None

    import torch
    loop = asyncio.get_event_loop()

    def _infer():
        inputs = _clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = _clip_model.get_image_features(**inputs)
            # L2 정규화
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy()

    return await loop.run_in_executor(None, _infer)


# ── 코사인 유사도 검색 ───────────────────────────────────────────
def _search_similar(query_emb: np.ndarray, top_k: int = 3) -> list[dict]:
    """임베딩 DB에서 가장 유사한 top_k 항목 반환"""
    # 코사인 유사도 (이미 정규화된 벡터이므로 내적 = 코사인 유사도)
    similarities = _embeddings @ query_emb  # (N,)

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    seen_brands = {}  # 브랜드별 최고 점수만 유지

    for idx in top_indices:
        label = _labels[idx]
        brand = label["brand"]
        model = label["model"]
        score = float(similarities[idx])

        if brand not in seen_brands:
            seen_brands[brand] = {"brand": brand, "model_name": model, "score": round(score, 4)}

    # 점수 기준 정렬
    top3 = sorted(seen_brands.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return top3


# ── Mock 응답 ────────────────────────────────────────────────────
def _mock_result(image_bytes: bytes) -> dict:
    import random
    brand = random.choice(list(BRAND_CATALOG.keys()))
    model = random.choice(BRAND_CATALOG[brand])
    confidence = round(random.uniform(0.72, 0.96), 3)
    return {
        "detected": True,
        "bbox": {"x1": 0.12, "y1": 0.08, "x2": 0.88, "y2": 0.92},
        "prediction": {
            "brand": brand,
            "model_name": model,
            "confidence": confidence,
            "top3": [
                {"brand": brand, "model_name": model, "score": confidence},
                {"brand": "Chanel", "model_name": "Classic Flap", "score": round(confidence - 0.15, 3)},
                {"brand": "Louis Vuitton", "model_name": "Neverfull", "score": round(confidence - 0.28, 3)},
            ]
        },
        "mode": "mock",
        "message": f"AI가 '{brand} {model}'으로 추측했습니다. (신뢰도 {int(confidence*100)}%)"
    }


# ── 메인 파이프라인 ──────────────────────────────────────────────
async def detect_and_classify(image_bytes: bytes) -> dict:
    # Mock 모드
    if _USE_MOCK:
        return _mock_result(image_bytes)

    # CLIP 로드
    clip_ok = await _load_clip()
    if not clip_ok:
        return _mock_result(image_bytes)

    # 임베딩 DB 로드 (최초 1회)
    if _embeddings is None:
        emb_ok = _load_embeddings()
        if not emb_ok:
            return _mock_result(image_bytes)

    # 이미지 열기
    image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

    # 임베딩 추출
    query_emb = await _get_image_embedding(image)
    if query_emb is None:
        return _mock_result(image_bytes)

    # 유사도 검색
    top3 = _search_similar(query_emb, top_k=3)
    if not top3:
        return {
            "detected": False,
            "bbox": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
            "prediction": None,
            "mode": "real",
            "message": "브랜드를 인식하지 못했습니다. 직접 입력해 주세요."
        }

    best = top3[0]
    confidence = best["score"]

    return {
        "detected": True,
        "bbox": {"x1": 0.05, "y1": 0.05, "x2": 0.95, "y2": 0.95},
        "prediction": {
            "brand": best["brand"],
            "model_name": best["model_name"],
            "confidence": confidence,
            "top3": top3,
        },
        "mode": "real",
        "message": f"AI가 '{best['brand']} {best['model_name']}'으로 추측했습니다. (신뢰도 {int(confidence * 100)}%)"
    }