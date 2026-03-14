"""
detector.py (CLIP Few-shot + YOLOv8 bbox + YOLOv8 브랜드 분류 버전)
------------------------------------------------
동작 방식:
  1. yolov8n.pt (COCO 베이스) → 이미지에서 가방 bbox 탐지
  2. best.pt (파인튜닝) → 브랜드 분류
  3. build_embeddings.py 로 브랜드별 CLIP 임베딩 DB 미리 생성 (1회)
  4. 새 이미지 입력 → CLIP 임베딩 추출 → 해당 브랜드 내에서 코사인 유사도 비교 → 모델명 반환
  5. bbox + 브랜드/모델 반환
"""

import asyncio
import io
import os
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image as PILImage

# ── 경로 설정 ────────────────────────────────────────────────────
BASE_DIR            = Path(__file__).parent
OUTPUT_DIR          = BASE_DIR / "output"
EMBEDDING_PATH      = OUTPUT_DIR / "clip_embeddings.npz"
LABEL_MAP_PATH      = OUTPUT_DIR / "label_map.json"
YOLO_BASE_PATH      = BASE_DIR / "yolov8n.pt"   # COCO 사전학습 베이스 모델 (bbox 탐지용)

_USE_MOCK = os.getenv("RECHECK_MOCK", "false").lower() == "true"

# 모델명 신뢰도 판단 기준: 상위 2개 모델 간 점수 차가 이보다 작으면 "모델 불확실"
MODEL_GAP_THRESHOLD = 0.05

# COCO 클래스 중 가방류: 24=backpack, 26=handbag, 28=suitcase
BAG_CLASSES = {24, 26, 28}

# label_map의 한국어/혼용 브랜드명 → CLIP 임베딩 DB의 브랜드명 매핑
BRAND_KO_TO_EN: dict[str, str] = {
    "샤넬":        "Chanel",
    "루이비통":    "Louis Vuitton",
    "구찌":        "Gucci",
    "에르메스":    "Hermès",
    "셀린느":      "Celine",
    "프라다":      "Prada",
    "디올":        "Dior",
    "보테가 베네타": "Bottega Veneta",
    "SaintLaurent": "Saint Laurent",
    "지방시":      "Givenchy",
    "발렌시아가":  "Balenciaga",
    "나이키":      "Nike",
    "아디다스":    "Adidas",
    "뉴발란스":    "New Balance",
    "반스":        "Vans",
    "닥터마틴":    "Dr. Martens",
    "노스페이스":  "The North Face",
    "보스":        "Boss",
    "레이번":      "Ray-Ban",
    "오클리":      "Oakley",
    "톰포드":      "Tom Ford",
    "젠틀몬스터":  "Gentle Monster",
    "티파니앤코":  "Tiffany & Co.",
    "유니클로":    "Uniqlo",
    "카카오프렌즈": "Kakao Friends",
    "라인프렌즈":  "Line Friends",
    "농심":        "Nongshim",
    # 영어 브랜드는 그대로 (Fendi, CJ, H&M 등)
}

# ── 한국어 모델명 → 영어 변환 ────────────────────────────────────
_KO_MODEL_WORD_MAP: list[tuple[str, str]] = sorted([
    # 복합어 (긴 것 우선)
    ("마더 오브 펄", "Mother of Pearl"),
    ("라이트 블루", "Light Blue"),
    ("라이트 그레이", "Light Gray"),
    ("도브 그레이", "Dove Gray"),
    ("멀티 컬러", "Multi-Color"),
    ("내추럴 컬러", "Natural Color"),
    ("스몰 사이즈", "Small Size"),
    ("재활용 플라스틱", "Recycled Plastic"),
    ("벨트 백", "Belt Bag"),
    ("보스턴 백", "Boston Bag"),
    ("미니 백", "Mini Bag"),
    ("쇼퍼백", "Shopper Bag"),
    # 단색
    ("그레이", "Gray"), ("블랙", "Black"), ("화이트", "White"), ("레드", "Red"),
    ("블루", "Blue"), ("그린", "Green"), ("베이지", "Beige"), ("버건디", "Burgundy"),
    ("라일락", "Lilac"), ("골드", "Gold"), ("오렌지", "Orange"), ("브라운", "Brown"),
    ("내추럴", "Natural"),
    # 소재
    ("가죽", "Leather"), ("나파", "Nappa"), ("패브릭", "Fabric"),
    ("나일론", "Nylon"), ("캔버스", "Canvas"), ("스웨이드", "Suede"),
    ("플란넬", "Flannel"), ("라피아", "Raffia"), ("스트로", "Straw"),
    ("메쉬", "Mesh"), ("플렉시글라스", "Plexiglass"),
    # 복합 백 종류 (긴 것 우선)
    ("탑핸들백", "Top Handle Bag"), ("숄더백", "Shoulder Bag"),
    ("크로스바디백", "Crossbody Bag"), ("카드 지갑", "Card Wallet"),
    # 백 종류
    ("핸드백", "Handbag"), ("미니백", "Mini Bag"), ("쇼퍼", "Shopper"),
    ("보스턴", "Boston"), ("파우치", "Pouch"), ("백팩", "Backpack"),
    ("포셰트", "Pochette"), ("클러치", "Clutch"), ("지갑", "Wallet"),
    ("버킷백", "Bucket Bag"), ("가방", "Bag"), ("백", "Bag"),
    # 수식어/크기
    ("컬러의", "Color"), ("컬러", "Color"), ("소재", ""), ("및", "and"),
    ("의", ""), ("스몰", "Small"), ("사이즈", "Size"),
    ("미디엄", "Medium"), ("라지", "Large"), ("맥시", "Maxi"),
    ("리미티드", "Limited"), ("캡슐", "Capsule"), ("컬렉션", "Collection"),
    ("모티프", "Motif"), ("프린트", "Print"), ("자수", "Embroidery"),
    ("장식", "Embellished"), ("인터레이스", "Interlaced"),
    ("셀러리아", "Selleria"), ("시그니처", "Signature"),
    ("재활용", "Recycled"), ("인서트", "Insert"),
    # 소재 추가
    ("사피아노", "Saffiano"), ("트위드", "Tweed"), ("퀼팅", "Quilted"),
    ("클래식", "Classic"), ("체인", "Chain"),
    # 부위/구조
    ("탑핸들", "Top Handle"), ("숄더", "Shoulder"), ("핸들", "Handle"),
    ("크로스바디", "Crossbody"), ("홀더", "Holder"), ("카드", "Card"),
    # 기타 카테고리
    ("선글라스", "Sunglasses"), ("스니커즈", "Sneakers"), ("운동화", "Sneakers"),
    ("플랩", "Flap"), ("버킷", "Bucket"), ("트래블", "Travel"),
], key=lambda x: -len(x[0]))  # 긴 것 우선


def _translate_model_name(name: str) -> str:
    """한국어/혼용 모델명을 영어로 변환. 이미 영어면 그대로 반환."""
    import re
    if not name:
        return name
    # ASCII 문자만 있으면 변환 불필요
    if all(ord(c) < 128 for c in name):
        return name
    result = name
    for ko, en in _KO_MODEL_WORD_MAP:
        replacement = f" {en} " if en else " "
        result = result.replace(ko, replacement)
    # 공백 정리
    result = re.sub(r'\s+', ' ', result).strip()
    return result


# ── 전역 모델/임베딩 캐시 ────────────────────────────────────────
_clip_model              = None
_clip_processor          = None
_yolo_detector           = None   # yolov8n.pt (bbox 탐지 전용)
_embeddings              = None   # shape: (N, 512)
_labels                  = None   # list of {"brand": ..., "model": ...}
_brand_text_embeddings   = None   # {brand_ko: np.ndarray} CLIP 텍스트 임베딩 캐시


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
    "Celine": ["Triomphe", "C Bag", "Cabas", "Besace", "Ava"],
}


# ── YOLOv8 베이스 모델 로드 (bbox 탐지 전용) ─────────────────────
async def _load_yolo_detector():
    global _yolo_detector
    if _yolo_detector is not None:
        return True
    if not YOLO_BASE_PATH.exists():
        print(f"[ReCheck] yolov8n.pt 없음: {YOLO_BASE_PATH}")
        return False
    try:
        from ultralytics import YOLO
        loop = asyncio.get_event_loop()
        _yolo_detector = await loop.run_in_executor(None, lambda: YOLO(str(YOLO_BASE_PATH)))
        print("[ReCheck] YOLOv8 베이스 모델 로드 완료 (bbox 탐지용)")
        return True
    except Exception as e:
        print(f"[ReCheck] YOLO 로드 실패: {e}")
        return False


async def _detect_bag_bbox(image: PILImage.Image) -> Optional[dict]:
    """
    yolov8n(COCO)으로 가방류(handbag/backpack/suitcase) 탐지 후
    가장 신뢰도 높은 박스를 normalized {x1,y1,x2,y2}로 반환.
    탐지 못하면 None 반환 → 전체 이미지 bbox 사용.
    """
    if _yolo_detector is None:
        return None

    loop = asyncio.get_event_loop()

    def _infer():
        results = _yolo_detector(image, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        best_box = None
        best_conf = -1.0

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf   = float(boxes.conf[i])
            if cls_id in BAG_CLASSES and conf > best_conf:
                best_conf = conf
                best_box  = boxes.xyxyn[i].tolist()  # normalized [x1,y1,x2,y2]

        if best_box is None:
            return None

        x1, y1, x2, y2 = best_box
        # 경계 보정
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        return {"x1": round(x1, 4), "y1": round(y1, 4),
                "x2": round(x2, 4), "y2": round(y2, 4)}

    return await loop.run_in_executor(None, _infer)


# ── CLIP 텍스트 기반 브랜드 분류 ──────────────────────────────────
async def _build_brand_text_embeddings() -> bool:
    """
    임베딩 DB의 브랜드별 이미지 임베딩 평균(prototype)을 생성해 캐시.
    텍스트 기반보다 정확한 이미지-이미지 브랜드 분류에 사용.
    """
    global _brand_text_embeddings
    if _brand_text_embeddings is not None:
        return True
    if _embeddings is None or _labels is None:
        return False

    unique_brands = list({label["brand"] for label in _labels})
    prototypes = {}

    for brand in unique_brands:
        mask = np.array([label["brand"] == brand for label in _labels])
        if mask.sum() == 0:
            continue
        proto = _embeddings[mask].mean(axis=0)   # 브랜드 평균 임베딩
        norm = np.linalg.norm(proto)
        if norm > 0:
            proto = proto / norm                 # L2 정규화
        prototypes[brand] = proto

    _brand_text_embeddings = prototypes
    print(f"[ReCheck] 브랜드 prototype 생성 완료: {len(prototypes)}개 브랜드")
    return True


def _classify_brand_by_text(query_emb: np.ndarray) -> Optional[str]:
    """
    CLIP 이미지 임베딩과 텍스트 브랜드 임베딩의 코사인 유사도로 브랜드 분류.
    임베딩 DB에 존재하는 브랜드 중 가장 유사한 브랜드명(원본 한국어/영문)을 반환.
    """
    if _brand_text_embeddings is None:
        return None

    best_brand = None
    best_score = -1.0

    for brand_ko, text_emb in _brand_text_embeddings.items():
        score = float(np.dot(query_emb, text_emb))
        if score > best_score:
            best_score = score
            best_brand = brand_ko

    # 상위 5개 브랜드 점수 출력 (진단용)
    sorted_brands = sorted(_brand_text_embeddings.items(),
                           key=lambda x: float(np.dot(query_emb, x[1])), reverse=True)
    print("[ReCheck] 브랜드 유사도 상위 5:")
    for b, emb in sorted_brands[:5]:
        print(f"   {b}: {float(np.dot(query_emb, emb)):.4f}")
    return best_brand


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
CORRECTIONS_PATH = OUTPUT_DIR / "user_corrections.npz"

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

        # 사용자 수정 임베딩 병합 (영구 반영)
        if CORRECTIONS_PATH.exists():
            corr = np.load(CORRECTIONS_PATH, allow_pickle=True)
            _embeddings = np.vstack([_embeddings, corr["embeddings"]])
            _labels.extend(corr["labels"].tolist())
            print(f"[ReCheck] 사용자 수정 임베딩 {len(corr['labels'])}개 추가 로드")

        return True
    except Exception as e:
        print(f"[ReCheck] 임베딩 DB 로드 실패: {e}")
        return False


# ── 사용자 수정 → 실시간 + 영구 임베딩 추가 ─────────────────────
async def add_embedding(image_b64: str, brand: str, model_name: str) -> bool:
    """
    사용자 확인/수정 데이터를 임베딩 DB에 추가.
    - 메모리(_embeddings, _labels): 즉시 반영
    - user_corrections.npz: 서버 재시작 후에도 영구 반영
    """
    global _embeddings, _labels, _brand_text_embeddings

    if _embeddings is None or _clip_model is None:
        return False

    try:
        import base64 as _b64
        img_bytes = _b64.b64decode(image_b64)
        image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")

        emb = await _get_image_embedding(image)
        if emb is None:
            return False

        new_emb = emb.reshape(1, -1)

        # ① 메모리에 즉시 추가
        _embeddings = np.vstack([_embeddings, new_emb])
        _labels.append({"brand": brand, "model": model_name})

        # ② user_corrections.npz에 영구 저장
        if CORRECTIONS_PATH.exists():
            existing = np.load(CORRECTIONS_PATH, allow_pickle=True)
            corr_embs   = np.vstack([existing["embeddings"], new_emb])
            corr_labels = existing["labels"].tolist() + [{"brand": brand, "model": model_name}]
        else:
            corr_embs   = new_emb
            corr_labels = [{"brand": brand, "model": model_name}]

        np.savez(CORRECTIONS_PATH, embeddings=corr_embs, labels=np.array(corr_labels))

        # ③ 브랜드 프로토타입 재생성
        _brand_text_embeddings = None
        await _build_brand_text_embeddings()

        print(f"[ReCheck] 임베딩 추가: {brand} / {model_name}")
        return True

    except Exception as e:
        print(f"[ReCheck] 임베딩 추가 실패: {e}")
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
            # transformers 버전에 따라 tensor 또는 output 객체 반환
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            # L2 정규화
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy()

    return await loop.run_in_executor(None, _infer)


# ── 특정 브랜드 내 모델 검색 ─────────────────────────────────────
def _search_model_in_brand(query_emb: np.ndarray, brand: str, top_k: int = 3) -> tuple[list[dict], bool]:
    """
    임베딩 DB에서 지정 브랜드에 속하는 항목만 필터링 후
    유사도 상위 top_k 모델 반환.
    brand와 일치하는 항목이 없으면 전체 DB에서 검색(fallback).
    """
    # 브랜드 필터 마스크
    mask = np.array([
        label["brand"].lower() == brand.lower()
        for label in _labels
    ])

    if mask.sum() == 0:
        print(f"[ReCheck] '{brand}' 임베딩 없음 → 전체 DB fallback")
        return _search_similar(query_emb, top_k)

    filtered_emb    = _embeddings[mask]          # (M, 512)
    filtered_labels = [l for l, m in zip(_labels, mask) if m]

    similarities = filtered_emb @ query_emb      # (M,)
    all_sorted   = np.argsort(similarities)[::-1]

    seen_models: dict[str, dict] = {}
    for idx in all_sorted:
        label = filtered_labels[idx]
        model = label["model"]
        if model not in seen_models:
            seen_models[model] = {
                "brand":      label["brand"],
                "model_name": model,
                "score":      round(float(similarities[idx]), 4),
            }
        if len(seen_models) >= top_k:
            break

    top3 = sorted(seen_models.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    scores = [item["score"] for item in top3]
    model_confident = (
        len(scores) < 2
        or (scores[0] - scores[1]) > MODEL_GAP_THRESHOLD
        or scores[0] >= 0.95
    )

    return top3, model_confident


# ── 코사인 유사도 검색 (브랜드 미지정 fallback용) ────────────────
def _search_similar(query_emb: np.ndarray, top_k: int = 3) -> tuple[list[dict], bool]:
    """
    임베딩 DB에서 유사도 상위 top_k (브랜드, 모델) 쌍 반환.
    같은 브랜드의 다른 모델도 각각 순위에 오를 수 있음
    (ex. 샤넬 Classic Flap 1위, 샤넬 Boy Bag 2위, 샤넬 19 Bag 3위).
    Returns:
        (top3, model_confident)
        model_confident: 1·2위 점수 차가 MODEL_GAP_THRESHOLD 이상이면 True
    """
    similarities = _embeddings @ query_emb  # (N,) 코사인 유사도
    all_sorted = np.argsort(similarities)[::-1]

    # ① (브랜드, 모델) 쌍 기준 상위 top_k 수집
    seen_pairs: dict[tuple, dict] = {}
    for idx in all_sorted:
        label = _labels[idx]
        pair  = (label["brand"], label["model"])
        if pair not in seen_pairs:
            seen_pairs[pair] = {
                "brand":      label["brand"],
                "model_name": label["model"],
                "score":      round(float(similarities[idx]), 4),
            }
        if len(seen_pairs) >= top_k:
            break

    top3 = sorted(seen_pairs.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    # ② 1·2위 점수 차로 모델 신뢰도 판단
    scores = [item["score"] for item in top3]
    model_confident = (
        len(scores) < 2
        or (scores[0] - scores[1]) > MODEL_GAP_THRESHOLD
        or scores[0] >= 0.95
    )

    return top3, model_confident


# ── Mock 응답 ────────────────────────────────────────────────────
def _mock_result(image_bytes: bytes) -> dict:  # noqa: ARG001
    import random
    brand = random.choice(list(BRAND_CATALOG.keys()))
    model = random.choice(BRAND_CATALOG[brand])
    confidence = round(random.uniform(0.72, 0.96), 3)
    # 30% 확률로 브랜드만 인식·모델명 불확실 케이스 시뮬레이션
    model_confident = random.random() > 0.3

    return {
        "detected": True,
        "bbox": {"x1": 0.12, "y1": 0.08, "x2": 0.88, "y2": 0.92},
        "prediction": {
            "brand": brand,
            "model_name": model if model_confident else None,
            "confidence": confidence,
            "model_confident": model_confident,
            "top3": [
                {"brand": brand, "model_name": model, "score": confidence},
                {"brand": "Chanel", "model_name": "Classic Flap", "score": round(confidence - 0.15, 3)},
                {"brand": "Louis Vuitton", "model_name": "Neverfull", "score": round(confidence - 0.28, 3)},
            ]
        },
        "mode": "mock",
        "message": (
            f"브랜드 '{brand}'는 확인했지만 모델명이 확실하지 않습니다. 직접 입력해 주세요."
            if not model_confident else
            f"AI가 '{brand} {model}'으로 추측했습니다. (신뢰도 {int(confidence*100)}%)"
        )
    }


# ── 메인 파이프라인 ──────────────────────────────────────────────
async def detect_and_classify(image_bytes: bytes) -> dict:
    # Mock 모드
    if _USE_MOCK:
        return _mock_result(image_bytes)

    # 이미지 열기 (YOLO·CLIP 공용)
    image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

    # ── ① YOLO base: 가방 bbox 탐지 ─────────────────────────────
    await _load_yolo_detector()
    bbox = await _detect_bag_bbox(image)
    if bbox is None:
        print("[ReCheck] YOLO: handbag 탐지 실패 → 전체 이미지 사용")
        bbox = {"x1": 0.05, "y1": 0.05, "x2": 0.95, "y2": 0.95}
    else:
        print(f"[ReCheck] YOLO bbox: {bbox}")

    # ── ② CLIP: 브랜드 + 모델 분류 ──────────────────────────────
    clip_ok = await _load_clip()
    if not clip_ok:
        return _mock_result(image_bytes)

    if _embeddings is None:
        if not _load_embeddings():
            return _mock_result(image_bytes)

    # 브랜드 텍스트 임베딩 준비 (최초 1회 생성)
    await _build_brand_text_embeddings()

    query_emb = await _get_image_embedding(image)
    if query_emb is None:
        return _mock_result(image_bytes)

    # 브랜드 분류 (CLIP 이미지↔텍스트 유사도)
    detected_brand = _classify_brand_by_text(query_emb)

    if detected_brand:
        # 해당 브랜드 내에서만 모델 검색
        top3, model_confident = _search_model_in_brand(query_emb, detected_brand, top_k=3)
        brand = BRAND_KO_TO_EN.get(detected_brand, detected_brand)
    else:
        # fallback: 전체 DB 검색
        top3, model_confident = _search_similar(query_emb, top_k=3)
        raw_brand = top3[0]["brand"] if top3 else None
        brand = BRAND_KO_TO_EN.get(raw_brand, raw_brand) if raw_brand else None

    if not top3:
        return {
            "detected": False,
            "bbox": bbox,
            "prediction": None,
            "mode": "real",
            "message": "브랜드를 인식하지 못했습니다. 직접 입력해 주세요."
        }

    # 브랜드명 + 모델명 영어 변환
    for item in top3:
        item["brand"] = BRAND_KO_TO_EN.get(item["brand"], item["brand"])
        item["model_name"] = _translate_model_name(item["model_name"])

    best = top3[0]
    confidence = best["score"]

    return {
        "detected": True,
        "bbox": bbox,
        "prediction": {
            "brand": brand,
            "model_name": best["model_name"] if model_confident else None,
            "confidence": confidence,
            "model_confident": model_confident,
            "top3": top3,
        },
        "mode": "real",
        "message": (
            f"브랜드 '{brand}'는 확인했지만 모델명이 확실하지 않습니다. 직접 입력해 주세요."
            if not model_confident else
            f"AI가 '{brand} {best['model_name']}'으로 추측했습니다. (신뢰도 {int(confidence * 100)}%)"
        )
    }