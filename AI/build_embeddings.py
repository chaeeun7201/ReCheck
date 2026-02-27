"""
build_embeddings.py
-------------------
전처리된 이미지로 CLIP 임베딩 DB 생성 (1회만 실행)

실행:
  python build_embeddings.py

출력:
  output/clip_embeddings.npz  ← detector.py가 이걸 로드해서 사용
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

BASE_DIR       = Path(__file__).parent
OUTPUT_DIR     = BASE_DIR / "output"
CLIP_CSV       = OUTPUT_DIR / "clip_train.csv"
EMBEDDING_PATH = OUTPUT_DIR / "clip_embeddings.npz"
BATCH_SIZE     = 16


def main():
    print("🔧 CLIP 임베딩 DB 생성 시작")

    # ── 1. CLIP 로드 ──────────────────────────────────────────────
    print("   CLIP 모델 로드 중...")
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"   디바이스: {device}")

    # ── 2. CSV 로드 ───────────────────────────────────────────────
    df = pd.read_csv(CLIP_CSV)
    print(f"   총 이미지: {len(df)}개")

    # ── 3. 배치 임베딩 추출 ───────────────────────────────────────
    all_embeddings = []
    all_labels     = []
    skip_count     = 0

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="임베딩 추출"):
        batch = df.iloc[i:i+BATCH_SIZE]
        images = []
        labels = []

        for _, row in batch.iterrows():
            img_path = Path(row["image_path"])
            if not img_path.exists():
                skip_count += 1
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                labels.append({"brand": row["brand"], "model": row["model"]})
            except Exception:
                skip_count += 1
                continue

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            features = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
            # L2 정규화
            features = features / features.norm(p=2, dim=-1, keepdim=True)

        all_embeddings.append(features.cpu().numpy())
        all_labels.extend(labels)

    # ── 4. 저장 ──────────────────────────────────────────────────
    embeddings_matrix = np.vstack(all_embeddings)  # (N, 512)
    labels_array      = np.array(all_labels)

    np.savez(
        EMBEDDING_PATH,
        embeddings=embeddings_matrix,
        labels=labels_array,
    )

    print(f"""
✅ 임베딩 DB 생성 완료!
   저장 경로: {EMBEDDING_PATH}
   총 임베딩: {len(all_labels)}개
   Skip:      {skip_count}개
   Shape:     {embeddings_matrix.shape}

→ 이제 백엔드 서버를 재시작하면 실제 AI 인식이 동작합니다!
""")


if __name__ == "__main__":
    main()