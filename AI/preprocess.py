"""
preprocess.py
-------------
Labelme JSON + CSV → YOLOv8 학습 포맷 변환 스크립트

입력 구조 (같은 폴더):
  dataset/
    ├── [가방] Fendi_Brown fabric belt bag_1.jpg
    ├── [가방] Fendi_Brown fabric belt bag_1.json
    ├── [가방] Fendi_Brown fabric belt bag_2.jpg
    ├── [가방] Fendi_Brown fabric belt bag_2.json
    ├── ...
    └── products.csv   ← name 컬럼 보유

출력 구조:
  output/
    ├── images/
    │   ├── train/   (80%)
    │   └── val/     (20%)
    ├── labels/
    │   ├── train/   (YOLO .txt 형식)
    │   └── val/
    ├── dataset.yaml          ← YOLOv8 학습 설정 파일
    ├── clip_train.csv        ← CLIP 분류 학습용 (image_path, brand, model)
    └── label_map.json        ← 클래스 ID ↔ 브랜드 매핑
"""
import json
import shutil
import random
import re
from pathlib import Path

import pandas as pd

# ── 설정 ────────────────────────────────────────────────────────
DATASET_DIR = Path("../file")      # 이미지 + JSON이 있는 폴더
CSV_PATH    = Path("../file_list.csv")
OUTPUT_DIR  = Path("output")
TRAIN_RATIO = 0.8
SEED        = 42

random.seed(SEED)


# ── 유틸 ────────────────────────────────────────────────────────
def parse_name(name: str) -> tuple[str, str, str]:
    """
    "[가방] Fendi_Brown fabric belt bag" →
    category="가방", brand="Fendi", model="Brown fabric belt bag"
    """
    name = name.strip()

    # 카테고리 추출: [가방], [신발] 등
    cat_match = re.match(r'\[(.+?)\]', name)
    category = cat_match.group(1) if cat_match else "기타"

    # 브랜드_모델명 부분 추출
    rest = re.sub(r'\[.+?\]\s*', '', name).strip()

    # 첫 번째 _ 기준으로 브랜드 / 모델 분리
    if '_' in rest:
        brand, model = rest.split('_', 1)
    else:
        brand, model = rest, "Unknown"

    return category.strip(), brand.strip(), model.strip()


def labelme_to_yolo(points, img_w, img_h) -> str:
    """
    Labelme rectangle points [[x1,y1],[x2,y2]] →
    YOLO 형식: "class_id cx cy w h" (0~1 정규화)
    """
    x1, y1 = points[0]
    x2, y2 = points[1]

    # 좌상단/우하단 보정 (포인트 순서가 뒤집힐 수 있음)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h

    # 범위 클램핑 (0~1)
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.001, min(1.0, w))
    h  = max(0.001, min(1.0, h))

    return cx, cy, w, h


# ── 메인 파이프라인 ──────────────────────────────────────────────
def main():
    # 출력 폴더 생성
    for split in ('train', 'val'):
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # ── 1. CSV에서 브랜드 목록 및 클래스 ID 매핑 생성 ────────────
    print("📄 CSV 파싱 중...")
    df = pd.read_csv(CSV_PATH)
    name_col = 'name'

    brands = set()
    for name in df[name_col].dropna():
        _, brand, _ = parse_name(str(name))
        brands.add(brand)

    brand_list = sorted(brands)
    brand_to_id = {b: i for i, b in enumerate(brand_list)}
    print(f"   → 총 브랜드 수: {len(brand_list)}개")

    # label_map.json 저장
    label_map = {str(i): b for i, b in enumerate(brand_list)}
    with open(OUTPUT_DIR / 'label_map.json', 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"   → label_map.json 저장 완료")

    # ── 2. JSON 파일 순회 및 변환 ─────────────────────────────────
    print("\n🔍 JSON 파일 파싱 및 YOLO 변환 중...")

    json_files = list(DATASET_DIR.glob("*.json"))
    print(f"   → JSON 파일 총 {len(json_files)}개 발견")

    # train/val 분할
    random.shuffle(json_files)
    split_idx = int(len(json_files) * TRAIN_RATIO)
    splits = {
        'train': json_files[:split_idx],
        'val':   json_files[split_idx:],
    }

    clip_rows = []   # CLIP 학습용 데이터
    stats = {'train': 0, 'val': 0, 'skip': 0}

    for split, files in splits.items():
        for json_path in files:
            try:
                with open(json_path, encoding='utf-8') as f:
                    data = json.load(f)

                image_filename = data.get('imagePath', '')
                img_w = data.get('imageWidth', 300)
                img_h = data.get('imageHeight', 300)
                shapes = data.get('shapes', [])

                # 이미지 파일 확인
                img_src = DATASET_DIR / image_filename
                if not img_src.exists():
                    # 확장자 다를 수 있음 → 유사 파일 탐색
                    stem = Path(image_filename).stem
                    candidates = list(DATASET_DIR.glob(f"{stem}.*"))
                    img_src = candidates[0] if candidates else None

                if img_src is None or not img_src.exists():
                    stats['skip'] += 1
                    continue

                # 이미지 이름에서 브랜드/모델 파싱
                name_str = Path(image_filename).stem
                # 파일명 끝의 _숫자 제거 (예: _1, _2)
                name_str = re.sub(r'_\d+$', '', name_str)
                category, brand, model = parse_name(name_str)

                if brand not in brand_to_id:
                    stats['skip'] += 1
                    continue

                class_id = brand_to_id[brand]

                # YOLO .txt 라벨 생성
                yolo_lines = []
                for shape in shapes:
                    if shape.get('shape_type') != 'rectangle':
                        continue
                    pts = shape['points']
                    if len(pts) < 2:
                        continue
                    cx, cy, w, h = labelme_to_yolo(pts, img_w, img_h)
                    yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                if not yolo_lines:
                    stats['skip'] += 1
                    continue

                # 파일 복사 및 저장
                stem = img_src.stem
                suffix = img_src.suffix

                dst_img = OUTPUT_DIR / 'images' / split / f"{stem}{suffix}"
                dst_lbl = OUTPUT_DIR / 'labels' / split / f"{stem}.txt"

                shutil.copy2(img_src, dst_img)
                with open(dst_lbl, 'w') as f:
                    f.write('\n'.join(yolo_lines))

                # CLIP 학습용 row 추가
                clip_rows.append({
                    'image_path': str(dst_img),
                    'category':   category,
                    'brand':      brand,
                    'model':      model,
                    'class_id':   class_id,
                    'split':      split,
                })

                stats[split] += 1

            except Exception as e:
                print(f"   ⚠️  {json_path.name} 처리 실패: {e}")
                stats['skip'] += 1

    # ── 3. dataset.yaml 생성 (YOLOv8 학습 설정) ──────────────────
    yaml_content = f"""# ReCheck YOLOv8 Dataset Config
path: {OUTPUT_DIR.resolve()}
train: images/train
val:   images/val

nc: {len(brand_list)}
names:
"""
    for i, b in enumerate(brand_list):
        yaml_content += f"  {i}: {b}\n"

    with open(OUTPUT_DIR / 'dataset.yaml', 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    # ── 4. CLIP 학습용 CSV 저장 ────────────────────────────────────
    clip_df = pd.DataFrame(clip_rows)
    clip_df.to_csv(OUTPUT_DIR / 'clip_train.csv', index=False, encoding='utf-8-sig')

    # ── 5. 완료 리포트 ─────────────────────────────────────────────
    print(f"""
✅ 전처리 완료!
   Train: {stats['train']}개
   Val:   {stats['val']}개
   Skip:  {stats['skip']}개 (이미지 없음 또는 bbox 없음)

📁 출력 폴더: {OUTPUT_DIR.resolve()}
   ├── images/train & val
   ├── labels/train & val
   ├── dataset.yaml    ← YOLOv8 학습에 사용
   ├── clip_train.csv  ← CLIP 분류 학습에 사용
   └── label_map.json  ← 클래스 ID ↔ 브랜드 매핑
""")


if __name__ == '__main__':
    main()