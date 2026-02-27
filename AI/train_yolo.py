"""
train_yolo.py
-------------
전처리된 데이터셋으로 YOLOv8 파인튜닝

실행:
  python train_yolo.py
"""

from pathlib import Path
from ultralytics import YOLO

YAML_PATH  = Path("output/dataset.yaml")
OUTPUT_DIR = Path("output/yolo_model")
BASE_MODEL = "yolov8n.pt"   # nano(빠름) | yolov8s.pt(균형) | yolov8m.pt(정확)
EPOCHS     = 50
IMG_SIZE   = 640
BATCH      = 16


def main():
    print(f"🚀 YOLOv8 파인튜닝 시작 (base: {BASE_MODEL})")
    model = YOLO(BASE_MODEL)

    results = model.train(
        data    = str(YAML_PATH),
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH,
        project = str(OUTPUT_DIR),
        name    = "recheck_v1",
        patience= 10,          # 10 epoch 개선 없으면 조기 종료
        cache   = True,        # 이미지 캐시 (RAM 여유 있을 때)
        exist_ok= True,
    )

    best_path = OUTPUT_DIR / "recheck_v1" / "weights" / "best.pt"
    print(f"\n✅ 학습 완료! 최적 모델: {best_path}")
    print("   → detector.py의 YOLO() 경로를 이 경로로 교체하세요.")


if __name__ == '__main__':
    main()