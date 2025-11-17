from fastapi import FastAPI, HTTPException
from pathlib import Path
from detection.yolov8_detector import YOLOv8Detector
from detection.matcher import match_objects
from detection.diff_detector import DiffDetector

app = FastAPI()

detector = YOLOv8Detector()
diff_detector = DiffDetector()

# データフォルダの絶対パス
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

@app.get("/detect")
def detect():
    base_img = DATA_DIR / "base.jpg"
    latest_img = DATA_DIR / "latest.jpg"
    prev_img = DATA_DIR / "prev.jpg"
    prev2_img = DATA_DIR / "prev2.jpg"

    # ファイル存在チェック
    for img in [base_img, latest_img]:
        if not img.exists():
            raise HTTPException(status_code=400, detail=f"Image not found: {img}")

    # YOLO 推論
    base_objects = detector.detect(str(base_img))
    latest_objects = detector.detect(str(latest_img))

    # 移動量（マッチング）
    movement = match_objects(base_objects, latest_objects)

    # 差分画像の計算（オプション）
    diff_score = diff_detector.compare(str(base_img), str(latest_img))

    return {
        "movement": movement,
        "diff_score": diff_score,
        "status": "ok"
    }
