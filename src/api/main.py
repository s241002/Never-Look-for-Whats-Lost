from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import cv2

from detection.yolov8_detector import YOLOv8Detector
from detection.diff import DiffDetector
from detection.movement import match_objects, filter_objects_by_regions
from detection.utils import draw_movement

app = FastAPI()

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

detector = YOLOv8Detector()
diff_detector = DiffDetector()

@app.get("/detect")
def detect():
    baseline_path = DATA_DIR / "baseline.jpg"
    prev_path = DATA_DIR / "prev.jpg"
    latest_path = DATA_DIR / "latest.jpg"
    buffer_dir = DATA_DIR / "buffer"
    buffer_imgs = sorted(buffer_dir.glob("*.jpg"))
    buffer_img = None
    if buffer_imgs:
        buffer_img = cv2.imread(str(buffer_imgs[-1]))

    imgs = []
    for p in [baseline_path, prev_path, latest_path]:
        img = cv2.imread(str(p))
        if img is None:
            return JSONResponse(content={"error": f"File not found: {p}"}, status_code=404)
        imgs.append(img)
    if buffer_img is not None:
        imgs.append(buffer_img)

    # 差分検出
    diff_mask = diff_detector.compute_combined_mask(imgs)
    movement_regions = diff_detector.get_movement_regions(diff_mask)

    # YOLO検出
    objects_prev = detector.detect(str(prev_path))
    objects_latest = detector.detect(str(latest_path))

    # 差分領域でフィルタリング
    objects_prev_filtered = filter_objects_by_regions(objects_prev, movement_regions)
    objects_latest_filtered = filter_objects_by_regions(objects_latest, movement_regions)

    # マッチングして移動計算
    movements = match_objects(objects_prev_filtered, objects_latest_filtered)

    # 可視化画像作成
    vis_img = imgs[-2].copy()  # latest.jpgを基に
    draw_movement(vis_img, movements)

    # 画像保存
    output_img_path = OUTPUT_DIR / "movement.jpg"
    cv2.imwrite(str(output_img_path), vis_img)

    return {
        "movement_count": len(movements),
        "movements": movements,
        "image_path": str(output_img_path)
    }