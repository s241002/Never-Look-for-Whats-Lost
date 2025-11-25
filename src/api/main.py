from fastapi import FastAPI, HTTPException
from pathlib import Path
import cv2

from detection.yolov8_detector import YOLOv8Detector
from detection.diff_detector import DiffDetector
from detection.movement import match_objects, filter_objects_by_regions
from detection.utils import draw_movement

app = FastAPI()

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

detector = YOLOv8Detector(device="cpu")
diff_detector = DiffDetector()

@app.get("/detect")
def detect_changes():
    try:
        baseline = cv2.imread(str(DATA_DIR / "baseline.jpg"))
        prev = cv2.imread(str(DATA_DIR / "prev.jpg"))
        latest = cv2.imread(str(DATA_DIR / "latest.jpg"))

        if baseline is None or prev is None or latest is None:
            raise HTTPException(404, "Missing input images")

        imgs = [baseline, prev, latest]

        # 差分領域
        diff_mask = diff_detector.compute_combined_mask(imgs)
        regions = diff_detector.get_movement_regions(diff_mask)

        # YOLO検出（高速化オプション）
        prev_objs = detector.detect(str(DATA_DIR / "prev.jpg"))
        latest_objs = detector.detect(str(DATA_DIR / "latest.jpg"))

        prev_filtered = filter_objects_by_regions(prev_objs, regions)
        latest_filtered = filter_objects_by_regions(latest_objs, regions)

        movements = match_objects(prev_filtered, latest_filtered)

        # 描画
        vis = draw_movement(latest, movements)
        OUTPUT_DIR.mkdir(exist_ok=True)
        out_path = OUTPUT_DIR / "movement.jpg"
        cv2.imwrite(str(out_path), vis)

        return {
            "movement_count": len(movements),
            "movements": movements,
            "result_image": str(out_path)
        }

    except Exception as e:
        raise HTTPException(500, f"Error: {e}")
