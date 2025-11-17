from fastapi import FastAPI
from detection.yolov8_detector import YOLOv8Detector
from detection.matcher import match_objects
from detection.diff_detector import DiffDetector

app = FastAPI()

detector = YOLOv8Detector()
diff = DiffDetector()

@app.get("/detect")
def detect():
    base = "data/base.jpg"
    latest = "data/latest.jpg"

    base_obj = detector.detect(base)
    latest_obj = detector.detect(latest)

    movement = match_objects(base_obj, latest_obj)

    return {
        "movement": movement
    }
