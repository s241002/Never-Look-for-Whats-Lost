from ultralytics import YOLO
from typing import List, Dict

class YOLOv8Detector:
    def __init__(self, model_path="yolov8n.pt", device="cpu"):
        self.device = device
        self.model = YOLO(model_path)

    def detect(self, image_path: str) -> List[Dict]:
        results = self.model.predict(
            source=image_path,
            device=self.device,
            imgsz=640,
            verbose=False
        )[0]
