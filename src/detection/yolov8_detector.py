from ultralytics import YOLO
from typing import List, Dict

class YOLOv8Detector:
    def __init__(self, model_path="yolov8n.pt", device=None):
        self.model = YOLO(model_path)
        if device:
            self.model.to(device)

    def detect(self, image_path: str) -> List[Dict]:
        results = self.model(image_path)[0]
        detected_objects = []
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            x1, y1, x2, y2 = xyxy.tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            class_name = self.model.names.get(cls_id, "unknown")
            detected_objects.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": conf,
                "center": [cx, cy]
            })
        return detected_objects