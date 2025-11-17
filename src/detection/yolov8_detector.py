from ultralytics import YOLO
import numpy as np


class YOLOv8Detector:
def __init__(self, model_path="yolov8n.pt", device=None):
"""device: None (autoselect) or 'cpu' / '0'など"""
self.model = YOLO(model_path)


def detect_image(self, image):
"""
image: BGR numpy array (cv2.imread の出力)
returns: list of dicts: {"bbox":[x1,y1,x2,y2], "class":int, "conf":float, "center":[cx,cy]}
"""
results = self.model(image)[0]
objs = []
for box in results.boxes:
xyxy = box.xyxy[0].cpu().numpy().astype(float)
cls = int(box.cls[0].cpu().numpy())
conf = float(box.conf[0].cpu().numpy())
x1, y1, x2, y2 = xyxy.tolist()
cx = (x1 + x2) / 2.0
cy = (y1 + y2) / 2.0
objs.append({
"bbox": [x1, y1, x2, y2],
"class": cls,
"conf": conf,
"center": [cx, cy]
})
return objs