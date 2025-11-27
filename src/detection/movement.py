import cv2
import numpy as np
from typing import List, Dict
from .diff_detector import DiffDetector
from .utils import draw_movement
from .yolov8_detector import YOLOv8Detector

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, (box1[2]-box1[0])) * max(0, (box1[3]-box1[1]))
    area2 = max(0, (box2[2]-box2[0])) * max(0, (box2[3]-box2[1]))

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0


class MovementDetector:
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.diff = DiffDetector()
        self.yolo = YOLOv8Detector()

    def detect_movement(self, prev_img_path, latest_img_path):
        print("★ detect_movement 呼び出し OK")

        prev = cv2.imread(str(prev_img_path))
        latest = cv2.imread(str(latest_img_path))

        if prev is None or latest is None:
            print("× 画像が読み込めない")
            return None

        # (1) 差分マスク
        mask = self.diff.compute_diff_mask(prev, latest)
        regions = self.diff.get_movement_regions(mask)
        print(f"差分領域数: {len(regions)}")

        # (2) YOLO 検出
        prev_boxes = self.yolo.detect(prev_img_path)
        latest_boxes = self.yolo.detect(latest_img_path)
        print(f"YOLO prev={len(prev_boxes)} latest={len(latest_boxes)}")

        # (3) マッチング
        matches = []
        for pb in prev_boxes:
            for lb in latest_boxes:
                if pb["class_id"] != lb["class_id"]:
                    continue

                if iou(pb["bbox"], lb["bbox"]) > self.iou_threshold:
                    matches.append({
                        "class_name": pb["class_name"],
                        "bbox_prev": pb["bbox"],
                        "bbox_latest": lb["bbox"]
                    })

        print(f"マッチ数: {len(matches)}")

        # (4) movement 描画
        output = draw_movement(latest, matches)

        print("◎ movement画像生成完了")
        return output
