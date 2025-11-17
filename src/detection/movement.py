
from typing import List, Dict
import numpy as np

def iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def bbox_to_np(bbox):
    return np.array(bbox)

def is_bbox_overlap(bbox1, bbox2, iou_threshold=0.1):
    return iou(bbox1, bbox2) > iou_threshold

def filter_objects_by_regions(objects: List[Dict], regions: List[List[int]], iou_threshold=0.1) -> List[Dict]:
    filtered = []
    for obj in objects:
        for region in regions:
            if is_bbox_overlap(obj["bbox"], region, iou_threshold):
                filtered.append(obj)
                break
    return filtered

def match_objects(
    objects_prev: List[Dict],
    objects_latest: List[Dict],
    iou_threshold: float = 0.3
) -> List[Dict]:
    matches = []
    used_latest_idx = set()

    for prev_obj in objects_prev:
        best_iou = 0
        best_idx = -1
        for i, latest_obj in enumerate(objects_latest):
            if i in used_latest_idx:
                continue
            if prev_obj["class_id"] != latest_obj["class_id"]:
                continue
            iou_score = iou(prev_obj["bbox"], latest_obj["bbox"])
            if iou_score > best_iou:
                best_iou = iou_score
                best_idx = i
        if best_iou >= iou_threshold and best_idx >= 0:
            latest_obj = objects_latest[best_idx]
            used_latest_idx.add(best_idx)
            dx = latest_obj["center"][0] - prev_obj["center"][0]
            dy = latest_obj["center"][1] - prev_obj["center"][1]
            matches.append({
                "class_id": prev_obj["class_id"],
                "class_name": prev_obj["class_name"],
                "confidence_prev": prev_obj["confidence"],
                "confidence_latest": latest_obj["confidence"],
                "bbox_prev": prev_obj["bbox"],
                "bbox_latest": latest_obj["bbox"],
                "dx": dx,
                "dy": dy,
            })
    return matches