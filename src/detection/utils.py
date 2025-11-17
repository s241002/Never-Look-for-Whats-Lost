from typing import List
import cv2

def draw_movement(img, matches: List[dict]) -> None:
    """
    移動を矢印で描画する簡単な可視化関数
    """
    for m in matches:
        x1_prev, y1_prev, x2_prev, y2_prev = map(int, m["bbox_prev"])
        x1_latest, y1_latest, x2_latest, y2_latest = map(int, m["bbox_latest"])
        cx_prev = int((x1_prev + x2_prev) / 2)
        cy_prev = int((y1_prev + y2_prev) / 2)
        cx_latest = int((x1_latest + x2_latest) / 2)
        cy_latest = int((y1_latest + y2_latest) / 2)

        cv2.rectangle(img, (x1_latest, y1_latest), (x2_latest, y2_latest), (0,255,0), 2)
        cv2.arrowedLine(img, (cx_prev, cy_prev), (cx_latest, cy_latest), (0,0,255), 2)
        cv2.putText(img, m["class_name"], (x1_latest, y1_latest - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)