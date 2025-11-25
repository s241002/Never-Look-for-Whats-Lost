def draw_movement(img: np.ndarray, matches: List[dict]) -> np.ndarray:
    out = img.copy()
    for m in matches:
        x1,y1,x2,y2 = map(int, m["bbox_latest"])
        px1,py1,px2,py2 = map(int, m["bbox_prev"])

        cx_prev = int((px1 + px2) / 2)
        cy_prev = int((py1 + py2) / 2)

        cx_latest = int((x1 + x2) / 2)
        cy_latest = int((y1 + y2) / 2)

        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.arrowedLine(out, (cx_prev,cy_prev), (cx_latest,cy_latest), (0,0,255), 2)
        cv2.putText(out, m["class_name"], (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return out