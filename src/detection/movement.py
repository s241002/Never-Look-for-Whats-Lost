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
