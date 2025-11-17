import math


def iou(boxA, boxB):
xA = max(boxA[0], boxB[0])
yA = max(boxA[1], boxB[1])
xB = min(boxA[2], boxB[2])
yB = min(boxA[3], boxB[3])
interW = max(0, xB - xA)
interH = max(0, yB - yA)
interArea = interW * interH
areaA = max(0.0, (boxA[2]-boxA[0]) * (boxA[3]-boxA[1]))
areaB = max(0.0, (boxB[2]-boxB[0]) * (boxB[3]-boxB[1]))
denom = (areaA + areaB - interArea)
if denom <= 0:
return 0.0
return interArea / denom




def match_by_iou(prev_objs, cur_objs, iou_thresh=0.1):
"""
単純な貪欲マッチング：prev_objsの各要素に対してcur_objsとのIOUが最大のものを割当
returns: list of matches: {"prev":prev,"cur":cur,"iou":v}
"""
matches = []
used = set()
for p in prev_objs:
best_j = None
best_iou = 0.0
for j, c in enumerate(cur_objs):
if j in used:
continue
if p["class"] != c["class"]:
continue
i = iou(p["bbox"], c["bbox"])
if i > best_iou:
best_iou = i
best_j = j
if best_j is not None and best_iou >= iou_thresh:
used.add(best_j)
matches.append({"prev": p, "cur": cur_objs[best_j], "iou": best_iou})
return matches




def center_distance(a, b):
ax, ay = a["center"]
bx, by = b["center"]
return math.hypot(bx-ax, by-ay)




def match_by_distance(prev_objs, cur_objs, max_dist=100):
matches = []
used = set()
for p in prev_objs:
best_j = None
best_d = None
for j, c in enumerate(cur_objs):
if j in used:
continue
if p["class"] != c["class"]:
continue
d = center_distance(p, c)
if best_j is None or d < best_d:
best_j = j
best_d = d
if best_j is not None and best_d is not None and best_d <= max_dist:
used.add(best_j)
matches.append({"prev": p, "cur": cur_objs[best_j], "dist": best_d})
return matches