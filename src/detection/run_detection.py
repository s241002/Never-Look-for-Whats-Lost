import os
# bbox
cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
# arrow
cv2.arrowedLine(out, (cx_prev, cy_prev), (cx_cur, cy_cur), (0,0,255), 2, tipLength=0.2)
cv2.putText(out, f"{name}", (x1, max(10, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
return out




def main():
# ファイル検出
p_base = find_one(BASE_CANDS)
p_latest = find_one(LATEST_CANDS)
p_prev1 = find_one(PREV1_CANDS)
p_prev2 = find_one(PREV2_CANDS)


if p_base is None or p_latest is None:
raise FileNotFoundError("base or latest image not found in data/ (checked candidates)")


print("Using:")
print(" base:", p_base)
print(" latest:", p_latest)
print(" prev1:", p_prev1)
print(" prev2:", p_prev2)


img_base = load_img(p_base)
img_latest = load_img(p_latest)


detector = YOLOv8Detector()
diff = DiffDetector()


objs_base = detector.detect_image(img_base)
objs_latest = detector.detect_image(img_latest)


# マッチング（IoUベース）
matches = match_by_iou(objs_base, objs_latest, iou_thresh=0.05)


# 結果を出力用に整形
names = detector.model.names if hasattr(detector.model, 'names') else {}
movement = []
for m in matches:
prev = m["prev"]
cur = m["cur"]
dx = cur["center"][0] - prev["center"][0]
dy = cur["center"][1] - prev["center"][1]
movement.append({
"class": int(prev["class"]),
"label": names.get(prev["class"], str(prev["class"])),
"conf_prev": prev.get("conf", None),
"conf_cur": cur.get("conf", None),
"dx": dx,
"dy": dy,
"iou": m.get("iou", None)
})


# 可視化画像
vis = draw_results(img_latest, matches, names)
out_img = OUTPUT_DIR / "diff_result.jpg"
cv2.imwrite(str(out_img), vis)


# JSON 保存
out_json = OUTPUT_DIR / "movement.json"
with open(out_json, 'w', encoding='utf-8') as f:
json.dump({"movement": movement}, f, ensure_ascii=False, indent=2)


print("Saved:", out_img, out_json)


if __name__ == '__main__':
main()