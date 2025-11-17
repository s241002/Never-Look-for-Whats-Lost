import cv2
import time
import os

def capture_loop(save_dir="data/captures", interval=2):
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available")
        return

    # --- 最初の1枚を基準画像として保存 ---
    ret, frame = cap.read()
    if ret:
        baseline = os.path.join(save_dir, "baseline.jpg")
        cv2.imwrite(baseline, frame)
        print("基準画像を保存:", baseline)
    else:
        print("基準画像が取得できませんでした")
        return

    # 前フレームの名前を固定
    prev1 = os.path.join(save_dir, "prev1.jpg")
    prev2 = os.path.join(save_dir, "prev2.jpg")
    latest = os.path.join(save_dir, "latest.jpg")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 最新→前1→前2 の順にローテーション
        if os.path.exists(prev1):
            os.replace(prev1, prev2)
        if os.path.exists(latest):
            os.replace(latest, prev1)

        # 最新フレームを保存
        cv2.imwrite(latest, frame)
        print("保存 → 最新:", latest)

        time.sleep(interval)

    cap.release()
