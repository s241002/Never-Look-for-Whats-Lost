import cv2
import time
from pathlib import Path
import shutil
from detection.diff import DiffDetector
import numpy as np

DATA_DIR = Path("data")
BUFFER_DIR = DATA_DIR / "buffer"
BUFFER_DIR.mkdir(parents=True, exist_ok=True)

BASE_IMG = DATA_DIR / "baseline.jpg"
PREV_IMG = DATA_DIR / "prev.jpg"
LATEST_IMG = DATA_DIR / "latest.jpg"

diff_detector = DiffDetector()

def is_significant_change(img1_path, img2_path, threshold=50):
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        return True  # 最初は必ずTrue
    mask = diff_detector.compute_diff_mask(img1, img2)
    changed_area = cv2.countNonZero(mask)
    print(f"changed_area={changed_area}")
    return changed_area > threshold

def capture_loop(interval=0.5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラを開けません")
        return

    frame_count = 0
    buffer_files = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 画像保存前にprev/latestを更新
        if LATEST_IMG.exists():
            shutil.copy(str(LATEST_IMG), PREV_IMG)

        cv2.imwrite(str(LATEST_IMG), frame)

        # 変化が小さい場合は次ループまで待機
        if PREV_IMG.exists():
            if not is_significant_change(PREV_IMG, LATEST_IMG):
                print("変化なし、待機中...")
                time.sleep(interval)
                continue

        # 変化あったらbufferに追加（1秒分＝2枚）
        buffer_path = BUFFER_DIR / f"frame_{int(time.time()*1000)}.jpg"
        cv2.imwrite(str(buffer_path), frame)
        buffer_files.append(buffer_path)
        if len(buffer_files) > 2:
            oldest = buffer_files.pop(0)
            oldest.unlink()

        frame_count += 1

        # 4枚+基準の画像揃ったら処理呼び出し（ここはAPIやメイン処理で制御）
        if len(buffer_files) == 2 and BASE_IMG.exists() and PREV_IMG.exists() and LATEST_IMG.exists():
            print("処理呼び出し可能状態")

        time.sleep(interval)

    cap.release()

if __name__ == "__main__":
    capture_loop(0.5)