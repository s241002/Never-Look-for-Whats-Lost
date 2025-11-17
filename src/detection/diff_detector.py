import cv2
import numpy as np


class DiffDetector:
def __init__(self, thresh=25, blur_ksize=(5,5)):
self.thresh = thresh
self.blur_ksize = blur_ksize


def compute_mask(self, imgA, imgB):
"""グレースケールで差分を取り、閾値処理でマスクを返す
imgA, imgB: BGR numpy arrays
returns: mask (binary uint8)
"""
grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
# ブラーでノイズ軽減
grayA = cv2.GaussianBlur(grayA, self.blur_ksize, 0)
grayB = cv2.GaussianBlur(grayB, self.blur_ksize, 0)
diff = cv2.absdiff(grayA, grayB)
_, mask = cv2.threshold(diff, self.thresh, 255, cv2.THRESH_BINARY)
# 輪郭を埋めてノイズ除去
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
return mask


def bbox_overlap_ratio(self, bbox, mask):
"""bbox: [x1,y1,x2,y2], mask: binary image. bbox領域におけるマスク占有率を返す"""
x1,y1,x2,y2 = [int(v) for v in bbox]
h, w = mask.shape
x1 = max(0, min(x1, w-1))
x2 = max(0, min(x2, w-1))
y1 = max(0, min(y1, h-1))
y2 = max(0, min(y2, h-1))
if x2 <= x1 or y2 <= y1:
return 0.0
region = mask[y1:y2, x1:x2]
if region.size == 0:
return 0.0
return float(np.count_nonzero(region)) / float(region.size)