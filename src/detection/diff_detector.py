import cv2
import numpy as np
from typing import List, Tuple

class DiffDetector:
    def __init__(self, threshold=30, blur_kernel=(5,5)):
        self.threshold = threshold
        self.blur_kernel = blur_kernel

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        return blurred

    def compute_diff_mask(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        pre1 = self._preprocess(img1)
        pre2 = self._preprocess(img2)
        diff = cv2.absdiff(pre1, pre2)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def compute_combined_mask(self, images: List[np.ndarray]) -> np.ndarray:
        combined_mask = None
        for i in range(len(images) - 1):
            mask = self.compute_diff_mask(images[i], images[i+1])
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
        return combined_mask

    def get_movement_regions(self, mask: np.ndarray, min_area=500) -> List[Tuple[int,int,int,int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h >= min_area:
                regions.append((x, y, x+w, y+h))
        return regions
