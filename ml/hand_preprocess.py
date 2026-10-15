from __future__ import annotations

import logging
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import Config

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

class HandPreprocessor:
    PADDING_RATIO: float = 0.25
    MIN_PADDING: float = 0.02

    def __init__(self, config: Config) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.config = config

        base_options = python.BaseOptions(model_asset_path=config.HAND_LANDMARKER_MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def close(self) -> None: self._detector.close()

    def process(self, frame_bgr: np.ndarray) -> Optional[dict]:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect(mp_image)

        if not result.hand_landmarks: return None

        hand = result.hand_landmarks[0]

        xs = [lm.x for lm in hand]
        ys = [lm.y for lm in hand]
        x_min_hand, x_max_hand = min(xs), max(xs)
        y_min_hand, y_max_hand = min(ys), max(ys)

        bbox_w = x_max_hand - x_min_hand
        bbox_h = y_max_hand - y_min_hand

        pad_x = max(bbox_w * self.PADDING_RATIO, self.MIN_PADDING)
        pad_y = max(bbox_h * self.PADDING_RATIO, self.MIN_PADDING)

        x_min = max(0.0, x_min_hand - pad_x)
        x_max = min(1.0, x_max_hand + pad_x)
        y_min = max(0.0, y_min_hand - pad_y)
        y_max = min(1.0, y_max_hand + pad_y)

        px_min, px_max = int(x_min * width), int(x_max * width)
        py_min, py_max = int(y_min * height), int(y_max * height)

        if px_max <= px_min or py_max <= py_min: return None

        crop = frame_rgb[py_min:py_max, px_min:px_max]
        if crop.size == 0: return None

        target = self.config.GESTURE_IMG_SIZE
        crop_resized = self._letterbox(crop, target)

        palm_center = ((hand[0].x + hand[9].x) / 2.0, (hand[0].y + hand[9].y) / 2.0,)

        return {
            "crop_rgb": crop_resized,
            "bbox_xyxy": (px_min, py_min, px_max, py_max),
            "landmarks": [(lm.x, lm.y, lm.z) for lm in hand],
            "palm_center": palm_center,
        }

    @staticmethod
    def _letterbox(crop_rgb: np.ndarray, target: int) -> np.ndarray:
        crop_h, crop_w = crop_rgb.shape[:2]
        scale = min(target / crop_w, target / crop_h)
        new_w = max(1, int(round(crop_w * scale)))
        new_h = max(1, int(round(crop_h * scale)))

        resized = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((target, target, 3), 128, dtype=np.uint8)
        x_off = (target - new_w) // 2
        y_off = (target - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        return canvas

    def draw_landmarks(self, frame_bgr: np.ndarray, landmarks: list[tuple[float, float, float]],) -> None:
        h, w = frame_bgr.shape[:2]
        pts = [(int(x * w), int(y * h)) for x, y, _ in landmarks]
        for x, y in pts: cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)
        for a, b in HAND_CONNECTIONS: cv2.line(frame_bgr, pts[a], pts[b], (0, 200, 255), 2)