from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from config import Config
from ml.dataset import build_eval_transform, load_meta
from ml.gestures import GestureClass
from ml.hand_preprocess import HandPreprocessor
from ml.model import GestureCNN

_VOTE_CONFIDENCE_THRESHOLD: float = 0.35
_VOTE_MARGIN_THRESHOLD: float = 0.15
_SMOOTHING_WINDOW: int = 5
_SMOOTHING_MIN_VOTES: int = 3

@dataclass
class GesturePrediction:
    gesture: GestureClass | None
    confidence: float
    margin: float
    top_probs: list[tuple[str, float]]
    palm_center: tuple[float, float] | None
    bbox_xyxy: tuple[int, int, int, int] | None
    landmarks: list[tuple[float, float, float]] | None
    crop_rgb: np.ndarray | None

class GestureRecognizer:
    def __init__(self, config: Config) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.config = config

        meta = load_meta(config.GESTURE_META_PATH)
        self.classes: list[str] = list(meta["classes"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GestureCNN(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(torch.load(config.GESTURE_MODEL_PATH, weights_only=True))
        self.model.eval()

        self.transform = build_eval_transform(config)
        self.pre = HandPreprocessor(config)

        self._vote_buffer: deque[str | None] = deque(maxlen=_SMOOTHING_WINDOW)

        self.logger.info("The service is standby (device=%s, classes=%s)", self.device, self.classes,)

    def close(self) -> None: self.pre.close()

    def recognize(self, frame_bgr: np.ndarray) -> GesturePrediction:
        result = self.pre.process(frame_bgr)
        if result is None:
            self._vote_buffer.append(None)
            return GesturePrediction(gesture=None, confidence=0.0, margin=0.0, top_probs=[], palm_center=None, bbox_xyxy=None, landmarks=None, crop_rgb=None,)

        crop_pil = Image.fromarray(result["crop_rgb"])
        tensor = self.transform(crop_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        top_probs, top_indexes = probs.topk(min(3, len(self.classes)))
        top1_prob = float(top_probs[0])
        top2_prob = float(top_probs[1]) if len(top_probs) > 1 else 0.0
        top1_idx = int(top_indexes[0])
        margin = top1_prob - top2_prob

        is_confident = (top1_prob >= _VOTE_CONFIDENCE_THRESHOLD and margin >= _VOTE_MARGIN_THRESHOLD)
        if is_confident: self._vote_buffer.append(self.classes[top1_idx])
        else: self._vote_buffer.append(None)

        smoothed = self._smoothed_gesture()
        top_pairs = [(self.classes[index], float(prob)) for prob, index in zip(top_probs.tolist(), top_indexes.tolist())]

        return GesturePrediction(gesture=smoothed, confidence=top1_prob, margin=margin, top_probs=top_pairs, palm_center=result["palm_center"], bbox_xyxy=result["bbox_xyxy"], landmarks=result["landmarks"], crop_rgb=result["crop_rgb"],)

    def draw_landmarks(self, frame_bgr: np.ndarray, landmarks: list[tuple[float, float, float]],) -> None: self.pre.draw_landmarks(frame_bgr, landmarks)

    def _smoothed_gesture(self) -> GestureClass | None:
        valid = [v for v in self._vote_buffer if v is not None]
        if len(valid) < _SMOOTHING_MIN_VOTES: return None
        most_common, count = Counter(valid).most_common(1)[0]
        if count < _SMOOTHING_MIN_VOTES: return None
        try: return GestureClass(most_common)
        except ValueError: return None