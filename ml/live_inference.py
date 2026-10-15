from __future__ import annotations

import logging
import sys

import cv2
import numpy as np

import config as config_outer
from logger_setup import setup_logging
from ml.gesture_recognizer import GestureRecognizer
from utils import set_seed

CROP_PREVIEW_SIZE = 160
CROP_PREVIEW_MARGIN = 10
COLOR_OK = (0, 255, 0)
COLOR_WARN = (0, 200, 255)
COLOR_BAD = (0, 0, 255)

def _draw_crop_preview(frame_bgr: np.ndarray, crop_rgb: np.ndarray) -> None:
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    crop_resized = cv2.resize(crop_bgr, (CROP_PREVIEW_SIZE, CROP_PREVIEW_SIZE), interpolation=cv2.INTER_NEAREST,)

    h_frame, w_frame = frame_bgr.shape[:2]
    x1 = w_frame - CROP_PREVIEW_SIZE - CROP_PREVIEW_MARGIN
    y1 = CROP_PREVIEW_MARGIN
    x2 = w_frame - CROP_PREVIEW_MARGIN
    y2 = CROP_PREVIEW_MARGIN + CROP_PREVIEW_SIZE
    if x1 < 0 or y1 < 0: return

    frame_bgr[y1:y2, x1:x2] = crop_resized
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), COLOR_OK, 2)
    cv2.putText(frame_bgr, "Crop", (x1 + 5, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_OK, 1, cv2.LINE_AA)

def _draw_top_probs(frame_bgr: np.ndarray, top_probs: list[tuple[str, float]], margin: float,) -> None:
    h_frame = frame_bgr.shape[0]
    y0 = h_frame - 15 * (len(top_probs) + 1) - 10

    for index, (name, prob) in enumerate(top_probs):
        color = COLOR_OK if index == 0 else COLOR_WARN
        text = f"{name:<10} {prob * 100:5.1f}%"
        cv2.putText(frame_bgr, text, (10, y0 + index * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"margin:    {margin * 100:5.1f}%", (10, y0 + len(top_probs) * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WARN, 1, cv2.LINE_AA)

def main() -> int:
    setup_logging()
    logger = logging.getLogger("ml.live_inference")

    config = config_outer.Config()
    set_seed(config.RANDOM_SEED)

    try: recognizer = GestureRecognizer(config)
    except FileNotFoundError as exc:
        logger.error("No model artifacts found (%s). First, train the model", exc,)
        return 1

    cap = cv2.VideoCapture(config.GESTURE_CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("Unable to open the camera (index=%d)", config.GESTURE_CAMERA_INDEX)
        recognizer.close()
        return 1

    logger.info("Inference (device=%s). [Q] - exit, [S] - save crop.", recognizer.device,)

    last_crop_rgb: np.ndarray | None = None
    save_counter = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)

            pred = recognizer.recognize(frame)

            hud = "No hand"
            hud_color = COLOR_BAD

            if pred.bbox_xyxy is not None:
                x1, y1, x2, y2 = pred.bbox_xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_OK, 2)
                if pred.landmarks is not None: recognizer.draw_landmarks(frame, pred.landmarks)

                if pred.crop_rgb is not None:
                    last_crop_rgb = pred.crop_rgb
                    _draw_crop_preview(frame, pred.crop_rgb)

                if pred.gesture is not None:
                    hud = f"{pred.gesture.value} - {pred.confidence * 100:.1f}%"
                    hud_color = COLOR_OK
                else:
                    hud = f"Unknown - {pred.confidence * 100:.1f}%"
                    hud_color = COLOR_WARN

                _draw_top_probs(frame, pred.top_probs, pred.margin)

            cv2.putText(frame, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, hud_color, 2, cv2.LINE_AA)
            cv2.imshow("Gesture inference", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("s") and last_crop_rgb is not None:
                path = f"debug_crop_{save_counter:04d}.png"
                cv2.imwrite(path, cv2.cvtColor(last_crop_rgb, cv2.COLOR_RGB2BGR))
                logger.info("Cropp was saved: %s", path)
                save_counter += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())