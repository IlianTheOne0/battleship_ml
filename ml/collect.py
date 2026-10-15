from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2

import config as outer_config
from logger_setup import setup_logging
from ml.gestures import GestureClass
from ml.hand_preprocess import HandPreprocessor

def _prompt_class_selection(classes: tuple[str, ...]) -> GestureClass:
    print("\nSelect a class for collection:")
    for index, name in enumerate(classes, start=1):
        label = GestureClass(name).name
        print(f"  {index}. {label:<10}")
    while True:
        raw = input("No of class (1-5): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(classes): return GestureClass(classes[int(raw) - 1])
        print("Invalid selection. Please try again")

def _count_existing(data_dir: Path, cls: GestureClass) -> int:
    cls_dir = data_dir / cls.value
    if not cls_dir.exists(): return 0
    return sum(1 for _ in cls_dir.glob("*.jpg"))

def main() -> int:
    setup_logging()
    logger = logging.getLogger("ml.collect")

    config = outer_config.Config()
    data_dir = Path(config.GESTURE_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in config.GESTURE_CLASSES: (data_dir / name).mkdir(exist_ok=True)

    target_class = _prompt_class_selection(config.GESTURE_CLASSES)
    target_dir = data_dir / target_class.value

    pre = HandPreprocessor(config)
    cap = cv2.VideoCapture(config.GESTURE_CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("Unable to open the camera (index=%d)", config.GESTURE_CAMERA_INDEX)
        return 1

    logger.info("Class: '%s' (%s). Target: %d shots.",target_class.value, target_class, config.GESTURE_COLLECT_TARGET_PER_CLASS,)
    logger.info("Controls: [Space] - take a screenshot, [A] - auto-collect, [Q] - exit")

    counter = _count_existing(data_dir, target_class)
    auto_mode = False
    last_auto_ts = 0.0
    auto_interval_sec = 0.25

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("The frame was not received")
                break

            frame = cv2.flip(frame, 1)  # дзеркалимо для зручності
            result = pre.process(frame)

            hud_lines = [
                f"Class: {target_class.value} ({target_class.name})",
                f"Collected: {counter} / {config.GESTURE_COLLECT_TARGET_PER_CLASS}",
                f"Auto: {'ON' if auto_mode else 'OFF'}",
                "Space - shot | A - auto | Q - quit",
            ]

            if result is not None:
                x1, y1, x2, y2 = result["bbox_xyxy"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                pre.draw_landmarks(frame, result["landmarks"])
            else:
                hud_lines.append("NO HAND DETECTED")

            for i, line in enumerate(hud_lines):
                cv2.putText(
                    frame, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
                )

            cv2.imshow("Gesture collection", frame)
            key = cv2.waitKey(1) & 0xFF

            take_shot = False
            if key == ord(" "): take_shot = True
            elif key == ord("a"):
                auto_mode = not auto_mode
                logger.info("Auto-mode: %s", "enabled" if auto_mode else "disabled")
            elif key == ord("q"): break

            if auto_mode and result is not None:
                now = time.time()
                if now - last_auto_ts >= auto_interval_sec:
                    take_shot = True
                    last_auto_ts = now

            if take_shot and result is not None:
                ts = int(time.time() * 1000)
                path = target_dir / f"{ts}_{counter:05d}.jpg"
                cv2.imwrite(str(path), cv2.cvtColor(result["crop_rgb"], cv2.COLOR_RGB2BGR))
                counter += 1
                if counter >= config.GESTURE_COLLECT_TARGET_PER_CLASS: logger.info("The goal for the class has been achieved '%s'", target_class.value)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pre.close()

    logger.info("%d images have been collected for the class '%s'", counter, target_class.value)
    return 0

if __name__ == "__main__": sys.exit(main())