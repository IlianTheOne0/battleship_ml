from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Sequence

import cv2
import numpy as np

from commands import Command, SetCursorCommand
from config import Config
from inputs.base import InputStrategy
from inputs.pygame_input import PygameInputStrategy
from ml.gesture_recognizer import GesturePrediction, GestureRecognizer
from ml.gestures import gesture_to_command

if TYPE_CHECKING:
    from game.battleship import BattleshipGame
    from ui.pygame_view import PygameView

_COLOR_OK = (0, 255, 0)
_COLOR_WARN = (0, 200, 255)
_COLOR_BAD = (0, 0, 255)

class GestureInputStrategy(InputStrategy):
    mode_name = "gesture"

    def __init__(self, config: Config, view: "PygameView", game: "BattleshipGame", ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.config = config
        self.view = view
        self.game = game

        self._pygame_fallback = PygameInputStrategy(view=view, game=game)
        self._recognizer = GestureRecognizer(config)

        self._cap = cv2.VideoCapture(config.GESTURE_CAMERA_INDEX)
        if not self._cap.isOpened():
            self._recognizer.close()
            raise RuntimeError(f"Unable to open the camera (index={config.GESTURE_CAMERA_INDEX}).")

        self._commands: queue.Queue[Command] = queue.Queue()
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_prediction: GesturePrediction | None = None
        self._palm_center: tuple[float, float] | None = None

        self._last_fired_gesture: str | None = None
        self._last_fire_ts: float = 0.0
        self._smoothed_palm: tuple[float, float] | None = None
        self._last_cell: tuple[int, int] | None = None

        self._running = threading.Event()
        self._running.set()
        self._thread = threading.Thread(target=self._worker_loop, name="GestureInputWorker", daemon=True,)
        self._thread.start()

        self.logger.info(
            "GestureInputStrategy was started: camera=%d, cooldown=%dms, confidence_threshold=%.2f",
            config.GESTURE_CAMERA_INDEX,
            config.GESTURE_COOLDOWN_MS,
            config.GESTURE_CONFIDENCE_THRESHOLD,
        )

    def poll(self) -> Sequence[Command]:
        commands: list[Command] = list(self._pygame_fallback.poll())

        while True:
            try: commands.append(self._commands.get_nowait())
            except queue.Empty: break

        cursor_cmd = self._cursor_command()
        if cursor_cmd is not None: commands.append(cursor_cmd)

        self._render_window_with_camera()

        return commands

    def shutdown(self) -> None:
        self.logger.info("Stopping the service...")
        self._running.clear()
        if self._thread.is_alive(): self._thread.join(timeout=2.0)

        try: self._cap.release()
        except Exception as exception: self.logger.warning("Camera Release Error: %s", exception)

        try: self._recognizer.close()
        except Exception as exception: self.logger.warning("Recognizer closure error: %s", exception)

        if self.config.GESTURE_DEBUG_WINDOW:
            try: cv2.destroyAllWindows()
            except Exception: pass
        self.logger.info("The service is stopped")

    def _worker_loop(self) -> None:
        cooldown_seconds = self.config.GESTURE_COOLDOWN_MS / 1000.0

        while self._running.is_set():
            ok, frame = self._cap.read()
            if not ok:
                self.logger.warning("Shot not received; stopping the stream")
                break

            frame = cv2.flip(frame, 1)

            try: prediction = self._recognizer.recognize(frame)
            except Exception as exception:
                self.logger.error("Inference Error: %s", exception)
                continue

            with self._lock:
                self._latest_frame = frame
                self._latest_prediction = prediction
                self._palm_center = prediction.palm_center

            now = time.monotonic()
            gesture_value = (prediction.gesture.value if prediction.gesture is not None else None)

            if gesture_value is None:
                if now - self._last_fire_ts >= cooldown_seconds: self._last_fired_gesture = None
                continue
            if prediction.confidence < self.config.GESTURE_CONFIDENCE_THRESHOLD: continue
            if gesture_value == self._last_fired_gesture: continue
            if now - self._last_fire_ts < cooldown_seconds: continue

            try: command = gesture_to_command(prediction.gesture)
            except ValueError as exception:
                self.logger.error("Gesture mapping failed: %s", exception)
                continue

            self._commands.put(command)
            self._last_fired_gesture = gesture_value
            self._last_fire_ts = now
            self.logger.info("Gesture %s (%.2f) → %s", gesture_value, prediction.confidence, command.name,)

    def _cursor_command(self) -> Command | None:
        with self._lock: palm = self._palm_center

        if palm is None: return None

        smoothed = self._smooth_palm(palm)
        if smoothed is None: return None

        cell = self._palm_to_cell(smoothed)
        if cell is None or cell == self._last_cell: return None

        self._last_cell = cell
        return SetCursorCommand(row=cell[0], col=cell[1])

    def _smooth_palm(self, palm: tuple[float, float]) -> tuple[float, float] | None:
        px, py = palm
        if self._smoothed_palm is None:
            self._smoothed_palm = (px, py)
            return self._smoothed_palm

        sx, sy = self._smoothed_palm
        deadzone = self.config.GESTURE_CURSOR_DEADZONE
        if abs(px - sx) < deadzone and abs(py - sy) < deadzone: return self._smoothed_palm

        alpha = self.config.GESTURE_CURSOR_SMOOTHING
        self._smoothed_palm = (alpha * px + (1.0 - alpha) * sx, alpha * py + (1.0 - alpha) * sy,)
        return self._smoothed_palm

    def _palm_to_cell(self, palm: tuple[float, float]) -> tuple[int, int] | None:
        px, py = palm
        x_lo, x_hi = self.config.GESTURE_CURSOR_X_RANGE
        y_lo, y_hi = self.config.GESTURE_CURSOR_Y_RANGE

        nx = (px - x_lo) / (x_hi - x_lo)
        ny = (py - y_lo) / (y_hi - y_lo)

        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0): return None

        max_idx = self.config.BOARD_SIZE - 1
        col = max(0, min(max_idx, int(round(nx * max_idx))))
        row = max(0, min(max_idx, int(round(ny * max_idx))))
        return (row, col)

    def _render_window_with_camera(self) -> None:
        with self._lock:
            frame = self._latest_frame
            prediction = self._latest_prediction

        if frame is None: return

        canvas = frame.copy()
        if prediction is not None: self._draw_prediction(canvas, prediction)

        palm = prediction.palm_center if prediction is not None else None
        self._draw_input_frame(canvas, palm)

        cv2.imshow("Camera window", canvas)
        cv2.waitKey(1)

    def _draw_prediction(self, canvas: np.ndarray, prediction: GesturePrediction) -> None:
        if prediction.bbox_xyxy is not None:
            x1, y1, x2, y2 = prediction.bbox_xyxy
            cv2.rectangle(canvas, (x1, y1), (x2, y2), _COLOR_OK, 2)
            if prediction.landmarks is not None: self._recognizer.draw_landmarks(canvas, prediction.landmarks)

        if prediction.gesture is not None:
            hud = f"{prediction.gesture.value} - {prediction.confidence * 100:.1f}%"
            color = _COLOR_OK
        elif prediction.bbox_xyxy is not None:
            hud = f"Unknown - {prediction.confidence * 100:.1f}%"
            color = _COLOR_WARN
        else:
            hud = "No hand"
            color = _COLOR_BAD

        cv2.putText(canvas, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA,)

        h = canvas.shape[0]
        y0 = h - 15 * (len(prediction.top_probs) + 1) - 10
        for i, (name, prob) in enumerate(prediction.top_probs):
            color = _COLOR_OK if i == 0 else _COLOR_WARN
            text = f"{name:<10} {prob * 100:5.1f}%"
            cv2.putText(canvas, text, (10, y0 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,)
        cv2.putText(
            canvas, f"margin:    {prediction.margin * 100:5.1f}%",
            (10, y0 + len(prediction.top_probs) * 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR_WARN, 1, cv2.LINE_AA,
        )

    def _draw_input_frame(self, canvas: np.ndarray, palm_center: tuple[float, float] | None,) -> None:
        h, w = canvas.shape[:2]
        x_lo, x_hi = self.config.GESTURE_CURSOR_X_RANGE
        y_lo, y_hi = self.config.GESTURE_CURSOR_Y_RANGE

        x1, x2 = int(x_lo * w), int(x_hi * w)
        y1, y2 = int(y_lo * h), int(y_hi * h)

        inside = (
            palm_center is not None
            and x_lo <= palm_center[0] <= x_hi
            and y_lo <= palm_center[1] <= y_hi
        )
        color = _COLOR_OK if inside else _COLOR_WARN

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, y1), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, y2), (w, h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, y1), (x1, y2), (0, 0, 0), -1)
        cv2.rectangle(overlay, (x2, y1), (w, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, "INPUT ZONE", (x1 + 8, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,)

        if palm_center is not None:
            px = int(palm_center[0] * w)
            py = int(palm_center[1] * h)
            cv2.circle(canvas, (px, py), 9, color, 2)
            cv2.circle(canvas, (px, py), 3, color, -1)
            if not inside:
                cv2.putText(canvas, "OUT", (px + 12, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,)

        if inside:
            cell = self._palm_to_cell(palm_center)
            if cell is not None:
                row, col = cell
                letter = "ABCDEFGHIJ"[col] if col < 10 else str(col + 1)
                label = f"-> {letter}{row + 1}"
                cv2.putText(canvas, label, (x1 + 8, y2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,)