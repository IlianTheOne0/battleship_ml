from __future__ import annotations

import logging
from enum import Enum
from typing import ClassVar

class InputMode(str, Enum):
    KEYBOARD = "keyboard"
    GESTURE = "gesture"
    VOICE = "voice"

class GameMode(str, Enum):
    HUMAN_VS_AI = "human_vs_ai"
    HUMAN_VS_HUMAN = "human_vs_human"


class SingletonMeta(type):
    _instances: ClassVar[dict[type, object]] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances: cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Config(metaclass=SingletonMeta):
    BOARD_SIZE: int = 10
    FLEET: tuple[int, ...] = (4, 3, 3, 2, 2, 2, 1, 1, 1, 1)

    GAME_MODE: GameMode = GameMode.HUMAN_VS_AI
    INPUT_MODE: InputMode = InputMode.KEYBOARD

    RANDOM_SEED: int = 873912

    CELL_SIZE: int = 40
    PADDING: int = 30
    BOARD_GAP: int = 60
    TITLE_HEIGHT: int = 60
    LABEL_HEIGHT: int = 30
    COORD_SIZE: int = 24
    HUD_HEIGHT: int = 80
    FPS: int = 60
    AI_DELAY_MS: int = 350

    COLOR_BG: tuple[int, int, int] = (18, 24, 38)
    COLOR_GRID_BG: tuple[int, int, int] = (28, 38, 58)
    COLOR_GRID_LINE: tuple[int, int, int] = (70, 95, 130)
    COLOR_SHIP: tuple[int, int, int] = (200, 210, 220)
    COLOR_HIT: tuple[int, int, int] = (220, 60, 60)
    COLOR_MISS: tuple[int, int, int] = (90, 150, 220)
    COLOR_CURSOR: tuple[int, int, int] = (255, 220, 80)
    COLOR_PREVIEW_OK: tuple[int, int, int] = (110, 220, 130)
    COLOR_PREVIEW_BAD: tuple[int, int, int] = (220, 90, 90)
    COLOR_TEXT: tuple[int, int, int] = (240, 240, 240)
    COLOR_SUBTEXT: tuple[int, int, int] = (170, 180, 200)
    COLOR_ACCENT: tuple[int, int, int] = (255, 200, 90)

    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    LOG_DATE_FORMAT: str = "%H:%M:%S"

    GESTURE_CURSOR_SMOOTHING: float = 0.35
    GESTURE_CURSOR_DEADZONE: float = 0.04
    GESTURE_CURSOR_X_RANGE: tuple[float, float] = (0.20, 0.80)
    GESTURE_CURSOR_Y_RANGE: tuple[float, float] = (0.20, 0.80)

    GESTURE_DATA_DIR: str = "data/gestures"
    GESTURE_MODEL_PATH: str = "models/gesture_cnn.pt"
    GESTURE_META_PATH: str = "models/gesture_cnn_meta.json"
    GESTURE_CLASSES: tuple[str, ...] = ("point", "fist", "palm", "victory", "thumbs_up",)
    HAND_LANDMARKER_MODEL_PATH: str = "./models/hand_landmarker.task"

    GESTURE_IMG_SIZE: int = 224
    GESTURE_BATCH_SIZE: int = 32
    GESTURE_EPOCHS: int = 30
    GESTURE_LEARNING_RATE: float = 1e-3
    GESTURE_WEIGHT_DECAY: float = 1e-4
    GESTURE_PATIENCE: int = 7
    GESTURE_SCHEDULER_PATIENCE: int = 3
    GESTURE_SCHEDULER_FACTOR: float = 0.5
    GESTURE_VAL_SPLIT: float = 0.15
    GESTURE_TEST_SPLIT: float = 0.15
    GESTURE_NUM_WORKERS: int = 2

    GESTURE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    GESTURE_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

    GESTURE_COLLECT_TARGET_PER_CLASS: int = 250
    GESTURE_CAMERA_INDEX: int = 0
    GESTURE_CROP_PADDING: float = 0.10

    GESTURE_INFER_INTERVAL_FRAMES: int = 2
    GESTURE_DEBOUNCE_FRAMES: int = 3
    GESTURE_COOLDOWN_MS: int = 800
    GESTURE_CONFIDENCE_THRESHOLD: float = 0.70

    GESTURE_CURSOR_SMOOTHING: float = 0.35
    GESTURE_CURSOR_DEADZONE: float = 0.04
    GESTURE_CURSOR_X_RANGE: tuple[float, float] = (0.15, 0.85)
    GESTURE_CURSOR_Y_RANGE: tuple[float, float] = (0.15, 0.85)