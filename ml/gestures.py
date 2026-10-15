from __future__ import annotations

from enum import Enum

from commands import (
    Command,
    ConfirmCommand,
    FireCommand,
    RotateCommand,
)

class GestureClass(str, Enum):
    POINT = "point"
    FIST = "fist"
    PALM = "palm"
    VICTORY = "victory"
    THUMBS_UP = "thumbs_up"


def gesture_to_command(gesture: GestureClass) -> Command:
    if gesture is GestureClass.POINT: return ConfirmCommand()
    if gesture is GestureClass.FIST: return FireCommand()
    if gesture is GestureClass.VICTORY: return RotateCommand()
    if gesture is GestureClass.THUMBS_UP: return ConfirmCommand()
    raise ValueError(f"Unknown gesture: {gesture.name!r}")