from __future__ import annotations

from enum import Enum

class GamePhase(str, Enum):
    PLACEMENT = "placement"
    BATTLE = "battle"
    GAME_OVER = "game_over"


class ShotOutcome(str, Enum):
    MISS = "miss"
    HIT = "hit"
    SUNK = "sunk"
    ALREADY_SHOT = "already_shot"
    OUT_OF_BOUNDS = "out_of_bounds"
    INVALID = "invalid"