from __future__ import annotations

from enum import Enum

class CellState(str, Enum):
    EMPTY = "empty"
    SHIP = "ship"
    HIT = "hit"
    MISS = "miss"

class Cell:
    def __init__(self, state: CellState = CellState.EMPTY, ship_id: int = -1) -> None:
        self.state = state
        self.ship_id = ship_id

    __slots__ = ("state", "ship_id")

    def __repr__(self) -> str: return f"Cell(state={self.state.value!r}, ship_id={self.ship_id})"