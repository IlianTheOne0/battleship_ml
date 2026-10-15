from __future__ import annotations

from enum import Enum

class Orientation(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"

class Ship:
    def __init__(self, ship_id: int, size: int, row: int, column: int, orientation: Orientation, ) -> None:
        self.ship_id = ship_id
        self.size = size
        self.row = row
        self.column = column
        self.orientation = orientation
        self.hits: set[tuple[int, int]] = set()

    def __repr__(self) -> str: return (f"Ship(id={self.ship_id}, size={self.size}, row={self.row}, col={self.column}, orientation={self.orientation.value}, hits={len(self.hits)}/{self.size})")
    __slots__ = ("ship_id", "size", "row", "column", "orientation", "hits")

    def occupies(self) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        for index in range(self.size):
            row = self.row + (index if self.orientation is Orientation.VERTICAL else 0)
            column = self.column + (index if self.orientation is Orientation.HORIZONTAL else 0)
            cells.append((row, column))
        return cells

    def register_hit(self, row: int, col: int) -> None: self.hits.add((row, col))
    def is_sunk(self) -> bool: return len(self.hits) >= self.size