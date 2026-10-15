from __future__ import annotations

import logging
import random
from typing import Optional

from game.cell import Cell, CellState
from game.game_state import ShotOutcome
from game.ship import Orientation, Ship

class Board:
    _MAX_FILL_ATTEMPTS: int = 2000

    def __init__(self, size: int) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.size = size
        self.grid: list[list[Cell]] = [
            [Cell() for _ in range(size)]
            for _ in range(size)
        ]
        self.ships: list[Ship] = []
        self.shots: set[tuple[int, int]] = set()
        self._next_ship_id = 0

    def __repr__(self) -> str: return f"Board(size={self.size}, ships={len(self.ships)})"

    def _cells_for(self, row: int, column: int, size: int, orientation: Orientation,) -> Optional[list[tuple[int, int]]]:
        cells: list[tuple[int, int]] = []

        for index in range(size):
            inner_row = row + (index if orientation is Orientation.VERTICAL else 0)
            inner_column = column + (index if orientation is Orientation.HORIZONTAL else 0)
            if not (0 <= inner_row < self.size and 0 <= inner_column < self.size): return None
            cells.append((inner_row, inner_column))
        return cells

    def can_place(self, row: int, column: int, size: int, orientation: Orientation,) -> bool:
        cells = self._cells_for(row, column, size, orientation)
        if cells is None: return False

        for inner_row, inner_column in cells:
            if self.grid[inner_row][inner_column].state is not CellState.EMPTY: return False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbor, neighbor_cell = inner_row + dr, inner_column + dc
                    if 0 <= neighbor < self.size and 0 <= neighbor_cell < self.size:
                        if self.grid[neighbor][neighbor_cell].state is CellState.SHIP:
                            return False
        return True

    def place_ship(self, row: int, column: int, size: int, orientation: Orientation,) -> Optional[Ship]:
        if not self.can_place(row, column, size, orientation): return None

        ship = Ship(self._next_ship_id, size, row, column, orientation)
        self._next_ship_id += 1

        for inner_row, inner_column in ship.occupies():
            self.grid[inner_row][inner_column].state = CellState.SHIP
            self.grid[inner_row][inner_column].ship_id = ship.ship_id

        self.ships.append(ship)
        self.logger.debug("The ship was placed at: %r", ship)

        return ship

    def random_fill(self, fleet: tuple[int, ...], range_outer: random.Random) -> None:
        for size in fleet:
            placed = False

            for _ in range(self._MAX_FILL_ATTEMPTS):
                orientation = range_outer.choice([Orientation.HORIZONTAL, Orientation.VERTICAL])
                row = range_outer.randrange(self.size)
                column = range_outer.randrange(self.size)

                if self.place_ship(row, column, size, orientation) is not None:
                    placed = True
                    break

            if not placed: raise RuntimeError(f"Unable to place the {size}-deck ship after {self._MAX_FILL_ATTEMPTS} attempts")

    def shoot(self, row: int, column: int) -> ShotOutcome:
        if not (0 <= row < self.size and 0 <= column < self.size): return ShotOutcome.OUT_OF_BOUNDS

        cell = self.grid[row][column]
        if cell.state in (CellState.HIT, CellState.MISS): return ShotOutcome.ALREADY_SHOT

        self.shots.add((row, column))

        if cell.state is CellState.SHIP:
            cell.state = CellState.HIT
            ship = self.ships[cell.ship_id]
            ship.register_hit(row, column)

            if ship.is_sunk():
                self._mark_ship_surroundings_as_miss(ship)
                return ShotOutcome.SUNK

            return ShotOutcome.HIT

        cell.state = CellState.MISS
        return ShotOutcome.MISS

    def _mark_ship_surroundings_as_miss(self, ship: Ship) -> None:
        for row, column in ship.occupies():
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0: continue
                    neighbor, neighbor_cell = row + dr, column + dc
                    
                    if not (0 <= neighbor < self.size and 0 <= neighbor_cell < self.size): continue
                    neighbor_inner = self.grid[neighbor][neighbor_cell]
                    if neighbor_inner.state is CellState.EMPTY:
                        neighbor_inner.state = CellState.MISS
                        self.shots.add((neighbor, neighbor_cell))

    def all_ships_sunk(self) -> bool: return bool(self.ships) and all(s.is_sunk() for s in self.ships)