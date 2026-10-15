from __future__ import annotations

import logging
import random
from abc import ABC

from game.board import Board
from game.game_state import ShotOutcome
from game.ship import Orientation

class Player(ABC):
    def __init__(self, name: str, board: Board) -> None:
        self.name = name
        self.board = board
        self.cursor_row: int = 0
        self.cursor_column: int = 0
        self.orientation: Orientation = Orientation.HORIZONTAL
        self.logger = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str: return f"{self.__class__.__name__}(name={self.name!r})"

class HumanPlayer(Player):
    def __init__(self, name: str, size: int) -> None: super().__init__(name, Board(size))

class AIPlayer(Player):
    def __init__(self, name: str, size: int, range: random.Random) -> None:
        super().__init__(name, Board(size))
        self._range = range
        self._hunt_queue: list[tuple[int, int]] = []
        self._current_hits: list[tuple[int, int]] = []

    def auto_place_fleet(self, fleet: tuple[int, ...]) -> None:
        self.board.random_fill(fleet, self._range)
        self.logger.info("%s settled positions of it's ship automatically", self.name)

    def choose_shot(self, enemy_board: Board) -> tuple[int, int]:
        while self._hunt_queue:
            row, column = self._hunt_queue.pop()
            if not (0 <= row < enemy_board.size and 0 <= column < enemy_board.size): continue

            if (row, column) in enemy_board.shots: continue
            return row, column

        candidates = [
            (row, column)
            for row in range(enemy_board.size)
            for column in range(enemy_board.size)
            if (row, column) not in enemy_board.shots
        ]
        if not candidates: raise RuntimeError("There are no available cells for a shot")
        return self._range.choice(candidates)


    def register_shot_outcome(self, row: int, column: int, outcome: ShotOutcome, ) -> None:
        if outcome is ShotOutcome.SUNK:
            self._hunt_queue.clear()
            self._current_hits.clear()
            self.logger.debug("%s sank the ship; the target stack has been cleared", self.name)
            return

        if outcome is ShotOutcome.HIT:
            self._current_hits.append((row, column))
            self._refresh_hunt_queue(row, column)

    def _refresh_hunt_queue(self, row: int, column: int) -> None:
        if len(self._current_hits) >= 2:
            rows = {row_inner for row_inner, _ in self._current_hits}
            columns = {column_inner for _, column_inner in self._current_hits}

            if len(rows) == 1:
                rows = next(iter(rows))
                cs = sorted(c for _, c in self._current_hits)
                self._hunt_queue = [(rows, cs[0] - 1), (rows, cs[-1] + 1)]
                self.logger.debug("%s: column (row %d), targets: %r", self.name, rows, self._hunt_queue)
                return

            if len(columns) == 1:
                column = next(iter(columns))
                rs = sorted(r for r, _ in self._current_hits)
                self._hunt_queue = [(rs[0] - 1, column), (rs[-1] + 1, column)]
                self.logger.debug("%s: column (row %d), targets: %r", self.name, column, self._hunt_queue)
                return

            self._current_hits = [(row, column)]

        self._hunt_queue = [
            (row + dr, column + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        ]
        self.logger.debug("%s: one hit (%d, %d), targets: %r", self.name, row, column, self._hunt_queue,)