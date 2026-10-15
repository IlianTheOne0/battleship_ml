from __future__ import annotations

from typing import TYPE_CHECKING

from commands.base import Command

if TYPE_CHECKING: from game.battleship import BattleshipGame

class MoveCursorCommand(Command):
    name = "move_cursor"

    def __init__(self, d_row: int, d_col: int) -> None:
        self.d_row = d_row
        self.d_col = d_col

    def execute(self, game: "BattleshipGame") -> None: game.move_cursor(self.d_row, self.d_col)

class SetCursorCommand(Command):
    name = "set_cursor"

    def __init__(self, row: int, col: int) -> None:
        self.row = row
        self.col = col

    def execute(self, game: "BattleshipGame") -> None: game.set_cursor(self.row, self.col)