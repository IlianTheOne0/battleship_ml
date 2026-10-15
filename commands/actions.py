from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from commands.base import Command

if TYPE_CHECKING: from game.battleship import BattleshipGame

class ConfirmCommand(Command):
    name = "confirm"

    def execute(self, game: "BattleshipGame") -> None:
        if game.handoff_pending:
            game.confirm_handoff()
            return
        game.try_place_current_ship()

class FireCommand(Command):
    name = "fire"

    def execute(self, game: "BattleshipGame") -> None:
        game.fire_at_cursor()

class RotateCommand(Command):
    name = "rotate"

    def execute(self, game: "BattleshipGame") -> None:
        game.rotate_orientation()

class QuitCommand(Command):
    name = "quit"

    def execute(self, game: "BattleshipGame") -> None:
        logging.getLogger(self.__class__.__name__).info("Користувач вийшов з гри.")