from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING: from game.battleship import BattleshipGame

class Command(ABC):
    name: str = "command"

    def __repr__(self) -> str: return f"{self.__class__.__name__}()"

    @abstractmethod
    def execute(self, game: "BattleshipGame") -> None: pass