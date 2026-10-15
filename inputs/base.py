from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from commands.base import Command

class InputStrategy(ABC):
    mode_name: str = "abstract"

    @abstractmethod
    def poll(self) -> Sequence[Command]: pass
    def shutdown(self) -> None: pass