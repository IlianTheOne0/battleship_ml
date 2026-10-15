from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import pygame

from commands import (
    Command,
    ConfirmCommand,
    FireCommand,
    MoveCursorCommand,
    QuitCommand,
    RotateCommand,
    SetCursorCommand,
)
from inputs.base import InputStrategy

if TYPE_CHECKING:
    from game.battleship import BattleshipGame


class PygameInputStrategy(InputStrategy):
    mode_name = "keyboard"

    def __init__(self, view=None, game: "BattleshipGame | None" = None) -> None:
        self._view = view
        self._game = game
        self.logger = logging.getLogger(self.__class__.__name__)

    def poll(self, events: Sequence[pygame.event.Event] | None = None) -> Sequence[Command]:
        if events is None:
            events = pygame.event.get()

        commands: list[Command] = []
        for event in events:
            if event.type == pygame.KEYDOWN:
                cmd = self._map_key(event.key)
                if cmd is not None:
                    commands.append(cmd)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                cmd = self._map_click(event.pos)
                if cmd is not None:
                    commands.append(cmd)
        return commands

    def _map_key(self, key: int) -> Command | None:
        mapping = {
            pygame.K_q: QuitCommand,
            pygame.K_UP: lambda: MoveCursorCommand(-1, 0),
            pygame.K_DOWN: lambda: MoveCursorCommand(1, 0),
            pygame.K_LEFT: lambda: MoveCursorCommand(0, -1),
            pygame.K_RIGHT: lambda: MoveCursorCommand(0, 1),
            pygame.K_RETURN: ConfirmCommand,
            pygame.K_KP_ENTER: ConfirmCommand,
            pygame.K_SPACE: FireCommand,
            pygame.K_r: RotateCommand,
        }
        factory = mapping.get(key)
        return factory() if factory is not None else None

    def _map_click(self, position: tuple[int, int]) -> Command | None:
        if self._view is None or self._game is None:
            return None

        cell = self._view.screen_to_cell(position, self._game)
        if cell is None:
            return None

        row, column = cell
        return SetCursorCommand(row, column)