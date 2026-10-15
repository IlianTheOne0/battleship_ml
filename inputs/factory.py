from __future__ import annotations

from typing import TYPE_CHECKING

from config import Config, InputMode
from inputs.base import InputStrategy
from inputs.pygame_input import PygameInputStrategy

if TYPE_CHECKING:
    from game.battleship import BattleshipGame
    from ui.pygame_view import PygameView

class InputStrategyFactory:
    @staticmethod
    def create(config: Config, view: "PygameView", game: "BattleshipGame", ) -> InputStrategy:
        if config.INPUT_MODE is InputMode.KEYBOARD: return PygameInputStrategy(view=view, game=game)

        if config.INPUT_MODE is InputMode.GESTURE:
            from inputs.gesture_input import GestureInputStrategy

            try: return GestureInputStrategy(config=config, view=view, game=game)
            except FileNotFoundError as exception:
                raise RuntimeError(f"No gesture model artifacts found: {exception}. First train the model") from exception

        if config.INPUT_MODE is InputMode.VOICE:
            from inputs.voice_input import VoiceInputStrategy

            return VoiceInputStrategy(config=config, view=view, game=game)

        raise ValueError(f"Unknown input mode: {config.INPUT_MODE!r}")