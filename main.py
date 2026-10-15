from __future__ import annotations

import logging
import random
import sys

import pygame

import config as config_outer
from commands import QuitCommand, ConfirmCommand
from game.battleship import BattleshipGame
from game.game_state import GamePhase
from inputs.factory import InputStrategyFactory
from logger_setup import setup_logging
from ui.pygame_view import PygameView
from utils import set_seed

def main() -> int:
    setup_logging()
    logger = logging.getLogger("main")

    config = config_outer.Config()
    set_seed(config.RANDOM_SEED)
    rng = random.Random(config.RANDOM_SEED)

    logger.info("Starting the game: mode=%s, field=%dx%d", config.GAME_MODE.value, config.BOARD_SIZE, config.BOARD_SIZE,)

    try: view = PygameView(config)
    except pygame.error as exc:
        logger.error("Failed to initialize the pygame window: %s", exc)
        return 1

    game = BattleshipGame(config, rng)
    try: strategy = InputStrategyFactory.create(config, view, game)
    except (NotImplementedError, RuntimeError) as exc:
        logger.error("Failed to initialize the input strategy: %s", exc)
        view.close()
        return 2

    running = True
    try:
        while running:
            commands = strategy.poll()
            for command in commands:
                if isinstance(command, QuitCommand):
                    running = False
                    break

                if game.handoff_pending and not isinstance(command, ConfirmCommand): continue
                command.execute(game)

            if not running: break

            view.render(game)

            if game.phase is GamePhase.BATTLE and game.is_ai_turn():
                pygame.time.delay(config.AI_DELAY_MS)
                game.ai_take_one_shot()
                view.render(game)
    except KeyboardInterrupt: logger.info("The game was stopped due to the interrupt")
    finally:
        try: strategy.shutdown()
        except Exception as exc: logger.error("The error while closing service: %s", exc)
        view.close()

    logger.info("The game has been finished!")
    return 0

if __name__ == "__main__": sys.exit(main())