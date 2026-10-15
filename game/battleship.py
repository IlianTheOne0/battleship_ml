from __future__ import annotations

import logging
import random

from game.game_state import GamePhase, ShotOutcome
from game.player import AIPlayer, HumanPlayer, Player
from game.ship import Orientation

from config import Config, GameMode

class BattleshipGame:
    def __init__(self, cfg: Config, rng: random.Random) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.config = cfg
        self.range = rng

        self.players: list[Player] = self._build_players()
        self.phase: GamePhase = GamePhase.PLACEMENT

        self.placing_player_index: int = 0
        self.placement_queue: list[int] = list(cfg.FLEET)

        self.turn_player_index: int = 0

        self.winner: Player | None = None
        self.message: str = "Ship Placement. Place the largest ship"

        self.handoff_pending: bool = False

        self._auto_place_ai_fleets()

    @property
    def cursor_row(self) -> int: return self.current_player().cursor_row
    @cursor_row.setter
    def cursor_row(self, value: int) -> None: self.current_player().cursor_row = value

    @property
    def cursor_col(self) -> int: return self.current_player().cursor_column
    @cursor_col.setter
    def cursor_col(self, value: int) -> None: self.current_player().cursor_column = value

    @property
    def orientation(self) -> Orientation: return self.current_player().orientation
    @orientation.setter
    def orientation(self, value: Orientation) -> None: self.current_player().orientation = value

    def _build_players(self) -> list[Player]:
        size = self.config.BOARD_SIZE
        if self.config.GAME_MODE is GameMode.HUMAN_VS_AI:
            return [
                HumanPlayer("Player", size),
                AIPlayer("AI", size, self.range),
            ]
        return [
            HumanPlayer("Player No. 1", size),
            HumanPlayer("Player No. 2", size),
        ]

    def _auto_place_ai_fleets(self) -> None:
        for player in self.players:
            if isinstance(player, AIPlayer): player.auto_place_fleet(self.config.FLEET)
        self._skip_ai_placers()

    def _skip_ai_placers(self) -> None:
        while (
            self.phase is GamePhase.PLACEMENT
            and self.placing_player_index < len(self.players)
            and isinstance(self.players[self.placing_player_index], AIPlayer)
        ):
            self.placing_player_index += 1
            self.placement_queue = list(self.config.FLEET)

        if (
            self.phase is GamePhase.PLACEMENT
            and self.placing_player_index >= len(self.players)
        ):
            self._start_battle()

    def _start_battle(self) -> None:
        self.phase = GamePhase.BATTLE
        self.turn_player_index = 0
        self.placement_queue = []
        self.message = (f"The battle is is heating up from this point of time... Move of {self.players[self.turn_player_index].name}.")
        self.logger.info("Phase of the battle. First move: %s", self.players[0].name)

        self._request_handoff_if_needed()

    def current_player(self) -> Player:
        if self.phase is GamePhase.PLACEMENT: return self.players[self.placing_player_index]
        return self.players[self.turn_player_index]

    def enemy_player(self) -> Player:
        if self.phase is GamePhase.PLACEMENT: return self.players[1 - self.placing_player_index]
        return self.players[1 - self.turn_player_index]
    def viewer_player(self) -> Player:
        if self.config.GAME_MODE is GameMode.HUMAN_VS_AI: return self.players[0]
        return self.current_player()

    def opponent_of(self, player: Player) -> Player: return self.players[1] if player is self.players[0] else self.players[0]

    def is_ai_turn(self) -> bool:
        return (
            self.phase is GamePhase.BATTLE
            and isinstance(self.players[self.turn_player_index], AIPlayer)
        )

    def current_ship_size(self) -> int | None:
        if self.phase is not GamePhase.PLACEMENT or not self.placement_queue: return None
        return self.placement_queue[0]

    def preview_cells(self) -> list[tuple[int, int]]:
        size = self.current_ship_size()
        if size is None: return []

        orientation = self.orientation
        cells: list[tuple[int, int]] = []
        for index in range(size):
            row = self.cursor_row + (index if orientation is Orientation.VERTICAL else 0)
            column = self.cursor_col + (index if orientation is Orientation.HORIZONTAL else 0)
            cells.append((row, column))
        return cells

    def can_place_at_cursor(self) -> bool:
        size = self.current_ship_size()
        if size is None: return False
        return self.players[self.placing_player_index].board.can_place(self.cursor_row, self.cursor_col, size, self.orientation)

    def try_place_current_ship(self) -> bool:
        if self.phase is not GamePhase.PLACEMENT:
            self.message = "This isn't the setup phase"
            return False

        size = self.current_ship_size()
        if size is None: return False

        player = self.players[self.placing_player_index]
        ship = player.board.place_ship(self.cursor_row, self.cursor_col, size, self.orientation)
        if ship is None:
            self.message = "You can't place the ship here. Try another position or orientation"
            return False

        self.placement_queue.pop(0)
        self.logger.info("%s placed %d-deck.", player.name, size)

        if not self.placement_queue:
            self.message = f"{player.name} positioned all the ships"
            self._advance_placer()
        else: self.message = (f"The next ship is: {self.placement_queue[0]}-deck.")
        return True

    def _advance_placer(self) -> None:
        self.placing_player_index += 1
        self.placement_queue = list(self.config.FLEET)
        self._skip_ai_placers()

        if self.phase is GamePhase.PLACEMENT: self._request_handoff_if_needed()

    def fire_at_cursor(self) -> ShotOutcome:
        if self.phase is not GamePhase.BATTLE:
            self.message = "Shooting is  available only during the combat phase"
            return ShotOutcome.INVALID

        attacker = self.players[self.turn_player_index]
        defender = self.players[1 - self.turn_player_index]
        row, col = self.cursor_row, self.cursor_col
        outcome = defender.board.shoot(row, col)
        self.logger.info("%s shoots (%d, %d) -> %s",attacker.name, row, col, outcome.value,)

        if isinstance(attacker, AIPlayer): attacker.register_shot_outcome(row, col, outcome)

        if outcome in (ShotOutcome.HIT, ShotOutcome.SUNK):
            if defender.board.all_ships_sunk():
                self.phase = GamePhase.GAME_OVER
                self.winner = attacker
                self.message = f"{attacker.name} won!"
                return outcome
            self.message = (f"{attacker.name}: sank!" if outcome is ShotOutcome.SUNK else f"{attacker.name}: Hit! The next move is by the same player!")
        elif outcome is ShotOutcome.MISS:
            self.message = f"{attacker.name}: miss"
            self.turn_player_index = 1 - self.turn_player_index
            self.message += f" Move: {self.players[self.turn_player_index].name}."
            self._request_handoff_if_needed()
        elif outcome is ShotOutcome.ALREADY_SHOT: self.message = "This area has already been shot at"
        elif outcome is ShotOutcome.OUT_OF_BOUNDS: self.message = "Outside the field"

        return outcome

    def ai_take_one_shot(self) -> None:
        if not self.is_ai_turn(): return

        attacker = self.players[self.turn_player_index]
        defender = self.players[1 - self.turn_player_index]
        row, col = attacker.choose_shot(defender.board)
        attacker.cursor_row, attacker.cursor_column = row, col

        self.fire_at_cursor()

    def move_cursor(self, d_row: int, d_col: int) -> None:
        max_rows = max_columns = self.config.BOARD_SIZE - 1

        self.cursor_row = max(0, min(max_rows, self.cursor_row + d_row))
        self.cursor_col = max(0, min(max_columns, self.cursor_col + d_col))

    def set_cursor(self, row: int, column: int) -> None:
        max_r = max_c = self.config.BOARD_SIZE - 1
        self.cursor_row = max(0, min(max_r, row))
        self.cursor_col = max(0, min(max_c, column))

    def rotate_orientation(self) -> None:
        if self.phase is not GamePhase.PLACEMENT: return

        self.orientation = (
            Orientation.VERTICAL
            if self.orientation is Orientation.HORIZONTAL
            else Orientation.HORIZONTAL
        )
        self.message = f"Orientation: {self.orientation.value}."

    def confirm_handoff(self) -> None: self.handoff_pending = False

    def _request_handoff_if_needed(self) -> None:
        if self.config.GAME_MODE is GameMode.HUMAN_VS_HUMAN: self.handoff_pending = True