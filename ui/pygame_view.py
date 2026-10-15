from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pygame

from config import Config, InputMode
from game.cell import CellState
from game.game_state import GamePhase

if TYPE_CHECKING: from game.battleship import BattleshipGame

_COORD_LETTERS = "ABCDEFGHIJ"

class PygameView:
    def __init__(self, config: Config) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.config = config

        pygame.init()
        pygame.display.set_caption("Battleship")

        self.cell = config.CELL_SIZE
        self.board_px = self.cell * config.BOARD_SIZE
        self.coord = config.COORD_SIZE

        width = (config.PADDING * 2 + self.coord + self.board_px + config.BOARD_GAP + self.coord + self.board_px)
        height = (config.PADDING * 2 + config.TITLE_HEIGHT + config.LABEL_HEIGHT + self.coord + self.board_px + config.HUD_HEIGHT)

        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        font_name = "dejavusans"

        self.font_title = pygame.font.SysFont(font_name, 30, bold=True)
        self.font_label = pygame.font.SysFont(font_name, 20, bold=True)
        self.font_hud = pygame.font.SysFont(font_name, 18)
        self.font_hint = pygame.font.SysFont(font_name, 14)
        self.font_coord = pygame.font.SysFont(font_name, 16, bold=True)

        board_top = (config.PADDING + config.TITLE_HEIGHT + config.LABEL_HEIGHT + self.coord)
        self.left_origin = (config.PADDING + self.coord, board_top)
        self.right_origin = (config.PADDING + self.coord + self.board_px + config.BOARD_GAP + self.coord, board_top,)

        self._controls_hint = self._build_controls_hint(config.INPUT_MODE)

    def screen_to_cell(self, pos: tuple[int, int], game: "BattleshipGame",) -> tuple[int, int] | None:
        origin = self.right_origin if game.phase is GamePhase.BATTLE else self.left_origin
        x, y = pos
        ox, oy = origin

        if not (ox <= x < ox + self.board_px and oy <= y < oy + self.board_px): return None
        return ((y - oy) // self.cell, (x - ox) // self.cell)

    def render(self, game: "BattleshipGame") -> None:
        self.screen.fill(self.config.COLOR_BG)

        if game.handoff_pending: self._draw_handoff_screen(game)
        else:
            self._draw_title()
            self._draw_board_pair(game)
            self._draw_hud(game)

        pygame.display.flip()
        self.clock.tick(self.config.FPS)

    def close(self) -> None: pygame.quit()

    def _draw_title(self) -> None:
        text = self.font_title.render("Battleship", True, self.config.COLOR_TEXT)
        self.screen.blit(text, (self.config.PADDING, self.config.PADDING))

    def _draw_board_pair(self, game: "BattleshipGame") -> None:
        viewer = game.viewer_player()
        opponent = game.opponent_of(viewer)

        label_y = self.left_origin[1] - self.coord - 28
        self._draw_label(f"Field of {viewer.name}", self.left_origin[0], label_y)
        self._draw_label(f"Field of opponent, {opponent.name}", self.right_origin[0], label_y,)

        self._draw_board(origin=self.left_origin, board=viewer.board, hide_ships=False)
        self._draw_board(origin=self.right_origin, board=opponent.board, hide_ships=True)

        self._draw_placement_preview(game)

    def _draw_label(self, text: str, x: int, y: int) -> None:
        rendered = self.font_label.render(text, True, self.config.COLOR_TEXT)
        self.screen.blit(rendered, (x, y))

    def _draw_coordinates(self, origin: tuple[int, int], board) -> None:
        ox, oy = origin
        cell = self.cell
        half = self.coord // 2

        for column in range(board.size):
            letter = (_COORD_LETTERS[column] if column < len(_COORD_LETTERS) else str(column + 1))
            surf = self.font_coord.render(letter, True, self.config.COLOR_SUBTEXT)
            rect = surf.get_rect(center=(ox + column * cell + cell // 2, oy - half))
            self.screen.blit(surf, rect)

        for row in range(board.size):
            surf = self.font_coord.render(str(row + 1), True, self.config.COLOR_SUBTEXT)
            rect = surf.get_rect(center=(ox - half, oy + row * cell + cell // 2))
            self.screen.blit(surf, rect)

    def _draw_board(self, origin: tuple[int, int], board, hide_ships: bool,) -> None:
        ox, oy = origin
        cell = self.cell

        pygame.draw.rect(self.screen, self.config.COLOR_GRID_BG, pygame.Rect(ox, oy, self.board_px, self.board_px),)

        for row in range(board.size):
            for column in range(board.size):
                state = board.grid[row][column].state
                rect = pygame.Rect(ox + column * cell, oy + row * cell, cell, cell)

                if state is CellState.HIT: pygame.draw.rect(self.screen, self.config.COLOR_HIT, rect)
                elif state is CellState.MISS: pygame.draw.rect(self.screen, self.config.COLOR_MISS, rect)
                elif state is CellState.SHIP and not hide_ships:
                    pygame.draw.rect(self.screen, self.config.COLOR_SHIP, rect.inflate(-4, -4),)

                pygame.draw.rect(self.screen, self.config.COLOR_GRID_LINE, rect, 1)
        self._draw_coordinates(origin, board)

    def _draw_placement_preview(self, game: "BattleshipGame") -> None:
        if game.phase is GamePhase.PLACEMENT:
            cells = game.preview_cells()
            color = (self.config.COLOR_PREVIEW_OK if game.can_place_at_cursor() else self.config.COLOR_PREVIEW_BAD)
            for row, column in cells:
                rect = pygame.Rect(self.left_origin[0] + column * self.cell, self.left_origin[1] + row * self.cell, self.cell, self.cell,)
                pygame.draw.rect(self.screen, color, rect, 3)

            cursor_rect = pygame.Rect(self.left_origin[0] + game.cursor_col * self.cell, self.left_origin[1] + game.cursor_row * self.cell, self.cell, self.cell,)
            pygame.draw.rect(self.screen, self.config.COLOR_CURSOR, cursor_rect, 3)
            return

        if game.phase is GamePhase.BATTLE:
            origin = self.left_origin if game.is_ai_turn() else self.right_origin
            cursor_rect = pygame.Rect(origin[0] + game.cursor_col * self.cell, origin[1] + game.cursor_row * self.cell, self.cell, self.cell,)
            pygame.draw.rect(self.screen, self.config.COLOR_CURSOR, cursor_rect, 3)

    def _draw_handoff_screen(self, game: "BattleshipGame") -> None:
        width, height = self.screen.get_size()
        cx = width // 2

        player_name = game.current_player().name
        if game.phase is GamePhase.PLACEMENT: phase_text = "Ship positioning phase"
        else: phase_text = "Battle Phase"

        title_surf = self.font_title.render("Pass the device to another player", True, self.config.COLOR_TEXT)
        name_surf = self.font_title.render(player_name, True, self.config.COLOR_ACCENT)
        phase_surf = self.font_hud.render(phase_text, True, self.config.COLOR_SUBTEXT)
        hint_surf = self.font_hud.render("Press Enter when you're ready", True, self.config.COLOR_SUBTEXT,)

        blocks = [title_surf, name_surf, phase_surf, hint_surf]
        gap = 40
        total_h = sum(b.get_height() for b in blocks) + gap * (len(blocks) - 1)
        y = (height - total_h) // 2

        for surf in blocks:
            rect = surf.get_rect(center=(cx, y + surf.get_height() // 2))
            self.screen.blit(surf, rect)
            y += surf.get_height() + gap

    def _build_controls_hint(self, mode: InputMode) -> str:
        if mode is InputMode.GESTURE: return ("Gestures: palm moves the cursor | index finger - confirm | fist - shoot | two fingers (victory) - rotate | thumb up - OK")
        if mode is InputMode.VOICE: return ("Voice: say a coordinate (e.g. 'A5', 'F7') | 'fire' | 'confirm' | 'rotate' | 'quit'")
        return ("Keyboard: ← → ↑ ↓ to move | R to rotate | Enter to place | Space: Shoot | Q: Exit")

    def _draw_hud(self, game: "BattleshipGame") -> None:
        x = self.config.PADDING
        y = (self.config.PADDING + self.config.TITLE_HEIGHT + self.config.LABEL_HEIGHT + self.coord + self.board_px + 15)

        if game.phase is GamePhase.PLACEMENT:
            ship = game.current_ship_size()
            header = (f"Positioning: {game.current_player().name} | The next ship is: {ship}-deck | Orientation: {game.orientation.value}")
        elif game.phase is GamePhase.BATTLE: header = f"Battle. Move of : {game.current_player().name}"
        else:
            winner = game.winner.name if game.winner else "-"
            header = f"The game is over. Winner is: {winner}"

        self.screen.blit(self.font_hud.render(header, True, self.config.COLOR_ACCENT), (x, y),)
        self.screen.blit(self.font_hud.render(game.message, True, self.config.COLOR_TEXT), (x, y + 24),)
        self.screen.blit(self.font_hint.render(self._controls_hint, True, self.config.COLOR_SUBTEXT,), (x, y + 50),)