from __future__ import annotations

import logging
import queue
import threading
import time
import numpy as np

from typing import TYPE_CHECKING, Sequence

import sounddevice as sd

from commands import Command
from config import Config
from game.game_state import GamePhase
from inputs.base import InputStrategy
from inputs.pygame_input import PygameInputStrategy
from voice.command_parser import parse_text
from voice.tts import TextToSpeech
from voice.vosk_recognizer import VoskRecognizer

if TYPE_CHECKING:
    from game.battleship import BattleshipGame
    from ui.pygame_view import PygameView

class VoiceInputStrategy(InputStrategy):
    mode_name = "voice"

    def __init__(self, config: Config, view: "PygameView", game: "BattleshipGame") -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.config = config
        self.view = view
        self.game = game

        self._pygame_fallback = PygameInputStrategy(view=view, game=game)
        self._tts = TextToSpeech(config)
        self._recognizer = VoskRecognizer(config)

        self._commands: queue.Queue[Command] = queue.Queue()
        self._stop = threading.Event()

        self._last_fire_ts: float = 0.0

        self._last_activity_ts: float = time.monotonic()
        self._last_turn_reminder_ts: float = 0.0

        self._game_over_spoken: bool = False

        self._stream = sd.RawInputStream(samplerate=config.VOICE_VOSK_SAMPLE_RATE, blocksize=8000, dtype="int16", channels=1, callback=self._audio_callback,)
        self._stream.start()
        self.logger.info("VoiceInputStrategy (Vosk) started (model=%s)", config.VOICE_VOSK_MODEL_PATH)

    def poll(self) -> Sequence[Command]:
        commands: list[Command] = list(self._pygame_fallback.poll())
        while True:
            try: commands.append(self._commands.get_nowait())
            except queue.Empty: break

        if commands: self._last_activity_ts = time.monotonic()

        if self.config.VOICE_TTS_ENABLED:
            self._maybe_speak_game_over()
            self._maybe_speak_turn_reminder()

        return commands

    def shutdown(self) -> None:
        self.logger.info("Stopping voice service...")
        self._stop.set()
        try:
            self._stream.stop()
            self._stream.close()
        except Exception as exception: self.logger.warning("Failed to close audio stream: %s", exception)
        self._tts.shutdown()
        self.logger.info("Voice service stopped")

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if self._stop.is_set(): return
        try: text = self._recognizer.accept_waveform(bytes(indata))
        except Exception as exc:
            self.logger.error("Vosk error: %s", exc)
            return

        if not text: return

        text = text.strip().lower()
        self.logger.info("Recognized: '%s'", text)

        now = time.monotonic()
        cooldown_seconds = self.config.VOICE_COOLDOWN_MS / 1000.0
        if now - self._last_fire_ts < cooldown_seconds:
            self.logger.debug("Cooldown active, ignoring: '%s'", text)
            return

        command = parse_text(text)
        if command is None:
            self.logger.debug("No command matched for: '%s'", text)
            if self.config.VOICE_TTS_ENABLED: self._tts.say("not recognized")
            self._last_fire_ts = now
            return

        self._last_fire_ts = now
        self._last_activity_ts = now
        self._commands.put(command)

        if self.config.VOICE_TTS_ENABLED: self._tts.say(self._feedback_for(command, text))

    def _feedback_for(self, command: Command, raw_text: str) -> str:
        name = command.name
        if name == "set_cursor":
            row = getattr(command, "row", None)
            col = getattr(command, "col", None)
            if isinstance(row, int) and isinstance(col, int):
                nato = { 0: "Alpha", 1: "Bravo", 2: "Charlie", 3: "Delta", 4: "Echo", 5: "Foxtrot", 6: "Golf", 7: "Hotel", 8: "India", 9: "Juliett", }
                letter = nato.get(col, chr(ord("A") + col))
                return f"{letter} {row + 1}"
            return "move"
        return name

    def _maybe_speak_game_over(self) -> None:
        if self._game_over_spoken: return
        if self.game.phase is not GamePhase.GAME_OVER: return
        winner = self.game.winner
        if winner is None: return

        self._game_over_spoken = True
        self._tts.say(f"{winner.name} won!")

    def _maybe_speak_turn_reminder(self) -> None:
        if self.game.phase is GamePhase.GAME_OVER: return

        if self.game.is_ai_turn():
            self._last_activity_ts = time.monotonic()
            return
        if self.game.handoff_pending:
            self._last_activity_ts = time.monotonic()
            return

        now = time.monotonic()
        idle = now - self._last_activity_ts
        reminder = self.config.VOICE_TURN_REMINDER_SECONDS

        if idle < reminder: return
        if now - self._last_turn_reminder_ts < reminder: return

        self._last_turn_reminder_ts = now
        self._beep()
        player = self.game.current_player().name
        self._tts.say(f"{player}, your turn")

    def _beep(self) -> None:
        try:
            sr = 44100
            duration = 0.75
            freq = 800
            amplitude = 0.9

            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            wave = amplitude * np.sin(2 * np.pi * freq * t)

            fade = int(sr * 0.01)
            wave[:fade] *= np.linspace(0, 1, fade)
            wave[-fade:] *= np.linspace(1, 0, fade)

            sd.play(wave.astype(np.float32), sr, blocking=True)
        except Exception as exception: self.logger.warning("Beep failed: %s", exception)