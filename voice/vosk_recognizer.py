from __future__ import annotations

import json
import logging
from typing import Optional

from vosk import KaldiRecognizer, Model

from config import Config

class VoskRecognizer:
    def __init__(self, config: Config) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        self.model = Model(config.VOICE_VOSK_MODEL_PATH)

        grammar = self._build_grammar(config)
        self._recognizer = KaldiRecognizer(self.model, config.VOICE_VOSK_SAMPLE_RATE, grammar)

        self.logger.info("Vosk recognizer initialized (model=%s)", config.VOICE_VOSK_MODEL_PATH)

    def _build_grammar(self, config: Config) -> str:
        letter_forms: dict[str, list[str]] = {
            "a": ["a", "ay", "alpha"],
            "b": ["b", "bee", "bravo"],
            "c": ["c", "see", "sea", "charlie"],
            "d": ["d", "dee", "delta"],
            "e": ["e", "ee", "echo"],
            "f": ["f", "eff", "foxtrot"],
            "g": ["g", "gee", "golf"],
            "h": ["h", "aitch", "hotel"],
            "i": ["i", "eye", "india"],
            "j": ["j", "jay", "juliet", "juliet"],
        }

        number_forms: dict[str, list[str]] = {
            "1": ["one"],
            "2": ["two", "to", "too"],
            "3": ["three"],
            "4": ["four", "for"],
            "5": ["five"],
            "6": ["six"],
            "7": ["seven"],
            "8": ["eight", "ate"],
            "9": ["nine"],
            "10": ["ten"],
        }

        phrases: set[str] = set()

        letters = "abcdefghij"[: config.BOARD_SIZE]
        for letter in letters:
            for letter_form in letter_forms[letter]:
                for number in range(1, config.BOARD_SIZE + 1):
                    for number_form in number_forms[str(number)]:
                        phrases.add(f"{letter_form} {number_form}")

        phrases.update(
            [
                "fire", "shoot", "shot",
                "confirm", "place", "ok", "yes",
                "rotate", "turn",
                "quit", "exit", "stop",
            ]
        )

        return json.dumps(sorted(phrases))

    def accept_waveform(self, data: bytes) -> Optional[str]:
        if self._recognizer.AcceptWaveform(data):
            result = json.loads(self._recognizer.Result())
            text = result.get("text", "").strip()
            return text if text else None
        return None

    def final_result(self) -> Optional[str]:
        result = json.loads(self._recognizer.FinalResult())
        text = result.get("text", "").strip()
        return text if text else None