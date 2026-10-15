from __future__ import annotations

import logging
import re
from typing import Optional

import nltk

from commands import (
    Command,
    ConfirmCommand,
    FireCommand,
    QuitCommand,
    RotateCommand,
    SetCursorCommand,
)

for _pkg in ("punkt", "punkt_tab"):
    try: nltk.data.find(f"tokenizers/{_pkg}")
    except LookupError: nltk.download(_pkg, quiet=True)

_SPOKEN_LETTERS: dict[str, str] = {
    "a": "a", "b": "b", "c": "c", "d": "d", "e": "e",
    "f": "f", "g": "g", "h": "h", "i": "i", "j": "j",

    "ay": "a", "bee": "b", "see": "c", "sea": "c", "dee": "d",
    "ee": "e", "eff": "f", "gee": "g", "aitch": "h",
    "eye": "i", "jay": "j",

    "alpha": "a", "bravo": "b", "charlie": "c", "delta": "d", "echo": "e",
    "foxtrot": "f", "golf": "g", "hotel": "h", "india": "i", "juliet": "j",
}

_SPOKEN_NUMBERS: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2", "to": "2", "too": "2",
    "three": "3",
    "four": "4", "for": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8", "ate": "8",
    "nine": "9",
    "ten": "10",
}

_COORD_RE = re.compile(r"^([a-jA-J])\s*(\d{1,2})$")

def _normalize_tokens(text: str) -> list[str]:
    tokens = nltk.word_tokenize(text.lower())
    result: list[str] = []
    for token in tokens:
        if token in _SPOKEN_LETTERS: result.append(_SPOKEN_LETTERS[token])
        elif token in _SPOKEN_NUMBERS: result.append(_SPOKEN_NUMBERS[token])
        else: result.append(token)
    return result


def parse_text(text: str) -> Optional[Command]:
    if not text: return None

    tokens = _normalize_tokens(text)

    for index, token in enumerate(tokens):
        if index + 1 < len(tokens):
            candidate = f"{token}{tokens[index + 1]}"
            match = _COORD_RE.match(candidate)
            if match:
                column = ord(match.group(1).upper()) - ord("A")
                row = int(match.group(2)) - 1
                if 0 <= column < 10 and 0 <= row < 10: return SetCursorCommand(row=row, col=column)

        match = _COORD_RE.match(token)
        if match:
            column = ord(match.group(1).upper()) - ord("A")
            row = int(match.group(2)) - 1
            if 0 <= column < 10 and 0 <= row < 10: return SetCursorCommand(row=row, col=column)

    joined = " ".join(tokens)
    if any(w in joined for w in ("fire", "shoot", "shot")): return FireCommand()
    if any(w in joined for w in ("confirm", "place", "ok", "yes")): return ConfirmCommand()
    if any(w in joined for w in ("rotate", "turn", "orientation")): return RotateCommand()
    if any(w in joined for w in ("quit", "exit", "stop", "close")): return QuitCommand()

    return None