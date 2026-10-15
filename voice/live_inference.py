from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import re

from typing import Optional

import sounddevice as sd

import config as config_outer
from commands import Command
from logger_setup import setup_logging
from voice.command_parser import parse_text
from voice.tts import TextToSpeech
from voice.vosk_recognizer import VoskRecognizer

_COLOR_RESET = "\033[0m"
_COLOR_OK = "\033[92m"
_COLOR_WARN = "\033[93m"
_COLOR_BAD = "\033[91m"
_COLOR_INFO = "\033[96m"

def _describe_command(command: Command) -> str:
    name = command.name
    if name == "set_cursor":
        row = getattr(command, "row", "?")
        column = getattr(command, "col", "?")
        letter = chr(ord("A") + column) if isinstance(column, int) and 0 <= column < 26 else "?"
        return f"set_cursor -> {letter}{row + 1}" if isinstance(row, int) else "set_cursor"
    return name

def _print_banner(config: config_outer.Config) -> None:
    print(f"{_COLOR_INFO}{'=' * 60}{_COLOR_RESET}")
    print(f"{_COLOR_INFO} Voice Live Inference (Stage 3, Vosk) {_COLOR_RESET}")
    print(f"{_COLOR_INFO}{'=' * 60}{_COLOR_RESET}")
    print(f" Model            : {config.VOICE_VOSK_MODEL_PATH}")
    print(f" Sample rate      : {config.VOICE_VOSK_SAMPLE_RATE}")
    print(f" Cooldown         : {config.VOICE_COOLDOWN_MS}ms")
    print(f" TTS enabled      : {config.VOICE_TTS_ENABLED}")
    print()
    print(" Say a command, e.g.:")
    print("   'A five', 'F seven', 'B ten'  -> move cursor")
    print("   'fire' / 'shoot'               -> shoot")
    print("   'confirm' / 'place'            -> confirm")
    print("   'rotate'                       -> rotate")
    print("   'quit' / 'exit'                -> exit")
    print()
    print(f" {_COLOR_WARN}Press Ctrl+C to stop.{_COLOR_RESET}")
    print(f"{_COLOR_INFO}{'=' * 60}{_COLOR_RESET}\n")

def main() -> int:
    setup_logging()
    logger = logging.getLogger("voice.live_inference")

    config = config_outer.Config()

    try: tts = TextToSpeech(config)
    except Exception as exception:
        logger.error("Failed to initialize TTS: %s", exception)
        return 1

    try: recognizer = VoskRecognizer(config)
    except Exception as exception:
        logger.error("Failed to initialize Vosk recognizer: %s. Check that the model exists at '%s'",
            exception, config.VOICE_VOSK_MODEL_PATH,
        )
        tts.shutdown()
        return 1

    _print_banner(config)

    command_queue: queue.Queue[Command] = queue.Queue()
    stop = threading.Event()

    last_fire_ts: float = 0.0
    cooldown_seconds = config.VOICE_COOLDOWN_MS / 1000.0

    def audio_callback(indata, frames, time_info, status) -> None:
        nonlocal last_fire_ts
        if stop.is_set(): return
        if status: logger.debug("Audio status: %s", status)

        try: text = recognizer.accept_waveform(bytes(indata))
        except Exception as exception:
            logger.error("Vosk error: %s", exception)
            return

        if not text: return

        text = text.strip().lower()
        print(f"{_COLOR_INFO} [HEARD] '{text}'{_COLOR_RESET}")
        logger.info("Recognized: '%s'", text)

        now = time.monotonic()
        if now - last_fire_ts < cooldown_seconds:
            remaining = cooldown_seconds - (now - last_fire_ts)
            print(f"{_COLOR_WARN} [SKIP ] cooldown ({remaining:.2f}s){_COLOR_RESET}")
            return

        command = parse_text(text)
        if command is None:
            print(f"{_COLOR_BAD} [MISS ] no command matched{_COLOR_RESET}")
            logger.debug("No command matched for: '%s'", text)
            if config.VOICE_TTS_ENABLED: tts.say("not recognized")
            last_fire_ts = now
            return

        last_fire_ts = now
        command_queue.put(command)
        print(f"{_COLOR_OK} [CMD  ] {_describe_command(command)}{_COLOR_RESET}")
        logger.info("Command parsed: %s", command)

        if config.VOICE_TTS_ENABLED: tts.say(_feedback_for(command, text))

    stream: Optional[sd.RawInputStream] = None
    try:
        stream = sd.RawInputStream(samplerate=config.VOICE_VOSK_SAMPLE_RATE, blocksize=8000, dtype="int16", channels=1, callback=audio_callback,)
        stream.start()
        logger.info("Listening... (Ctrl+C to stop)")

        while True: time.sleep(0.2)
    except KeyboardInterrupt:
        print(f"\n{_COLOR_WARN} Interrupted by user.{_COLOR_RESET}")
        logger.info("Stopped by user")
    except Exception as exception:
        logger.error("Fatal error in listen loop: %s", exception)
        return 2
    finally:
        stop.set()
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exception: logger.warning("Failed to close audio stream: %s", exception)
        try: tts.shutdown()
        except Exception as exception: logger.warning("Failed to shutdown TTS: %s", exception)

    print(f"{_COLOR_INFO} Bye!{_COLOR_RESET}")
    return 0

def _feedback_for(command: Command, raw_text: str) -> str:
    name = command.name
    if name == "set_cursor":
        match = re.search(r"([a-j])\s+(\w+)", raw_text)
        return f"{match.group(1).upper()} {match.group(2)}" if match else "move"
    if name in ("fire", "confirm", "rotate", "quit"): return name
    return name

if __name__ == "__main__": sys.exit(main())