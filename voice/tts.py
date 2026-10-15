from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import pyttsx3

from config import Config

class TextToSpeech:
    def __init__(self, config: Config) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._config = config

        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        if config.VOICE_TTS_ENABLED:
            self._thread = threading.Thread(target=self._worker_loop, name="TTSCWorker", daemon=True,)
            self._thread.start()
            self.logger.info("TTS worker started (rate=%d, volume=%.2f)", config.VOICE_TTS_RATE, config.VOICE_TTS_VOLUME)

    def say(self, phrase: str) -> None:
        if not self._config.VOICE_TTS_ENABLED or not phrase: return
        self._queue.put(phrase)

    def shutdown(self) -> None:
        self._stop.set()
        self._queue.put(None)
        if self._thread and self._thread.is_alive(): self._thread.join(timeout=2.0)
        self.logger.info("TTS worker stopped")

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try: phrase = self._queue.get(timeout=0.2)
            except queue.Empty: continue

            if phrase is None: break

            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self._config.VOICE_TTS_RATE)
                engine.setProperty("volume", self._config.VOICE_TTS_VOLUME)
                engine.say(phrase)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as exception: self.logger.error("TTS error: %s", exception)
            finally: self._queue.task_done()