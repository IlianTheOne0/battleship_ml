from __future__ import annotations

import logging
import sys

import config as config_outer

def setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers: return

    config = config_outer.Config()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(config.LOG_LEVEL)

def get_logger(name: str) -> logging.Logger: return logging.getLogger(name)