#!/usr/bin/env python3
"""Smoke-test deepseek_pisanitizer across PIArena scopes."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _pi_defense_smoke import (  # noqa: E402
    AGENT_SMOKE_MATRIX,
    DEFAULT_BACKEND_LLM,
    STANDARD_SMOKE_MATRIX,
    build_commands,
    main as _main,
    parse_args as _parse_args,
)


DEFENSE = "deepseek_pisanitizer"
DEFAULT_CONFIG = "configs/experiments/deepseek_pisanitizer.yaml"
DEFAULT_NAME = "deepseek_pisanitizer_smoke"
DEFAULT_WORK_DIR = ".tmp/deepseek_pisanitizer_smoke"


def parse_args(argv: list[str] | None = None):
    return _parse_args(
        argv,
        defense=DEFENSE,
        default_config=DEFAULT_CONFIG,
        default_name=DEFAULT_NAME,
        default_work_dir=DEFAULT_WORK_DIR,
    )


def main(argv: list[str] | None = None) -> int:
    return _main(
        argv,
        defense=DEFENSE,
        default_config=DEFAULT_CONFIG,
        default_name=DEFAULT_NAME,
        default_work_dir=DEFAULT_WORK_DIR,
    )


if __name__ == "__main__":
    raise SystemExit(main())
