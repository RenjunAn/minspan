#!/usr/bin/env python3
"""Run full non-WASP leaderboard benchmarks for modernbert_tagger."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _pi_defense_full import (  # noqa: E402
    LEADERBOARD_AGENT_MATRIX,
    LEADERBOARD_STANDARD_MATRIX,
    build_commands,
    export_leaderboard_results,
    main as _main,
    parse_args as _parse_args,
)


DEFENSE = "modernbert_tagger"
DEFAULT_CONFIG = "configs/experiments/modernbert_tagger.yaml"
DEFAULT_NAME = "modernbert_tagger_full"
SCRIPT_NAME = Path(__file__).name


def parse_args(argv: list[str] | None = None):
    return _parse_args(
        argv,
        defense=DEFENSE,
        default_config=DEFAULT_CONFIG,
        default_name=DEFAULT_NAME,
        script_name=SCRIPT_NAME,
    )


def main(argv: list[str] | None = None) -> int:
    return _main(
        argv,
        defense=DEFENSE,
        default_config=DEFAULT_CONFIG,
        default_name=DEFAULT_NAME,
        script_name=SCRIPT_NAME,
    )


if __name__ == "__main__":
    raise SystemExit(main())
