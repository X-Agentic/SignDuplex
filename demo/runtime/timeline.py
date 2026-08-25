"""Append-only per-run prediction timeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TimelineLogger:
    def __init__(
        self, logs_dir: Path, run_id: str, prefix: str = "prediction_timeline"
    ):
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = logs_dir / f"{prefix}_{run_id}.jsonl"

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
