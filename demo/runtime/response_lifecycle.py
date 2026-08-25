"""Thread-safe response playback lifecycle with stale-callback rejection."""

from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResponseLifecycleEvent:
    run_id: str
    response_id: str
    event: str


@dataclass
class _RunLifecycle:
    active_response_id: str | None = None
    status_by_response: dict[str, str] = field(default_factory=dict)
    events: deque[ResponseLifecycleEvent] = field(default_factory=deque)
    stale_callback_count: int = 0


class ResponseLifecycleRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, _RunLifecycle] = {}

    def create_response(self, run_id: str) -> tuple[str, str]:
        response_id = uuid.uuid4().hex[:12]
        with self._lock:
            lifecycle = self._runs.setdefault(run_id, _RunLifecycle())
            lifecycle.active_response_id = response_id
            lifecycle.status_by_response[response_id] = "PENDING"
        return response_id, f"RESPONSE:{run_id}:{response_id}"

    @staticmethod
    def parse_token(token: str | None) -> tuple[str, str] | None:
        parts = str(token or "").split(":")
        if len(parts) != 3 or parts[0] not in {"RESPONSE", "STOP"}:
            return None
        return parts[1], parts[2]

    def record_started(self, token: str | None) -> bool:
        parsed = self.parse_token(token)
        if parsed is None:
            return False
        run_id, response_id = parsed
        with self._lock:
            lifecycle = self._runs.get(run_id)
            if (
                lifecycle is None
                or lifecycle.active_response_id != response_id
                or lifecycle.status_by_response.get(response_id) != "PENDING"
            ):
                if lifecycle is not None:
                    lifecycle.stale_callback_count += 1
                return False
            lifecycle.status_by_response[response_id] = "PLAYING"
            lifecycle.events.append(
                ResponseLifecycleEvent(run_id, response_id, "SYSTEM_RESPONSE_STARTED")
            )
            return True

    def record_finished(self, token: str | None) -> bool:
        parsed = self.parse_token(token)
        if parsed is None:
            return False
        run_id, response_id = parsed
        with self._lock:
            lifecycle = self._runs.get(run_id)
            if (
                lifecycle is None
                or lifecycle.active_response_id != response_id
                or lifecycle.status_by_response.get(response_id) != "PLAYING"
            ):
                if lifecycle is not None:
                    lifecycle.stale_callback_count += 1
                return False
            lifecycle.status_by_response[response_id] = "FINISHED"
            lifecycle.active_response_id = None
            lifecycle.events.append(
                ResponseLifecycleEvent(run_id, response_id, "SYSTEM_RESPONSE_FINISHED")
            )
            return True

    def cancel_response(self, run_id: str, response_id: str) -> bool:
        with self._lock:
            lifecycle = self._runs.get(run_id)
            if (
                lifecycle is None
                or lifecycle.active_response_id != response_id
            ):
                return False
            lifecycle.status_by_response[response_id] = "CANCELLED"
            lifecycle.active_response_id = None
            return True

    def drain(self, run_id: str) -> list[ResponseLifecycleEvent]:
        with self._lock:
            lifecycle = self._runs.get(run_id)
            if lifecycle is None:
                return []
            events = list(lifecycle.events)
            lifecycle.events.clear()
            return events

    def active_response_id(self, run_id: str) -> str | None:
        with self._lock:
            lifecycle = self._runs.get(run_id)
            return (
                lifecycle.active_response_id if lifecycle is not None else None
            )

    def stale_callback_count(self, run_id: str) -> int:
        with self._lock:
            lifecycle = self._runs.get(run_id)
            return lifecycle.stale_callback_count if lifecycle else 0

    def close_run(self, run_id: str) -> None:
        with self._lock:
            self._runs.pop(run_id, None)
