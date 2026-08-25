"""Single-slot latest-only causal inference worker."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from runtime.model_runtime import ModelRuntime
from runtime.stream_cache import DecisionBoundary, PreparedVideo


@dataclass(frozen=True)
class InferenceRequest:
    decision: DecisionBoundary
    submitted_wall_seconds: float
    task_prefix_start_frame: int = 0


@dataclass(frozen=True)
class InferenceResult:
    request: InferenceRequest
    prediction: dict[str, Any]
    worker_total_ms: float
    completed_wall_seconds: float


class LatestOnlyInferenceWorker:
    def __init__(
        self,
        model: ModelRuntime,
        prepared: PreparedVideo,
        system_state: str,
        wall_started: float,
        *,
        pause_after_result: bool = False,
    ):
        self.model = model
        self.prepared = prepared
        self.system_state = system_state
        self.wall_started = wall_started
        self.pause_after_result = pause_after_result
        self._condition = threading.Condition()
        self._pending: InferenceRequest | None = None
        self._busy = False
        self._awaiting_result_ack = False
        self._closed = False
        self._results: queue.Queue[InferenceResult] = queue.Queue()
        self.submitted = 0
        self.dropped = 0
        self.completed = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, request: InferenceRequest) -> None:
        with self._condition:
            if self._closed:
                return
            if self._pending is not None:
                self.dropped += 1
            self._pending = request
            self.submitted += 1
            self._condition.notify()

    def drain(self) -> list[InferenceResult | Exception]:
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                return results

    def is_idle(self) -> bool:
        with self._condition:
            return (
                not self._busy
                and self._pending is None
                and not self._awaiting_result_ack
            )

    def has_results(self) -> bool:
        return not self._results.empty()

    def clear_pending(self) -> None:
        with self._condition:
            if self._pending is not None:
                self.dropped += 1
            self._pending = None

    def acknowledge_result(self) -> None:
        """Allow a paused worker to route its next pending request."""
        with self._condition:
            self._awaiting_result_ack = False
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._awaiting_result_ack = False
            self._condition.notify_all()
        self.thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed
                    or (
                        self._pending is not None
                        and not self._awaiting_result_ack
                    )
                )
                if self._closed:
                    return
                request = self._pending
                self._pending = None
                self._busy = True
            started = time.perf_counter()
            try:
                prediction = self.model.predict(
                    self.prepared.rgb_sample(
                        request.decision.prefix_end_frame,
                        request.task_prefix_start_frame,
                    ),
                    self.prepared.pose_prefix(
                        request.decision.prefix_end_frame,
                        request.task_prefix_start_frame,
                    ),
                    self.system_state,
                )
                finished = time.perf_counter()
                completed = InferenceResult(
                    request=request,
                    prediction=prediction,
                    worker_total_ms=(finished - started) * 1000.0,
                    completed_wall_seconds=finished - self.wall_started,
                )
                with self._condition:
                    self.completed += 1
                    self._awaiting_result_ack = self.pause_after_result
                self._results.put(completed)
            except Exception as error:
                self._results.put(error)
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()
