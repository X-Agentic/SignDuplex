"""Metadata-driven timing plans for a balanced paper-demo interaction."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Iterable

from runtime.subtitle_timeline import ReferenceSubtitleSegment


DEMO_SCENARIO_MODE = "BALANCED_IDLE_BARGE"
EXPECTED_PAUSE_DURATION_RANGE = (5.0, 6.0)
IDLE_MARGIN_SECONDS = 1.5
OVERLAP_MARGIN_SECONDS = 1.75
MIN_SHORT_TTS_DURATION_SECONDS = 1.0
MAX_SHORT_TTS_DURATION_SECONDS = 2.5
MIN_LONG_TTS_DURATION_SECONDS = 3.5
_SCENARIO_SEED_SALT = 0x53444736


@dataclass(frozen=True)
class PauseValidation:
    segment_index: int
    segment_end: float
    next_segment_start: float
    pause_duration: float
    expected_range: tuple[float, float]
    status: str

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["expected_range"] = list(self.expected_range)
        return record


@dataclass(frozen=True)
class ReplyScenarioPlan:
    slot: int
    scenario_reply_mode: str
    complete_time: float
    next_sign_start: float

    def as_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DemoValidation:
    segment_count: int
    pauses: tuple[PauseValidation, ...]
    planned_reply_modes: tuple[str, ...]

    @property
    def pause_duration_validation(self) -> bool:
        return bool(self.pauses) and all(
            pause.status == "PASS" for pause in self.pauses
        )

    @property
    def expected_idle_opportunity_count(self) -> int:
        return self.planned_reply_modes.count("SHORT")

    @property
    def expected_barge_opportunity_count(self) -> int:
        return self.planned_reply_modes.count("LONG")

    def as_record(self, *, video_id: str) -> dict[str, object]:
        return {
            "video_id": video_id,
            "demo_scenario_mode": DEMO_SCENARIO_MODE,
            "segment_count": self.segment_count,
            "pause_durations": [
                round(pause.pause_duration, 6) for pause in self.pauses
            ],
            "pause_details": [pause.as_record() for pause in self.pauses],
            "pause_duration_validation": self.pause_duration_validation,
            "planned_reply_modes": list(self.planned_reply_modes),
            "expected_idle_opportunity_count": (
                self.expected_idle_opportunity_count
            ),
            "expected_barge_opportunity_count": (
                self.expected_barge_opportunity_count
            ),
        }


def validate_segment_pauses(
    segments: Iterable[ReferenceSubtitleSegment],
) -> tuple[PauseValidation, ...]:
    ordered = tuple(segments)
    low, high = EXPECTED_PAUSE_DURATION_RANGE
    records = []
    for current, following in zip(ordered, ordered[1:]):
        duration = round(following.start_time - current.end_time, 6)
        records.append(
            PauseValidation(
                segment_index=current.segment_index,
                segment_end=current.end_time,
                next_segment_start=following.start_time,
                pause_duration=duration,
                expected_range=EXPECTED_PAUSE_DURATION_RANGE,
                status="PASS" if low <= duration <= high else "WARNING",
            )
        )
    return tuple(records)


class BalancedDemoScenarioPlanner:
    """Plan SHORT/LONG timing without touching model predictions."""

    def __init__(
        self,
        segments: Iterable[ReferenceSubtitleSegment],
        random_seed: int,
    ) -> None:
        self.segments = tuple(segments)
        opportunity_count = min(2, max(0, len(self.segments) - 1))
        modes = ["SHORT", "LONG"][:opportunity_count]
        if len(modes) == 2:
            random.Random(
                int(random_seed) ^ _SCENARIO_SEED_SALT
            ).shuffle(modes)
        self.planned_reply_modes = tuple(modes)
        self._next_slot = 0

    def validation(self) -> DemoValidation:
        return DemoValidation(
            segment_count=len(self.segments),
            pauses=validate_segment_pauses(self.segments),
            planned_reply_modes=self.planned_reply_modes,
        )

    def reserve(self, complete_time: float) -> ReplyScenarioPlan | None:
        if self._next_slot >= len(self.planned_reply_modes):
            return None
        next_sign_start = next(
            (
                segment.start_time
                for segment in self.segments
                if segment.start_time > complete_time + 1e-6
            ),
            None,
        )
        if next_sign_start is None:
            return None
        slot = self._next_slot
        self._next_slot += 1
        return ReplyScenarioPlan(
            slot=slot,
            scenario_reply_mode=self.planned_reply_modes[slot],
            complete_time=float(complete_time),
            next_sign_start=float(next_sign_start),
        )

    def is_final_reply_opportunity(self, complete_time: float) -> bool:
        """Return true only in the last metadata segment with no future sign."""
        if not self.segments:
            return False
        in_final_segment = complete_time >= self.segments[-1].start_time - 1e-6
        future_sign_exists = any(
            segment.start_time > complete_time + 1e-6
            for segment in self.segments
        )
        return in_final_segment and not future_sign_exists


def adaptive_tts_duration(
    scenario_reply_mode: str,
    *,
    response_start_time: float,
    next_sign_start: float,
) -> float:
    """Create a real timing opportunity, never a synthetic prediction."""
    available_gap = float(next_sign_start) - float(response_start_time)
    if scenario_reply_mode == "SHORT":
        duration = min(
            MAX_SHORT_TTS_DURATION_SECONDS,
            max(
                MIN_SHORT_TTS_DURATION_SECONDS,
                available_gap - IDLE_MARGIN_SECONDS,
            ),
        )
        if duration >= available_gap:
            duration = max(0.1, available_gap - 0.1)
        return duration
    if scenario_reply_mode == "LONG":
        return max(
            MIN_LONG_TTS_DURATION_SECONDS,
            available_gap + OVERLAP_MARGIN_SECONDS,
        )
    raise ValueError(f"Unsupported scenario reply mode: {scenario_reply_mode}")
