"""Deterministic demo-only simulation of thinking and TTS response duration."""

from __future__ import annotations

import random
from dataclasses import dataclass

from runtime.demo_scenario import ReplyScenarioPlan, adaptive_tts_duration


DEFAULT_RANDOM_SEED = 42
NORMAL_REPLY_PROBABILITY = 0.70
LONG_REPLY_PROBABILITY = 0.30
THINKING_TIME_RANGE = (0.3, 0.8)
NORMAL_TTS_DURATION_RANGE = (1.0, 2.5)
LONG_TTS_DURATION_RANGE = (3.5, 5.5)

RESPONSE_STATUSES = {
    "IDLE",
    "THINKING",
    "SPEAKING",
    "FINISHED",
    "CANCELLED_BY_BARGE",
}


@dataclass
class SimulatedResponse:
    response_id: str
    response_token: str
    complete_time: float
    thinking_time: float
    response_start_time: float
    tts_duration: float
    planned_response_end_time: float
    reply_mode: str
    random_seed: int
    scenario_reply_mode: str | None = None
    next_sign_start: float | None = None
    reply_timing_margin: float | None = None
    sampled_tts_duration: float | None = None
    final_reply_extension_seconds: float = 0.0
    status: str = "THINKING"

    def as_log_record(self) -> dict[str, object]:
        return {
            "event": "RESPONSE_SCHEDULED",
            "response_id": self.response_id,
            "complete_time": self.complete_time,
            "thinking_time": self.thinking_time,
            "response_start_time": self.response_start_time,
            "tts_duration": self.tts_duration,
            "planned_response_end_time": self.planned_response_end_time,
            "reply_mode": self.reply_mode,
            "scenario_reply_mode": self.scenario_reply_mode,
            "next_sign_start": self.next_sign_start,
            "reply_timing_margin": self.reply_timing_margin,
            "sampled_tts_duration": self.sampled_tts_duration,
            "final_reply_extension_seconds": (
                self.final_reply_extension_seconds
            ),
            "final_reply_extension_applied": (
                self.final_reply_extension_seconds > 0
            ),
            "random_seed": self.random_seed,
            "response_status": self.status,
        }

    def as_debug_record(self) -> dict[str, object]:
        return {
            "Reply Mode": self.reply_mode,
            "Scenario Reply Mode": self.scenario_reply_mode or "RANDOM",
            "Final Reply Extension": round(
                self.final_reply_extension_seconds,
                6,
            ),
            "Thinking Time": round(self.thinking_time, 6),
            "TTS Duration": round(self.tts_duration, 6),
            "Response Start Time": round(self.response_start_time, 6),
            "Planned Response End Time": round(
                self.planned_response_end_time, 6
            ),
            "Response Status": self.status,
        }


@dataclass(frozen=True)
class SimulatedResponseEvent:
    event: str
    event_time: float
    response_id: str
    response_token: str


class TTSResponseSimulator:
    """Sample and advance response lifecycles using only elapsed demo time."""

    def __init__(self, random_seed: int = DEFAULT_RANDOM_SEED) -> None:
        self.random_seed = int(random_seed)
        self._random = random.Random(self.random_seed)
        self._active: SimulatedResponse | None = None
        self._latest: SimulatedResponse | None = None
        self._responses: list[SimulatedResponse] = []

    @property
    def active_response_id(self) -> str | None:
        return self._active.response_id if self._active is not None else None

    @property
    def latest_response(self) -> SimulatedResponse | None:
        return self._latest

    @property
    def responses(self) -> tuple[SimulatedResponse, ...]:
        return tuple(self._responses)

    def schedule(
        self,
        response_id: str,
        response_token: str,
        complete_time: float,
        *,
        scenario_plan: ReplyScenarioPlan | None = None,
        final_reply_extension_seconds: float = 0.0,
    ) -> SimulatedResponse:
        if self._active is not None:
            raise RuntimeError("Cannot schedule while another response is active")

        # Sampling order is fixed so a given seed reproduces the full run.
        thinking_time = self._random.uniform(*THINKING_TIME_RANGE)
        sampled_reply_mode = (
            "NORMAL"
            if self._random.random() < NORMAL_REPLY_PROBABILITY
            else "LONG"
        )
        duration_range = (
            NORMAL_TTS_DURATION_RANGE
            if sampled_reply_mode == "NORMAL"
            else LONG_TTS_DURATION_RANGE
        )
        sampled_tts_duration = self._random.uniform(*duration_range)
        response_start_time = float(complete_time) + thinking_time
        scenario_reply_mode = (
            scenario_plan.scenario_reply_mode
            if scenario_plan is not None
            else None
        )
        next_sign_start = (
            scenario_plan.next_sign_start
            if scenario_plan is not None
            else None
        )
        if scenario_plan is None:
            reply_mode = sampled_reply_mode
            tts_duration = sampled_tts_duration
            reply_timing_margin = None
        else:
            reply_mode = (
                "NORMAL"
                if scenario_plan.scenario_reply_mode == "SHORT"
                else "LONG"
            )
            tts_duration = adaptive_tts_duration(
                scenario_plan.scenario_reply_mode,
                response_start_time=response_start_time,
                next_sign_start=scenario_plan.next_sign_start,
            )
            planned_end = response_start_time + tts_duration
            if scenario_plan.scenario_reply_mode == "SHORT":
                reply_timing_margin = (
                    scenario_plan.next_sign_start - planned_end
                )
            else:
                reply_timing_margin = (
                    planned_end - scenario_plan.next_sign_start
                )
        extension = max(0.0, float(final_reply_extension_seconds))
        base_tts_duration = tts_duration
        tts_duration += extension
        response = SimulatedResponse(
            response_id=response_id,
            response_token = response_token,
            complete_time=float(complete_time),
            thinking_time=thinking_time,
            response_start_time=response_start_time,
            tts_duration=tts_duration,
            planned_response_end_time=response_start_time + tts_duration,
            reply_mode=reply_mode,
            random_seed=self.random_seed,
            scenario_reply_mode=scenario_reply_mode,
            next_sign_start=next_sign_start,
            reply_timing_margin=reply_timing_margin,
            sampled_tts_duration=base_tts_duration,
            final_reply_extension_seconds=extension,
        )
        self._active = response
        self._latest = response
        self._responses.append(response)
        return response

    def poll(self, elapsed_seconds: float) -> list[SimulatedResponseEvent]:
        response = self._active
        if response is None:
            return []

        events: list[SimulatedResponseEvent] = []
        if (
            response.status == "THINKING"
            and elapsed_seconds >= response.response_start_time
        ):
            response.status = "SPEAKING"
            events.append(
                SimulatedResponseEvent(
                    "SYSTEM_RESPONSE_STARTED",
                    response.response_start_time,
                    response.response_id,
                    response.response_token,
                )
            )
        if (
            response.status == "SPEAKING"
            and elapsed_seconds >= response.planned_response_end_time
        ):
            response.status = "FINISHED"
            events.append(
                SimulatedResponseEvent(
                    "SYSTEM_RESPONSE_FINISHED",
                    response.planned_response_end_time,
                    response.response_id,
                    response.response_token,
                )
            )
            self._active = None
        return events

    def cancel(self, response_id: str) -> bool:
        response = self._active
        if response is None or response.response_id != response_id:
            return False
        if response.status not in {"THINKING", "SPEAKING"}:
            return False
        response.status = "CANCELLED_BY_BARGE"
        self._active = None
        return True

    def debug_record(self) -> dict[str, object]:
        if self._latest is None:
            return {
                "Reply Mode": "—",
                "Scenario Reply Mode": "—",
                "Final Reply Extension": "—",
                "Thinking Time": "—",
                "TTS Duration": "—",
                "Response Start Time": "—",
                "Planned Response End Time": "—",
                "Response Status": "IDLE",
            }
        return self._latest.as_debug_record()

    def counts(self) -> dict[str, int]:
        return {
            "response_scheduled_count": len(self._responses),
            "normal_reply_count": sum(
                response.reply_mode == "NORMAL" for response in self._responses
            ),
            "long_reply_count": sum(
                response.reply_mode == "LONG" for response in self._responses
            ),
            "planned_short_reply_count": sum(
                response.scenario_reply_mode == "SHORT"
                for response in self._responses
            ),
            "planned_long_reply_count": sum(
                response.scenario_reply_mode == "LONG"
                for response in self._responses
            ),
            "final_reply_extension_count": sum(
                response.final_reply_extension_seconds > 0
                for response in self._responses
            ),
            "response_cancelled_by_barge_count": sum(
                response.status == "CANCELLED_BY_BARGE"
                for response in self._responses
            ),
        }
