"""State-relative causal context for one user turn at a time."""

from __future__ import annotations

from dataclasses import dataclass

from runtime.causal_sign_evidence import CausalSignEvidence


@dataclass(frozen=True)
class TurnContextSnapshot:
    turn_start_frame: int
    turn_start_time: float
    state_enter_frame: int
    state_enter_time: float
    turn_has_sign_activity: bool
    turn_consumed: bool
    current_motion_episode_start: int | None


class TurnContextManager:
    """Prevent completed prefixes from leaking into a later state or turn."""

    def __init__(self, fps: float) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = float(fps)
        self.turn_start_frame = 0
        self.turn_start_time = 0.0
        self.state_enter_frame = 0
        self.state_enter_time = 0.0
        self.turn_has_sign_activity = False
        self.turn_consumed = False
        self.current_motion_episode_start: int | None = None

    def observe(self, evidence: CausalSignEvidence) -> None:
        if evidence.causal_sign_evidence:
            self.turn_has_sign_activity = True
        self.current_motion_episode_start = (
            evidence.motion_episode_start_frame
            if evidence.current_motion_active
            else None
        )

    def consume_turn(self) -> None:
        self.turn_consumed = True

    def enter_turn_ready(self, frame: int, time_seconds: float) -> None:
        self.state_enter_frame = int(frame)
        self.state_enter_time = float(time_seconds)

    def enter_speaking(
        self,
        frame: int,
        time_seconds: float,
        evidence: CausalSignEvidence,
    ) -> None:
        self.state_enter_frame = int(frame)
        self.state_enter_time = float(time_seconds)
        onset = (
            evidence.motion_episode_start_frame
            if evidence.current_motion_active
            else int(frame)
        )
        self.turn_start_frame = max(0, min(onset, int(frame)))
        self.turn_start_time = self.turn_start_frame / self.fps
        self.turn_has_sign_activity = bool(evidence.causal_sign_evidence)
        self.turn_consumed = False
        self.current_motion_episode_start = (
            evidence.motion_episode_start_frame
            if evidence.current_motion_active
            else None
        )

    def enter_listening(
        self,
        frame: int,
        time_seconds: float,
        evidence: CausalSignEvidence,
    ) -> None:
        self.state_enter_frame = int(frame)
        self.state_enter_time = float(time_seconds)
        onset = (
            evidence.motion_episode_start_frame
            if evidence.current_motion_active
            else int(frame)
        )
        self.turn_start_frame = max(0, min(onset, int(frame)))
        self.turn_start_time = self.turn_start_frame / self.fps
        self.turn_has_sign_activity = bool(evidence.causal_sign_evidence)
        self.turn_consumed = False
        self.current_motion_episode_start = (
            evidence.motion_episode_start_frame
            if evidence.current_motion_active
            else None
        )

    def task_prefix_start_frame(self, system_state: str) -> int:
        if system_state == "LISTENING":
            return self.turn_start_frame
        if system_state == "SPEAKING":
            return self.state_enter_frame
        raise RuntimeError(f"No inference prefix for state {system_state}")

    def snapshot(self) -> TurnContextSnapshot:
        return TurnContextSnapshot(
            turn_start_frame=self.turn_start_frame,
            turn_start_time=self.turn_start_time,
            state_enter_frame=self.state_enter_frame,
            state_enter_time=self.state_enter_time,
            turn_has_sign_activity=self.turn_has_sign_activity,
            turn_consumed=self.turn_consumed,
            current_motion_episode_start=self.current_motion_episode_start,
        )
