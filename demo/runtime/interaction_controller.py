"""Stateful interaction constraints layered after the V1.8 model prediction."""

from __future__ import annotations

from dataclasses import dataclass

from runtime.causal_sign_evidence import CausalSignEvidence
from runtime.interaction_state import InteractionStateManager, InteractionTransition


POST_BARGE_PROTECTION_SECONDS = 2.0
BACKCHANNEL_COOLDOWN_SECONDS = 3.0
POST_BARGE_RECOVERY = "POST_BARGE_RECOVERY"
NORMAL_CONTROLLER_STATE = "NORMAL"


@dataclass(frozen=True)
class ControllerDecision:
    raw_prediction: str
    raw_confidence: float
    controller_prediction: str
    controller_override_reason: str
    controller_override: str
    previous_effective_prediction: str | None
    decision_time: float
    controller_state: str
    post_barge_active: bool
    post_barge_until: float | None
    backchannel_cooldown_active: bool
    backchannel_cooldown_until: float | None
    turn_has_sign_activity: bool
    turn_consumed: bool
    evidence: CausalSignEvidence

    @property
    def effective_prediction(self) -> str:
        """Compatibility alias for downstream state-transition code."""
        return self.controller_prediction

    @property
    def controller_decision(self) -> str:
        return self.controller_prediction

    @property
    def override_reason(self) -> str:
        return self.controller_override_reason

    @property
    def idle_source(self) -> str | None:
        if self.controller_prediction == "IDLE":
            return "MODEL"
        return None

    @property
    def idle_learned(self) -> bool:
        return self.raw_prediction == "IDLE"

    @property
    def activity_label(self) -> str:
        return self.evidence.activity_label


class InteractionController:
    """Validate actions without mutating the model's raw prediction record."""

    def __init__(
        self,
        *,
        post_barge_seconds: float = POST_BARGE_PROTECTION_SECONDS,
        backchannel_cooldown_seconds: float = BACKCHANNEL_COOLDOWN_SECONDS,
    ) -> None:
        if post_barge_seconds < 0 or backchannel_cooldown_seconds < 0:
            raise ValueError("Controller durations must be non-negative")
        self.post_barge_seconds = float(post_barge_seconds)
        self.backchannel_cooldown_seconds = float(backchannel_cooldown_seconds)
        self.reset()

    def reset(self) -> None:
        self.previous_effective_prediction: str | None = None
        self.post_barge_until: float | None = None
        self.backchannel_cooldown_until: float | None = None
        self._last_decision_time: float | None = None
        self.controller_state = NORMAL_CONTROLLER_STATE

    def evaluate(
        self,
        system_state: str,
        raw_prediction: str,
        raw_confidence: float,
        evidence: CausalSignEvidence,
        *,
        decision_time: float = 0.0,
        turn_has_sign_activity: bool,
        turn_consumed: bool,
    ) -> ControllerDecision:
        now = float(decision_time)
        if now < 0:
            raise ValueError("decision_time must be non-negative")
        if self._last_decision_time is not None and now < self._last_decision_time:
            raise ValueError("decision_time must be monotonic")
        self._last_decision_time = now

        allowed = {
            "LISTENING": {"IDLE", "INCOMPLETE", "BACKCHANNEL", "COMPLETE"},
            "SPEAKING": {"BARGE", "NO_BARGE"},
        }
        if raw_prediction not in allowed.get(system_state, set()):
            raise RuntimeError(f"Invalid routed prediction: {system_state}/{raw_prediction}")

        previous = self.previous_effective_prediction
        post_barge_active = (
            system_state == "LISTENING"
            and self.post_barge_until is not None
            and now < self.post_barge_until
        )
        cooldown_active = (
            system_state == "LISTENING"
            and self.backchannel_cooldown_until is not None
            and now < self.backchannel_cooldown_until
        )

        controller_prediction = raw_prediction
        reason = "NONE"

        # Priority: BARGE > COMPLETE > INCOMPLETE > BACKCHANNEL > IDLE.
        if raw_prediction == "BARGE":
            self.post_barge_until = now + self.post_barge_seconds
            post_barge_active = self.post_barge_seconds > 0
        elif raw_prediction in {"COMPLETE", "INCOMPLETE", "NO_BARGE"}:
            pass
        elif raw_prediction == "BACKCHANNEL":
            if cooldown_active:
                controller_prediction = "INCOMPLETE"
                reason = "BACKCHANNEL_COOLDOWN"
            else:
                self.backchannel_cooldown_until = now + self.backchannel_cooldown_seconds
                cooldown_active = self.backchannel_cooldown_seconds > 0
        elif raw_prediction == "IDLE" and post_barge_active:
            controller_prediction = "INCOMPLETE"
            reason = "POST_BARGE_PROTECTION"

        override = (
            "NONE"
            if controller_prediction == raw_prediction
            else f"{raw_prediction} → {controller_prediction}"
        )
        self.controller_state = (
            POST_BARGE_RECOVERY
            if self.post_barge_until is not None and now < self.post_barge_until
            else NORMAL_CONTROLLER_STATE
        )
        self.previous_effective_prediction = controller_prediction
        return ControllerDecision(
            raw_prediction=raw_prediction,
            raw_confidence=max(0.0, float(raw_confidence)),
            controller_prediction=controller_prediction,
            controller_override_reason=reason,
            controller_override=override,
            previous_effective_prediction=previous,
            decision_time=now,
            controller_state=self.controller_state,
            post_barge_active=post_barge_active,
            post_barge_until=self.post_barge_until,
            backchannel_cooldown_active=cooldown_active,
            backchannel_cooldown_until=self.backchannel_cooldown_until,
            turn_has_sign_activity=bool(turn_has_sign_activity),
            turn_consumed=bool(turn_consumed),
            evidence=evidence,
        )

    def end_of_stream(
        self, state_manager: InteractionStateManager
    ) -> InteractionTransition:
        self.reset()
        return state_manager.reset("END_OF_STREAM")
