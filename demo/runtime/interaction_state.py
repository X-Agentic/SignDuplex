"""Automatic interaction-state transitions for the full-duplex demo."""

from __future__ import annotations

from dataclasses import dataclass

from runtime.state_controller import action_for


INITIAL_SYSTEM_STATE = "LISTENING"
INFERENCE_STATES = {"LISTENING", "SPEAKING"}

_NEXT_STATE = {
    "LISTENING": {
        "IDLE": "LISTENING",
        "INCOMPLETE": "LISTENING",
        "BACKCHANNEL": "LISTENING",
        "COMPLETE": "TURN_READY",
    },
    "SPEAKING": {
        "NO_BARGE": "SPEAKING",
        "BARGE": "LISTENING",
    },
}


@dataclass(frozen=True)
class InteractionTransition:
    state_before: str
    event: str
    action: str
    state_after: str


class InteractionStateManager:
    """Single source of truth for the automatic interaction state."""

    def __init__(self) -> None:
        self.current_system_state = INITIAL_SYSTEM_STATE

    def handle_prediction(self, prediction: str) -> InteractionTransition:
        state_before = self.current_system_state
        if state_before == "TURN_READY":
            raise RuntimeError(
                "Response must start before another prediction is routed"
            )
        try:
            state_after = _NEXT_STATE[state_before][prediction]
        except KeyError as error:
            raise RuntimeError(
                f"Invalid automatic transition: {state_before}/{prediction}"
            ) from error
        action, _ = action_for(prediction)
        self.current_system_state = state_after
        return InteractionTransition(
            state_before=state_before,
            event=prediction,
            action=action,
            state_after=state_after,
        )

    def start_response(self) -> InteractionTransition:
        if self.current_system_state != "TURN_READY":
            raise RuntimeError(
                "Response can start only after COMPLETE enters TURN_READY"
            )
        transition = InteractionTransition(
            state_before="TURN_READY",
            event="SYSTEM_RESPONSE_STARTED",
            action="START_OUTPUT",
            state_after="SPEAKING",
        )
        self.current_system_state = transition.state_after
        return transition

    def finish_response(self) -> InteractionTransition:
        if self.current_system_state != "SPEAKING":
            raise RuntimeError(
                "Response can finish only while the system is speaking"
            )
        transition = InteractionTransition(
            state_before="SPEAKING",
            event="SYSTEM_RESPONSE_FINISHED",
            action="FINISH_OUTPUT",
            state_after="LISTENING",
        )
        self.current_system_state = transition.state_after
        return transition

    def reset(self, event: str = "RESET") -> InteractionTransition:
        transition = InteractionTransition(
            state_before=self.current_system_state,
            event=event,
            action="RESET",
            state_after=INITIAL_SYSTEM_STATE,
        )
        self.current_system_state = transition.state_after
        return transition

    @property
    def inference_state(self) -> str:
        if self.current_system_state not in INFERENCE_STATES:
            raise RuntimeError(
                f"{self.current_system_state} is not an inference state"
            )
        return self.current_system_state
