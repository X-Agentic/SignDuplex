from __future__ import annotations

from runtime.causal_sign_evidence import CausalSignEvidence
from runtime.interaction_controller import (
    BACKCHANNEL_COOLDOWN_SECONDS,
    POST_BARGE_PROTECTION_SECONDS,
    InteractionController,
)
from runtime.state_controller import action_for


def evidence(active: bool = False) -> CausalSignEvidence:
    return CausalSignEvidence(
        motion_score=0.01 if active else 0.0,
        hand_motion_score=0.01 if active else 0.0,
        recent_active_fraction=1.0 if active else 0.0,
        sustained_active_duration=0.5 if active else 0.0,
        hand_articulation_score=0.01 if active else 0.0,
        trajectory_score=0.1 if active else 0.0,
        articulation_active_fraction=1.0 if active else 0.0,
        causal_confirmation_score=1.0 if active else 0.0,
        causal_sign_evidence=active,
        current_motion_active=active,
        motion_episode_start_frame=0 if active else None,
    )


def decide(
    controller: InteractionController,
    state: str,
    prediction: str,
    decision_time: float,
):
    return controller.evaluate(
        state,
        prediction,
        0.9,
        evidence(),
        decision_time=decision_time,
        turn_has_sign_activity=False,
        turn_consumed=False,
    )


def test_default_controller_durations():
    assert POST_BARGE_PROTECTION_SECONDS == 2.0
    assert BACKCHANNEL_COOLDOWN_SECONDS == 3.0


def test_case_1_normal_barge_sequence():
    controller = InteractionController()
    sequence = (
        ("SPEAKING", "NO_BARGE", 19.0),
        ("SPEAKING", "NO_BARGE", 20.0),
        ("SPEAKING", "BARGE", 21.0),
        ("LISTENING", "INCOMPLETE", 22.0),
        ("LISTENING", "INCOMPLETE", 23.0),
        ("LISTENING", "COMPLETE", 24.0),
    )
    decisions = [decide(controller, *item) for item in sequence]
    assert [item.controller_prediction for item in decisions] == [
        "NO_BARGE",
        "NO_BARGE",
        "BARGE",
        "INCOMPLETE",
        "INCOMPLETE",
        "COMPLETE",
    ]
    assert [action_for(item.controller_prediction)[0] for item in decisions] == [
        "CONTINUE_SPEAKING",
        "CONTINUE_SPEAKING",
        "SWITCH_TO_LISTENING",
        "KEEP_LISTENING",
        "KEEP_LISTENING",
        "SWITCH_TO_SPEAKING",
    ]


def test_case_2_idle_is_suppressed_during_post_barge_recovery():
    controller = InteractionController()
    barge = decide(controller, "SPEAKING", "BARGE", 21.0)
    protected_idle = decide(controller, "LISTENING", "IDLE", 22.0)
    incomplete = decide(controller, "LISTENING", "INCOMPLETE", 23.0)

    assert barge.controller_prediction == "BARGE"
    assert barge.controller_state == "POST_BARGE_RECOVERY"
    assert protected_idle.raw_prediction == "IDLE"
    assert protected_idle.controller_prediction == "INCOMPLETE"
    assert protected_idle.controller_override_reason == "POST_BARGE_PROTECTION"
    assert protected_idle.controller_state == "POST_BARGE_RECOVERY"
    assert action_for(protected_idle.controller_prediction)[0] == "KEEP_LISTENING"
    assert incomplete.controller_prediction == "INCOMPLETE"


def test_post_barge_protection_expires_at_exactly_two_seconds():
    controller = InteractionController()
    decide(controller, "SPEAKING", "BARGE", 21.0)
    expired = decide(controller, "LISTENING", "IDLE", 23.0)
    assert expired.controller_prediction == "IDLE"
    assert expired.controller_override_reason == "NONE"
    assert expired.controller_state == "NORMAL"


def test_case_3_consecutive_backchannels_are_cooled_down():
    controller = InteractionController()
    first = decide(controller, "LISTENING", "BACKCHANNEL", 14.0)
    second = decide(controller, "LISTENING", "BACKCHANNEL", 15.0)
    third = decide(controller, "LISTENING", "BACKCHANNEL", 16.0)

    assert [first.controller_prediction, second.controller_prediction, third.controller_prediction] == [
        "BACKCHANNEL",
        "INCOMPLETE",
        "INCOMPLETE",
    ]
    assert first.controller_override_reason == "NONE"
    assert second.controller_override_reason == "BACKCHANNEL_COOLDOWN"
    assert third.controller_override_reason == "BACKCHANNEL_COOLDOWN"
    assert action_for(first.controller_prediction)[0] == "SHORT_FEEDBACK"
    assert action_for(second.controller_prediction)[0] == "KEEP_LISTENING"


def test_backchannel_cooldown_expires_at_exactly_three_seconds():
    controller = InteractionController()
    decide(controller, "LISTENING", "BACKCHANNEL", 14.0)
    expired = decide(controller, "LISTENING", "BACKCHANNEL", 17.0)
    assert expired.controller_prediction == "BACKCHANNEL"
    assert expired.controller_override_reason == "NONE"


def test_case_4_complete_is_never_blocked_by_backchannel_cooldown():
    controller = InteractionController()
    decide(controller, "LISTENING", "BACKCHANNEL", 14.0)
    incomplete = decide(controller, "LISTENING", "INCOMPLETE", 15.0)
    complete = decide(controller, "LISTENING", "COMPLETE", 16.0)
    assert incomplete.controller_prediction == "INCOMPLETE"
    assert complete.controller_prediction == "COMPLETE"
    assert complete.controller_override_reason == "NONE"
    assert action_for(complete.controller_prediction)[0] == "SWITCH_TO_SPEAKING"


def test_controller_never_mutates_raw_prediction_or_uses_pose_to_override():
    controller = InteractionController()
    decide(controller, "SPEAKING", "BARGE", 1.0)
    result = controller.evaluate(
        "LISTENING",
        "IDLE",
        0.8,
        evidence(active=True),
        decision_time=1.5,
        turn_has_sign_activity=True,
        turn_consumed=False,
    )
    assert result.raw_prediction == "IDLE"
    assert result.controller_prediction == "INCOMPLETE"
    assert isinstance(result.evidence.motion_score, float)


def test_controller_rejects_non_monotonic_decision_time():
    controller = InteractionController()
    decide(controller, "LISTENING", "IDLE", 2.0)
    try:
        decide(controller, "LISTENING", "IDLE", 1.0)
    except ValueError as error:
        assert "monotonic" in str(error)
    else:
        raise AssertionError("non-monotonic decision time was accepted")
