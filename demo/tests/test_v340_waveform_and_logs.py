from __future__ import annotations

from pathlib import Path

from runtime.causal_sign_evidence import CausalSignEvidence
from runtime.interaction_controller import InteractionController
from runtime.state_controller import action_for
from runtime.tts_response_simulator import TTSResponseSimulator
from runtime.waveform_display import (
    BACKCHANNEL_FEEDBACK_DURATION_RANGE,
    BackchannelFeedbackSampler,
)


SOURCE = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")


def evidence() -> CausalSignEvidence:
    return CausalSignEvidence(
        motion_score=0.0,
        hand_motion_score=0.0,
        recent_active_fraction=0.0,
        sustained_active_duration=0.0,
        hand_articulation_score=0.0,
        trajectory_score=0.0,
        articulation_active_fraction=0.0,
        causal_confirmation_score=0.0,
        causal_sign_evidence=False,
        current_motion_active=False,
        motion_episode_start_frame=None,
    )


def decide(
    controller: InteractionController,
    prediction: str,
    decision_time: float,
):
    return controller.evaluate(
        "LISTENING",
        prediction,
        0.9,
        evidence(),
        decision_time=decision_time,
        turn_has_sign_activity=False,
        turn_consumed=False,
    )


def test_case_1_idle_is_quiet_and_case_2_incomplete_is_quiet():
    assert 'data-mode="QUIET" d="M 0 70 L 640 70"' in SOURCE
    trigger_guard = SOURCE[SOURCE.index("backchannel_feedback_plan = None"):]
    assert 'controller_prediction == "BACKCHANNEL"' in trigger_guard
    assert 'action == "SHORT_FEEDBACK"' in trigger_guard
    assert "SHORT_FEEDBACK_START" in SOURCE
    assert "IDLE_FEEDBACK_START" not in SOURCE
    assert "INCOMPLETE_FEEDBACK_START" not in SOURCE


def test_case_3_accepted_backchannel_gets_seeded_short_feedback_in_listening():
    assert BACKCHANNEL_FEEDBACK_DURATION_RANGE == (0.8, 1.2)
    first = BackchannelFeedbackSampler(42)
    second = BackchannelFeedbackSampler(42)
    durations_a = [first.sample().duration for _ in range(5)]
    durations_b = [second.sample().duration for _ in range(5)]
    assert durations_a == durations_b
    assert all(0.8 <= value <= 1.2 for value in durations_a)

    controller = InteractionController()
    accepted = decide(controller, "BACKCHANNEL", 1.0)
    assert accepted.controller_prediction == "BACKCHANNEL"
    assert action_for(accepted.controller_prediction)[0] == "SHORT_FEEDBACK"
    assert action_for(accepted.controller_prediction)[0] != "SWITCH_TO_SPEAKING"
    assert 'waveform_mode="SHORT_FEEDBACK"' in SOURCE
    assert 'trigger="BACKCHANNEL"' in SOURCE


def test_backchannel_sampler_does_not_change_existing_tts_random_sequence():
    baseline = TTSResponseSimulator(42).schedule("r1", "t1", 1.0)
    sampler = BackchannelFeedbackSampler(42)
    for _ in range(4):
        sampler.sample()
    after_feedback = TTSResponseSimulator(42).schedule("r1", "t1", 1.0)
    assert after_feedback.thinking_time == baseline.thinking_time
    assert after_feedback.reply_mode == baseline.reply_mode
    assert after_feedback.tts_duration == baseline.tts_duration


def test_case_4_cooldown_suppression_does_not_trigger_second_waveform():
    controller = InteractionController()
    first = decide(controller, "BACKCHANNEL", 1.0)
    suppressed = decide(controller, "BACKCHANNEL", 2.0)
    assert first.controller_prediction == "BACKCHANNEL"
    assert suppressed.raw_prediction == "BACKCHANNEL"
    assert suppressed.controller_prediction == "INCOMPLETE"
    assert suppressed.controller_override_reason == "BACKCHANNEL_COOLDOWN"
    assert 'controller_prediction == "BACKCHANNEL"' in SOURCE
    assert 'raw_prediction == "BACKCHANNEL"' not in SOURCE


def test_case_5_complete_thinking_does_not_start_full_reply_early():
    complete_branch = SOURCE[SOURCE.index('if controller_prediction == "COMPLETE":'):]
    complete_branch = complete_branch[:complete_branch.index('elif controller_prediction == "BARGE":')]
    assert "TTS_START:" not in complete_branch
    assert 'lifecycle_event.event == "SYSTEM_RESPONSE_STARTED"' in SOURCE
    assert 'waveform_mode="FULL_REPLY"' in SOURCE
    assert 'trigger="SYSTEM_RESPONSE_STARTED"' in SOURCE


def test_case_6_full_reply_uses_natural_stationary_speech_envelope():
    for expected in (
        "buildPoints",
        "smoothstep",
        "fadeInSeconds",
        "fadeOutSeconds",
        "0.55 * Math.sin",
        "0.30 * Math.sin",
        "0.15 * Math.sin",
        "point.currentAmplitude +=",
        "draw(stationarySamples(timestamp), state.mode)",
        "const targetFps = 60",
        " Q ",
    ):
        assert expected in SOURCE
    assert "state.samples.shift()" not in SOURCE
    assert "Math.random()" not in SOURCE
    assert 'begin("FULL_REPLY", duration, token)' in SOURCE
    assert 'trigger="SYSTEM_RESPONSE_FINISHED"' in SOURCE
    assert "waveform.stopNormal();" in SOURCE


def test_case_7_barge_immediately_stops_full_reply_without_fade_delay():
    barge = SOURCE[SOURCE.index('if aligned.prediction == "BARGE":'):]
    barge = barge[:barge.index('elif (\n                    aligned.prediction == "BACKCHANNEL"')]
    assert 'trigger="CANCELLED_BY_BARGE"' in barge
    assert 'f"STOP:BARGE:{run_id}:{aligned.record[\'response_id\']}"' in barge
    stop_handler = SOURCE[SOURCE.index('signal.startsWith("STOP:")'):]
    assert "waveform.stopForBarge();" in stop_handler
    assert "stopForBarge: () => stopWithFade(65)" in SOURCE


def test_waveform_priority_is_full_reply_then_short_feedback_then_quiet():
    assert 'if (state.mode === "FULL_REPLY") { return; }' in SOURCE
    assert "startFullReply:" in SOURCE
    assert "startShortFeedback:" in SOURCE
    assert "stopNormal:" in SOURCE
    assert "stopForBarge:" in SOURCE


def test_waveform_event_log_has_required_fields_and_time_domain():
    assert 'TimelineLogger(LOGS_DIR, run_id, "waveform_timeline")' in SOURCE
    assert 'print("[waveform-event] "' in SOURCE
    for field in (
        '"event"',
        '"waveform_mode"',
        '"trigger"',
        '"start_time"',
        '"planned_end_time"',
        '"actual_end_time"',
        '"response_id"',
        '"event_time_domain": "INTERACTION"',
        '"video_time"',
        '"interaction_wall_time"',
    ):
        assert field in SOURCE


def test_summary_semantics_and_end_of_stream_time_are_unambiguous():
    assert '"controller_changes_model_prediction": False' not in SOURCE
    assert '"controller_preserves_raw_prediction": True' in SOURCE
    assert '"raw_prediction_preserved": True' in SOURCE
    assert 'event_time_domain="INTERACTION"' in SOURCE
    assert 'video_time=prepared.duration' in SOURCE
    assert '"interaction_wall_time": end_interaction_wall_time' in SOURCE


def test_reference_sentence_and_prediction_action_mapping_remain_independent():
    assert "Prediction-Action Mapping" not in SOURCE
    publish = SOURCE.index('"event": "REFERENCE_SENTENCE_PUBLISHED"')
    guard = SOURCE.rfind("if (", 0, publish)
    branch = SOURCE[guard:publish]
    assert 'aligned.prediction == "COMPLETE"' in branch
    assert "waveform" not in branch.lower()
