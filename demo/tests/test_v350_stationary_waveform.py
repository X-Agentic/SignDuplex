from __future__ import annotations

from pathlib import Path

from runtime.causal_sign_evidence import CausalSignEvidence
from runtime.interaction_controller import InteractionController
from runtime.state_controller import action_for
from runtime.waveform_display import (
    BACKCHANNEL_FEEDBACK_DURATION_RANGE,
    BackchannelFeedbackSampler,
)


SOURCE = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")


def waveform_javascript() -> str:
    start = SOURCE.index("          const waveform = (() => {")
    end = SOURCE.index("        }\"\"\"", start)
    return SOURCE[start:end]


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


def decide(controller: InteractionController, prediction: str, time: float):
    return controller.evaluate(
        "LISTENING",
        prediction,
        0.9,
        evidence(),
        decision_time=time,
        turn_has_sign_activity=False,
        turn_consumed=False,
    )


def test_case_1_idle_and_incomplete_are_motionless_center_lines():
    assert SOURCE.count('d="M 0 70 L 640 70"') == 2
    script = waveform_javascript()
    assert 'state.mode = "QUIET";' in script
    assert 'draw(Array(anchorCount).fill(0), "QUIET")' in script
    assert "requestAnimationFrame" not in script[
        script.index("const toQuiet"):script.index("const stopWithFade")
    ]


def test_case_2_accepted_backchannel_is_visible_for_0_8_to_1_2_seconds():
    assert BACKCHANNEL_FEEDBACK_DURATION_RANGE == (0.8, 1.2)
    first = BackchannelFeedbackSampler(42)
    second = BackchannelFeedbackSampler(42)
    values_a = [first.sample().duration for _ in range(12)]
    values_b = [second.sample().duration for _ in range(12)]
    assert values_a == values_b
    assert all(0.8 <= value <= 1.2 for value in values_a)

    controller = InteractionController()
    result = decide(controller, "BACKCHANNEL", 1.0)
    assert result.controller_prediction == "BACKCHANNEL"
    assert action_for(result.controller_prediction)[0] == "SHORT_FEEDBACK"
    assert 'begin("SHORT_FEEDBACK", duration, token)' in SOURCE
    assert "const amplitudeScale = isFullReply ? 49 : 31;" in SOURCE


def test_backchannel_cooldown_guard_and_listening_state_are_unchanged():
    controller = InteractionController()
    accepted = decide(controller, "BACKCHANNEL", 1.0)
    suppressed = decide(controller, "BACKCHANNEL", 2.0)
    assert accepted.controller_prediction == "BACKCHANNEL"
    assert suppressed.raw_prediction == "BACKCHANNEL"
    assert suppressed.controller_prediction == "INCOMPLETE"
    assert suppressed.controller_override_reason == "BACKCHANNEL_COOLDOWN"
    assert "SWITCH_TO_SPEAKING" not in action_for(
        accepted.controller_prediction
    )
    trigger = SOURCE[SOURCE.index("backchannel_feedback_plan = None"):]
    assert 'controller_prediction == "BACKCHANNEL"' in trigger
    assert 'raw_prediction == "BACKCHANNEL"' not in trigger


def test_case_3_thinking_does_not_start_waveform_before_lifecycle_start():
    complete = SOURCE[SOURCE.index('if controller_prediction == "COMPLETE":'):]
    complete = complete[:complete.index('elif controller_prediction == "BARGE":')]
    assert "TTS_START:" not in complete
    assert 'lifecycle_event.event == "SYSTEM_RESPONSE_STARTED"' in SOURCE


def test_cases_4_and_5_full_reply_starts_and_finishes_on_lifecycle_events():
    assert 'trigger="SYSTEM_RESPONSE_STARTED"' in SOURCE
    assert 'f"TTS_START:{run_id}:{lifecycle_event.response_id}:"' in SOURCE
    assert 'trigger="SYSTEM_RESPONSE_FINISHED"' in SOURCE
    assert 'f"STOP:NORMAL:{run_id}:{lifecycle_event.response_id}"' in SOURCE
    assert "stopNormal: () => stopWithFade(185)" in SOURCE


def test_case_6_barge_uses_a_65_ms_fast_stop():
    assert 'trigger="CANCELLED_BY_BARGE"' in SOURCE
    assert 'f"STOP:BARGE:{run_id}:{aligned.record[\'response_id\']}"' in SOURCE
    assert "stopForBarge: () => stopWithFade(65)" in SOURCE
    assert 'signal.startsWith("STOP:BARGE:")' in SOURCE


def test_stationary_request_animation_frame_renderer_targets_60_fps():
    script = waveform_javascript()
    for expected in (
        "const anchorCount = 64;",
        "const targetFps = 60;",
        "const frameInterval = 1000 / targetFps;",
        "requestAnimationFrame(tick)",
        "draw(stationarySamples(timestamp), state.mode)",
        "index * width / (samples.length - 1)",
        "point.currentAmplitude +=",
        "* 0.12;",
        'transform="translate(0 140) scale(1 -1)"',
    ):
        assert expected in SOURCE
    for scrolling in (
        "state.samples.shift()",
        "state.samples.push(",
        "lastSampleAt",
        "timestamp - state.lastSampleAt >= 50",
    ):
        assert scrolling not in script


def test_waveform_has_smooth_deterministic_speech_like_modulation():
    script = waveform_javascript()
    assert "Math.random()" not in script
    for expected in (
        "unitNoise",
        "targetAmplitude",
        "currentAmplitude",
        "smoothstep",
        "fadeInSeconds",
        "fadeOutSeconds",
        "0.55 * Math.sin",
        "0.30 * Math.sin",
        "0.15 * Math.sin",
        "const amplitudeScale = isFullReply ? 49 : 31;",
    ):
        assert expected in script


def test_frontend_animation_is_state_only_and_does_not_block_inference():
    script = waveform_javascript()
    assert "requestAnimationFrame(tick)" in script
    assert "fetch(" not in script
    assert "gr.update" not in script
    assert "yield " not in script
    assert "setInterval(" not in script
    assert SOURCE.count("SHORT_FEEDBACK_START:") == 2
    assert SOURCE.count("TTS_START:") >= 2


def test_summary_reports_v350_animation_contract():
    for expected in (
        '"waveform_animation_engine": "FRONTEND_REQUEST_ANIMATION_FRAME"',
        '"waveform_target_fps": 60',
        '"waveform_scrolling": False',
        '"waveform_stationary_animation": True',
        '"backchannel_feedback_duration_range"',
    ):
        assert expected in SOURCE
