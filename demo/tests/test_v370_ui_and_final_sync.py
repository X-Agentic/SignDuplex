from __future__ import annotations

from pathlib import Path

import pytest

from app import format_test_video, video_display_name
from config import (
    FINAL_REPLY_EXTENSION_SECONDS,
    POST_EOS_DISPLAY_HOLD_SECONDS,
)
from runtime.demo_scenario import BalancedDemoScenarioPlanner
from runtime.subtitle_timeline import ReferenceSubtitleSegment
from runtime.tts_response_simulator import TTSResponseSimulator
from runtime.waveform_display import BACKCHANNEL_FEEDBACK_DURATION_RANGE


ROOT = Path(__file__).parents[1]
SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")


def segments() -> tuple[ReferenceSubtitleSegment, ...]:
    return (
        ReferenceSubtitleSegment(0, "a.mp4", "a", 0.0, 5.0, "A"),
        ReferenceSubtitleSegment(1, "b.mp4", "b", 10.2, 15.0, "B"),
        ReferenceSubtitleSegment(2, "c.mp4", "c", 20.4, 25.4, "C"),
    )


def test_case_1_video_player_is_compact_and_preserves_aspect_ratio():
    assert "#upload-video video" in SOURCE
    assert "max-height: 560px" in SOURCE
    assert "object-fit: contain" in SOURCE
    assert "with gr.Column(scale=11)" in SOURCE
    assert "with gr.Column(scale=9)" in SOURCE


def test_case_2_reference_note_removed_and_sentence_enlarged():
    assert "Reference annotation from CSL-Daily; not a model translation." not in SOURCE
    assert "#reference-subtitle" in SOURCE
    assert "font-size: 22px" in SOURCE
    assert "text-align: center" in SOURCE
    assert "min-height: 0" in SOURCE


def test_cases_3_and_4_test_video_name_is_visible_and_safe():
    path = "/tmp/gradio/a1b2/muted_concat_021_P0004.mp4"
    assert video_display_name(path) == "muted_concat_021_P0004.mp4"
    assert format_test_video(path) == (
        "**Test Video:** muted_concat_021_P0004.mp4"
    )
    assert "/tmp/gradio" not in format_test_video(path)
    assert 'elem_id="test-video-name"' in SOURCE
    assert "upload.change(\n        format_test_video" in SOURCE
    assert "outputs=[upload, test_video_name] + stream_outputs" in SOURCE
    assert '"video_display_name": video_display_name(prepared.path)' in SOURCE


def test_cases_5_and_6_only_final_reply_gets_three_second_extension():
    planner = BalancedDemoScenarioPlanner(segments(), random_seed=42)
    assert not planner.is_final_reply_opportunity(15.0)
    assert planner.is_final_reply_opportunity(25.4)

    baseline = TTSResponseSimulator(42).schedule("a", "ta", 25.4)
    extended = TTSResponseSimulator(42).schedule(
        "b",
        "tb",
        25.4,
        final_reply_extension_seconds=FINAL_REPLY_EXTENSION_SECONDS,
    )
    assert extended.sampled_tts_duration == pytest.approx(
        baseline.tts_duration
    )
    assert extended.tts_duration == pytest.approx(
        baseline.tts_duration + 3.0
    )
    assert extended.planned_response_end_time == pytest.approx(
        extended.response_start_time + extended.tts_duration
    )
    assert baseline.final_reply_extension_seconds == 0.0


def test_case_7_extension_keeps_full_waveform_until_response_finish():
    assert 'trigger="SYSTEM_RESPONSE_STARTED"' in SOURCE
    assert "duration=response.tts_duration" in SOURCE
    assert (
        'f"TTS_START:{run_id}:{lifecycle_event.response_id}:"' in SOURCE
    )
    finish = SOURCE.index(
        'elif lifecycle_event.event == "SYSTEM_RESPONSE_FINISHED"'
    )
    end = SOURCE.index("end_of_stream = controller.end_of_stream", finish)
    lifecycle = SOURCE[finish:end]
    assert 'trigger="SYSTEM_RESPONSE_FINISHED"' in lifecycle
    assert "final_response_finished_wall_time = wall_elapsed" in lifecycle
    assert "final_waveform_stop_wall_time = wall_elapsed" in lifecycle
    assert "final_waveform_sync_error_seconds < 0.1" in SOURCE


def test_case_8_no_quiet_hold_or_inference_during_final_extension():
    assert POST_EOS_DISPLAY_HOLD_SECONDS == 0.0
    assert "POST_EOS_HOLD" not in SOURCE
    assert '"event": "POST_EOS_DISPLAY_HOLD"' not in SOURCE
    started = SOURCE[
        SOURCE.index('if lifecycle_event.event == "SYSTEM_RESPONSE_STARTED"'):
        SOURCE.index(
            'elif lifecycle_event.event == "SYSTEM_RESPONSE_FINISHED"'
        )
    ]
    assert "lifecycle_event.response_id != final_response_id" in started
    assert "retire_worker()" in SOURCE


def test_balanced_backchannel_and_barge_contracts_remain_present():
    assert BACKCHANNEL_FEEDBACK_DURATION_RANGE == (0.8, 1.2)
    assert 'DEMO_SCENARIO_MODE = "BALANCED_IDLE_BARGE"' in (
        ROOT / "runtime/demo_scenario.py"
    ).read_text(encoding="utf-8")
    for required in (
        'controller_prediction == "BACKCHANNEL"',
        'controller_prediction == "BARGE"',
        'trigger="CANCELLED_BY_BARGE"',
        "stopForBarge: () => stopWithFade(65)",
    ):
        assert required in SOURCE


def test_v370_summary_contains_required_audit_fields():
    for field in (
        "video_display_name",
        "video_player_compact",
        "reference_annotation_note_visible",
        "reference_sentence_font_px",
        "final_reply_extension_seconds",
        "final_reply_extension_applied",
        "post_eos_quiet_hold_seconds",
        "final_waveform_synced_with_response",
        "final_waveform_sync_error_seconds",
    ):
        assert f'"{field}"' in SOURCE
