from __future__ import annotations

from pathlib import Path

from config import (
    CHUNK_SIZE_SECONDS,
    DECISION_INTERVAL_SECONDS,
    FINAL_REPLY_EXTENSION_SECONDS,
    POST_EOS_DISPLAY_HOLD_SECONDS,
)
from runtime.demo_scenario import (
    DEMO_SCENARIO_MODE,
    EXPECTED_PAUSE_DURATION_RANGE,
    BalancedDemoScenarioPlanner,
    validate_segment_pauses,
)
from runtime.subtitle_timeline import ReferenceSubtitleSegment
from runtime.tts_response_simulator import TTSResponseSimulator


ROOT = Path(__file__).parents[1]
SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")


def segments_with_pauses(
    first_pause: float = 5.3,
    second_pause: float = 5.6,
) -> tuple[ReferenceSubtitleSegment, ...]:
    first_end = 5.9
    second_start = first_end + first_pause
    second_end = second_start + 5.7
    third_start = second_end + second_pause
    return (
        ReferenceSubtitleSegment(0, "a.mp4", "a", 0.0, first_end, "A"),
        ReferenceSubtitleSegment(
            1,
            "b.mp4",
            "b",
            second_start,
            second_end,
            "B",
        ),
        ReferenceSubtitleSegment(
            2,
            "c.mp4",
            "c",
            third_start,
            third_start + 5.0,
            "C",
        ),
    )


def test_case_1_decision_interval_is_fixed_at_one_second_without_selector():
    assert CHUNK_SIZE_SECONDS == 1.0
    assert DECISION_INTERVAL_SECONDS == 1.0
    assert "DECISION_INTERVALS_SECONDS" not in SOURCE
    assert "DEFAULT_DECISION_INTERVAL_SECONDS" not in SOURCE
    assert "decision_interval = gr.Radio" not in SOURCE
    assert 'label="Decision Interval"' not in SOURCE
    assert "inputs=[upload, runtime]" in SOURCE
    assert "inputs=[upload, runtime, decision_interval]" not in SOURCE


def test_metadata_pause_validation_uses_actual_segment_boundaries():
    records = validate_segment_pauses(segments_with_pauses())
    assert EXPECTED_PAUSE_DURATION_RANGE == (5.0, 6.0)
    assert [record.pause_duration for record in records] == [5.3, 5.6]
    assert [record.status for record in records] == ["PASS", "PASS"]

    warnings = validate_segment_pauses(segments_with_pauses(4.13, 6.2))
    assert [record.status for record in warnings] == ["WARNING", "WARNING"]
    assert "[demo-validation]" in SOURCE
    assert {
        "segment_end",
        "next_segment_start",
        "pause_duration",
        "expected_range",
        "status",
    }.issubset(records[0].as_record())


def test_balanced_mode_plans_one_short_and_one_long_reproducibly():
    first = BalancedDemoScenarioPlanner(segments_with_pauses(), 42)
    second = BalancedDemoScenarioPlanner(segments_with_pauses(), 42)
    assert DEMO_SCENARIO_MODE == "BALANCED_IDLE_BARGE"
    assert first.planned_reply_modes == second.planned_reply_modes
    assert sorted(first.planned_reply_modes) == ["LONG", "SHORT"]
    validation = first.validation()
    assert validation.expected_idle_opportunity_count == 1
    assert validation.expected_barge_opportunity_count == 1


def test_cases_2_and_3_adaptive_short_and_long_create_real_opportunities():
    planner = BalancedDemoScenarioPlanner(segments_with_pauses(), 42)
    simulator = TTSResponseSimulator(42)
    plans = []
    complete_times = [6.0, 17.0]
    for index, complete_time in enumerate(complete_times):
        scenario = planner.reserve(complete_time)
        assert scenario is not None
        response = simulator.schedule(
            f"r{index}",
            f"token{index}",
            complete_time,
            scenario_plan=scenario,
        )
        plans.append(response)
        simulator.poll(response.planned_response_end_time)

    short = next(
        response
        for response in plans
        if response.scenario_reply_mode == "SHORT"
    )
    long = next(
        response
        for response in plans
        if response.scenario_reply_mode == "LONG"
    )
    assert short.next_sign_start is not None
    assert long.next_sign_start is not None
    assert short.planned_response_end_time < short.next_sign_start
    assert long.planned_response_end_time > long.next_sign_start
    assert short.reply_timing_margin is not None
    assert long.reply_timing_margin is not None
    assert short.reply_timing_margin > 0
    assert long.reply_timing_margin >= 1.75


def test_scenario_changes_only_tts_timing_and_never_forces_predictions():
    scenario_source = (
        ROOT / "runtime/demo_scenario.py"
    ).read_text(encoding="utf-8")
    for label in ('"IDLE"', '"BARGE"'):
        assert label not in scenario_source
    assert "scenario_planner.reserve(decision_time)" in SOURCE
    assert "scenario_plan=scenario_plan" in SOURCE
    assert "raw_prediction = result[\"predicted_class\"]" in SOURCE
    assert "controller_result = controller.evaluate(" in SOURCE
    assert "controller_prediction = controller_result.controller_prediction" in SOURCE


def test_case_4_summary_audits_observed_labels_without_rewriting_them():
    assert 'idle_observed = any(row[1] == "IDLE"' in SOURCE
    assert 'barge_observed = any(row[1] == "BARGE"' in SOURCE
    assert "balanced_demo_success = idle_observed and barge_observed" in SOURCE
    for expected in (
        '"idle_observed": idle_observed',
        '"barge_observed": barge_observed',
        '"balanced_demo_success": balanced_demo_success',
        '"planned_short_reply_count"',
        '"planned_long_reply_count"',
    ):
        assert expected in SOURCE or expected in (
            ROOT / "runtime/tts_response_simulator.py"
        ).read_text(encoding="utf-8")


def test_case_5_final_reply_extends_tts_instead_of_post_eos_hold():
    assert FINAL_REPLY_EXTENSION_SECONDS == 3.0
    assert POST_EOS_DISPLAY_HOLD_SECONDS == 0.0
    assert "POST_EOS_HOLD" not in SOURCE
    assert '"event": "POST_EOS_DISPLAY_HOLD"' not in SOURCE
    assert "final_reply_extension_seconds=final_extension" in SOURCE
    assert '"final_reply_extension_applied": final_reply_extension_applied' in SOURCE


def test_case_6_final_extension_performs_no_new_inference():
    response_start = SOURCE.index(
        'if lifecycle_event.event == "SYSTEM_RESPONSE_STARTED"'
    )
    response_finish = SOURCE.index(
        'elif lifecycle_event.event == "SYSTEM_RESPONSE_FINISHED"',
        response_start,
    )
    started = SOURCE[response_start:response_finish]
    assert "lifecycle_event.response_id != final_response_id" in started
    assert 'create_worker("SPEAKING")' in started
    assert "retire_worker()" in SOURCE


def test_final_waveform_stops_with_response_finish():
    finish = SOURCE.index(
        'elif lifecycle_event.event == "SYSTEM_RESPONSE_FINISHED"'
    )
    end = SOURCE.index("end_of_stream = controller.end_of_stream", finish)
    lifecycle = SOURCE[finish:end]
    assert 'waveform_mode="FULL_REPLY"' in lifecycle
    assert 'trigger="SYSTEM_RESPONSE_FINISHED"' in lifecycle
    assert "final_response_finished_wall_time = wall_elapsed" in lifecycle
    assert "final_waveform_stop_wall_time = wall_elapsed" in lifecycle
