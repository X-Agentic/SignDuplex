from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from config import (
    BARGE_LABELS,
    CHUNK_SIZE_SECONDS,
    DECISION_INTERVAL_SECONDS,
    DISPLAY_DELAY_MAX_SECONDS,
    DISPLAY_DELAY_MIN_SECONDS,
    TURN_LABELS,
)
from runtime.model_runtime import prepare_pose_prefix
from runtime.playback_alignment import (
    display_delay_from_latencies,
    percentile_95,
)
from runtime.async_worker import InferenceRequest, LatestOnlyInferenceWorker
from runtime.causal_sign_evidence import (
    CAUSAL_ACTIVE_DURATION_MIN_SECONDS,
    CONFIRMATION_SCORE_MIN,
    RECENT_ACTIVE_FRACTION_MIN,
    CausalSignEvidence,
    causal_sign_evidence,
)
from runtime.interaction_controller import InteractionController
from runtime.interaction_state import InteractionStateManager
from runtime.response_lifecycle import ResponseLifecycleRegistry
from runtime.state_controller import action_for, assert_routed_label
from runtime.timeline import TimelineLogger
from runtime.tts_response_simulator import (
    LONG_TTS_DURATION_RANGE,
    NORMAL_TTS_DURATION_RANGE,
    THINKING_TIME_RANGE,
    TTSResponseSimulator,
)
from runtime.turn_context import TurnContextManager
from runtime.video_stream import uniform_indices
from runtime.session_state import can_start, empty_state, runtime_state, transition
from runtime.stream_cache import (
    DecisionBoundary,
    PreparedVideo,
    PreparedVideoRegistry,
    build_decisions,
)


def test_training_sampler_parity():
    assert uniform_indices(4, 8) == [0, 1, 2, 3]
    assert uniform_indices(10, 4) == [0, 3, 6, 9]
    assert max(uniform_indices(400, 300)) == 399
    assert len(uniform_indices(400, 300)) == 300


def test_pose_prefix_has_training_parts_and_length():
    pose = {
        "keypoints": [np.zeros((1, 133, 2), np.float32) for _ in range(4)],
        "scores": [np.zeros((1, 133), np.float32) for _ in range(4)],
    }
    result = prepare_pose_prefix(pose)
    assert set(result) == {"body", "left", "right", "face_all"}
    assert result["body"].shape == (4, 9, 3)
    assert result["left"].shape == (4, 21, 3)


def test_factorized_routes_and_actions():
    assert TURN_LABELS == ("IDLE", "INCOMPLETE", "BACKCHANNEL", "COMPLETE")
    assert BARGE_LABELS == ("BARGE", "NO_BARGE")
    assert len(TURN_LABELS) == 4
    assert len(BARGE_LABELS) == 2
    for label in TURN_LABELS:
        assert_routed_label("LISTENING", label)
        assert action_for(label)[0]
    for label in BARGE_LABELS:
        assert_routed_label("SPEAKING", label)
        assert action_for(label)[0]


def test_automatic_interaction_state_flow():
    manager = InteractionStateManager()
    assert manager.current_system_state == "LISTENING"

    for prediction, action in (
        ("IDLE", "KEEP_LISTENING"),
        ("INCOMPLETE", "KEEP_LISTENING"),
        ("BACKCHANNEL", "SHORT_FEEDBACK"),
    ):
        event = manager.handle_prediction(prediction)
        assert event.state_before == "LISTENING"
        assert event.action == action
        assert event.state_after == "LISTENING"

    complete = manager.handle_prediction("COMPLETE")
    assert complete.action == "SWITCH_TO_SPEAKING"
    assert complete.state_after == "TURN_READY"
    assert manager.current_system_state == "TURN_READY"

    response = manager.start_response()
    assert response.event == "SYSTEM_RESPONSE_STARTED"
    assert response.state_before == "TURN_READY"
    assert response.state_after == "SPEAKING"
    assert manager.current_system_state == "SPEAKING"

    finished = manager.finish_response()
    assert finished.event == "SYSTEM_RESPONSE_FINISHED"
    assert finished.state_after == "LISTENING"

    manager.handle_prediction("COMPLETE")
    manager.start_response()

    no_barge = manager.handle_prediction("NO_BARGE")
    assert no_barge.action == "CONTINUE_SPEAKING"
    assert no_barge.state_after == "SPEAKING"

    barge = manager.handle_prediction("BARGE")
    assert barge.action == "SWITCH_TO_LISTENING"
    assert barge.state_after == "LISTENING"
    assert manager.current_system_state == "LISTENING"


def test_complete_cannot_skip_turn_ready():
    manager = InteractionStateManager()
    complete = manager.handle_prediction("COMPLETE")
    assert (complete.state_before, complete.state_after) == (
        "LISTENING",
        "TURN_READY",
    )
    try:
        manager.handle_prediction("NO_BARGE")
    except RuntimeError:
        pass
    else:
        raise AssertionError("TURN_READY accepted inference before response start")


def _pose_sequence(offsets, *, articulate=False):
    keypoints = []
    scores = []
    for offset in offsets:
        frame = np.zeros((1, 133, 2), np.float32)
        frame[0, 91:133, 0] = offset
        frame[0, 9:11, 0] = offset
        if articulate:
            frame[0, 92:112, 1] = offset * 0.7
            frame[0, 113:133, 1] = offset * 0.5
        keypoints.append(frame)
        scores.append(np.ones((1, 133), np.float32))
    return keypoints, scores


def _evidence(active: bool, onset: int | None = 10) -> CausalSignEvidence:
    return CausalSignEvidence(
        motion_score=0.01 if active else 0.0,
        hand_motion_score=0.01 if active else 0.0,
        recent_active_fraction=0.9 if active else 0.0,
        sustained_active_duration=0.4 if active else 0.0,
        hand_articulation_score=0.005 if active else 0.0,
        trajectory_score=0.1 if active else 0.0,
        articulation_active_fraction=0.8 if active else 0.0,
        causal_confirmation_score=0.8 if active else 0.0,
        causal_sign_evidence=active,
        current_motion_active=active,
        motion_episode_start_frame=onset if active else None,
    )


def test_v16_aligned_evidence_separates_static_and_articulated_signing():
    frozen_keypoints, frozen_scores = _pose_sequence([0.2] * 60)
    active_keypoints, active_scores = _pose_sequence(
        [(index / 60) ** 2 * 2.0 for index in range(60)],
        articulate=True,
    )
    frozen = causal_sign_evidence(
        frozen_keypoints, frozen_scores, 0, 60, 30.0
    )
    active = causal_sign_evidence(
        active_keypoints, active_scores, 0, 60, 30.0
    )
    assert not frozen.causal_sign_evidence
    assert active.causal_sign_evidence
    assert active.recent_active_fraction >= RECENT_ACTIVE_FRACTION_MIN
    assert active.sustained_active_duration >= CAUSAL_ACTIVE_DURATION_MIN_SECONDS
    assert active.causal_confirmation_score >= CONFIRMATION_SCORE_MIN


def test_causal_evidence_never_reads_future_frames():
    keypoints, scores = _pose_sequence([0.2] * 60)
    baseline = causal_sign_evidence(keypoints, scores, 0, 60, 30.0)
    future_keypoints, future_scores = _pose_sequence(
        [0.2] * 60 + [index * 0.3 for index in range(1, 16)],
        articulate=True,
    )
    future = causal_sign_evidence(
        future_keypoints, future_scores, 0, 60, 30.0
    )
    assert future == baseline


def test_controller_never_rewrites_model_barge_prediction():
    controller = InteractionController()
    blocked = controller.evaluate(
        "SPEAKING",
        "BARGE",
        0.999,
        _evidence(False),
        turn_has_sign_activity=False,
        turn_consumed=False,
    )
    assert blocked.effective_prediction == "BARGE"
    assert blocked.override_reason == "NONE"
    allowed = controller.evaluate(
        "SPEAKING",
        "BARGE",
        0.10,
        _evidence(True),
        turn_has_sign_activity=True,
        turn_consumed=False,
    )
    assert allowed.effective_prediction == "BARGE"


def test_controller_never_synthesizes_idle_or_rewrites_complete():
    controller = InteractionController()
    held_complete = controller.evaluate(
        "LISTENING",
        "COMPLETE",
        0.99,
        _evidence(False),
        turn_has_sign_activity=True,
        turn_consumed=False,
    )
    assert held_complete.effective_prediction == "COMPLETE"
    empty_turn = controller.evaluate(
        "LISTENING",
        "COMPLETE",
        0.99,
        _evidence(False),
        turn_has_sign_activity=False,
        turn_consumed=False,
    )
    assert empty_turn.effective_prediction == "COMPLETE"
    assert empty_turn.controller_decision == "COMPLETE"
    assert empty_turn.override_reason == "NONE"
    assert empty_turn.controller_override == "NONE"
    assert empty_turn.idle_source is None
    assert not empty_turn.idle_learned
    consumed = controller.evaluate(
        "LISTENING",
        "COMPLETE",
        0.99,
        _evidence(False),
        turn_has_sign_activity=True,
        turn_consumed=True,
    )
    assert consumed.effective_prediction == "COMPLETE"
    assert consumed.override_reason == "NONE"


def test_controller_is_pass_through_for_every_listening_prediction():
    controller = InteractionController()
    for raw_prediction in ("COMPLETE", "INCOMPLETE", "BACKCHANNEL"):
        decision = controller.evaluate(
            "LISTENING",
            raw_prediction,
            0.99,
            _evidence(False),
            turn_has_sign_activity=False,
            turn_consumed=False,
        )
        assert decision.raw_prediction == raw_prediction
        assert decision.effective_prediction == raw_prediction
        assert decision.controller_override == "NONE"
        assert decision.override_reason == "NONE"

    learned_idle = controller.evaluate(
        "LISTENING",
        "IDLE",
        0.99,
        _evidence(False),
        turn_has_sign_activity=False,
        turn_consumed=False,
    )
    assert learned_idle.effective_prediction == "IDLE"
    assert learned_idle.idle_source == "MODEL"
    assert learned_idle.idle_learned


def test_sign_hold_never_becomes_idle_after_turn_has_started():
    controller = InteractionController()
    for raw_prediction in ("COMPLETE", "INCOMPLETE", "BACKCHANNEL"):
        decision = controller.evaluate(
            "LISTENING",
            raw_prediction,
            0.99,
            _evidence(False),
            turn_has_sign_activity=True,
            turn_consumed=False,
        )
        assert decision.effective_prediction == raw_prediction
        assert decision.idle_source is None
        assert not decision.idle_learned


def test_speaking_head_never_emits_controller_idle():
    controller = InteractionController()
    for raw_prediction in ("BARGE", "NO_BARGE"):
        decision = controller.evaluate(
            "SPEAKING",
            raw_prediction,
            0.99,
            _evidence(False),
            turn_has_sign_activity=False,
            turn_consumed=False,
        )
        assert decision.effective_prediction in {"BARGE", "NO_BARGE"}
        assert decision.effective_prediction != "IDLE"
        assert decision.idle_source is None


def test_turn_context_preserves_motion_onset_across_response_finish_and_barge():
    context = TurnContextManager(30.0)
    context.observe(_evidence(True, 12))
    assert context.turn_has_sign_activity
    context.consume_turn()
    context.enter_turn_ready(90, 3.0)
    context.enter_speaking(92, 92 / 30.0, _evidence(False))
    assert context.task_prefix_start_frame("SPEAKING") == 92
    context.enter_listening(150, 5.0, _evidence(True, 120))
    assert context.turn_start_frame == 120
    assert context.task_prefix_start_frame("LISTENING") == 120
    assert not context.turn_consumed


def test_response_lifecycle_rejects_cancelled_callback():
    registry = ResponseLifecycleRegistry()
    response_id, token = registry.create_response("run-a")
    assert registry.record_started(token)
    assert registry.drain("run-a")[0].event == "SYSTEM_RESPONSE_STARTED"
    assert registry.cancel_response("run-a", response_id)
    assert not registry.record_finished(token)
    assert registry.drain("run-a") == []
    assert registry.stale_callback_count("run-a") == 1


def test_response_lifecycle_records_natural_finish():
    registry = ResponseLifecycleRegistry()
    _, token = registry.create_response("run-a")
    assert registry.record_started(token)
    assert registry.record_finished(token)
    assert [event.event for event in registry.drain("run-a")] == [
        "SYSTEM_RESPONSE_STARTED",
        "SYSTEM_RESPONSE_FINISHED",
    ]


def test_tts_simulator_seed_42_is_reproducible_and_covers_both_modes():
    first = TTSResponseSimulator(42)
    second = TTSResponseSimulator(42)
    first_plans = []
    second_plans = []
    for index in range(2):
        complete_time = index * 10.0
        first_plan = first.schedule(
            f"a-{index}", f"RESPONSE:run:a-{index}", complete_time
        )
        second_plan = second.schedule(
            f"b-{index}", f"RESPONSE:run:b-{index}", complete_time
        )
        first_plans.append(
            (
                first_plan.reply_mode,
                first_plan.thinking_time,
                first_plan.tts_duration,
            )
        )
        second_plans.append(
            (
                second_plan.reply_mode,
                second_plan.thinking_time,
                second_plan.tts_duration,
            )
        )
        first.poll(first_plan.planned_response_end_time)
        second.poll(second_plan.planned_response_end_time)
    assert [plan[0] for plan in first_plans] == ["NORMAL", "LONG"]
    assert first_plans == second_plans


def test_tts_simulator_emits_scheduled_start_and_natural_finish():
    simulator = TTSResponseSimulator(42)
    plan = simulator.schedule("r1", "RESPONSE:run:r1", 6.0)
    assert THINKING_TIME_RANGE[0] <= plan.thinking_time <= THINKING_TIME_RANGE[1]
    assert NORMAL_TTS_DURATION_RANGE[0] <= plan.tts_duration <= NORMAL_TTS_DURATION_RANGE[1]
    assert simulator.poll(plan.response_start_time - 1e-6) == []
    started = simulator.poll(plan.response_start_time)
    assert [(event.event, event.event_time) for event in started] == [
        ("SYSTEM_RESPONSE_STARTED", plan.response_start_time)
    ]
    assert plan.status == "SPEAKING"
    finished = simulator.poll(plan.planned_response_end_time)
    assert [(event.event, event.event_time) for event in finished] == [
        ("SYSTEM_RESPONSE_FINISHED", plan.planned_response_end_time)
    ]
    assert plan.status == "FINISHED"
    assert simulator.active_response_id is None


def test_tts_simulator_barge_cancels_old_finish_event():
    simulator = TTSResponseSimulator(42)
    plan = simulator.schedule("r1", "RESPONSE:run:r1", 6.0)
    simulator.poll(plan.response_start_time)
    assert simulator.cancel("r1")
    assert plan.status == "CANCELLED_BY_BARGE"
    assert simulator.poll(plan.planned_response_end_time + 10.0) == []
    assert simulator.counts()["response_cancelled_by_barge_count"] == 1


def test_tts_simulator_fifty_reply_mix_is_near_configured_probability():
    simulator = TTSResponseSimulator(42)
    for index in range(50):
        plan = simulator.schedule(
            str(index), f"RESPONSE:run:{index}", index * 10.0
        )
        simulator.poll(plan.planned_response_end_time)
    counts = simulator.counts()
    assert counts["response_scheduled_count"] == 50
    assert 30 <= counts["normal_reply_count"] <= 40
    assert 10 <= counts["long_reply_count"] <= 20
    assert counts["normal_reply_count"] + counts["long_reply_count"] == 50
    long_plans = [
        response
        for response in simulator.responses
        if response.reply_mode == "LONG"
    ]
    assert all(
        LONG_TTS_DURATION_RANGE[0]
        <= response.tts_duration
        <= LONG_TTS_DURATION_RANGE[1]
        for response in long_plans
    )


def test_end_of_stream_resets_controller_and_state_manager():
    manager = InteractionStateManager()
    manager.handle_prediction("COMPLETE")
    manager.start_response()
    controller = InteractionController()
    event = controller.end_of_stream(manager)
    assert event.event == "END_OF_STREAM"
    assert event.state_before == "SPEAKING"
    assert event.state_after == "LISTENING"
    assert manager.current_system_state == "LISTENING"


def test_app_preserves_raw_model_prediction_before_controller_validation():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    assert "PredictionSmoother" not in source
    assert "controller_result = controller.evaluate(" in source
    assert "controller_prediction = controller_result.controller_prediction" in source
    assert '"raw_prediction": raw_prediction' in source
    assert '"controller_prediction": controller_prediction' in source
    assert '"controller_override_reason": (' in source
    assert 'result["probabilities"][raw_prediction]' in source


def test_timeline_is_append_only_jsonl(tmp_path):
    logger = TimelineLogger(tmp_path, "test")
    logger.append({"timestamp": 0.25, "predicted_class": "INCOMPLETE"})
    logger.append({"timestamp": 0.50, "predicted_class": "COMPLETE"})
    rows = [json.loads(line) for line in logger.path.read_text().splitlines()]
    assert [row["timestamp"] for row in rows] == [0.25, 0.5]


def test_playback_delay_uses_recent_p95_plus_small_margin():
    assert percentile_95([400.0, 450.0, 500.0]) == 495.0
    delay, source_p95 = display_delay_from_latencies([400.0, 450.0, 500.0])
    assert source_p95 == 0.495
    assert delay == 0.57
    assert display_delay_from_latencies([10.0])[0] == DISPLAY_DELAY_MIN_SECONDS
    assert display_delay_from_latencies([1000.0])[0] == DISPLAY_DELAY_MAX_SECONDS


def test_fallback_playback_delay_stays_small():
    delay, source_p95 = display_delay_from_latencies([])
    assert source_p95 == 0.5
    assert delay == 0.575
    assert 0.30 <= delay <= 0.80


def test_fixed_one_second_decision_interval_includes_exact_video_endpoint():
    assert CHUNK_SIZE_SECONDS == 1.0
    assert DECISION_INTERVAL_SECONDS == 1.0
    decisions = build_decisions(
        7.47,
        30.0,
        225,
        DECISION_INTERVAL_SECONDS,
    )
    assert [decision.end_seconds for decision in decisions[:4]] == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    assert decisions[-1].end_seconds == 7.47
    assert decisions[-1].prefix_end_frame == 225


def test_one_second_prefix_grows_with_decision_time():
    decisions = build_decisions(7.47, 30.0, 225, 1.0)
    assert [decision.index for decision in decisions] == list(range(1, 9))
    assert [decision.end_seconds for decision in decisions[:4]] == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    assert [decision.prefix_end_frame for decision in decisions[:4]] == [
        30,
        60,
        90,
        120,
    ]


def test_prepared_cache_access_is_strictly_prefix_bounded():
    frames = {index: np.full((2, 2, 3), index, np.uint8) for index in range(5)}
    prepared = PreparedVideo(
        video_id="video-a",
        path="/tmp/test.mp4",
        fps=1.0,
        duration=5.0,
        total_frames=5,
        decisions=build_decisions(5.0, 1.0, 5),
        frames_by_index=frames,
        pose_keypoints=[np.full((1, 133, 2), index) for index in range(5)],
        pose_scores=[np.ones((1, 133)) for _ in range(5)],
        pose_views_by_end={index: frames[index - 1] for index in range(1, 6)},
        decode_ms=0.0,
        pose_extraction_ms=0.0,
        pose_workers=1,
    )
    assert len(prepared.pose_prefix(3)["keypoints"]) == 3
    assert max(int(frame[0, 0, 0]) for frame in prepared.rgb_sample(3)) == 2
    assert len(prepared.pose_prefix(5, 2)["keypoints"]) == 3
    assert prepared.rgb_source_indices(5, 2) == [2, 3, 4]


def test_latest_only_worker_keeps_at_most_one_pending_request():
    started = threading.Event()
    release = threading.Event()

    class FakeModel:
        calls = 0

        def predict(self, frames, pose, state):
            self.calls += 1
            if self.calls == 1:
                started.set()
                assert release.wait(2.0)
            return {"predicted_class": "INCOMPLETE"}

    class FakePrepared:
        def rgb_sample(self, prefix_end, prefix_start=0):
            return [prefix_start, prefix_end]

        def pose_prefix(self, prefix_end, prefix_start=0):
            return {"prefix_start": prefix_start, "prefix_end": prefix_end}

    wall = time.perf_counter()
    worker = LatestOnlyInferenceWorker(FakeModel(), FakePrepared(), "LISTENING", wall)
    try:
        decisions = [DecisionBoundary(index, index, index) for index in (1, 2, 3)]
        worker.submit(InferenceRequest(decisions[0], 1.0))
        assert started.wait(1.0)
        worker.submit(InferenceRequest(decisions[1], 2.0))
        worker.submit(InferenceRequest(decisions[2], 3.0))
        assert worker.dropped == 1
        release.set()
        deadline = time.time() + 2.0
        while worker.completed < 2 and time.time() < deadline:
            time.sleep(0.01)
        results = worker.drain()
        assert [result.request.decision.index for result in results] == [1, 3]
        assert worker.submitted == 3
        assert worker.completed == 2
    finally:
        worker.close()


def test_worker_state_switch_barrier_waits_for_result_ack():
    class FakeModel:
        def predict(self, frames, pose, state):
            return {"predicted_class": state}

    class FakePrepared:
        def rgb_sample(self, prefix_end, prefix_start=0):
            return [prefix_start, prefix_end]

        def pose_prefix(self, prefix_end, prefix_start=0):
            return {"prefix_start": prefix_start, "prefix_end": prefix_end}

    worker = LatestOnlyInferenceWorker(
        FakeModel(),
        FakePrepared(),
        "LISTENING",
        time.perf_counter(),
        pause_after_result=True,
    )
    try:
        first = DecisionBoundary(1, 1.0, 1)
        second = DecisionBoundary(2, 2.0, 2)
        worker.submit(InferenceRequest(first, 1.0))
        deadline = time.time() + 1.0
        while worker.completed < 1 and time.time() < deadline:
            time.sleep(0.01)
        worker.submit(InferenceRequest(second, 2.0))
        time.sleep(0.05)
        assert worker.completed == 1
        assert worker.drain()[0].request.decision.index == 1
        worker.acknowledge_result()
        deadline = time.time() + 1.0
        while worker.completed < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert worker.drain()[0].request.decision.index == 2
        worker.acknowledge_result()
    finally:
        worker.close()


def test_runtime_state_is_the_only_start_gate():
    assert not can_start(empty_state())
    assert not can_start(runtime_state("PREPARING"))
    ready = runtime_state(
        "READY",
        video_id="video-a",
        prepared_key="prepared",
        video_path="/tmp/video.mp4",
        pose_cache_ready=True,
    )
    assert can_start(ready)
    assert not can_start({**ready, "pose_cache_ready": False})
    assert not can_start({**ready, "phase": "RUNNING"})


def test_finished_and_stop_style_transitions_preserve_timeline():
    ready = runtime_state(
        "READY",
        video_id="video-a",
        prepared_key="video-a",
        video_path="/tmp/video.mp4",
        pose_cache_ready=True,
    )
    rows = [[0.5, "COMPLETE", "TAKE_TURN"]]
    interaction_rows = [
        [0.0, "LISTENING", "Initial state"],
        [0.5, "COMPLETE", "LISTENING → TURN_READY"],
    ]
    running = transition(
        ready,
        "RUNNING",
        prediction_history=rows,
        interaction_history=interaction_rows,
        current_system_state="SPEAKING",
        current_time=0.5,
        current_prefix=15,
        latest_prediction="COMPLETE",
    )
    finished = transition(running, "FINISHED")
    assert finished["prediction_history"] == rows
    assert finished["interaction_history"] == interaction_rows
    assert finished["current_system_state"] == "SPEAKING"
    assert finished["current_time"] == 0.5
    assert finished["current_prefix"] == 15
    assert empty_state()["prediction_history"] == []


def test_new_video_registry_clear_prevents_old_cache_reuse():
    registry = PreparedVideoRegistry(capacity=1)

    class FakePrepared:
        def __init__(self, video_id):
            self.video_id = video_id

    registry.put(FakePrepared("video-a"))
    assert registry.get("video-a").video_id == "video-a"
    registry.clear()
    registry.put(FakePrepared("video-b"))
    assert len(registry) == 1
    assert registry.get("video-b").video_id == "video-b"
    try:
        registry.get("video-a")
    except KeyError:
        pass
    else:
        raise AssertionError("old video cache survived a new upload")


def test_reviewer_ui_has_no_raw_controller_or_motion_debug_panel():
    source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'TIMELINE_HEADERS = ["Decision Time", "Prediction", "Action"]' in source
    assert 'label="Decision Interval"' not in source
    assert "decision_interval = gr.Radio" not in source
    assert 'label="Current System State"' in source
    assert 'label="Interaction Timeline"' in source
    assert 'label="Motion Score"' not in source
    assert 'label="Hand Motion Score"' not in source
    assert 'label="Activity"' not in source
    assert 'label="Controller Override"' not in source
    assert 'label="Controller Override Reason"' not in source
    assert 'label="Raw Prediction (Debug)"' not in source
    assert 'label="Controller-level IDLE (Debug)"' not in source
    assert '"Raw Model Prediction"' in source
    assert '"Controller Prediction"' in source
    assert '"IDLE Source"' in source
    assert '"Learned IDLE": raw_prediction == "IDLE"' in source
    assert '"Override Reason"' in source
    assert 'label="System State"' not in source
    assert "system_state = gr.Radio" not in source
    assert "inputs=[upload, runtime]" in source
    assert 'f"PLAY_VIDEO:{run_id}:{display_delay:.6f}"' in source
    assert "decision_time + display_delay" in source
    assert '"predictions_released_at_display_time": True' in source
    for field in (
        '"video_id"',
        '"global_decision_time"',
        '"global_prefix_end_frame"',
        '"task_prefix_start_frame"',
        '"task_prefix_end_frame"',
        '"state_before"',
        '"state_after"',
        '"action"',
        '"motion_score"',
        '"hand_motion_score"',
        '"recent_active_fraction"',
        '"sustained_active_duration"',
            '"causal_sign_evidence"',
            '"activity"',
            '"raw_prediction"',
            '"final_prediction"',
            '"system_action"',
            '"current_system_state"',
            '"raw_confidence"',
    ):
        assert field in source
    forbidden_labels = (
        'label="Confidence"',
        'label="Stable Prediction"',
        'label="Streaming Pose"',
        'label="Chunk Range"',
        'label="Prefix Range"',
        'label="Inference Latency',
        'gr.JSON(label="Class Probabilities")',
    )
    assert not any(label in source for label in forbidden_labels)
