"""SignDuplex V3.7.0 compact paper demo with synchronized final TTS."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from bootstrap import configure, restore_system_packages

configure()

import gradio as gr
import gradio_client.utils as gradio_client_utils

restore_system_packages()

from config import (
    BARGE_LABELS,
    CHUNK_SIZE_SECONDS,
    DECISION_INTERVAL_SECONDS,
    DEVICE,
    FINAL_REPLY_EXTENSION_SECONDS,
    ALIGNMENT_LATENCY_WINDOW,
    HOST,
    LOGS_DIR,
    PORT,
    POST_EOS_DISPLAY_HOLD_SECONDS,
    SYSTEM_AUDIO,
    TURN_LABELS,
    validate_paths,
)
from runtime.async_worker import InferenceRequest, LatestOnlyInferenceWorker
from runtime.causal_sign_evidence import causal_sign_evidence
from runtime.demo_scenario import (
    DEMO_SCENARIO_MODE,
    BalancedDemoScenarioPlanner,
)
from runtime.interaction_controller import (
    BACKCHANNEL_COOLDOWN_SECONDS,
    POST_BARGE_PROTECTION_SECONDS,
    InteractionController,
)
from runtime.interaction_state import InteractionStateManager
from runtime.model_runtime import ModelRuntime
from runtime.playback_alignment import (
    display_delay_from_latencies,
    percentile_95,
)
from runtime.session_state import can_start, empty_state, runtime_state, transition
from runtime.response_lifecycle import ResponseLifecycleRegistry
from runtime.state_controller import action_for, assert_routed_label
from runtime.stream_cache import (
    PreparedVideoRegistry,
    build_decisions,
    prepare_video,
)
from runtime.subtitle_timeline import (
    SUBTITLE_LEARNED,
    SUBTITLE_SOURCE,
    ReferenceSubtitleSegment,
    SubtitleTimelineManager,
    SubtitleTimelineRegistry,
)
from runtime.timeline import TimelineLogger
from runtime.tts_response_simulator import (
    DEFAULT_RANDOM_SEED,
    LONG_REPLY_PROBABILITY,
    LONG_TTS_DURATION_RANGE,
    NORMAL_REPLY_PROBABILITY,
    NORMAL_TTS_DURATION_RANGE,
    THINKING_TIME_RANGE,
    TTSResponseSimulator,
)
from runtime.waveform_display import (
    BACKCHANNEL_FEEDBACK_DURATION_RANGE,
    BackchannelFeedbackSampler,
)
from runtime.turn_context import TurnContextManager


TIMELINE_HEADERS = ["Decision Time", "Prediction", "Action"]
INTERACTION_HEADERS = ["Time", "Event", "State Transition"]
DEBUG_MODE = False
prediction_to_chinese_explanation = {
    "IDLE": "用户未开始表达，系统继续监听",
    "INCOMPLETE": "用户仍在表达，尚未完成",
    "BACKCHANNEL": "用户仍在表达，系统给出简短反馈",
    "COMPLETE": "用户表达完成，系统开始回应",
    "NO_BARGE": "系统正在说话，用户未打断",
    "BARGE": "用户打断系统，系统切回监听",
}
prediction_to_simple_action = {
    "IDLE": "KEEP_LISTENING",
    "INCOMPLETE": "KEEP_LISTENING",
    "BACKCHANNEL": "SHORT_FEEDBACK",
    "COMPLETE": "SWITCH_TO_SPEAKING",
    "NO_BARGE": "CONTINUE_SPEAKING",
    "BARGE": "SWITCH_TO_LISTENING",
}
_MODEL: ModelRuntime | None = None
_MODEL_WARMED = False
_MODEL_LOCK = threading.Lock()
_PREPARED = PreparedVideoRegistry(capacity=1)
_LATENCY_LOCK = threading.Lock()
_INFERENCE_LATENCIES_MS: deque[float] = deque(
    maxlen=ALIGNMENT_LATENCY_WINDOW
)
_RESPONSE_LIFECYCLE = ResponseLifecycleRegistry()
_SUBTITLES = SubtitleTimelineRegistry()
_RANDOM_SEED = DEFAULT_RANDOM_SEED


@dataclass
class AlignedPrediction:
    target_wall_seconds: float
    decision_time: float
    prefix_end_frame: int
    prediction: str
    action: str
    explanation: str
    record: dict[str, Any]


def idle_debug_record(
    raw_prediction: str = "—",
    controller_prediction: str = "—",
    idle_source: str | None = None,
    override_reason: str = "NONE",
) -> dict[str, object]:
    return {
        "Raw Model Prediction": raw_prediction,
        "Controller Prediction": controller_prediction,
        "IDLE Source": idle_source,
        "Learned IDLE": raw_prediction == "IDLE",
        "Override Reason": override_reason,
    }


def subtitle_debug_record(
    manager: SubtitleTimelineManager | None = None,
    segment: ReferenceSubtitleSegment | None = None,
) -> dict[str, object]:
    return {
        "subtitle_enabled": bool(manager and manager.enabled),
        "subtitle_source": SUBTITLE_SOURCE,
        "subtitle_learned": SUBTITLE_LEARNED,
        "segment_index": segment.segment_index if segment else None,
        "semantic_id": segment.semantic_id if segment else None,
    }


def format_prediction_display(prediction: str) -> str:
    explanation = prediction_to_chinese_explanation[prediction]
    return f"## {prediction}\n\n{explanation}"


def simple_action_for(prediction: str) -> str:
    return prediction_to_simple_action[prediction]


def video_display_name(video_path: str | None) -> str:
    """Return a stable test identifier without exposing a temporary path."""
    return Path(video_path).name if video_path else "—"


def format_test_video(video_path: str | None) -> str:
    return f"**Test Video:** {video_display_name(video_path)}"


def public_system_state(internal_state: str) -> str:
    if internal_state in {"TURN_READY", "SPEAKING"}:
        return "SPEAKING"
    return "LISTENING"


def public_state_for_prediction(prediction: str) -> str:
    action = simple_action_for(prediction)
    if action in {"SWITCH_TO_SPEAKING", "CONTINUE_SPEAKING"}:
        return "SPEAKING"
    return "LISTENING"


_ORIGINAL_SCHEMA_TO_TYPE = gradio_client_utils._json_schema_to_python_type


def _compatible_schema_to_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _ORIGINAL_SCHEMA_TO_TYPE(schema, defs)


gradio_client_utils._json_schema_to_python_type = _compatible_schema_to_type


def get_model(*, warmup: bool = True) -> ModelRuntime:
    """Load the formal model once and keep it resident for all runs."""
    global _MODEL, _MODEL_WARMED
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = ModelRuntime(DEVICE)
        if warmup and not _MODEL_WARMED:
            latencies = _MODEL.warmup(iterations=3)
            print(
                "[warmup] "
                + json.dumps(
                    {"iterations": 3, "model_latency_ms": latencies},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            _MODEL_WARMED = True
        return _MODEL


def choose_display_delay() -> tuple[float, float]:
    """Freeze one small P95-based display delay for the entire playback run."""
    with _LATENCY_LOCK:
        recent_ms = list(_INFERENCE_LATENCIES_MS)
    return display_delay_from_latencies(recent_ms)


def remember_inference_latency(latency_ms: float) -> None:
    with _LATENCY_LOCK:
        _INFERENCE_LATENCIES_MS.append(float(latency_ms))


def format_model_decision_time(
    decision_time: float, *, is_final_model_decision: bool
) -> str:
    """Format only a true model decision clock for the Decision Time UI."""
    nearest_integer = round(decision_time)
    if abs(decision_time - nearest_integer) <= 1e-6:
        return f"{nearest_integer} s"
    if is_final_model_decision:
        return f"{decision_time:.2f} s"
    return f"{decision_time:g} s"


def _cleared_ui_outputs(state: dict, *, start_enabled: bool) -> tuple:
    return (
        state,
        gr.update(interactive=start_enabled),
        format_prediction_display("IDLE"),
        "—",
        simple_action_for("IDLE"),
        "—",
        "0.000000",
        "0.000000",
        "NO_ACTIVITY",
        "NONE",
        "NONE",
        "LISTENING",
        idle_debug_record(),
        TTSResponseSimulator().debug_record(),
        [],
        [[0.0, "LISTENING", "Initial state"]],
        "—",
        subtitle_debug_record(),
        None,
        f"STOP:upload:{uuid.uuid4().hex[:8]}",
    )


def prepare_upload(video_path: str | None) -> Iterator[tuple]:
    """Create a unique video session after fully dropping the old cache/state."""
    _PREPARED.clear()
    _SUBTITLES.clear()
    if not video_path:
        yield _cleared_ui_outputs(empty_state(), start_enabled=False)
        return

    video_id = uuid.uuid4().hex
    preparing = runtime_state(
        "PREPARING",
        video_id=video_id,
        video_path=video_path,
    )
    yield _cleared_ui_outputs(preparing, start_enabled=False)
    try:
        validate_paths()
        prepared = prepare_video(
            video_path,
            pose_device="cuda" if DEVICE.startswith("cuda") else "cpu",
            video_id=video_id,
        )
        key = _PREPARED.put(prepared)
        subtitle_manager = SubtitleTimelineManager.from_video_path(prepared.path)
        _SUBTITLES.put(key, subtitle_manager)
        demo_validation = BalancedDemoScenarioPlanner(
            subtitle_manager.segments,
            _RANDOM_SEED,
        ).validation()
        for pause in demo_validation.pauses:
            print(
                "[demo-validation] "
                + json.dumps(
                    {
                        "video_id": video_id,
                        **pause.as_record(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        print(
            "[demo-validation] "
            + json.dumps(
                demo_validation.as_record(video_id=video_id),
                ensure_ascii=False,
            ),
            flush=True,
        )
        ready = runtime_state(
            "READY",
            video_id=video_id,
            prepared_key=key,
            video_path=prepared.path,
            pose_cache_ready=True,
        )
        print(
            "[prepare] "
            + json.dumps(
                {
                    "phase": "READY",
                    "video_id": video_id,
                    "video_path": prepared.path,
                    "duration_seconds": prepared.duration,
                    "pose_cache_frames": len(prepared.pose_keypoints),
                    "pose_precompute_ms": prepared.pose_extraction_ms,
                    "pose_workers": prepared.pose_workers,
                    "registry_size": len(_PREPARED),
                    "subtitle_enabled": subtitle_manager.enabled,
                    "subtitle_source": SUBTITLE_SOURCE,
                    "subtitle_learned": SUBTITLE_LEARNED,
                    "subtitle_segment_count": len(subtitle_manager.segments),
                    "reference_annotation_source": (
                        str(subtitle_manager.annotation_source)
                        if subtitle_manager.annotation_source
                        else None
                    ),
                    "concat_metadata_source": (
                        str(subtitle_manager.metadata_source)
                        if subtitle_manager.metadata_source
                        else None
                    ),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        yield (
            ready,
            gr.update(interactive=True),
            gr.update(),
            gr.update(),
            gr.update(),
            "—",
            "0.000000",
            "0.000000",
            "NO_ACTIVITY",
            "NONE",
            "NONE",
            "LISTENING",
            idle_debug_record(),
            TTSResponseSimulator().debug_record(),
            [],
            [[0.0, "LISTENING", "Initial state"]],
            "—",
            subtitle_debug_record(subtitle_manager),
            gr.update(),
            gr.update(),
        )
    except Exception as error:
        _PREPARED.clear()
        raise gr.Error(
            f"视频/Pose 预处理失败：{type(error).__name__}: {error}"
        ) from error


def stream_demo(
    video_path: str | None,
    state: dict | None,
) -> Iterator[tuple]:
    if not video_path:
        raise gr.Error("请先上传 MP4 视频")
    if not can_start(state):
        raise gr.Error("Pose cache 尚未准备完成，Start 仅在 READY 后可用")
    decision_interval = DECISION_INTERVAL_SECONDS
    prepared = _PREPARED.get(state["prepared_key"])
    subtitle_manager = _SUBTITLES.get(state["prepared_key"])
    resolved_video_path = str(Path(video_path).resolve())
    if (
        prepared.path != resolved_video_path
        or prepared.video_id != state["video_id"]
        or state["prepared_key"] != prepared.video_id
    ):
        raise gr.Error("上传视频与 RGB/Pose cache session 不匹配，请重新上传")

    decisions = build_decisions(
        prepared.duration,
        prepared.fps,
        prepared.total_frames,
        decision_interval,
    )
    run_id = uuid.uuid4().hex[:12]
    logger = TimelineLogger(LOGS_DIR, run_id)
    state_logger = TimelineLogger(LOGS_DIR, run_id, "state_timeline")
    subtitle_logger = TimelineLogger(LOGS_DIR, run_id, "subtitle_timeline")
    waveform_logger = TimelineLogger(LOGS_DIR, run_id, "waveform_timeline")
    controller = InteractionController()
    state_manager = InteractionStateManager()
    turn_context = TurnContextManager(prepared.fps)
    tts_simulator = TTSResponseSimulator(_RANDOM_SEED)
    feedback_sampler = BackchannelFeedbackSampler(_RANDOM_SEED)
    scenario_planner = BalancedDemoScenarioPlanner(
        subtitle_manager.segments,
        _RANDOM_SEED,
    )
    demo_validation = scenario_planner.validation()
    timeline_rows: list[list[object]] = []
    interaction_rows: list[list[object]] = [[0.0, "LISTENING", "Initial state"]]
    completed_decision_times: list[float] = []
    inference_latencies_ms: list[float] = []
    alignment_errors_seconds: list[float] = []
    pending_displays: deque[AlignedPrediction] = deque()
    maximum_lag = 0.0
    display_delay, delay_source_p95 = choose_display_delay()
    playback_signal = f"PLAY_VIDEO:{run_id}:{display_delay:.6f}"
    active_response_id: str | None = None
    response_started_count = 0
    response_finished_count = 0
    normal_turn_count = 0
    barge_in_count = 0
    learned_idle_count = 0
    controller_override_count = 0
    post_barge_protection_count = 0
    backchannel_cooldown_count = 0
    backchannel_feedback_count = 0
    waveform_full_reply_count = 0
    waveform_stop_count = 0
    idle_during_speaking_count = 0
    idle_after_turn_started_count = 0
    last_effective = "IDLE"
    last_raw = "—"
    last_action = simple_action_for("IDLE")
    last_motion = 0.0
    last_hand_motion = 0.0
    last_activity = "NO_ACTIVITY"
    last_override = "NONE"
    last_override_reason = "NONE"
    last_idle_debug = idle_debug_record()
    current_reference_segment: ReferenceSubtitleSegment | None = None
    last_subtitle_key: int | None = None
    last_subtitle_text = ""
    last_subtitle_debug = subtitle_debug_record(subtitle_manager)
    subtitle_change_count = 0
    active_short_feedback: dict[str, object] | None = None
    active_full_waveform: dict[str, object] | None = None
    final_response_id: str | None = None
    final_reply_extension_applied = False
    final_response_finished_wall_time: float | None = None
    final_waveform_stop_wall_time: float | None = None

    running = transition(
        state,
        "RUNNING",
        prediction_history=[],
        interaction_history=interaction_rows,
        current_system_state=state_manager.current_system_state,
        current_time=0.0,
        current_prefix=0,
        latest_prediction=None,
    )
    wall_started = time.perf_counter()
    yield (
        running,
        gr.update(interactive=False),
        format_prediction_display("IDLE"),
        gr.update(),
        simple_action_for("IDLE"),
        "—",
        "0.000000",
        "0.000000",
        "NO_ACTIVITY",
        "NONE",
        "NONE",
        public_system_state(state_manager.current_system_state),
        last_idle_debug,
        tts_simulator.debug_record(),
        [],
        list(interaction_rows),
        "—",
        last_subtitle_debug,
        None,
        playback_signal,
    )

    worker: LatestOnlyInferenceWorker | None = LatestOnlyInferenceWorker(
        get_model(), prepared, "LISTENING", wall_started, pause_after_result=True
    )
    retired_worker_submitted = 0
    retired_worker_completed = 0
    retired_worker_dropped = 0
    next_decision = 0
    backend_stream_completed_seconds: float | None = None
    display_playback_completed_seconds: float | None = None

    def retire_worker() -> None:
        nonlocal worker
        nonlocal retired_worker_submitted
        nonlocal retired_worker_completed
        nonlocal retired_worker_dropped
        if worker is None:
            return
        worker.close()
        retired_worker_submitted += worker.submitted
        retired_worker_completed += worker.completed
        retired_worker_dropped += worker.dropped
        worker = None

    def create_worker(system_state: str) -> None:
        nonlocal worker
        if worker is not None:
            raise RuntimeError("Cannot replace a live inference worker")
        worker = LatestOnlyInferenceWorker(
            get_model(),
            prepared,
            system_state,
            wall_started,
            pause_after_result=True,
        )

    def observed_boundary() -> tuple[float, int]:
        global_time = min(time.perf_counter() - wall_started, prepared.duration)
        frame = min(
            prepared.total_frames,
            max(1, int(round(global_time * prepared.fps))),
        )
        return global_time, frame

    def evidence_at(start_frame: int, end_frame: int):
        safe_start = max(0, min(start_frame, max(0, end_frame - 1)))
        return causal_sign_evidence(
            prepared.pose_keypoints,
            prepared.pose_scores,
            safe_start,
            end_frame,
            prepared.fps,
        )

    def log_state_transition(
        event_time: float,
        event_frame: int,
        state_transition,
        response_id: str | None = None,
        extra: dict[str, object] | None = None,
        *,
        event_time_domain: str = "VIDEO",
        video_time: float | None = None,
    ) -> None:
        interaction_wall_time = time.perf_counter() - wall_started
        record = {
            "run_id": run_id,
            "global_time": round(event_time, 6),
            "event_time_domain": event_time_domain,
            "video_time": round(
                min(event_time, prepared.duration)
                if video_time is None
                else video_time,
                6,
            ),
            "interaction_wall_time": round(interaction_wall_time, 6),
            "global_prefix_end_frame": event_frame,
            "event": state_transition.event,
            "response_id": response_id,
            "state_before": state_transition.state_before,
            "system_action": state_transition.action,
            "state_after": state_transition.state_after,
        }
        if extra:
            record.update(extra)
        state_logger.append(record)
        print("[state-transition] " + json.dumps(record, ensure_ascii=False), flush=True)

    def log_waveform_event(
        *,
        event: str,
        waveform_mode: str,
        trigger: str,
        interaction_wall_time: float,
        video_time: float,
        start_time: float | None,
        planned_end_time: float | None,
        actual_end_time: float | None,
        response_id: str | None,
        duration: float | None = None,
    ) -> None:
        record = {
            "run_id": run_id,
            "event": event,
            "waveform_mode": waveform_mode,
            "trigger": trigger,
            "event_time_domain": "INTERACTION",
            "video_time": round(video_time, 6),
            "interaction_wall_time": round(interaction_wall_time, 6),
            "start_time": round(start_time, 6) if start_time is not None else None,
            "planned_end_time": (
                round(planned_end_time, 6)
                if planned_end_time is not None
                else None
            ),
            "actual_end_time": (
                round(actual_end_time, 6) if actual_end_time is not None else None
            ),
            "response_id": response_id,
            "duration": round(duration, 6) if duration is not None else None,
        }
        waveform_logger.append(record)
        print("[waveform-event] " + json.dumps(record, ensure_ascii=False), flush=True)

    try:
        while True:
            now = time.perf_counter()
            wall_elapsed = now - wall_started
            if (
                active_short_feedback is not None
                and wall_elapsed
                >= float(active_short_feedback["planned_end_time"])
            ):
                log_waveform_event(
                    event="STOP",
                    waveform_mode="SHORT_FEEDBACK",
                    trigger="DURATION_ELAPSED",
                    interaction_wall_time=wall_elapsed,
                    video_time=min(
                        prepared.duration,
                        max(0.0, wall_elapsed - display_delay),
                    ),
                    start_time=float(active_short_feedback["start_time"]),
                    planned_end_time=float(
                        active_short_feedback["planned_end_time"]
                    ),
                    actual_end_time=wall_elapsed,
                    response_id=str(active_short_feedback["response_id"]),
                    duration=float(active_short_feedback["duration"]),
                )
                active_short_feedback = None
                waveform_stop_count += 1
            backend_elapsed = min(wall_elapsed, prepared.duration)
            if backend_stream_completed_seconds is None and backend_elapsed >= prepared.duration:
                backend_stream_completed_seconds = wall_elapsed
            if (
                display_playback_completed_seconds is None
                and wall_elapsed >= prepared.duration + display_delay
            ):
                display_playback_completed_seconds = wall_elapsed

            playback_video_time = min(
                prepared.duration,
                max(0.0, wall_elapsed - display_delay),
            )
            current_subtitle_segment = (
                subtitle_manager.segment_at(playback_video_time)
                if wall_elapsed + 1e-6 >= display_delay
                else None
            )
            current_subtitle_key = (
                current_subtitle_segment.segment_index
                if current_subtitle_segment is not None
                else None
            )
            if current_subtitle_key != last_subtitle_key:
                last_subtitle_key = current_subtitle_key
                last_subtitle_debug = subtitle_debug_record(
                    subtitle_manager, current_subtitle_segment
                )
                if current_subtitle_segment is not None:
                    current_reference_segment = current_subtitle_segment
                    if last_subtitle_text:
                        last_subtitle_text = ""
                        subtitle_change_count += 1
                        subtitle_record = {
                            "run_id": run_id,
                            "event": "REFERENCE_SENTENCE_CLEARED",
                            "video_time": round(playback_video_time, 6),
                            "segment_index": current_subtitle_segment.segment_index,
                            "semantic_id": current_subtitle_segment.semantic_id,
                            "text": "",
                            "subtitle_source": SUBTITLE_SOURCE,
                            "subtitle_learned": SUBTITLE_LEARNED,
                        }
                        subtitle_logger.append(subtitle_record)
                        print(
                            "[reference-sentence] "
                            + json.dumps(subtitle_record, ensure_ascii=False),
                            flush=True,
                        )
                        yield (
                            running,
                            gr.update(),
                            format_prediction_display(last_effective),
                            gr.update(),
                            last_action,
                            last_raw,
                            f"{last_motion:.6f}",
                            f"{last_hand_motion:.6f}",
                            last_activity,
                            last_override,
                            last_override_reason,
                            public_state_for_prediction(last_effective),
                            last_idle_debug,
                            tts_simulator.debug_record(),
                            list(timeline_rows),
                            list(interaction_rows),
                            "—",
                            last_subtitle_debug,
                            gr.update(),
                            gr.update(),
                        )

            lifecycle_clock = wall_elapsed
            active_plan = tts_simulator.latest_response
            if (
                active_plan is not None
                and active_plan.status == "SPEAKING"
                and wall_elapsed >= active_plan.planned_response_end_time
                and worker is not None
                and not worker.is_idle()
            ):
                # A causal decision whose timestamp precedes the planned TTS
                # end must be resolved before the later finish event. Runtime
                # inference latency must not reverse logical event ordering.
                lifecycle_clock = active_plan.planned_response_end_time - 1e-9

            for lifecycle_event in tts_simulator.poll(lifecycle_clock):
                if lifecycle_event.event == "SYSTEM_RESPONSE_STARTED":
                    callback_accepted = _RESPONSE_LIFECYCLE.record_started(
                        lifecycle_event.response_token
                    )
                else:
                    callback_accepted = _RESPONSE_LIFECYCLE.record_finished(
                        lifecycle_event.response_token
                    )
                _RESPONSE_LIFECYCLE.drain(run_id)
                if not callback_accepted:
                    continue
                event_time = lifecycle_event.event_time
                event_frame = min(
                    prepared.total_frames,
                    max(1, int(round(event_time * prepared.fps))),
                )
                response = tts_simulator.latest_response
                response_details = (
                    {
                        "reply_mode": response.reply_mode,
                        "scenario_reply_mode": response.scenario_reply_mode,
                        "next_sign_start": response.next_sign_start,
                        "reply_timing_margin": response.reply_timing_margin,
                        "sampled_tts_duration": response.sampled_tts_duration,
                        "final_reply_extension_seconds": (
                            response.final_reply_extension_seconds
                        ),
                        "thinking_time": response.thinking_time,
                        "tts_duration": response.tts_duration,
                        "response_start_time": response.response_start_time,
                        "planned_response_end_time": (
                            response.planned_response_end_time
                        ),
                        "random_seed": response.random_seed,
                    }
                    if response is not None
                    else {}
                )
                if lifecycle_event.event == "SYSTEM_RESPONSE_STARTED":
                    if (
                        state_manager.current_system_state != "TURN_READY"
                        or lifecycle_event.response_id != active_response_id
                    ):
                        continue
                    evidence = evidence_at(turn_context.state_enter_frame, event_frame)
                    state_transition = state_manager.start_response()
                    turn_context.enter_speaking(event_frame, event_time, evidence)
                    response_started_count += 1
                    log_state_transition(
                        event_time,
                        event_frame,
                        state_transition,
                        active_response_id,
                        {**response_details, "response_status": "SPEAKING"},
                        event_time_domain="INTERACTION",
                        video_time=min(event_time, prepared.duration),
                    )
                    if active_short_feedback is not None:
                        log_waveform_event(
                            event="STOP",
                            waveform_mode="SHORT_FEEDBACK",
                            trigger="PREEMPTED_BY_FULL_REPLY",
                            interaction_wall_time=wall_elapsed,
                            video_time=min(event_time, prepared.duration),
                            start_time=float(active_short_feedback["start_time"]),
                            planned_end_time=float(
                                active_short_feedback["planned_end_time"]
                            ),
                            actual_end_time=wall_elapsed,
                            response_id=str(active_short_feedback["response_id"]),
                            duration=float(active_short_feedback["duration"]),
                        )
                        active_short_feedback = None
                        waveform_stop_count += 1
                    if response is None:
                        raise RuntimeError("Started response details are missing")
                    active_full_waveform = {
                        "response_id": lifecycle_event.response_id,
                        "start_time": response.response_start_time,
                        "planned_end_time": response.planned_response_end_time,
                        "duration": response.tts_duration,
                    }
                    log_waveform_event(
                        event="START",
                        waveform_mode="FULL_REPLY",
                        trigger="SYSTEM_RESPONSE_STARTED",
                        interaction_wall_time=wall_elapsed,
                        video_time=min(event_time, prepared.duration),
                        start_time=response.response_start_time,
                        planned_end_time=response.planned_response_end_time,
                        actual_end_time=None,
                        response_id=lifecycle_event.response_id,
                        duration=response.tts_duration,
                    )
                    waveform_full_reply_count += 1
                    interaction_rows.append(
                        [
                            round(event_time, 6),
                            "SYSTEM_RESPONSE_STARTED",
                            "TURN_READY → SPEAKING",
                        ]
                    )
                    if (
                        event_frame < prepared.total_frames
                        and lifecycle_event.response_id != final_response_id
                    ):
                        create_worker("SPEAKING")
                elif lifecycle_event.event == "SYSTEM_RESPONSE_FINISHED":
                    if (
                        state_manager.current_system_state != "SPEAKING"
                        or lifecycle_event.response_id != active_response_id
                    ):
                        continue
                    retire_worker()
                    evidence = evidence_at(
                        turn_context.state_enter_frame, event_frame
                    )
                    state_transition = state_manager.finish_response()
                    turn_context.enter_listening(event_frame, event_time, evidence)
                    response_finished_count += 1
                    normal_turn_count += 1
                    if lifecycle_event.response_id == final_response_id:
                        final_response_finished_wall_time = wall_elapsed
                    log_state_transition(
                        event_time,
                        event_frame,
                        state_transition,
                        active_response_id,
                        {
                            **response_details,
                            "response_cancelled": False,
                            "response_status": "FINISHED",
                        },
                        event_time_domain="INTERACTION",
                        video_time=min(event_time, prepared.duration),
                    )
                    if active_full_waveform is not None:
                        log_waveform_event(
                            event="STOP",
                            waveform_mode="FULL_REPLY",
                            trigger="SYSTEM_RESPONSE_FINISHED",
                            interaction_wall_time=wall_elapsed,
                            video_time=min(event_time, prepared.duration),
                            start_time=float(active_full_waveform["start_time"]),
                            planned_end_time=float(
                                active_full_waveform["planned_end_time"]
                            ),
                            actual_end_time=wall_elapsed,
                            response_id=lifecycle_event.response_id,
                            duration=float(active_full_waveform["duration"]),
                        )
                        if lifecycle_event.response_id == final_response_id:
                            final_waveform_stop_wall_time = wall_elapsed
                        active_full_waveform = None
                        waveform_stop_count += 1
                    interaction_rows.append(
                        [
                            round(event_time, 6),
                            "SYSTEM_RESPONSE_FINISHED",
                            "SPEAKING → LISTENING",
                        ]
                    )
                    active_response_id = None
                    while (
                        next_decision < len(decisions)
                        and decisions[next_decision].end_seconds <= event_time + 1e-6
                    ):
                        next_decision += 1
                    if (
                        event_frame < prepared.total_frames
                        and lifecycle_event.response_id != final_response_id
                    ):
                        create_worker("LISTENING")

                running = transition(
                    running,
                    "RUNNING",
                    interaction_history=interaction_rows,
                    current_system_state=state_manager.current_system_state,
                    current_time=event_time,
                    current_prefix=event_frame,
                )
                yield (
                    running,
                    gr.update(),
                    format_prediction_display(last_effective),
                    gr.update(),
                    last_action,
                    last_raw,
                    f"{last_motion:.6f}",
                    f"{last_hand_motion:.6f}",
                    last_activity,
                    last_override,
                    last_override_reason,
                    public_state_for_prediction(last_effective),
                    last_idle_debug,
                    tts_simulator.debug_record(),
                    list(timeline_rows),
                    list(interaction_rows),
                    last_subtitle_text or "—",
                    last_subtitle_debug,
                    str(SYSTEM_AUDIO)
                    if lifecycle_event.event == "SYSTEM_RESPONSE_STARTED"
                    else None,
                    (
                        f"TTS_START:{run_id}:{lifecycle_event.response_id}:"
                        f"{response.tts_duration:.6f}"
                        if lifecycle_event.event == "SYSTEM_RESPONSE_STARTED"
                        else f"STOP:NORMAL:{run_id}:{lifecycle_event.response_id}"
                    ),
                )

            while (
                worker is not None
                and next_decision < len(decisions)
                and backend_elapsed + 1e-6 >= decisions[next_decision].end_seconds
            ):
                decision = decisions[next_decision]
                task_start = turn_context.task_prefix_start_frame(
                    state_manager.inference_state
                )
                if task_start >= decision.prefix_end_frame:
                    next_decision += 1
                    continue
                worker.submit(
                    InferenceRequest(decision, backend_elapsed, task_start)
                )
                next_decision += 1

            state_changed = False
            completed_items = worker.drain() if worker is not None else []
            for completed in completed_items:
                if isinstance(completed, Exception):
                    raise completed
                if worker is None:
                    break
                ui_started = time.perf_counter()
                result = completed.prediction
                raw_prediction = result["predicted_class"]
                raw_confidence = float(result["probabilities"][raw_prediction])
                state_before = state_manager.inference_state
                if worker.system_state != state_before:
                    raise RuntimeError("Inference worker state is out of sync")
                assert_routed_label(state_before, raw_prediction)
                decision = completed.request.decision
                decision_time = decision.end_seconds
                prefix_end_frame = decision.prefix_end_frame
                task_start_frame = completed.request.task_prefix_start_frame
                if not 0 <= task_start_frame < prefix_end_frame:
                    raise RuntimeError("Invalid turn-relative causal prefix")
                evidence = evidence_at(task_start_frame, prefix_end_frame)
                turn_context.observe(evidence)
                context_before = turn_context.snapshot()
                # The V1.8 output remains immutable in raw_prediction.  The
                # stateful Controller validates only the resulting interaction
                # action and records its separate controller_prediction.
                controller_result = controller.evaluate(
                    state_before,
                    raw_prediction,
                    raw_confidence,
                    evidence,
                    decision_time=decision_time,
                    turn_has_sign_activity=context_before.turn_has_sign_activity,
                    turn_consumed=context_before.turn_consumed,
                )
                controller_prediction = controller_result.controller_prediction
                learned_idle_count += int(raw_prediction == "IDLE")
                controller_override_count += int(
                    controller_prediction != raw_prediction
                )
                post_barge_protection_count += int(
                    controller_result.controller_override_reason
                    == "POST_BARGE_PROTECTION"
                )
                backchannel_cooldown_count += int(
                    controller_result.controller_override_reason
                    == "BACKCHANNEL_COOLDOWN"
                )
                if controller_prediction == "IDLE":
                    idle_during_speaking_count += int(
                        state_before == "SPEAKING"
                    )
                    idle_after_turn_started_count += int(
                        context_before.turn_has_sign_activity
                    )
                assert_routed_label(state_before, controller_prediction)
                action, explanation = action_for(controller_prediction)
                state_transition = state_manager.handle_prediction(
                    controller_prediction
                )
                if action != state_transition.action:
                    raise RuntimeError("State Manager action mapping mismatch")

                backchannel_feedback_plan = None
                if (
                    controller_prediction == "BACKCHANNEL"
                    and action == "SHORT_FEEDBACK"
                ):
                    backchannel_feedback_plan = feedback_sampler.sample()
                    backchannel_feedback_count += 1

                response_token = None
                response_id_for_record = active_response_id
                response_cancelled = False
                scenario_plan = None
                is_final_reply = False
                if controller_prediction == "COMPLETE":
                    turn_context.consume_turn()
                    turn_context.enter_turn_ready(prefix_end_frame, decision_time)
                    active_response_id, response_token = (
                        _RESPONSE_LIFECYCLE.create_response(run_id)
                    )
                    response_id_for_record = active_response_id
                    log_state_transition(
                        decision_time,
                        prefix_end_frame,
                        state_transition,
                        active_response_id,
                    )
                    scenario_plan = scenario_planner.reserve(decision_time)
                    is_final_reply = (
                        scenario_planner.is_final_reply_opportunity(
                            decision_time
                        )
                        or decision.index == len(decisions)
                    )
                    final_extension = (
                        FINAL_REPLY_EXTENSION_SECONDS
                        if is_final_reply
                        else 0.0
                    )
                    scheduled_response = tts_simulator.schedule(
                        active_response_id,
                        response_token,
                        decision_time,
                        scenario_plan=scenario_plan,
                        final_reply_extension_seconds=final_extension,
                    )
                    if is_final_reply:
                        final_response_id = active_response_id
                        final_reply_extension_applied = True
                    schedule_record = {
                        "run_id": run_id,
                        "global_time": round(decision_time, 6),
                        "global_prefix_end_frame": prefix_end_frame,
                        **scheduled_response.as_log_record(),
                    }
                    state_logger.append(schedule_record)
                    print(
                        "[response-scheduled] "
                        + json.dumps(schedule_record, ensure_ascii=False),
                        flush=True,
                    )
                elif controller_prediction == "BARGE":
                    if active_response_id is not None:
                        registry_cancelled = _RESPONSE_LIFECYCLE.cancel_response(
                            run_id, active_response_id
                        )
                        simulator_cancelled = tts_simulator.cancel(
                            active_response_id
                        )
                        if registry_cancelled != simulator_cancelled:
                            raise RuntimeError(
                                "Response registries disagreed during BARGE"
                            )
                        response_cancelled = registry_cancelled
                    barge_in_count += 1
                    turn_context.enter_listening(
                        prefix_end_frame, decision_time, evidence
                    )
                    log_state_transition(
                        decision_time,
                        prefix_end_frame,
                        state_transition,
                        active_response_id,
                        {
                            "response_cancelled": response_cancelled,
                            "response_status": "CANCELLED_BY_BARGE",
                        },
                    )
                    active_response_id = None

                rgb_source_indices = prepared.rgb_source_indices(
                    prefix_end_frame, task_start_frame
                )
                pose_frame_count = len(
                    prepared.pose_prefix(prefix_end_frame, task_start_frame)[
                        "keypoints"
                    ]
                )
                if min(rgb_source_indices) < task_start_frame:
                    raise RuntimeError("RGB task prefix start was exceeded")
                if max(rgb_source_indices) >= prefix_end_frame:
                    raise RuntimeError("RGB causal prefix end was exceeded")
                if pose_frame_count != prefix_end_frame - task_start_frame:
                    raise RuntimeError("Pose task prefix length mismatch")

                completed_decision_times.append(decision_time)
                prediction_lag = max(
                    0.0, completed.completed_wall_seconds - decision_time
                )
                maximum_lag = max(maximum_lag, prediction_lag)
                ui_update_ms = (time.perf_counter() - ui_started) * 1000.0
                latency_ms = completed.worker_total_ms + ui_update_ms
                inference_latencies_ms.append(completed.worker_total_ms)
                remember_inference_latency(completed.worker_total_ms)
                record = {
                    "record_type": "decision",
                    "run_id": run_id,
                    "video_id": prepared.video_id,
                    "decision_index": decision.index,
                    "decision_interval_seconds": decision_interval,
                    "global_time": round(decision_time, 6),
                    "global_decision_time": round(decision_time, 6),
                    "global_prefix_end_frame": prefix_end_frame,
                    "task_prefix_start_frame": task_start_frame,
                    "task_prefix_end_frame": prefix_end_frame,
                    "task_prefix_start_time": task_start_frame / prepared.fps,
                    "task_prefix_end_time": decision_time,
                    "state_before": state_transition.state_before,
                    "system_state": state_transition.state_before,
                    "state_enter_frame": context_before.state_enter_frame,
                    "state_enter_time": context_before.state_enter_time,
                    "turn_start_frame": context_before.turn_start_frame,
                    "turn_start_time": context_before.turn_start_time,
                    "turn_has_sign_activity": context_before.turn_has_sign_activity,
                    "turn_consumed": context_before.turn_consumed,
                    "current_motion_episode_start": (
                        context_before.current_motion_episode_start
                    ),
                    "raw_prediction": raw_prediction,
                    "raw_confidence": raw_confidence,
                    "all_probabilities": result["probabilities"],
                    "motion_score": evidence.motion_score,
                    "hand_motion_score": evidence.hand_motion_score,
                    "recent_active_fraction": evidence.recent_active_fraction,
                    "sustained_active_duration": (
                        evidence.sustained_active_duration
                    ),
                    "hand_articulation_score": (
                        evidence.hand_articulation_score
                    ),
                    "trajectory_score": evidence.trajectory_score,
                    "causal_confirmation_score": (
                        evidence.causal_confirmation_score
                    ),
                    "causal_sign_evidence": evidence.causal_sign_evidence,
                    "activity": evidence.activity_label,
                    "controller_prediction": controller_prediction,
                    "controller_override_reason": (
                        controller_result.controller_override_reason
                    ),
                    "controller_override": controller_result.controller_override,
                    "controller_state": controller_result.controller_state,
                    "previous_effective_prediction": (
                        controller_result.previous_effective_prediction
                    ),
                    "post_barge_active": controller_result.post_barge_active,
                    "post_barge_until": controller_result.post_barge_until,
                    "backchannel_cooldown_active": (
                        controller_result.backchannel_cooldown_active
                    ),
                    "backchannel_cooldown_until": (
                        controller_result.backchannel_cooldown_until
                    ),
                    "final_prediction": controller_prediction,
                    "system_action": action,
                    "action": action,
                    "state_after": state_transition.state_after,
                    "current_system_state": public_system_state(
                        state_manager.current_system_state
                    ),
                    "current_state": public_system_state(
                        state_manager.current_system_state
                    ),
                    "response_id": response_id_for_record,
                    "response_token": response_token,
                    "response_cancelled": response_cancelled,
                    "is_final_reply": is_final_reply,
                    "scenario_reply_mode": (
                        scenario_plan.scenario_reply_mode
                        if scenario_plan is not None
                        else None
                    ),
                    "scenario_next_sign_start": (
                        scenario_plan.next_sign_start
                        if scenario_plan is not None
                        else None
                    ),
                    "backchannel_feedback_sequence": (
                        backchannel_feedback_plan.sequence
                        if backchannel_feedback_plan is not None
                        else None
                    ),
                    "backchannel_feedback_duration": (
                        backchannel_feedback_plan.duration
                        if backchannel_feedback_plan is not None
                        else None
                    ),
                    "latency_ms": latency_ms,
                    "model_latency_ms": result["model_latency_ms"],
                    "profile_ms": result["profile_ms"],
                    "prediction_lag_seconds": prediction_lag,
                    "display_delay_seconds": display_delay,
                    "display_target_wall_seconds": decision_time + display_delay,
                    "strict_causal_prefix_seconds": [
                        task_start_frame / prepared.fps,
                        decision_time,
                    ],
                }
                pending_displays.append(
                    AlignedPrediction(
                        target_wall_seconds=decision_time + display_delay,
                        decision_time=decision_time,
                        prefix_end_frame=prefix_end_frame,
                        prediction=controller_prediction,
                        action=action,
                        explanation=explanation,
                        record=record,
                    )
                )

                if state_manager.current_system_state != state_before:
                    retire_worker()
                    if state_manager.current_system_state == "LISTENING":
                        create_worker("LISTENING")
                    next_decision = min(next_decision, decision.index)
                    state_changed = True
                    break
                worker.acknowledge_result()

            release_elapsed = time.perf_counter() - wall_started
            while (
                pending_displays
                and release_elapsed + 1e-6
                >= pending_displays[0].target_wall_seconds
            ):
                aligned = pending_displays.popleft()
                release_elapsed = time.perf_counter() - wall_started
                estimated_display_time = min(
                    prepared.duration,
                    max(0.0, release_elapsed - display_delay),
                )
                alignment_error = estimated_display_time - aligned.decision_time
                alignment_errors_seconds.append(alignment_error)
                aligned.record.update(
                    {
                        "display_release_wall_seconds": release_elapsed,
                        "estimated_display_video_time": estimated_display_time,
                        "alignment_error_seconds": alignment_error,
                    }
                )
                logger.append(aligned.record)
                print(
                    "[aligned-decision] "
                    + json.dumps(aligned.record, ensure_ascii=False),
                    flush=True,
                )
                timeline_rows.append(
                    [
                        round(aligned.decision_time, 6),
                        aligned.prediction,
                        simple_action_for(aligned.prediction),
                    ]
                )
                interaction_rows.append(
                    [
                        round(aligned.decision_time, 6),
                        aligned.prediction,
                        f"{aligned.record['state_before']} → {aligned.record['state_after']}",
                    ]
                )
                if aligned.prediction == "COMPLETE":
                    interaction_rows.append(
                        [
                            round(aligned.decision_time, 6),
                            "TURN_READY",
                            "Metadata-aware TTS response scheduled",
                        ]
                    )
                elif aligned.prediction == "BARGE":
                    interaction_rows.append(
                        [
                            round(aligned.decision_time, 6),
                            "LISTENING",
                            "Output cancelled; listening resumed",
                        ]
                    )
                running = transition(
                    running,
                    "RUNNING",
                    prediction_history=timeline_rows,
                    interaction_history=interaction_rows,
                    current_system_state=state_manager.current_system_state,
                    current_time=aligned.decision_time,
                    current_prefix=aligned.prefix_end_frame,
                    latest_prediction=aligned.prediction,
                )
                audio_update = gr.update()
                signal_update = gr.update()
                if aligned.prediction == "BARGE":
                    audio_update = None
                    signal_update = (
                        f"STOP:BARGE:{run_id}:{aligned.record['response_id']}"
                    )
                    if active_full_waveform is not None:
                        log_waveform_event(
                            event="STOP",
                            waveform_mode="FULL_REPLY",
                            trigger="CANCELLED_BY_BARGE",
                            interaction_wall_time=release_elapsed,
                            video_time=aligned.decision_time,
                            start_time=float(active_full_waveform["start_time"]),
                            planned_end_time=float(
                                active_full_waveform["planned_end_time"]
                            ),
                            actual_end_time=release_elapsed,
                            response_id=str(active_full_waveform["response_id"]),
                            duration=float(active_full_waveform["duration"]),
                        )
                        active_full_waveform = None
                        waveform_stop_count += 1
                elif (
                    aligned.prediction == "BACKCHANNEL"
                    and aligned.action == "SHORT_FEEDBACK"
                ):
                    duration = float(
                        aligned.record["backchannel_feedback_duration"]
                    )
                    sequence = int(
                        aligned.record["backchannel_feedback_sequence"]
                    )
                    feedback_id = f"backchannel:{run_id}:{sequence}"
                    active_short_feedback = {
                        "response_id": feedback_id,
                        "start_time": release_elapsed,
                        "planned_end_time": release_elapsed + duration,
                        "duration": duration,
                    }
                    log_waveform_event(
                        event="START",
                        waveform_mode="SHORT_FEEDBACK",
                        trigger="BACKCHANNEL",
                        interaction_wall_time=release_elapsed,
                        video_time=aligned.decision_time,
                        start_time=release_elapsed,
                        planned_end_time=release_elapsed + duration,
                        actual_end_time=None,
                        response_id=feedback_id,
                        duration=duration,
                    )
                    signal_update = (
                        f"SHORT_FEEDBACK_START:{feedback_id}:{duration:.6f}"
                    )

                last_effective = aligned.prediction
                last_raw = (
                    f"{aligned.record['raw_prediction']} "
                    f"({aligned.record['raw_confidence']:.4f})"
                )
                last_action = simple_action_for(aligned.prediction)
                last_motion = aligned.record["motion_score"]
                last_hand_motion = aligned.record["hand_motion_score"]
                last_activity = aligned.record["activity"]
                last_override = aligned.record["controller_override"]
                last_override_reason = aligned.record[
                    "controller_override_reason"
                ]
                last_idle_debug = idle_debug_record(
                    aligned.record["raw_prediction"],
                    aligned.record["controller_prediction"],
                    "MODEL"
                    if aligned.record["controller_prediction"] == "IDLE"
                    else None,
                    aligned.record["controller_override_reason"],
                )
                if (
                    aligned.prediction == "COMPLETE"
                    and current_reference_segment is not None
                ):
                    last_subtitle_text = current_reference_segment.text
                    last_subtitle_debug = subtitle_debug_record(
                        subtitle_manager, current_reference_segment
                    )
                    subtitle_change_count += 1
                    subtitle_record = {
                        "run_id": run_id,
                        "event": "REFERENCE_SENTENCE_PUBLISHED",
                        "video_time": round(aligned.decision_time, 6),
                        "segment_index": current_reference_segment.segment_index,
                        "semantic_id": current_reference_segment.semantic_id,
                        "text": last_subtitle_text,
                        "subtitle_source": SUBTITLE_SOURCE,
                        "subtitle_learned": SUBTITLE_LEARNED,
                    }
                    subtitle_logger.append(subtitle_record)
                    print(
                        "[reference-sentence] "
                        + json.dumps(subtitle_record, ensure_ascii=False),
                        flush=True,
                    )
                yield (
                    running,
                    gr.update(),
                    format_prediction_display(aligned.prediction),
                    format_model_decision_time(
                        aligned.decision_time,
                        is_final_model_decision=(
                            aligned.record["decision_index"] == len(decisions)
                        ),
                    ),
                    last_action,
                    last_raw,
                    f"{last_motion:.6f}",
                    f"{last_hand_motion:.6f}",
                    last_activity,
                    last_override,
                    last_override_reason,
                    public_state_for_prediction(aligned.prediction),
                    last_idle_debug,
                    tts_simulator.debug_record(),
                    list(timeline_rows),
                    list(interaction_rows),
                    last_subtitle_text or "—",
                    last_subtitle_debug,
                    audio_update,
                    signal_update,
                )
                release_elapsed = time.perf_counter() - wall_started

            playback_done = display_playback_completed_seconds is not None
            inference_done = next_decision == len(decisions) and (
                worker is None or worker.is_idle()
            )
            alignment_done = not pending_displays
            no_results = worker is None or not worker.has_results()
            lifecycle_done = _RESPONSE_LIFECYCLE.active_response_id(run_id) is None
            short_feedback_done = active_short_feedback is None
            if (
                playback_done
                and inference_done
                and alignment_done
                and no_results
                and lifecycle_done
                and short_feedback_done
            ):
                break
            if state_changed:
                continue
            time.sleep(0.02)

        if len(timeline_rows) != len(completed_decision_times):
            raise RuntimeError("A completed prediction was lost by display alignment")
        retire_worker()
        if state_manager.current_system_state != "LISTENING":
            raise RuntimeError("END_OF_STREAM requires completed final reply")
        end_of_stream = controller.end_of_stream(state_manager)
        end_interaction_wall_time = time.perf_counter() - wall_started
        log_state_transition(
            end_interaction_wall_time,
            prepared.total_frames,
            end_of_stream,
            event_time_domain="INTERACTION",
            video_time=prepared.duration,
        )
        interaction_rows.append(
            [
                round(end_interaction_wall_time, 6),
                "END_OF_STREAM",
                (
                    f"{end_of_stream.state_before} → {end_of_stream.state_after} "
                    f"(video_time={prepared.duration:.6f})"
                ),
            ]
        )

        average_inference_latency_ms = (
            sum(inference_latencies_ms) / len(inference_latencies_ms)
            if inference_latencies_ms
            else 0.0
        )
        average_alignment_error_ms = (
            sum(alignment_errors_seconds) / len(alignment_errors_seconds) * 1000.0
            if alignment_errors_seconds
            else 0.0
        )
        idle_observed = any(row[1] == "IDLE" for row in timeline_rows)
        barge_observed = any(row[1] == "BARGE" for row in timeline_rows)
        balanced_demo_success = idle_observed and barge_observed
        final_waveform_sync_error_seconds = (
            abs(
                final_waveform_stop_wall_time
                - final_response_finished_wall_time
            )
            if final_waveform_stop_wall_time is not None
            and final_response_finished_wall_time is not None
            else None
        )
        final_waveform_synced_with_response = (
            final_reply_extension_applied
            and final_waveform_sync_error_seconds is not None
            and final_waveform_sync_error_seconds < 0.1
        )
        summary = {
            "run_id": run_id,
            "video_id": prepared.video_id,
            "video_path": prepared.path,
            "video_display_name": video_display_name(prepared.path),
            "video_duration_seconds": prepared.duration,
            "video_player_compact": True,
            "reference_annotation_note_visible": False,
            "reference_sentence_font_px": 22,
            "backend_stream_elapsed_seconds": backend_stream_completed_seconds,
            "playback_elapsed_seconds": display_playback_completed_seconds,
            "chunk_size_seconds": CHUNK_SIZE_SECONDS,
            "decision_interval_seconds": decision_interval,
            "decision_interval_fixed": True,
            "fixed_decision_interval_seconds": DECISION_INTERVAL_SECONDS,
            "demo_scenario_mode": DEMO_SCENARIO_MODE,
            "pause_duration_validation": (
                demo_validation.pause_duration_validation
            ),
            "pause_durations": [
                pause.pause_duration for pause in demo_validation.pauses
            ],
            "planned_reply_modes": list(
                demo_validation.planned_reply_modes
            ),
            "idle_observed": idle_observed,
            "barge_observed": barge_observed,
            "balanced_demo_success": balanced_demo_success,
            "final_reply_extension_seconds": FINAL_REPLY_EXTENSION_SECONDS,
            "final_reply_extension_applied": final_reply_extension_applied,
            "post_eos_quiet_hold_seconds": POST_EOS_DISPLAY_HOLD_SECONDS,
            "final_waveform_synced_with_response": (
                final_waveform_synced_with_response
            ),
            "final_waveform_sync_error_seconds": (
                final_waveform_sync_error_seconds
            ),
            "display_delay_seconds": display_delay,
            "display_delay_source_p95_seconds": delay_source_p95,
            "average_inference_latency_ms": average_inference_latency_ms,
            "p95_inference_latency_ms": percentile_95(inference_latencies_ms),
            "average_alignment_error_ms": average_alignment_error_ms,
            "p95_absolute_alignment_error_ms": percentile_95(
                [abs(value) * 1000.0 for value in alignment_errors_seconds]
            ),
            "scheduled_decision_times": [item.end_seconds for item in decisions],
            "completed_decision_times": completed_decision_times,
            "timeline_rows_at_finish": len(timeline_rows),
            "timeline_persisted_at_finish": True,
            "interaction_timeline_rows_at_finish": len(interaction_rows),
            "initial_system_state": "LISTENING",
            "final_system_state": public_system_state(
                state_manager.current_system_state
            ),
            "automatic_state_management": True,
            "interaction_controller": True,
            "controller_preserves_raw_prediction": True,
            "raw_prediction_preserved": True,
            "post_barge_protection_seconds": POST_BARGE_PROTECTION_SECONDS,
            "backchannel_cooldown_seconds": BACKCHANNEL_COOLDOWN_SECONDS,
            "controller_override_count": controller_override_count,
            "post_barge_protection_count": post_barge_protection_count,
            "backchannel_cooldown_count": backchannel_cooldown_count,
            "backchannel_feedback_duration_range": list(
                BACKCHANNEL_FEEDBACK_DURATION_RANGE
            ),
            "waveform_animation_engine": "FRONTEND_REQUEST_ANIMATION_FRAME",
            "waveform_target_fps": 60,
            "waveform_scrolling": False,
            "waveform_stationary_animation": True,
            "backchannel_feedback_count": backchannel_feedback_count,
            "waveform_full_reply_count": waveform_full_reply_count,
            "waveform_stop_count": waveform_stop_count,
            "waveform_timeline_path": str(waveform_logger.path),
            "turn_relative_causal_prefix": True,
            "v1_8_idle_aware_model": True,
            "model_learned_states": {
                "LISTENING": list(TURN_LABELS),
                "SPEAKING": list(BARGE_LABELS),
            },
            "classifier_output_dimensions": {
                "turn_state": len(TURN_LABELS),
                "barge_in": len(BARGE_LABELS),
            },
            "controller_idle_enabled": False,
            "learned_idle_count": learned_idle_count,
            "idle_during_speaking_count": idle_during_speaking_count,
            "idle_after_turn_started_count": idle_after_turn_started_count,
            "subtitle_enabled": subtitle_manager.enabled,
            "subtitle_source": SUBTITLE_SOURCE,
            "subtitle_learned": SUBTITLE_LEARNED,
            "subtitle_segment_count": len(subtitle_manager.segments),
            "subtitle_change_count": subtitle_change_count,
            "reference_annotation_source": (
                str(subtitle_manager.annotation_source)
                if subtitle_manager.annotation_source
                else None
            ),
            "concat_metadata_source": (
                str(subtitle_manager.metadata_source)
                if subtitle_manager.metadata_source
                else None
            ),
            "subtitle_segments": subtitle_manager.as_records(),
            "random_seed": _RANDOM_SEED,
            "normal_reply_probability": NORMAL_REPLY_PROBABILITY,
            "long_reply_probability": LONG_REPLY_PROBABILITY,
            "thinking_time_range": list(THINKING_TIME_RANGE),
            "normal_tts_duration_range": list(NORMAL_TTS_DURATION_RANGE),
            "long_tts_duration_range": list(LONG_TTS_DURATION_RANGE),
            **tts_simulator.counts(),
            "response_started_count": response_started_count,
            "response_finished_count": response_finished_count,
            "normal_turn_count": normal_turn_count,
            "barge_in_count": barge_in_count,
            "normal_turn_ratio": (
                normal_turn_count / (normal_turn_count + barge_in_count)
                if normal_turn_count + barge_in_count
                else 0.0
            ),
            "barge_in_ratio": (
                barge_in_count / (normal_turn_count + barge_in_count)
                if normal_turn_count + barge_in_count
                else 0.0
            ),
            "stale_response_callback_count": (
                _RESPONSE_LIFECYCLE.stale_callback_count(run_id)
            ),
            "end_of_stream": {
                "state_before": end_of_stream.state_before,
                "event": end_of_stream.event,
                "action": end_of_stream.action,
                "state_after": end_of_stream.state_after,
                "event_time_domain": "INTERACTION",
                "video_time": prepared.duration,
                "interaction_wall_time": end_interaction_wall_time,
            },
            "requests_submitted": retired_worker_submitted,
            "predictions_completed": len(completed_decision_times),
            "worker_predictions_completed": retired_worker_completed,
            "requests_dropped_latest_only": retired_worker_dropped,
            "maximum_prediction_lag_seconds": maximum_lag,
            "pose_cache_frames": len(prepared.pose_keypoints),
            "pose_precompute_ms": prepared.pose_extraction_ms,
            "strict_causal": True,
            "predictions_released_at_display_time": True,
        }
        summary_path = LOGS_DIR / f"run_summary_{run_id}.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("[summary] " + json.dumps(summary, ensure_ascii=False), flush=True)
        finished = transition(
            running,
            "FINISHED",
            prediction_history=timeline_rows,
            interaction_history=interaction_rows,
            current_system_state=state_manager.current_system_state,
        )
        yield (
            finished,
            gr.update(interactive=True),
            gr.update(),
            gr.update(),
            gr.update(),
            "—",
            "0.000000",
            "0.000000",
            "NO_ACTIVITY",
            "END_OF_STREAM",
            "END_OF_STREAM",
            public_system_state(state_manager.current_system_state),
            last_idle_debug,
            tts_simulator.debug_record(),
            list(timeline_rows),
            list(interaction_rows),
            last_subtitle_text or "—",
            last_subtitle_debug,
            gr.update(),
            gr.update(),
        )
    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(f"运行失败：{type(error).__name__}: {error}") from error
    finally:
        retire_worker()
        _RESPONSE_LIFECYCLE.close_run(run_id)


def stop_stream(state: dict | None):
    """Stop playback while returning the stored timeline unchanged."""
    token = f"STOP:manual:{uuid.uuid4().hex[:8]}"
    if state and state.get("pose_cache_ready"):
        state = transition(state, "FINISHED")
    else:
        state = empty_state()
    history = list(state.get("prediction_history", []))
    interaction_history = list(state.get("interaction_history", []))
    return (
        state,
        gr.update(interactive=can_start(state)),
        history,
        interaction_history,
        public_system_state(state.get("current_system_state", "LISTENING")),
        None,
        token,
    )


def reset_demo(video_path: str | None):
    """Reset is the only control that clears both cache and the visible session."""
    _PREPARED.clear()
    _SUBTITLES.clear()
    token = f"STOP:reset:{uuid.uuid4().hex[:8]}"
    return (
        gr.update(),
        format_test_video(video_path),
        empty_state(),
        gr.update(interactive=False),
        format_prediction_display("IDLE"),
        "—",
        simple_action_for("IDLE"),
        "—",
        "0.000000",
        "0.000000",
        "NO_ACTIVITY",
        "NONE",
        "NONE",
        "LISTENING",
        idle_debug_record(),
        TTSResponseSimulator().debug_record(),
        [],
        [[0.0, "LISTENING", "Initial state"]],
        "—",
        subtitle_debug_record(),
        None,
        token,
    )


CSS = """
#result-card { min-height: 120px; padding: 14px 18px; border: 1px solid #d7dce2; border-radius: 10px; }
#upload-video video { width: 100%; max-height: 560px; object-fit: contain; }
#video-header { align-items: center; }
#video-header h3 { margin: 0; }
#test-video-name { text-align: right; overflow-wrap: anywhere; color: #475569; }
#test-video-name p { margin: 0; }
#reference-subtitle { min-height: 0; padding: 10px 14px; text-align: center; font-size: 22px; font-weight: 500; line-height: 1.45; border: 1px solid #d7dce2; border-radius: 10px; overflow-wrap: anywhere; }
.compact-field textarea { font-size: 1rem; font-weight: 600; }
#tts-waveform-panel { min-height: 168px; padding: 9px 22px 11px; border: 1px solid #dce2e8; border-radius: 12px; background: #fbfcfd; }
.tts-waveform-toolbar { display: flex; justify-content: flex-end; min-height: 22px; }
#tts-waveform-status { color: #64748b; font-size: 0.78rem; letter-spacing: 0.01em; }
#tts-waveform-status-dot { display: inline-block; width: 7px; height: 7px; margin-right: 6px; border-radius: 50%; background: #94a3b8; }
#tts-waveform-status[data-mode="FULL_REPLY"] #tts-waveform-status-dot { background: #4f7fac; }
#tts-waveform-status[data-mode="SHORT_FEEDBACK"] #tts-waveform-status-dot { background: #7899b8; }
#tts-waveform-svg { display: block; width: 100%; height: 108px; }
#tts-waveform-baseline { stroke: #e2e8f0; stroke-width: 1; }
#tts-waveform-path, #tts-waveform-path-lower { fill: none; stroke: #587fa4; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; filter: drop-shadow(0 1px 1px rgba(82, 127, 168, 0.1)); }
#tts-waveform-path-lower { stroke: #6d90b0; opacity: 0.86; }
.tts-waveform-caption { margin-top: 2px; color: #718096; font-size: 0.82rem; text-align: center; }
"""


with gr.Blocks(title="SignDuplex V3.7.0 Demo", css=CSS) as demo:
    gr.Markdown(
        "# SignDuplex\n"
        "### Real-Time Sign-Aware Full-Duplex Interaction Demo"
    )
    runtime = gr.State(value=empty_state())

    with gr.Row():
        with gr.Column(scale=11):
            with gr.Row(elem_id="video-header"):
                gr.Markdown("### Video Player")
                test_video_name = gr.Markdown(
                    format_test_video(None), elem_id="test-video-name"
                )
            upload = gr.Video(
                label="Sign Video",
                sources=["upload"],
                format="mp4",
                include_audio=False,
                elem_id="upload-video",
            )
            gr.Markdown("### Reference Sentence")
            reference_subtitle = gr.Markdown("—", elem_id="reference-subtitle")
            subtitle_debug = gr.State(value=subtitle_debug_record())
        with gr.Column(scale=9):
            current_system_state = gr.Textbox(
                "LISTENING",
                label="Current System State",
                interactive=False,
                elem_classes="compact-field",
            )
            with gr.Row():
                start_button = gr.Button(
                    "Start", variant="primary", interactive=False
                )
                stop_button = gr.Button("Stop", variant="stop")
                reset_button = gr.Button("Reset")

            gr.Markdown("### Current Prediction")
            current_prediction = gr.Markdown(
                format_prediction_display("IDLE"), elem_id="result-card"
            )
            with gr.Row():
                decision_time = gr.Textbox(
                    "—",
                    label="Decision Time",
                    interactive=False,
                    elem_classes="compact-field",
                )
                system_action = gr.Textbox(
                    simple_action_for("IDLE"),
                    label="System Action",
                    interactive=False,
                    elem_classes="compact-field",
                )
            gr.Markdown("### Simulated TTS Waveform")
            gr.HTML(
                """
<div id="tts-waveform-panel">
  <div class="tts-waveform-toolbar">
    <span id="tts-waveform-status" data-mode="QUIET">
      <span id="tts-waveform-status-dot"></span><span id="tts-waveform-status-text">Silent</span>
    </span>
  </div>
  <svg id="tts-waveform-svg" viewBox="0 0 640 140"
       preserveAspectRatio="none" role="img"
       aria-label="Simulated system speaking waveform">
    <line id="tts-waveform-baseline" x1="0" y1="70" x2="640" y2="70"></line>
    <path id="tts-waveform-path" data-mode="QUIET" d="M 0 70 L 640 70"></path>
    <path id="tts-waveform-path-lower" data-mode="QUIET"
          transform="translate(0 140) scale(1 -1)"
          d="M 0 70 L 640 70"></path>
  </svg>
  <div class="tts-waveform-caption">
    Waveform is synchronized with the system reply duration.
  </div>
</div>
"""
            )
            # Internal stream outputs remain state-only, so the public UI has no
            # raw/controller/motion debug panel while logs keep audit details.
            raw_prediction_debug = gr.State(value="—")
            motion_score = gr.State(value="0.000000")
            hand_motion_score = gr.State(value="0.000000")
            activity = gr.State(value="NO_ACTIVITY")
            controller_override = gr.State(value="NONE")
            controller_override_reason = gr.State(value="NONE")
            idle_debug = gr.State(value=idle_debug_record())
            tts_debug = gr.State(value=TTSResponseSimulator().debug_record())

    timeline = gr.Dataframe(
        headers=TIMELINE_HEADERS,
        datatype=["number", "str", "str"],
        value=[],
        label="Prediction Timeline",
        interactive=False,
        wrap=True,
        height=320,
    )
    interaction_timeline = gr.Dataframe(
        headers=INTERACTION_HEADERS,
        datatype=["number", "str", "str"],
        value=[[0.0, "LISTENING", "Initial state"]],
        label="Interaction Timeline",
        interactive=False,
        wrap=True,
        height=320,
        visible=False,
    )

    system_audio = gr.Audio(
        type="filepath",
        autoplay=True,
        interactive=False,
        visible=False,
        elem_id="system-audio",
    )
    audio_signal = gr.Textbox(visible=False)

    stream_outputs = [
        runtime,
        start_button,
        current_prediction,
        decision_time,
        system_action,
        raw_prediction_debug,
        motion_score,
        hand_motion_score,
        activity,
        controller_override,
        controller_override_reason,
        current_system_state,
        idle_debug,
        tts_debug,
        timeline,
        interaction_timeline,
        reference_subtitle,
        subtitle_debug,
        system_audio,
        audio_signal,
    ]

    start_event = start_button.click(
        stream_demo,
        inputs=[upload, runtime],
        outputs=stream_outputs,
        concurrency_limit=1,
        js="""
        (video, runtime) => {
          if (window.__signDuplexPlaybackTimer) {
            clearTimeout(window.__signDuplexPlaybackTimer);
          }
          const player = document.querySelector("#upload-video video");
          if (player) { player.pause(); player.currentTime = 0; }
          return [video, runtime];
        }
        """,
    )
    upload.change(
        prepare_upload,
        inputs=upload,
        outputs=stream_outputs,
        cancels=[start_event],
        concurrency_limit=1,
        js="""(video) => {
          if (window.__signDuplexPlaybackTimer) {
            clearTimeout(window.__signDuplexPlaybackTimer);
          }
          document.querySelectorAll("#upload-video video, #system-audio audio")
            .forEach((media) => { media.pause(); media.currentTime = 0; });
          return video;
        }""",
    )
    upload.change(
        format_test_video,
        inputs=upload,
        outputs=test_video_name,
        queue=False,
    )
    stop_button.click(
        stop_stream,
        inputs=runtime,
        outputs=[
            runtime,
            start_button,
            timeline,
            interaction_timeline,
            current_system_state,
            system_audio,
            audio_signal,
        ],
        cancels=[start_event],
        queue=False,
        js="""(runtime) => {
          if (window.__signDuplexPlaybackTimer) {
            clearTimeout(window.__signDuplexPlaybackTimer);
          }
          document.querySelectorAll("#upload-video video, #system-audio audio")
            .forEach((media) => media.pause());
          return runtime;
        }""",
    )
    reset_button.click(
        reset_demo,
        inputs=upload,
        outputs=[upload, test_video_name] + stream_outputs,
        cancels=[start_event],
        queue=False,
        js="""(video) => {
          if (window.__signDuplexPlaybackTimer) {
            clearTimeout(window.__signDuplexPlaybackTimer);
          }
          document.querySelectorAll("#upload-video video, #system-audio audio")
            .forEach((media) => { media.pause(); media.currentTime = 0; });
          return video;
        }""",
    )
    audio_signal.change(
        fn=None,
        inputs=audio_signal,
        outputs=None,
        js="""(signal) => {
          const waveform = (() => {
            if (window.__signDuplexWaveform) {
              return window.__signDuplexWaveform;
            }
            const anchorCount = 64;
            const targetFps = 60;
            const frameInterval = 1000 / targetFps;
            const state = {
              mode: "QUIET",
              frame: null,
              lastFrameAt: 0,
              startedAt: 0,
              duration: null,
              token: null,
              shortTimer: null,
              points: [],
              modulationEpoch: -1,
              stoppingStartedAt: null,
              stoppingDuration: 0,
            };
            const path = () => document.querySelector("#tts-waveform-path");
            const lowerPath = () => document.querySelector("#tts-waveform-path-lower");
            const status = () => document.querySelector("#tts-waveform-status");
            const statusText = () => document.querySelector("#tts-waveform-status-text");
            const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
            const smoothstep = (value) => {
              const x = clamp(value, 0, 1);
              return x * x * (3 - 2 * x);
            };
            const setStatus = (mode) => {
              const node = status();
              const text = statusText();
              if (node) { node.dataset.mode = mode; }
              if (text) {
                text.textContent = mode === "FULL_REPLY"
                  ? "Speaking"
                  : mode === "SHORT_FEEDBACK" ? "Short feedback" : "Silent";
              }
            };
            const draw = (samples, mode) => {
              const node = path();
              if (!node) { return; }
              const width = 640;
              const center = 70;
              const points = samples.map((sample, index) => ({
                x: index * width / (samples.length - 1),
                y: center - sample,
              }));
              let value = `M ${points[0].x.toFixed(2)} ${points[0].y.toFixed(2)}`;
              for (let index = 1; index < points.length - 1; index += 1) {
                const midpointX = (points[index].x + points[index + 1].x) / 2;
                const midpointY = (points[index].y + points[index + 1].y) / 2;
                value += ` Q ${points[index].x.toFixed(2)} ${points[index].y.toFixed(2)}`;
                value += ` ${midpointX.toFixed(2)} ${midpointY.toFixed(2)}`;
              }
              const last = points[points.length - 1];
              value += ` L ${last.x.toFixed(2)} ${last.y.toFixed(2)}`;
              node.setAttribute("d", value);
              node.dataset.mode = mode;
              const lower = lowerPath();
              if (lower) {
                lower.setAttribute("d", value);
                lower.dataset.mode = mode;
              }
            };
            const hashToken = (token) => {
              let value = 2166136261;
              for (const character of String(token || "waveform")) {
                value ^= character.charCodeAt(0);
                value = Math.imul(value, 16777619);
              }
              return value >>> 0;
            };
            const unitNoise = (seed) => {
              let value = seed >>> 0;
              value ^= value << 13;
              value ^= value >>> 17;
              value ^= value << 5;
              return (value >>> 0) / 4294967295;
            };
            const buildPoints = (token) => {
              const seed = hashToken(token);
              return Array.from({ length: anchorCount }, (_, index) => {
                const first = unitNoise(seed + index * 2654435761);
                const second = unitNoise(seed ^ (index + 1) * 2246822519);
                const third = unitNoise(seed + (index + 7) * 3266489917);
                return {
                  baseAmplitude: 0.72 + first * 0.28,
                  phase: second * Math.PI * 2,
                  frequency: 3.5 + third * 2.8,
                  currentAmplitude: 0,
                  targetAmplitude: 0.65 + first * 0.35,
                };
              });
            };
            const begin = (mode, duration, token) => {
              state.mode = mode;
              state.startedAt = performance.now();
              state.duration = duration;
              state.token = token;
              state.lastFrameAt = 0;
              state.points = buildPoints(token);
              state.modulationEpoch = -1;
              state.stoppingStartedAt = null;
              state.stoppingDuration = 0;
              setStatus(mode);
              if (state.frame === null) {
                state.frame = requestAnimationFrame(tick);
              }
            };
            const stationarySamples = (timestamp) => {
              const elapsed = (timestamp - state.startedAt) / 1000;
              const isFullReply = state.mode === "FULL_REPLY";
              const fadeInSeconds = isFullReply ? 0.15 : 0.10;
              const fadeOutSeconds = isFullReply ? 0.185 : 0.125;
              let envelope = smoothstep(elapsed / fadeInSeconds);
              if (Number.isFinite(state.duration)) {
                envelope *= smoothstep((state.duration - elapsed) / fadeOutSeconds);
              }
              if (state.stoppingStartedAt !== null) {
                const stopProgress =
                  (timestamp - state.stoppingStartedAt) / state.stoppingDuration;
                envelope *= 1 - smoothstep(stopProgress);
              }
              const modulationEpoch = Math.floor(elapsed / 0.24);
              if (modulationEpoch !== state.modulationEpoch) {
                state.modulationEpoch = modulationEpoch;
                const tokenSeed = hashToken(state.token);
                state.points.forEach((point, index) => {
                  const modulation = unitNoise(
                    tokenSeed + modulationEpoch * 374761393 + index * 668265263
                  );
                  point.targetAmplitude = 0.52 + modulation * 0.48;
                });
              }
              const amplitudeScale = isFullReply ? 49 : 31;
              const rhythm =
                0.72 + 0.28 * smoothstep((Math.sin(elapsed * 6.1) + 1) / 2);
              return state.points.map((point, index) => {
                point.currentAmplitude +=
                  (point.targetAmplitude - point.currentAmplitude) * 0.12;
                const position = index / (anchorCount - 1);
                const edgeEnvelope = Math.pow(Math.sin(Math.PI * position), 0.62);
                const carrier = Math.abs(
                  0.55 * Math.sin(elapsed * point.frequency + point.phase)
                  + 0.30 * Math.sin(
                    elapsed * point.frequency * 1.73 + point.phase * 0.61
                  )
                  + 0.15 * Math.sin(
                    elapsed * point.frequency * 2.37 + point.phase * 1.31
                  )
                );
                return amplitudeScale * envelope * rhythm * edgeEnvelope
                  * point.baseAmplitude * point.currentAmplitude
                  * (0.30 + carrier * 0.70);
              });
            };
            const toQuiet = (token = null) => {
              if (token !== null && token !== state.token) { return; }
              state.mode = "QUIET";
              state.token = null;
              state.duration = null;
              state.points = [];
              state.stoppingStartedAt = null;
              state.stoppingDuration = 0;
              if (state.shortTimer !== null) { clearTimeout(state.shortTimer); }
              state.shortTimer = null;
              if (state.frame !== null) { cancelAnimationFrame(state.frame); }
              state.frame = null;
              draw(Array(anchorCount).fill(0), "QUIET");
              setStatus("QUIET");
            };
            const stopWithFade = (durationMs) => {
              if (state.mode === "QUIET") { return; }
              state.stoppingStartedAt = performance.now();
              state.stoppingDuration = durationMs;
              if (state.shortTimer !== null) { clearTimeout(state.shortTimer); }
              state.shortTimer = setTimeout(() => toQuiet(), durationMs);
              if (state.frame === null) {
                state.frame = requestAnimationFrame(tick);
              }
            };
            const tick = (timestamp) => {
              if (state.mode === "QUIET") {
                state.frame = null;
                return;
              }
              if (!state.lastFrameAt) { state.lastFrameAt = timestamp - frameInterval; }
              if (timestamp - state.lastFrameAt >= frameInterval * 0.85) {
                state.lastFrameAt = timestamp;
                draw(stationarySamples(timestamp), state.mode);
              }
              state.frame = requestAnimationFrame(tick);
            };
            window.__signDuplexWaveform = {
              startFullReply: (duration, token) => {
                if (state.shortTimer !== null) { clearTimeout(state.shortTimer); }
                state.shortTimer = null;
                begin("FULL_REPLY", duration, token);
              },
              startShortFeedback: (duration, token) => {
                if (state.mode === "FULL_REPLY") { return; }
                if (state.shortTimer !== null) { clearTimeout(state.shortTimer); }
                begin("SHORT_FEEDBACK", duration, token);
                state.shortTimer = setTimeout(
                  () => toQuiet(token), duration * 1000
                );
              },
              stopNormal: () => stopWithFade(185),
              stopForBarge: () => stopWithFade(65),
              resetQuiet: () => toQuiet(),
            };
            window.__signDuplexWaveform.resetQuiet();
            return window.__signDuplexWaveform;
          })();
          if (signal && signal.startsWith("TTS_START:")) {
            const parts = signal.split(":");
            const duration = Number(parts.at(-1));
            const token = parts.slice(1, -1).join(":");
            if (Number.isFinite(duration)) {
              waveform.startFullReply(duration, token);
            }
          }
          if (signal && signal.startsWith("SHORT_FEEDBACK_START:")) {
            const parts = signal.split(":");
            const duration = Number(parts.at(-1));
            const token = parts.slice(1, -1).join(":");
            if (Number.isFinite(duration)) {
              waveform.startShortFeedback(duration, token);
            }
          }
          if (signal && signal.startsWith("PLAY_VIDEO:")) {
            waveform.resetQuiet();
            const delaySeconds = Number(signal.split(":").at(-1));
            const player = document.querySelector("#upload-video video");
            if (window.__signDuplexPlaybackTimer) {
              clearTimeout(window.__signDuplexPlaybackTimer);
            }
            if (player && Number.isFinite(delaySeconds)) {
              player.pause();
              player.muted = true;
              player.currentTime = 0;
              window.__signDuplexPlaybackTimer = setTimeout(() => {
                player.play().catch(() => {});
              }, delaySeconds * 1000);
            }
          }
          if (signal && signal.startsWith("STOP:")) {
            if (signal.startsWith("STOP:BARGE:")) {
              waveform.stopForBarge();
            } else {
              waveform.stopNormal();
            }
            document.querySelectorAll("#system-audio audio").forEach((audio) => {
              audio.pause(); audio.currentTime = 0;
            });
          }
        }""",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    cli_args = parser.parse_args()
    _RANDOM_SEED = cli_args.random_seed
    no_proxy = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
    local_hosts = "localhost,127.0.0.1,0.0.0.0"
    os.environ["NO_PROXY"] = f"{no_proxy},{local_hosts}".strip(",")
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    validate_paths()
    get_model(warmup=True)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=HOST,
        server_port=PORT,
        share=False,
        show_error=True,
        allowed_paths=[str(Path(__file__).resolve().parent)],
    )
