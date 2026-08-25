"""Single source of truth for upload readiness and demo lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PHASES = {"EMPTY", "PREPARING", "READY", "RUNNING", "FINISHED"}


def runtime_state(
    phase: str,
    *,
    video_id: str | None = None,
    prepared_key: str | None = None,
    video_path: str | None = None,
    pose_cache_ready: bool = False,
    prediction_history: list[list[Any]] | None = None,
    interaction_history: list[list[Any]] | None = None,
    current_system_state: str = "LISTENING",
    current_time: float = 0.0,
    current_prefix: int = 0,
    latest_prediction: str | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"Unknown runtime phase: {phase}")
    if pose_cache_ready != bool(video_id and prepared_key and video_path):
        raise ValueError("pose_cache_ready must match the prepared video identity")
    return {
        "phase": phase,
        "video_id": video_id,
        "prepared_key": prepared_key,
        "video_path": str(Path(video_path).resolve()) if video_path else None,
        "pose_cache_ready": pose_cache_ready,
        "prediction_history": list(prediction_history or []),
        "interaction_history": list(interaction_history or []),
        "current_system_state": current_system_state,
        "current_time": float(current_time),
        "current_prefix": int(current_prefix),
        "latest_prediction": latest_prediction,
    }


def empty_state() -> dict[str, Any]:
    return runtime_state("EMPTY")


def can_start(state: dict[str, Any] | None) -> bool:
    return bool(
        state
        and state.get("phase") in {"READY", "FINISHED"}
        and state.get("pose_cache_ready") is True
        and state.get("video_id")
        and state.get("prepared_key")
        and state.get("video_path")
    )


def transition(state: dict[str, Any], phase: str, **updates: Any) -> dict[str, Any]:
    """Return a complete state while preserving cache and prediction history."""
    values = {
        "video_id": state.get("video_id"),
        "prepared_key": state.get("prepared_key"),
        "video_path": state.get("video_path"),
        "pose_cache_ready": state.get("pose_cache_ready", False),
        "prediction_history": state.get("prediction_history", []),
        "interaction_history": state.get("interaction_history", []),
        "current_system_state": state.get(
            "current_system_state", "LISTENING"
        ),
        "current_time": state.get("current_time", 0.0),
        "current_prefix": state.get("current_prefix", 0),
        "latest_prediction": state.get("latest_prediction"),
    }
    values.update(updates)
    return runtime_state(phase, **values)
