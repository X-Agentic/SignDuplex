"""V1.6-aligned, past-only communicative-sign evidence from cached Pose."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np


CONFIDENCE_THRESHOLD = 0.30
RECENT_WINDOW_SECONDS = 0.25
CAUSAL_ACTIVE_DURATION_MIN_SECONDS = 0.30
RECENT_ACTIVE_FRACTION_MIN = 0.75
ARTICULATION_WINDOW_SECONDS = 0.25
RECENT_ARTICULATION_ABSOLUTE_MIN = 0.004
HANDSHAPE_ARTICULATION_FRACTION_MIN = 0.50
TRAJECTORY_ARTICULATION_FRACTION_MIN = 0.35
CONFIRMATION_SCORE_MIN = 0.66
CUMULATIVE_DISPLACEMENT_GAP_UNITS_MIN = 0.60
ACTIVE_THRESHOLD_FRACTION = 0.38
TRAILING_MEDIAN_SECONDS = 0.13
TRAILING_MEAN_SECONDS = 0.23

LEFT_HAND = slice(91, 112)
RIGHT_HAND = slice(112, 133)
LEFT_BODY_WRIST = 9
RIGHT_BODY_WRIST = 10
LEFT_HAND_ROOT = 91
RIGHT_HAND_ROOT = 112


@dataclass(frozen=True)
class CausalSignEvidence:
    motion_score: float
    hand_motion_score: float
    recent_active_fraction: float
    sustained_active_duration: float
    hand_articulation_score: float
    trajectory_score: float
    articulation_active_fraction: float
    causal_confirmation_score: float
    causal_sign_evidence: bool
    current_motion_active: bool
    motion_episode_start_frame: int | None

    @property
    def activity_label(self) -> str:
        return "ACTIVE" if self.causal_sign_evidence else "NO_ACTIVITY"


def _past_fill(values: np.ndarray) -> np.ndarray:
    """Forward-fill missing values without consulting a future observation."""
    output = np.asarray(values, dtype=np.float64).copy()
    previous = 0.0
    for index, value in enumerate(output):
        if np.isfinite(value):
            previous = float(value)
        else:
            output[index] = previous
    return output


def _hand_motion(
    keypoints: np.ndarray,
    scores: np.ndarray,
    hand_slice: slice,
    body_wrist: int,
) -> np.ndarray:
    if len(keypoints) < 2:
        return np.zeros(len(keypoints), dtype=np.float64)
    hand_xy = keypoints[:, hand_slice]
    hand_scores = np.clip(scores[:, hand_slice], 0.0, 1.0)
    displacement = np.linalg.norm(np.diff(hand_xy, axis=0), axis=2)
    pair_confidence = np.minimum(hand_scores[:-1], hand_scores[1:])
    valid = pair_confidence >= CONFIDENCE_THRESHOLD
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        per_pair = np.nanmedian(np.where(valid, displacement, np.nan), axis=1)
    wrist_delta = np.linalg.norm(
        np.diff(keypoints[:, body_wrist], axis=0), axis=1
    )
    wrist_valid = (
        np.minimum(scores[:-1, body_wrist], scores[1:, body_wrist])
        >= CONFIDENCE_THRESHOLD
    )
    per_pair = np.where(
        np.sum(valid, axis=1) >= 5,
        per_pair,
        np.where(wrist_valid, wrist_delta, np.nan),
    )
    return np.r_[0.0, _past_fill(per_pair)]


def _wrist_speed(
    keypoints: np.ndarray,
    scores: np.ndarray,
    body_wrist: int,
    hand_root: int,
    fps: float,
) -> np.ndarray:
    if len(keypoints) < 2:
        return np.zeros(len(keypoints), dtype=np.float64)
    body_delta = np.linalg.norm(
        np.diff(keypoints[:, body_wrist], axis=0), axis=1
    )
    root_delta = np.linalg.norm(
        np.diff(keypoints[:, hand_root], axis=0), axis=1
    )
    body_valid = (
        np.minimum(scores[:-1, body_wrist], scores[1:, body_wrist])
        >= CONFIDENCE_THRESHOLD
    )
    root_valid = (
        np.minimum(scores[:-1, hand_root], scores[1:, hand_root])
        >= CONFIDENCE_THRESHOLD
    )
    per_pair = np.where(
        body_valid,
        body_delta,
        np.where(root_valid, root_delta, np.nan),
    )
    return np.r_[0.0, _past_fill(per_pair) * fps]


def _hand_articulation(
    keypoints: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    if len(keypoints) < 2:
        return np.zeros(len(keypoints), dtype=np.float64)
    sides: list[np.ndarray] = []
    for hand_slice, wrist in (
        (LEFT_HAND, LEFT_HAND_ROOT),
        (RIGHT_HAND, RIGHT_HAND_ROOT),
    ):
        relative = keypoints[:, hand_slice] - keypoints[:, wrist, None]
        delta = np.linalg.norm(np.diff(relative, axis=0), axis=2)
        joint_valid = (
            np.minimum(scores[:-1, hand_slice], scores[1:, hand_slice])
            >= CONFIDENCE_THRESHOLD
        )
        wrist_valid = (
            np.minimum(scores[:-1, wrist], scores[1:, wrist])
            >= CONFIDENCE_THRESHOLD
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(
                np.where(joint_valid & wrist_valid[:, None], delta, np.nan),
                axis=1,
            )
        sides.append(_past_fill(median))
    return np.r_[0.0, np.maximum(sides[0], sides[1])]


def _trailing_smooth(values: np.ndarray, fps: float) -> np.ndarray:
    """V1.6 trailing median then trailing mean; every sample is past-only."""
    values = np.asarray(values, dtype=np.float64)
    median_width = max(1, int(round(TRAILING_MEDIAN_SECONDS * fps)))
    mean_width = max(1, int(round(TRAILING_MEAN_SECONDS * fps)))
    median = np.empty_like(values)
    for index in range(len(values)):
        median[index] = np.median(
            values[max(0, index - median_width + 1) : index + 1]
        )
    cumulative = np.r_[0.0, np.cumsum(median)]
    mean = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index - mean_width + 1)
        mean[index] = (
            cumulative[index + 1] - cumulative[start]
        ) / (index + 1 - start)
    return mean


def _recent(values: np.ndarray, seconds: float, fps: float) -> np.ndarray:
    width = max(1, int(math.ceil(seconds * fps)))
    return np.asarray(values[max(0, len(values) - width) :])


def _trailing_true_start(mask: np.ndarray) -> int | None:
    if not len(mask) or not bool(mask[-1]):
        return None
    start = len(mask) - 1
    while start > 0 and bool(mask[start - 1]):
        start -= 1
    return start


def causal_sign_evidence(
    pose_keypoints: Sequence[np.ndarray],
    pose_scores: Sequence[np.ndarray],
    task_prefix_start_frame: int,
    task_prefix_end_frame: int,
    fps: float,
) -> CausalSignEvidence:
    """Evaluate V1.6-style evidence within one observed turn/state prefix."""
    available = min(len(pose_keypoints), len(pose_scores))
    start = max(0, min(int(task_prefix_start_frame), available))
    end = max(start, min(int(task_prefix_end_frame), available))
    if fps <= 0 or end - start < 2:
        return CausalSignEvidence(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, False, None
        )

    keypoints = np.asarray(pose_keypoints[start:end], dtype=np.float64)[:, 0]
    scores = np.asarray(pose_scores[start:end], dtype=np.float64)[:, 0]
    left = _hand_motion(keypoints, scores, LEFT_HAND, LEFT_BODY_WRIST)
    right = _hand_motion(keypoints, scores, RIGHT_HAND, RIGHT_BODY_WRIST)
    combined_raw = np.sqrt(left * left + right * right)
    combined = _trailing_smooth(combined_raw, fps)
    left_smooth = _trailing_smooth(left, fps)
    right_smooth = _trailing_smooth(right, fps)
    left_wrist = _trailing_smooth(
        _wrist_speed(
            keypoints, scores, LEFT_BODY_WRIST, LEFT_HAND_ROOT, fps
        ),
        fps,
    )
    right_wrist = _trailing_smooth(
        _wrist_speed(
            keypoints, scores, RIGHT_BODY_WRIST, RIGHT_HAND_ROOT, fps
        ),
        fps,
    )
    wrist = np.maximum(left_wrist, right_wrist)
    articulation = _hand_articulation(keypoints, scores)

    q20, q50, q80 = np.quantile(combined, [0.20, 0.50, 0.80])
    gap = max(float(q80 - q20), 1e-8)
    active_threshold = max(
        float(q20 + ACTIVE_THRESHOLD_FRACTION * gap), 1e-8
    )
    active_mask = combined >= active_threshold
    episode_start = _trailing_true_start(active_mask)
    sustained = (
        (len(active_mask) - episode_start) / fps
        if episode_start is not None
        else 0.0
    )

    recent_motion = _recent(combined, RECENT_WINDOW_SECONDS, fps)
    recent_active = recent_motion >= active_threshold
    active_fraction = (
        float(np.mean(recent_active)) if len(recent_active) else 0.0
    )
    recent_mean = float(np.mean(recent_motion)) if len(recent_motion) else 0.0
    recent_wrist = _recent(wrist, RECENT_WINDOW_SECONDS, fps)
    trajectory_score = (
        float(np.mean(recent_wrist)) if len(recent_wrist) else 0.0
    )

    art_q20, art_q50, _ = np.quantile(articulation, [0.20, 0.50, 0.80])
    recent_articulation = _recent(
        articulation, ARTICULATION_WINDOW_SECONDS, fps
    )
    articulation_score = (
        float(np.mean(recent_articulation))
        if len(recent_articulation)
        else 0.0
    )
    articulation_fraction = (
        float(np.mean(recent_articulation >= art_q50))
        if len(recent_articulation)
        else 0.0
    )
    handshape_evidence = bool(
        articulation_score
        >= max(RECENT_ARTICULATION_ABSOLUTE_MIN, float(art_q50))
        and articulation_fraction >= HANDSHAPE_ARTICULATION_FRACTION_MIN
    )
    onset = episode_start if episode_start is not None else len(combined) - 1
    cumulative_displacement = float(np.sum(combined_raw[onset:]))
    displacement_gap_units = cumulative_displacement / gap
    trajectory_articulation = bool(
        recent_mean >= float(q50)
        and displacement_gap_units
        >= CUMULATIVE_DISPLACEMENT_GAP_UNITS_MIN
        and articulation_score
        >= max(0.75 * RECENT_ARTICULATION_ABSOLUTE_MIN, float(art_q20))
        and articulation_fraction >= TRAJECTORY_ARTICULATION_FRACTION_MIN
    )
    sign_articulation = handshape_evidence or trajectory_articulation
    bilateral = float(
        np.mean(
            (_recent(left_smooth, RECENT_WINDOW_SECONDS, fps) >= np.quantile(left_smooth, 0.50))
            & (_recent(right_smooth, RECENT_WINDOW_SECONDS, fps) >= np.quantile(right_smooth, 0.50))
        )
    )
    motion_ratio = recent_mean / max(float(q50), 1e-8)
    wrist_ratio = trajectory_score / max(float(np.quantile(wrist, 0.50)), 1e-8)
    confirmation = float(
        np.clip(
            0.25
            * np.clip(
                sustained / CAUSAL_ACTIVE_DURATION_MIN_SECONDS, 0.0, 1.0
            )
            + 0.25
            * np.clip(
                active_fraction / RECENT_ACTIVE_FRACTION_MIN, 0.0, 1.0
            )
            + 0.18 * np.clip(motion_ratio, 0.0, 1.0)
            + 0.10
            * np.clip(
                displacement_gap_units
                / CUMULATIVE_DISPLACEMENT_GAP_UNITS_MIN,
                0.0,
                1.0,
            )
            + 0.07 * np.clip(wrist_ratio, 0.0, 1.0)
            + 0.05 * bilateral,
            0.0,
            1.0,
        )
    )
    evidence = bool(
        sustained >= CAUSAL_ACTIVE_DURATION_MIN_SECONDS
        and active_fraction >= RECENT_ACTIVE_FRACTION_MIN
        and recent_mean >= float(q50)
        and combined[-1] >= active_threshold
        and displacement_gap_units
        >= CUMULATIVE_DISPLACEMENT_GAP_UNITS_MIN
        and sign_articulation
        and confirmation >= CONFIRMATION_SCORE_MIN
    )
    current_motion_active = bool(
        active_mask[-1]
        and active_fraction >= 0.50
        and articulation_score >= 0.75 * RECENT_ARTICULATION_ABSOLUTE_MIN
    )
    return CausalSignEvidence(
        motion_score=float(combined[-1]),
        hand_motion_score=float(max(left_smooth[-1], right_smooth[-1])),
        recent_active_fraction=active_fraction,
        sustained_active_duration=sustained,
        hand_articulation_score=articulation_score,
        trajectory_score=trajectory_score,
        articulation_active_fraction=articulation_fraction,
        causal_confirmation_score=confirmation,
        causal_sign_evidence=evidence,
        current_motion_active=current_motion_active,
        motion_episode_start_frame=(
            start + episode_start if episode_start is not None else None
        ),
    )
