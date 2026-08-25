#!/usr/bin/env python3
"""Build SignDuplex V1.8 with a strict pre-sign IDLE class.

The five existing labels are inherited as raw JSONL lines, in source order:

* COMPLETE from V1.7
* INCOMPLETE and BACKCHANNEL from V1.5
* NO_BARGE and BARGE from V1.6

IDLE is mined by streaming the verified V1.5 causal Pose/sign feature JSONL.
No RGB, Pose, source dataset, or training file is modified.  The output is
built in a sibling staging directory and published only after every hard
validation succeeds.  Existing output directories are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_ROOT = Path(os.environ.get("SIGNDUPLEX_DATA_ROOT", REPO_ROOT / "data")).expanduser()
DEFAULT_V15 = DATA_ROOT / "Sign_Aware_Raw_Screen_v1_5"
DEFAULT_V16 = DATA_ROOT / "Sign_Aware_Raw_Screen_v1_6"
DEFAULT_V17 = DATA_ROOT / "Sign_Aware_Raw_Screen_v1_7"
DEFAULT_OUTPUT = DATA_ROOT / "Sign_Aware_Raw_Screen_v1_8"

VERSION = "Sign_Aware_Raw_Screen_v1_8"
SPLITS = ("train", "dev", "test")
LABELS = ("IDLE", "INCOMPLETE", "BACKCHANNEL", "COMPLETE", "NO_BARGE", "BARGE")
LISTENING_SOURCE_LABELS = frozenset(("INCOMPLETE", "BACKCHANNEL", "COMPLETE"))
SOURCE_LABELS: dict[str, tuple[str, ...]] = {
    "v1_5": ("INCOMPLETE", "BACKCHANNEL"),
    "v1_7": ("COMPLETE",),
    "v1_6": ("NO_BARGE", "BARGE"),
}
SOURCE_ORDER = ("v1_5", "v1_7", "v1_6")

CONFIG: dict[str, Any] = {
    "version": VERSION,
    "random_seed": 42,
    "inheritance": {
        "COMPLETE": "Sign_Aware_Raw_Screen_v1_7",
        "INCOMPLETE": "Sign_Aware_Raw_Screen_v1_5",
        "BACKCHANNEL": "Sign_Aware_Raw_Screen_v1_5",
        "NO_BARGE": "Sign_Aware_Raw_Screen_v1_6",
        "BARGE": "Sign_Aware_Raw_Screen_v1_6",
        "IDLE": "V1.8 strict pre-sign construction from V1.5 causal features",
    },
    "idle": {
        "candidate_nominal_times_sec": [0.50, 0.75, 1.00, 1.25, 1.50],
        "candidate_frame_alignment": (
            "nearest positive prefix-end frame using floor(seconds*fps+0.5); "
            "actual decision_time_sec=prefix_end_frame_exclusive/fps"
        ),
        "minimum_decision_time_sec": 0.50,
        "onset_safety_margin_sec": 0.30,
        "recent_window_sec": 0.50,
        "recent_active_ratio_max": 0.10,
        "stable_active_prefix_min_sec": 0.30,
        "future_first_active_run_min_sec": 0.70,
        "maximum_samples_per_video": 1,
        "target_per_split": "V1.5 INCOMPLETE count in the same official split",
        "insufficient_pool_policy": "retain every valid sample; never relax a hard rule",
        "quality_gate_pass_required": True,
        "minimum_feature_reliability": 1.0,
        "required_reliability_fields": [
            "left_hand_motion_reliability",
            "right_hand_motion_reliability",
            "left_wrist_speed_reliability",
            "right_wrist_speed_reliability",
        ],
        "per_frame_recent_low_motion_required": True,
        "thresholds": {
            "baseline_quantiles": [0.20, 0.50, 0.80],
            "baseline_dynamic_multiplier": 0.50,
            "internal_dynamic_multiplier": 0.75,
            "combined_motion_absolute_min": 0.0035,
            "hand_motion_absolute_min": 0.0035,
            "wrist_speed_absolute_min": 0.06,
            "internal_motion_absolute_min": 0.008,
            "combined_and_hand_cap": "V1.5 active_motion_threshold for the current video",
            "strict_comparison": "every frame in the recent window must be < its threshold",
        },
        "internal_motion_proxy": (
            "per hand max(causal_hand_motion - causal_wrist_speed/fps, 0); "
            "same definition as the verified V1.6 builder"
        ),
    },
    "contracts": {
        "factorized_tasks": {
            "LISTENING": ["IDLE", "INCOMPLETE", "BACKCHANNEL", "COMPLETE"],
            "SPEAKING": ["NO_BARGE", "BARGE"],
        },
        "prefix_start_frame": 0,
        "prefix_end": "decision_frame + 1",
        "causal_model_input": "RGB/Pose [0:prefix_end_frame_exclusive) only",
        "future_metadata_is_offline_validation_only": True,
        "source_jsonl_lines_and_per_source_order_preserved": True,
        "official_split_preserved": True,
        "physical_split_key": "video_id",
        "semantic_overlap": "reported transparently, never repaired by resplitting",
        "raw_rgb_pose_read_only": True,
        "source_datasets_read_only": True,
        "existing_output_never_overwritten": True,
        "root_jsonl_and_final_dataset_are_hard_links": True,
        "train_model": False,
    },
}


class SourceIntegrityError(RuntimeError):
    """Raised when a supposedly frozen/verified source violates its contract."""


@dataclass
class PreparedFeature:
    feature: dict[str, Any]
    arrays: dict[str, np.ndarray]
    active_mask: np.ndarray
    runs: list[dict[str, int]]


@dataclass
class CandidateEvaluation:
    record: dict[str, Any] | None
    reasons: tuple[str, ...]
    metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-5", type=Path, default=DEFAULT_V15)
    parser.add_argument("--v1-6", type=Path, default=DEFAULT_V16)
    parser.add_argument("--v1-7", type=Path, default=DEFAULT_V17)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def rounded(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def describe(values: Iterable[float]) -> dict[str, Any]:
    data = sorted(float(value) for value in values)
    if not data:
        return {"count": 0}
    return {
        "count": len(data),
        "min": rounded(data[0]),
        "p05": rounded(percentile(data, 0.05)),
        "p25": rounded(percentile(data, 0.25)),
        "median": rounded(percentile(data, 0.50)),
        "mean": rounded(statistics.fmean(data)),
        "p75": rounded(percentile(data, 0.75)),
        "p95": rounded(percentile(data, 0.95)),
        "max": rounded(data[-1]),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        handle.write("\n")
    temporary.replace(path)


def source_jsonl(root: Path, split: str) -> Path:
    path = root / "final_dataset" / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def feature_jsonl(v15: Path) -> Path:
    path = v15 / "01_pose_sign_features" / "pose_sign_features.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def iter_raw_jsonl(path: Path) -> Iterator[tuple[bytes, dict[str, Any]]]:
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                raise SourceIntegrityError(f"blank JSONL line: {path}:{line_number}")
            if not raw_line.endswith(b"\n"):
                raise SourceIntegrityError(
                    f"JSONL line lacks newline: {path}:{line_number}"
                )
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise SourceIntegrityError(
                    f"invalid JSONL: {path}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise SourceIntegrityError(
                    f"JSONL record is not an object: {path}:{line_number}"
                )
            yield raw_line, value


def derive_semantic_id(video_id: str) -> str:
    parts = video_id.split("_")
    if len(parts) != 3 or not parts[1].startswith("P"):
        raise SourceIntegrityError(f"unexpected video_id format: {video_id}")
    return f"{parts[0]}_{parts[2]}"


def derive_signer_id(video_id: str) -> str:
    parts = video_id.split("_")
    if len(parts) != 3 or not parts[1].startswith("P"):
        raise SourceIntegrityError(f"unexpected video_id format: {video_id}")
    return parts[1]


def core_record_checks(item: Mapping[str, Any], split: str, context: str) -> None:
    required = (
        "sample_id",
        "video_id",
        "semantic_id",
        "signer_id",
        "split",
        "rgb_path",
        "pose_path",
        "label",
        "task",
        "system_state",
        "system_action",
        "decision_frame",
        "decision_time_sec",
        "prefix_start_frame",
        "prefix_end_frame_exclusive",
        "fps",
        "video_frames",
    )
    missing = [key for key in required if item.get(key) is None]
    if missing:
        raise SourceIntegrityError(f"missing fields {missing}: {context}")
    if item["split"] != split:
        raise SourceIntegrityError(f"record/file split mismatch: {context}")
    video_id = str(item["video_id"])
    if item["semantic_id"] != derive_semantic_id(video_id):
        raise SourceIntegrityError(f"semantic_id mismatch: {context}")
    if item["signer_id"] != derive_signer_id(video_id):
        raise SourceIntegrityError(f"signer_id mismatch: {context}")
    decision = int(item["decision_frame"])
    prefix_start = int(item["prefix_start_frame"])
    prefix_end = int(item["prefix_end_frame_exclusive"])
    frames = int(item["video_frames"])
    fps = float(item["fps"])
    if prefix_start != 0 or prefix_end != decision + 1:
        raise SourceIntegrityError(f"invalid causal prefix: {context}")
    if not (0 <= decision < frames) or fps <= 0.0:
        raise SourceIntegrityError(f"decision outside source video: {context}")
    if abs(float(item["decision_time_sec"]) - prefix_end / fps) > 1e-5:
        raise SourceIntegrityError(f"decision time/frame mismatch: {context}")
    rgb = Path(str(item["rgb_path"]))
    pose = Path(str(item["pose_path"]))
    if rgb.stem != video_id or pose.stem != video_id:
        raise SourceIntegrityError(f"RGB/Pose basename mismatch: {context}")


def source_manifest(roots: Mapping[str, Path], feature_path: Path) -> dict[str, Any]:
    return {
        "datasets": {
            source_name: {
                split: {
                    "path": str(source_jsonl(root, split)),
                    "size": source_jsonl(root, split).stat().st_size,
                    "sha256": sha256(source_jsonl(root, split)),
                }
                for split in SPLITS
            }
            for source_name, root in roots.items()
        },
        "v1_5_pose_sign_features": {
            "path": str(feature_path),
            "size": feature_path.stat().st_size,
            "sha256": sha256(feature_path),
        },
    }


def scan_sources(roots: Mapping[str, Path]) -> dict[str, Any]:
    full_counts: dict[str, dict[str, Counter[str]]] = defaultdict(dict)
    inherited_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    label_digests: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    group_digests: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    listening_decisions: set[tuple[str, int]] = set()
    sample_ids: set[str] = set()
    official_video_split: dict[str, str] = {}
    checked_media: set[tuple[str, str]] = set()

    for source_name in SOURCE_ORDER:
        labels = frozenset(SOURCE_LABELS[source_name])
        root = roots[source_name]
        for split in SPLITS:
            path = source_jsonl(root, split)
            counts: Counter[str] = Counter()
            per_label_hash = {label: hashlib.sha256() for label in labels}
            per_label_count: Counter[str] = Counter()
            group_hash = hashlib.sha256()
            group_count = 0
            for raw_line, item in iter_raw_jsonl(path):
                context = f"{path}:{item.get('sample_id', '<unknown>')}"
                core_record_checks(item, split, context)
                label = str(item["label"])
                counts[label] += 1
                video_id = str(item["video_id"])
                if source_name == "v1_5":
                    previous = official_video_split.setdefault(video_id, split)
                    if previous != split:
                        raise SourceIntegrityError(
                            f"V1.5 physical split leakage: {video_id}"
                        )
                if label not in labels:
                    continue
                if item["sample_id"] in sample_ids:
                    raise SourceIntegrityError(
                        f"duplicate inherited sample_id: {item['sample_id']}"
                    )
                sample_ids.add(str(item["sample_id"]))
                inherited_counts[split][label] += 1
                per_label_hash[label].update(raw_line)
                per_label_count[label] += 1
                group_hash.update(raw_line)
                group_count += 1
                if label in LISTENING_SOURCE_LABELS:
                    listening_decisions.add(
                        (video_id, int(item["prefix_end_frame_exclusive"]))
                    )
                media_key = (str(item["rgb_path"]), str(item["pose_path"]))
                if media_key not in checked_media:
                    if not Path(media_key[0]).is_file():
                        raise SourceIntegrityError(
                            f"missing inherited RGB: {media_key[0]}"
                        )
                    if not Path(media_key[1]).is_file():
                        raise SourceIntegrityError(
                            f"missing inherited Pose: {media_key[1]}"
                        )
                    checked_media.add(media_key)
            full_counts[source_name][split] = counts
            label_digests[source_name][split] = {
                label: {
                    "count": per_label_count[label],
                    "ordered_raw_line_sha256": per_label_hash[label].hexdigest(),
                }
                for label in SOURCE_LABELS[source_name]
            }
            group_digests[source_name][split] = {
                "labels": list(SOURCE_LABELS[source_name]),
                "count": group_count,
                "ordered_raw_line_sha256": group_hash.hexdigest(),
            }

    expected = set(LABELS) - {"IDLE"}
    observed = {label for counter in inherited_counts.values() for label in counter}
    if observed != expected:
        raise SourceIntegrityError(f"inherited label set mismatch: {sorted(observed)}")
    target_idle = {
        split: int(full_counts["v1_5"][split].get("INCOMPLETE", 0)) for split in SPLITS
    }
    if any(value <= 0 for value in target_idle.values()):
        raise SourceIntegrityError(f"invalid V1.5 INCOMPLETE targets: {target_idle}")
    return {
        "full_counts": full_counts,
        "inherited_counts": inherited_counts,
        "target_idle": target_idle,
        "label_digests": label_digests,
        "group_digests": group_digests,
        "listening_decisions": listening_decisions,
        "official_video_split": official_video_split,
        "inherited_sample_ids": sample_ids,
        "checked_unique_media_pairs": len(checked_media),
    }


def active_mask_from_runs(
    num_frames: int, runs: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    mask = np.zeros(num_frames, dtype=bool)
    for run in runs:
        start = max(0, int(run["start_frame"]))
        end = min(num_frames, int(run["end_frame_exclusive"]))
        if end > start:
            mask[start:end] = True
    return mask


def longest_true_run(mask: Sequence[bool] | np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def align_prefix_end(nominal_time_sec: float, fps: float, num_frames: int) -> int:
    if nominal_time_sec <= 0.0 or fps <= 0.0 or num_frames <= 0:
        raise ValueError("candidate time, fps, and frame count must be positive")
    return min(num_frames, max(1, int(math.floor(nominal_time_sec * fps + 0.5))))


def adaptive_threshold(
    values: np.ndarray,
    absolute_minimum: float,
    dynamic_multiplier: float,
    cap: float | None = None,
) -> dict[str, float]:
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise SourceIntegrityError("threshold baseline is empty or non-finite")
    q20, q50, q80 = np.quantile(values, [0.20, 0.50, 0.80])
    threshold = max(float(q80 + dynamic_multiplier * (q80 - q20)), absolute_minimum)
    if cap is not None:
        if not math.isfinite(cap) or cap <= 0.0:
            raise SourceIntegrityError(f"invalid active threshold cap: {cap}")
        threshold = min(threshold, cap)
    return {
        "q20": float(q20),
        "q50": float(q50),
        "q80": float(q80),
        "threshold": float(threshold),
    }


def prepare_feature(feature: dict[str, Any]) -> PreparedFeature:
    required = (
        "video_id",
        "semantic_id",
        "split",
        "rgb_path",
        "pose_path",
        "fps",
        "num_frames",
        "active_start_frame",
        "active_end_frame",
        "active_motion_threshold",
        "active_runs",
        "causal_motion_smooth_curve",
        "causal_left_hand_smooth_curve",
        "causal_right_hand_smooth_curve",
        "causal_left_wrist_speed_curve",
        "causal_right_wrist_speed_curve",
    )
    missing = [key for key in required if feature.get(key) is None]
    if missing:
        raise SourceIntegrityError(
            f"V1.5 feature missing {missing}: {feature.get('video_id', '<unknown>')}"
        )
    video_id = str(feature["video_id"])
    if feature["semantic_id"] != derive_semantic_id(video_id):
        raise SourceIntegrityError(f"feature semantic_id mismatch: {video_id}")
    if feature["split"] not in SPLITS:
        raise SourceIntegrityError(
            f"feature split invalid: {video_id}: {feature['split']}"
        )
    fps = float(feature["fps"])
    frames = int(feature["num_frames"])
    if fps <= 0.0 or frames <= 0:
        raise SourceIntegrityError(f"invalid feature geometry: {video_id}")
    curve_keys = {
        "combined": "causal_motion_smooth_curve",
        "left_hand": "causal_left_hand_smooth_curve",
        "right_hand": "causal_right_hand_smooth_curve",
        "left_wrist": "causal_left_wrist_speed_curve",
        "right_wrist": "causal_right_wrist_speed_curve",
    }
    arrays = {
        name: np.asarray(feature[key], dtype=np.float64)
        for name, key in curve_keys.items()
    }
    lengths = {array.size for array in arrays.values()}
    if lengths != {frames}:
        raise SourceIntegrityError(
            f"causal feature length mismatch: {video_id}: {lengths}"
        )
    if any(not np.all(np.isfinite(array)) for array in arrays.values()):
        raise SourceIntegrityError(f"non-finite causal feature: {video_id}")
    arrays["left_internal"] = np.maximum(
        arrays["left_hand"] - arrays["left_wrist"] / fps, 0.0
    )
    arrays["right_internal"] = np.maximum(
        arrays["right_hand"] - arrays["right_wrist"] / fps, 0.0
    )
    arrays["wrist"] = np.maximum(arrays["left_wrist"], arrays["right_wrist"])
    arrays["internal"] = np.maximum(arrays["left_internal"], arrays["right_internal"])

    runs: list[dict[str, int]] = []
    previous_end = 0
    for raw_run in feature.get("active_runs") or []:
        start = int(raw_run["start_frame"])
        end = int(raw_run["end_frame_exclusive"])
        if not (0 <= start < end <= frames) or start < previous_end:
            raise SourceIntegrityError(
                f"invalid/overlapping active run: {video_id}: {raw_run}"
            )
        runs.append({"start_frame": start, "end_frame_exclusive": end})
        previous_end = end
    active_start = int(feature["active_start_frame"])
    active_end = int(feature["active_end_frame"])
    if (
        not runs
        or runs[0]["start_frame"] != active_start
        or runs[-1]["end_frame_exclusive"] != active_end + 1
    ):
        raise SourceIntegrityError(f"active boundary/run mismatch: {video_id}")
    return PreparedFeature(
        feature=feature,
        arrays=arrays,
        active_mask=active_mask_from_runs(frames, runs),
        runs=runs,
    )


def feature_filter_reasons(feature: Mapping[str, Any]) -> tuple[str, ...]:
    reasons: list[str] = []
    cfg = CONFIG["idle"]
    if feature.get("quality_gate_pass") is not True:
        reasons.append("QUALITY_GATE_FAILED")
    required_reliability = float(cfg["minimum_feature_reliability"])
    for field in cfg["required_reliability_fields"]:
        value = feature.get(field)
        if value is None or abs(float(value) - required_reliability) > 1e-12:
            reasons.append(f"RELIABILITY_NOT_ONE:{field}")
    return tuple(reasons)


def candidate_thresholds(
    prepared: PreparedFeature, prefix_end: int
) -> dict[str, dict[str, float]]:
    cfg = CONFIG["idle"]["thresholds"]
    arrays = prepared.arrays
    active_cap = float(prepared.feature["active_motion_threshold"])
    common_multiplier = float(cfg["baseline_dynamic_multiplier"])
    internal_multiplier = float(cfg["internal_dynamic_multiplier"])
    thresholds = {
        "combined": adaptive_threshold(
            arrays["combined"][:prefix_end],
            float(cfg["combined_motion_absolute_min"]),
            common_multiplier,
            cap=active_cap,
        ),
        "left_hand": adaptive_threshold(
            arrays["left_hand"][:prefix_end],
            float(cfg["hand_motion_absolute_min"]),
            common_multiplier,
            cap=active_cap,
        ),
        "right_hand": adaptive_threshold(
            arrays["right_hand"][:prefix_end],
            float(cfg["hand_motion_absolute_min"]),
            common_multiplier,
            cap=active_cap,
        ),
        "left_wrist": adaptive_threshold(
            arrays["left_wrist"][:prefix_end],
            float(cfg["wrist_speed_absolute_min"]),
            common_multiplier,
        ),
        "right_wrist": adaptive_threshold(
            arrays["right_wrist"][:prefix_end],
            float(cfg["wrist_speed_absolute_min"]),
            common_multiplier,
        ),
        "left_internal": adaptive_threshold(
            arrays["left_internal"][:prefix_end],
            float(cfg["internal_motion_absolute_min"]),
            internal_multiplier,
        ),
        "right_internal": adaptive_threshold(
            arrays["right_internal"][:prefix_end],
            float(cfg["internal_motion_absolute_min"]),
            internal_multiplier,
        ),
    }
    return thresholds


def evaluate_idle_candidate(
    prepared: PreparedFeature,
    nominal_time_sec: float,
    listening_decisions: set[tuple[str, int]],
) -> CandidateEvaluation:
    feature = prepared.feature
    arrays = prepared.arrays
    cfg = CONFIG["idle"]
    fps = float(feature["fps"])
    frames = int(feature["num_frames"])
    video_id = str(feature["video_id"])
    prefix_end = align_prefix_end(nominal_time_sec, fps, frames)
    decision_frame = prefix_end - 1
    decision_time = prefix_end / fps
    active_start = int(feature["active_start_frame"])
    active_start_sec = active_start / fps
    margin = active_start_sec - decision_time
    recent_width = max(1, int(math.ceil(float(cfg["recent_window_sec"]) * fps - 1e-12)))
    recent_start = max(0, prefix_end - recent_width)
    thresholds = candidate_thresholds(prepared, prefix_end)

    recent_active = prepared.active_mask[recent_start:prefix_end]
    recent_active_ratio = float(np.mean(recent_active)) if recent_active.size else 0.0
    offline_prefix_longest = longest_true_run(prepared.active_mask[:prefix_end]) / fps
    causal_prefix_longest = (
        longest_true_run(
            arrays["combined"][:prefix_end] >= thresholds["combined"]["threshold"]
        )
        / fps
    )

    future_runs = [
        run for run in prepared.runs if run["end_frame_exclusive"] > prefix_end
    ]
    future_first = future_runs[0] if future_runs else None
    future_duration = (
        (future_first["end_frame_exclusive"] - future_first["start_frame"]) / fps
        if future_first is not None
        else 0.0
    )

    recent_maxima: dict[str, float] = {
        name: float(np.max(arrays[name][recent_start:prefix_end]))
        for name in thresholds
    }
    recent_means: dict[str, float] = {
        name: float(np.mean(arrays[name][recent_start:prefix_end]))
        for name in thresholds
    }
    low_streams = {
        name: bool(
            np.all(arrays[name][recent_start:prefix_end] < threshold_info["threshold"])
        )
        for name, threshold_info in thresholds.items()
    }

    reasons: list[str] = []
    epsilon = 1e-9
    if decision_time + epsilon < float(cfg["minimum_decision_time_sec"]):
        reasons.append("DECISION_BEFORE_MINIMUM_TIME")
    if not decision_time < active_start_sec:
        reasons.append("DECISION_NOT_BEFORE_ACTIVE_START")
    if margin + epsilon < float(cfg["onset_safety_margin_sec"]):
        reasons.append("ONSET_SAFETY_MARGIN_TOO_SHORT")
    if recent_active_ratio > float(cfg["recent_active_ratio_max"]) + epsilon:
        reasons.append("RECENT_ACTIVE_RATIO_TOO_HIGH")
    for name, passed in low_streams.items():
        if not passed:
            reasons.append(f"RECENT_{name.upper()}_NOT_STRICTLY_LOW")
    stable_limit = float(cfg["stable_active_prefix_min_sec"])
    if offline_prefix_longest + epsilon >= stable_limit:
        reasons.append("STABLE_OFFLINE_ACTIVE_ALREADY_IN_PREFIX")
    if causal_prefix_longest + epsilon >= stable_limit:
        reasons.append("STABLE_CAUSAL_ACTIVE_ALREADY_IN_PREFIX")
    if future_first is None:
        reasons.append("NO_FUTURE_ACTIVE_RUN")
    else:
        if future_first["start_frame"] != active_start:
            reasons.append("FIRST_FUTURE_RUN_NOT_ACTIVE_START")
        if future_duration + epsilon < float(cfg["future_first_active_run_min_sec"]):
            reasons.append("FUTURE_FIRST_ACTIVE_RUN_TOO_SHORT")
    if (video_id, prefix_end) in listening_decisions:
        reasons.append("LISTENING_DECISION_CONFLICT")

    metrics: dict[str, Any] = {
        "nominal_time_sec": nominal_time_sec,
        "prefix_end_frame_exclusive": prefix_end,
        "decision_frame": decision_frame,
        "decision_time_sec": decision_time,
        "active_start_sec": active_start_sec,
        "onset_safety_margin_sec": margin,
        "recent_start_frame": recent_start,
        "recent_end_frame_exclusive": prefix_end,
        "recent_active_ratio": recent_active_ratio,
        "offline_prefix_longest_active_sec": offline_prefix_longest,
        "causal_prefix_longest_active_sec": causal_prefix_longest,
        "future_first_active_run": future_first,
        "future_first_active_run_duration_sec": future_duration,
        "thresholds": thresholds,
        "recent_maxima": recent_maxima,
        "recent_means": recent_means,
        "low_streams": low_streams,
    }
    if reasons:
        return CandidateEvaluation(record=None, reasons=tuple(reasons), metrics=metrics)
    return CandidateEvaluation(
        record=make_idle_record(prepared, metrics), reasons=(), metrics=metrics
    )


def make_idle_record(
    prepared: PreparedFeature, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    feature = prepared.feature
    arrays = prepared.arrays
    fps = float(feature["fps"])
    frames = int(feature["num_frames"])
    prefix_end = int(metrics["prefix_end_frame_exclusive"])
    decision_frame = int(metrics["decision_frame"])
    decision_time = float(metrics["decision_time_sec"])
    active_start = int(feature["active_start_frame"])
    active_end = int(feature["active_end_frame"])
    recent_start = int(metrics["recent_start_frame"])
    recent = arrays["combined"][recent_start:prefix_end]
    midpoint = max(1, recent.size // 2)
    first_half = float(np.mean(recent[:midpoint])) if recent.size else 0.0
    second_half = (
        float(np.mean(recent[midpoint:])) if midpoint < recent.size else first_half
    )
    dynamic_range = max(
        float(feature.get("hand_q80", 0.0)) - float(feature.get("hand_q20", 0.0)), 1e-12
    )
    thresholds = metrics["thresholds"]
    maxima = metrics["recent_maxima"]
    means = metrics["recent_means"]
    future_run = metrics["future_first_active_run"]
    video_id = str(feature["video_id"])
    return {
        "sample_id": f"turn:{video_id}:{prefix_end}:IDLE:v1_8",
        "video_id": video_id,
        "semantic_id": feature["semantic_id"],
        "signer_id": derive_signer_id(video_id),
        "split": feature["split"],
        "rgb_path": feature["rgb_path"],
        "pose_path": feature["pose_path"],
        "task": "turn",
        "task_type": "turn",
        "system_state": "LISTENING",
        "system_speaking": 0,
        "decision_frame": decision_frame,
        "decision_time_sec": rounded(decision_time),
        "system_state_before": "LISTENING",
        "system_speaking_before": 0,
        "label": "IDLE",
        "system_action": "KEEP_LISTENING",
        "system_state_after": "LISTENING",
        "candidate_label": "IDLE",
        "candidate_frame": decision_frame,
        "candidate_time_sec": rounded(decision_time),
        "prefix_start_frame": 0,
        "prefix_end_frame_exclusive": prefix_end,
        "window_end_frame": prefix_end,
        "total_frames": frames,
        "video_frames": frames,
        "fps": rounded(fps),
        "video_duration_sec": rounded(frames / fps),
        "normalized_progress": rounded(prefix_end / frames),
        "active_start_frame": active_start,
        "active_start_sec": rounded(active_start / fps),
        "active_end_frame": active_end,
        "active_end_sec": rounded((active_end + 1) / fps),
        "motion_at_decision": rounded(arrays["combined"][decision_frame]),
        "recent_motion_mean": rounded(means["combined"]),
        "recent_active_fraction": rounded(metrics["recent_active_ratio"]),
        "causal_active_duration_sec": rounded(
            metrics["causal_prefix_longest_active_sec"]
        ),
        "wrist_speed_recent": rounded(max(means["left_wrist"], means["right_wrist"])),
        "internal_motion_recent": rounded(
            max(means["left_internal"], means["right_internal"])
        ),
        "motion_rising_score": rounded((second_half - first_half) / dynamic_range),
        "hand_q20": rounded(feature.get("hand_q20", 0.0)),
        "hand_q50": rounded(feature.get("hand_q50", 0.0)),
        "hand_q80": rounded(feature.get("hand_q80", 0.0)),
        "active_motion_threshold": rounded(feature["active_motion_threshold"]),
        "low_motion_threshold": rounded(feature.get("low_motion_threshold", 0.0)),
        "pose_quality_score": rounded(feature.get("pose_quality_score", 0.0)),
        "boundary_agreement_score": rounded(
            feature.get("boundary_agreement_score", 0.0)
        ),
        "motion_dynamic_range": rounded(feature.get("motion_dynamic_range", 0.0)),
        "quality_gate_pass": True,
        "quality_gate_reason": None,
        "causal_observable": True,
        "future_used_for_label_position": False,
        "future_used_for_accept_reject_only": True,
        "future_validation_only": True,
        "future_metadata_excluded_from_model_input": True,
        "candidate_source": "v1_8_strict_pre_sign_stillness",
        "selection_reason": (
            "fixed frame-aligned prefix is before the first active run with >=0.30s margin; "
            "the recent 0.50s is inactive and every trailing-causal hand/wrist/internal stream "
            "is strictly below its adaptive threshold; future first active run is >=0.70s"
        ),
        "confidence": 0.99,
        "confidence_type": "strict_rule_gate",
        "source_version": "Sign_Aware_Raw_Screen_v1_8",
        "idle_feature_source_version": "Sign_Aware_Raw_Screen_v1_5",
        "idle_nominal_candidate_time_sec": rounded(metrics["nominal_time_sec"]),
        "idle_pre_active_ratio": rounded(metrics["recent_active_ratio"]),
        "idle_recent_motion": rounded(means["combined"]),
        "onset_safety_margin": rounded(metrics["onset_safety_margin_sec"]),
        "onset_safety_margin_sec": rounded(metrics["onset_safety_margin_sec"]),
        "future_active_duration_for_offline_verification": rounded(
            metrics["future_first_active_run_duration_sec"]
        ),
        "future_active_duration_for_offline_verification_sec": rounded(
            metrics["future_first_active_run_duration_sec"]
        ),
        "idle_recent_window_start_frame": recent_start,
        "idle_recent_window_end_frame_exclusive": prefix_end,
        "idle_prefix_offline_longest_active_sec": rounded(
            metrics["offline_prefix_longest_active_sec"]
        ),
        "idle_prefix_causal_longest_active_sec": rounded(
            metrics["causal_prefix_longest_active_sec"]
        ),
        "idle_future_first_active_run_start_frame": int(future_run["start_frame"]),
        "idle_future_first_active_run_end_frame_exclusive": int(
            future_run["end_frame_exclusive"]
        ),
        "idle_all_recent_streams_strictly_low": True,
        "idle_recent_combined_motion_max": rounded(maxima["combined"]),
        "idle_recent_left_hand_motion_max": rounded(maxima["left_hand"]),
        "idle_recent_right_hand_motion_max": rounded(maxima["right_hand"]),
        "idle_recent_left_wrist_speed_max": rounded(maxima["left_wrist"]),
        "idle_recent_right_wrist_speed_max": rounded(maxima["right_wrist"]),
        "idle_recent_left_internal_motion_max": rounded(maxima["left_internal"]),
        "idle_recent_right_internal_motion_max": rounded(maxima["right_internal"]),
        "idle_combined_motion_threshold": rounded(thresholds["combined"]["threshold"]),
        "idle_left_hand_motion_threshold": rounded(
            thresholds["left_hand"]["threshold"]
        ),
        "idle_right_hand_motion_threshold": rounded(
            thresholds["right_hand"]["threshold"]
        ),
        "idle_left_wrist_speed_threshold": rounded(
            thresholds["left_wrist"]["threshold"]
        ),
        "idle_right_wrist_speed_threshold": rounded(
            thresholds["right_wrist"]["threshold"]
        ),
        "idle_left_internal_motion_threshold": rounded(
            thresholds["left_internal"]["threshold"]
        ),
        "idle_right_internal_motion_threshold": rounded(
            thresholds["right_internal"]["threshold"]
        ),
        "left_hand_motion_reliability": 1.0,
        "right_hand_motion_reliability": 1.0,
        "left_wrist_speed_reliability": 1.0,
        "right_wrist_speed_reliability": 1.0,
        "decision_feature_pipeline": (
            "V1.5 serialized trailing-causal combined/left/right hand and wrist curves; "
            "V1.6-compatible causal internal-motion proxy"
        ),
        "model_input_contract": "RGB/Pose prefix [0,prefix_end_frame_exclusive) only",
    }


def screen_idle_features(
    feature_path: Path,
    source_audit: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    seed = int(CONFIG["random_seed"])
    random.seed(seed)
    pools: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    seen_videos: set[str] = set()
    feature_rejections: Counter[str] = Counter()
    candidate_rejections: Counter[str] = Counter()
    candidate_rejections_by_split: Counter[str] = Counter()
    candidate_attempts: Counter[str] = Counter()
    candidate_valid: Counter[str] = Counter()
    valid_point_count_per_video: Counter[str] = Counter()
    feature_rows_by_split: Counter[str] = Counter()
    feature_rows = 0
    feature_rows_eligible = 0
    videos_with_valid_candidate = 0

    with feature_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise SourceIntegrityError(
                    f"blank feature line: {feature_path}:{line_number}"
                )
            feature = json.loads(line)
            feature_rows += 1
            video_id = str(feature.get("video_id", ""))
            if not video_id or video_id in seen_videos:
                raise SourceIntegrityError(
                    f"missing/duplicate feature video_id: {video_id}"
                )
            seen_videos.add(video_id)
            split = str(feature.get("split", ""))
            if split not in SPLITS:
                raise SourceIntegrityError(
                    f"invalid feature split: {video_id}: {split}"
                )
            feature_rows_by_split[split] += 1
            reasons = feature_filter_reasons(feature)
            if reasons:
                feature_rejections.update(reasons)
                continue
            official_split = source_audit["official_video_split"].get(video_id)
            if official_split is None or official_split != split:
                raise SourceIntegrityError(
                    f"eligible feature missing/mismatched in V1.5 official split: {video_id}"
                )
            prepared = prepare_feature(feature)
            feature_rows_eligible += 1
            valid: list[dict[str, Any]] = []
            for nominal in CONFIG["idle"]["candidate_nominal_times_sec"]:
                key = f"{float(nominal):.2f}"
                candidate_attempts[key] += 1
                evaluation = evaluate_idle_candidate(
                    prepared,
                    float(nominal),
                    source_audit["listening_decisions"],
                )
                if evaluation.record is None:
                    for reason in evaluation.reasons:
                        candidate_rejections[reason] += 1
                        candidate_rejections_by_split[f"{split}:{reason}"] += 1
                else:
                    candidate_valid[key] += 1
                    valid.append(evaluation.record)
            valid_point_count_per_video[str(len(valid))] += 1
            if valid:
                # Exactly one randomly chosen fixed candidate per original video.
                pools[split].append(random.choice(valid))
                videos_with_valid_candidate += 1
            else:
                feature_rejections["NO_VALID_FIXED_CANDIDATE"] += 1
            if line_number % 5000 == 0:
                print(
                    f"screened V1.5 causal feature rows: {line_number}; "
                    f"valid-video pool: {videos_with_valid_candidate}",
                    flush=True,
                )

    targets: Mapping[str, int] = source_audit["target_idle"]
    selected: dict[str, list[dict[str, Any]]] = {}
    selection_stats: dict[str, Any] = {}
    for split in SPLITS:
        pool = sorted(pools[split], key=lambda item: item["video_id"])
        target = int(targets[split])
        if len(pool) > target:
            chosen = random.sample(pool, target)
        else:
            chosen = list(pool)
        chosen.sort(key=lambda item: item["video_id"])
        selected[split] = chosen
        selection_stats[split] = {
            "target_v1_5_incomplete_count": target,
            "valid_one_per_video_pool": len(pool),
            "selected": len(chosen),
            "target_reached": len(chosen) == target,
            "downsampled": len(pool) - len(chosen),
            "shortfall_without_rule_relaxation": max(0, target - len(chosen)),
        }

    stats = {
        "version": VERSION,
        "random_seed": seed,
        "feature_path": str(feature_path),
        "feature_rows": feature_rows,
        "feature_rows_by_split": dict(sorted(feature_rows_by_split.items())),
        "feature_rows_passing_quality_and_reliability": feature_rows_eligible,
        "videos_with_at_least_one_valid_fixed_candidate": videos_with_valid_candidate,
        "feature_rejection_reasons": dict(sorted(feature_rejections.items())),
        "candidate_attempts_by_nominal_time_sec": dict(
            sorted(candidate_attempts.items())
        ),
        "valid_candidates_by_nominal_time_sec": dict(sorted(candidate_valid.items())),
        "candidate_rejection_reasons": dict(sorted(candidate_rejections.items())),
        "candidate_rejection_reasons_by_split": dict(
            sorted(candidate_rejections_by_split.items())
        ),
        "valid_fixed_point_count_per_video": dict(
            sorted(valid_point_count_per_video.items())
        ),
        "one_candidate_random_choice": {
            "method": "random.choice(valid fixed points) in V1.5 feature-stream order",
            "seed": seed,
            "maximum_per_video": 1,
        },
        "split_downsampling": {
            "method": (
                "sort one-per-video pool by video_id, then random.sample with the same "
                "global random.seed(42) stream; retain all if below target"
            ),
            "per_split": selection_stats,
        },
        "hard_rules_relaxed": False,
    }
    return selected, stats


def copy_inherited_group(
    source_path: Path,
    labels: frozenset[str],
    target: Any,
) -> int:
    count = 0
    for raw_line, item in iter_raw_jsonl(source_path):
        if item["label"] in labels:
            target.write(raw_line)
            count += 1
    return count


def write_dataset(
    roots: Mapping[str, Path],
    staging: Path,
    idle_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    final_dataset = staging / "final_dataset"
    final_dataset.mkdir()
    for split in SPLITS:
        destination = staging / f"{split}.jsonl"
        temporary = destination.with_suffix(".jsonl.tmp")
        with temporary.open("wb") as target:
            for source_name in SOURCE_ORDER:
                copy_inherited_group(
                    source_jsonl(roots[source_name], split),
                    frozenset(SOURCE_LABELS[source_name]),
                    target,
                )
            for item in idle_by_split[split]:
                target.write(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                    + b"\n"
                )
        temporary.replace(destination)
        os.link(destination, final_dataset / destination.name)


def digest_output_group(path: Path, labels: frozenset[str]) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    for raw_line, item in iter_raw_jsonl(path):
        if item["label"] in labels:
            digest.update(raw_line)
            count += 1
    return {"count": count, "ordered_raw_line_sha256": digest.hexdigest()}


def inherited_raw_line_audit(
    staging: Path,
    source_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    report: dict[str, Any] = {}
    all_exact = True
    for source_name in SOURCE_ORDER:
        report[source_name] = {}
        labels = frozenset(SOURCE_LABELS[source_name])
        for split in SPLITS:
            source_group = source_audit["group_digests"][source_name][split]
            output_group = digest_output_group(staging / f"{split}.jsonl", labels)
            group_exact = source_group == {
                "labels": list(SOURCE_LABELS[source_name]),
                **output_group,
            }
            label_report: dict[str, Any] = {}
            for label in SOURCE_LABELS[source_name]:
                output_label = digest_output_group(
                    staging / f"{split}.jsonl", frozenset((label,))
                )
                source_label = source_audit["label_digests"][source_name][split][label]
                exact = output_label == source_label
                label_report[label] = {
                    "source": source_label,
                    "output": output_label,
                    "byte_and_order_exact": exact,
                }
                all_exact = all_exact and exact
            report[source_name][split] = {
                "labels": list(SOURCE_LABELS[source_name]),
                "source_group": source_group,
                "output_group": output_group,
                "group_raw_lines_and_relative_order_exact": group_exact,
                "per_label": label_report,
            }
            all_exact = all_exact and group_exact
    return report, all_exact


def semantic_overlap_report(semantic_splits: Mapping[str, set[str]]) -> dict[str, Any]:
    overlaps = {
        semantic_id: sorted(splits)
        for semantic_id, splits in semantic_splits.items()
        if len(splits) > 1
    }
    combinations = Counter("+".join(splits) for splits in overlaps.values())
    return {
        "unique_semantic_id_count": len(semantic_splits),
        "cross_split_semantic_id_count": len(overlaps),
        "split_combination_counts": dict(sorted(combinations.items())),
        "examples": [
            {"semantic_id": semantic_id, "splits": splits}
            for semantic_id, splits in sorted(overlaps.items())[:20]
        ],
        "hard_failure": False,
        "note": (
            "Official CSL-Daily splits are preserved by physical video_id.  Existing "
            "semantic_id overlap across different signer recordings is reported and is not "
            "removed, because resplitting would violate the inheritance contract."
        ),
    }


def scan_output(
    staging: Path,
    expected_counts: Mapping[str, Counter[str]],
) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    idles: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    duplicate_sample_ids = 0
    physical_splits: defaultdict[str, set[str]] = defaultdict(set)
    semantic_splits: defaultdict[str, set[str]] = defaultdict(set)
    prefix_start_failures = 0
    prefix_end_failures = 0
    decision_range_failures = 0
    inherited_listening_before_active_start = 0
    split_field_failures = 0
    unexpected_labels = 0
    missing_media = 0
    checked_media: set[tuple[str, str]] = set()
    listening_at_decision: defaultdict[tuple[str, int], set[str]] = defaultdict(set)

    for split in SPLITS:
        path = staging / f"{split}.jsonl"
        for _, item in iter_raw_jsonl(path):
            label = str(item.get("label"))
            counts[split][label] += 1
            if item.get("split") != split:
                split_field_failures += 1
            sample_id = str(item.get("sample_id"))
            if sample_id in sample_ids:
                duplicate_sample_ids += 1
            sample_ids.add(sample_id)
            video_id = str(item.get("video_id"))
            semantic_id = str(item.get("semantic_id"))
            physical_splits[video_id].add(split)
            semantic_splits[semantic_id].add(split)
            if label not in LABELS:
                unexpected_labels += 1
            if int(item.get("prefix_start_frame", -1)) != 0:
                prefix_start_failures += 1
            decision = int(item.get("decision_frame", -1))
            prefix_end = int(item.get("prefix_end_frame_exclusive", -1))
            frames = int(item.get("video_frames", item.get("total_frames", -1)))
            if prefix_end != decision + 1:
                prefix_end_failures += 1
            if not (0 <= decision < frames) or not (1 <= prefix_end <= frames):
                decision_range_failures += 1
            if label in LISTENING_SOURCE_LABELS or label == "IDLE":
                listening_at_decision[(video_id, prefix_end)].add(label)
            if label in LISTENING_SOURCE_LABELS:
                fps = float(item.get("fps", 0.0))
                active_start_frame = int(item.get("active_start_frame", -1))
                decision_time = float(item.get("decision_time_sec", -1.0))
                if fps <= 0.0 or active_start_frame < 0 or decision_time + 1e-9 < active_start_frame / fps:
                    inherited_listening_before_active_start += 1
            media_key = (str(item.get("rgb_path", "")), str(item.get("pose_path", "")))
            if media_key not in checked_media:
                if not Path(media_key[0]).is_file() or not Path(media_key[1]).is_file():
                    missing_media += 1
                checked_media.add(media_key)
            if label == "IDLE":
                idles.append(item)

    physical_leaks = {
        video_id: sorted(splits)
        for video_id, splits in physical_splits.items()
        if len(splits) > 1
    }
    idle_conflicts = {
        f"{video_id}:{prefix_end}": sorted(labels)
        for (video_id, prefix_end), labels in listening_at_decision.items()
        if "IDLE" in labels and len(labels) > 1
    }
    observed_label_set = {label for counter in counts.values() for label in counter}
    count_match = observed_label_set == set(LABELS) and all(
        counts[split] == expected_counts[split] for split in SPLITS
    )
    return {
        "counts": counts,
        "idles": idles,
        "unique_sample_ids": len(sample_ids),
        "duplicate_sample_ids": duplicate_sample_ids,
        "unique_video_ids": len(physical_splits),
        "unique_semantic_ids": len(semantic_splits),
        "physical_split_leakage_count": len(physical_leaks),
        "physical_split_leakage_examples": dict(
            list(sorted(physical_leaks.items()))[:20]
        ),
        "semantic_overlap": semantic_overlap_report(semantic_splits),
        "prefix_start_failures": prefix_start_failures,
        "prefix_end_failures": prefix_end_failures,
        "decision_range_failures": decision_range_failures,
        "inherited_listening_before_active_start": inherited_listening_before_active_start,
        "split_field_failures": split_field_failures,
        "unexpected_labels": unexpected_labels,
        "observed_label_set": sorted(observed_label_set),
        "missing_unique_media_pairs": missing_media,
        "idle_listening_decision_conflict_count": len(idle_conflicts),
        "idle_listening_decision_conflict_examples": dict(
            list(sorted(idle_conflicts.items()))[:20]
        ),
        "expected_counts_match": count_match,
    }


def validate_idle_records(idles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cfg = CONFIG["idle"]
    failures: Counter[str] = Counter()
    video_ids: set[str] = set()
    nominal_allowed = {float(value) for value in cfg["candidate_nominal_times_sec"]}
    for item in idles:
        video_id = str(item["video_id"])
        if video_id in video_ids:
            failures["multiple_idle_per_video"] += 1
        video_ids.add(video_id)
        fps = float(item["fps"])
        decision_time = float(item["decision_time_sec"])
        active_start_time = int(item["active_start_frame"]) / fps
        if not decision_time < active_start_time:
            failures["decision_not_before_active_start"] += 1
        # Selection and the source-feature replay below use the original fps
        # and unrounded timestamps.  JSONL stores fps/times to 8 decimals, so
        # validate the serialized margin field with one serialization quantum
        # instead of recomputing a boundary value from two rounded fields.
        if float(item["onset_safety_margin_sec"]) + 1e-8 < float(
            cfg["onset_safety_margin_sec"]
        ):
            failures["onset_safety_margin"] += 1
        if (
            float(item["idle_pre_active_ratio"])
            > float(cfg["recent_active_ratio_max"]) + 1e-9
        ):
            failures["recent_active_ratio"] += 1
        if float(item["idle_prefix_offline_longest_active_sec"]) + 1e-9 >= float(
            cfg["stable_active_prefix_min_sec"]
        ):
            failures["stable_offline_active_prefix"] += 1
        if float(item["idle_prefix_causal_longest_active_sec"]) + 1e-9 >= float(
            cfg["stable_active_prefix_min_sec"]
        ):
            failures["stable_causal_active_prefix"] += 1
        if float(
            item["future_active_duration_for_offline_verification_sec"]
        ) + 1e-9 < float(cfg["future_first_active_run_min_sec"]):
            failures["future_active_duration"] += 1
        if item.get("idle_all_recent_streams_strictly_low") is not True:
            failures["recent_streams_not_low"] += 1
        comparisons = (
            ("idle_recent_combined_motion_max", "idle_combined_motion_threshold"),
            ("idle_recent_left_hand_motion_max", "idle_left_hand_motion_threshold"),
            ("idle_recent_right_hand_motion_max", "idle_right_hand_motion_threshold"),
            ("idle_recent_left_wrist_speed_max", "idle_left_wrist_speed_threshold"),
            ("idle_recent_right_wrist_speed_max", "idle_right_wrist_speed_threshold"),
            (
                "idle_recent_left_internal_motion_max",
                "idle_left_internal_motion_threshold",
            ),
            (
                "idle_recent_right_internal_motion_max",
                "idle_right_internal_motion_threshold",
            ),
        )
        # These audit values are rounded to 8 decimals in JSONL.  The strict
        # unrounded comparison is repeated against source curves below; allow
        # only the serialization quantum here so a true x<threshold pair that
        # rounds to equality is not reported as a failure.
        if any(
            float(item[left]) > float(item[right]) + 1e-8 for left, right in comparisons
        ):
            failures["stored_per_frame_max_not_below_threshold"] += 1
        if any(
            abs(float(item[field]) - 1.0) > 1e-12
            for field in CONFIG["idle"]["required_reliability_fields"]
        ):
            failures["feature_reliability"] += 1
        if item.get("quality_gate_pass") is not True:
            failures["quality_gate"] += 1
        if item.get("source_version") != VERSION:
            failures["idle_source_version"] += 1
        if item.get("idle_feature_source_version") != "Sign_Aware_Raw_Screen_v1_5":
            failures["idle_feature_source_version"] += 1
        if float(item["idle_nominal_candidate_time_sec"]) not in nominal_allowed:
            failures["candidate_not_from_fixed_set"] += 1
        expected_end = align_prefix_end(
            float(item["idle_nominal_candidate_time_sec"]),
            fps,
            int(item["video_frames"]),
        )
        if expected_end != int(item["prefix_end_frame_exclusive"]):
            failures["candidate_frame_alignment"] += 1
        if item.get("future_used_for_label_position") is not False:
            failures["future_label_position_flag"] += 1
        if item.get("future_metadata_excluded_from_model_input") is not True:
            failures["future_metadata_model_input_flag"] += 1
    return {
        "checked_idle_records": len(idles),
        "unique_idle_video_ids": len(video_ids),
        "failure_counts": dict(sorted(failures.items())),
        "all_pass": not failures,
    }


def deep_revalidate_selected_idle(
    feature_path: Path,
    selected_idles: Sequence[Mapping[str, Any]],
    listening_decisions: set[tuple[str, int]],
) -> dict[str, Any]:
    selected = {str(item["video_id"]): item for item in selected_idles}
    if len(selected) != len(selected_idles):
        return {
            "checked": 0,
            "missing": len(selected_idles) - len(selected),
            "mismatched": 0,
            "all_pass": False,
        }
    found: set[str] = set()
    mismatches: list[str] = []
    with feature_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            feature = json.loads(line)
            video_id = str(feature["video_id"])
            original = selected.get(video_id)
            if original is None:
                continue
            found.add(video_id)
            if feature_filter_reasons(feature):
                mismatches.append(f"{video_id}:feature_filter")
                continue
            evaluation = evaluate_idle_candidate(
                prepare_feature(feature),
                float(original["idle_nominal_candidate_time_sec"]),
                listening_decisions,
            )
            rebuilt = evaluation.record
            if rebuilt is None:
                mismatches.append(f"{video_id}:{','.join(evaluation.reasons)}")
                continue
            exact_fields = (
                "sample_id",
                "split",
                "decision_frame",
                "prefix_end_frame_exclusive",
                "active_start_frame",
                "idle_future_first_active_run_start_frame",
                "idle_future_first_active_run_end_frame_exclusive",
            )
            numeric_fields = (
                "decision_time_sec",
                "onset_safety_margin_sec",
                "idle_pre_active_ratio",
                "idle_recent_combined_motion_max",
                "future_active_duration_for_offline_verification_sec",
            )
            if any(rebuilt[field] != original[field] for field in exact_fields) or any(
                abs(float(rebuilt[field]) - float(original[field])) > 1e-8
                for field in numeric_fields
            ):
                mismatches.append(f"{video_id}:record_rebuild_mismatch")
    missing = sorted(set(selected) - found)
    return {
        "checked": len(found),
        "missing": len(missing),
        "missing_examples": missing[:20],
        "mismatched": len(mismatches),
        "mismatch_examples": mismatches[:20],
        "all_pass": len(found) == len(selected) and not missing and not mismatches,
    }


def hard_link_audit(staging: Path) -> tuple[dict[str, Any], bool]:
    report: dict[str, Any] = {}
    all_hard_links = True
    for split in SPLITS:
        root_file = staging / f"{split}.jsonl"
        final_file = staging / "final_dataset" / f"{split}.jsonl"
        root_stat = root_file.stat()
        final_stat = final_file.stat()
        exact = (
            root_stat.st_dev == final_stat.st_dev
            and root_stat.st_ino == final_stat.st_ino
            and sha256(root_file) == sha256(final_file)
        )
        report[split] = {
            "root_path_relative_to_output": f"{split}.jsonl",
            "final_dataset_path_relative_to_output": f"final_dataset/{split}.jsonl",
            "same_device_and_inode": (
                root_stat.st_dev == final_stat.st_dev
                and root_stat.st_ino == final_stat.st_ino
            ),
            "sha256": sha256(root_file),
            "hard_link_exact": exact,
        }
        all_hard_links = all_hard_links and exact
    return report, all_hard_links


def build_reports(
    output: Path,
    staging: Path,
    roots: Mapping[str, Path],
    feature_path: Path,
    source_audit: Mapping[str, Any],
    idle_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    idle_filter_stats: dict[str, Any],
    source_manifest_before: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_counts: dict[str, Counter[str]] = {}
    for split in SPLITS:
        expected_counts[split] = Counter(source_audit["inherited_counts"][split])
        expected_counts[split]["IDLE"] = len(idle_by_split[split])
    output_scan = scan_output(staging, expected_counts)
    inherited_report, inherited_exact = inherited_raw_line_audit(staging, source_audit)
    idle_validation = validate_idle_records(output_scan["idles"])
    deep_idle = deep_revalidate_selected_idle(
        feature_path,
        output_scan["idles"],
        source_audit["listening_decisions"],
    )
    hard_links, hard_links_exact = hard_link_audit(staging)
    source_manifest_after = source_manifest(roots, feature_path)
    sources_unchanged = source_manifest_before == source_manifest_after

    checks = {
        "01_six_class_label_counts_correct": bool(output_scan["expected_counts_match"]),
        "02_all_idle_decisions_before_active_start": all(
            float(item["decision_time_sec"])
            < int(item["active_start_frame"]) / float(item["fps"])
            for item in output_scan["idles"]
        ),
        "03_all_idle_onset_safety_margins_at_least_0_30s": all(
            float(item["onset_safety_margin_sec"]) + 1e-9
            >= float(CONFIG["idle"]["onset_safety_margin_sec"])
            for item in output_scan["idles"]
        ),
        "04_all_idle_prefixes_have_no_stable_active_signing": (
            idle_validation["failure_counts"].get("stable_offline_active_prefix", 0)
            == 0
            and idle_validation["failure_counts"].get("stable_causal_active_prefix", 0)
            == 0
        ),
        "05_all_idle_have_future_active_signing_at_least_0_70s": (
            idle_validation["failure_counts"].get("future_active_duration", 0) == 0
        ),
        "06_idle_has_no_exact_decision_conflict_with_other_listening_labels": (
            output_scan["idle_listening_decision_conflict_count"] == 0
        ),
        "07_other_five_labels_raw_lines_and_source_order_exact": inherited_exact,
        "08_physical_video_id_split_leakage_zero": (
            output_scan["physical_split_leakage_count"] == 0
        ),
        "09_all_causal_prefixes_start_at_zero": output_scan["prefix_start_failures"]
        == 0,
        "10_all_decisions_within_original_video": output_scan["decision_range_failures"]
        == 0,
        "idle_all_quality_reliability_and_low_motion_checks_pass": idle_validation[
            "all_pass"
        ],
        "idle_second_pass_feature_revalidation_pass": deep_idle["all_pass"],
        "all_inherited_listening_decisions_at_or_after_active_start": (
            output_scan["inherited_listening_before_active_start"] == 0
        ),
        "all_prefix_ends_equal_decision_plus_one": output_scan["prefix_end_failures"]
        == 0,
        "all_record_split_fields_match_files": output_scan["split_field_failures"] == 0,
        "no_duplicate_sample_ids": output_scan["duplicate_sample_ids"] == 0,
        "no_unexpected_labels": output_scan["unexpected_labels"] == 0,
        "all_referenced_rgb_pose_exist": output_scan["missing_unique_media_pairs"] == 0,
        "source_hashes_unchanged_during_build": sources_unchanged,
        "root_and_final_dataset_jsonl_are_hard_links": hard_links_exact,
    }
    hard_pass = all(checks.values())
    validation_report = {
        "version": VERSION,
        "status": "PASS" if hard_pass else "FAIL",
        "checks": checks,
        "metrics": {
            "physical_split_leakage_count": output_scan["physical_split_leakage_count"],
            "semantic_cross_split_overlap_count_report_only": output_scan[
                "semantic_overlap"
            ]["cross_split_semantic_id_count"],
            "idle_listening_decision_conflict_count": output_scan[
                "idle_listening_decision_conflict_count"
            ],
            "duplicate_sample_ids": output_scan["duplicate_sample_ids"],
            "prefix_start_failures": output_scan["prefix_start_failures"],
            "prefix_end_failures": output_scan["prefix_end_failures"],
            "decision_range_failures": output_scan["decision_range_failures"],
            "inherited_listening_before_active_start": output_scan[
                "inherited_listening_before_active_start"
            ],
            "missing_unique_media_pairs": output_scan["missing_unique_media_pairs"],
        },
        "idle_validation": idle_validation,
        "idle_second_pass_feature_revalidation": deep_idle,
        "inherited_raw_line_audit": inherited_report,
        "semantic_overlap_report_only": output_scan["semantic_overlap"],
        "hard_link_audit": hard_links,
        "source_manifest_before": source_manifest_before,
        "source_manifest_after": source_manifest_after,
    }

    idles = output_scan["idles"]
    dataset_stats = {
        "version": VERSION,
        "output_directory": str(output),
        "counts": {
            split: {
                "total": sum(output_scan["counts"][split].values()),
                "by_label": {
                    label: int(output_scan["counts"][split].get(label, 0))
                    for label in LABELS
                },
            }
            for split in SPLITS
        },
        "total": sum(
            sum(counter.values()) for counter in output_scan["counts"].values()
        ),
        "idle_actual": {
            split: int(output_scan["counts"][split].get("IDLE", 0)) for split in SPLITS
        },
        "idle_target_v1_5_incomplete": dict(source_audit["target_idle"]),
        "idle_distributions": {
            "decision_time_sec": describe(item["decision_time_sec"] for item in idles),
            "onset_safety_margin_sec": describe(
                item["onset_safety_margin_sec"] for item in idles
            ),
            "recent_active_ratio": describe(
                item["idle_pre_active_ratio"] for item in idles
            ),
            "recent_combined_motion_max": describe(
                item["idle_recent_combined_motion_max"] for item in idles
            ),
            "future_first_active_run_duration_sec": describe(
                item["future_active_duration_for_offline_verification_sec"]
                for item in idles
            ),
        },
        "unique_video_ids": output_scan["unique_video_ids"],
        "unique_semantic_ids": output_scan["unique_semantic_ids"],
        "semantic_overlap_report_only": output_scan["semantic_overlap"],
        "source_versions_by_label": CONFIG["inheritance"],
        "causal_prefix_contract": (
            "prefix_start_frame=0; prefix_end_frame_exclusive=decision_frame+1; "
            "decision_time_sec=prefix_end_frame_exclusive/fps"
        ),
        "other_five_labels_raw_line_and_order_exact": inherited_exact,
        "validation_status": validation_report["status"],
    }

    idle_filter_stats = dict(idle_filter_stats)
    idle_filter_stats.update(
        {
            "feature_sha256": source_manifest_before["v1_5_pose_sign_features"][
                "sha256"
            ],
            "idle_selected_total": len(idles),
            "idle_selected_by_split": {
                split: len(idle_by_split[split]) for split in SPLITS
            },
            "idle_selected_decision_time_distribution": describe(
                item["decision_time_sec"] for item in idles
            ),
            "idle_selected_nominal_time_counts": dict(
                sorted(
                    Counter(
                        str(item["idle_nominal_candidate_time_sec"]) for item in idles
                    ).items()
                )
            ),
        }
    )

    build_config = json.loads(json.dumps(CONFIG))
    build_config.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "resolved_sources": {name: str(root) for name, root in roots.items()},
            "resolved_v1_5_feature_path": str(feature_path),
            "resolved_output": str(output),
            "source_manifest_before": source_manifest_before,
            "source_manifest_after": source_manifest_after,
            "source_hashes_unchanged_during_build": sources_unchanged,
            "output_line_block_order": [
                {
                    "source": source_name,
                    "labels": list(SOURCE_LABELS[source_name]),
                    "within_source_order": "exact raw source order",
                }
                for source_name in SOURCE_ORDER
            ]
            + [
                {
                    "source": "v1_8_idle",
                    "labels": ["IDLE"],
                    "within_source_order": "selected records sorted by video_id",
                }
            ],
            "output_jsonl": {
                split: {
                    "path": str(output / f"{split}.jsonl"),
                    "sha256": sha256(staging / f"{split}.jsonl"),
                    "size": (staging / f"{split}.jsonl").stat().st_size,
                }
                for split in SPLITS
            },
        }
    )
    return dataset_stats, idle_filter_stats, validation_report, build_config


def ensure_independent_paths(roots: Mapping[str, Path], output: Path) -> None:
    if output.exists():
        raise FileExistsError(
            f"V1.8 output already exists; refusing to overwrite: {output}"
        )
    for source_name, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(root)
        if output == root or output in root.parents or root in output.parents:
            raise RuntimeError(
                f"output must be independent of source {source_name}: output={output} source={root}"
            )


def main() -> int:
    args = parse_args()
    roots = {
        "v1_5": args.v1_5.resolve(),
        "v1_6": args.v1_6.resolve(),
        "v1_7": args.v1_7.resolve(),
    }
    output = args.output.resolve()
    ensure_independent_paths(roots, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_path = feature_jsonl(roots["v1_5"])

    print("hashing immutable V1.5/V1.6/V1.7 sources", flush=True)
    manifest_before = source_manifest(roots, feature_path)
    print("auditing source schemas and raw-line inheritance sets", flush=True)
    source_audit = scan_sources(roots)
    print("streaming V1.5 causal features for strict IDLE candidates", flush=True)
    idle_by_split, idle_filter_stats = screen_idle_features(feature_path, source_audit)
    print(
        "selected IDLE: "
        + ", ".join(f"{split}={len(idle_by_split[split])}" for split in SPLITS),
        flush=True,
    )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=str(output.parent))
    )
    try:
        write_dataset(roots, staging, idle_by_split)
        dataset_stats, filter_stats, validation, build_config = build_reports(
            output=output,
            staging=staging,
            roots=roots,
            feature_path=feature_path,
            source_audit=source_audit,
            idle_by_split=idle_by_split,
            idle_filter_stats=idle_filter_stats,
            source_manifest_before=manifest_before,
        )
        if validation["status"] != "PASS":
            raise RuntimeError(
                "V1.8 hard validation failed: "
                + json.dumps(validation["checks"], ensure_ascii=False, sort_keys=True)
            )
        atomic_json(staging / "dataset_stats.json", dataset_stats)
        atomic_json(staging / "idle_filter_stats.json", filter_stats)
        atomic_json(staging / "validation_report.json", validation)
        atomic_json(staging / "BUILD_CONFIG.json", build_config)
        if output.exists():
            raise FileExistsError(
                f"output appeared during build; refusing to overwrite: {output}"
            )
        staging.replace(output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(json.dumps(dataset_stats["counts"], ensure_ascii=False, indent=2), flush=True)
    print(f"validation: {validation['status']}", flush=True)
    print(f"output: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
