#!/usr/bin/env python3
"""Export the deterministic SignDuplex V1.8 six-class visual audit package.

The exporter reads the already-built V1.8 JSONL files.  It never changes the
dataset or source RGB/Pose files.  Each selected sample gets an exact causal
RGB prefix, a byte-identical full RGB video (hard link when possible), and a
label/audit JSON file.  Automatic media validation is deliberately reported
separately from the pending human review.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get("SIGNDUPLEX_DATA_DIR", REPO_ROOT / "data" / "Sign_Aware_Raw_Screen_v1_8")
).expanduser()
DEFAULT_OUTPUT = Path(
    os.environ.get("SIGNDUPLEX_VISUAL_CHECK_DIR", REPO_ROOT / "outputs" / "visual_check")
).expanduser()
DEFAULT_SEED = 42
DEFAULT_SAMPLES_PER_CLASS = 10
SPLITS = ("train", "dev", "test")
LABELS = ("IDLE", "INCOMPLETE", "BACKCHANNEL", "COMPLETE", "NO_BARGE", "BARGE")
SOURCE_VERSION_BY_LABEL = {
    "IDLE": "Sign_Aware_Raw_Screen_v1_8",
    "INCOMPLETE": "Sign_Aware_Raw_Screen_v1_5",
    "BACKCHANNEL": "Sign_Aware_Raw_Screen_v1_5",
    "COMPLETE": "Sign_Aware_Raw_Screen_v1_7",
    "NO_BARGE": "Sign_Aware_Raw_Screen_v1_6",
    "BARGE": "Sign_Aware_Raw_Screen_v1_6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help="default 10 as required by V1.8; a smaller value is useful only for isolated smoke tests",
    )
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "").strip()
        raise RuntimeError(
            f"command failed ({error.returncode}): {' '.join(command)}"
            + (f"\n{stderr}" if stderr else "")
        ) from error


def valid_number(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def valid_integer(value: Any) -> int | None:
    number = valid_number(value)
    if number is None or number < 0 or not float(number).is_integer():
        return None
    return int(number)


def probe_video(path: Path) -> dict[str, Any]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration,width,height:format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"expected exactly one video stream in {path}, got {len(streams)}")
    stream = streams[0]
    frame_count = valid_integer(stream.get("nb_read_frames"))
    if frame_count is None:
        frame_count = valid_integer(stream.get("nb_frames"))
    if frame_count is None:
        raise RuntimeError(f"ffprobe did not return a frame count for {path}")
    fps_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    try:
        fps = float(Fraction(str(fps_text)))
    except (ValueError, ZeroDivisionError) as error:
        raise RuntimeError(f"invalid ffprobe frame rate for {path}: {fps_text}") from error
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"invalid ffprobe fps for {path}: {fps}")
    duration = valid_number(stream.get("duration"))
    if duration is None:
        duration = valid_number((payload.get("format") or {}).get("duration"))
    if duration is None or duration <= 0:
        raise RuntimeError(f"ffprobe did not return a positive duration for {path}")
    return {
        "path": str(path),
        "frame_count": frame_count,
        "fps": fps,
        "duration_sec": duration,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "size_bytes": path.stat().st_size,
    }


def make_input_prefix(source: Path, destination: Path, prefix_end_frame: int) -> None:
    if prefix_end_frame <= 0:
        raise ValueError(f"prefix_end_frame must be positive: {prefix_end_frame}")
    run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vf",
            f"trim=start_frame=0:end_frame={prefix_end_frame},setpts=PTS-STARTPTS",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )


def link_or_copy_full(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def require_fields(row: dict[str, Any], fields: Iterable[str], context: str) -> None:
    missing = [field for field in fields if field not in row or row[field] is None]
    if missing:
        raise RuntimeError(f"{context}: missing required fields {missing}")


def load_rows(data_root: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    seen_sample_ids: set[str] = set()
    for split in SPLITS:
        path = data_root / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"{path}:{line_number}: invalid JSON: {error}") from error
                context = f"{path}:{line_number}"
                require_fields(
                    row,
                    (
                        "sample_id",
                        "video_id",
                        "semantic_id",
                        "split",
                        "label",
                        "rgb_path",
                        "pose_path",
                    ),
                    context,
                )
                if row["split"] != split:
                    raise RuntimeError(f"{context}: split field is {row['split']!r}, expected {split!r}")
                label = row["label"]
                if label not in rows:
                    raise RuntimeError(f"{context}: unexpected V1.8 label {label!r}")
                sample_id = str(row["sample_id"])
                if sample_id in seen_sample_ids:
                    raise RuntimeError(f"{context}: duplicate sample_id {sample_id}")
                seen_sample_ids.add(sample_id)
                row["_source_jsonl"] = str(path)
                row["_source_line_number"] = line_number
                rows[label].append(row)
    return rows


def choose_rows(
    rows: dict[str, list[dict[str, Any]]], seed: int, count: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if count <= 0:
        raise ValueError("samples-per-class must be positive")
    rng = random.Random(seed)
    selected: dict[str, list[dict[str, Any]]] = {}
    global_videos: set[str] = set()
    global_semantics: set[str] = set()
    fallback_labels: list[str] = []

    for label in LABELS:
        candidates = sorted(
            rows[label],
            key=lambda row: (
                str(row["sample_id"]),
                str(row["video_id"]),
                str(row["semantic_id"]),
            ),
        )
        rng.shuffle(candidates)
        chosen: list[dict[str, Any]] = []
        used_videos: set[str] = set()
        used_semantics: set[str] = set()
        used_samples: set[str] = set()

        # Prefer all-60 global uniqueness while always preserving the stricter
        # per-class video_id and semantic_id uniqueness contract.
        for row in candidates:
            video_id = str(row["video_id"])
            semantic_id = str(row["semantic_id"])
            if video_id in global_videos or semantic_id in global_semantics:
                continue
            if video_id in used_videos or semantic_id in used_semantics:
                continue
            chosen.append(row)
            used_videos.add(video_id)
            used_semantics.add(semantic_id)
            used_samples.add(str(row["sample_id"]))
            if len(chosen) == count:
                break

        # A global collision must never make an otherwise valid class short.
        # Fall back only to candidates still unique inside this class.
        if len(chosen) < count:
            fallback_labels.append(label)
            for row in candidates:
                video_id = str(row["video_id"])
                semantic_id = str(row["semantic_id"])
                sample_id = str(row["sample_id"])
                if sample_id in used_samples:
                    continue
                if video_id in used_videos or semantic_id in used_semantics:
                    continue
                chosen.append(row)
                used_videos.add(video_id)
                used_semantics.add(semantic_id)
                used_samples.add(sample_id)
                if len(chosen) == count:
                    break

        if len(chosen) != count:
            unique_videos = len({str(row["video_id"]) for row in candidates})
            unique_semantics = len({str(row["semantic_id"]) for row in candidates})
            raise RuntimeError(
                f"cannot select {count} {label} rows with unique video_id and semantic_id; "
                f"selected={len(chosen)}, candidates={len(candidates)}, "
                f"unique_videos={unique_videos}, unique_semantics={unique_semantics}"
            )
        selected[label] = chosen
        global_videos.update(str(row["video_id"]) for row in chosen)
        global_semantics.update(str(row["semantic_id"]) for row in chosen)

    flat = [row for label in LABELS for row in selected[label]]
    global_video_count = len({str(row["video_id"]) for row in flat})
    global_semantic_count = len({str(row["semantic_id"]) for row in flat})
    metadata = {
        "strategy": (
            "seeded shuffle after stable sort; prefer global video_id+semantic_id uniqueness, "
            "then fall back to per-class uniqueness only if necessary"
        ),
        "global_uniqueness_preferred": True,
        "global_uniqueness_fallback_labels": fallback_labels,
        "global_video_id_unique": global_video_count == len(flat),
        "global_semantic_id_unique": global_semantic_count == len(flat),
        "global_unique_video_id_count": global_video_count,
        "global_unique_semantic_id_count": global_semantic_count,
    }
    return selected, metadata


def source_version(row: dict[str, Any]) -> str:
    expected = SOURCE_VERSION_BY_LABEL[str(row["label"])]
    present = row.get("source_version")
    if present is not None and str(present) != expected:
        raise RuntimeError(
            f"{row['sample_id']}: source_version {present!r} conflicts with required {expected!r}"
        )
    return expected


def first_value(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def row_timing(row: dict[str, Any]) -> dict[str, Any]:
    context = str(row["sample_id"])
    require_fields(
        row,
        (
            "decision_frame",
            "prefix_start_frame",
            "prefix_end_frame_exclusive",
            "window_end_frame",
            "fps",
            "video_frames",
            "active_start_frame",
            "active_end_frame",
        ),
        context,
    )
    fps = float(row["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"{context}: invalid fps {fps}")
    decision_frame = int(row["decision_frame"])
    prefix_start = int(row["prefix_start_frame"])
    prefix_end = int(row["prefix_end_frame_exclusive"])
    window_end = int(row["window_end_frame"])
    video_frames = int(row["video_frames"])
    if prefix_start != 0:
        raise RuntimeError(f"{context}: prefix_start_frame must be 0, got {prefix_start}")
    if not (prefix_end == window_end == decision_frame + 1):
        raise RuntimeError(
            f"{context}: prefix contract mismatch: prefix_end={prefix_end}, "
            f"window_end={window_end}, decision_frame={decision_frame}"
        )
    if not 0 < prefix_end <= video_frames:
        raise RuntimeError(f"{context}: prefix_end {prefix_end} outside video_frames {video_frames}")
    decision_time = float(
        first_value(row, ("decision_time_sec", "candidate_time_sec"))
        if first_value(row, ("decision_time_sec", "candidate_time_sec")) is not None
        else prefix_end / fps
    )
    expected_decision_time = prefix_end / fps
    if abs(decision_time - expected_decision_time) > 1e-6:
        raise RuntimeError(
            f"{context}: decision_time {decision_time} != prefix_end/fps {expected_decision_time}"
        )
    video_duration = float(
        row.get("video_duration_sec", video_frames / fps)
    )
    if abs(video_duration - video_frames / fps) > max(0.002, 0.02 / fps):
        raise RuntimeError(
            f"{context}: video_duration_sec {video_duration} != video_frames/fps {video_frames/fps}"
        )
    active_start_frame = int(row["active_start_frame"])
    active_end_frame = int(row["active_end_frame"])
    active_start_time = float(
        first_value(row, ("active_start_time", "active_start_time_sec", "active_start_sec"))
        if first_value(row, ("active_start_time", "active_start_time_sec", "active_start_sec"))
        is not None
        else active_start_frame / fps
    )
    active_end_time = float(
        first_value(row, ("active_end_time", "active_end_time_sec", "active_end_sec"))
        if first_value(row, ("active_end_time", "active_end_time_sec", "active_end_sec"))
        is not None
        else (active_end_frame + 1) / fps
    )
    return {
        "fps": fps,
        "decision_frame": decision_frame,
        "decision_time": decision_time,
        "prefix_start": prefix_start,
        "prefix_end": prefix_end,
        "video_frames": video_frames,
        "video_duration": video_duration,
        "active_start_frame": active_start_frame,
        "active_start_time": active_start_time,
        "active_end_frame": active_end_frame,
        "active_end_time": active_end_time,
    }


def idle_audit_fields(row: dict[str, Any], timing: dict[str, Any]) -> dict[str, Any]:
    ratio = first_value(row, ("idle_pre_active_ratio", "recent_active_fraction"))
    recent_motion = first_value(row, ("idle_recent_motion",))
    if recent_motion is None:
        motion_parts = {
            key: row[key]
            for key in ("recent_motion_mean", "wrist_speed_recent", "internal_motion_recent")
            if key in row and row[key] is not None
        }
        recent_motion = motion_parts or None
    margin = first_value(row, ("onset_safety_margin", "onset_safety_margin_sec"))
    if margin is None:
        margin = timing["active_start_time"] - timing["decision_time"]
    future_duration = first_value(
        row,
        (
            "future_active_duration_for_offline_verification",
            "future_active_duration_sec",
            "future_longest_active_sec",
        ),
    )
    values = {
        "idle_pre_active_ratio": ratio,
        "idle_recent_motion": recent_motion,
        "onset_safety_margin": margin,
        "future_active_duration_for_offline_verification": future_duration,
    }
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise RuntimeError(f"{row['sample_id']}: cannot populate IDLE audit fields {missing}")
    if float(ratio) > 0.10 + 1e-8:
        raise RuntimeError(f"{row['sample_id']}: IDLE active ratio exceeds 0.10: {ratio}")
    if float(margin) < 0.30 - 1e-8:
        raise RuntimeError(f"{row['sample_id']}: IDLE onset margin below 0.30s: {margin}")
    if float(future_duration) < 0.70 - 1e-8:
        raise RuntimeError(
            f"{row['sample_id']}: future active duration below 0.70s: {future_duration}"
        )
    if timing["decision_time"] >= timing["active_start_time"]:
        raise RuntimeError(f"{row['sample_id']}: IDLE decision is not before active_start")
    return values


def safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "sample"


def duration_matches(actual: float, expected: float, fps: float) -> bool:
    return abs(actual - expected) <= max(0.015, 1.05 / fps)


def validate_media(
    row: dict[str, Any],
    timing: dict[str, Any],
    source_probe: dict[str, Any],
    input_probe: dict[str, Any],
    full_probe: dict[str, Any],
    source: Path,
    full_video: Path,
    storage_mode: str,
) -> dict[str, Any]:
    expected_input_duration = timing["prefix_end"] / timing["fps"]
    checks = {
        "source_frame_count_matches_record": source_probe["frame_count"] == timing["video_frames"],
        "source_duration_matches_record": duration_matches(
            source_probe["duration_sec"], timing["video_duration"], timing["fps"]
        ),
        "input_frame_count_matches_prefix_end": input_probe["frame_count"] == timing["prefix_end"],
        "input_duration_matches_decision_time": duration_matches(
            input_probe["duration_sec"], expected_input_duration, timing["fps"]
        ),
        "full_frame_count_matches_record": full_probe["frame_count"] == timing["video_frames"],
        "full_duration_matches_source": duration_matches(
            full_probe["duration_sec"], source_probe["duration_sec"], timing["fps"]
        ),
        "full_file_size_matches_source": full_video.stat().st_size == source.stat().st_size,
        "input_not_longer_than_full": input_probe["duration_sec"] <= full_probe["duration_sec"] + 1e-6,
    }
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"{row['sample_id']}: media validation failed: {failures}")
    return {
        "status": "PASS_AUTOMATIC_MEDIA_VALIDATION",
        "full_storage_mode": storage_mode,
        "expected": {
            "input_frame_count": timing["prefix_end"],
            "input_duration_sec": expected_input_duration,
            "full_frame_count": timing["video_frames"],
            "full_duration_sec": timing["video_duration"],
        },
        "observed": {
            "source": source_probe,
            "input": input_probe,
            "full": full_probe,
        },
        "checks": checks,
    }


def make_label_record(
    row: dict[str, Any],
    timing: dict[str, Any],
    relative_input: Path,
    relative_full: Path,
    media_validation: dict[str, Any],
) -> dict[str, Any]:
    state = first_value(row, ("system_state_before", "system_state"))
    if state is None:
        raise RuntimeError(f"{row['sample_id']}: missing system state")
    record: dict[str, Any] = {
        "sample_id": row["sample_id"],
        "video_id": row["video_id"],
        "semantic_id": row["semantic_id"],
        "split": row["split"],
        "label": row["label"],
        "system_state": state,
        "decision_time": timing["decision_time"],
        "decision_time_sec": timing["decision_time"],
        "decision_frame": timing["decision_frame"],
        "video_duration": timing["video_duration"],
        "video_duration_sec": timing["video_duration"],
        "video_frames": timing["video_frames"],
        "fps": timing["fps"],
        "active_start_time": timing["active_start_time"],
        "active_start_time_sec": timing["active_start_time"],
        "active_start_frame": timing["active_start_frame"],
        "active_end_time": timing["active_end_time"],
        "active_end_time_sec": timing["active_end_time"],
        "active_end_frame": timing["active_end_frame"],
        "rgb_path": row["rgb_path"],
        "pose_path": row["pose_path"],
        "prefix_start": timing["prefix_start"],
        "prefix_end": timing["prefix_end"],
        "prefix_start_frame": timing["prefix_start"],
        "prefix_end_frame_exclusive": timing["prefix_end"],
        "source_version": source_version(row),
        "input_video": str(relative_input),
        "full_video": str(relative_full),
        "prefix_contract": "RGB frames [0,prefix_end_frame_exclusive); decision_frame is included",
        "media_validation": media_validation,
        "manual_review_completed": False,
        "manual_review_status": "PENDING_MANUAL_REVIEW",
        "source_training_jsonl": row["_source_jsonl"],
        "source_training_line_number": row["_source_line_number"],
    }
    if row["label"] == "IDLE":
        record.update(idle_audit_fields(row, timing))
    # Keep the exact parsed training record available for detailed review while
    # excluding only exporter-private provenance helpers.
    record["source_training_record"] = {
        key: value for key, value in row.items() if not key.startswith("_source_")
    }
    return record


def export_package(
    data_root: Path,
    output: Path,
    seed: int,
    samples_per_class: int,
) -> dict[str, Any]:
    data_root = data_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing visual-check output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(data_root)
    selected, selection_metadata = choose_rows(rows, seed, samples_per_class)
    for label in LABELS:
        if len(selected[label]) != samples_per_class:
            raise RuntimeError(f"selection count mismatch for {label}")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    probe_cache: dict[Path, dict[str, Any]] = {}
    summary_items: list[dict[str, Any]] = []
    try:
        for label in LABELS:
            label_dir = staging / label
            label_dir.mkdir()
            for index, row in enumerate(selected[label], 1):
                timing = row_timing(row)
                source = Path(str(row["rgb_path"])).resolve()
                pose_path = Path(str(row["pose_path"])).resolve()
                if not source.is_file():
                    raise FileNotFoundError(source)
                if not pose_path.is_file():
                    raise FileNotFoundError(pose_path)
                base = f"{index:02d}_{safe_component(str(row['video_id']))}"
                input_video = label_dir / f"{base}_input.mp4"
                full_video = label_dir / f"{base}_full.mp4"
                label_json = label_dir / f"{base}_label.json"
                make_input_prefix(source, input_video, timing["prefix_end"])
                storage_mode = link_or_copy_full(source, full_video)

                if source not in probe_cache:
                    probe_cache[source] = probe_video(source)
                source_probe = probe_cache[source]
                input_probe = probe_video(input_video)
                full_probe = probe_video(full_video)
                media_validation = validate_media(
                    row,
                    timing,
                    source_probe,
                    input_probe,
                    full_probe,
                    source,
                    full_video,
                    storage_mode,
                )
                relative_input = input_video.relative_to(staging)
                relative_full = full_video.relative_to(staging)
                relative_label = label_json.relative_to(staging)
                label_record = make_label_record(
                    row,
                    timing,
                    relative_input,
                    relative_full,
                    media_validation,
                )
                write_json(label_json, label_record)
                summary_items.append(
                    {
                        "class_index": index,
                        "sample_id": row["sample_id"],
                        "video_id": row["video_id"],
                        "semantic_id": row["semantic_id"],
                        "split": row["split"],
                        "label": label,
                        "source_version": source_version(row),
                        "input_video": str(relative_input),
                        "full_video": str(relative_full),
                        "label_json": str(relative_label),
                        "media_validation": media_validation,
                        "manual_review_completed": False,
                    }
                )

        per_class: dict[str, Any] = {}
        for label in LABELS:
            items = [item for item in summary_items if item["label"] == label]
            per_class[label] = {
                "count": len(items),
                "split_counts": dict(sorted(Counter(item["split"] for item in items).items())),
                "unique_video_id_count": len({item["video_id"] for item in items}),
                "unique_semantic_id_count": len({item["semantic_id"] for item in items}),
                "video_id_unique": len({item["video_id"] for item in items}) == len(items),
                "semantic_id_unique": len({item["semantic_id"] for item in items}) == len(items),
            }
        automatic_checks = {
            "six_class_counts_exact": all(
                per_class[label]["count"] == samples_per_class for label in LABELS
            ),
            "per_class_video_id_unique": all(per_class[label]["video_id_unique"] for label in LABELS),
            "per_class_semantic_id_unique": all(
                per_class[label]["semantic_id_unique"] for label in LABELS
            ),
            "all_input_frame_counts_verified": all(
                item["media_validation"]["checks"]["input_frame_count_matches_prefix_end"]
                for item in summary_items
            ),
            "all_full_frame_counts_verified": all(
                item["media_validation"]["checks"]["full_frame_count_matches_record"]
                for item in summary_items
            ),
            "all_input_durations_verified": all(
                item["media_validation"]["checks"]["input_duration_matches_decision_time"]
                for item in summary_items
            ),
            "all_full_durations_verified": all(
                item["media_validation"]["checks"]["full_duration_matches_source"]
                for item in summary_items
            ),
        }
        if not all(automatic_checks.values()):
            raise RuntimeError(f"summary automatic checks failed: {automatic_checks}")
        summary = {
            "version": "SignDuplex_V1.8_Visual_Check",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_data_root": str(data_root),
            "output_directory": str(output),
            "seed": seed,
            "labels": list(LABELS),
            "samples_per_class": samples_per_class,
            "total_samples": len(summary_items),
            "source_version_by_label": SOURCE_VERSION_BY_LABEL,
            "selection": selection_metadata,
            "per_class": per_class,
            "automatic_media_validation": {
                "status": "PASS",
                "checks": automatic_checks,
                "note": "PASS covers deterministic selection and media/frame/duration checks only.",
            },
            "manual_review_completed": False,
            "manual_review_status": "PENDING_MANUAL_REVIEW",
            "manual_review_note": (
                "Media export success is not a human visual-review verdict. "
                "A reviewer must inspect every input/full pair before changing this status."
            ),
            "samples": summary_items,
        }
        write_json(staging / "samples_summary.json", summary)
        if output.exists():
            raise FileExistsError(f"output appeared during export; refusing to overwrite: {output}")
        staging.rename(output)
        return summary
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> int:
    args = parse_args()
    summary = export_package(
        args.data_root,
        args.output,
        args.seed,
        args.samples_per_class,
    )
    print(
        json.dumps(
            {
                "output": summary["output_directory"],
                "total_samples": summary["total_samples"],
                "automatic_media_validation": summary["automatic_media_validation"]["status"],
                "manual_review_completed": summary["manual_review_completed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
