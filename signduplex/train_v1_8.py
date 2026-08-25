#!/usr/bin/env python3
"""Train SignDuplex V1.8 with a shared backbone and state-routed 4/2 heads.

This is deliberately a thin V1.8 entry point over the validated V1.6 trainer.
The imported module supplies the RGB/Pose preprocessing, MiniCPM/Uni-Sign/Qwen
backbone, LoRA injection, optimizer construction, evaluation, and checkpoint
serialization.  This file changes only the V1.8 data contract, the LISTENING
head size, the fixed eight-epoch lifecycle, and the required output/audits.

Important import detail: the V1.6 module configures a log handler while it is
imported.  We close that handler immediately and do not call its ``main``.  Its
old output directory already exists and no V1.6 artifact is overwritten.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

# GPU isolation must happen before the imported V1.6 module imports torch.
PHYSICAL_GPU = os.environ.get("SIGNDUPLEX_PHYSICAL_GPU", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = PHYSICAL_GPU
os.environ["V16_PHYSICAL_GPU"] = PHYSICAL_GPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
V16_SCRIPT = HERE / "factorized_training.py"


def _load_v16_module():
    spec = importlib.util.spec_from_file_location("sign_duplex_v16_reuse", V16_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import V1.6 training base: {V16_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v16 = _load_v16_module()

# The imported module and its historical base both set CUDA visibility.  Assert
# the requested physical card again before the first CUDA allocation.
os.environ["CUDA_VISIBLE_DEVICES"] = PHYSICAL_GPU
v16.SELECTED_PHYSICAL_GPU = PHYSICAL_GPU
v16.base.DEVICE = v16.DEVICE

import torch  # noqa: E402  (CUDA visibility must be set before this import)


VERSION = "SignDuplex_V1_8_IDLE_4x2_GPU3"
DATA_ROOT = Path(
    os.environ.get(
        "SIGNDUPLEX_DATA_DIR",
        REPO_ROOT / "data" / "Sign_Aware_Raw_Screen_v1_8",
    )
).expanduser()
DATA_DIR = DATA_ROOT
FORMAL_OUT = Path(
    os.environ.get("SIGNDUPLEX_OUTPUT_DIR", REPO_ROOT / "outputs" / "train_v1_8")
).expanduser()
SMOKE_OUT = Path(str(FORMAL_OUT) + "_smoke")
OUT_DIR = SMOKE_OUT if int(v16.base.ARGS.smoke) > 0 else FORMAL_OUT
BACKING_ENV_NAME = "V18_OUTPUT_BACKING_DIR"

TURN_LABELS = ("IDLE", "INCOMPLETE", "BACKCHANNEL", "COMPLETE")
BARGE_LABELS = ("BARGE", "NO_BARGE")
TASK_LABELS = {"turn_state": TURN_LABELS, "barge_in": BARGE_LABELS}
LABEL_TO_INDEX = {
    task: {label: index for index, label in enumerate(labels)}
    for task, labels in TASK_LABELS.items()
}
NUM_EPOCHS = int(v16.base.ARGS.epochs)
REQUIRED_EPOCHS = 8
MIN_FORMAL_FREE_BYTES = 3 * 1024**3
OVERFIT_MACRO_F1_DROP = 0.005

# Patch only the V1.6 module globals that its reusable functions resolve at
# runtime.  The model/preprocessing/hyperparameters remain unchanged.
v16.DATA_ROOT = DATA_ROOT
v16.DATA_DIR = DATA_DIR
v16.FORMAL_OUT = FORMAL_OUT
v16.OUT_DIR = OUT_DIR
v16.VERSION = VERSION
v16.TURN_LABELS = TURN_LABELS
v16.BARGE_LABELS = BARGE_LABELS
v16.TASK_LABELS = TASK_LABELS
v16.LABEL_TO_INDEX = LABEL_TO_INDEX
v16.NUM_EPOCHS = NUM_EPOCHS
v16.base.DATA_DIR = str(DATA_DIR)
v16.base.OUT_DIR = str(OUT_DIR)

v16.BANNED_MODEL_FIELDS = set(v16.BANNED_MODEL_FIELDS) | {
    "active_start_frame",
    "active_start_sec",
    "active_start_time",
    "active_start_time_sec",
    "active_end_frame",
    "active_end_sec",
    "idle_pre_active_ratio",
    "idle_recent_motion",
    "idle_recent_hand_motion",
    "idle_recent_wrist_motion",
    "idle_recent_internal_motion",
    "onset_safety_margin",
    "future_active_duration_for_offline_verification",
    "source_version",
}


def _detach_import_log_handlers() -> None:
    """Undo logging side effects from importing the V1.6 implementation."""
    for handler in list(logging.root.handlers):
        handler.close()
        logging.root.removeHandler(handler)


_detach_import_log_handlers()
log = logging.getLogger("sign_duplex_v1_8_train")
v16.log = log
v16.base.log = log


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def write_jsonl_atomic(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sequence_v18(
    tokenizer, n_rgb: int, n_pose: int, task: str, system_state: str
):
    """Build the same V1.6 multimodal prompt with the V1.8 label inventory."""

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    tokens = v16.TOKENS
    im_start = [tokens["im_start"]]
    im_end = [tokens["im_end"]]
    newline = encode("\n")
    placeholder = tokens["placeholder"]
    ids = (
        im_start
        + encode("system")
        + newline
        + encode(v16.SYSTEM_TEXT)
        + im_end
        + newline
    )
    ids += im_start + encode("user") + newline
    if v16.ROUTED_TASK_BY_STATE.get(system_state) != task:
        raise RuntimeError(f"system-state router mismatch: {system_state}/{task}")
    ids += encode(f"已知系统状态：{system_state}。") + newline
    if task == "turn_state":
        ids += encode(
            "路由：Turn-State Head，类别为 IDLE / INCOMPLETE / BACKCHANNEL / COMPLETE。"
        ) + newline
    elif task == "barge_in":
        ids += encode("路由：Barge-In Head，类别为 BARGE / NO_BARGE。") + newline
    else:
        raise KeyError(task)
    ids += encode("手语视频：")
    rgb_slots: list[int] = []
    for _ in range(n_rgb):
        ids += [tokens["img_start"]]
        rgb_slots.extend(range(len(ids), len(ids) + 64))
        ids += [placeholder] * 64 + [tokens["img_end"]]
    ids += newline + encode("手语骨架：")
    pose_slots = list(range(len(ids), len(ids) + n_pose))
    ids += [placeholder] * n_pose + newline
    ids += encode("输出该任务的唯一类别：") + im_end + newline
    ids += im_start + encode("assistant") + newline
    return ids, rgb_slots, pose_slots, len(ids) - 1


# V1.6 forward/evaluation helpers resolve this function from their module.
v16.build_sequence = build_sequence_v18


def configure_logging() -> None:
    for handler in list(logging.root.handlers):
        handler.close()
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(OUT_DIR / "logs" / "train.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def ensure_new_output() -> dict[str, Any]:
    """Create a new logical output, optionally backed by another filesystem.

    ``OUT_DIR`` always remains the user-requested logical path.  When
    V18_OUTPUT_BACKING_DIR is set, that absolute, not-yet-existing directory is
    created first and the logical path becomes a directory symlink to it.  We
    use lexists for both paths so a broken symlink is never mistaken for a safe
    empty destination.
    """
    logical = OUT_DIR
    if os.path.lexists(logical):
        raise RuntimeError(f"refusing to overwrite existing V1.8 run: {logical}")
    required = MIN_FORMAL_FREE_BYTES if int(v16.base.ARGS.smoke) == 0 else 512 * 1024**2
    requested_backing = os.environ.get(BACKING_ENV_NAME)
    backing: Path | None = None
    target_parent = logical.parent
    via_symlink = False
    if requested_backing:
        backing = Path(requested_backing)
        if not backing.is_absolute():
            raise RuntimeError(f"{BACKING_ENV_NAME} must be an absolute path")
        if os.path.lexists(backing):
            raise RuntimeError(f"backing output path already exists: {backing}")
        target_parent = backing.parent
        if not target_parent.is_dir():
            raise RuntimeError(f"backing parent does not exist: {target_parent}")
        if target_parent.stat().st_dev == logical.parent.stat().st_dev:
            raise RuntimeError(
                f"{BACKING_ENV_NAME} must be on a filesystem independent of {logical.parent}"
            )
        via_symlink = True
    free_bytes = shutil.disk_usage(target_parent).free
    if free_bytes < required:
        raise RuntimeError(
            "insufficient free space for V1.8 output: "
            f"parent={target_parent} free={free_bytes} required={required}"
        )
    if backing is not None:
        backing.mkdir(parents=False, exist_ok=False)
        # Never remove an existing or newly created path on failure.  A failed
        # symlink attempt is reported and requires an explicit new destination.
        logical.symlink_to(backing, target_is_directory=True)
        if not logical.is_dir() or logical.resolve() != backing.resolve():
            raise RuntimeError(f"logical/backing output verification failed: {logical} -> {backing}")
    else:
        logical.mkdir(parents=False, exist_ok=False)
    for relative in (
        "checkpoints",
        "checkpoints/best",
        "checkpoints/last",
        "logs",
        "metrics",
        "confusion_matrix",
        "predictions",
    ):
        (logical / relative).mkdir(
            parents=True, exist_ok=False if relative == "checkpoints" else True
        )
    return {
        "logical_output_dir": str(logical),
        "backing_output_dir": str(backing) if backing is not None else None,
        "via_directory_symlink": via_symlink,
        "space_checked_parent": str(target_parent),
        "free_bytes_before_create": free_bytes,
        "minimum_required_free_bytes": required,
    }


def load_source_validation() -> dict[str, Any]:
    required_files = (
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "dataset_stats.json",
        "idle_filter_stats.json",
        "validation_report.json",
        "BUILD_CONFIG.json",
    )
    missing = [name for name in required_files if not (DATA_ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"V1.8 data is incomplete; missing {missing}")
    report = json.loads((DATA_ROOT / "validation_report.json").read_text(encoding="utf-8"))
    passed = report.get("status") == "PASS" or report.get("all_passed") is True
    if not passed:
        raise RuntimeError("V1.8 validation_report.json is not PASS")
    return report


def copy_source_audits() -> None:
    for name in (
        "dataset_stats.json",
        "idle_filter_stats.json",
        "validation_report.json",
        "BUILD_CONFIG.json",
    ):
        shutil.copy2(DATA_ROOT / name, OUT_DIR / "metrics" / f"source_{name}")


def static_model_input_audit() -> dict[str, Any]:
    source = inspect.getsource(build_sequence_v18) + inspect.getsource(v16.forward_step)
    mentions = {
        field: source.count(field) for field in sorted(v16.BANNED_MODEL_FIELDS)
    }
    failures = {field: count for field, count in mentions.items() if count}
    if failures:
        raise RuntimeError(f"forbidden audit metadata appears in model-input code: {failures}")
    return {
        "status": "PASS",
        "model_input_fields": ["RGB prefix", "Pose prefix", "system_state_before"],
        "router": "LISTENING/0 -> turn_state(4-way); SPEAKING/1 -> barge_in(2-way)",
        "turn_state_labels": list(TURN_LABELS),
        "barge_in_labels": list(BARGE_LABELS),
        "forbidden_field_mentions_in_model_input_code": mentions,
        "unified_six_class_softmax": False,
    }


def validate_rows(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    expected_labels = set(TURN_LABELS + BARGE_LABELS)
    seen_ids: dict[str, str] = {}
    seen_videos: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, dict[str, int]] = {}
    for split, rows in rows_by_split.items():
        counter = Counter(row["label"] for row in rows)
        if set(counter) != expected_labels:
            raise RuntimeError(
                f"{split} label inventory mismatch: {sorted(counter)} != {sorted(expected_labels)}"
            )
        counts[split] = {label: counter[label] for label in TURN_LABELS + BARGE_LABELS}
        for row in rows:
            sample_id = row["sample_id"]
            prior = seen_ids.setdefault(sample_id, split)
            if prior != split:
                raise RuntimeError(f"sample_id crosses split: {sample_id} {prior}/{split}")
            seen_videos[row["video_id"]].add(split)
            if row["label"] == "IDLE" and row["system_state_before"] != "LISTENING":
                raise RuntimeError(f"IDLE is not LISTENING: {sample_id}")
    crossing_videos = {
        video_id: sorted(splits)
        for video_id, splits in seen_videos.items()
        if len(splits) > 1
    }
    if crossing_videos:
        first = next(iter(crossing_videos.items()))
        raise RuntimeError(f"video_id crosses split: {first}")
    return {
        "status": "PASS",
        "counts": counts,
        "sample_id_cross_split": 0,
        "video_id_cross_split": 0,
        "semantic_id_scope": "not re-split; official inherited split is authoritative",
    }


def physical_gpu3_preflight() -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != PHYSICAL_GPU:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 3")
    gpu_text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpus = []
    for line in gpu_text.strip().splitlines():
        index, uuid, pci, name, total, used, utilization = [
            part.strip() for part in line.split(",", 6)
        ]
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "pci_bus_id": pci,
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "utilization_gpu_percent": int(utilization),
            }
        )
    selected = next((item for item in gpus if item["index"] == 3), None)
    if selected is None:
        raise RuntimeError("physical GPU3 does not exist")
    process_text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    other_processes = []
    for line in process_text.splitlines() if process_text else []:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and parts[0] == selected["uuid"] and parts[1] != str(os.getpid()):
            other_processes.append(
                {"gpu_uuid": parts[0], "pid": int(parts[1]), "used_memory_mib": int(parts[2])}
            )
    if other_processes:
        raise RuntimeError(f"GPU3 is not exclusive: {other_processes}")
    return {
        "CUDA_VISIBLE_DEVICES": PHYSICAL_GPU,
        "selected_physical_gpu": selected,
        "other_compute_processes": other_processes,
        "all_passed": True,
    }


def label_mapping() -> dict[str, Any]:
    return {
        "known_state_router": {
            "LISTENING": "turn_state",
            "SPEAKING": "barge_in",
        },
        "turn_state": {
            "labels": list(TURN_LABELS),
            "label_to_index": LABEL_TO_INDEX["turn_state"],
            "output_dim": 4,
        },
        "barge_in": {
            "labels": list(BARGE_LABELS),
            "label_to_index": LABEL_TO_INDEX["barge_in"],
            "output_dim": 2,
            "roc_auc_positive_label": "BARGE",
        },
        "unified_six_class_softmax": False,
    }


def select_all_class_smoke_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for label in TURN_LABELS + BARGE_LABELS:
        selected.append(next(row for row in rows if row["label"] == label))
    return selected


def save_confusion_artifacts(
    metrics: dict[str, Any], stem: str, *, title_prefix: str
) -> None:
    payload = {
        task: {
            "labels": metrics[task]["confusion_matrix_labels"],
            "matrix": metrics[task]["confusion_matrix"],
        }
        for task in TASK_LABELS
    }
    write_json_atomic(OUT_DIR / "confusion_matrix" / f"{stem}.json", payload)
    for task in TASK_LABELS:
        v16.save_confusion_matrix_png(
            metrics[task],
            OUT_DIR / "confusion_matrix" / f"{stem}_{task}.png",
            f"{title_prefix} {task}",
        )


def replace_with_hardlink_or_copy(source: Path, destination: Path) -> str:
    """Atomically update best/last while avoiding duplicate 176 MB files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    if os.path.lexists(temp):
        raise RuntimeError(f"refusing to remove stale checkpoint temp: {temp}")
    method = "hardlink"
    try:
        os.link(source, temp)
    except OSError:
        method = "copy"
        shutil.copy2(source, temp)
    os.replace(temp, destination)
    return method


def save_epoch_checkpoint(
    epoch: int,
    projector,
    adapter,
    llm,
    pose_encoder,
    heads,
    train_loss: float,
    dev_metrics: dict[str, Any],
    config: dict[str, Any],
    *,
    is_best: bool,
) -> dict[str, Any]:
    epoch_dir = OUT_DIR / "checkpoints" / f"epoch_{epoch}"
    epoch_dir.mkdir(parents=False, exist_ok=False)
    checkpoint_path = epoch_dir / "ckpt.pt"
    v16.save_checkpoint(
        checkpoint_path,
        epoch,
        projector,
        adapter,
        llm,
        pose_encoder,
        heads,
        train_loss,
        dev_metrics,
        config,
    )
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        raise RuntimeError(f"checkpoint save verification failed: {checkpoint_path}")
    last_method = replace_with_hardlink_or_copy(
        checkpoint_path, OUT_DIR / "checkpoints" / "last" / "ckpt.pt"
    )
    best_method = None
    if is_best:
        best_method = replace_with_hardlink_or_copy(
            checkpoint_path, OUT_DIR / "checkpoints" / "best" / "ckpt.pt"
        )
    metadata = {
        "epoch": epoch,
        "checkpoint": str(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "sha256": sha256_file(checkpoint_path),
        "last_alias_method": last_method,
        "best_alias_method": best_method,
        "is_best": is_best,
    }
    write_json_atomic(epoch_dir / "metadata.json", metadata)
    write_json_atomic(
        OUT_DIR / "checkpoints" / "last" / "metadata.json", metadata
    )
    if is_best:
        write_json_atomic(
            OUT_DIR / "checkpoints" / "best" / "metadata.json", metadata
        )
    return metadata


def idle_complete_confusion(metrics: dict[str, Any]) -> dict[str, Any]:
    turn = metrics["turn_state"]
    labels = turn["confusion_matrix_labels"]
    matrix = turn["confusion_matrix"]
    idle = labels.index("IDLE")
    complete = labels.index("COMPLETE")
    idle_support = sum(matrix[idle])
    complete_support = sum(matrix[complete])
    return {
        "IDLE_true_pred_COMPLETE": matrix[idle][complete],
        "COMPLETE_true_pred_IDLE": matrix[complete][idle],
        "IDLE_support": idle_support,
        "COMPLETE_support": complete_support,
        "IDLE_to_COMPLETE_rate": round(matrix[idle][complete] / idle_support, 6)
        if idle_support
        else None,
        "COMPLETE_to_IDLE_rate": round(matrix[complete][idle] / complete_support, 6)
        if complete_support
        else None,
    }


def temporal_examples(records: list[dict[str, Any]], limit: int = 10) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["task"] == "turn_state":
            grouped[record["video_id"]].append(record)
    candidates = []
    for video_id, items in grouped.items():
        items.sort(key=lambda item: (float(item["decision_time"]), item["true"]))
        true_labels = {item["true"] for item in items}
        if "IDLE" in true_labels and len(true_labels) >= 2:
            candidates.append((video_id, items))
    candidates.sort(key=lambda pair: (-len(pair[1]), pair[0]))
    examples = []
    for video_id, items in candidates[:limit]:
        examples.append(
            {
                "video_id": video_id,
                "timeline": [
                    {
                        "decision_time": item["decision_time"],
                        "decision_frame": item["decision_frame"],
                        "true": item["true"],
                        "pred": item["pred"],
                        "confidence": item["confidence"],
                    }
                    for item in items
                ],
            }
        )
    return {
        "scope": "available labeled V1.8 test prefixes only; no label rule modification",
        "candidate_video_count": len(candidates),
        "examples": examples,
    }


def detect_overfitting(history: list[dict[str, Any]]) -> dict[str, Any]:
    first = None
    best_so_far = -1.0
    minimum_train_loss = float("inf")
    for item in history:
        score = float(item["dev"]["combined_macro_f1"])
        train_loss = float(item["train_loss"])
        if (
            best_so_far >= 0
            and train_loss < minimum_train_loss
            and score <= best_so_far - OVERFIT_MACRO_F1_DROP
        ):
            first = int(item["epoch"])
            break
        best_so_far = max(best_so_far, score)
        minimum_train_loss = min(minimum_train_loss, train_loss)
    return {
        "detected": first is not None,
        "first_epoch": first,
        "rule": (
            "train_loss makes a new low while dev combined per-class macro-F1 is "
            f"at least {OVERFIT_MACRO_F1_DROP:.3f} below its prior best"
        ),
        "all_checkpoints_retained": True,
    }


def source_count_table(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        split: {
            label: sum(row["label"] == label for row in rows)
            for label in TURN_LABELS + BARGE_LABELS
        }
        for split, rows in rows_by_split.items()
    }


def make_train_config(
    rows_by_split: dict[str, list[dict[str, Any]]],
    storage_audit: dict[str, Any],
    preflight_gpu: dict[str, Any],
    runtime_gpu: dict[str, Any],
    parameter_audit: dict[str, Any],
    class_weight_report: dict[str, Any],
    model_input_audit: dict[str, Any],
    warm_start: dict[str, Any],
    source_hashes: dict[str, str],
    row_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "data_dir": str(DATA_DIR),
        "output_dir": str(OUT_DIR),
        "storage": storage_audit,
        "gpu_preflight": preflight_gpu,
        "gpu_runtime_mapping": runtime_gpu,
        "factorized_heads": label_mapping(),
        "inputs": ["RGB causal prefix", "Pose causal prefix", "known system_state_before"],
        "model_input_audit": model_input_audit,
        "excluded_from_model_input": sorted(v16.BANNED_MODEL_FIELDS),
        "audio_or_tts": False,
        "rgb_frames": int(v16.base.RGB_FRAMES),
        "rgb_tokens_per_frame": 64,
        "pose_max_frames": int(v16.base.MAX_POSE_LEN),
        "pose_temporal_downsample": 2,
        "batch_size": v16.BATCH_SIZE,
        "gradient_accumulation_steps": v16.GRAD_ACCUM,
        "effective_batch_size": v16.BATCH_SIZE * v16.GRAD_ACCUM,
        "epochs": NUM_EPOCHS,
        "bf16": True,
        "optimizer": "AdamW",
        "learning_rates": {
            "sign_projector": v16.LR_PROJECTOR,
            "temporal_adapter": v16.LR_ADAPTER,
            "classification_heads": v16.LR_CLASSIFICATION_HEAD,
            "qwen_lora": v16.LR_LORA,
            "unisign_unfrozen": v16.LR_UNISIGN,
        },
        "weight_decay": v16.WEIGHT_DECAY,
        "warmup_ratio": v16.WARMUP_RATIO,
        "scheduler": "cosine",
        "max_grad_norm": v16.MAX_GRAD_NORM,
        "seed": v16.SEED,
        "early_stopping": False,
        "class_weights": class_weight_report,
        "best_checkpoint_selection": (
            "dev combined Macro-F1: equal-weight mean of all six per-class F1 values"
        ),
        "test_policy": "evaluate once after training with best checkpoint only",
        "n_train": len(rows_by_split["train"]),
        "n_dev": len(rows_by_split["dev"]),
        "n_test": len(rows_by_split["test"]),
        "class_counts": source_count_table(rows_by_split),
        "row_audit": row_audit,
        "parameter_audit": parameter_audit,
        "warm_start": warm_start,
        "source_data_sha256_before": source_hashes,
        "dataloader": {
            "num_workers": int(v16.base.ARGS.workers),
            "pin_memory": True,
            "persistent_workers": int(v16.base.ARGS.workers) > 0,
            "prefetch_factor": 4 if int(v16.base.ARGS.workers) > 0 else None,
        },
        "reuse": {
            "v1_6_entry": str(V16_SCRIPT),
            "v1_6_imported_main_called": False,
            "v1_6_import_logging_handler_closed_immediately": True,
            "backbone_and_preprocessing_unchanged": True,
        },
    }


def final_report_text(summary: dict[str, Any]) -> str:
    counts = summary["class_counts"]
    dev = summary["best_dev_metrics"]
    test = summary["test_metrics"]
    idle_stats = summary["idle_filter_stats"]
    rejection: dict[str, Any] = {}
    for key in (
        "feature_rejection_reasons",
        "candidate_rejection_reasons",
        "rejected_by_reason",
        "filter_reasons",
    ):
        value = idle_stats.get(key)
        if isinstance(value, dict):
            for reason, count in value.items():
                rejection[f"{key}.{reason}"] = count
    lines = [
        "# SignDuplex V1.8 GPU3 Final Report",
        "",
        f"1. V1.8 数据目录：`{DATA_ROOT}`",
        "2. 六类 train/dev/test 数量：",
        "",
        "| split | IDLE | INCOMPLETE | BACKCHANNEL | COMPLETE | BARGE | NO_BARGE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "dev", "test"):
        item = counts[split]
        lines.append(
            f"| {split} | {item['IDLE']} | {item['INCOMPLETE']} | "
            f"{item['BACKCHANNEL']} | {item['COMPLETE']} | {item['BARGE']} | "
            f"{item['NO_BARGE']} |"
        )
    lines += [
        "",
        "3. IDLE 实际构造数量："
        f"train/dev/test = {counts['train']['IDLE']}/{counts['dev']['IDLE']}/{counts['test']['IDLE']}",
        f"4. IDLE 被过滤的主要原因：`{json.dumps(rejection, ensure_ascii=False)}`",
        "5. 六类各 10 条视觉核验目录："
        f"`{os.environ.get('SIGNDUPLEX_VISUAL_CHECK_DIR', REPO_ROOT / 'outputs' / 'visual_check')}`",
        f"6. 自动校验：{summary['source_validation_status']}",
        f"7. GPU3 训练完成：{summary['training_complete']}",
        f"8. 8 个 epoch checkpoint 全部保存：{summary['all_epoch_checkpoints_saved']}",
        f"9. best epoch：{summary['best_epoch']}",
        f"10. best dev combined Macro-F1：{dev['combined_macro_f1']:.6f}",
        f"11. best checkpoint test combined Macro-F1：{test['combined_macro_f1']:.6f}",
        "12. IDLE/COMPLETE confusion："
        f"`{json.dumps(summary['idle_complete_confusion'], ensure_ascii=False)}`",
        f"13. 过拟合：`{json.dumps(summary['overfitting'], ensure_ascii=False)}`",
        f"14. 训练输出目录：`{OUT_DIR}`",
        "15. Demo best checkpoint："
        f"`{OUT_DIR / 'checkpoints' / 'best' / 'ckpt.pt'}`",
        "",
        "测试集仅在八个 epoch 全部结束并重新载入 best checkpoint 后评估一次。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    started = time.time()
    if int(v16.base.ARGS.smoke) == 0 and NUM_EPOCHS != REQUIRED_EPOCHS:
        raise RuntimeError(f"formal V1.8 run requires exactly {REQUIRED_EPOCHS} epochs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != PHYSICAL_GPU:
        raise RuntimeError("V1.8 is restricted to physical GPU3")
    load_source_validation()
    preflight_gpu = physical_gpu3_preflight()
    storage_audit = ensure_new_output()
    configure_logging()
    copy_source_audits()
    write_json_atomic(OUT_DIR / "label_mapping.json", label_mapping())
    v16.set_seed(v16.SEED)

    model_input_audit = static_model_input_audit()
    source_hashes_before = {
        split: sha256_file(DATA_DIR / f"{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    rows_by_split = {
        split: v16.load_jsonl(split) for split in ("train", "dev", "test")
    }
    row_audit = validate_rows(rows_by_split)
    for split, rows in rows_by_split.items():
        task_counts = dict(Counter(row["task"] for row in rows))
        label_counts = dict(Counter(row["label"] for row in rows))
        log.info(
            f"{split} n={len(rows)} tasks={task_counts} labels={label_counts}"
        )
    v16.log_random_samples(rows_by_split)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    from transformers import AutoModel, AutoTokenizer

    log.info("loading unchanged MiniCPM-o-4.5 RGB tower + LLM in bf16")
    tokenizer = AutoTokenizer.from_pretrained(v16.base.MINICPM_DIR, trust_remote_code=True)
    minicpm = AutoModel.from_pretrained(
        v16.base.MINICPM_DIR,
        trust_remote_code=True,
        torch_dtype=v16.DTYPE,
        device_map={"": v16.DEVICE},
        attn_implementation="eager",
        init_audio=False,
        init_tts=False,
    )
    minicpm.eval()
    for parameter in minicpm.parameters():
        parameter.requires_grad = False
    llm = minicpm.llm
    if llm.config.num_hidden_layers != 36:
        raise RuntimeError("unexpected Qwen architecture")
    hidden_size = int(llm.config.hidden_size)
    v16.setup_placeholders(tokenizer)
    v16.base.inject_lora(
        llm,
        v16.LORA_LAYERS,
        v16.LORA_TARGETS,
        v16.LORA_R,
        v16.LORA_ALPHA,
        v16.LORA_DROPOUT,
    )
    v16.base.set_lora_mode(llm, False)
    pose_encoder, _ = v16.base.build_pose_encoder()
    projector = v16.base.SignProjector(768, hidden_size).to(v16.DEVICE, dtype=v16.TDTYPE)
    adapter = v16.base.TemporalAdapter().to(v16.DEVICE, dtype=v16.TDTYPE)
    for module in (projector, adapter):
        for parameter in module.parameters():
            parameter.requires_grad = True
    heads = v16.FactorizedHeads(hidden_size).to(v16.DEVICE, dtype=v16.TDTYPE)
    for parameter in heads.parameters():
        parameter.requires_grad = True
    warm_start_checkpoint = v16.base.warm_start_from_exp25(
        v16.base.EXP25_CKPT, projector, adapter, llm, pose_encoder
    )
    warm_start = {
        "path": v16.base.EXP25_CKPT,
        "sha256": sha256_file(Path(v16.base.EXP25_CKPT)),
        "source_epoch": warm_start_checkpoint.get("epoch"),
        "adapter_variant": warm_start_checkpoint.get("adapter_variant"),
        "classification_heads": "new deterministic 4-way/2-way initialization",
    }

    runtime_gpu = v16.gpu_audit()
    parameter_audit = v16.trainable_audit(
        projector, adapter, llm, pose_encoder, heads, minicpm
    )
    if parameter_audit["turn_head_output_dim"] != 4 or parameter_audit["barge_head_output_dim"] != 2:
        raise RuntimeError(f"factorized head audit failed: {parameter_audit}")
    class_weights, class_weight_report = v16.class_weights_from_train(
        rows_by_split["train"]
    )
    config = make_train_config(
        rows_by_split,
        storage_audit,
        preflight_gpu,
        runtime_gpu,
        parameter_audit,
        class_weight_report,
        model_input_audit,
        warm_start,
        source_hashes_before,
        row_audit,
    )

    if int(v16.base.ARGS.smoke) > 0:
        smoke_rows = select_all_class_smoke_rows(rows_by_split["train"])
        smoke_report = v16.run_forward_backward_smoke(
            smoke_rows,
            minicpm,
            llm,
            pose_encoder,
            adapter,
            projector,
            heads,
            tokenizer,
            class_weights,
        )
        smoke_report["requested_smoke_argument"] = int(v16.base.ARGS.smoke)
        smoke_report["all_six_classes_covered"] = True
        smoke_report["head_dimensions"] = {"turn_state": 4, "barge_in": 2}
        smoke_report["gpu"] = runtime_gpu
        smoke_report["model_input_audit"] = model_input_audit
        write_json_atomic(OUT_DIR / "train_config.json", config)
        write_json_atomic(OUT_DIR / "metrics" / "smoke_report.json", smoke_report)
        log.info("V1.8 six-class forward/backward smoke PASS")
        return

    optimizer, all_trainable, optimizer_groups = v16.build_optimizer(
        projector, adapter, llm, pose_encoder, heads
    )
    initial_optimizer_groups = {
        group["name"]: {
            "initial_lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
        }
        for group in optimizer_groups
    }
    steps_per_epoch = math.ceil(len(rows_by_split["train"]) / v16.GRAD_ACCUM)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = max(1, int(total_steps * v16.WARMUP_RATIO))
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    config["optimizer_groups"] = initial_optimizer_groups
    config["steps_per_epoch"] = steps_per_epoch
    config["total_steps"] = total_steps
    config["warmup_steps"] = warmup_steps
    write_json_atomic(OUT_DIR / "train_config.json", config)

    metrics_path = OUT_DIR / "metrics" / "metrics.jsonl"
    history: list[dict[str, Any]] = []
    checkpoint_manifest: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = -1
    dev_loader = v16.make_loader(rows_by_split["dev"], "dev")
    train_loader = v16.make_loader(rows_by_split["train"], "train", shuffle=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_started = time.time()
        torch.cuda.reset_peak_memory_stats(v16.DEVICE)
        log.info("%s", "=" * 96)
        log.info("Epoch %d/%d", epoch, NUM_EPOCHS)
        v16.set_train_mode(projector, adapter, llm, heads, True)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        used = 0
        skipped = 0
        accumulated = 0
        last_grad_norm = 0.0
        for batch in train_loader:
            if not batch["ok"]:
                skipped += 1
                log.error("train data failure %s: %s", batch["sample_id"], batch["err"])
                continue
            loss, _, _, _, _, _ = v16.forward_step(
                batch,
                minicpm,
                llm,
                pose_encoder,
                adapter,
                projector,
                heads,
                tokenizer,
                class_weights,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite train loss: {batch['sample_id']}")
            (loss / v16.GRAD_ACCUM).backward()
            total_loss += float(loss.item())
            used += 1
            accumulated += 1
            if accumulated % v16.GRAD_ACCUM == 0:
                last_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(all_trainable, v16.MAX_GRAD_NORM).item()
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if used % 1000 == 0:
                elapsed = time.time() - epoch_started
                log.info(
                    "E%d %d/%d loss=%.5f grad_norm=%.4f %.2f it/s mem=%.1fGiB",
                    epoch,
                    used,
                    len(rows_by_split["train"]),
                    total_loss / used,
                    last_grad_norm,
                    used / elapsed,
                    torch.cuda.max_memory_allocated() / 2**30,
                )
        if skipped:
            raise RuntimeError(f"epoch {epoch} had {skipped} unreadable training samples")
        if accumulated % v16.GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_(all_trainable, v16.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if used != len(rows_by_split["train"]):
            raise RuntimeError(f"epoch {epoch} train cardinality mismatch {used}")
        train_loss = total_loss / used

        dev_metrics, dev_predictions = v16.evaluate(
            dev_loader,
            minicpm,
            llm,
            pose_encoder,
            adapter,
            projector,
            heads,
            tokenizer,
            class_weights,
            f"epoch {epoch} dev",
        )
        if dev_metrics["n_skipped"]:
            raise RuntimeError(f"epoch {epoch} dev skipped {dev_metrics['n_skipped']} samples")
        if sum(dev_metrics[task]["n"] for task in TASK_LABELS) != len(rows_by_split["dev"]):
            raise RuntimeError(f"epoch {epoch} dev cardinality mismatch")
        log.info("%s", v16.format_metrics(dev_metrics, f"Epoch {epoch} DEV"))
        score = float(dev_metrics["combined_macro_f1"])
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_used": used,
            "train_skipped": skipped,
            "dev_loss": dev_metrics["loss"],
            "dev": dev_metrics,
            "idle_complete_confusion": idle_complete_confusion(dev_metrics),
            "elapsed_min": round((time.time() - epoch_started) / 60.0, 3),
            "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "is_best": is_best,
        }
        history.append(epoch_record)
        append_jsonl(metrics_path, epoch_record)
        write_json_atomic(OUT_DIR / "metrics" / "history.json", history)
        write_json_atomic(
            OUT_DIR / "metrics" / f"dev_epoch_{epoch}.json", dev_metrics
        )
        write_jsonl_atomic(
            OUT_DIR / "predictions" / f"dev_epoch_{epoch}.jsonl", dev_predictions
        )
        save_confusion_artifacts(
            dev_metrics, f"dev_epoch_{epoch}", title_prefix=f"DEV epoch {epoch}"
        )
        checkpoint_metadata = save_epoch_checkpoint(
            epoch,
            projector,
            adapter,
            llm,
            pose_encoder,
            heads,
            train_loss,
            dev_metrics,
            config,
            is_best=is_best,
        )
        checkpoint_manifest.append(checkpoint_metadata)
        write_json_atomic(
            OUT_DIR / "checkpoints" / "checkpoint_manifest.json", checkpoint_manifest
        )
        if is_best:
            write_json_atomic(OUT_DIR / "metrics" / "best_dev_metrics.json", dev_metrics)
            write_jsonl_atomic(
                OUT_DIR / "predictions" / "dev_best_predictions.jsonl", dev_predictions
            )
            log.info("new best epoch=%d dev combined Macro-F1=%.6f", epoch, score)
        write_json_atomic(
            OUT_DIR / "training_state.json",
            {
                "completed_epochs": epoch,
                "required_epochs": REQUIRED_EPOCHS,
                "best_epoch": best_epoch,
                "best_dev_macro_f1": best_score,
                "history": history,
                "test_evaluated": False,
            },
        )

    expected_epoch_dirs = [
        OUT_DIR / "checkpoints" / f"epoch_{epoch}" / "ckpt.pt"
        for epoch in range(1, REQUIRED_EPOCHS + 1)
    ]
    if len(history) != REQUIRED_EPOCHS or not all(path.is_file() for path in expected_epoch_dirs):
        raise RuntimeError("formal run did not retain all eight epoch checkpoints")
    best_path = OUT_DIR / "checkpoints" / "best" / "ckpt.pt"
    checkpoint = v16.load_checkpoint(
        best_path, projector, adapter, llm, pose_encoder, heads
    )
    if int(checkpoint["epoch"]) != best_epoch:
        raise RuntimeError("best checkpoint epoch mismatch")

    # This is the only test evaluation in the formal lifecycle.
    test_loader = v16.make_loader(rows_by_split["test"], "test")
    test_metrics, test_predictions = v16.evaluate(
        test_loader,
        minicpm,
        llm,
        pose_encoder,
        adapter,
        projector,
        heads,
        tokenizer,
        class_weights,
        "test best checkpoint",
    )
    if test_metrics["n_skipped"]:
        raise RuntimeError(f"test skipped {test_metrics['n_skipped']} samples")
    if sum(test_metrics[task]["n"] for task in TASK_LABELS) != len(rows_by_split["test"]):
        raise RuntimeError("test cardinality mismatch")
    log.info("%s", v16.format_metrics(test_metrics, f"TEST best epoch={best_epoch}"))
    write_json_atomic(OUT_DIR / "metrics" / "test_metrics.json", test_metrics)
    write_jsonl_atomic(
        OUT_DIR / "predictions" / "test_predictions.jsonl", test_predictions
    )
    save_confusion_artifacts(test_metrics, "test_best", title_prefix="TEST best")
    idle_complete = idle_complete_confusion(test_metrics)
    write_json_atomic(
        OUT_DIR / "metrics" / "idle_complete_confusion.json", idle_complete
    )
    pair_metrics = v16.same_video_pair_metrics(test_predictions)
    write_json_atomic(
        OUT_DIR / "metrics" / "same_video_pair_metrics.json", pair_metrics
    )
    length_shortcut = v16.length_shortcut_audit(
        rows_by_split["train"], test_predictions, tokenizer
    )
    write_json_atomic(
        OUT_DIR / "metrics" / "length_shortcut_audit.json", length_shortcut
    )
    errors = v16.random_error_cases(test_predictions)
    write_json_atomic(OUT_DIR / "predictions" / "random_error_cases.json", errors)
    timelines = temporal_examples(test_predictions)
    write_json_atomic(
        OUT_DIR / "predictions" / "test_temporal_examples.json", timelines
    )

    source_hashes_after = {
        split: sha256_file(DATA_DIR / f"{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("V1.8 source dataset changed during training")
    overfitting = detect_overfitting(history)
    idle_filter_stats = json.loads(
        (DATA_ROOT / "idle_filter_stats.json").read_text(encoding="utf-8")
    )
    summary = {
        "version": VERSION,
        "training_complete": True,
        "completed_epochs": REQUIRED_EPOCHS,
        "all_epoch_checkpoints_saved": True,
        "best_epoch": best_epoch,
        "best_dev_macro_f1": best_score,
        "best_dev_metrics": history[best_epoch - 1]["dev"],
        "test_metrics": test_metrics,
        "idle_complete_confusion": idle_complete,
        "overfitting": overfitting,
        "class_counts": source_count_table(rows_by_split),
        "idle_filter_stats": idle_filter_stats,
        "source_validation_status": "PASS",
        "source_data_sha256_after": source_hashes_after,
        "source_data_preserved": True,
        "gpu": runtime_gpu,
        "same_video_pair_metrics": pair_metrics,
        "length_shortcut_audit": length_shortcut,
        "temporal_examples": timelines,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(OUT_DIR / "checkpoints" / "last" / "ckpt.pt"),
        "output_dir": str(OUT_DIR),
        "total_hours": round((time.time() - started) / 3600.0, 4),
        "all_passed": True,
    }
    write_json_atomic(OUT_DIR / "run_summary.json", summary)
    (OUT_DIR / "final_report.md").write_text(
        final_report_text(summary), encoding="utf-8"
    )
    write_json_atomic(
        OUT_DIR / "training_state.json",
        {
            "completed_epochs": REQUIRED_EPOCHS,
            "required_epochs": REQUIRED_EPOCHS,
            "best_epoch": best_epoch,
            "best_dev_macro_f1": best_score,
            "history": history,
            "test_evaluated": True,
            "test_checkpoint_epoch": best_epoch,
            "all_passed": True,
        },
    )
    log.info(
        "V1.8 complete: best epoch=%d dev Macro-F1=%.6f test Macro-F1=%.6f",
        best_epoch,
        best_score,
        test_metrics["combined_macro_f1"],
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("V1.8 training failed")
        raise
