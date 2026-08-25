#!/usr/bin/env python3
"""Train Sign-Aware V1.6 for 10 epochs with two state-routed heads.

The RGB/Pose backbone and warm start are reused from the already validated
FullDuplex training entry.  Only the dataset adapter and final classifier are
task-specific here: Turn-State has 3 outputs, and Barge-in has 2 outputs.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

# This must be set before importing torch through the reused training module.
# The physical GPU is selected once from the preflight nvidia-smi audit and the
# base module's historical hard-code is immediately overridden after import.
SELECTED_PHYSICAL_GPU = os.environ.get("V16_PHYSICAL_GPU", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = SELECTED_PHYSICAL_GPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
BASE_SCRIPT = HERE / "training_base.py"


def _load_base_module():
    spec = importlib.util.spec_from_file_location("full_duplex_train_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import training base: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()
# The historical base hard-codes GPU 3 at module import.  No CUDA call has
# occurred yet; restore the selected physical GPU before the first allocation.
os.environ["CUDA_VISIBLE_DEVICES"] = SELECTED_PHYSICAL_GPU

import logging
import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_ROOT = Path(
    os.environ.get(
        "SIGNDUPLEX_V16_DATA_DIR",
        REPO_ROOT / "data" / "Sign_Aware_Raw_Screen_v1_6",
    )
).expanduser()
DATA_DIR = DATA_ROOT / "final_dataset"
FORMAL_OUT = Path(
    os.environ.get("SIGNDUPLEX_V16_OUTPUT_DIR", REPO_ROOT / "outputs" / "train_v1_6")
).expanduser()
OUT_DIR = (
    Path(str(FORMAL_OUT) + "_smoke") if int(base.ARGS.smoke) > 0 else FORMAL_OUT
)
VERSION = "Sign_Aware_Raw_Screen_v1_6_10Epoch_SingleGPU"
DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
TDTYPE = torch.float32
PDTYPE = torch.float32

TURN_LABELS = ("COMPLETE", "INCOMPLETE", "BACKCHANNEL")
BARGE_LABELS = ("BARGE", "NO_BARGE")
TASK_LABELS = {"turn_state": TURN_LABELS, "barge_in": BARGE_LABELS}
LABEL_TO_INDEX = {
    task: {label: index for index, label in enumerate(labels)}
    for task, labels in TASK_LABELS.items()
}

BATCH_SIZE = 1
GRAD_ACCUM = 8
NUM_EPOCHS = int(base.ARGS.epochs)
SEED = 42
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0
LR_PROJECTOR = 1e-4
LR_ADAPTER = 1e-4
LR_CLASSIFICATION_HEAD = 1e-4
LR_LORA = 1e-5
LR_UNISIGN = 1e-5
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_LAYERS = list(range(20, 36))
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

for handler in list(logging.root.handlers):
    handler.close()
    logging.root.removeHandler(handler)
OUT_DIR.mkdir(parents=True, exist_ok=True)
for subdir in ("predictions",):
    (OUT_DIR / subdir).mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(OUT_DIR / "train.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("sign_aware_v1_6_train")
base.log = log
base.DEVICE = DEVICE
base.DTYPE = DTYPE
base.TDTYPE = TDTYPE
base.PDTYPE = PDTYPE
base.DATA_DIR = str(DATA_DIR)
base.OUT_DIR = str(OUT_DIR)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json_atomic(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temp, path)


def write_jsonl_atomic(path: Path, values: Iterable[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
    os.replace(temp, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


ALLOWED_FIELDS = {
    "sample_id",
    "video_id",
    "split",
    "task",
    "label",
    "rgb_path",
    "pose_path",
    "window_end_frame",
    "video_frames",
    "candidate_frame",
    "candidate_time_sec",
    "decision_frame",
    "decision_time_sec",
    "prefix_end_frame_exclusive",
    "system_state_before",
    "system_speaking_before",
    "pair_id",
}

BANNED_MODEL_FIELDS = {
    "decision_time_sec",
    "normalized_progress",
    "video_duration_sec",
    "pair_id",
    "pair_gap_sec",
    "t_motion",
    "t_sign",
    "t_barge",
    "motion_at_decision",
    "recent_motion_mean",
    "sign_evidence_score",
    "pose_quality_score",
    "boundary_agreement_score",
    "motion_dynamic_range",
}
SOURCE_TASK_BY_STATE = {"LISTENING": "turn", "SPEAKING": "barge"}
ROUTED_TASK_BY_STATE = {"LISTENING": "turn_state", "SPEAKING": "barge_in"}
SPEAKING_BY_STATE = {"LISTENING": 0, "SPEAKING": 1}


def load_jsonl(split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = DATA_DIR / f"{split}.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            source = json.loads(line)
            row = {key: source[key] for key in ALLOWED_FIELDS if key in source}
            required = {
                "sample_id",
                "video_id",
                "split",
                "task",
                "label",
                "rgb_path",
                "pose_path",
                "window_end_frame",
                "video_frames",
                "system_state_before",
                "system_speaking_before",
            }
            missing = required - set(row)
            if missing:
                raise RuntimeError(f"{path}:{line_number} missing {sorted(missing)}")
            if row["split"] != split:
                raise RuntimeError(f"{path}:{line_number} split mismatch")
            state = row["system_state_before"]
            if state not in ROUTED_TASK_BY_STATE:
                raise RuntimeError(f"{path}:{line_number} invalid system state {state}")
            if int(row["system_speaking_before"]) != SPEAKING_BY_STATE[state]:
                raise RuntimeError(f"{path}:{line_number} state/speaking mismatch")
            if row["task"] != SOURCE_TASK_BY_STATE[state]:
                raise RuntimeError(f"{path}:{line_number} task/state router mismatch")
            routed_task = ROUTED_TASK_BY_STATE[state]
            if row["label"] not in TASK_LABELS[routed_task]:
                raise RuntimeError(
                    f"{path}:{line_number} invalid routed task/label {routed_task}/{row['label']}"
                )
            if source.get("quality_gate_pass") is not True:
                raise RuntimeError(f"{path}:{line_number} quality gate failure")
            if str(source.get("candidate_label", "")).startswith("AMBIGUOUS"):
                raise RuntimeError(f"{path}:{line_number} AMBIGUOUS entered training")
            if int(row["window_end_frame"]) != int(source["prefix_end_frame_exclusive"]):
                raise RuntimeError(f"{path}:{line_number} prefix contract mismatch")
            row["source_task"] = row.pop("task")
            row["task"] = routed_task
            row["system_speaking"] = int(row["system_speaking_before"])
            rows.append(row)
    return rows


class FactorizedDataset(base.TurnTakingDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        item["task"] = self.rows[index]["task"]
        item["system_state_before"] = self.rows[index]["system_state_before"]
        if item["ok"]:
            row = self.rows[index]
            item["audit"] = {
                "video_id": row["video_id"],
                "decision_frame": int(row.get("decision_frame", row.get("candidate_frame", row["window_end_frame"] - 1))),
                "decision_time": row.get("decision_time_sec", row.get("candidate_time_sec")),
                "system_state": row["system_state_before"],
                "pair_id": row.get("pair_id"),
                "prefix_frame_count": int(row["window_end_frame"]),
            }
        return item


def collate_one(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise RuntimeError("batch_size must be 1")
    return batch[0]


def make_loader(
    rows: list[dict[str, Any]], name: str, *, shuffle: bool = False, workers: int | None = None
) -> torch.utils.data.DataLoader:
    worker_count = int(base.ARGS.workers) if workers is None else workers
    return torch.utils.data.DataLoader(
        FactorizedDataset(rows, name),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=worker_count,
        collate_fn=collate_one,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4 if worker_count > 0 else None,
        persistent_workers=worker_count > 0,
    )


class FactorizedHeads(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.turn_state = nn.Linear(hidden_size, len(TURN_LABELS))
        self.barge_in = nn.Linear(hidden_size, len(BARGE_LABELS))

    def forward(self, hidden: torch.Tensor, system_state: str) -> torch.Tensor:
        if system_state == "LISTENING":
            return self.turn_state(hidden.float())
        if system_state == "SPEAKING":
            return self.barge_in(hidden.float())
        raise KeyError(system_state)


TOKENS: dict[str, int] = {}
SYSTEM_TEXT = (
    "你是手语实时对话的轮次决策器。只根据当前可见的手语视频、骨架前缀，"
    "以及已知系统状态完成对应分类任务。"
)


def setup_placeholders(tokenizer) -> int:
    base_vocab = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    added = tokenizer.get_added_vocab()
    if added:
        base_vocab = max(base_vocab, max(added.values()) + 1)
    for name, text in (
        ("im_start", "<|im_start|>"),
        ("im_end", "<|im_end|>"),
        ("img_start", "<image>"),
        ("img_end", "</image>"),
    ):
        token_id = tokenizer.convert_tokens_to_ids(text)
        if token_id is None or token_id == tokenizer.unk_token_id or token_id >= base_vocab:
            raise RuntimeError(f"required tokenizer token missing: {text}")
        TOKENS[name] = int(token_id)
    placeholder = tokenizer.convert_tokens_to_ids("<unk>")
    if placeholder is None or placeholder >= base_vocab:
        placeholder = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
    TOKENS["placeholder"] = int(placeholder)
    return int(base_vocab)


def build_sequence(tokenizer, n_rgb: int, n_pose: int, task: str, system_state: str):
    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    im_start = [TOKENS["im_start"]]
    im_end = [TOKENS["im_end"]]
    newline = encode("\n")
    placeholder = TOKENS["placeholder"]
    ids = im_start + encode("system") + newline + encode(SYSTEM_TEXT) + im_end + newline
    ids += im_start + encode("user") + newline
    if ROUTED_TASK_BY_STATE.get(system_state) != task:
        raise RuntimeError(f"system-state router mismatch: {system_state}/{task}")
    ids += encode(f"已知系统状态：{system_state}。") + newline
    if task == "turn_state":
        ids += encode("路由：Turn-State Head，类别为 COMPLETE / INCOMPLETE / BACKCHANNEL。") + newline
    elif task == "barge_in":
        ids += encode("路由：Barge-In Head，类别为 BARGE / NO_BARGE。") + newline
    else:
        raise KeyError(task)
    ids += encode("手语视频：")
    rgb_slots: list[int] = []
    for _ in range(n_rgb):
        ids += [TOKENS["img_start"]]
        rgb_slots.extend(range(len(ids), len(ids) + 64))
        ids += [placeholder] * 64 + [TOKENS["img_end"]]
    ids += newline + encode("手语骨架：")
    pose_slots = list(range(len(ids), len(ids) + n_pose))
    ids += [placeholder] * n_pose + newline
    ids += encode("输出该任务的唯一类别：") + im_end + newline
    ids += im_start + encode("assistant") + newline
    return ids, rgb_slots, pose_slots, len(ids) - 1


def forward_step(
    batch: dict[str, Any],
    minicpm,
    llm,
    pose_encoder,
    adapter,
    projector,
    heads,
    tokenizer,
    class_weights: dict[str, torch.Tensor],
):
    rgb = base.rgb_embeds(minicpm, batch["pixel_values"], batch["tgt_sizes"])
    pose = base.pose_embeds(pose_encoder, adapter, projector, batch["kps"])
    routed_task = ROUTED_TASK_BY_STATE[batch["system_state_before"]]
    if routed_task != batch["task"]:
        raise RuntimeError("batch state did not route to its declared task")
    ids, rgb_slots, pose_slots, prediction_position = build_sequence(
        tokenizer, batch["n_rgb"], pose.shape[0], routed_task, batch["system_state_before"]
    )
    if len(rgb_slots) != rgb.shape[0]:
        raise RuntimeError(f"RGB slot mismatch {len(rgb_slots)} != {rgb.shape[0]}")
    ids_tensor = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    embeddings = llm.model.embed_tokens(ids_tensor).to(DTYPE).clone()
    embeddings[0, rgb_slots] = rgb
    embeddings[0, pose_slots] = pose
    attention = torch.ones(
        1, embeddings.shape[1], dtype=torch.long, device=DEVICE
    )
    hidden = llm.model(inputs_embeds=embeddings, attention_mask=attention).last_hidden_state
    logits = heads(hidden[0, prediction_position], batch["system_state_before"])
    target_index = LABEL_TO_INDEX[routed_task][batch["label"]]
    target = torch.tensor([target_index], dtype=torch.long, device=DEVICE)
    # With batch_size=1, CrossEntropy's default weighted mean divides by the
    # same single class weight and cancels it.  Apply the train-derived weight
    # explicitly so class balancing remains effective under gradient
    # accumulation.
    unweighted_loss = F.cross_entropy(logits.unsqueeze(0), target)
    loss = unweighted_loss * class_weights[routed_task][target_index]
    return loss, logits, target_index, tuple(rgb.shape), tuple(pose.shape), embeddings.shape[1]


def compute_multiclass_metrics(
    y_true: list[str], y_pred: list[str], labels: tuple[str, ...]
) -> dict[str, Any]:
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true, predicted in zip(y_true, y_pred):
        matrix[index[true]][index[predicted]] += 1
    per_class: dict[str, dict[str, Any]] = {}
    for position, label in enumerate(labels):
        tp = matrix[position][position]
        fp = sum(matrix[row][position] for row in range(len(labels))) - tp
        fn = sum(matrix[position]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(matrix[position]),
        }
    total = len(y_true)
    accuracy = sum(matrix[i][i] for i in range(len(labels))) / total if total else 0.0
    return {
        "n": total,
        "accuracy": round(accuracy, 6),
        "balanced_accuracy": round(
            sum(value["recall"] for value in per_class.values()) / len(labels), 6
        ),
        "macro_precision": round(
            sum(value["precision"] for value in per_class.values()) / len(labels), 6
        ),
        "macro_recall": round(
            sum(value["recall"] for value in per_class.values()) / len(labels), 6
        ),
        "macro_f1": round(
            sum(value["f1"] for value in per_class.values()) / len(labels), 6
        ),
        "per_class": per_class,
        "confusion_matrix_labels": list(labels),
        "confusion_matrix": matrix,
    }


def compute_factorized_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    all_f1: list[float] = []
    for task, labels in TASK_LABELS.items():
        selected = [record for record in records if record["task"] == task]
        metrics = compute_multiclass_metrics(
            [record["true"] for record in selected],
            [record["pred"] for record in selected],
            labels,
        )
        result[task] = metrics
        all_f1.extend(metrics["per_class"][label]["f1"] for label in labels)
    barge = result["barge_in"]["per_class"]["BARGE"]
    result["barge_in"]["precision"] = barge["precision"]
    result["barge_in"]["recall"] = barge["recall"]
    result["barge_in"]["f1"] = barge["f1"]
    barge_records = [record for record in records if record["task"] == "barge_in"]
    y_binary = [1 if record["true"] == "BARGE" else 0 for record in barge_records]
    p_barge = [record["probabilities"][0] for record in barge_records]
    result["barge_in"]["roc_auc"] = round(
        float(roc_auc_score(y_binary, p_barge)), 6
    ) if len(set(y_binary)) == 2 else None
    result["combined_macro_f1"] = round(sum(all_f1) / len(all_f1), 6)
    return result


def set_train_mode(projector, adapter, llm, heads, training: bool) -> None:
    projector.train(training)
    adapter.train(training)
    heads.train(training)
    base.set_lora_mode(llm, training)


@torch.no_grad()
def evaluate(
    loader,
    minicpm,
    llm,
    pose_encoder,
    adapter,
    projector,
    heads,
    tokenizer,
    class_weights,
    tag: str,
):
    set_train_mode(projector, adapter, llm, heads, False)
    records: list[dict[str, Any]] = []
    total_loss = 0.0
    used = 0
    skipped = 0
    started = time.time()
    for index, batch in enumerate(loader, 1):
        if not batch["ok"]:
            skipped += 1
            log.warning("%s data skip %s: %s", tag, batch["sample_id"], batch["err"])
            continue
        try:
            loss, logits, _, rgb_shape, pose_shape, sequence_length = forward_step(
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
        except Exception as error:
            skipped += 1
            log.warning("%s forward skip %s: %s", tag, batch["sample_id"], error)
            continue
        if not torch.isfinite(loss):
            raise RuntimeError(f"{tag} non-finite loss: {batch['sample_id']}")
        total_loss += float(loss.item())
        used += 1
        labels = TASK_LABELS[batch["task"]]
        probabilities = torch.softmax(logits.float(), dim=0)
        predicted_index = int(logits.argmax().item())
        predicted = labels[predicted_index]
        audit = batch["audit"]
        records.append(
            {
                "sample_id": batch["sample_id"],
                "video_id": audit["video_id"],
                "task": batch["task"],
                "true": batch["label"],
                "pred": predicted,
                "logits": [round(float(value), 5) for value in logits.float().tolist()],
                "probabilities": [round(float(value), 8) for value in probabilities.tolist()],
                "confidence": round(float(probabilities[predicted_index]), 8),
                "decision_frame": audit["decision_frame"],
                "decision_time": audit["decision_time"],
                "system_state": audit["system_state"],
                "pair_id": audit["pair_id"],
                "prefix_frame_count": audit["prefix_frame_count"],
                "pose_length": int(batch["n_pose"]),
                "final_token_length": int(sequence_length),
                "rgb_token_shape": list(rgb_shape),
                "pose_token_shape": list(pose_shape),
            }
        )
        if index % 500 == 0:
            log.info("%s %d/%d elapsed=%.0fs", tag, index, len(loader), time.time() - started)
    metrics = compute_factorized_metrics(records)
    metrics["loss"] = round(total_loss / used, 6) if used else None
    metrics["n_skipped"] = skipped
    return metrics, records


def format_metrics(metrics: dict[str, Any], title: str) -> str:
    lines = [
        "=" * 96,
        title,
        f"loss={metrics['loss']} combined Macro-F1={metrics['combined_macro_f1']:.6f}",
    ]
    for task, labels in TASK_LABELS.items():
        value = metrics[task]
        lines.append(
            f"{task}: n={value['n']} acc={value['accuracy']:.6f} "
            f"balanced_acc={value['balanced_accuracy']:.6f} macro_f1={value['macro_f1']:.6f}"
        )
        for label in labels:
            item = value["per_class"][label]
            lines.append(
                f"  {label:<12} P={item['precision']:.6f} R={item['recall']:.6f} "
                f"F1={item['f1']:.6f} support={item['support']}"
            )
        lines.append(
            "  confusion rows=true cols=pred labels="
            + json.dumps(value["confusion_matrix_labels"], ensure_ascii=False)
        )
        lines.extend("  " + str(row) for row in value["confusion_matrix"])
    lines.append("=" * 96)
    return "\n".join(lines)


def class_weights_from_train(rows: list[dict[str, Any]]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    report: dict[str, Any] = {}
    for task, labels in TASK_LABELS.items():
        counts = Counter(row["label"] for row in rows if row["task"] == task)
        total = sum(counts.values())
        weights = [total / (len(labels) * counts[label]) for label in labels]
        tensors[task] = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
        report[task] = {
            "labels": list(labels),
            "counts": {label: counts[label] for label in labels},
            "weights": {label: round(weight, 8) for label, weight in zip(labels, weights)},
            "source_split": "train",
            "dev_test_unchanged": True,
        }
    return tensors, report


def build_optimizer(projector, adapter, llm, pose_encoder, heads):
    groups = [
        {
            "params": list(projector.parameters()),
            "lr": LR_PROJECTOR,
            "weight_decay": WEIGHT_DECAY,
            "name": "sign_projector",
        },
        {
            "params": list(adapter.parameters()),
            "lr": LR_ADAPTER,
            "weight_decay": WEIGHT_DECAY,
            "name": "temporal_adapter",
        },
        {
            "params": list(heads.parameters()),
            "lr": LR_CLASSIFICATION_HEAD,
            "weight_decay": WEIGHT_DECAY,
            "name": "factorized_heads",
        },
        {
            "params": [parameter for _, parameter in base.lora_parameters(llm)],
            "lr": LR_LORA,
            "weight_decay": 0.0,
            "name": "lora",
        },
        {
            "params": list(pose_encoder.pose_proj.parameters()),
            "lr": LR_UNISIGN,
            "weight_decay": WEIGHT_DECAY,
            "name": "unisign_pose_proj",
        },
        {
            "params": list(pose_encoder.fusion_gcn_modules.parameters()),
            "lr": LR_UNISIGN,
            "weight_decay": WEIGHT_DECAY,
            "name": "unisign_fusion_gcn",
        },
    ]
    parameters = [parameter for group in groups for parameter in group["params"]]
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise RuntimeError("optimizer parameter groups overlap")
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    return optimizer, parameters, groups


def trainable_audit(projector, adapter, llm, pose_encoder, heads, minicpm) -> dict[str, Any]:
    counts = {
        "sign_projector": sum(p.numel() for p in projector.parameters() if p.requires_grad),
        "temporal_adapter": sum(p.numel() for p in adapter.parameters() if p.requires_grad),
        "factorized_heads": sum(p.numel() for p in heads.parameters() if p.requires_grad),
        "lora": sum(p.numel() for _, p in base.lora_parameters(llm)),
        "unisign_unfrozen": sum(
            p.numel() for p in pose_encoder.parameters() if p.requires_grad
        ),
    }
    errors: list[str] = []
    for name, module in (
        ("embed_tokens", llm.model.embed_tokens),
        ("lm_head", llm.lm_head),
        ("vpm", minicpm.vpm),
        ("resampler", minicpm.resampler),
    ):
        if any(parameter.requires_grad for parameter in module.parameters()):
            errors.append(f"{name} is not frozen")
    if any(parameter.requires_grad for parameter in pose_encoder.gcn_modules.parameters()):
        errors.append("Uni-Sign base gcn_modules is not frozen")
    if any(not parameter.requires_grad for parameter in pose_encoder.pose_proj.parameters()):
        errors.append("Uni-Sign pose_proj is not fully trainable")
    if any(not parameter.requires_grad for parameter in pose_encoder.fusion_gcn_modules.parameters()):
        errors.append("Uni-Sign fusion_gcn_modules is not fully trainable")
    if any(not parameter.requires_grad for parameter in heads.parameters()):
        errors.append("factorized head has frozen parameters")
    counts["total_trainable"] = sum(counts.values())
    counts["turn_head_output_dim"] = heads.turn_state.out_features
    counts["barge_head_output_dim"] = heads.barge_in.out_features
    counts["all_passed"] = not errors
    counts["errors"] = errors
    if errors:
        raise RuntimeError("trainable audit failed: " + "; ".join(errors))
    return counts


def save_checkpoint(
    path: Path,
    epoch: int,
    projector,
    adapter,
    llm,
    pose_encoder,
    heads,
    train_loss: float,
    dev_metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    value = {
        "epoch": epoch,
        "sign_projector": {
            key: tensor.detach().float().cpu() for key, tensor in projector.state_dict().items()
        },
        "temporal_adapter": {
            key: tensor.detach().float().cpu() for key, tensor in adapter.state_dict().items()
        },
        "lora": base.lora_state_dict(llm),
        "unisign_unfrozen": {
            name: parameter.detach().float().cpu()
            for name, parameter in pose_encoder.named_parameters()
            if parameter.requires_grad
        },
        "factorized_heads": {
            key: tensor.detach().float().cpu() for key, tensor in heads.state_dict().items()
        },
        "train_loss": train_loss,
        "dev_metrics": dev_metrics,
        "config": config,
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def load_checkpoint(path: Path, projector, adapter, llm, pose_encoder, heads):
    checkpoint = torch.load(path, map_location="cpu")
    projector.load_state_dict(checkpoint["sign_projector"])
    adapter.load_state_dict(checkpoint["temporal_adapter"])
    heads.load_state_dict(checkpoint["factorized_heads"])
    llm_parameters = dict(llm.named_parameters())
    pose_parameters = dict(pose_encoder.named_parameters())
    with torch.no_grad():
        for name, value in checkpoint["lora"].items():
            llm_parameters[name].copy_(
                value.to(llm_parameters[name].device, dtype=llm_parameters[name].dtype)
            )
        for name, value in checkpoint["unisign_unfrozen"].items():
            pose_parameters[name].copy_(
                value.to(pose_parameters[name].device, dtype=pose_parameters[name].dtype)
            )
    return checkpoint


def gpu_audit() -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != SELECTED_PHYSICAL_GPU:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be exactly {SELECTED_PHYSICAL_GPU}")
    allocation = torch.zeros(1, device=DEVICE)
    del allocation
    torch.cuda.synchronize()
    current = int(torch.cuda.current_device())
    gpu_lines = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    physical = {}
    for line in gpu_lines:
        index, uuid, pci, name, memory = [part.strip() for part in line.split(",", 4)]
        physical[uuid] = {
            "index": int(index),
            "pci_bus_id": pci,
            "name": name,
            "memory_total_mib": int(memory),
        }
    process_lines = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    matches = []
    for line in process_lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and parts[1] == str(os.getpid()):
            matches.append(
                {
                    "gpu_uuid": parts[0],
                    "physical_index": physical.get(parts[0], {}).get("index"),
                    "used_memory_mib": int(parts[2]),
                }
            )
    selected_index = int(SELECTED_PHYSICAL_GPU)
    if current != 0 or not any(item["physical_index"] == selected_index for item in matches):
        raise RuntimeError(
            f"GPU mapping audit failed: current={current}, process_matches={matches}"
        )
    return {
        "CUDA_VISIBLE_DEVICES": SELECTED_PHYSICAL_GPU,
        "torch_cuda_current_device": current,
        "torch_device_name": torch.cuda.get_device_name(current),
        "process_matches": matches,
        "selected_physical_gpu": next(
            value for value in physical.values() if value["index"] == selected_index
        ),
        "all_passed": True,
    }


def log_random_samples(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    for label_index, label in enumerate(TURN_LABELS + BARGE_LABELS):
        candidates = [row for row in all_rows if row["label"] == label]
        chosen = random.Random(20260816 + label_index).sample(candidates, 20)
        log.info("[random sample] %s (20)", label)
        for row in chosen:
            log.info(
                "  %s split=%s frame=%s time=%s label=%s quality=%s confidence=%s type=%s",
                row["video_id"],
                row["split"],
                row.get("candidate_frame"),
                row.get("candidate_time_sec"),
                row["label"],
                row.get("quality_gate_pass"),
                row.get("confidence"),
                row.get("confidence_type"),
            )


def select_smoke_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for label in TURN_LABELS + BARGE_LABELS:
        match = next(row for row in rows if row["label"] == label)
        selected.append(match)
    return selected[: max(2, min(count, 5))]


def run_forward_backward_smoke(
    rows,
    minicpm,
    llm,
    pose_encoder,
    adapter,
    projector,
    heads,
    tokenizer,
    class_weights,
) -> dict[str, Any]:
    loader = make_loader(rows, "smoke", workers=0)
    optimizer, _, optimizer_groups = build_optimizer(
        projector, adapter, llm, pose_encoder, heads
    )
    optimizer.zero_grad(set_to_none=True)
    set_train_mode(projector, adapter, llm, heads, True)
    outputs = []
    for batch in loader:
        if not batch["ok"]:
            raise RuntimeError(f"smoke dataloader failed: {batch}")
        expected_task = ROUTED_TASK_BY_STATE[batch["system_state_before"]]
        if batch["task"] != expected_task:
            raise RuntimeError("smoke state router mismatch")
        loss, logits, target, rgb_shape, pose_shape, sequence_length = forward_step(
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
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise RuntimeError("smoke loss/logits contain NaN or Inf")
        expected_dim = len(TASK_LABELS[batch["task"]])
        if logits.numel() != expected_dim:
            raise RuntimeError(
                f"{batch['task']} output dim {logits.numel()} != {expected_dim}"
            )
        loss.backward()
        outputs.append(
            {
                "sample_id": batch["sample_id"],
                "system_state": batch["system_state_before"],
                "task": batch["task"],
                "label": batch["label"],
                "target_index": target,
                "output_dim": logits.numel(),
                "loss": round(float(loss.item()), 6),
                "rgb_shape": list(rgb_shape),
                "pose_shape": list(pose_shape),
                "sequence_length": sequence_length,
            }
        )
    gradient_report = {}
    for group in optimizer_groups:
        grads = [parameter.grad for parameter in group["params"] if parameter.grad is not None]
        finite = all(torch.isfinite(grad).all().item() for grad in grads)
        nonzero = sum(bool(torch.count_nonzero(grad).item()) for grad in grads)
        gradient_report[group["name"]] = {
            "gradient_tensor_count": len(grads),
            "nonzero_gradient_tensor_count": nonzero,
            "all_finite": finite,
        }
        if not grads or not finite or nonzero == 0:
            raise RuntimeError(f"smoke allowed-gradient audit failed: {group['name']}")
    frozen_with_grad = []
    for module_name, module in (
        ("vpm", minicpm.vpm),
        ("resampler", minicpm.resampler),
        ("qwen_embed_tokens", llm.model.embed_tokens),
        ("qwen_lm_head", llm.lm_head),
        ("unisign_base_gcn", pose_encoder.gcn_modules),
    ):
        if any(parameter.grad is not None for parameter in module.parameters()):
            frozen_with_grad.append(module_name)
    if frozen_with_grad:
        raise RuntimeError(f"smoke frozen modules received gradients: {frozen_with_grad}")
    optimizer.zero_grad(set_to_none=True)
    return {
        "forward_only": False,
        "forward_backward": True,
        "batch_count": len(outputs),
        "batches": outputs,
        "head_dimensions": {"turn_state": 3, "barge_in": 2},
        "loss_all_finite": True,
        "rgb_pose_inputs_ok": True,
        "label_mapping_ok": True,
        "state_router_ok": True,
        "allowed_gradient_report": gradient_report,
        "frozen_modules_with_gradients": frozen_with_grad,
        "all_passed": True,
    }


def sha256_file(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def static_model_input_audit() -> dict[str, Any]:
    source = inspect.getsource(build_sequence) + inspect.getsource(forward_step)
    mentions = {field: source.count(field) for field in sorted(BANNED_MODEL_FIELDS)}
    failures = {field: count for field, count in mentions.items() if count}
    if failures:
        raise RuntimeError(f"forbidden audit metadata appears in model-input code: {failures}")
    return {
        "status": "PASS",
        "model_input_fields": ["RGB prefix", "Pose prefix", "system_state_before"],
        "router": "LISTENING/0 -> turn_state(3-way); SPEAKING/1 -> barge_in(2-way)",
        "forbidden_field_mentions_in_model_input_code": mentions,
        "unified_five_class_softmax": False,
    }


def copy_data_audits() -> None:
    shutil.copy2(DATA_ROOT / "DATA_STATS.json", OUT_DIR / "v1_6_data_stats.json")
    shutil.copy2(DATA_ROOT / "AUDIT_REPORT.md", OUT_DIR / "v1_6_data_audit_report.md")
    shutil.copy2(DATA_ROOT / "V1_6_MANIFEST.json", OUT_DIR / "v1_6_data_manifest.json")


def row_length_features(row: dict[str, Any], tokenizer) -> list[int]:
    prefix = int(row["window_end_frame"])
    available = min(prefix, int(row["video_frames"]))
    n_rgb = min(max(1, available), int(base.RGB_FRAMES))
    pose_length = min(max(1, available), int(base.MAX_POSE_LEN))
    pose_tokens = (pose_length + 1) // 2
    ids, _, _, _ = build_sequence(
        tokenizer, n_rgb, pose_tokens, row["task"], row["system_state_before"]
    )
    return [prefix, pose_length, len(ids)]


def same_video_pair_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [record for record in records if record["task"] == "barge_in"]
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for record in selected:
        if not record.get("pair_id"):
            raise RuntimeError(f"test barge record missing pair_id: {record['sample_id']}")
        grouped.setdefault(record["pair_id"], {})[record["true"]] = record
    valid = [pair for pair in grouped.values() if set(pair) == set(BARGE_LABELS)]
    if len(valid) * 2 != len(selected):
        raise RuntimeError("test same-video pair cardinality mismatch")
    no_barge_correct = sum(pair["NO_BARGE"]["pred"] == "NO_BARGE" for pair in valid)
    barge_correct = sum(pair["BARGE"]["pred"] == "BARGE" for pair in valid)
    both = sum(
        pair["NO_BARGE"]["pred"] == "NO_BARGE" and pair["BARGE"]["pred"] == "BARGE"
        for pair in valid
    )
    count = len(valid)
    return {
        "pair_count": count,
        "NO_BARGE_accuracy": round(no_barge_correct / count, 6),
        "BARGE_accuracy": round(barge_correct / count, 6),
        "pair_both_correct_accuracy": round(both / count, 6),
    }


def length_shortcut_audit(
    train_rows: list[dict[str, Any]], test_records: list[dict[str, Any]], tokenizer,
) -> dict[str, Any]:
    train = [row for row in train_rows if row["task"] == "barge_in"]
    test = [record for record in test_records if record["task"] == "barge_in"]
    x_train = np.asarray([row_length_features(row, tokenizer) for row in train], dtype=np.float64)
    y_train = np.asarray([1 if row["label"] == "BARGE" else 0 for row in train], dtype=np.int64)
    x_test = np.asarray(
        [[record["prefix_frame_count"], record["pose_length"], record["final_token_length"]] for record in test],
        dtype=np.float64,
    )
    y_test = np.asarray([1 if record["true"] == "BARGE" else 0 for record in test], dtype=np.int64)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(random_state=SEED, max_iter=1000)
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]
    balanced = float(balanced_accuracy_score(y_test, prediction))

    feature_names = ("prefix_frame_count", "pose_length", "final_token_length")
    bucket_metrics: dict[str, Any] = {}
    for feature_index, feature_name in enumerate(feature_names):
        quartiles = np.quantile(x_train[:, feature_index], [0.25, 0.50, 0.75])
        boundaries = sorted(set(float(value) for value in quartiles))
        buckets = []
        for bucket_index in range(len(boundaries) + 1):
            members = []
            for record in test:
                value = float(record[feature_name])
                assigned = int(np.digitize([value], boundaries, right=True)[0])
                if assigned == bucket_index:
                    members.append(record)
            by_label = {}
            for label in BARGE_LABELS:
                subset = [record for record in members if record["true"] == label]
                by_label[label] = {
                    "count": len(subset),
                    "accuracy": round(
                        sum(record["pred"] == label for record in subset) / len(subset), 6
                    ) if subset else None,
                }
            buckets.append({"bucket": bucket_index, "count": len(members), "by_true_label": by_label})
        bucket_metrics[feature_name] = {"train_quartile_boundaries": boundaries, "test_buckets": buckets}
    warning_threshold = 0.60
    return {
        "status": "REPORT_ONLY",
        "allowed_inputs": list(feature_names),
        "classifier": "StandardScaler + LogisticRegression(random_state=42)",
        "train_count": len(train),
        "test_count": len(test),
        "accuracy": round(float(np.mean(prediction == y_test)), 6),
        "balanced_accuracy": round(balanced, 6),
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 6),
        "warning_threshold": warning_threshold,
        "LENGTH_SHORTCUT_WARNING": "YES" if balanced > warning_threshold else "NO",
        "main_model_accuracy_by_length_bucket": bucket_metrics,
    }


def save_confusion_matrix_png(metrics: dict[str, Any], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    labels = metrics["confusion_matrix_labels"]
    figure, axis = plt.subplots(figsize=(5.5, 4.8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > threshold else "black")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def random_error_cases(records: list[dict[str, Any]], per_class: int = 10) -> dict[str, Any]:
    result = {"seed": SEED, "per_class_limit": per_class, "classes": {}}
    for index, label in enumerate(TURN_LABELS + BARGE_LABELS):
        errors = [record for record in records if record["true"] == label and record["pred"] != label]
        random.Random(SEED + index).shuffle(errors)
        result["classes"][label] = [
            {
                "video_id": record["video_id"],
                "label": record["true"],
                "prediction": record["pred"],
                "decision_frame": record["decision_frame"],
                "decision_time": record["decision_time"],
                "system_state": record["system_state"],
                "confidence": record["confidence"],
                "logits": record["logits"],
            }
            for record in errors[:per_class]
        ]
    return result


def main() -> None:
    started = time.time()
    set_seed()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    source_audit = json.loads((DATA_ROOT / "DATA_STATS.json").read_text(encoding="utf-8"))
    if source_audit.get("ready_for_manual_audit") is not True:
        raise RuntimeError("V1.6 data hard assertions did not pass")
    if set(source_audit.get("hard_assertions", {}).values()) != {"PASS"}:
        raise RuntimeError("V1.6 data contains a failed hard assertion")
    if int(base.ARGS.smoke) == 0 and NUM_EPOCHS != 10:
        raise RuntimeError("formal V1.6 run must use exactly 10 epochs")
    if int(base.ARGS.smoke) == 0 and (OUT_DIR / "config.json").exists():
        raise RuntimeError(f"refusing to overwrite an existing formal run: {OUT_DIR}")

    model_input_audit = static_model_input_audit()
    source_hashes_before = {
        split: sha256_file(DATA_DIR / f"{split}.jsonl") for split in ("train", "dev", "test")
    }
    copy_data_audits()
    rows_by_split = {split: load_jsonl(split) for split in ("train", "dev", "test")}
    log_random_samples(rows_by_split)
    for split, rows in rows_by_split.items():
        log.info(
            "%s n=%d tasks=%s labels=%s",
            split,
            len(rows),
            dict(Counter(row["task"] for row in rows)),
            dict(Counter(row["label"] for row in rows)),
        )

    from transformers import AutoModel, AutoTokenizer

    log.info("loading MiniCPM-o-4.5 RGB tower + LLM in bf16 (audio/tts disabled)")
    tokenizer = AutoTokenizer.from_pretrained(base.MINICPM_DIR, trust_remote_code=True)
    minicpm = AutoModel.from_pretrained(
        base.MINICPM_DIR,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map={"": DEVICE},
        attn_implementation="eager",
        init_audio=False,
        init_tts=False,
    )
    minicpm.eval()
    for parameter in minicpm.parameters():
        parameter.requires_grad = False
    llm = minicpm.llm
    if llm.config.num_hidden_layers != 36:
        raise RuntimeError("unexpected LLM architecture")
    hidden_size = int(llm.config.hidden_size)
    setup_placeholders(tokenizer)
    base.inject_lora(
        llm,
        LORA_LAYERS,
        LORA_TARGETS,
        LORA_R,
        LORA_ALPHA,
        LORA_DROPOUT,
    )
    base.set_lora_mode(llm, False)
    pose_encoder, _ = base.build_pose_encoder()
    projector = base.SignProjector(768, hidden_size).to(DEVICE, dtype=TDTYPE)
    adapter = base.TemporalAdapter().to(DEVICE, dtype=TDTYPE)
    for module in (projector, adapter):
        for parameter in module.parameters():
            parameter.requires_grad = True
    heads = FactorizedHeads(hidden_size).to(DEVICE, dtype=TDTYPE)
    for parameter in heads.parameters():
        parameter.requires_grad = True
    warm_start = base.warm_start_from_exp25(
        base.EXP25_CKPT, projector, adapter, llm, pose_encoder
    )

    hardware_audit = gpu_audit()
    log.info("GPU audit: %s", json.dumps(hardware_audit, ensure_ascii=False))
    parameter_audit = trainable_audit(
        projector, adapter, llm, pose_encoder, heads, minicpm
    )
    class_weights, class_weight_report = class_weights_from_train(
        rows_by_split["train"]
    )
    log.info("class weights (train only): %s", json.dumps(class_weight_report))

    config = {
        "version": VERSION,
        "data_dir": str(DATA_DIR),
        "output_dir": str(OUT_DIR),
        "gpu": hardware_audit,
        "factorized_heads": {
            "turn_state": {"labels": list(TURN_LABELS), "output_dim": 3},
            "barge_in": {"labels": list(BARGE_LABELS), "output_dim": 2},
            "not_five_class_softmax": True,
        },
        "inputs": ["RGB prefix", "Pose prefix", "known system_state_before"],
        "model_input_audit": model_input_audit,
        "excluded_from_model_input": sorted(BANNED_MODEL_FIELDS),
        "audio_or_tts": False,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "epochs": NUM_EPOCHS,
        "bf16": True,
        "optimizer": "AdamW",
        "learning_rates": {
            "projector": LR_PROJECTOR,
            "temporal_adapter": LR_ADAPTER,
            "classification_heads": LR_CLASSIFICATION_HEAD,
            "lora": LR_LORA,
            "unisign_unfrozen": LR_UNISIGN,
        },
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "scheduler": "cosine",
        "max_grad_norm": MAX_GRAD_NORM,
        "seed": SEED,
        "early_stopping": False,
        "class_weights": class_weight_report,
        "best_checkpoint_selection": "dev combined Macro-F1 across five factorized class metrics",
        "n_train": len(rows_by_split["train"]),
        "n_dev": len(rows_by_split["dev"]),
        "n_test": len(rows_by_split["test"]),
        "parameter_audit": parameter_audit,
        "warm_start": {
            "path": base.EXP25_CKPT,
            "sha256": sha256_file(Path(base.EXP25_CKPT)),
            "source_epoch": warm_start.get("epoch"),
            "adapter_variant": warm_start.get("adapter_variant"),
        },
        "source_data_sha256_before": source_hashes_before,
        "dataloader": {
            "num_workers": int(base.ARGS.workers),
            "pin_memory": True,
            "persistent_workers": int(base.ARGS.workers) > 0,
            "prefetch_factor": 4,
        },
    }
    write_json_atomic(OUT_DIR / "config.json", config)

    if int(base.ARGS.smoke) > 0:
        smoke_rows = select_smoke_rows(rows_by_split["train"], int(base.ARGS.smoke))
        smoke_report = run_forward_backward_smoke(
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
        smoke_report["gpu"] = hardware_audit
        smoke_report["parameter_audit"] = parameter_audit
        write_json_atomic(OUT_DIR / "smoke_report.json", smoke_report)
        smoke_report["model_input_audit"] = model_input_audit
        write_json_atomic(OUT_DIR / "smoke_report.json", smoke_report)
        log.info("forward/backward smoke PASS: %s", json.dumps(smoke_report, ensure_ascii=False))
        return

    optimizer, all_trainable, optimizer_groups = build_optimizer(
        projector, adapter, llm, pose_encoder, heads
    )
    steps_per_epoch = math.ceil(len(rows_by_split["train"]) / GRAD_ACCUM)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    config["optimizer_groups"] = {
        group["name"]: {
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
        }
        for group in optimizer_groups
    }
    config["steps_per_epoch"] = steps_per_epoch
    config["total_steps"] = total_steps
    config["warmup_steps"] = warmup_steps
    write_json_atomic(OUT_DIR / "config.json", config)

    metrics_path = OUT_DIR / "metrics.jsonl"
    best_score = -1.0
    best_epoch = -1
    history: list[dict[str, Any]] = []
    dev_loader = make_loader(rows_by_split["dev"], "dev")
    train_loader = make_loader(rows_by_split["train"], "train", shuffle=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_started = time.time()
        log.info("%s", "=" * 96)
        log.info("Epoch %d/%d", epoch, NUM_EPOCHS)
        set_train_mode(projector, adapter, llm, heads, True)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        used = 0
        skipped = 0
        accumulated = 0
        last_grad_norm = 0.0
        for batch in train_loader:
            if not batch["ok"]:
                skipped += 1
                log.warning("train data skip %s: %s", batch["sample_id"], batch["err"])
                continue
            loss, _, _, _, _, _ = forward_step(
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
            (loss / GRAD_ACCUM).backward()
            total_loss += float(loss.item())
            used += 1
            accumulated += 1
            if accumulated % GRAD_ACCUM == 0:
                last_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(all_trainable, MAX_GRAD_NORM).item()
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
        if accumulated % GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_(all_trainable, MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        train_loss = total_loss / used if used else None
        if train_loss is None:
            raise RuntimeError("no usable training batches")

        dev_metrics, dev_predictions = evaluate(
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
        log.info("%s", format_metrics(dev_metrics, f"Epoch {epoch} DEV"))
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_used": used,
            "train_skipped": skipped,
            "dev_loss": dev_metrics["loss"],
            "dev": dev_metrics,
            "elapsed_min": round((time.time() - epoch_started) / 60.0, 3),
            "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        }
        history.append(epoch_record)
        append_jsonl(metrics_path, epoch_record)
        write_json_atomic(OUT_DIR / "history.json", history)
        save_checkpoint(
            OUT_DIR / "last.pt",
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
        score = float(dev_metrics["combined_macro_f1"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(
                OUT_DIR / "best.pt",
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
            write_json_atomic(
                OUT_DIR / "dev_metrics.json",
                {"best_epoch": epoch, **dev_metrics},
            )
            write_jsonl_atomic(OUT_DIR / "predictions" / "dev_best_predictions.jsonl", dev_predictions)
            log.info("new best epoch=%d dev combined Macro-F1=%.6f", epoch, score)
        write_json_atomic(
            OUT_DIR / "training_state.json",
            {
                "completed_epochs": epoch,
                "best_epoch": best_epoch,
                "best_dev_macro_f1": best_score,
                "history": history,
            },
        )

    if len(history) != 10:
        raise RuntimeError(f"formal training ended at {len(history)} epochs, expected 10")
    checkpoint = load_checkpoint(
        OUT_DIR / "best.pt",
        projector,
        adapter,
        llm,
        pose_encoder,
        heads,
    )
    if int(checkpoint["epoch"]) != best_epoch:
        raise RuntimeError("best checkpoint epoch mismatch")
    test_loader = make_loader(rows_by_split["test"], "test")
    test_metrics, test_predictions = evaluate(
        test_loader,
        minicpm,
        llm,
        pose_encoder,
        adapter,
        projector,
        heads,
        tokenizer,
        class_weights,
        "test",
    )
    log.info("%s", format_metrics(test_metrics, f"TEST best epoch={best_epoch}"))
    write_jsonl_atomic(
        OUT_DIR / "predictions" / "test_predictions.jsonl", test_predictions
    )
    write_json_atomic(OUT_DIR / "test_metrics.json", test_metrics)
    save_confusion_matrix_png(
        test_metrics["turn_state"], OUT_DIR / "turn_confusion_matrix.png", "Turn-State test"
    )
    save_confusion_matrix_png(
        test_metrics["barge_in"], OUT_DIR / "barge_confusion_matrix.png", "Barge-In test"
    )
    pair_metrics = same_video_pair_metrics(test_predictions)
    write_json_atomic(OUT_DIR / "same_video_pair_metrics.json", pair_metrics)
    shortcut = length_shortcut_audit(rows_by_split["train"], test_predictions, tokenizer)
    write_json_atomic(OUT_DIR / "length_shortcut_audit.json", shortcut)
    errors = random_error_cases(test_predictions)
    write_json_atomic(OUT_DIR / "random_error_cases.json", errors)
    source_hashes_after = {
        split: sha256_file(DATA_DIR / f"{split}.jsonl") for split in ("train", "dev", "test")
    }
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("V1.6 source dataset changed during training")
    confusions = []
    for task, labels in TASK_LABELS.items():
        matrix = test_metrics[task]["confusion_matrix"]
        for true_index, true_label in enumerate(labels):
            for pred_index, pred_label in enumerate(labels):
                if true_index != pred_index and matrix[true_index][pred_index]:
                    confusions.append({
                        "task": task, "true": true_label, "predicted": pred_label,
                        "count": matrix[true_index][pred_index],
                    })
    confusions.sort(key=lambda item: (-item["count"], item["task"], item["true"], item["predicted"]))
    write_json_atomic(
        OUT_DIR / "run_summary.json",
        {
            "version": VERSION,
            "completed_epochs": 10,
            "best_epoch": best_epoch,
            "best_dev_macro_f1": best_score,
            "test_macro_f1": test_metrics["combined_macro_f1"],
            "turn_state_test_accuracy": test_metrics["turn_state"]["accuracy"],
            "barge_in_test_accuracy": test_metrics["barge_in"]["accuracy"],
            "best_checkpoint": str(OUT_DIR / "best.pt"),
            "last_checkpoint": str(OUT_DIR / "last.pt"),
            "gpu": hardware_audit,
            "total_hours": round((time.time() - started) / 3600.0, 4),
            "same_video_pair_metrics": pair_metrics,
            "length_shortcut_audit": shortcut,
            "major_confusions": confusions[:3],
            "source_data_sha256_after": source_hashes_after,
            "source_data_preserved": True,
            "all_passed": True,
        },
    )
    log.info(
        "training complete: best epoch=%d dev Macro-F1=%.6f test Macro-F1=%.6f",
        best_epoch,
        best_score,
        test_metrics["combined_macro_f1"],
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("training failed")
        raise
