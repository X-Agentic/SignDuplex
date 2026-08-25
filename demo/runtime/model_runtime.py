"""Exact V1.8 IDLE-aware two-head inference runtime.

The architecture, prompt template, preprocessing, and label order mirror
`Easy_Turn_Sign_Duplex_Data/train_sign_duplex_v1_8.py`. No model component is
redesigned for the demo.
"""

from __future__ import annotations

import copy
import importlib.machinery
import os
import sys
import time
from argparse import Namespace
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from config import (
    BARGE_LABELS,
    CHECKPOINT,
    DEVICE,
    MAX_POSE_LEN,
    MINICPM_DIR,
    TASK_BY_STATE,
    TURN_LABELS,
    UNISIGN_CHECKPOINT,
    UNISIGN_DIR,
)
from runtime.video_stream import uniform_indices


DTYPE = torch.bfloat16
TENSOR_DTYPE = torch.float32
POSE_DTYPE = torch.float32
LORA_LAYERS = list(range(20, 36))
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
SYSTEM_TEXT = (
    "你是手语实时对话的轮次决策器。只根据当前可见的手语视频、骨架前缀，"
    "以及已知系统状态完成对应分类任务。"
)


def _patch_broken_flash_attn() -> None:
    """Match the validated training environment's flash-attn compatibility shim."""
    def mock(name: str, is_package: bool = False, attrs: tuple[str, ...] = ()):
        module = ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(
            name, loader=None, is_package=is_package
        )
        module.__loader__ = None
        module.__package__ = name if is_package else name.rsplit(".", 1)[0]
        if is_package:
            module.__path__ = []
        for attr in attrs:
            setattr(module, attr, None)
        sys.modules[name] = module

    mock("flash_attn_2_cuda")
    mock("flash_attn", True, ("flash_attn_func", "flash_attn_varlen_func"))
    mock(
        "flash_attn.flash_attn_interface",
        attrs=("flash_attn_func", "flash_attn_varlen_func", "flash_attn_with_kvcache"),
    )
    mock("flash_attn.bert_padding", attrs=("index_first_axis", "pad_input", "unpad_input"))


class PoseEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.modes = model.modes
        self.proj_linear = model.proj_linear
        self.gcn_modules = model.gcn_modules
        self.fusion_gcn_modules = model.fusion_gcn_modules
        self.part_para = model.part_para
        self.pose_proj = model.pose_proj

    def forward(self, source):
        with torch.no_grad():
            gcn_outputs: dict[str, torch.Tensor] = {}
            body = None
            for part in self.modes:
                feature = self.proj_linear[part](source[part]).permute(0, 3, 1, 2)
                feature = self.gcn_modules[part](feature)
                if part == "body":
                    body = feature
                elif part == "left":
                    feature = feature + body[..., -2][..., None].detach()
                elif part == "right":
                    feature = feature + body[..., -1][..., None].detach()
                elif part == "face_all":
                    feature = feature + body[..., 0][..., None].detach()
                gcn_outputs[part] = feature
        features = []
        for part in self.modes:
            feature = self.fusion_gcn_modules[part](gcn_outputs[part])
            features.append(feature.mean(-1).transpose(1, 2))
        return self.pose_proj(torch.cat(features, dim=-1) + self.part_para)


class SignProjector(nn.Module):
    def __init__(self, input_dim: int = 768, output_dim: int = 4096):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(output_dim, output_dim)

    def forward(self, value):
        return self.linear2(self.activation(self.linear1(value.float())))


class LearnedCausalDownsample(nn.Module):
    def __init__(self, model_dim: int = 768):
        super().__init__()
        self.conv = nn.Conv1d(model_dim, model_dim, kernel_size=2, stride=2, padding=0)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, value):
        value = F.pad(value.transpose(1, 2), (1, 0))
        return self.norm(self.conv(value).transpose(1, 2))


class TemporalAdapter(nn.Module):
    def __init__(self, model_dim: int = 768):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=2,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.down = LearnedCausalDownsample(model_dim)

    def forward(self, value):
        length = value.shape[1]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            length, device=value.device, dtype=value.dtype
        )
        return self.down(self.encoder(value, mask=mask, is_causal=True))


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))

    def forward(self, value):
        output = self.base(value)
        hidden = self.lora_dropout(value.float())
        delta = (hidden @ self.lora_A.t()) @ self.lora_B.t()
        return output + (delta * self.scaling).to(output.dtype)


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


class ModelRuntime:
    """Lazy-loading, inference-only wrapper around the formal V1.8 checkpoint."""

    def __init__(self, device: str = DEVICE):
        if not torch.cuda.is_available() and str(device).startswith("cuda"):
            raise RuntimeError("CUDA is required by the formal bf16 V1.6 runtime")
        self.device = torch.device(device)
        self.tokens: dict[str, int] = {}
        self.image_processor = None
        self._load()

    def warmup(self, iterations: int = 3) -> list[float]:
        """Warm all inference branches without loading any user video."""
        frames = [np.zeros((448, 448, 3), dtype=np.uint8) for _ in range(8)]
        pose = {
            "keypoints": [np.zeros((1, 133, 2), np.float32) for _ in range(30)],
            "scores": [np.zeros((1, 133), np.float32) for _ in range(30)],
        }
        latencies = []
        for index in range(iterations):
            state = "LISTENING" if index % 2 == 0 else "SPEAKING"
            latencies.append(self.predict(frames, pose, state)["model_latency_ms"])
        return latencies

    def _load(self) -> None:
        _patch_broken_flash_attn()
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            MINICPM_DIR, trust_remote_code=True
        )
        self.image_processor = AutoProcessor.from_pretrained(
            MINICPM_DIR, trust_remote_code=True
        ).image_processor
        self.minicpm = AutoModel.from_pretrained(
            MINICPM_DIR,
            trust_remote_code=True,
            torch_dtype=DTYPE,
            device_map={"": self.device},
            attn_implementation="eager",
            init_audio=False,
            init_tts=False,
        )
        self.minicpm.eval()
        self.llm = self.minicpm.llm
        if int(self.llm.config.num_hidden_layers) != 36:
            raise RuntimeError("Unexpected LLM architecture; V1.8 expects 36 layers")
        self._setup_placeholders()
        self._inject_lora()

        self.pose_encoder = self._build_pose_encoder()
        hidden_size = int(self.llm.config.hidden_size)
        self.projector = SignProjector(768, hidden_size).to(
            self.device, dtype=TENSOR_DTYPE
        )
        self.adapter = TemporalAdapter().to(self.device, dtype=TENSOR_DTYPE)
        self.heads = FactorizedHeads(hidden_size).to(self.device, dtype=TENSOR_DTYPE)
        self._load_checkpoint()

        for module in (
            self.minicpm,
            self.llm,
            self.pose_encoder,
            self.projector,
            self.adapter,
            self.heads,
        ):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    def _setup_placeholders(self) -> None:
        base_vocab = getattr(self.tokenizer, "vocab_size", len(self.tokenizer))
        added = self.tokenizer.get_added_vocab()
        if added:
            base_vocab = max(base_vocab, max(added.values()) + 1)
        for name, text in (
            ("im_start", "<|im_start|>"),
            ("im_end", "<|im_end|>"),
            ("img_start", "<image>"),
            ("img_end", "</image>"),
        ):
            token_id = self.tokenizer.convert_tokens_to_ids(text)
            if token_id is None or token_id == self.tokenizer.unk_token_id or token_id >= base_vocab:
                raise RuntimeError(f"Required tokenizer token is missing: {text}")
            self.tokens[name] = int(token_id)
        placeholder = self.tokenizer.convert_tokens_to_ids("<unk>")
        if placeholder is None or placeholder >= base_vocab:
            placeholder = self.tokenizer.convert_tokens_to_ids("<|vision_pad|>")
        self.tokens["placeholder"] = int(placeholder)

    def _inject_lora(self) -> None:
        layers = self.llm.model.layers
        for layer_index in LORA_LAYERS:
            attention = layers[layer_index].self_attn
            for target in LORA_TARGETS:
                base = getattr(attention, target)
                if not isinstance(base, nn.Linear):
                    raise RuntimeError(f"Unexpected LoRA target: layer {layer_index}/{target}")
                setattr(attention, target, LoRALinear(base).to(self.device))

    def _build_pose_encoder(self) -> PoseEncoder:
        previous_cwd = os.getcwd()
        demo_config_module = sys.modules.get("config")
        try:
            os.chdir(UNISIGN_DIR)
            unisign_path = str(UNISIGN_DIR)
            if unisign_path in sys.path:
                sys.path.remove(unisign_path)
            sys.path.insert(0, unisign_path)
            # Uni-Sign's historical `models.py` uses `from config import ...`.
            # Temporarily expose its own config instead of this demo's config.
            sys.modules.pop("config", None)
            from models import Uni_Sign

            args = Namespace(
                hidden_dim=256,
                dataset="CSL_Daily",
                rgb_support=False,
                max_length=MAX_POSE_LEN,
                label_smoothing=0.2,
            )
            full_model = Uni_Sign(args)
            checkpoint = torch.load(UNISIGN_CHECKPOINT, map_location="cpu", weights_only=False)
            missing, _ = full_model.load_state_dict(checkpoint["model"], strict=False)
            pose_missing = [name for name in missing if "mt5" not in name]
            if pose_missing:
                raise RuntimeError(f"Uni-Sign pose weights missing: {pose_missing[:8]}")
            encoder = PoseEncoder(full_model).to(self.device, dtype=POSE_DTYPE)
            del full_model, checkpoint
            return encoder
        finally:
            sys.modules.pop("config", None)
            if demo_config_module is not None:
                sys.modules["config"] = demo_config_module
            os.chdir(previous_cwd)

    def _load_checkpoint(self) -> None:
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        self.projector.load_state_dict(checkpoint["sign_projector"])
        self.adapter.load_state_dict(checkpoint["temporal_adapter"])
        self.heads.load_state_dict(checkpoint["factorized_heads"])
        llm_parameters = dict(self.llm.named_parameters())
        pose_parameters = dict(self.pose_encoder.named_parameters())
        with torch.no_grad():
            for name, value in checkpoint["lora"].items():
                parameter = llm_parameters[name]
                parameter.copy_(value.to(parameter.device, dtype=parameter.dtype))
            for name, value in checkpoint["unisign_unfrozen"].items():
                parameter = pose_parameters[name]
                parameter.copy_(value.to(parameter.device, dtype=parameter.dtype))
        self.checkpoint_epoch = int(checkpoint["epoch"])
        del checkpoint

    def _build_sequence(self, rgb_count: int, pose_count: int, system_state: str):
        def encode(text: str) -> list[int]:
            return self.tokenizer.encode(text, add_special_tokens=False)

        im_start = [self.tokens["im_start"]]
        im_end = [self.tokens["im_end"]]
        newline = encode("\n")
        placeholder = self.tokens["placeholder"]
        task = TASK_BY_STATE[system_state]

        ids = im_start + encode("system") + newline + encode(SYSTEM_TEXT) + im_end + newline
        ids += im_start + encode("user") + newline
        ids += encode(f"已知系统状态：{system_state}。") + newline
        if task == "turn_state":
            ids += encode(
                "路由：Turn-State Head，类别为 IDLE / INCOMPLETE / BACKCHANNEL / COMPLETE。"
            ) + newline
        else:
            ids += encode("路由：Barge-In Head，类别为 BARGE / NO_BARGE。") + newline
        ids += encode("手语视频：")
        rgb_slots: list[int] = []
        for _ in range(rgb_count):
            ids += [self.tokens["img_start"]]
            rgb_slots.extend(range(len(ids), len(ids) + 64))
            ids += [placeholder] * 64 + [self.tokens["img_end"]]
        ids += newline + encode("手语骨架：")
        pose_slots = list(range(len(ids), len(ids) + pose_count))
        ids += [placeholder] * pose_count + newline
        ids += encode("输出该任务的唯一类别：") + im_end + newline
        ids += im_start + encode("assistant") + newline
        return ids, rgb_slots, pose_slots, len(ids) - 1

    @torch.inference_mode()
    def predict(
        self,
        bgr_frames: list[np.ndarray],
        pose_data: dict[str, list[np.ndarray]],
        system_state: str,
    ) -> dict[str, Any]:
        if system_state not in TASK_BY_STATE:
            raise KeyError(system_state)
        self._synchronize()
        started = time.perf_counter()
        stage_started = started
        pixel_values, target_sizes = self._prepare_rgb(bgr_frames)
        rgb_preprocess_ms = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        keypoints = prepare_pose_prefix(pose_data)
        pose_preprocess_ms = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        rgb = self._rgb_embeddings(pixel_values, target_sizes)
        self._synchronize()
        rgb_encoder_ms = (time.perf_counter() - stage_started) * 1000.0

        source = {
            key: value.unsqueeze(0).to(self.device, dtype=POSE_DTYPE)
            for key, value in keypoints.items()
        }
        stage_started = time.perf_counter()
        pose_features = self.pose_encoder(source)
        self._synchronize()
        unisign_encoder_ms = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        pose = self.adapter(pose_features)
        pose = self.projector(pose).to(DTYPE)[0]
        self._synchronize()
        adapter_projector_ms = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        ids, rgb_slots, pose_slots, prediction_position = self._build_sequence(
            len(bgr_frames), pose.shape[0], system_state
        )
        if len(rgb_slots) != rgb.shape[0]:
            raise RuntimeError(f"RGB embedding mismatch: {len(rgb_slots)} != {rgb.shape[0]}")
        ids_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        embeddings = self.llm.model.embed_tokens(ids_tensor).to(DTYPE).clone()
        embeddings[0, rgb_slots] = rgb
        embeddings[0, pose_slots] = pose
        attention = torch.ones(1, embeddings.shape[1], dtype=torch.long, device=self.device)
        hidden = self.llm.model(
            inputs_embeds=embeddings, attention_mask=attention
        ).last_hidden_state
        logits = self.heads(hidden[0, prediction_position], system_state)
        self._synchronize()
        classifier_ms = (time.perf_counter() - stage_started) * 1000.0

        stage_started = time.perf_counter()
        probabilities_tensor = torch.softmax(logits.float(), dim=-1).cpu()
        labels = TURN_LABELS if system_state == "LISTENING" else BARGE_LABELS
        probabilities = {
            label: float(probabilities_tensor[index])
            for index, label in enumerate(labels)
        }
        predicted = labels[int(torch.argmax(probabilities_tensor).item())]
        postprocess_ms = (time.perf_counter() - stage_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        return {
            "predicted_class": predicted,
            "confidence": probabilities[predicted],
            "probabilities": probabilities,
            "model_latency_ms": total_ms,
            "profile_ms": {
                "video_preprocess": rgb_preprocess_ms + rgb_encoder_ms,
                "rgb_cpu_preprocess": rgb_preprocess_ms,
                "rgb_encoder": rgb_encoder_ms,
                "pose_preprocess": pose_preprocess_ms,
                "unisign_encoder": unisign_encoder_ms,
                "temporal_adapter_projector": adapter_projector_ms,
                "classifier": classifier_ms,
                "postprocess": postprocess_ms,
                "model_total": total_ms,
            },
        }

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _prepare_rgb(self, bgr_frames: list[np.ndarray]):
        images = [
            Image.fromarray(frame[:, :, ::-1]).convert("RGB").resize(
                (448, 448), Image.Resampling.BICUBIC
            )
            for frame in bgr_frames
        ]
        output = self.image_processor(
            [images], do_pad=True, max_slice_nums=1, return_tensors="pt"
        )
        return output["pixel_values"][0], output["tgt_sizes"][0]

    def _rgb_embeddings(self, pixel_values, target_sizes):
        data = {
            "pixel_values": [[value.to(self.device, dtype=DTYPE) for value in pixel_values]],
            "tgt_sizes": [target_sizes.to(self.device)],
        }
        vision = self.minicpm.get_vision_embedding(data)[0]
        return vision.reshape(-1, vision.shape[-1]).to(DTYPE)

    def _pose_embeddings(self, keypoints):
        source = {
            key: value.unsqueeze(0).to(self.device, dtype=POSE_DTYPE)
            for key, value in keypoints.items()
        }
        feature = self.pose_encoder(source)
        feature = self.adapter(feature)
        return self.projector(feature).to(DTYPE)[0]


def prepare_pose_prefix(pose_data: dict[str, list[np.ndarray]]) -> dict[str, torch.Tensor]:
    """Exact training-time prefix selection and Uni-Sign part normalization."""
    total = min(len(pose_data["keypoints"]), len(pose_data["scores"]))
    if total <= 0:
        raise ValueError("The causal pose prefix is empty")
    indices = uniform_indices(total, MAX_POSE_LEN)
    skeletons = [pose_data["keypoints"][index] for index in indices]
    confidences = [pose_data["scores"][index] for index in indices]
    return load_part_keypoints(skeletons, confidences)


def _crop_scale(motion: np.ndarray, threshold: float = 0.3):
    result = copy.deepcopy(motion)
    valid = motion[motion[..., 2] > threshold][:, :2]
    if len(valid) < 4:
        return np.zeros(motion.shape), 0.0
    xmin, xmax = min(valid[:, 0]), max(valid[:, 0])
    ymin, ymax = min(valid[:, 1]), max(valid[:, 1])
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(motion.shape), 0.0
    x_start = (xmin + xmax - scale) / 2
    y_start = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [x_start, y_start]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    result[result[..., 2] <= threshold] = 0
    return result, float(scale)


def load_part_keypoints(skeletons, confidences) -> dict[str, torch.Tensor]:
    threshold = 0.3
    output: dict[str, torch.Tensor] = {}
    scale: float | None = None
    for part in ("body", "left", "right", "face_all"):
        points, scores = [], []
        for skeleton, confidence in zip(skeletons, confidences):
            skeleton = skeleton[0]
            confidence = confidence[0]
            if part == "body":
                indices = [0] + list(range(3, 11))
                point, score = skeleton[indices, :], confidence[indices]
            elif part == "left":
                point = skeleton[91:112, :] - skeleton[91, :]
                score = confidence[91:112]
            elif part == "right":
                point = skeleton[112:133, :] - skeleton[112, :]
                score = confidence[112:133]
            else:
                indices = [i for i in list(range(23, 40))[::2]] + list(range(83, 91)) + [53]
                point = skeleton[indices, :] - skeleton[53, :]
                score = confidence[indices]
            points.append(point)
            scores.append(score)
        points_array = np.stack(points, 0)
        scores_array = np.stack(scores, 0)
        if part == "body":
            normalized, scale = _crop_scale(
                np.concatenate([points_array, scores_array[..., None]], -1), threshold
            )
        else:
            normalized = np.concatenate([points_array, scores_array[..., None]], -1)
            if not scale:
                normalized = np.zeros(normalized.shape)
            else:
                normalized[..., :2] = normalized[..., :2] / scale
                normalized = np.clip(normalized, -1, 1)
                normalized[normalized[..., 2] <= threshold] = 0
        output[part] = torch.tensor(normalized, dtype=torch.float32)
    return output
