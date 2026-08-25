"""Incremental RTMPose extraction and cached pose visualization."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from config import POSE_BACKEND, POSE_MODE, RTMLIB_DIR


class PoseRuntime:
    def __init__(self, device: str = "cuda"):
        if str(RTMLIB_DIR) not in sys.path:
            sys.path.insert(0, str(RTMLIB_DIR))
        from rtmlib import Wholebody, draw_skeleton

        self._draw_skeleton = draw_skeleton
        if POSE_BACKEND == "onnxruntime" and device == "cuda":
            import onnxruntime

            if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
                device = "cpu"
        self.device = device
        self.estimator = Wholebody(
            to_openpose=False,
            mode=POSE_MODE,
            backend=POSE_BACKEND,
            device=device,
        )
        self.keypoints: list[np.ndarray] = []
        self.scores: list[np.ndarray] = []
        self.last_visualization: np.ndarray | None = None

    def process_new(self, frames: list[np.ndarray]) -> None:
        """Run RTMPose exactly once for every newly visible RGB frame."""
        for frame in frames:
            keypoints, scores = self._estimate(frame)
            self._append(frame, keypoints, scores)

    def process_all_parallel(
        self, frames: list[np.ndarray], max_workers: int = 48
    ) -> None:
        """Extract independent per-frame poses once, retaining original order."""
        if self.keypoints or self.scores:
            raise RuntimeError("Parallel pose precompute requires an empty cache")
        workers = max(1, min(int(max_workers), len(frames)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(self._estimate, frames))
        for frame, (keypoints, scores) in zip(frames, results):
            self._append(frame, keypoints, scores, render=False)

    def _estimate(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keypoints, scores = self.estimator(np.uint8(frame))
        return _one_person_or_zeros(keypoints, scores)

    def _append(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        *,
        render: bool = True,
    ) -> None:
        height, width = frame.shape[:2]
        normalized = keypoints / np.array([width, height], dtype=np.float32)[None, None]
        self.keypoints.append(normalized.astype(np.float32, copy=False))
        self.scores.append(scores.astype(np.float32, copy=False))
        if render:
            self.last_visualization = self.render(frame, len(self.keypoints) - 1)

    def render(self, frame: np.ndarray, index: int) -> np.ndarray:
        height, width = frame.shape[:2]
        keypoints = self.keypoints[index] * np.array(
            [width, height], dtype=np.float32
        )[None, None]
        canvas = np.zeros_like(frame)
        return self._draw_skeleton(
            canvas,
            keypoints,
            self.scores[index],
            openpose_skeleton=False,
            kpt_thr=0.3,
            radius=2,
            line_width=2,
        )

    def prefix(self, end: int | None = None) -> dict[str, list[np.ndarray]]:
        return {"keypoints": self.keypoints[:end], "scores": self.scores[:end]}


def _one_person_or_zeros(keypoints, scores) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if keypoints.size == 0 or scores.size == 0:
        return np.zeros((1, 133, 2), np.float32), np.zeros((1, 133), np.float32)
    if keypoints.ndim == 2:
        keypoints = keypoints[None, ...]
    if scores.ndim == 1:
        scores = scores[None, ...]
    # Training pose files keep the detector's first instance.
    if keypoints.shape[1:] != (133, 2) or scores.shape[1:] != (133,):
        return np.zeros((1, 133, 2), np.float32), np.zeros((1, 133), np.float32)
    return keypoints[:1], scores[:1]
