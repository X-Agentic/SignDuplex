"""Sequential MP4 replay with an explicit causal-prefix cache."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


class VideoStream:
    """Decode each frame once and expose only frames decoded up to the clock."""

    def __init__(self, video_path: str | Path):
        self.path = str(video_path)
        self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            raise ValueError(f"Cannot open MP4: {self.path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.fps = 25.0
        self.total_frames = max(0, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.duration = self.total_frames / self.fps if self.total_frames else 0.0
        self.frames: list[np.ndarray] = []
        self.exhausted = False

    @property
    def prefix_end_frame(self) -> int:
        return len(self.frames)

    def advance_to(self, current_time: float) -> list[np.ndarray]:
        """Decode new frames in [old_end, ceil(current_time * fps))."""
        wanted = max(1, int(math.ceil(max(0.0, current_time) * self.fps)))
        if self.total_frames:
            wanted = min(wanted, self.total_frames)
        new_frames: list[np.ndarray] = []
        while len(self.frames) < wanted and not self.exhausted:
            ok, frame = self.capture.read()
            if not ok:
                self.exhausted = True
                break
            self.frames.append(frame)
            new_frames.append(frame)
        return new_frames

    def rgb_sample(self, count: int) -> list[np.ndarray]:
        if not self.frames:
            raise ValueError("The causal prefix has no decoded frames")
        indices = uniform_indices(len(self.frames), count)
        return [self.frames[index] for index in indices]

    def close(self) -> None:
        self.capture.release()


def uniform_indices(end: int, count: int) -> list[int]:
    """Exact copy of the deterministic training-time [0, end) sampler."""
    if end <= 0:
        return []
    if end <= count:
        return list(range(end))
    return sorted(
        set(int(round(i * (end - 1) / (count - 1))) for i in range(count))
    )
