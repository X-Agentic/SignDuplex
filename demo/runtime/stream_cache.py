"""Upload-time video/Pose cache with strict causal-prefix accessors."""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import (
    DECISION_INTERVAL_SECONDS,
    POSE_WORKERS,
    RGB_FRAMES,
)
from runtime.pose_runtime import PoseRuntime
from runtime.video_stream import VideoStream, uniform_indices


@dataclass(frozen=True)
class DecisionBoundary:
    index: int
    end_seconds: float
    prefix_end_frame: int


@dataclass
class PreparedVideo:
    video_id: str
    path: str
    fps: float
    duration: float
    total_frames: int
    decisions: tuple[DecisionBoundary, ...]
    frames_by_index: dict[int, np.ndarray]
    pose_keypoints: list[np.ndarray]
    pose_scores: list[np.ndarray]
    pose_views_by_end: dict[int, np.ndarray]
    decode_ms: float
    pose_extraction_ms: float
    pose_workers: int

    def rgb_source_indices(
        self, prefix_end_frame: int, prefix_start_frame: int = 0
    ) -> list[int]:
        if not 0 <= prefix_start_frame < prefix_end_frame <= self.total_frames:
            raise ValueError(
                "Invalid causal task prefix: "
                f"[{prefix_start_frame}, {prefix_end_frame})"
            )
        return [
            prefix_start_frame + index
            for index in uniform_indices(
                prefix_end_frame - prefix_start_frame, RGB_FRAMES
            )
        ]

    def rgb_sample(
        self, prefix_end_frame: int, prefix_start_frame: int = 0
    ) -> list[np.ndarray]:
        indices = self.rgb_source_indices(
            prefix_end_frame, prefix_start_frame
        )
        missing = [index for index in indices if index not in self.frames_by_index]
        if missing:
            capture = cv2.VideoCapture(self.path)
            try:
                if not capture.isOpened():
                    raise RuntimeError(f"Cannot reopen cached video: {self.path}")
                for index in sorted(set(missing)):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"Cannot decode cached frame {index}")
                    self.frames_by_index[index] = frame
            finally:
                capture.release()
        return [self.frames_by_index[index] for index in indices]

    def rgb_view(self, prefix_end_frame: int) -> np.ndarray:
        return self.frames_by_index[prefix_end_frame - 1]

    def pose_view(self, prefix_end_frame: int) -> np.ndarray:
        return self.pose_views_by_end[prefix_end_frame]

    def pose_prefix(
        self, prefix_end_frame: int, prefix_start_frame: int = 0
    ) -> dict[str, list[np.ndarray]]:
        if not 0 <= prefix_start_frame < prefix_end_frame <= self.total_frames:
            raise ValueError(
                "Invalid causal task prefix: "
                f"[{prefix_start_frame}, {prefix_end_frame})"
            )
        return {
            "keypoints": self.pose_keypoints[
                prefix_start_frame:prefix_end_frame
            ],
            "scores": self.pose_scores[prefix_start_frame:prefix_end_frame],
        }


def prepare_video(video_path: str, pose_device: str, video_id: str) -> PreparedVideo:
    stream = VideoStream(video_path)
    try:
        decode_started = time.perf_counter()
        all_frames = stream.advance_to(stream.duration)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        if len(all_frames) != stream.total_frames:
            raise RuntimeError(
                f"Video decode mismatch: {len(all_frames)} != {stream.total_frames}"
            )
        decisions = build_decisions(
            stream.duration,
            stream.fps,
            stream.total_frames,
            DECISION_INTERVAL_SECONDS,
        )

        pose = PoseRuntime(device=pose_device)
        pose_started = time.perf_counter()
        pose.process_all_parallel(all_frames, max_workers=POSE_WORKERS)
        pose_extraction_ms = (time.perf_counter() - pose_started) * 1000.0
        if len(pose.keypoints) != stream.total_frames:
            raise RuntimeError("Pose cache frame count mismatch")

        retained_indices: set[int] = set()
        for decision in decisions:
            retained_indices.update(
                uniform_indices(decision.prefix_end_frame, RGB_FRAMES)
            )
            retained_indices.add(decision.prefix_end_frame - 1)
        frames_by_index = {index: all_frames[index] for index in retained_indices}
        pose_views = {
            decision.prefix_end_frame: pose.render(
                all_frames[decision.prefix_end_frame - 1],
                decision.prefix_end_frame - 1,
            )
            for decision in decisions
        }
        return PreparedVideo(
            video_id=video_id,
            path=str(Path(video_path).resolve()),
            fps=stream.fps,
            duration=stream.duration,
            total_frames=stream.total_frames,
            decisions=decisions,
            frames_by_index=frames_by_index,
            pose_keypoints=pose.keypoints,
            pose_scores=pose.scores,
            pose_views_by_end=pose_views,
            decode_ms=decode_ms,
            pose_extraction_ms=pose_extraction_ms,
            pose_workers=min(POSE_WORKERS, stream.total_frames),
        )
    finally:
        stream.close()


def build_decisions(
    duration: float,
    fps: float,
    total_frames: int,
    decision_interval: float = DECISION_INTERVAL_SECONDS,
) -> tuple[DecisionBoundary, ...]:
    """Return interval decision points plus the exact real video endpoint."""
    if duration <= 0 or total_frames <= 0:
        raise ValueError("Video has no usable duration")
    if decision_interval != DECISION_INTERVAL_SECONDS:
        raise ValueError(
            f"Decision interval is fixed at {DECISION_INTERVAL_SECONDS}"
        )
    ends: list[float] = []
    whole_strides = int(math.floor(duration / decision_interval))
    ends.extend(
        (index + 1) * decision_interval for index in range(whole_strides)
    )
    if not ends or duration - ends[-1] > 1e-6:
        ends.append(duration)
    else:
        ends[-1] = duration
    decisions = []
    for index, end in enumerate(ends, 1):
        prefix_end = min(total_frames, max(1, int(math.ceil(end * fps - 1e-9))))
        decisions.append(DecisionBoundary(index, end, prefix_end))
    if decisions[-1].prefix_end_frame != total_frames:
        last = decisions[-1]
        decisions[-1] = DecisionBoundary(
            last.index, duration, total_frames
        )
    return tuple(decisions)


class PreparedVideoRegistry:
    """Small bounded in-process cache keyed by opaque per-upload IDs."""

    def __init__(self, capacity: int = 2):
        self.capacity = capacity
        self._items: OrderedDict[str, PreparedVideo] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, prepared: PreparedVideo) -> str:
        key = prepared.video_id
        with self._lock:
            self._items[key] = prepared
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return key

    def get(self, key: str) -> PreparedVideo:
        with self._lock:
            prepared = self._items[key]
            self._items.move_to_end(key)
            return prepared

    def clear(self) -> None:
        """Drop every RGB/Pose cache reference before preparing a new upload."""
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
