"""Reference-annotation subtitle metadata for the Gradio display only."""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


SUBTITLE_SOURCE = "REFERENCE_ANNOTATION"
SUBTITLE_LEARNED = False
CONCAT_METADATA_DIR = Path(
    os.environ.get("SIGNDUPLEX_CONCAT_METADATA_DIR", "metadata")
).expanduser()
CSL_DAILY_ANNOTATIONS = Path(
    os.environ.get("SIGNDUPLEX_ANNOTATIONS", "data/CSL_Daily/labels.json")
).expanduser()


@dataclass(frozen=True)
class ReferenceSubtitleSegment:
    segment_index: int
    source_video: str
    semantic_id: str
    start_time: float
    end_time: float
    text: str

    def as_record(self) -> dict[str, object]:
        return asdict(self)


class SubtitleTimelineManager:
    """Map playback time to reference text without any inference dependency."""

    def __init__(
        self,
        segments: tuple[ReferenceSubtitleSegment, ...] = (),
        *,
        metadata_source: Path | None = None,
        annotation_source: Path | None = None,
    ):
        self.segments = segments
        self.metadata_source = metadata_source
        self.annotation_source = annotation_source
        previous_end = 0.0
        for segment in segments:
            if segment.start_time < previous_end - 1e-9:
                raise ValueError("Subtitle segments overlap or are out of order")
            if segment.end_time <= segment.start_time:
                raise ValueError("Subtitle segment duration must be positive")
            if not segment.text.strip():
                raise ValueError("Reference subtitle text must not be empty")
            previous_end = segment.end_time

    @property
    def enabled(self) -> bool:
        return bool(self.segments)

    @classmethod
    def from_video_path(cls, video_path: str | Path) -> SubtitleTimelineManager:
        match = re.search(r"(concat_\d{3})", Path(video_path).stem)
        if match is None:
            return cls()
        metadata_path = CONCAT_METADATA_DIR / f"{match.group(1)}.json"
        if not metadata_path.is_file():
            return cls()

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        annotations = json.loads(
            CSL_DAILY_ANNOTATIONS.read_text(encoding="utf-8")
        )
        cursor = 0.0
        segments: list[ReferenceSubtitleSegment] = []
        for zero_index, source in enumerate(metadata["source_videos"]):
            source_name = str(source["name"])
            annotation = annotations[source_name]
            reference_text = str(annotation["text"])
            if reference_text != str(source["text"]):
                raise ValueError(
                    f"Concat/reference annotation mismatch for {source_name}"
                )
            start_time = round(cursor, 6)
            end_time = round(start_time + float(source["duration"]), 6)
            segments.append(
                ReferenceSubtitleSegment(
                    segment_index=zero_index,
                    source_video=str(source["original_path"]),
                    semantic_id=str(source["sentence_id"]),
                    start_time=start_time,
                    end_time=end_time,
                    text=reference_text,
                )
            )
            cursor = round(
                end_time + float(source.get("pause_after", 0.0)), 6
            )
        return cls(
            tuple(segments),
            metadata_source=metadata_path,
            annotation_source=CSL_DAILY_ANNOTATIONS,
        )

    def segment_at(self, video_time: float) -> ReferenceSubtitleSegment | None:
        for segment in self.segments:
            if segment.start_time <= video_time < segment.end_time:
                return segment
        return None

    def get_subtitle(self, video_time: float) -> str:
        segment = self.segment_at(video_time)
        return segment.text if segment is not None else ""

    def as_records(self) -> list[dict[str, object]]:
        return [segment.as_record() for segment in self.segments]


class SubtitleTimelineRegistry:
    """Single-upload registry kept separate from prepared model inputs."""

    def __init__(self):
        self._items: dict[str, SubtitleTimelineManager] = {}
        self._lock = threading.Lock()

    def put(self, key: str, manager: SubtitleTimelineManager) -> None:
        with self._lock:
            self._items = {key: manager}

    def get(self, key: str) -> SubtitleTimelineManager:
        with self._lock:
            return self._items[key]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
