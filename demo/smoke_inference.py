"""Run one real causal-prefix inference outside Gradio."""

from __future__ import annotations

import argparse
import time

from config import DEVICE, RGB_FRAMES
from runtime.model_runtime import ModelRuntime
from runtime.pose_runtime import PoseRuntime
from runtime.video_stream import VideoStream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--state", choices=("LISTENING", "SPEAKING"), default="LISTENING")
    parser.add_argument("--prefix", type=float, default=0.5)
    args = parser.parse_args()

    stream = VideoStream(args.video)
    try:
        frames = stream.advance_to(args.prefix)
        pose = PoseRuntime(device="cuda" if DEVICE.startswith("cuda") else "cpu")
        started = time.perf_counter()
        pose.process_new(frames)
        model = ModelRuntime(DEVICE)
        result = model.predict(stream.rgb_sample(RGB_FRAMES), pose.prefix(), args.state)
        result["end_to_end_seconds"] = time.perf_counter() - started
        result["prefix_end_frame_exclusive"] = stream.prefix_end_frame
        print(result)
    finally:
        stream.close()


if __name__ == "__main__":
    main()
