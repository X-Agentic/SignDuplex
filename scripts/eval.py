#!/usr/bin/env python3
"""Run one real causal-prefix SignDuplex V1.8 inference evaluation."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_ROOT = REPO_ROOT / "demo"
EVAL_ENTRY = DEMO_ROOT / "smoke_inference.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "signduplex_v1_8.pt")
    parser.add_argument("--minicpm-dir", type=Path, required=True)
    parser.add_argument("--unisign-dir", type=Path, required=True)
    parser.add_argument("--state", choices=("LISTENING", "SPEAKING"), default="LISTENING")
    parser.add_argument("--prefix", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.update(
        {
            "SIGNDUPLEX_CHECKPOINT": str(args.checkpoint),
            "SIGNDUPLEX_MINICPM_DIR": str(args.minicpm_dir),
            "SIGNDUPLEX_UNISIGN_DIR": str(args.unisign_dir),
            "SIGNDUPLEX_DEVICE": args.device,
        }
    )
    sys.path.insert(0, str(DEMO_ROOT))
    sys.argv = [
        str(EVAL_ENTRY),
        str(args.video),
        "--state",
        args.state,
        "--prefix",
        str(args.prefix),
    ]
    runpy.run_path(str(EVAL_ENTRY), run_name="__main__")


if __name__ == "__main__":
    main()
