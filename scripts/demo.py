#!/usr/bin/env python3
"""Launch the real SignDuplex V3.7.0 Gradio demo with portable paths."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_ROOT = REPO_ROOT / "demo"
DEMO_ENTRY = DEMO_ROOT / "app.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "signduplex_v1_8.pt")
    parser.add_argument("--minicpm-dir", type=Path, required=True)
    parser.add_argument("--unisign-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.update(
        {
            "SIGNDUPLEX_CHECKPOINT": str(args.checkpoint),
            "SIGNDUPLEX_MINICPM_DIR": str(args.minicpm_dir),
            "SIGNDUPLEX_UNISIGN_DIR": str(args.unisign_dir),
            "SIGNDUPLEX_DEVICE": args.device,
            "SIGNDUPLEX_HOST": args.host,
            "SIGNDUPLEX_PORT": str(args.port),
        }
    )
    sys.path.insert(0, str(DEMO_ROOT))
    sys.argv = [str(DEMO_ENTRY), "--random-seed", str(args.random_seed)]
    runpy.run_path(str(DEMO_ENTRY), run_name="__main__")


if __name__ == "__main__":
    main()
