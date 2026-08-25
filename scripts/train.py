#!/usr/bin/env python3
"""Portable launcher for the real SignDuplex V1.8 training entry point."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_ENTRY = REPO_ROOT / "signduplex" / "train_v1_8.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "Sign_Aware_Raw_Screen_v1_8")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "train_v1_8")
    parser.add_argument("--minicpm-dir", type=Path, required=True)
    parser.add_argument("--unisign-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--physical-gpu", default="0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--rgb-frames", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.update(
        {
            "SIGNDUPLEX_DATA_DIR": str(args.data_dir),
            "SIGNDUPLEX_OUTPUT_DIR": str(args.output_dir),
            "SIGNDUPLEX_MINICPM_DIR": str(args.minicpm_dir),
            "SIGNDUPLEX_UNISIGN_DIR": str(args.unisign_dir),
            "SIGNDUPLEX_WARM_START": str(args.warm_start),
            "SIGNDUPLEX_PHYSICAL_GPU": str(args.physical_gpu),
        }
    )
    sys.argv = [
        str(TRAIN_ENTRY),
        "--epochs",
        str(args.epochs),
        "--rgb_frames",
        str(args.rgb_frames),
        "--workers",
        str(args.workers),
        "--smoke",
        str(args.smoke),
    ]
    runpy.run_path(str(TRAIN_ENTRY), run_name="__main__")


if __name__ == "__main__":
    main()
