"""Runtime configuration for the SignDuplex V3.7.0 compact paper demo."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
ASSETS_DIR = ROOT / "assets"
LOGS_DIR = Path(os.environ.get("SIGNDUPLEX_LOGS_DIR", REPO_ROOT / "logs")).expanduser()

TRAIN_SCRIPT = REPO_ROOT / "signduplex" / "train_v1_8.py"
BASE_TRAIN_SCRIPT = REPO_ROOT / "signduplex" / "training_base.py"
CHECKPOINT = Path(
    os.environ.get("SIGNDUPLEX_CHECKPOINT", REPO_ROOT / "checkpoints" / "signduplex_v1_8.pt")
).expanduser()
MINICPM_DIR = Path(
    os.environ.get("SIGNDUPLEX_MINICPM_DIR", REPO_ROOT / "models" / "MiniCPM-o-4_5")
).expanduser()
UNISIGN_DIR = Path(
    os.environ.get("SIGNDUPLEX_UNISIGN_DIR", REPO_ROOT / "models" / "Uni-Sign")
).expanduser()
UNISIGN_CHECKPOINT = UNISIGN_DIR / "hf_assets/csl_daily_pose_only_slt.pth"
RTMLIB_DIR = UNISIGN_DIR / "demo/rtmlib-main"

SYSTEM_AUDIO = ASSETS_DIR / "system_speaking.wav"
CHUNK_SIZE_SECONDS = 1.0
DECISION_INTERVAL_SECONDS = 1.0
FINAL_REPLY_EXTENSION_SECONDS = 3.0
POST_EOS_DISPLAY_HOLD_SECONDS = 0.0
DISPLAY_DELAY_MIN_SECONDS = 0.30
DISPLAY_DELAY_MAX_SECONDS = 0.80
DISPLAY_DELAY_MARGIN_SECONDS = 0.075
DISPLAY_DELAY_FALLBACK_LATENCY_SECONDS = 0.50
ALIGNMENT_LATENCY_WINDOW = 32
RGB_FRAMES = 8
MAX_POSE_LEN = 300
POSE_MODE = os.environ.get("SIGNDUPLEX_POSE_MODE", "lightweight")
POSE_BACKEND = os.environ.get("SIGNDUPLEX_POSE_BACKEND", "onnxruntime")
POSE_WORKERS = int(os.environ.get("SIGNDUPLEX_POSE_WORKERS", "48"))
DEVICE = os.environ.get("SIGNDUPLEX_DEVICE", "cuda:0")
HOST = os.environ.get("SIGNDUPLEX_HOST", "0.0.0.0")
PORT = int(os.environ.get("SIGNDUPLEX_PORT", "7863"))

TURN_LABELS = ("IDLE", "INCOMPLETE", "BACKCHANNEL", "COMPLETE")
BARGE_LABELS = ("BARGE", "NO_BARGE")
LABELS_BY_STATE = {
    "LISTENING": TURN_LABELS,
    "SPEAKING": BARGE_LABELS,
}
TASK_BY_STATE = {
    "LISTENING": "turn_state",
    "SPEAKING": "barge_in",
}


def validate_paths() -> None:
    required = (
        CHECKPOINT,
        MINICPM_DIR,
        UNISIGN_DIR,
        UNISIGN_CHECKPOINT,
        RTMLIB_DIR,
        SYSTEM_AUDIO,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required assets: " + ", ".join(missing))


ASSETS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
