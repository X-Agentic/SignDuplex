# SignDuplex
Real-Time Sign-Aware Full-Duplex Interaction

## Overview

SignDuplex is a real-time sign-aware turn-taking system built from the true V1.8 IDLE-aware model and the V3.7.0 interactive demo. It predicts from causal RGB/Pose prefixes at a fixed 1.0-second decision interval and keeps raw model prediction separate from controller policy.

## Architecture

The model combines the MiniCPM-o visual tower, Uni-Sign pose features, a learned causal Temporal Adapter, a Sign Projector, attention LoRA, and state-conditioned 4×2 factorized prediction heads. It is not a unified six-class softmax.

See [docs/architecture.md](docs/architecture.md) for the routed model contract.

## Interaction States

```text
LISTENING:
IDLE        -> KEEP_LISTENING
INCOMPLETE  -> KEEP_LISTENING
BACKCHANNEL -> SHORT_FEEDBACK
COMPLETE    -> SWITCH_TO_SPEAKING

SPEAKING:
NO_BARGE    -> CONTINUE_SPEAKING
BARGE       -> SWITCH_TO_LISTENING
```

The V3.7.0 controller retains Post-BARGE Protection, Backchannel Cooldown, `BALANCED_IDLE_BARGE`, simulated TTS waveform, short BACKCHANNEL feedback, immediate BARGE interruption, reference sentences, test-video display, and final-reply extension.

## Installation

Python 3.10 and a CUDA-capable PyTorch environment are recommended.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

MiniCPM-o-4.5 and Uni-Sign are external dependencies. Set their local directories with `SIGNDUPLEX_MINICPM_DIR` and `SIGNDUPLEX_UNISIGN_DIR`, or pass them to the launch scripts. Copy `configs/env.example` as a path reference; do not commit private assets.

## Weights

Place the V1.8 checkpoint at `checkpoints/signduplex_v1_8.pt`. The download link is to be released. Model weights are excluded from this repository; see [checkpoints/README.md](checkpoints/README.md).

## Training

The original V1.8 lifecycle runs eight epochs and preserves the validated backbone, causal inputs, Temporal Adapter, Projector, and factorized classifier heads.

```bash
python scripts/train.py --help
python scripts/train.py \
  --minicpm-dir /path/to/MiniCPM-o-4_5 \
  --unisign-dir /path/to/Uni-Sign \
  --warm-start /path/to/warm_start.pth \
  --data-dir /path/to/Sign_Aware_Raw_Screen_v1_8 \
  --epochs 8
```

Build the six-label V1.8 data contract with `python scripts/build_dataset.py --help`. Raw CSL-Daily and Uni-Sign data are not included.

## Evaluation

Run one real causal-prefix evaluation without Gradio:

```bash
python scripts/eval.py --help
python scripts/eval.py sample.mp4 \
  --minicpm-dir /path/to/MiniCPM-o-4_5 \
  --unisign-dir /path/to/Uni-Sign \
  --state LISTENING --prefix 1.0
```

## Demo

Launch the preserved V3.7.0 demo:

```bash
python scripts/demo.py --help
python scripts/demo.py \
  --minicpm-dir /path/to/MiniCPM-o-4_5 \
  --unisign-dir /path/to/Uni-Sign
```

## Repository Structure

```text
signduplex/   V1.8 trainer, factorized training logic, and validated base
demo/         V3.7.0 Gradio demo and Interaction Controller
scripts/      training, evaluation, demo, and data-construction entry points
configs/      portable path/environment example
docs/         architecture notes
checkpoints/  weight placement instructions only
```

## Citation

A formal citation will be added with the public paper release. Until then, please cite this repository and the specific commit used.

## Acknowledgements

This project builds on MiniCPM-o and Uni-Sign. Please follow their licenses and dataset terms. CSL-Daily source media, restricted datasets, external model weights, generated logs, and test videos are intentionally excluded.
