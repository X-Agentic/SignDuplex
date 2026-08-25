"""Fast structural audit of the formal two-head checkpoint."""

from __future__ import annotations

import torch

from config import CHECKPOINT


def main() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    expected = {
        "turn_state.weight": (4, 4096),
        "turn_state.bias": (4,),
        "barge_in.weight": (2, 4096),
        "barge_in.bias": (2,),
    }
    actual = {
        name: tuple(value.shape)
        for name, value in checkpoint["factorized_heads"].items()
    }
    if actual != expected:
        raise RuntimeError(f"Factorized head mismatch: {actual}")
    required = {
        "sign_projector",
        "temporal_adapter",
        "lora",
        "unisign_unfrozen",
        "factorized_heads",
    }
    missing = required - checkpoint.keys()
    if missing:
        raise RuntimeError(f"Checkpoint sections missing: {sorted(missing)}")
    print(f"PASS: {CHECKPOINT}")
    print(f"best epoch: {checkpoint['epoch']}")
    print(f"heads: {actual}")


if __name__ == "__main__":
    main()
