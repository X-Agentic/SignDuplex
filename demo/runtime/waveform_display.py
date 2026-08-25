"""Deterministic display-only timing for accepted BACKCHANNEL feedback."""

from __future__ import annotations

import random
from dataclasses import dataclass


BACKCHANNEL_FEEDBACK_DURATION_RANGE = (0.8, 1.2)
_BACKCHANNEL_SEED_SALT = 0x53444734


@dataclass(frozen=True)
class BackchannelFeedbackPlan:
    sequence: int
    duration: float


class BackchannelFeedbackSampler:
    """Use an isolated seeded RNG so TTS reply sampling remains unchanged."""

    def __init__(self, random_seed: int) -> None:
        self.random_seed = int(random_seed)
        self._random = random.Random(self.random_seed ^ _BACKCHANNEL_SEED_SALT)
        self._sequence = 0

    def sample(self) -> BackchannelFeedbackPlan:
        self._sequence += 1
        duration = self._random.uniform(*BACKCHANNEL_FEEDBACK_DURATION_RANGE)
        return BackchannelFeedbackPlan(self._sequence, duration)
