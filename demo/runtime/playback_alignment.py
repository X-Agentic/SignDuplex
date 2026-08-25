"""Small P95-based playback buffer helpers for V2.3."""

from __future__ import annotations

from config import (
    DISPLAY_DELAY_FALLBACK_LATENCY_SECONDS,
    DISPLAY_DELAY_MARGIN_SECONDS,
    DISPLAY_DELAY_MAX_SECONDS,
    DISPLAY_DELAY_MIN_SECONDS,
)


def percentile_95(values: list[float]) -> float:
    """Linearly interpolated P95 without over-weighting one short-run outlier."""
    if not values:
        raise ValueError("P95 requires at least one sample")
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def display_delay_from_latencies(
    recent_latency_ms: list[float],
) -> tuple[float, float]:
    """Return clamped delay seconds and the latency P95 used to derive it."""
    p95_seconds = (
        percentile_95(recent_latency_ms) / 1000.0
        if recent_latency_ms
        else DISPLAY_DELAY_FALLBACK_LATENCY_SECONDS
    )
    delay = min(
        DISPLAY_DELAY_MAX_SECONDS,
        max(
            DISPLAY_DELAY_MIN_SECONDS,
            p95_seconds + DISPLAY_DELAY_MARGIN_SECONDS,
        ),
    )
    return delay, p95_seconds
