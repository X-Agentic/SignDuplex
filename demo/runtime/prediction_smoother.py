"""Light temporal smoothing without adding BARGE delay."""

from __future__ import annotations


class PredictionSmoother:
    def __init__(self, confirmations: int = 2):
        self.confirmations = confirmations
        self.state: str | None = None
        self.stable: str | None = None
        self.pending: str | None = None
        self.pending_count = 0

    def update(self, raw: str, system_state: str) -> str:
        if system_state != self.state:
            self.state = system_state
            self.stable = None
            self.pending = None
            self.pending_count = 0

        # BARGE is intentionally unsmoothed so the simulated output stops now.
        if system_state == "SPEAKING":
            self.stable = raw
            return raw

        if self.stable is None:
            self.stable = raw
            return raw
        if raw == self.stable:
            self.pending = None
            self.pending_count = 0
            return self.stable
        if raw != self.pending:
            self.pending = raw
            self.pending_count = 1
        else:
            self.pending_count += 1
        if self.pending_count >= self.confirmations:
            self.stable = raw
            self.pending = None
            self.pending_count = 0
        return self.stable
