"""SignDuplex V1.8 state-conditioned full-duplex interaction code."""

__version__ = "1.8"

TURN_LABELS = ("IDLE", "INCOMPLETE", "BACKCHANNEL", "COMPLETE")
BARGE_LABELS = ("BARGE", "NO_BARGE")
LABELS_BY_STATE = {
    "LISTENING": TURN_LABELS,
    "SPEAKING": BARGE_LABELS,
}
