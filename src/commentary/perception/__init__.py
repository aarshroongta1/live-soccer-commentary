"""Turning pixels into claims: what the system can see for itself."""

from commentary.perception.board import (
    BOARD_SYSTEM,
    BoardChange,
    BoardPending,
    BoardReader,
    BoardTracker,
    crop_score_bug,
)
from commentary.perception.players import (
    Detector,
    NullTracker,
    PlayerTracker,
    Track,
    Tracker,
    VisionExtraMissing,
    default_tracker,
)

__all__ = [
    "BOARD_SYSTEM",
    "BoardChange",
    "BoardPending",
    "BoardReader",
    "BoardTracker",
    "Detector",
    "NullTracker",
    "PlayerTracker",
    "Track",
    "Tracker",
    "VisionExtraMissing",
    "crop_score_bug",
    "default_tracker",
]
