"""Grading. Nothing in here is ever imported by the runtime.

The play-by-play feed and the human commentator's transcript live on this
side of the wall and only on this side. That is the project's central claim
made structural rather than promised.
"""

from commentary.grading.metrics import (
    FactualError,
    Lag,
    Recall,
    Run,
    SpokenLine,
    event_recall,
    factual_errors,
    gate_rejection_rate,
    gate_table,
    lag,
    load_run,
    repetition_rate,
    silence_ratio,
)
from commentary.grading.report import Scorecard, detail, score, table

__all__ = [
    "FactualError",
    "Lag",
    "Recall",
    "Run",
    "Scorecard",
    "SpokenLine",
    "detail",
    "event_recall",
    "factual_errors",
    "gate_rejection_rate",
    "gate_table",
    "lag",
    "load_run",
    "repetition_rate",
    "score",
    "silence_ratio",
    "table",
]
