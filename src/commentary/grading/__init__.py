"""Grading. Nothing in here is ever imported by the runtime.

The play-by-play feed and the human commentator's transcript live on this
side of the wall and only on this side. That is the project's central claim
made structural rather than promised.
"""

from commentary.grading.captions import load_json3
from commentary.grading.feed import Alignment, Feed, FeedEvent, align, load_feed, shift
from commentary.grading.judge import (
    FactualityReport,
    JudgeReport,
    PairwiseReport,
    Verdict,
    judge_factuality,
    judge_pairwise,
    judge_run,
)
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
from commentary.grading.transcripts import Transcript, segments_near, transcribe

__all__ = [
    "Alignment",
    "FactualError",
    "FactualityReport",
    "Feed",
    "FeedEvent",
    "JudgeReport",
    "Lag",
    "PairwiseReport",
    "Recall",
    "Run",
    "Scorecard",
    "SpokenLine",
    "Transcript",
    "Verdict",
    "align",
    "detail",
    "event_recall",
    "factual_errors",
    "gate_rejection_rate",
    "gate_table",
    "judge_factuality",
    "judge_pairwise",
    "judge_run",
    "lag",
    "load_feed",
    "load_json3",
    "load_run",
    "repetition_rate",
    "score",
    "segments_near",
    "shift",
    "silence_ratio",
    "table",
    "transcribe",
]
