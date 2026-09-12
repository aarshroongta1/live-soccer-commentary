"""One run, or several, as a table you can put in a README.

The comparison is the deliverable, not any single run. worldcupvoice's loop
is baseline one, and every change this project makes has to show up as a
number against it, including the changes that turn out not to help.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from commentary.grading import metrics
from commentary.schemas import GroundTruthEvent, KnowledgePack, WireEvent


@dataclass
class Scorecard:
    """Every headline number for one run."""

    name: str
    lines: int = 0
    factual_error_rate: float = 0.0
    errors_by_kind: dict[str, int] = field(default_factory=dict)
    event_recall: float = 0.0
    recall_by_event: dict[str, tuple[int, int]] = field(default_factory=dict)
    gate_rejection_rate: float = 0.0
    gate_reasons: dict[str, int] = field(default_factory=dict)
    lag_p50: float = 0.0
    lag_p95: float = 0.0
    silence_ratio: float = 0.0
    repetition_rate: float = 0.0
    #: What fraction of lines name a player, and how many of those names are
    #: the right one. The pair is the question the vision chain exists for.
    name_rate: float = 0.0
    name_precision: float = 0.0
    corrections: int = 0
    preempted: int = 0
    cost_usd: float = 0.0
    #: Seconds of match this run actually saw. Recall and silence mean
    #: nothing without it, because both are fractions of exactly this.
    watched_s: float = 0.0

    def row(self) -> str:
        return (
            f"| {self.name} | {self.lines} | {self.factual_error_rate:.1%} | "
            f"{self.event_recall:.0%} | {self.gate_rejection_rate:.1%} | "
            f"{self.lag_p50:.1f} / {self.lag_p95:.1f} | {self.silence_ratio:.0%} | "
            f"{self.repetition_rate:.1%} | {self.name_rate:.0%} | "
            f"{self.name_precision:.0%} | ${self.cost_usd:.2f} |"
        )


HEADER = (
    "| run | lines | factual err | recall | gate rej | lag p50/p95 s | silence | repeat "
    "| named | name prec | cost |\n"
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
)


def score(
    name: str,
    trace_path: Path,
    truth: list[GroundTruthEvent],
    pack: KnowledgePack,
    *,
    duration_s: float | None = None,
    wire_events: list[WireEvent] | None = None,
) -> Scorecard:
    """Every headline number for one trace.

    ``wire_events`` is the feed, and it is the only way to say whether a name
    in a line was the right one — so the two name columns are zero without it
    rather than guessed at.
    """
    run = metrics.load_run(trace_path)
    errors = metrics.factual_errors(run, truth, pack)
    recall = metrics.event_recall(run, truth)
    lag = metrics.lag(run)
    named = metrics.names(run, wire_events or [])

    by_kind: dict[str, int] = {}
    for error in errors:
        by_kind[error.kind] = by_kind.get(error.kind, 0) + 1

    return Scorecard(
        name=name,
        lines=len(run.lines),
        factual_error_rate=len(errors) / len(run.lines) if run.lines else 0.0,
        errors_by_kind=by_kind,
        event_recall=recall.rate,
        recall_by_event=recall.by_event,
        gate_rejection_rate=metrics.gate_rejection_rate(run),
        gate_reasons=metrics.gate_table(run),
        lag_p50=lag.p50,
        lag_p95=lag.p95,
        silence_ratio=metrics.silence_ratio(run, duration_s),
        repetition_rate=metrics.repetition_rate(run),
        name_rate=named.rate,
        name_precision=named.precision,
        corrections=run.corrections,
        preempted=run.preempted,
        cost_usd=run.cost_usd,
    )


def table(cards: list[Scorecard]) -> str:
    return "\n".join([HEADER, *(card.row() for card in cards)])


def detail(card: Scorecard) -> str:
    """The breakdowns that explain a row, for when a number looks wrong."""
    lines = [f"### {card.name}", ""]
    if card.errors_by_kind:
        lines.append("Factual errors by kind: " + ", ".join(
            f"{k} {v}" for k, v in sorted(card.errors_by_kind.items(), key=lambda kv: -kv[1])
        ))
    if card.gate_reasons:
        lines.append("Gate rejections by reason: " + ", ".join(
            f"{k} {v}" for k, v in card.gate_reasons.items()
        ))
    if card.recall_by_event:
        lines.append("Recall by event: " + ", ".join(
            f"{k} {got}/{need}" for k, (got, need) in sorted(card.recall_by_event.items())
        ))
    lines.append(f"Lines cut off mid-sentence: {card.preempted}")
    if card.corrections:
        lines.append(f"Corrections from the wire: {card.corrections}")
    if card.watched_s:
        lines.append(f"Match watched: {card.watched_s:.0f}s")
    return "\n".join(lines)
