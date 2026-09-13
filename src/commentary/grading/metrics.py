"""Turning a run into numbers.

Everything here reads a trace file and, where grading is involved, a ground
truth that the runtime never saw. That separation is the point: the feed and
the human commentator's transcript exist only in this module and the ones
next to it, so there is no path by which they could leak into a live call.

The metrics are chosen to catch the ways this system fails rather than the
ways it succeeds. Anyone can produce a sentence per four seconds; the
questions that matter are whether the sentences are true, whether they arrive
while the moment is still happening, and whether the silences are in the
right places.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from commentary.gate import OPENERS
from commentary.schemas import Event, GroundTruthEvent, KnowledgePack, WireEvent
from commentary.trace import read_trace, rows_of
from commentary.voice.speaker import WORDS_PER_SECOND

WORD = re.compile(r"[a-z0-9']+")
SCORELINE = re.compile(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\b")
_STOPWORD_TEXT = (
    "the a an and or but it is in on at to of for with as its his her their they he she "
    "we us our you your me my this that now up down out off over into from by"
)
STOPWORDS = frozenset(_STOPWORD_TEXT.split(" "))


#: Ways a line can claim a goal. The first version listed "goal", "scores"
#: and "it's in", and therefore missed "It is in! ... has scored" entirely —
#: so phantom goals in that phrasing were counted nowhere and every factual
#: error rate was understated. A commentator has many ways to say it and a
#: checker that only knows three is measuring its own vocabulary.
#:
#: The bare word "goal" came off the list for the opposite reason. "Back
#: towards their own goal" is not a claim that one was scored, and the
#: trace's ``event`` field says outright whether the caller thought it was.
GOAL_CLAIMS: tuple[str, ...] = (
    "scores",
    "scored",
    "it's in",
    "it is in",
    "finds the net",
    "back of the net",
    "makes it",
    "puts them ahead",
    "levels it",
    "equaliser",
)

_SUFFIXES = ("ing", "edly", "ed", "es", "s")

#: How long after a goal a line may still be about it. The broadcaster spends
#: a minute or more on the celebration, the replays and the scorer's face, and
#: every line over that material is a line about a goal that really happened.
#: The same number as ``runtime.GOAL_TALK_CAP_S``, and a test holds them
#: equal: a grader stricter than the gate counts phantom goals the gate was
#: right to let through, which is the A19 bug scored instead of run.
GOAL_TALK_S = 150.0


def normalise(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in folded if not unicodedata.combining(c))


def stem(word: str) -> str:
    """Crude suffix stripping, so push and pushing count as the same word.

    Repetition in commentary is almost never verbatim. It is the same thought
    in a slightly different tense, which is exactly what an unstemmed token
    comparison sails past.
    """
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokens(text: str) -> set[str]:
    return {stem(w) for w in WORD.findall(normalise(text)) if w not in STOPWORDS}


@dataclass
class SpokenLine:
    """One line that actually reached a voice, with both clocks kept.

    ``video_ts`` is the instant in the match the line is about; ``live_ts`` is
    where the live edge had reached by the time the line existed. Lag is the
    gap between them, and both are on the video clock — an earlier version
    measured against wall time and reported a lag of eight days.
    """

    video_ts: float
    voice: str
    text: str
    live_ts: float = 0.0
    event: str = "none"

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class Run:
    """One match, as reconstructed from its trace."""

    run_id: str
    lines: list[SpokenLine] = field(default_factory=list)
    gate_rejections: list[str] = field(default_factory=list)
    gate_passes: int = 0
    board_reads: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    duration_s: float = 0.0
    preempted: int = 0
    corrections: int = 0

    @property
    def spoken_seconds(self) -> float:
        """Wall time occupied by speech, at a realistic speaking rate."""
        return sum(line.words for line in self.lines) / WORDS_PER_SECOND


def load_run(path: Path) -> Run:
    rows = read_trace(path)
    run = Run(run_id=path.stem)
    for row in rows_of(rows, "spoken"):
        run.lines.append(
            SpokenLine(
                video_ts=float(row.get("ts", 0.0)),
                live_ts=float(row.get("live_ts", 0.0)),
                voice=str(row.get("voice", "caller")),
                # "spoken" is what came out of the voice; a preempted line's
                # "text" is what it would have been. The director writes
                # "spoken" on every spoken row, so there is nothing to fall
                # back to and nothing that should.
                text=str(row.get("spoken", "")),
                event=str(row.get("event", "none")),
            )
        )
    for row in rows_of(rows, "preempted"):
        if row.get("reason") == "cut":
            run.preempted += 1
    for row in rows_of(rows, "gate"):
        if row.get("passed"):
            run.gate_passes += 1
        else:
            run.gate_rejections.extend(str(r) for r in row.get("reasons", []))
    run.board_reads = rows_of(rows, "board")
    run.corrections = len(rows_of(rows, "correction"))
    costs = rows_of(rows, "cost")
    if costs:
        run.cost_usd = float(costs[-1].get("total_usd", 0.0))
    stamps = [float(r.get("ts", 0.0)) for r in rows if "ts" in r]
    run.duration_s = max(stamps) - min(stamps) if stamps else 0.0
    return run


# -- timing -------------------------------------------------------------


@dataclass(frozen=True)
class Lag:
    p50: float
    p95: float
    n: int

    def as_dict(self) -> dict[str, float]:
        return {"p50_s": round(self.p50, 2), "p95_s": round(self.p95, 2), "n": self.n}


def lag(run: Run) -> Lag:
    """Buffer depth plus generation time: how late a line is against the pitch.

    Lines written before the trace carried a live edge are skipped rather than
    counted as zero, so an old trace reports a smaller ``n`` instead of a
    flattering number.
    """
    gaps = sorted(
        max(0.0, line.live_ts - line.video_ts) for line in run.lines if line.live_ts > 0.0
    )
    if not gaps:
        return Lag(0.0, 0.0, 0)
    return Lag(
        p50=statistics.median(gaps),
        p95=gaps[min(len(gaps) - 1, int(0.95 * len(gaps)))],
        n=len(gaps),
    )


def silence_ratio(run: Run, duration_s: float | None = None) -> float:
    """Fraction of the match with nobody talking.

    Real commentary sits somewhere around a third to a half silent. Zero means
    the system is chattering; close to one means it has nothing to say.

    Compared against a human broadcast this number has to be read carefully
    when the human side came from captions rather than from a transcription:
    a gap in auto-captions is either a commentator saying nothing or an ASR
    giving up in crowd noise, so the human's silence ratio is a ceiling.
    """
    total = duration_s or run.duration_s
    if total <= 0:
        return 1.0
    return max(0.0, 1.0 - run.spoken_seconds / total)


# -- content ------------------------------------------------------------


def repetition_rate(run: Run, threshold: float = 0.62, window: int = 5) -> float:
    """Fraction of lines that substantially restate a recent line."""
    if len(run.lines) < 2:
        return 0.0
    repeats = 0
    for i, line in enumerate(run.lines):
        recent = run.lines[max(0, i - window) : i]
        current = tokens(line.text)
        if not current:
            continue
        for earlier in recent:
            other = tokens(earlier.text)
            if not other:
                continue
            overlap = len(current & other) / len(current | other)
            if overlap >= threshold:
                repeats += 1
                break
    return repeats / len(run.lines)


@dataclass
class Recall:
    matched: int
    total: int
    by_event: dict[str, tuple[int, int]] = field(default_factory=dict)
    missed: list[GroundTruthEvent] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.matched / self.total if self.total else 0.0


def event_recall(run: Run, truth: list[GroundTruthEvent], window_s: float = 10.0) -> Recall:
    """Did a line land near each thing that actually happened?

    Matched on video time, not wall time, so a system with a deeper buffer is
    not punished for the buffer — only for being slow to react within it.
    """
    result = Recall(matched=0, total=len(truth))
    for event in truth:
        hit = any(abs(line.video_ts - event.video_ts) <= window_s for line in run.lines)
        got, need = result.by_event.get(event.event.value, (0, 0))
        result.by_event[event.event.value] = (got + int(hit), need + 1)
        if hit:
            result.matched += 1
        else:
            result.missed.append(event)
    return result


# -- truth --------------------------------------------------------------


@dataclass
class FactualError:
    video_ts: float
    kind: str
    detail: str
    line: str


def roster_names(pack: KnowledgePack) -> set[str]:
    """Every spelling of a squad member the grader will accept in a line.

    Each word of a full name goes in as well as the whole, because a line is
    scanned word by word and a compound surname arrives as two of them.
    ``Player.surname`` splits on the last space, so Di María's is "María" and
    De Paul's is "Paul"; without the parts, the grader counted "Di" and "De"
    as names off the roster every time the system got one right.

    The managers are in for the same reason the gate has them: a commentator
    says "Deschamps has a decision to make", the gate allows it, and a grader
    without them scores the line as an invented name.
    """
    names: set[str] = set()
    for sheet in (pack.home, pack.away):
        for player in sheet.squad:
            full = normalise(player.name)
            names.add(full)
            names.add(normalise(player.surname))
            names.update(full.split())
        if sheet.manager:
            manager = normalise(sheet.manager)
            names.add(manager)
            names.update(manager.split())
    names.discard("")
    return names


def score_at(truth: list[GroundTruthEvent], ts: float) -> tuple[int, int]:
    home = away = 0
    for event in truth:
        if event.video_ts <= ts:
            home, away = event.home_score, event.away_score
    return home, away


def factual_errors(
    run: Run,
    truth: list[GroundTruthEvent],
    pack: KnowledgePack,
    *,
    goal_window_s: float = 12.0,
) -> list[FactualError]:
    """The errors a machine can find without asking a model.

    This does not replace the judge — a line can be fluent, on-roster and
    still describe something that did not happen. It catches the three
    failures that are checkable outright, which are also the three the fact
    gate is supposed to make impossible, so anything found here is a hole in
    the gate rather than a subtlety of language.
    """
    known = roster_names(pack)
    goals = [e.video_ts for e in truth if e.event is Event.GOAL]
    errors: list[FactualError] = []

    for line in run.lines:
        lowered = normalise(line.text)

        for word in re.findall(r"\b[A-ZÁÉÍÓÚÄÖÜÑ][a-zá-ü]+\b", line.text):
            candidate = normalise(word)
            if candidate in known or candidate in STOPWORDS:
                continue
            if _is_team_word(candidate, pack) or _is_ordinary_opener(line.text, word):
                continue
            errors.append(FactualError(line.video_ts, "name_off_roster", word, line.text))

        match = SCORELINE.search(line.text)
        if match:
            said = (int(match.group(1)), int(match.group(2)))
            actual = score_at(truth, line.video_ts)
            if said != actual and said[::-1] != actual:
                errors.append(
                    FactualError(
                        line.video_ts,
                        "wrong_score",
                        f"said {said[0]}-{said[1]}, was {actual[0]}-{actual[1]}",
                        line.text,
                    )
                )

        claims_goal = line.event == Event.GOAL.value or any(
            phrase in lowered for phrase in GOAL_CLAIMS
        )
        # Asymmetric on purpose: a claim ahead of the goal is the system
        # inventing one, a claim behind it is the celebration.
        if claims_goal and not any(
            -goal_window_s <= line.video_ts - g <= GOAL_TALK_S for g in goals
        ):
            errors.append(FactualError(line.video_ts, "phantom_goal", "no goal nearby", line.text))

    return errors


def _is_team_word(candidate: str, pack: KnowledgePack) -> bool:
    """A word that names one of the two sides, the ground, or the competition.

    The ground is in here because a commentator says where they are — "away
    at Ashcombe Park" — and without it the grader reports "Park" as a name
    the system invented. The gate has always allowed these; the two have to
    agree or the results table counts errors the gate deliberately let by.
    """
    for sheet in (pack.home, pack.away):
        for part in (sheet.name, sheet.short, sheet.demonym):
            if part and candidate in normalise(part).split():
                return True
    for label in (pack.venue, pack.competition):
        if label and candidate in normalise(label).split():
            return True
    return False


def _is_ordinary_opener(text: str, word: str) -> bool:
    """A capital at the start of a sentence may be grammar rather than a name.

    This is a heuristic and it is biased towards flagging: a name the system
    invented and then put first in the sentence is the error that matters
    most, so anything not recognisably an ordinary opener is treated as a
    name claim. The model judge is what settles the genuinely ambiguous ones.

    The list lives in the gate, which applies the same rule to the same
    words. A grader with its own copy marks names the gate has already
    trimmed, or lets through words the gate cut — either way the table stops
    describing the system it is scoring.
    """
    return text.strip().startswith(word) and normalise(word) in OPENERS


# -- naming players ------------------------------------------------------


@dataclass
class Names:
    """Does it name players, and is it right? The two numbers that answer."""

    lines_with_name: int = 0
    names: int = 0
    correct: int = 0
    #: Lines in the run, so ``rate`` is a fraction of something stated.
    total: int = 0

    @property
    def rate(self) -> float:
        """Lines carrying at least one player name, as a fraction of all lines."""
        return self.lines_with_name / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        return self.correct / self.names if self.names else 0.0


def names(run: Run, wire_events: list[WireEvent], *, window_s: float = 3.0) -> Names:
    """Score every player named in a line against what the feed says happened.

    A name is correct when the feed has that player doing something within
    ``window_s`` of the line — as the actor or as the other party, because
    "foul by Rabiot on Messi" names two people and both are right.

    The window is small on purpose. A name that happens to belong to somebody
    who touched the ball twenty seconds earlier is not the system naming the
    right player, it is the system naming a player who is on the pitch.
    """
    scored = Names(total=len(run.lines))
    for line in run.lines:
        found = named_in(line.text, wire_events)
        if not found:
            continue
        scored.lines_with_name += 1
        scored.names += len(found)
        near = _people_near(wire_events, line.video_ts, window_s)
        scored.correct += sum(1 for name in found if name in near)
    return scored


def named_in(text: str, wire_events: list[WireEvent]) -> list[str]:
    """Every surname from the feed's cast that this line says out loud."""
    words = {normalise(w) for w in re.findall(r"\b[A-ZÁÉÍÓÚÄÖÜÑ][\w'-]+\b", text)}
    found: list[str] = []
    for person in _cast(wire_events):
        if normalise(person.rsplit(" ", 1)[-1]) in words:
            found.append(person)
    return found


def _cast(wire_events: list[WireEvent]) -> set[str]:
    people: set[str] = set()
    for event in wire_events:
        people.update(name for name in (event.player, event.recipient) if name)
    return people


def _people_near(wire_events: list[WireEvent], ts: float, window_s: float) -> set[str]:
    near: set[str] = set()
    for event in wire_events:
        if event.video_ts is None or abs(event.video_ts - ts) > window_s:
            continue
        near.update(name for name in (event.player, event.recipient) if name)
    return near


def corrections(run: Run) -> int:
    """How many times the statistician changed something the system believed."""
    return run.corrections


# -- the gate -----------------------------------------------------------


def gate_table(run: Run) -> dict[str, int]:
    """Rejections counted by reason tag, which is a headline result."""
    counts: dict[str, int] = {}
    for reason in run.gate_rejections:
        tag = reason.split(":", 1)[0].strip()
        counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def gate_rejection_rate(run: Run) -> float:
    judged = run.gate_passes + len(run.gate_rejections)
    return len(run.gate_rejections) / judged if judged else 0.0
