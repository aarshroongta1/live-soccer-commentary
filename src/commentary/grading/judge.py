"""The model judge: the two questions a checker cannot answer.

:mod:`commentary.grading.metrics` catches the errors a machine can prove —
a name that is on no roster, a scoreline that contradicts the board, a goal
claimed where the feed has none. What it cannot catch is the failure this
system is most likely to produce: a fluent line, every name on the roster,
the score right, describing something that did not happen. Only a reader who
knows what did happen can see that, so the feed and a model are pointed at
each line one at a time.

The two rates are reported side by side and never averaged. They measure
different things and have different error bars — one is arithmetic, the
other is a model's opinion — and a single blended number would hide both.

The second question is whether any of it is worth listening to, which no
factual check reaches at all. For that, our line and the human
commentator's line for the same moment go to the judge unlabelled, in both
orders. A win counts only when the two orders agree; a judge that picks
whichever line it read first is reporting its own position bias, not a
preference, and that is worth knowing rather than worth averaging away.

Nothing here is imported by the runtime, and nothing here may be: the feed
and the transcript both live on this side of the wall.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from commentary.config import JUDGE_MODEL
from commentary.grading import metrics
from commentary.grading.metrics import Run, SpokenLine
from commentary.grading.transcripts import Transcript, said_near
from commentary.llm.base import Block, LLMBackend, text_block
from commentary.llm.schema import strict_schema
from commentary.schemas import GroundTruthEvent, KnowledgePack

#: How much of the feed the judge sees around a line. Wide enough that a
#: line about the build-up to a goal has the goal in front of it; narrow
#: enough that it cannot excuse a line by pointing at a different minute.
CONTEXT_WINDOW_S = 20.0

#: How far apart our line and a human's line may be and still be treated as
#: describing the same moment. The plan's pairwise comparison is over ten
#: second windows, so this is half of one.
PAIR_WINDOW_S = 5.0


class JudgeError(RuntimeError):
    """The judge could not be asked, or did not answer in the right shape."""


class Verdict(StrEnum):
    """What the feed says about one spoken line."""

    TRUE = "true"
    FALSE = "false"
    #: The feed neither confirms nor contradicts it. Feeds record goals,
    #: cards and corners; they do not record "he shifts it onto his left",
    #: which is most of what a commentator says and is not an error.
    UNSUPPORTED = "unsupported"


class FactVerdict(BaseModel):
    """The judge's answer about one line."""

    verdict: Verdict
    reason: str = Field(description="One sentence. Name the feed event that settles it.")
    event_ref: str | None = Field(
        default=None,
        description="The feed event this line refers to, as shown in the list, or null",
    )


class Choice(StrEnum):
    A = "a"
    B = "b"
    TIE = "tie"


class PairVerdict(BaseModel):
    """Which of two unlabelled lines is the better call of the same moment."""

    choice: Choice
    reason: str = Field(description="One sentence.")


FACT_SYSTEM = """You are grading live football commentary against the official \
play-by-play record of the same match.

You are shown the events the feed recorded near one moment, the two squads, and \
one line a commentator said at that moment. Decide whether the line is true.

- true: the feed supports what the line says.
- false: the feed contradicts it — wrong scorer, wrong team, wrong score, or an \
event the feed places nowhere near this moment.
- unsupported: the feed says nothing either way. Most commentary is like this. A \
line about the run of play, a player's touch, the crowd, or the general shape of \
the game is unsupported, not false. Only say false when the feed actually \
contradicts the line.

Be strict about claims and generous about description. "Arsenal are pressing" is \
unsupported. "Saka makes it two" when the feed has no second goal is false."""

PAIR_SYSTEM = """You are comparing two lines of live football commentary describing \
the same moment of the same match.

Pick the better line. Better means: accurate about what is happening, specific \
rather than vague, well-timed for the moment, and something a listener would want \
to hear. Length is not quality either way.

Answer "tie" when they are genuinely close — that is a real answer, not a \
cop-out. You are not told where either line came from and you should not \
speculate; judge only what is written."""


# -- asking, without widening the backend protocol ---------------------


@dataclass(frozen=True)
class Question:
    """One thing to ask the judge, independent of how it gets asked.

    Separating the question from the asking is what lets the Batch API in
    without a batching flag on :class:`~commentary.llm.base.LLMBackend`. The
    runtime's backends answer one call at a time because the runtime has a
    deadline; grading has none and would rather pay half. Both shapes
    consume the same list of questions.
    """

    key: str
    tag: str
    system: str
    blocks: list[Block]
    output_format: type[BaseModel]
    max_tokens: int = 512
    effort: str | None = "low"


class Asker(Protocol):
    """Anything that can answer a batch of questions, in order."""

    async def ask(self, questions: Sequence[Question]) -> list[BaseModel]: ...


@dataclass
class SequentialAsker:
    """One call per question through the ordinary backend.

    The default, and the only one a :class:`ScriptedBackend` can serve. A
    small amount of concurrency because grading a match is a few hundred
    calls and doing them strictly one after another is an afternoon.
    """

    backend: LLMBackend
    model: str = JUDGE_MODEL
    concurrency: int = 4

    async def ask(self, questions: Sequence[Question]) -> list[BaseModel]:
        limit = asyncio.Semaphore(self.concurrency)

        async def one(question: Question) -> BaseModel:
            async with limit:
                parsed = await self.backend.parse(
                    model=self.model,
                    system=question.system,
                    blocks=question.blocks,
                    output_format=question.output_format,
                    max_tokens=question.max_tokens,
                    effort=question.effort,
                    tag=question.tag,
                )
                return parsed.value

        return list(await asyncio.gather(*(one(q) for q in questions)))


@dataclass
class BatchAsker:
    """The same questions through the Batch API, at half the price.

    Grading has no latency constraint — the match finished hours ago — so
    the only thing that matters is cost, and batch is half. This talks to
    the Anthropic client directly rather than through
    :class:`~commentary.llm.base.LLMBackend`: batching is not something the
    caller or the board reader could ever use, and putting it on the shared
    protocol would mean a parameter that means nothing at five of its six
    call sites. Composition instead — same questions, different asker.

    Results come back in whatever order they finish, so they are keyed by
    ``custom_id`` and put back in question order before returning.
    """

    model: str = JUDGE_MODEL
    poll_s: float = 20.0
    timeout_s: float = 24 * 3600
    client: Any = None

    async def ask(self, questions: Sequence[Question]) -> list[BaseModel]:
        if not questions:
            return []
        from commentary.llm.anthropic_backend import thinking_params

        client = self._client()
        requests = [
            {
                "custom_id": question.key,
                "params": {
                    "model": self.model,
                    "max_tokens": question.max_tokens,
                    "system": [text_block(question.system, cache=True)],
                    "messages": [{"role": "user", "content": question.blocks}],
                    **_with_format(
                        thinking_params(self.model, question.effort), question.output_format
                    ),
                },
            }
            for question in questions
        ]
        batch = await client.messages.batches.create(requests=requests)
        await self._wait(client, batch.id)

        answers: dict[str, BaseModel] = {}
        async for entry in await client.messages.batches.results(batch.id):
            answers[entry.custom_id] = _decode(entry, _format_for(questions, entry.custom_id))
        try:
            return [answers[question.key] for question in questions]
        except KeyError as exc:
            raise JudgeError(f"batch {batch.id} returned no result for {exc}") from exc

    def _client(self) -> Any:
        if self.client is None:
            import anthropic

            self.client = anthropic.AsyncAnthropic()
        return self.client

    async def _wait(self, client: Any, batch_id: str) -> None:
        waited = 0.0
        while waited < self.timeout_s:
            batch = await client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                return
            await asyncio.sleep(self.poll_s)
            waited += self.poll_s
        raise JudgeError(f"batch {batch_id} did not finish within {self.timeout_s:.0f}s")


def _with_format(params: dict[str, Any], output_format: type[BaseModel]) -> dict[str, Any]:
    config = dict(params.get("output_config", {}))
    config["format"] = {"type": "json_schema", "schema": strict_schema(output_format)}
    return {**params, "output_config": config}


def _format_for(questions: Sequence[Question], key: str) -> type[BaseModel]:
    for question in questions:
        if question.key == key:
            return question.output_format
    raise JudgeError(f"batch returned an unknown custom_id {key!r}")


def _decode(entry: Any, output_format: type[BaseModel]) -> BaseModel:
    if entry.result.type != "succeeded":
        raise JudgeError(f"{entry.custom_id}: batch result {entry.result.type}")
    message = entry.result.message
    if message.stop_reason == "refusal":
        raise JudgeError(f"{entry.custom_id}: judge refused")
    text = next((b.text for b in message.content if b.type == "text"), None)
    if text is None:
        raise JudgeError(f"{entry.custom_id}: no text block in batch result")
    return output_format.model_validate(json.loads(text))


def _typed[M: BaseModel](values: Sequence[BaseModel], kind: type[M]) -> list[M]:
    out: list[M] = []
    for value in values:
        if not isinstance(value, kind):
            raise JudgeError(f"judge returned {type(value).__name__}, wanted {kind.__name__}")
        out.append(value)
    return out


# -- is it true? -------------------------------------------------------


@dataclass(frozen=True)
class JudgedLine:
    video_ts: float
    text: str
    verdict: Verdict
    reason: str
    event_ref: str | None = None


@dataclass
class FactualityReport:
    """Every line, judged, with the denominator kept in plain sight."""

    judged: list[JudgedLine] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.judged)

    @property
    def false_lines(self) -> list[JudgedLine]:
        return [j for j in self.judged if j.verdict is Verdict.FALSE]

    @property
    def unsupported(self) -> int:
        return sum(1 for j in self.judged if j.verdict is Verdict.UNSUPPORTED)

    @property
    def error_rate(self) -> float:
        """False lines over every line judged — not over the ones the feed covers.

        The other denominator is tempting and wrong: a system that says
        nothing the feed can check would score a perfect zero on it.
        """
        return len(self.false_lines) / self.n if self.n else 0.0

    @property
    def unsupported_rate(self) -> float:
        return self.unsupported / self.n if self.n else 0.0


async def judge_factuality(
    run: Run,
    truth: list[GroundTruthEvent],
    pack: KnowledgePack,
    backend: LLMBackend,
    *,
    window_s: float = CONTEXT_WINDOW_S,
    asker: Asker | None = None,
) -> FactualityReport:
    """Ask the judge whether each spoken line is true, given the feed.

    ``asker`` is the composition point: leave it out and every line goes
    through ``backend`` one call at a time, or pass a :class:`BatchAsker`
    for the half-price offline path. The backend is still what a
    :class:`ScriptedBackend` hooks into, so tests need neither.
    """
    roster = _roster_text(pack)
    questions = [
        Question(
            key=f"fact-{i:04d}",
            tag="judge_factuality",
            system=FACT_SYSTEM,
            blocks=[
                text_block(roster, cache=True),
                text_block(_fact_prompt(line, truth, window_s)),
            ],
            output_format=FactVerdict,
        )
        for i, line in enumerate(run.lines)
    ]
    answers = _typed(await _run(asker, backend, questions), FactVerdict)
    return FactualityReport(
        judged=[
            JudgedLine(
                video_ts=line.video_ts,
                text=line.text,
                verdict=answer.verdict,
                reason=answer.reason,
                event_ref=answer.event_ref,
            )
            for line, answer in zip(run.lines, answers, strict=True)
        ]
    )


def _fact_prompt(line: SpokenLine, truth: list[GroundTruthEvent], window_s: float) -> str:
    near = [e for e in truth if abs(e.video_ts - line.video_ts) <= window_s]
    home, away = metrics.score_at(truth, line.video_ts)
    rows = "\n".join(_event_row(e, line.video_ts) for e in near) or "(nothing in this window)"
    return (
        f"Moment: {line.video_ts:.1f}s of video. Score at this point: {home}-{away}.\n\n"
        f"Feed events within {window_s:.0f}s:\n{rows}\n\n"
        f"The line said at this moment:\n{line.text!r}"
    )


def _event_row(event: GroundTruthEvent, at: float) -> str:
    delta = event.video_ts - at
    who = f" by {event.player}" if event.player else ""
    return (
        f"- {delta:+.1f}s: {event.event.value} ({event.side.value}){who}, "
        f"score after {event.home_score}-{event.away_score}"
    )


def _roster_text(pack: KnowledgePack) -> str:
    parts = []
    for label, sheet in (("Home", pack.home), ("Away", pack.away)):
        squad = ", ".join(
            f"{p.name}" + (f" ({p.number})" if p.number is not None else "") for p in sheet.squad
        )
        parts.append(f"{label}: {sheet.name}. Squad: {squad or 'unknown'}")
    return "Teams and squads for this match.\n" + "\n".join(parts)


# -- is it any good? ---------------------------------------------------


@dataclass(frozen=True)
class Matchup:
    """Our line and somebody else's, describing the same moment."""

    video_ts: float
    ours: str
    theirs: str


@dataclass(frozen=True)
class Outcome:
    """One matchup, judged both ways round."""

    video_ts: float
    ours: str
    theirs: str
    #: The verdict with ours presented as A, then with ours presented as B.
    first: Choice
    second: Choice
    reason: str = ""

    @property
    def agreed(self) -> bool:
        """Swapping the two lines swapped the answer, as it should.

        Both orderings saying tie counts as agreement. Anything else — the
        same letter twice, or a tie one way and a pick the other — means the
        judge moved when only the presentation did.
        """
        flipped = {Choice.A: Choice.B, Choice.B: Choice.A, Choice.TIE: Choice.TIE}
        return self.second is flipped[self.first]

    @property
    def result(self) -> str:
        if not self.agreed:
            return "tie"
        if self.first is Choice.A:
            return "win"
        if self.first is Choice.B:
            return "loss"
        return "tie"


@dataclass
class PairwiseReport:
    """Wins, losses, ties, and how often the judge contradicted itself."""

    outcomes: list[Outcome] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.outcomes)

    @property
    def wins(self) -> int:
        return sum(1 for o in self.outcomes if o.result == "win")

    @property
    def losses(self) -> int:
        return sum(1 for o in self.outcomes if o.result == "loss")

    @property
    def ties(self) -> int:
        return self.n - self.wins - self.losses

    @property
    def disagreements(self) -> int:
        """Matchups where swapping A and B swapped the answer.

        Counted and reported rather than resolved. A judge that flips is
        expressing a position preference, and a win rate built on those is
        measuring the prompt, not the commentary.
        """
        return sum(1 for o in self.outcomes if not o.agreed)

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def disagreement_rate(self) -> float:
        return self.disagreements / self.n if self.n else 0.0


def matchups(
    ours: Run | Sequence[SpokenLine],
    theirs: Transcript | Run | Sequence[SpokenLine],
    *,
    window_s: float = PAIR_WINDOW_S,
) -> list[Matchup]:
    """Pair each of our lines with what the other source said at that moment.

    Moments where the other source said nothing are dropped rather than
    compared against an empty string — a line that beats silence has not
    beaten anything, and counting those would inflate the win rate by
    exactly the amount the human commentators chose to stay quiet.
    """
    pairs: list[Matchup] = []
    for line in _lines(ours):
        other = _said_at(theirs, line.video_ts, window_s)
        if not other or not line.text.strip():
            continue
        pairs.append(Matchup(video_ts=line.video_ts, ours=line.text, theirs=other))
    return pairs


async def judge_pairwise(
    ours: Run | Sequence[SpokenLine],
    theirs: Transcript | Run | Sequence[SpokenLine],
    backend: LLMBackend,
    *,
    window_s: float = PAIR_WINDOW_S,
    asker: Asker | None = None,
) -> PairwiseReport:
    """Blind A/B against the human commentator, both orderings, every pair.

    Both orderings are asked because a single ordering is not a measurement:
    LLM judges have a well-known position bias, and with one pass there is
    no way to tell a real preference from a preference for going first.
    Neither prompt says which line came from where.
    """
    pairs = matchups(ours, theirs, window_s=window_s)
    questions: list[Question] = []
    for i, pair in enumerate(pairs):
        questions.append(_pair_question(f"pair-{i:04d}-fwd", pair.video_ts, pair.ours, pair.theirs))
        questions.append(_pair_question(f"pair-{i:04d}-rev", pair.video_ts, pair.theirs, pair.ours))

    answers = _typed(await _run(asker, backend, questions), PairVerdict)
    outcomes = [
        Outcome(
            video_ts=pair.video_ts,
            ours=pair.ours,
            theirs=pair.theirs,
            first=answers[2 * i].choice,
            second=answers[2 * i + 1].choice,
            reason=answers[2 * i].reason,
        )
        for i, pair in enumerate(pairs)
    ]
    return PairwiseReport(outcomes=outcomes)


def _pair_question(key: str, video_ts: float, first: str, second: str) -> Question:
    prompt = (
        f"Moment: {video_ts:.1f}s into the match.\n\n"
        f"Line A: {first}\n"
        f"Line B: {second}\n\n"
        "Which is the better call of this moment?"
    )
    return Question(
        key=key,
        tag="judge_pairwise",
        system=PAIR_SYSTEM,
        blocks=[text_block(prompt)],
        output_format=PairVerdict,
        max_tokens=384,
    )


def _lines(source: Run | Sequence[SpokenLine]) -> list[SpokenLine]:
    return list(source.lines) if isinstance(source, Run) else list(source)


def _said_at(
    source: Transcript | Run | Sequence[SpokenLine], video_ts: float, window_s: float
) -> str:
    if isinstance(source, Transcript):
        return said_near(source, video_ts, window_s)
    near = [
        line
        for line in _lines(source)
        if abs(line.video_ts - video_ts) <= window_s and line.text.strip()
    ]
    if not near:
        return ""
    return min(near, key=lambda line: abs(line.video_ts - video_ts)).text


async def _run(
    asker: Asker | None, backend: LLMBackend, questions: Sequence[Question]
) -> list[BaseModel]:
    if not questions:
        return []
    return await (asker or SequentialAsker(backend)).ask(questions)


# -- the write-up ------------------------------------------------------


@dataclass
class JudgeReport:
    """What the judge found, rendered like :func:`report.detail` next door."""

    name: str
    factuality: FactualityReport | None = None
    pairwise: PairwiseReport | None = None
    #: From :func:`metrics.factual_errors`. Kept beside the judged rate, never
    #: folded into it.
    deterministic_errors: int | None = None
    deterministic_lines: int | None = None

    @property
    def deterministic_rate(self) -> float | None:
        if self.deterministic_errors is None or not self.deterministic_lines:
            return None
        return self.deterministic_errors / self.deterministic_lines

    def markdown(self) -> str:
        out = [f"### {self.name} — judged", ""]
        rate = self.deterministic_rate
        if rate is not None:
            out.append(
                f"Factual errors, deterministic: {rate:.1%} "
                f"({self.deterministic_errors} of {self.deterministic_lines} lines)"
            )
        if self.factuality is not None:
            fact = self.factuality
            out.append(
                f"Factual errors, judged by {JUDGE_MODEL}: {fact.error_rate:.1%} "
                f"({len(fact.false_lines)} of {fact.n} lines false; "
                f"{fact.unsupported} unsupported by the feed and not counted)"
            )
            for judged in fact.false_lines[:10]:
                out.append(f"- {judged.video_ts:.1f}s {judged.text!r} — {judged.reason}")
        if self.pairwise is not None:
            pair = self.pairwise
            out.append(
                f"Pairwise vs the human commentary: {pair.win_rate:.0%} win "
                f"({pair.wins}W {pair.losses}L {pair.ties}T of {pair.n}), "
                f"judge disagreed with itself on {pair.disagreement_rate:.0%} "
                f"of orderings"
            )
        return "\n".join(out)


async def judge_run(
    name: str,
    run: Run,
    truth: list[GroundTruthEvent],
    pack: KnowledgePack,
    backend: LLMBackend,
    *,
    theirs: Transcript | Run | Sequence[SpokenLine] | None = None,
    asker: Asker | None = None,
) -> JudgeReport:
    """Both judgements plus the deterministic count, in one object."""
    factuality = await judge_factuality(run, truth, pack, backend, asker=asker)
    pairwise = (
        await judge_pairwise(run, theirs, backend, asker=asker) if theirs is not None else None
    )
    return JudgeReport(
        name=name,
        factuality=factuality,
        pairwise=pairwise,
        deterministic_errors=len(metrics.factual_errors(run, truth, pack)),
        deterministic_lines=len(run.lines),
    )
