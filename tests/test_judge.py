"""The judge, with a scripted model. No network, no key, no real call."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from commentary.grading.judge import (
    BatchAsker,
    Choice,
    FactVerdict,
    JudgeError,
    JudgeReport,
    PairVerdict,
    Verdict,
    judge_factuality,
    judge_pairwise,
    judge_run,
    matchups,
)
from commentary.grading.metrics import Run, SpokenLine
from commentary.grading.transcripts import Segment, Transcript
from commentary.llm.base import Block
from commentary.llm.fake import ScriptedBackend
from commentary.schemas import Event, GroundTruthEvent, KnowledgePack, Player, Side, TeamSheet


@pytest.fixture
def pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Arsenal", short="ARS", starters=[Player(name="Bukayo Saka", number=7)]
        ),
        away=TeamSheet(name="Real Madrid", short="RMA", starters=[Player(name="Jude Bellingham")]),
    )


@pytest.fixture
def truth() -> list[GroundTruthEvent]:
    return [
        GroundTruthEvent(video_ts=30.0, event=Event.SHOT, side=Side.HOME),
        GroundTruthEvent(
            video_ts=60.0, event=Event.GOAL, side=Side.HOME, player="Bukayo Saka", home_score=1
        ),
    ]


def run_of(*lines: tuple[float, str]) -> Run:
    return Run(
        run_id="scripted",
        lines=[
            SpokenLine(video_ts=ts, voice="caller", text=text) for ts, text in lines
        ],
    )


def prompt_of(blocks: list[Block]) -> str:
    return "\n".join(b["text"] for b in blocks if b.get("type") == "text")


# -- factuality --------------------------------------------------------


def verdicts(**by_phrase: Verdict):
    """A judge that answers by looking for a phrase in the line under test.

    It reads only the quoted line, not the whole prompt — the roster and the
    feed rows carry the same surnames, and a handler that matched on those
    would agree with itself for the wrong reason.
    """

    def handler(blocks: list[Block], _fmt: type) -> FactVerdict:
        text = prompt_of(blocks).rsplit("The line said at this moment:", 1)[-1]
        for phrase, verdict in by_phrase.items():
            if phrase in text:
                return FactVerdict(verdict=verdict, reason=f"matched {phrase}", event_ref=None)
        return FactVerdict(verdict=Verdict.UNSUPPORTED, reason="feed is silent", event_ref=None)

    return handler


async def test_a_line_the_judge_calls_false_is_an_error(pack, truth) -> None:
    backend = ScriptedBackend()
    backend.register("judge_factuality", verdicts(Bellingham=Verdict.FALSE))
    run = run_of((61.0, "Bellingham puts Madrid in front"))

    report = await judge_factuality(run, truth, pack, backend)

    assert [j.verdict for j in report.judged] == [Verdict.FALSE]
    assert report.error_rate == pytest.approx(1.0)
    assert report.false_lines[0].text == "Bellingham puts Madrid in front"


async def test_the_rate_counts_every_line_judged_not_just_the_checkable_ones(pack, truth) -> None:
    """Unsupported lines stay in the denominator and out of the numerator.

    Dropping them would let a system that only ever says "what a game" score
    a perfect zero, and counting them as errors would punish it for the
    feed not recording ordinary play.
    """
    backend = ScriptedBackend()
    backend.register("judge_factuality", verdicts(Bellingham=Verdict.FALSE, Saka=Verdict.TRUE))
    run = run_of(
        (61.0, "Saka makes it one"),
        (62.0, "Bellingham puts Madrid in front"),
        (63.0, "Arsenal are all over them"),
        (64.0, "the noise in here"),
    )

    report = await judge_factuality(run, truth, pack, backend)

    assert report.n == 4
    assert report.unsupported == 2
    assert report.error_rate == pytest.approx(0.25)


async def test_the_judge_is_shown_the_roster_and_the_feed_around_the_line(pack, truth) -> None:
    backend = ScriptedBackend()
    backend.register("judge_factuality", verdicts())
    await judge_factuality(run_of((61.0, "in it goes")), truth, pack, backend)

    shown = prompt_of(backend.calls_tagged("judge_factuality")[0].blocks)
    assert "Bukayo Saka (7)" in shown
    assert "goal (home) by Bukayo Saka" in shown
    assert "Score at this point: 1-0" in shown


async def test_a_run_with_nothing_spoken_asks_nothing(pack, truth) -> None:
    backend = ScriptedBackend()
    report = await judge_factuality(Run(run_id="silent"), truth, pack, backend)
    assert report.n == 0 and report.error_rate == 0.0
    assert backend.calls == []


# -- pairwise ----------------------------------------------------------


HUMAN = Transcript(
    segments=[Segment(58.0, 64.0, "Saka! Arsenal have the lead")],
    source="broadcast.wav",
)


def prefers(text: str):
    """A judge that consistently picks whichever side ``text`` is on."""

    def handler(blocks: list[Block], _fmt: type) -> PairVerdict:
        prompt = prompt_of(blocks)
        a = prompt.split("Line A: ", 1)[1].split("\n", 1)[0]
        choice = Choice.A if text in a else Choice.B
        return PairVerdict(choice=choice, reason="scripted")

    return handler


def always(choice: Choice):
    """A judge with a position preference and no opinion about the lines."""
    return lambda _blocks, _fmt: PairVerdict(choice=choice, reason="scripted")


async def test_a_win_needs_both_orderings_to_agree() -> None:
    backend = ScriptedBackend()
    backend.register("judge_pairwise", prefers("Saka slots it home"))
    ours = run_of((60.0, "Saka slots it home"))

    report = await judge_pairwise(ours, HUMAN, backend)

    assert report.n == 1
    assert report.wins == 1 and report.losses == 0 and report.ties == 0
    assert report.win_rate == pytest.approx(1.0)
    assert report.disagreements == 0
    assert len(backend.calls_tagged("judge_pairwise")) == 2


async def test_the_other_side_winning_both_ways_is_a_loss() -> None:
    backend = ScriptedBackend()
    backend.register("judge_pairwise", prefers("Arsenal have the lead"))
    report = await judge_pairwise(run_of((60.0, "Saka slots it home")), HUMAN, backend)
    assert report.losses == 1 and report.wins == 0
    assert report.disagreements == 0


async def test_a_judge_that_flips_with_the_ordering_is_a_tie_not_a_win() -> None:
    backend = ScriptedBackend()
    backend.register("judge_pairwise", always(Choice.A))
    report = await judge_pairwise(run_of((60.0, "Saka slots it home")), HUMAN, backend)

    assert report.wins == 0 and report.losses == 0
    assert report.ties == 1
    assert report.disagreements == 1
    assert report.disagreement_rate == pytest.approx(1.0)
    assert report.outcomes[0].result == "tie"


async def test_both_orderings_calling_it_level_is_an_agreed_tie() -> None:
    backend = ScriptedBackend()
    backend.register("judge_pairwise", always(Choice.TIE))
    report = await judge_pairwise(run_of((60.0, "Saka slots it home")), HUMAN, backend)
    assert report.ties == 1 and report.disagreements == 0


async def test_neither_prompt_says_which_line_is_ours() -> None:
    backend = ScriptedBackend()
    backend.register("judge_pairwise", always(Choice.TIE))
    await judge_pairwise(run_of((60.0, "Saka slots it home")), HUMAN, backend)

    for call in backend.calls_tagged("judge_pairwise"):
        whole = call.system + "\n" + prompt_of(call.blocks)
        assert "human" not in whole.lower()
        assert "commentator" not in whole.lower()
        assert "ours" not in whole.lower()


def test_moments_the_humans_left_alone_are_not_compared() -> None:
    """Beating silence is not beating anything, so those pairs are dropped."""
    ours = run_of((60.0, "Saka slots it home"), (200.0, "quiet spell here"))
    assert [m.video_ts for m in matchups(ours, HUMAN)] == [60.0]


def test_our_lines_can_be_compared_against_a_baseline_run_too() -> None:
    ours = run_of((60.0, "Saka slots it home"))
    theirs = run_of((59.0, "a goal"), (400.0, "nowhere near"))
    pairs = matchups(ours, theirs)
    assert len(pairs) == 1 and pairs[0].theirs == "a goal"


# -- the write-up ------------------------------------------------------


async def test_the_report_keeps_the_two_error_rates_apart(pack, truth) -> None:
    backend = ScriptedBackend()
    backend.register("judge_factuality", verdicts(Kowalczyk=Verdict.FALSE))
    backend.register("judge_pairwise", always(Choice.TIE))
    run = run_of(
        (61.0, "And there is Kowalczyk turning to shoot"),
        (62.0, "Arsenal keep it moving"),
    )

    report = await judge_run("full", run, truth, pack, backend, theirs=HUMAN)
    rendered = report.markdown()

    assert report.deterministic_errors == 1  # the invented name, caught by arithmetic
    assert report.factuality is not None and report.factuality.error_rate == pytest.approx(0.5)
    assert "Factual errors, deterministic: 50.0%" in rendered
    assert "Factual errors, judged by" in rendered
    assert "Pairwise vs the human commentary" in rendered


def test_an_empty_report_renders_without_numbers_it_does_not_have() -> None:
    assert JudgeReport(name="nothing").markdown().startswith("### nothing — judged")


# -- the batch path, against a stub client -----------------------------


class FakeBatches:
    """Just enough of ``client.messages.batches`` to exercise the shape.

    Results come back shuffled on purpose: the API returns them in whatever
    order they finished, and keying by position instead of ``custom_id``
    would attach every verdict to the wrong line while looking fine.
    """

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.sent: list[dict] = []

    async def create(self, *, requests: list[dict]) -> SimpleNamespace:
        self.sent = requests
        return SimpleNamespace(id="msgbatch_1")

    async def retrieve(self, batch_id: str) -> SimpleNamespace:
        return SimpleNamespace(id=batch_id, processing_status="ended")

    async def results(self, batch_id: str) -> AsyncIterator[SimpleNamespace]:
        async def rows() -> AsyncIterator[SimpleNamespace]:
            for custom_id, body in reversed(list(self.answers.items())):
                yield SimpleNamespace(
                    custom_id=custom_id,
                    result=SimpleNamespace(
                        type="succeeded",
                        message=SimpleNamespace(
                            stop_reason="end_turn",
                            content=[SimpleNamespace(type="text", text=body)],
                        ),
                    ),
                )

        return rows()


async def test_the_batch_asker_keys_results_by_custom_id(pack, truth) -> None:
    answers = {
        "fact-0000": '{"verdict": "true", "reason": "the feed has it", "event_ref": "goal"}',
        "fact-0001": '{"verdict": "false", "reason": "no such goal", "event_ref": null}',
    }
    batches = FakeBatches(answers)
    client = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    run = run_of((60.0, "Saka makes it one"), (90.0, "and a second for Saka"))

    report = await judge_factuality(
        run, truth, pack, ScriptedBackend(), asker=BatchAsker(client=client)
    )

    assert [j.verdict for j in report.judged] == [Verdict.TRUE, Verdict.FALSE]
    assert [r["custom_id"] for r in batches.sent] == ["fact-0000", "fact-0001"]
    params = batches.sent[0]["params"]
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert params["output_config"]["format"]["schema"]["additionalProperties"] is False


async def test_a_failed_batch_entry_is_raised_not_swallowed(pack, truth) -> None:
    batches = FakeBatches({"fact-0000": "{}"})

    async def errored(batch_id: str) -> AsyncIterator[SimpleNamespace]:
        async def rows() -> AsyncIterator[SimpleNamespace]:
            yield SimpleNamespace(custom_id="fact-0000", result=SimpleNamespace(type="expired"))

        return rows()

    batches.results = errored  # type: ignore[method-assign]
    client = SimpleNamespace(messages=SimpleNamespace(batches=batches))

    with pytest.raises(JudgeError, match="expired"):
        await judge_factuality(
            run_of((60.0, "one")), truth, pack, ScriptedBackend(), asker=BatchAsker(client=client)
        )
