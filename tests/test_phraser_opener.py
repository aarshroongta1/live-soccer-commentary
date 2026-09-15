"""The opener retry: reject a repeated first word once, and re-ask.

``docs/research/real-commentary-corpus.md`` section 8.2 and Gap 5. The
phraser opened 35-48% of its lines on the same first word as one of the last
five lead lines against 11-19% in real commentary, and seven prompt-only
attempts to move it did not. This is the code fix: after the model answers,
if its opening word repeats one of the last five spoken lead lines, the same
call is re-asked once, body extended with a note naming the offending word,
before the line is settled and returned.

What is protected here is the mechanics around that one re-ask: that it
fires only when it should, that it fires at most once, that a retry which
repeats again is kept rather than dropped, and that its cost is folded into
the usual usage rather than lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from commentary.agents.phraser import Phraser
from commentary.config import PhraserConfig
from commentary.llm.base import Block, Parsed, Usage
from commentary.llm.fake import ScriptedBackend
from commentary.schemas import CallerLine, Event, PhrasedLine, Scene, Side, Sighting

# -- fixtures ------------------------------------------------------------


def a_form(
    line: str = "Molina moves forward down the right and looks for a way inside.",
    *,
    event: Event = Event.CARRY,
) -> CallerLine:
    return CallerLine(
        scene=Scene.LIVE_PLAY,
        event=event,
        side=Side.HOME,
        team="Argentina",
        sightings=[Sighting(number=26, name="Molina")],
        confidence=0.7,
        speak=True,
        line=line,
    )


def a_phraser(backend: ScriptedBackend, *, recent_lines: int = 6) -> Phraser:
    return Phraser(
        backend,
        config=PhraserConfig(model="claude-haiku-4-5", recent_lines=recent_lines),
        home="Argentina",
        away="France",
    )


def queued(*lines: PhrasedLine) -> ScriptedBackend:
    backend = ScriptedBackend()
    backend.queue("phraser", list(lines))
    return backend


@dataclass
class CostedBackend:
    """A minimal backend whose queued answers each carry their own cost.

    :class:`ScriptedBackend` always reports ``cost_usd=0.0`` — its handlers
    choose the value, not the price — so it cannot show two calls' costs
    adding up. This does the one thing that test needs and nothing else.
    """

    answers: list[tuple[PhrasedLine, float]]
    calls: list[list[Block]] = field(default_factory=list)
    _total: Usage = field(default_factory=Usage)

    async def parse(
        self,
        *,
        model: str,
        system: str,
        blocks: list[Block],
        output_format: type,
        max_tokens: int = 1024,
        effort: str | None = None,
        cache_system: bool = True,
        tag: str = "",
    ) -> Parsed[PhrasedLine]:
        self.calls.append(blocks)
        value, cost = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        usage = Usage(cost_usd=cost)
        self._total = self._total + usage
        return Parsed(value=value, usage=usage, model=model)

    @property
    def total(self) -> Usage:
        return self._total


def text_of(blocks: list[Block]) -> str:
    return "\n".join(b["text"] for b in blocks if b.get("type") == "text")


# -- a repeat triggers one retry ------------------------------------------


@pytest.mark.asyncio
async def test_a_repeated_opener_triggers_one_retry() -> None:
    backend = queued(
        PhrasedLine(line="France push forward again.", excitement=0.3),
        PhrasedLine(line="Down the left now.", excitement=0.3),
    )
    phraser = a_phraser(backend)
    phraser.accept("France push forward.", Event.CARRY)

    phrased = await phraser.phrase(a_form(), "")

    assert phrased is not None
    assert phrased.line == "Down the left now."
    assert phrased.opener_retry is True
    calls = backend.calls_tagged("phraser")
    assert len(calls) == 2


def _last_text(backend: ScriptedBackend) -> str:
    return backend.calls_tagged("phraser")[-1].text


@pytest.mark.asyncio
async def test_the_retry_names_the_offending_opener_and_asks_for_a_change() -> None:
    backend = queued(
        PhrasedLine(line="France push forward again.", excitement=0.3),
        PhrasedLine(line="Down the left now.", excitement=0.3),
    )
    phraser = a_phraser(backend)
    phraser.accept("France push forward.", Event.CARRY)

    await phraser.phrase(a_form(), "")

    retry_text = _last_text(backend)
    assert '"France"' in retry_text
    assert "Open differently" in retry_text
    # the first call's body is still there — this is the same call, extended
    assert "THE LAST LINES SPOKEN" in retry_text


@pytest.mark.asyncio
async def test_the_retry_reuses_the_same_cached_system_prefix() -> None:
    backend = queued(
        PhrasedLine(line="France push forward again.", excitement=0.3),
        PhrasedLine(line="Down the left now.", excitement=0.3),
    )
    phraser = a_phraser(backend)
    phraser.accept("France push forward.", Event.CARRY)

    await phraser.phrase(a_form(), "")

    calls = backend.calls_tagged("phraser")
    assert calls[0].system == calls[1].system == phraser.system


# -- a retry that repeats again is kept -----------------------------------


@pytest.mark.asyncio
async def test_a_retry_that_repeats_again_is_kept_not_dropped() -> None:
    backend = queued(
        PhrasedLine(line="France push forward again.", excitement=0.3),
        PhrasedLine(line="France break at pace.", excitement=0.3),
    )
    phraser = a_phraser(backend)
    phraser.accept("France push forward.", Event.CARRY)

    phrased = await phraser.phrase(a_form(), "")

    assert phrased is not None
    assert phrased.line == "France break at pace."
    assert phrased.opener_retry is True
    # asked once, not twice, even though the retry still opened the same way
    assert len(backend.calls_tagged("phraser")) == 2


# -- skips: loud stacked repetition, and a bare surname --------------------


@pytest.mark.asyncio
async def test_excitement_at_or_above_point_nine_skips_the_retry() -> None:
    backend = queued(PhrasedLine(line="Mbappé! Mbappé!", excitement=0.95))
    phraser = a_phraser(backend)
    phraser.accept("Mbappé drives at goal.", Event.GOAL)

    phrased = await phraser.phrase(a_form(event=Event.GOAL), "")

    assert phrased is not None
    assert phrased.line == "Mbappé! Mbappé!"
    assert phrased.opener_retry is False
    assert len(backend.calls_tagged("phraser")) == 1


@pytest.mark.asyncio
async def test_just_below_point_nine_still_gets_a_retry() -> None:
    backend = queued(
        PhrasedLine(line="Mbappé drives again.", excitement=0.89),
        PhrasedLine(line="Straight through the middle.", excitement=0.89),
    )
    phraser = a_phraser(backend)
    phraser.accept("Mbappé drives at goal.", Event.CARRY)

    phrased = await phraser.phrase(a_form(), "")

    assert phrased is not None
    assert phrased.opener_retry is True
    assert len(backend.calls_tagged("phraser")) == 2


@pytest.mark.asyncio
async def test_a_bare_surname_skips_the_retry() -> None:
    backend = queued(PhrasedLine(line="Molina.", excitement=0.2))
    phraser = a_phraser(backend)
    phraser.accept("Molina drives forward.", Event.CARRY)

    phrased = await phraser.phrase(a_form(), "")

    assert phrased is not None
    assert phrased.line == "Molina."
    assert phrased.opener_retry is False
    assert len(backend.calls_tagged("phraser")) == 1


@pytest.mark.asyncio
async def test_a_bare_surname_with_punctuation_still_counts_as_bare() -> None:
    backend = queued(PhrasedLine(line="Molina!", excitement=0.2))
    phraser = a_phraser(backend)
    phraser.accept("Molina drives forward.", Event.CARRY)

    phrased = await phraser.phrase(a_form(), "")

    assert phrased is not None
    assert phrased.opener_retry is False
    assert len(backend.calls_tagged("phraser")) == 1


# -- no repeat, no retry ----------------------------------------------------


@pytest.mark.asyncio
async def test_no_repeated_opener_means_no_retry() -> None:
    backend = queued(PhrasedLine(line="Down the left now.", excitement=0.3))
    phraser = a_phraser(backend)
    phraser.accept("France push forward.", Event.CARRY)

    phrased = await phraser.phrase(a_form(), "")

    assert phrased is not None
    assert phrased.opener_retry is False
    assert len(backend.calls_tagged("phraser")) == 1


@pytest.mark.asyncio
async def test_a_possessive_form_of_the_same_opener_still_counts_as_a_repeat() -> None:
    """"France's" and "France" are the same opener, possessive stripped."""
    backend = queued(
        PhrasedLine(line="France's back four scramble.", excitement=0.3),
        PhrasedLine(line="Under pressure now.", excitement=0.3),
    )
    phraser = a_phraser(backend)
    phraser.accept("France push forward.", Event.CARRY)

    phrased = await phraser.phrase(a_form(), "")

    assert phrased is not None
    assert phrased.opener_retry is True
    assert len(backend.calls_tagged("phraser")) == 2


@pytest.mark.asyncio
async def test_an_opener_used_only_six_lines_back_is_not_treated_as_a_repeat() -> None:
    """Further back than five is allowed and normal — one line in eight, real."""
    backend = queued(PhrasedLine(line="France break away down the flank.", excitement=0.3))
    phraser = a_phraser(backend, recent_lines=6)
    phraser.accept("France push forward.", Event.CARRY)  # will be the 6th line back
    for i in range(5):
        phraser.accept(f"Filler line number {i}.", Event.CARRY)

    phrased = await phraser.phrase(a_form(), "")

    assert phrased is not None
    assert phrased.opener_retry is False
    assert len(backend.calls_tagged("phraser")) == 1


# -- usd sums ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_usd_of_both_calls_is_summed_on_last_usage() -> None:
    backend = CostedBackend(
        answers=[
            (PhrasedLine(line="France push forward again.", excitement=0.3), 0.0004),
            (PhrasedLine(line="Down the left now.", excitement=0.3), 0.0003),
        ]
    )
    phraser = a_phraser(backend)
    phraser.accept("France push forward.", Event.CARRY)

    phrased = await phraser.phrase(a_form(), "")

    assert phrased is not None
    assert phrased.opener_retry is True
    assert phraser.last_usage.cost_usd == pytest.approx(0.0007)
    assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_a_single_call_carries_only_its_own_usd() -> None:
    backend = CostedBackend(
        answers=[(PhrasedLine(line="Down the left now.", excitement=0.3), 0.0004)]
    )
    phraser = a_phraser(backend)
    phraser.accept("France push forward.", Event.CARRY)

    await phraser.phrase(a_form(), "")

    assert phraser.last_usage.cost_usd == pytest.approx(0.0004)
    assert len(backend.calls) == 1
