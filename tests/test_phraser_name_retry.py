"""Beat 3's own retry: a number with nobody's name on it, re-asked once.

Beat 3 of the goal follow-up (``goalfollow.py`` / ``GOAL_BEATS[3]`` in
``prompts/phraser.py``) is one number about the scorer, and the prompt
already says "Name him". About half the time the first answer does not, and
comes back a fact about nobody — "Six in the tournament now." — which the
gate's ``note_claim`` rule refuses outright as a statistic attached to no
one. This is the code fix, the same shape as the opener retry in
``test_phraser_opener.py``: after the model answers, if this was beat 3, the
scorer is known, the line carries a number and does not name him, the same
call is re-asked once with a note naming him, before the line is settled and
returned.
"""

from __future__ import annotations

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


def a_phraser(backend: ScriptedBackend) -> Phraser:
    return Phraser(
        backend,
        config=PhraserConfig(model="claude-haiku-4-5"),
        home="Argentina",
        away="France",
    )


def queued(*lines: PhrasedLine) -> ScriptedBackend:
    backend = ScriptedBackend()
    backend.queue("phraser", list(lines))
    return backend


def _last_text(backend: ScriptedBackend) -> str:
    return backend.calls_tagged("phraser")[-1].text


# -- a nameless numbered beat-3 line triggers one retry --------------------


@pytest.mark.asyncio
async def test_a_nameless_numbered_beat_3_line_triggers_one_retry() -> None:
    backend = queued(
        PhrasedLine(line="Six in the tournament now.", excitement=0.3),
        PhrasedLine(line="Mbappé's sixth in the tournament now.", excitement=0.3),
    )
    phraser = a_phraser(backend)

    phrased = await phraser.phrase(a_form(), "", goal_beat=3, scorer="Kylian Mbappé")

    assert phrased is not None
    assert phrased.line == "Mbappé's sixth in the tournament now."
    assert phrased.name_retry is True
    assert len(backend.calls_tagged("phraser")) == 2


@pytest.mark.asyncio
async def test_the_retry_names_the_scorer_and_asks_for_a_change() -> None:
    backend = queued(
        PhrasedLine(line="Six in the tournament now.", excitement=0.3),
        PhrasedLine(line="Mbappé's sixth in the tournament now.", excitement=0.3),
    )
    phraser = a_phraser(backend)

    await phraser.phrase(a_form(), "", goal_beat=3, scorer="Kylian Mbappé")

    retry_text = _last_text(backend)
    assert "Kylian Mbappé" in retry_text
    assert "nobody's name" in retry_text.lower()
    # the first call's body is still there — this is the same call, extended
    assert "THE LAST LINES SPOKEN" in retry_text or "Molina" in retry_text


@pytest.mark.asyncio
async def test_the_retry_reuses_the_same_cached_system_prefix() -> None:
    backend = queued(
        PhrasedLine(line="Six in the tournament now.", excitement=0.3),
        PhrasedLine(line="Mbappé's sixth in the tournament now.", excitement=0.3),
    )
    phraser = a_phraser(backend)

    await phraser.phrase(a_form(), "", goal_beat=3, scorer="Kylian Mbappé")

    calls = backend.calls_tagged("phraser")
    assert calls[0].system == calls[1].system == phraser.system


@pytest.mark.asyncio
async def test_a_retry_that_is_still_nameless_is_kept_not_dropped() -> None:
    backend = queued(
        PhrasedLine(line="Six in the tournament now.", excitement=0.3),
        PhrasedLine(line="Seven in the tournament now.", excitement=0.3),
    )
    phraser = a_phraser(backend)

    phrased = await phraser.phrase(a_form(), "", goal_beat=3, scorer="Kylian Mbappé")

    assert phrased is not None
    assert phrased.line == "Seven in the tournament now."
    assert phrased.name_retry is True
    # asked once, not twice, even though the retry still named nobody
    assert len(backend.calls_tagged("phraser")) == 2


# -- skips -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_beat_3_line_that_already_names_the_scorer_skips_the_retry() -> None:
    backend = queued(PhrasedLine(line="Mbappé's sixth in the tournament now.", excitement=0.3))
    phraser = a_phraser(backend)

    phrased = await phraser.phrase(a_form(), "", goal_beat=3, scorer="Kylian Mbappé")

    assert phrased is not None
    assert phrased.name_retry is False
    assert len(backend.calls_tagged("phraser")) == 1


@pytest.mark.asyncio
async def test_the_scorers_surname_alone_counts_as_naming_him() -> None:
    """The gate's own name matching is generous about surname-only reads."""
    backend = queued(PhrasedLine(line="Mbappé's sixth in the tournament.", excitement=0.3))
    phraser = a_phraser(backend)

    phrased = await phraser.phrase(a_form(), "", goal_beat=3, scorer="Kylian Mbappé")

    assert phrased is not None
    assert phrased.name_retry is False


@pytest.mark.asyncio
async def test_a_beat_3_line_with_no_number_skips_the_retry() -> None:
    """No figure, nothing for ``note_claim`` to attach to nobody — no retry needed."""
    backend = queued(PhrasedLine(line="He has been here before.", excitement=0.3))
    phraser = a_phraser(backend)

    phrased = await phraser.phrase(a_form(), "", goal_beat=3, scorer="Kylian Mbappé")

    assert phrased is not None
    assert phrased.name_retry is False
    assert len(backend.calls_tagged("phraser")) == 1


@pytest.mark.asyncio
async def test_a_numbered_line_outside_beat_3_skips_the_retry() -> None:
    """Only beat 3 is one number about the scorer; the other beats carry none."""
    backend = queued(PhrasedLine(line="Six in the tournament now.", excitement=0.3))
    phraser = a_phraser(backend)

    phrased = await phraser.phrase(a_form(), "", goal_beat=2, scorer="Kylian Mbappé")

    assert phrased is not None
    assert phrased.name_retry is False
    assert len(backend.calls_tagged("phraser")) == 1


@pytest.mark.asyncio
async def test_no_beat_at_all_skips_the_retry() -> None:
    backend = queued(PhrasedLine(line="Six in the tournament now.", excitement=0.3))
    phraser = a_phraser(backend)

    phrased = await phraser.phrase(a_form(), "", goal_beat=None, scorer="Kylian Mbappé")

    assert phrased is not None
    assert phrased.name_retry is False
    assert len(backend.calls_tagged("phraser")) == 1


@pytest.mark.asyncio
async def test_no_scorer_known_skips_the_retry_even_on_beat_3() -> None:
    """``goalfollow.py`` has no name to ask for; nothing to re-ask about."""
    backend = queued(PhrasedLine(line="Six in the tournament now.", excitement=0.3))
    phraser = a_phraser(backend)

    phrased = await phraser.phrase(a_form(), "", goal_beat=3, scorer=None)

    assert phrased is not None
    assert phrased.name_retry is False
    assert len(backend.calls_tagged("phraser")) == 1


@pytest.mark.asyncio
async def test_a_chosen_silence_on_beat_3_skips_the_retry() -> None:
    """An empty line carries no number and is not a nameless statistic."""
    backend = queued(PhrasedLine(line="", excitement=0.0))
    phraser = a_phraser(backend)

    phrased = await phraser.phrase(a_form(), "", goal_beat=3, scorer="Kylian Mbappé")

    assert phrased is not None
    assert phrased.name_retry is False
    assert len(backend.calls_tagged("phraser")) == 1


# -- interaction with the opener retry --------------------------------------


@pytest.mark.asyncio
async def test_a_repeated_opener_and_a_nameless_beat_3_line_both_retry() -> None:
    """The two checks are independent and can both fire on the same call."""
    backend = queued(
        PhrasedLine(line="France six in the tournament now.", excitement=0.3),
        PhrasedLine(line="Still six in the tournament now.", excitement=0.3),
        PhrasedLine(line="Mbappé's sixth in the tournament now.", excitement=0.3),
    )
    phraser = a_phraser(backend)
    phraser.accept("France push forward.", Event.CARRY)

    phrased = await phraser.phrase(a_form(), "", goal_beat=3, scorer="Kylian Mbappé")

    assert phrased is not None
    assert phrased.line == "Mbappé's sixth in the tournament now."
    assert phrased.opener_retry is True
    assert phrased.name_retry is True
    assert len(backend.calls_tagged("phraser")) == 3


# -- usd sums ----------------------------------------------------------------


class CostedBackend:
    """A minimal backend whose queued answers each carry their own cost.

    :class:`ScriptedBackend` always reports ``cost_usd=0.0`` — its handlers
    choose the value, not the price — so it cannot show two calls' costs
    adding up. Copied from ``test_phraser_opener.py``, which does the one
    thing that test needs and nothing else.
    """

    def __init__(self, answers: list[tuple[PhrasedLine, float]]) -> None:
        self.answers = answers
        self.calls: list[list[Block]] = []
        self._total = Usage()

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


@pytest.mark.asyncio
async def test_the_usd_of_both_calls_is_summed_on_last_usage() -> None:
    backend = CostedBackend(
        answers=[
            (PhrasedLine(line="Six in the tournament now.", excitement=0.3), 0.0004),
            (PhrasedLine(line="Mbappé's sixth in the tournament now.", excitement=0.3), 0.0003),
        ]
    )
    phraser = a_phraser(backend)  # type: ignore[arg-type]

    phrased = await phraser.phrase(a_form(), "", goal_beat=3, scorer="Kylian Mbappé")

    assert phrased is not None
    assert phrased.name_retry is True
    assert phraser.last_usage.cost_usd == pytest.approx(0.0007)
    assert len(backend.calls) == 2
