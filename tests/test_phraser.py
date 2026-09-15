"""The phrasing stage: the caller's form, said the way a commentator says it.

The thing being protected here is not the register — a prompt cannot be
unit-tested into sounding right, and what it sounds like is in
``runs/rephrased/``. It is everything around the register: that the gate sees
what is actually going to the speaker, that the stage can be switched off and
leave nothing behind, and that a phraser which falls over costs a rewrite and
never a line.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from commentary.agents.phraser import Phraser, phraser_enabled
from commentary.capture.buffer import Frame
from commentary.config import (
    CallerConfig,
    CaptureConfig,
    DirectorConfig,
    PhraserConfig,
    PredictorConfig,
    Settings,
)
from commentary.llm.base import Block
from commentary.llm.fake import ScriptedBackend
from commentary.prompts.commentary_examples import EXAMPLES, KINDS
from commentary.prompts.phraser import (
    GOAL_BEATS,
    PHRASER_RULES,
    _examples,
    notes_allowed,
    phraser_blocks,
    phraser_system,
    replay_block,
)
from commentary.rephrase import rephrase
from commentary.runtime import Runtime
from commentary.schemas import (
    Beat,
    CallerLine,
    Event,
    Incident,
    KnowledgePack,
    MatchState,
    Note,
    PhrasedLine,
    Player,
    Scene,
    Side,
    Sighting,
    TeamSheet,
    Trigger,
)
from commentary.sim import MatchSim, SimOracle, SimSource
from commentary.state import notes_for
from commentary.trace import read_trace
from commentary.voice import LogSpeaker

# -- fixtures ----------------------------------------------------------------


def a_form(
    line: str = "Molina moves forward down the right and looks for a way inside.",
    *,
    event: Event = Event.CARRY,
    sightings: list[Sighting] | None = None,
    detail: str | None = None,
) -> CallerLine:
    return CallerLine(
        scene=Scene.LIVE_PLAY,
        event=event,
        side=Side.HOME,
        team="Argentina",
        sightings=sightings if sightings is not None else [Sighting(number=26, name="Molina")],
        confidence=0.7,
        speak=True,
        line=line,
        detail=detail,
    )


def a_pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Argentina",
            short="ARG",
            demonym="Argentine",
            starters=[
                Player(name="Nahuel Molina", number=26),
                Player(name="Lionel Messi", number=10),
            ],
        ),
        away=TeamSheet(
            name="France",
            short="FRA",
            demonym="French",
            starters=[Player(name="Kylian Mbappé", number=10)],
        ),
    )


#: Every surname on the roster of ``clips/pack-argfra-2022.json``, restated
#: here rather than read off disk because ``clips/`` is where the broadcast
#: footage lives and is not in the repository (see ``argfra_notes`` below).
#: This is the pack the phraser is measured against, so a worked example in
#: the rules that pairs one of these names with a made-up outcome is a name
#: the model can echo verbatim onto a real player in a real call — which is
#: exactly what happened on the Mbappé trace (82.5 s copied the "kept" line
#: for "Mbappé! The volley, buried." onto a penalty, and 94.8 s copied the
#: beat-4 rewrite onto the wrong goal).
ARGFRA_2022_SURNAMES = frozenset(
    {
        "Acuña",
        "Almada",
        "Areola",
        "Armani",
        "Camavinga",
        "Coman",
        "Correa",
        "Dembélé",
        "Disasi",
        "Dybala",
        "Fernandez",
        "Fofana",
        "Foyth",
        "Giroud",
        "Griezmann",
        "Guendouzi",
        "Gómez",
        "Hernández",
        "Konaté",
        "Koundé",
        "Lloris",
        "MacAllister",
        "Mandanda",
        "Martínez",
        "María",
        "Mbappé",
        "Messi",
        "Molina",
        "Montiel",
        "Muani",
        "Otamendi",
        "Palacios",
        "Paredes",
        "Paul",
        "Pavard",
        "Pezzella",
        "Rabiot",
        "Rodríguez",
        "Romero",
        "Rulli",
        "Saliba",
        "Tagliafico",
        "Tchouaméni",
        "Thuram",
        "Upamecano",
        "Varane",
        "Veretout",
        "Álvarez",
    }
)

#: Surnames from that roster that legitimately still appear in the rules or
#: the goal follow-up block: bare, generic name mentions and genuine verbatim
#: real-broadcast quotes sourced in ``docs/research/real-commentary-corpus.md``
#: (the same "real captions from other matches" exemption the REAL EXAMPLES
#: block in ``commentary_examples.py`` gets, just inline instead of in that
#: file). None of them pairs the name with an invented, specific outcome the
#: way the fixed examples used to. Pinned counts so a new example that sneaks
#: a pack name back in has to touch this list to pass.
ARGFRA_2022_EXEMPT_IN_RULES = {
    "Tagliafico": 1,  # "the form gives you a name, use it" — bare mention
    "Griezmann": 2,  # opener-shape table, sourced from real-commentary-corpus.md
    "Varane": 1,  # "at speed it is still short" — bare fragment, no outcome
    "Messi": 3,  # opener-shape table + "is offside" fragment — bare mentions
    "Paul": 2,  # "a bare surname is a line" — bare mention, no outcome
}
ARGFRA_2022_EXEMPT_IN_GOAL_BEATS = {
    "Griezmann": 3,  # verbatim bar-mal-2019 quotes, section 8.4
}


def _surname_counts(text: str, surnames: frozenset[str]) -> dict[str, int]:
    return {
        surname: len(re.findall(r"\b" + re.escape(surname) + r"\b", text))
        for surname in surnames
        if re.search(r"\b" + re.escape(surname) + r"\b", text)
    }


def test_no_worked_example_pairs_an_invented_detail_with_a_name_from_the_measured_pack() -> None:
    rules_text = PHRASER_RULES.format(max_words=28)
    beats_text = "\n".join(GOAL_BEATS.values())

    assert _surname_counts(rules_text, ARGFRA_2022_SURNAMES) == ARGFRA_2022_EXEMPT_IN_RULES
    assert _surname_counts(beats_text, ARGFRA_2022_SURNAMES) == ARGFRA_2022_EXEMPT_IN_GOAL_BEATS


def saying(*lines: PhrasedLine) -> ScriptedBackend:
    backend = ScriptedBackend()
    backend.queue("phraser", list(lines))
    return backend


def a_phraser(backend: ScriptedBackend) -> Phraser:
    return Phraser(
        backend,
        config=PhraserConfig(model="claude-haiku-4-5"),
        home="Argentina",
        away="France",
    )


def text_of(blocks: list[Block]) -> str:
    return "\n".join(b["text"] for b in blocks if b.get("type") == "text")


# -- the prompt --------------------------------------------------------------


def test_the_system_prompt_carries_real_utterances_from_every_kind() -> None:
    system = phraser_system()
    for kind in KINDS:
        assert EXAMPLES[kind], f"{kind} has no examples"
        assert EXAMPLES[kind][0] in system, f"{kind} is not shown in the prompt"


def test_the_system_prompt_is_the_same_bytes_every_time() -> None:
    assert phraser_system() == phraser_system()


def test_the_prompt_shows_real_replay_talk() -> None:
    """Replay mode's whole teaching material is this bucket.

    Nothing else in the set is in the past tense, so there is no neighbouring
    kind to borrow the register from. Section 3.2 of the corpus study is where
    these come from.
    """
    system = phraser_system()
    assert "replay" in KINDS
    assert "A replay, talked over in the past tense" in system
    assert EXAMPLES["replay"], "the replay bucket is empty"
    assert any("replay" in text.lower() for text in EXAMPLES["replay"])


def test_a_replay_form_gets_the_replay_block_and_not_the_goal_followup() -> None:
    """Beat 4 and a replay rebuild are the same line; two of them is one too many."""
    body = text_of(
        phraser_blocks(
            a_form("The leg was in behind him.", event=Event.FOUL).model_copy(
                update={"scene": Scene.REPLAY}
            ),
            "Argentina 1-0 France",
            [],
            home="Argentina",
            away="France",
            followup=GOAL_BEATS[4],
        )
    )
    assert "THIS IS A REPLAY OF A FOUL" in body
    assert "PAST TENSE FROM THE FIRST VERB" in body
    assert "REBUILD THE MOVE" not in body


def test_the_replay_is_named_once_and_only_in_the_first_line_of_a_sequence() -> None:
    """"Watch this." opens a sequence; the lines after it go straight at it."""
    first = replay_block(Event.GOAL, first=True)
    later = replay_block(Event.GOAL, first=False)
    assert "THIS IS THE FIRST LINE OF THIS REPLAY" in first
    assert "having seen the replay" in first
    assert "THIS REPLAY HAS ALREADY BEEN NAMED" in later
    assert "do not say so a second time" in later


def test_a_replay_line_is_offered_no_number_at_all() -> None:
    """Never a tally on a replay, so the clauses are not offered rather than refused."""
    quiet = a_form(event=Event.BUILD_UP)
    assert notes_allowed(quiet) is True
    assert notes_allowed(quiet.model_copy(update={"scene": Scene.REPLAY})) is False


def test_a_replay_takes_the_plain_state_heading_even_on_a_goal() -> None:
    """No score is appended to a replay line, so nothing may promise one."""
    body = text_of(
        phraser_blocks(
            a_form("It had gone in off the post.", event=Event.GOAL).model_copy(
                update={"scene": Scene.REPLAY}
            ),
            "Argentina 1-0 France",
            [],
            home="Argentina",
            away="France",
        )
    )
    assert "Never say the score or the clock." in body
    assert "the broadcast\nappends it after your words" not in body


def test_the_call_shows_the_form_the_state_and_what_was_just_said() -> None:
    body = text_of(
        phraser_blocks(
            a_form(),
            "Argentina 2-0 France\n78:09",
            ["Fernández.", "Now De Paul."],
            home="Argentina",
            away="France",
        )
    )
    assert "Molina" in body
    assert "carry" in body
    assert "Argentina 2-0 France" in body
    assert "Now De Paul." in body
    assert "Molina moves forward down the right" in body


def test_the_call_carries_no_pictures() -> None:
    """A phraser with a frame describes the frame, which is the bug."""
    blocks = phraser_blocks(a_form(), "", [], home="Argentina", away="France")
    assert all(block.get("type") == "text" for block in blocks)


def test_a_form_that_named_nobody_says_so_rather_than_saying_nothing() -> None:
    body = text_of(
        phraser_blocks(
            a_form(sightings=[]), "", [], home="Argentina", away="France"
        )
    )
    assert "none read off the picture" in body


def test_the_name_carried_from_the_last_line_is_offered_as_usable() -> None:
    body = text_of(
        phraser_blocks(
            a_form(sightings=[]),
            "",
            [],
            home="Argentina",
            away="France",
            on_the_ball="Messi",
        )
    )
    assert "Messi" in body


def test_every_example_is_shorter_than_a_commentator_ever_goes() -> None:
    """Twenty-eight words was the longest thing said in half an hour."""
    for kind in KINDS:
        for text in EXAMPLES[kind]:
            assert len(text.split()) < 30, f"{kind}: {text!r}"
            assert text.strip() == text


def test_the_example_set_is_big_enough_to_be_a_register_and_small_enough_to_cache() -> None:
    """The file is the corpus; the prompt is a sample of it.

    The bound moved from 300 to 600 when the corpus study split eight kinds
    into twenty-one (its Gap 8) and pooled six matches instead of one. What
    has to stay small is the *prompt*, not the file — :data:`SHOWN` decides
    how many of each kind are printed — so the second assertion here is the
    one that guards the cache.
    """
    total = sum(len(EXAMPLES[kind]) for kind in KINDS)
    assert 300 <= total <= 600, total
    assert len(_examples(10)) < 8000, len(_examples(10))


def test_the_example_set_holds_enough_goals_and_chances_to_teach_one() -> None:
    """The loud moments are the ones the phraser was flattening to a surname."""
    loud = sum(len(EXAMPLES[kind]) for kind in ("goal", "shot", "save"))
    assert loud >= 30, loud


# -- keeping the detail, and whose line it is --------------------------------


def test_the_rules_ask_for_one_concrete_detail_and_say_which_lines_were_thin() -> None:
    """"Okafor!" over a volley is true, short, and says nothing."""
    rules = phraser_system()
    assert "COMPRESSING IS NOT DELETING" in rules
    assert "One, not two, and not the whole clause" in rules
    assert "kept:  Okafor! Off the ground!" in rules
    assert "kept:  France, through the middle at speed." in rules


def test_the_rules_forbid_a_subject_that_is_only_on_the_sightings_list() -> None:
    """A France break called "Harlow." because Harlow's shirt was legible."""
    rules = phraser_system()
    assert "WHO THE LINE IS ABOUT" in rules
    assert "may not be your subject" in rules
    assert "Never the\nsubject of the line." in rules


def test_the_goal_example_is_the_name_then_the_how_and_stops() -> None:
    """The third beat is still the score. It is no longer the model's to write."""
    rules = phraser_system()
    assert "A GOAL IS THREE BEATS, AND YOU WRITE TWO OF THEM" in rules
    assert "<Scorer>! On the volley!" in rules
    assert "<Scorer>! Over the wall!" in rules
    # The examples stop where the words stop. A number in one of them is a
    # number the model has been shown and will copy.
    assert "<Scorer>! On the volley! Two-two." not in rules
    assert "<Scorer>! Over the wall! Three-three." not in rules
    assert "THE THIRD BEAT IS THE SCORE AND IT IS NOT YOURS" in rules
    assert "The broadcast adds it to" in rules


def test_the_rules_ask_the_excitement_and_the_words_to_move_together() -> None:
    rules = phraser_system()
    assert "EXCITEMENT, AND IT HAS TO MOVE" in rules
    assert "a break at speed, a run at a defender" in rules
    assert "DO NOT SOUND LIKE THE LINE BEFORE IT" in rules
    assert "same word as either of the two above it" in rules


def test_the_decoration_ban_names_erupts_for_any_group_not_only_the_crowd() -> None:
    """86.8 said "the whole crowd erupts", 180.5 said "the corner erupts" —
    both atmosphere decoration the rule already forbade in spirit but never
    named, and naming only "crowd" and "corner" left "the bench erupts"
    untouched on the next round."""
    rules = " ".join(phraser_system().split())
    assert "Never decorate" in rules
    assert "the crowd rises" in rules
    assert "any group of people made to erupt" in rules
    assert "The whole crowd erupts" in rules
    assert "and the corner erupts" in rules
    assert "the bench erupts" in rules


def test_a_form_that_carries_a_detail_puts_it_in_front_of_the_phraser() -> None:
    body = text_of(
        phraser_blocks(
            a_form(
                "Mbappé hooks it out of the air and the net bulges behind Martínez.",
                event=Event.GOAL,
                detail="off the ground, on the volley",
            ),
            "Argentina 2-1 France\n81:03",
            [],
            home="Argentina",
            away="France",
        )
    )
    assert "detail: off the ground, on the volley" in body
    assert "could not guess. Keep it." in body


def test_a_form_without_one_says_nothing_about_a_detail() -> None:
    """Older traces have no such field, and an empty label teaches nothing."""
    body = text_of(phraser_blocks(a_form(), "", [], home="Argentina", away="France"))
    assert "detail:" not in body
    assert "keeping one concrete detail" in body


def test_a_goal_is_told_the_score_is_appended_for_it() -> None:
    """Everywhere else MATCH STATE is context and repeating it is a score claim."""
    ordinary = text_of(phraser_blocks(a_form(), "Argentina 2-1 France", [], home="A", away="F"))
    assert "for context only. Never say the score" in ordinary

    goal = text_of(
        phraser_blocks(
            a_form("Mbappé turns away, arms wide.", event=Event.GOAL),
            "Argentina 2-1 France",
            [],
            home="A",
            away="F",
        )
    )
    # A goal used to be the one moment the model was allowed a number. It is
    # not any more: the heading tells it the score is already going out.
    assert "the broadcast\nappends it after your words" in goal
    assert "Write no number at all." in goal
    assert "Never say the score" not in goal


# -- the agent ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_phraser_returns_the_line_the_model_wrote() -> None:
    phraser = a_phraser(saying(PhrasedLine(line="Molina, forward.", excitement=0.3)))
    phrased = await phraser.phrase(a_form(), "")
    assert phrased is not None
    assert phrased.line == "Molina, forward."
    assert phrased.excitement == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_a_phrased_line_is_stripped_and_capped_like_the_callers() -> None:
    phraser = a_phraser(saying(PhrasedLine(line='"Commentary: Molina."', excitement=0.2)))
    phrased = await phraser.phrase(a_form(), "")
    assert phrased is not None
    assert phrased.line == "Molina."


@pytest.mark.asyncio
async def test_a_failed_call_is_a_reason_not_an_exception() -> None:
    phraser = a_phraser(ScriptedBackend())  # no handler registered for "phraser"
    assert await phraser.phrase(a_form(), "") is None
    assert "model call failed" in phraser.last_reason


@pytest.mark.asyncio
async def test_what_was_said_is_shown_back_so_the_voice_does_not_repeat_itself() -> None:
    backend = saying(PhrasedLine(line="Molina.", excitement=0.2))
    phraser = a_phraser(backend)
    phraser.accept("Fernández.")
    await phraser.phrase(a_form(), "")
    assert "Fernández." in backend.calls_tagged("phraser")[0].text


def test_the_stage_is_off_when_the_model_says_off() -> None:
    assert phraser_enabled("claude-haiku-4-5")
    assert not phraser_enabled("off")
    assert not phraser_enabled("OFF")
    assert not phraser_enabled("")


# -- the runtime -------------------------------------------------------------


def fast_settings(**kw: Any) -> Settings:
    return Settings(
        capture=CaptureConfig(
            width=640, height=360, fps=8, delay_s=4.0, history_s=3.0, present_offset_s=3.0
        ),
        caller=CallerConfig(min_gap_s=2.0),
        predictor=PredictorConfig(tick_s=0.05),
        director=DirectorConfig(max_beat_age_s=30.0),
        **kw,
    )


def a_runtime(settings: Settings | None = None) -> Runtime:
    """A runtime holding the sim's teams, with frames in the buffer and no clock.

    Built rather than run. ``Runtime.run`` stops on a wall-clock deadline
    while the sim source pours frames in as fast as the machine will take
    them, so two runs of the same seed with the same settings produce
    different commentary — measured at nineteen beats against nine. That is
    fine for the tests that ask "did anything reasonable happen" and useless
    for one that asks "is this identical", so here the frames go in by hand
    and the calls are made one at a time.
    """
    sim = MatchSim(seed=5, duration_s=120.0)
    chosen = settings or fast_settings()
    runtime = Runtime(
        source=SimSource(sim, chosen.capture, realtime=False),
        backend=SimOracle(sim=sim),
        pack=sim.knowledge_pack,
        settings=chosen,
        speaker=LogSpeaker(words_per_second=120),
    )
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    for index in range(96):
        runtime.buffer.append(Frame(ts=index / chosen.capture.fps, image=blank))
    return runtime


def caught_beats(runtime: Runtime) -> list[Beat]:
    """Wrap the director so a test can see what it was handed."""
    beats: list[Beat] = []
    original = runtime.director.submit

    def spy(beat: Beat) -> None:
        beats.append(beat)
        original(beat)

    runtime.director.submit = spy  # type: ignore[method-assign]
    return beats


def caught_gate(runtime: Runtime) -> list[CallerLine]:
    judged: list[CallerLine] = []
    original = runtime.gate.judge

    def spy(line: CallerLine, *args: Any, **kw: Any) -> Any:
        judged.append(line)
        return original(line, *args, **kw)

    runtime.gate.judge = spy  # type: ignore[method-assign]
    return judged


def speaking(runtime: Runtime, line: CallerLine) -> None:
    """Make the caller return this form and nothing else."""

    async def call(*_args: Any, **_kw: Any) -> CallerLine:
        return line

    runtime.caller.call = call  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_the_gate_judges_the_phrased_line_not_the_callers() -> None:
    runtime = a_runtime()
    assert runtime.pack is not None
    player = runtime.pack.home.starters[0]
    form = a_form(
        line=f"{player.surname} carries it forward past two challenges towards the box.",
        sightings=[Sighting(number=player.number, name=player.name, side=Side.HOME)],
    )
    speaking(runtime, form)
    runtime.phraser = a_phraser(
        saying(PhrasedLine(line=f"{player.surname}, away.", excitement=0.4))
    )
    judged, beats = caught_gate(runtime), caught_beats(runtime)

    await runtime._call([Trigger.SCHEDULED])

    assert [line.line for line in judged] == [f"{player.surname}, away."]
    assert [beat.text for beat in beats] == [f"{player.surname}, away."]
    assert beats[0].excitement == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_a_phraser_that_fails_costs_a_rewrite_and_never_a_line() -> None:
    runtime = a_runtime()
    assert runtime.pack is not None
    player = runtime.pack.home.starters[0]
    form = a_form(
        line=f"{player.surname} carries it forward past two challenges towards the box.",
        sightings=[Sighting(number=player.number, name=player.name, side=Side.HOME)],
    )
    speaking(runtime, form)
    runtime.phraser = a_phraser(ScriptedBackend())
    beats = caught_beats(runtime)

    await runtime._call([Trigger.SCHEDULED])

    assert [beat.text for beat in beats] == [form.line]
    assert beats[0].excitement == 0.0


def _a_carrying_form(runtime: Runtime) -> CallerLine:
    assert runtime.pack is not None
    player = runtime.pack.home.starters[0]
    return a_form(
        line=f"{player.surname} carries it forward past two challenges towards the box.",
        sightings=[Sighting(number=player.number, name=player.name, side=Side.HOME)],
    )


def _watch_published(runtime: Runtime) -> list[tuple[str, dict[str, Any]]]:
    """Every ``_publish`` the runtime makes, topic and keywords."""
    seen: list[tuple[str, dict[str, Any]]] = []
    original = runtime._publish

    def spy(topic: Any, ts: float, value: Any = None, **extra: Any) -> None:
        seen.append((str(getattr(topic, "value", topic)), extra))
        original(topic, ts, value, **extra)

    runtime._publish = spy  # type: ignore[method-assign]
    return seen


@pytest.mark.asyncio
async def test_a_phraser_that_chose_silence_speaks_no_line_at_all() -> None:
    """A quarter of build-up touches pass in silence in real commentary.

    ``docs/research/real-commentary-corpus.md`` section 3: 24% of carries and
    passes have nothing said within three seconds, 43% of goal kicks and 37%
    of throw-in deliveries. The phraser could never do that — an empty line
    fell back to the caller's words — so the system spoke on every call it
    made. An empty line is now a choice, and it produces no beat, no gate
    row, and a ``phrased`` row that says silence was chosen.
    """
    runtime = a_runtime()
    form = _a_carrying_form(runtime)
    speaking(runtime, form)
    runtime.phraser = a_phraser(saying(PhrasedLine(line="", excitement=0.0)))
    beats = caught_beats(runtime)
    published = _watch_published(runtime)

    await runtime._call([Trigger.SCHEDULED])

    assert beats == []
    phrased = [extra for topic, extra in published if topic == "phrased"]
    assert phrased and phrased[0]["line"] == ""
    assert "silence" in phrased[0]["reason"]
    assert not [topic for topic, _ in published if topic == "gate"]


@pytest.mark.asyncio
async def test_a_phrasing_that_cleaned_away_to_nothing_still_falls_back() -> None:
    """Silence is an empty answer, not an answer that survived nothing.

    A model that writes a label and no line has failed, and the caller's own
    words go out as they always did. Only a model that returns nothing has
    decided anything.
    """
    runtime = a_runtime()
    form = _a_carrying_form(runtime)
    speaking(runtime, form)
    runtime.phraser = a_phraser(saying(PhrasedLine(line='"Commentary:"', excitement=0.9)))
    beats = caught_beats(runtime)

    await runtime._call([Trigger.SCHEDULED])

    assert [beat.text for beat in beats] == [form.line]


@pytest.mark.asyncio
async def test_the_stage_switched_off_leaves_the_runtime_exactly_as_it_was() -> None:
    """``PHRASER_MODEL=off`` is not a degraded mode; it is the old runtime.

    The same sim forms go through two runtimes that differ only in the
    setting. The one with the stage on has the sim's oracle behind it, and
    the oracle answers by decoding a timestamp out of a picture — the phraser
    sends no pictures — so every call fails and every line falls back. That
    is the second half of what this asserts: a stage that is absent and a
    stage that has fallen over produce the same commentary, down to the
    event, the excitement and whether the director may cut it off.
    """
    arms: list[list[tuple[str, Event, float, bool]]] = []
    for model in ("off", "claude-haiku-4-5"):
        runtime = a_runtime(fast_settings(phraser=PhraserConfig(model=model)))
        assert (runtime.phraser is None) == (model == "off")
        beats = caught_beats(runtime)
        for form in sim_forms(runtime):
            speaking(runtime, form)
            await runtime._call([Trigger.SCHEDULED])
        arms.append([(b.text, b.event, b.excitement, b.preemptable) for b in beats])

    assert arms[0], "nothing was spoken, so nothing was compared"
    assert arms[0] == arms[1]


def sim_forms(runtime: Runtime) -> list[CallerLine]:
    """A handful of caller forms off the sim's own team sheets."""
    assert runtime.pack is not None
    home = runtime.pack.home.starters[0]
    away = runtime.pack.away.starters[0]
    return [
        a_form(
            line=f"{home.surname} collects it inside his own half and turns upfield.",
            sightings=[Sighting(number=home.number, name=home.name, side=Side.HOME)],
        ),
        a_form(
            line=f"{away.surname} closes him down and wins it back near the touchline.",
            event=Event.TACKLE,
            sightings=[Sighting(number=away.number, name=away.name, side=Side.AWAY)],
        ),
        a_form(
            line=f"{home.surname} is away again down the right with space to run into.",
            sightings=[Sighting(number=home.number, name=home.name, side=Side.HOME)],
        ),
    ]


# -- rephrase ----------------------------------------------------------------


def a_trace() -> list[dict[str, Any]]:
    """Four rows of a run: one state, one spoken caller line, one analyst beat."""
    return [
        {
            "topic": "state",
            "ts": 0.5,
            "home": "Argentina",
            "away": "France",
            "home_score": 1,
            "away_score": 0,
            "clock": "78:09",
            "period": 2,
        },
        {"topic": "board", "ts": 1.0, "bug_visible": True, "confidence": 0.9},
        {
            "topic": "caller",
            "ts": 10.0,
            "scene": "live_play",
            "event": "carry",
            "side": "home",
            "team": "Argentina",
            "sightings": [{"number": 26, "name": "Molina", "side": "home"}],
            "confidence": 0.7,
            "speak": True,
            "line": "Molina drives forward down the right, looking for a way inside.",
        },
        {
            "topic": "gate",
            "ts": 10.0,
            "passed": True,
            "reasons": [],
            "line": "Molina drives forward down the right, looking for a way inside.",
            "event": "carry",
        },
        {
            "topic": "beat",
            "ts": 10.0,
            "id": "b1",
            "voice": "caller",
            "text": "Molina drives forward down the right, looking for a way inside.",
            "video_ts": 10.0,
            "created_ts": 1.0,
            "live_ts": 13.0,
            "event": "carry",
            "preemptable": True,
        },
        {
            "topic": "spoken",
            "ts": 10.0,
            "id": "b1",
            "voice": "caller",
            "text": "Molina drives forward down the right, looking for a way inside.",
            "video_ts": 10.0,
            "created_ts": 1.0,
            "spoken": "Molina drives forward down the right, looking for a way inside.",
            "seconds": 3.4,
        },
        {
            "topic": "caller",
            "ts": 20.0,
            "scene": "live_play",
            "event": "shot",
            "side": "home",
            "team": "Argentina",
            "sightings": [{"number": 10, "name": "Messi", "side": "home"}],
            "confidence": 0.8,
            "speak": True,
            "line": "Messi shapes and strikes it towards the near post from the edge.",
        },
        {
            "topic": "beat",
            "ts": 20.0,
            "id": "b2",
            "voice": "caller",
            "text": "Messi shapes and strikes it towards the near post from the edge.",
            "video_ts": 20.0,
            "created_ts": 1.0,
            "live_ts": 23.0,
            "event": "shot",
            "preemptable": True,
        },
        {
            "topic": "beat",
            "ts": 30.0,
            "id": "b3",
            "voice": "analyst",
            "text": "Argentina have found the overload down that right side all evening.",
            "video_ts": 30.0,
            "created_ts": 1.0,
            "live_ts": 33.0,
            "event": "none",
            "preemptable": True,
        },
        {"topic": "cost", "ts": 30.0, "total_usd": 0.41},
    ]


def rows_of(rows: list[dict[str, Any]], topic: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("topic") == topic]


@pytest.mark.asyncio
async def test_rephrasing_rewrites_the_caller_beats_and_leaves_the_analyst_alone() -> None:
    backend = saying(
        PhrasedLine(line="Molina, down the right.", excitement=0.3),
        PhrasedLine(line="Messi strikes.", excitement=0.75),
    )
    result = await rephrase(a_trace(), backend, pack=a_pack())

    beats = {row["id"]: row["text"] for row in rows_of(result.rows, "beat")}
    assert beats["b1"] == "Molina, down the right."
    assert beats["b2"] == "Messi strikes."
    assert beats["b3"] == "Argentina have found the overload down that right side all evening."


@pytest.mark.asyncio
async def test_rephrasing_drops_the_speaking_rows_and_keeps_everything_else() -> None:
    backend = saying(PhrasedLine(line="Molina, down the right.", excitement=0.3))
    result = await rephrase(a_trace(), backend, pack=a_pack())

    assert rows_of(result.rows, "spoken") == []
    assert rows_of(result.rows, "preempted") == []
    assert rows_of(result.rows, "board") == rows_of(a_trace(), "board")
    assert rows_of(result.rows, "state") == rows_of(a_trace(), "state")
    assert rows_of(result.rows, "cost") == rows_of(a_trace(), "cost")
    assert len(rows_of(result.rows, "phrased")) == 2


@pytest.mark.asyncio
async def test_a_phrased_line_that_claims_a_score_loses_the_number_not_the_line() -> None:
    """An ordinal is a scoreline with the number left out, and it comes out.

    This test used to assert the opposite: the gate refused the line and the
    beat never happened. That is still the right answer *at the gate* — there
    is no trimming a number out of a sentence once it has got that far — but
    a line is no longer allowed to reach the gate with a number in it. Code
    takes the claim out first, and what is left is the words the model was
    asked for. The words here were "Molina!" and they are worth keeping.
    """
    backend = saying(
        PhrasedLine(line="Molina! Argentina's third.", excitement=1.0),
        PhrasedLine(line="Messi strikes.", excitement=0.75),
    )
    result = await rephrase(a_trace(), backend, pack=a_pack())

    beats = {row["id"]: row["text"] for row in rows_of(result.rows, "beat")}
    assert beats["b1"] == "Molina!"
    assert "b2" in beats

    phrased = rows_of(result.rows, "phrased")[0]
    assert phrased["score_stripped"] == ["Argentina's third"]
    # A carry is not a goal, so nothing is appended in its place: the one
    # scoreline a match gets per goal belongs to the goal.
    assert phrased["score_appended"] == ""

    assert [row for row in rows_of(result.rows, "gate") if row.get("where") == "rephrase"] == []
    assert result.rejected == 0
    assert result.lines[0].stripped == ("Argentina's third",)


def test_the_score_is_settled_only_once_the_state_has_taken_the_goal_in() -> None:
    """The gate is handed the state row at ``ts``, so this has to match it.

    The cover question — is this a line about a goal — starts ten seconds
    before the graphic catches up, because a caller watching the ball cross
    the line is talking about the goal the board is about to show. The
    arithmetic question cannot start there: in that gap the board still reads
    2-0 while the ball is in the net for 2-1, and telling the gate the number
    was settled struck out two correct scorelines on the Mbappé trace.
    """
    from commentary.rephrase import Cover

    states = [
        (0.0, MatchState(home="Argentina", away="France", home_score=2, away_score=0)),
        (
            86.3,
            MatchState(
                home="Argentina",
                away="France",
                home_score=2,
                away_score=1,
                incidents=[
                    Incident(
                        event=Event.GOAL,
                        side=Side.AWAY,
                        player=None,
                        video_ts=82.5,
                        source="board",
                    )
                ],
            ),
        ),
    ]
    cover = Cover(states)
    assert not cover.goal_in_state(82.5)
    assert cover.goal_in_state(86.8)
    assert cover.goal_in_state(200.0)


def test_a_second_goal_arriving_unsettles_a_number_the_first_one_settled() -> None:
    """The line at 176.7 is about the goal the state takes in at 180.9.

    Counting any earlier goal as proof that the number is settled rejected
    "Mbappé! Off the ground! Two-two." against a row still reading 2-1 — the
    right score, and the second one this rule threw away.
    """
    from commentary.rephrase import Cover

    def scored(home: int, away: int, goals: int) -> MatchState:
        return MatchState(
            home="Argentina",
            away="France",
            home_score=home,
            away_score=away,
            incidents=[
                Incident(
                    event=Event.GOAL, side=Side.AWAY, player=None, video_ts=0.0, source="board"
                )
            ]
            * goals,
        )

    cover = Cover([(0.0, scored(2, 0, 0)), (86.3, scored(2, 1, 1)), (180.9, scored(2, 2, 2))])
    assert not cover.goal_in_state(82.5)
    assert cover.goal_in_state(86.8)
    assert cover.goal_in_state(120.0)
    assert not cover.goal_in_state(176.7)
    assert cover.goal_in_state(180.9)


@pytest.mark.asyncio
async def test_a_goal_may_be_one_ahead_of_a_board_that_has_not_moved() -> None:
    """"Two-one." over a state row that still says 2-0 is the third beat working."""
    rows = a_trace()
    rows[2] = rows[2] | {"event": "goal", "line": "Molina turns it in at the near post."}
    rows[3] = rows[3] | {"event": "goal", "line": "Molina turns it in at the near post."}
    rows[4] = rows[4] | {"event": "goal"}
    backend = saying(
        PhrasedLine(line="Molina! At the near post! Two-nil.", excitement=1.0),
        PhrasedLine(line="Messi strikes.", excitement=0.75),
    )
    result = await rephrase(rows, backend, pack=a_pack())

    assert result.lines[0].passed, result.lines[0].reason
    beats = {row["id"]: row["text"] for row in rows_of(result.rows, "beat")}
    assert set(beats) == {"b1", "synth-14.0", "synth-18.0", "b2", "b3"}

    # The number on the goal line is the state's, put there by code, once.
    # The model wrote one too — the same one — and it was taken out first.
    phrased = rows_of(result.rows, "phrased")[0]
    assert phrased["score_appended"] == "Two-nil."
    assert phrased["score_stripped"] == ["Two-nil"]
    assert beats["b1"].endswith("Two-nil.")
    assert beats["b1"].count("Two-nil") == 1

    # The caller says nothing between 10 s and 20 s, which is longer than any
    # gap the corpus leaves inside the half-minute after a goal, so the
    # follow-up beats are called for and filled.
    extra = [line for line in result.lines if line.synthetic]
    assert [line.ts for line in extra] == [14.0, 18.0]
    assert all(not line.appended for line in extra), "the score goes out once"


def a_replay_row(ts: float, line: str, *, event: str = "foul") -> dict[str, Any]:
    """One replay form as every trace on disk has them: filled in, not spoken.

    ``speak`` is false because the caller vetoed its own replays until replay
    mode removed that veto, so every recorded trace carries the flag of a rule
    that no longer exists. The rephrase reads the form, not the flag.
    """
    return {
        "topic": "caller",
        "ts": ts,
        "scene": "replay",
        "event": event,
        "side": "home",
        "team": "Argentina",
        "sightings": [],
        "confidence": 0.8,
        "speak": False,
        "line": line,
    }


def a_trace_with_replays(*replays: dict[str, Any]) -> list[dict[str, Any]]:
    """The four-row trace with replay forms in it, in timestamp order.

    The state row is given a foul, because the gate's ``replay_of_nothing``
    rule is exactly that a replay is a second look at something that happened.
    """
    rows = a_trace()
    rows[0] = rows[0] | {"last_events": ["foul"]}
    return sorted([*rows, *replays], key=lambda row: float(row["ts"]))


@pytest.mark.asyncio
async def test_a_replay_form_the_run_never_said_becomes_a_lead_beat() -> None:
    """Replay mode, offline.

    The whole point of the offline path: every trace in ``runs/`` was recorded
    with the veto in place, so the replay lines are sitting in the caller rows
    with ``speak`` false and nothing downstream ever looked at them.
    """
    rows = a_trace_with_replays(
        a_replay_row(25.0, "The replay: the trailing leg, and down he goes."),
        a_replay_row(29.0, "Tight in: the foot was never near the ball."),
    )
    backend = saying(
        PhrasedLine(line="Molina, down the right.", excitement=0.3),
        PhrasedLine(line="Messi strikes.", excitement=0.75),
        PhrasedLine(line="Having seen it again, the trailing leg caught him.", excitement=0.4),
        PhrasedLine(line="The foot was never near the ball there.", excitement=0.4),
    )
    result = await rephrase(rows, backend, pack=a_pack(), colour=False)

    beats = [row for row in rows_of(result.rows, "beat") if row.get("replay")]
    assert [row["id"] for row in beats] == ["replay-25.0", "replay-29.0"]
    assert [row["ts"] for row in beats] == [25.0, 29.0]
    # A lead beat, so the register measurement and the colour pass see it as
    # the lead line it is — and preemptable, whatever it is a replay of.
    assert all(row["voice"] == "caller" for row in beats)
    assert all(row["preemptable"] is True for row in beats)
    assert beats[0]["text"] == "Having seen it again, the trailing leg caught him."

    phrased = [row for row in rows_of(result.rows, "phrased") if row.get("replay")]
    assert [row["ts"] for row in phrased] == [25.0, 29.0]
    assert [line.verdict for line in result.lines if line.replay] == ["passed replay"] * 2


@pytest.mark.asyncio
async def test_a_replay_sequence_gets_three_lines_offline_and_no_more() -> None:
    rows = a_trace_with_replays(
        *(a_replay_row(ts, f"Another angle, {ts:.0f}.") for ts in (25.0, 29.0, 33.0, 37.0, 41.0))
    )
    backend = saying(
        PhrasedLine(line="Molina, down the right.", excitement=0.3),
        PhrasedLine(line="Messi strikes.", excitement=0.75),
        PhrasedLine(line="The trailing leg caught him and down he went.", excitement=0.4),
    )
    result = await rephrase(rows, backend, pack=a_pack(), colour=False)

    assert len([row for row in rows_of(result.rows, "beat") if row.get("replay")]) == 3


@pytest.mark.asyncio
async def test_a_replay_too_close_to_a_line_the_lead_already_has_gives_way() -> None:
    """Two voices on one moment is worse than one, and the trace already has a beat."""
    rows = a_trace_with_replays(a_replay_row(21.0, "The replay: the trailing leg."))
    backend = saying(
        PhrasedLine(line="Molina, down the right.", excitement=0.3),
        PhrasedLine(line="Messi strikes.", excitement=0.75),
        PhrasedLine(line="The trailing leg caught him.", excitement=0.4),
    )
    result = await rephrase(rows, backend, pack=a_pack(), colour=False)

    assert not [row for row in rows_of(result.rows, "beat") if row.get("replay")]
    assert not [line for line in result.lines if line.replay]


@pytest.mark.asyncio
async def test_a_trace_with_no_replay_forms_rephrases_exactly_as_it_used_to() -> None:
    """Every trace recorded before replay mode has to come out unchanged."""
    backend = saying(
        PhrasedLine(line="Molina, down the right.", excitement=0.3),
        PhrasedLine(line="Messi strikes.", excitement=0.75),
    )
    result = await rephrase(a_trace(), backend, pack=a_pack(), colour=False)

    assert not [row for row in result.rows if row.get("replay")]
    assert not [line for line in result.lines if line.replay]
    beats = {row["id"]: row["text"] for row in rows_of(result.rows, "beat")}
    assert beats["b1"] == "Molina, down the right."
    assert beats["b2"] == "Messi strikes."


@pytest.mark.asyncio
async def test_inside_a_goal_window_the_replay_is_the_rebuild() -> None:
    """One rebuild of one move, and the one with a picture behind it wins.

    Without the replay this trace synthesises two follow-up beats into the
    gap between 10 s and 20 s (see the test above). A replay form at 14 s is
    the caller about to fill that gap itself, so the synthesiser stands down
    and the past-tense rebuild goes out over the pictures it is about.
    """
    rows = a_trace()
    rows[2] = rows[2] | {"event": "goal", "line": "Molina turns it in at the near post."}
    rows[3] = rows[3] | {"event": "goal", "line": "Molina turns it in at the near post."}
    rows[4] = rows[4] | {"event": "goal"}
    rows = sorted(
        [
            *rows,
            a_replay_row(14.0, "The replay: the ball across, and Molina at the near post.",
                         event="build_up"),
        ],
        key=lambda row: float(row["ts"]),
    )
    backend = saying(
        PhrasedLine(line="Molina! At the near post!", excitement=1.0),
        PhrasedLine(line="The ball had come across and Molina got there first.", excitement=0.5),
        PhrasedLine(line="Messi strikes.", excitement=0.75),
    )
    result = await rephrase(rows, backend, pack=a_pack(), colour=False)

    beats = {row["id"] for row in rows_of(result.rows, "beat")}
    assert "replay-14.0" in beats
    # The two the same trace synthesises without a replay in it. The gap the
    # synthesiser was going to fill is the gap the replay fills.
    assert "synth-14.0" not in beats
    assert "synth-18.0" not in beats
    assert [line.ts for line in result.lines if line.replay] == [14.0]


@pytest.mark.asyncio
async def test_a_rephrase_that_fails_keeps_the_line_the_run_paid_for() -> None:
    result = await rephrase(a_trace(), ScriptedBackend(), pack=a_pack())

    beats = {row["id"]: row["text"] for row in rows_of(result.rows, "beat")}
    assert beats["b1"].startswith("Molina drives forward")
    assert len(rows_of(result.rows, "error")) == 2
    assert all(line.fallback for line in result.lines)


@pytest.mark.asyncio
async def test_a_chosen_silence_leaves_no_beat_and_no_gate_row_but_is_recorded() -> None:
    """The offline path has to count silence the same way the runtime does.

    Otherwise a rephrase pass looks like it dropped a line, and the judge
    cannot tell a moment the phraser passed over from one it failed on.
    """
    backend = saying(
        PhrasedLine(line="", excitement=0.0),
        PhrasedLine(line="Messi strikes.", excitement=0.75),
    )
    result = await rephrase(a_trace(), backend, pack=a_pack())

    phrased = rows_of(result.rows, "phrased")
    assert [row["line"] for row in phrased] == ["", "Messi strikes."]
    assert "silence" in phrased[0]["reason"]
    assert [row["id"] for row in rows_of(result.rows, "beat") if row["voice"] == "caller"] == ["b2"]
    assert not [row for row in rows_of(result.rows, "gate") if row.get("where") == "rephrase"]
    assert [line.verdict for line in result.lines] == ["chose silence", "passed"]


@pytest.mark.asyncio
async def test_the_table_shows_both_versions_of_every_line() -> None:
    backend = saying(
        PhrasedLine(line="Molina, down the right.", excitement=0.3),
        PhrasedLine(line="Messi strikes.", excitement=0.75),
    )
    result = await rephrase(a_trace(), backend, pack=a_pack())
    table = result.table()

    assert "Molina drives forward down the right" in table
    assert "Molina, down the right." in table
    assert "Messi strikes." in table


@pytest.mark.asyncio
async def test_a_rephrased_trace_is_a_trace(tmp_path: Path) -> None:
    """It has to load through the same reader the watch page replays from."""
    backend = saying(PhrasedLine(line="Molina, down the right.", excitement=0.3))
    result = await rephrase(a_trace(), backend, pack=a_pack())
    path = result.write(tmp_path / "phrased.jsonl")

    from commentary.replay import cues

    rows = read_trace(path)
    assert rows == result.rows
    assert any(cue.topic.value == "beat" for cue in cues(rows, delay_s=8.0))


# -- the form's event is binding ---------------------------------------------


def test_the_call_makes_the_forms_event_the_thing_the_line_must_be_about() -> None:
    """Every invented outcome on the first two traces reached past this word.

    A penalty being waited on was written as a penalty struck; a foul given
    was written as a booking. The event is in the body twice now, once in
    capitals beside the instruction and once in the line that hands over the
    description.
    """
    body = text_of(
        phraser_blocks(
            a_form(event=Event.PENALTY), "", [], home="Argentina", away="France"
        )
    )
    assert "EVENT: penalty" in body
    assert "only about a penalty" in body


def test_the_rules_forbid_promoting_the_event_and_implying_the_score() -> None:
    rules = phraser_system()
    for forbidden in ("in the book", "levels it", "the equaliser", "YOU COMPRESS"):
        assert forbidden in rules, forbidden


@pytest.mark.asyncio
async def test_the_table_shows_the_event_and_says_when_the_words_moved_it() -> None:
    backend = saying(
        PhrasedLine(line="Molina, down the right.", excitement=0.3),
        PhrasedLine(line="Messi scores.", excitement=1.0),
    )
    result = await rephrase(a_trace(), backend, pack=a_pack())

    events = [line.events for line in result.lines]
    assert events[0] == "carry"
    # The form said shot; the words say the ball went in, which is the
    # reading the director uses to decide whether it may be cut off.
    assert events[1] == "shot->goal"
    assert "shot->goal" in result.table()

    rows = [row for row in result.rows if row.get("topic") == "phrased"]
    assert [row["form_event"] for row in rows] == ["carry", "shot"]
    assert [row["event"] for row in rows] == ["carry", "goal"]


# -- notes in the prompt -----------------------------------------------------
#
# The pack has always held context and none of it has ever been said. These
# pin down the two halves of the fix: that the notes about the people on this
# form reach the body of the call, and that they are absent from the moments a
# commentator would never drop a statistic into.


def argfra_notes() -> list[Note]:
    """The Argentina v France notes, as ``clips/add_notes_argfra.py`` writes them.

    Restated here rather than read off disk because ``clips/`` is where the
    broadcast footage lives and is not in the repository; the names and the
    counts are the ones in that file.
    """
    return [
        Note(
            about="Kylian Mbappé",
            text="five goals in this tournament",
            kind="stat",
            source="FIFA World Cup 2022, six matches played before the final",
        ),
        Note(
            about="Kylian Mbappé",
            text="a goal in the 2018 World Cup final at nineteen",
            kind="storyline",
            source="FIFA World Cup 2018 final, France 4-2 Croatia",
        ),
        Note(
            about="Lionel Messi",
            text="five goals in this tournament",
            kind="stat",
            source="FIFA World Cup 2022, six matches played before the final",
        ),
        Note(
            about="Argentina",
            text="chasing a first World Cup since 1986",
            kind="storyline",
            source="FIFA World Cup winners, 1930 to 2018",
        ),
    ]


def argfra_pack() -> KnowledgePack:
    return a_pack().model_copy(update={"notes": argfra_notes()})


def context_of(body: str) -> str:
    """Everything between the `context:` header and the next block."""
    _, _, rest = body.partition("context:")
    head, _, _ = rest.partition("THE LAST LINES SPOKEN")
    return head


def test_a_penalty_is_quiet_enough_to_carry_the_pack_notes() -> None:
    """The kick has not been taken. That gap is exactly where a note belongs."""
    form = a_form(
        "Mbappé places the ball on the spot and steps back.",
        event=Event.PENALTY,
        sightings=[Sighting(number=10, name="Mbappé", side=Side.AWAY)],
    )
    pack = argfra_pack()
    body = text_of(
        phraser_blocks(
            form,
            "Argentina 2-0 France\n78:09",
            [],
            home="Argentina",
            away="France",
            notes=notes_for(pack, ["Mbappé", "Argentina", "France"]),
        )
    )
    context = context_of(body)
    assert "five goals in this tournament" in context
    assert "2018 World Cup final" in context
    assert "Kylian Mbappé" in context


def test_a_goal_gets_an_empty_context_block() -> None:
    """Nobody reads a statistic over a goal, and the block says so rather than vanishing."""
    form = a_form(
        "Mbappé turns away with his arms out as the net bulges.",
        event=Event.GOAL,
        sightings=[Sighting(number=10, name="Mbappé", side=Side.AWAY)],
    )
    pack = argfra_pack()
    body = text_of(
        phraser_blocks(
            form,
            "Argentina 2-0 France\n78:09",
            [],
            home="Argentina",
            away="France",
            notes=notes_for(pack, ["Mbappé", "Argentina", "France"]),
        )
    )
    context = context_of(body)
    assert "(none" in context
    assert "five goals in this tournament" not in context


def test_the_block_is_there_even_when_the_pack_has_nothing() -> None:
    """Always printed. A block that comes and goes reads as a licence to improvise."""
    body = text_of(
        phraser_blocks(
            a_form(),
            "Argentina 0-0 France",
            [],
            home="Argentina",
            away="France",
        )
    )
    assert "context:" in body
    assert "(none" in context_of(body)


def test_the_man_on_the_form_outranks_his_country() -> None:
    """Four places, and the specific notes take them."""
    pack = argfra_pack()
    picked = notes_for(pack, ["Mbappé", "Argentina", "France"], limit=2)
    assert [note.about for note in picked] == ["Kylian Mbappé", "Kylian Mbappé"]


def test_a_note_about_somebody_not_on_the_form_is_not_offered() -> None:
    pack = argfra_pack()
    picked = notes_for(pack, ["Molina"])
    assert picked == []


@pytest.mark.asyncio
async def test_the_phraser_passes_the_notes_it_is_given_into_the_call() -> None:
    """End to end through the agent, so the wiring is pinned and not just the builder."""
    backend = saying(PhrasedLine(line="Mbappé. Five in the tournament already.", excitement=0.4))
    phraser = a_phraser(backend)
    form = a_form(
        "Mbappé stands over it.",
        event=Event.PENALTY,
        sightings=[Sighting(number=10, name="Mbappé", side=Side.AWAY)],
    )
    await phraser.phrase(
        form, "Argentina 2-0 France", notes=notes_for(argfra_pack(), ["Mbappé"])
    )
    body = backend.calls_tagged("phraser")[0].text
    assert "five goals in this tournament" in context_of(body)


def test_the_last_lines_are_shown_with_the_kind_of_moment_each_was() -> None:
    """Silence and opener variety both need the kind, not just the words.

    A model shown "Upamecano works it forward." cannot tell whether that was
    a carry it should now pass over in silence — the corpus says a quarter of
    repeated build-up touches get nothing said — or a tackle that has ended.
    """
    phraser = a_phraser(saying(PhrasedLine(line="Messi.", excitement=0.2)))
    phraser.accept("Upamecano works it forward.", Event.CARRY)
    phraser.accept("Squeezed back towards halfway.")

    assert phraser.recent == [
        "Upamecano works it forward.   (carry)",
        "Squeezed back towards halfway.   (no kind)",
    ]


def test_a_second_build_up_line_in_a_row_is_told_to_consider_silence() -> None:
    """The rule is in the system prompt; four rounds showed that is not enough.

    One chosen silence in 35 calls with the paragraph alone, and the moment it
    names — a second consecutive build-up line about the same man, which real
    commentary passes over a quarter of the time — went by every time. The
    condition is cheap to compute, so it is computed and said again in the
    body, where the model is looking.
    """
    quiet = text_of(
        phraser_blocks(
            a_form(event=Event.CARRY),
            "",
            [],
            home="Argentina",
            away="France",
            last_event=Event.PASS,
        )
    )
    assert "THE LAST LINE WAS THIS SAME KIND OF MOMENT" in quiet

    loud = text_of(
        phraser_blocks(
            a_form(event=Event.SHOT),
            "",
            [],
            home="Argentina",
            away="France",
            last_event=Event.PASS,
        )
    )
    assert "THE LAST LINE WAS THIS SAME KIND OF MOMENT" not in loud
