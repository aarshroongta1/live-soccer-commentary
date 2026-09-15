"""The register instrument: the counted layer by hand, the judged one scripted.

The counted layer is tested against a five-line trace whose every number was
worked out on paper, because the whole point of the module is that its
numbers can be trusted without a model. The judged layer is tested through
:class:`ScriptedBackend` with a handler that builds its answer by parsing
JSON, so the path a real reply takes — text, ``json.loads``,
``model_validate`` — is the path the test takes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from commentary.grading import register
from commentary.grading.judge import JudgeError
from commentary.grading.register import (
    Names,
    RegisterVerdict,
    Source,
    is_bare_name,
    measure,
    name_words,
    people_of,
    says_a_number_off_score,
)
from commentary.llm.base import Block
from commentary.llm.fake import ScriptedBackend
from commentary.llm.schema import strict_schema
from commentary.prompts.commentary_examples import EXAMPLES, KINDS
from commentary.schemas import KnowledgePack, Note, Player, TeamSheet
from commentary.tallies import Tallies

# -- a trace whose every number was worked out on paper -----------------

STATE = {
    "topic": "state",
    "ts": 0.0,
    "home": "Argentina",
    "away": "France",
    "home_score": 0,
    "away_score": 0,
    "on_pitch": {"10": "Lionel Messi", "7": "Rodrigo De Paul"},
    "incidents": [],
}


def beat(ts: float, text: str, *, voice: str = "caller", event: str = "build_up") -> dict[str, Any]:
    return {"topic": "beat", "ts": ts, "voice": voice, "text": text, "event": event}


def phrased(ts: float, text: str) -> dict[str, Any]:
    return {"topic": "phrased", "ts": ts, "original": text, "line": text, "usd": 0.001}


#: Four lead lines: one word, two words, fourteen words, three words.
#: Words 1, 2, 14, 3 -> median 2.5. Two of four are nothing but a name.
#: Openers messi, de, messi, messi -> two repeats inside the window of five.
#: Gaps 2, 2, 8 -> median 2.0, one of three over four seconds, none over eight.
LINES = [
    (0.0, "Messi."),
    (2.0, "De Paul."),
    (4.0, "Messi drives at the defence and the crowd are up on their feet now."),
    (12.0, "Messi drives again."),
]


@pytest.fixture
def rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [STATE]
    for ts, text in LINES:
        out.append(phrased(ts, text))
        out.append(beat(ts, text))
    # The gate refused a fifth line on this pass. `where` is what separates a
    # rephrase's own verdicts from the original run's, which judged words
    # that are not in the file any more.
    out.append(
        {
            "topic": "gate",
            "ts": 6.0,
            "passed": False,
            "line": "Messi! Four-three.",
            "reasons": ["scoreline_mismatch: said 4-3, board 0-0"],
            "where": "rephrase",
        }
    )
    # The original run's verdict on a line that no longer exists. Counted by
    # nothing here, and a test below says so.
    out.append({"topic": "gate", "ts": 4.0, "passed": True, "line": "old words"})
    return out


def test_counts_the_shape_of_a_known_trace(rows: list[dict[str, Any]]) -> None:
    shape = measure(rows)
    assert shape.source is Source.PHRASED
    assert (shape.lines, shape.colour_lines) == (4, 0)
    assert shape.median_words == pytest.approx(2.5)
    assert shape.share_le_2 == pytest.approx(0.5)
    assert shape.share_le_4 == pytest.approx(0.75)
    assert shape.share_ge_9 == pytest.approx(0.25)
    assert shape.bare_name_share == pytest.approx(0.5)
    assert shape.name_share == pytest.approx(1.0)
    assert shape.opener_repeat_share == pytest.approx(0.5)
    assert shape.median_gap_s == pytest.approx(2.0)
    assert shape.share_gap_gt_4 == pytest.approx(1 / 3)
    assert shape.share_gap_gt_8 == pytest.approx(0.0)
    assert shape.number_share == pytest.approx(0.0)


def test_only_this_pass_s_refusals_are_counted(rows: list[dict[str, Any]]) -> None:
    """One refused line of five judged, and the old run's verdict ignored."""
    shape = measure(rows)
    assert [r.text for r in shape.refused] == ["Messi! Four-three."]
    assert shape.refusal_reasons == {"scoreline_mismatch": 1}
    assert shape.gate_refused_share == pytest.approx(1 / 5)


def test_a_second_voice_counts_apart_but_moves_the_cadence(
    rows: list[dict[str, Any]],
) -> None:
    """Colour is not scored on line length, and is heard in the gaps."""
    shape = measure([*rows, beat(3.0, "They have been chasing this all half.", voice="analyst")])
    assert (shape.lines, shape.colour_lines) == (4, 1)
    assert shape.median_words == pytest.approx(2.5)
    # Gaps are now 2, 1, 1, 8 over the five lines a listener would hear.
    assert shape.median_gap_s == pytest.approx(1.5)


def test_the_split_by_event_is_commonest_first_and_stable(rows: list[dict[str, Any]]) -> None:
    """A goal is called differently from build-up, and the median hides it."""
    extra = [
        phrased(20.0, "Messi!"),
        beat(20.0, "Messi!", event="goal"),
        phrased(24.0, "Messi buries it past the keeper from eight yards."),
        beat(24.0, "Messi buries it past the keeper from eight yards.", event="goal"),
    ]
    shape = measure([*rows, *extra])
    assert [(e.event, e.lines) for e in shape.by_event] == [("build_up", 4), ("goal", 2)]

    goal = shape.by_event[1]
    assert goal.median_words == pytest.approx(5.0)
    assert goal.share_le_4 == pytest.approx(0.5)
    assert goal.name_share == pytest.approx(1.0)
    # Words 1, 2, 14, 3, 1, 9: the whole-trace median of 2.5 is a number
    # neither group's lines are anywhere near.
    assert shape.median_words == pytest.approx(2.5)


def test_names_come_from_the_trace_when_there_is_no_pack(rows: list[dict[str, Any]]) -> None:
    assert people_of(rows) == frozenset({"Lionel Messi", "Rodrigo De Paul"})


def test_a_particle_is_part_of_a_bare_name_but_is_not_a_name() -> None:
    """"De Paul." is a bare name; a line whose only hit is "de" is not."""
    names = name_words(frozenset({"Rodrigo De Paul"}))
    assert is_bare_name("De Paul.", names)
    assert not register.says_a_name("Cut in from the left", names)
    assert register.says_a_name("Paul again", names)


def test_an_empty_name_set_says_nothing_rather_than_everything() -> None:
    assert not is_bare_name("Messi.", Names())


def test_the_scoreline_is_subtracted_before_a_number_is_counted() -> None:
    """The score is the number the system already says; the rest is the ask."""
    assert not says_a_number_off_score("Mbappé! Buried past Martínez! Two-one.")
    assert not says_a_number_off_score("Argentina lead 2-1.")
    assert says_a_number_off_score("Mbappé steps up. Five in the tournament.")
    assert says_a_number_off_score("Ten minutes remaining.")


def test_a_trace_with_no_phrased_rows_falls_back_to_the_caller() -> None:
    rows = [
        STATE,
        {"topic": "caller", "ts": 1.0, "line": "Messi carries it forward.", "speak": True},
        {"topic": "caller", "ts": 5.0, "line": "A thought that was not spoken.", "speak": False},
    ]
    shape = measure(rows)
    assert shape.source is Source.CALLER
    assert [line.text for line in shape.lead] == ["Messi carries it forward."]


def test_spoken_rows_beat_caller_rows_when_a_voice_ran() -> None:
    rows = [
        STATE,
        {"topic": "caller", "ts": 1.0, "line": "the form", "speak": True},
        {"topic": "spoken", "ts": 1.0, "voice": "caller", "spoken": "what came out"},
    ]
    shape = measure(rows)
    assert shape.source is Source.SPOKEN
    assert [line.text for line in shape.lead] == ["what came out"]


# -- the judged layer, scripted ----------------------------------------


def verdict_json(*, overall: float = 5.0, register_score: float = 11.0) -> str:
    """A reply in the shape the API would send it: text, to be parsed."""
    score = {"score": 4.0, "why": "even lengths, no fragments"}
    return json.dumps(
        {
            "voice_register": {"score": register_score, "why": "reads as description"},
            "economy": score,
            "variety": score,
            "event_fit": score,
            "goal_call": score,
            "build_up": score,
            "colour": score,
            "invention": {"score": 8.0, "why": "nothing unsupported reached air"},
            "overall": {"score": overall, "why": "accurate and lifeless"},
            "worst": [
                {"ts": 4.0, "line": "Messi drives at the defence", "why": "too long"},
                {"ts": 12.0, "line": "Messi drives again.", "why": "repeats the opener"},
                {"ts": 0.0, "line": "Messi.", "why": "fine, but three of these in a row"},
            ],
        }
    )


def scripted(reply: str | None = None) -> ScriptedBackend:
    """A backend whose handler parses JSON, the way the real one does."""
    text = reply if reply is not None else verdict_json()

    def handler(_blocks: list[Block], fmt: type) -> Any:
        return fmt.model_validate(json.loads(text))

    backend = ScriptedBackend()
    backend.register("judge_register", handler)
    return backend


async def test_the_judge_parses_json_into_the_verdict(rows: list[dict[str, Any]]) -> None:
    backend = scripted()
    verdict, usage = await register.judge_register(measure(rows), backend, name="t.jsonl")
    assert verdict.overall.score == pytest.approx(5.0)
    assert verdict.invention.why == "nothing unsupported reached air"
    assert len(verdict.worst) == 3
    assert usage.cost_usd == pytest.approx(0.0)


async def test_the_judge_is_shown_the_real_utterances_and_every_line(
    rows: list[dict[str, Any]],
) -> None:
    backend = scripted()
    await register.judge_register(measure(rows), backend, name="t.jsonl")
    call = backend.calls_tagged("judge_register")[0]
    # A real utterance, whichever one the generated example set happens to
    # start with. Naming one by hand ties this test to a rebuild of that
    # file, and the file is rebuilt whenever the corpus grows.
    assert EXAMPLES[KINDS[0]][0] in call.text
    assert "Messi drives again." in call.text
    # The refused line is shown, marked, so the judge can see what the
    # phraser tried to say without scoring it as if it had been said.
    assert "REFUSED by the fact gate" in call.text
    assert "Messi! Four-three." in call.text
    assert "commentator" in call.system


async def test_one_call_for_the_whole_passage(rows: list[dict[str, Any]]) -> None:
    backend = scripted()
    await register.judge_register(measure(rows), backend)
    assert len(backend.calls_tagged("judge_register")) == 1


async def test_a_trace_with_no_lines_is_refused_rather_than_judged() -> None:
    with pytest.raises(JudgeError):
        await register.judge_register(measure([STATE]), scripted())


def a_pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Argentina", short="ARG", starters=[Player(name="Lionel Messi", number=10)]
        ),
        away=TeamSheet(
            name="France", short="FRA", starters=[Player(name="Kylian Mbappé", number=10)]
        ),
        notes=[
            Note(
                about="Lionel Messi",
                text="five goals in this tournament",
                kind="stat",
                counts="goals",
            )
        ],
    )


async def test_pack_facts_reach_the_judge_after_the_reference_examples(
    rows: list[dict[str, Any]],
) -> None:
    """The judge sees what the broadcast was researched with, when it exists."""
    backend = scripted()
    await register.judge_register(measure(rows), backend, name="t.jsonl", pack=a_pack())
    call = backend.calls_tagged("judge_register")[0]
    assert "FACTS THE BROADCAST WAS GIVEN BEFORE KICKOFF" in call.text
    assert "Lionel Messi: five goals in this tournament (stat)" in call.text
    # After the reference examples, not before: the cached prefix those sit
    # behind is unaffected by whether a pack was given.
    assert call.text.index(EXAMPLES[KINDS[0]][0]) < call.text.index(
        "FACTS THE BROADCAST WAS GIVEN BEFORE KICKOFF"
    )
    assert call.blocks[0].get("cache_control") == {"type": "ephemeral"}


async def test_no_pack_no_facts_block(rows: list[dict[str, Any]]) -> None:
    backend = scripted()
    await register.judge_register(measure(rows), backend, name="t.jsonl")
    call = backend.calls_tagged("judge_register")[0]
    assert "FACTS THE BROADCAST WAS GIVEN BEFORE KICKOFF" not in call.text


async def test_a_shared_tallies_advances_the_note_shown_to_the_judge(
    rows: list[dict[str, Any]],
) -> None:
    """Cheap tallies, when the caller has them, read the way the match ended."""
    backend = scripted()
    tallies = Tallies()
    tallies.credit_goal("Lionel Messi", 4.0)
    await register.judge_register(
        measure(rows), backend, name="t.jsonl", pack=a_pack(), tallies=tallies
    )
    call = backend.calls_tagged("judge_register")[0]
    assert "Lionel Messi: six goals in this tournament (stat)" in call.text
    # No tallies given: the raw, researched note plus the caveat.
    backend2 = scripted()
    await register.judge_register(measure(rows), backend2, name="t.jsonl", pack=a_pack())
    call2 = backend2.calls_tagged("judge_register")[0]
    assert "Lionel Messi: five goals in this tournament (stat)" in call2.text
    assert "running count" in call2.text.lower()


async def test_a_score_out_of_range_is_clamped_not_raised(rows: list[dict[str, Any]]) -> None:
    """A judge answering 11 is answering 10. Not worth failing a paid call."""
    verdict, usage = await register.judge_register(measure(rows), scripted())
    report = register.RegisterReport(
        name="t.jsonl", shape=measure(rows), verdict=verdict, usage=usage, model="scripted"
    )
    assert report.as_dict()["judge"]["scores"]["register"]["score"] == pytest.approx(10.0)


def test_the_verdict_survives_the_strict_schema() -> None:
    """Structured outputs need every object closed and every field required."""
    schema = strict_schema(RegisterVerdict)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    for definition in schema["$defs"].values():
        assert definition["additionalProperties"] is False


def test_the_reference_sample_is_the_same_every_time() -> None:
    first = register.reference_utterances()
    assert first == register.reference_utterances()
    assert len(first) == register.REFERENCE_N
    # Spread across the kinds rather than forty bare surnames in a row.
    assert len({kind for kind, _ in first}) >= 6


# -- end to end, on disk ----------------------------------------------


async def test_score_trace_writes_a_report_beside_the_trace(
    tmp_path: Path, rows: list[dict[str, Any]]
) -> None:
    trace = tmp_path / "run.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = await register.score_trace(trace, backend=scripted(), model="scripted")
    out = register.report_path(trace)
    report.write(out)

    assert out == tmp_path / "run.register.json"
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["measured"]["median_words"]["trace"] == pytest.approx(2.5)
    assert written["measured"]["median_words"]["real"] == pytest.approx(8.0)
    assert written["reference"] == "club football, six matches"
    assert written["reference_is_provisional"] is False
    assert written["judge"]["scores"]["overall"]["score"] == pytest.approx(5.0)
    assert written["usd"] == pytest.approx(0.0)
    assert "trace" in report.table()


async def test_no_model_leaves_the_judged_half_out(
    tmp_path: Path, rows: list[dict[str, Any]]
) -> None:
    trace = tmp_path / "run.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = await register.score_trace(trace)
    assert report.verdict is None
    assert report.usage.cost_usd == pytest.approx(0.0)
    assert "--no-model" in report.table()
    assert report.as_dict()["judge"] is None


async def test_a_free_rerun_does_not_throw_away_a_paid_verdict(
    tmp_path: Path, rows: list[dict[str, Any]]
) -> None:
    """The cheap command must not be the expensive one to use."""
    trace = tmp_path / "run.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    paid = await register.score_trace(trace, backend=scripted(), model="scripted")
    paid.write(register.report_path(trace))

    free = await register.score_trace(trace)
    free.write(register.report_path(trace))

    written = json.loads(register.report_path(trace).read_text(encoding="utf-8"))
    assert written["judge"]["scores"]["overall"]["score"] == pytest.approx(5.0)
    assert written["judge"]["carried_forward"] is True
    assert "an earlier judgement" in free.table()


async def test_a_judgement_of_a_different_line_count_is_not_carried(
    tmp_path: Path, rows: list[dict[str, Any]]
) -> None:
    """A report whose numbers describe other lines is worse than none."""
    trace = tmp_path / "run.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    paid = await register.score_trace(trace, backend=scripted(), model="scripted")
    paid.write(register.report_path(trace))

    longer = [*rows, phrased(20.0, "Messi once more."), beat(20.0, "Messi once more.")]
    trace.write_text("\n".join(json.dumps(row) for row in longer), encoding="utf-8")
    assert (await register.score_trace(trace)).carried is None
