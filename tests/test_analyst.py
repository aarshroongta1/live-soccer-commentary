import numpy as np

from commentary.agents.analyst import ECHO_THRESHOLD, Analyst, restates_score
from commentary.capture.buffer import DelayBuffer, Frame
from commentary.config import ANALYST_MODEL, AnalystConfig
from commentary.llm.base import Block, LLMError
from commentary.llm.fake import ScriptedBackend
from commentary.prompts.analyst import analyst_system
from commentary.schemas import (
    AnalystLine,
    Angle,
    Event,
    KnowledgePack,
    MatchState,
    Player,
    Side,
    TeamSheet,
)
from commentary.tools import MatchTools

CONFIG = AnalystConfig()


def buffer_with(
    seconds: float = 34.0,
    fps: int = 15,
    delay_s: float = 8.0,
    history_s: float = 24.0,
) -> DelayBuffer:
    """Long enough that the analyst's twenty-second window fits behind the cursor."""
    buf = DelayBuffer(fps=fps, delay_s=delay_s, history_s=history_s)
    for i in range(int(fps * seconds)):
        image = np.full((90, 160, 3), (i * 3) % 256, dtype=np.uint8)
        buf.append(Frame(ts=i / fps, image=image))
    return buf


def pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Arsenal",
            short="ARS",
            kit="red",
            formation="4-3-3",
            manager="Arteta",
            starters=[
                Player(name="David Raya", number=1, position="GK"),
                Player(name="Bukayo Saka", number=7, position="RW"),
            ],
            bench=[Player(name="Gabriel Jesus", number=9, position="ST")],
        ),
        away=TeamSheet(
            name="Chelsea",
            short="CHE",
            kit="blue",
            formation="4-2-3-1",
            manager="Maresca",
            starters=[Player(name="Cole Palmer", number=10, position="AM")],
        ),
        competition="Premier League",
        venue="Emirates Stadium",
        storylines=["Arsenal have not beaten Chelsea at home in four attempts"],
        form={"ARS": "WWDLW", "CHE": "LDLWL"},
        key_matchups=["Saka against the Chelsea left-back"],
    )


def state(**overrides: object) -> MatchState:
    base: dict[str, object] = {"home": "Arsenal", "away": "Chelsea"}
    base.update(overrides)
    return MatchState(**base)  # type: ignore[arg-type]


def analyst_with(
    *lines: AnalystLine,
    match_state: MatchState | None = None,
    config: AnalystConfig = CONFIG,
) -> tuple[Analyst, ScriptedBackend]:
    backend = ScriptedBackend()
    if lines:
        backend.queue("analyst", list(lines))
    knowledge = pack()
    tools = MatchTools(state=match_state or state(), pack=knowledge)
    return Analyst(backend, config, tools, knowledge), backend


def aside(
    text: str = "Third time down that left side in ten minutes, and nobody is tracking back.",
    *,
    angle: Angle = Angle.TACTICS,
    confidence: float = 0.8,
    speak: bool = True,
) -> AnalystLine:
    return AnalystLine(
        angle=angle,
        cites=["Saka against the Chelsea left-back"],
        confidence=confidence,
        speak=speak,
        line=text,
    )


# -- the prompt ---------------------------------------------------------


def test_the_system_prompt_is_byte_identical_across_two_builds():
    first = analyst_system(pack(), CONFIG)
    second = analyst_system(pack(), CONFIG)

    assert first == second
    assert "Bukayo Saka" in first
    assert "WWDLW" in first


def test_the_system_prompt_forbids_doing_the_callers_job():
    text = analyst_system(pack(), CONFIG)

    assert "never narrate the frames" in text
    assert "it is an echo" in text
    assert "what the viewer can already see" in text
    assert str(CONFIG.max_words) in text


# -- choosing what to look up -------------------------------------------


def test_gather_reaches_for_the_notes_when_the_match_is_quiet():
    analyst, _ = analyst_with(match_state=state(last_events=[Event.BUILD_UP]))

    facts = analyst.gather(analyst.tools.state)  # type: ignore[union-attr]

    assert facts["storylines"] == ["Arsenal have not beaten Chelsea at home in four attempts"]
    assert facts["form"] == {"ARS": "WWDLW", "CHE": "LDLWL"}
    assert "player" not in facts
    assert "key_matchups" not in facts
    assert "possession" not in facts


def test_gather_looks_up_the_player_on_the_side_with_the_ball():
    analyst, _ = analyst_with(
        match_state=state(
            last_events=[Event.CORNER],
            possession=Side.AWAY,
            on_pitch={"7": "Bukayo Saka", "10": "Cole Palmer"},
        )
    )

    facts = analyst.gather(analyst.tools.state)  # type: ignore[union-attr]

    assert facts["player"]["name"] == "Cole Palmer"
    assert facts["player"]["side"] == "away"
    assert facts["possession"] == "away"
    assert facts["key_matchups"] == ["Saka against the Chelsea left-back"]


def test_gather_does_not_hand_over_every_tool_on_every_call():
    quiet, _ = analyst_with(match_state=state(last_events=[Event.BUILD_UP]))
    busy, _ = analyst_with(
        match_state=state(last_events=[Event.CORNER], on_pitch={"7": "Bukayo Saka"})
    )

    quiet_facts = set(quiet.gather(quiet.tools.state))  # type: ignore[union-attr]
    busy_facts = set(busy.gather(busy.tools.state))  # type: ignore[union-attr]

    everything = {"storylines", "form", "key_matchups", "player", "team_sheet"}
    assert not everything <= quiet_facts
    assert not everything <= busy_facts
    assert "key_matchups" not in quiet_facts
    assert "storylines" not in busy_facts
    assert "team_sheet" not in quiet_facts | busy_facts


def test_gather_after_a_goal_frames_it_with_the_season_not_the_preamble():
    analyst, _ = analyst_with(match_state=state(last_events=[Event.SHOT, Event.GOAL]))

    facts = analyst.gather(analyst.tools.state)  # type: ignore[union-attr]

    assert facts["form"] == {"ARS": "WWDLW", "CHE": "LDLWL"}
    assert "storylines" not in facts
    assert "key_matchups" not in facts


def test_gather_with_no_tools_looks_up_nothing():
    analyst = Analyst(ScriptedBackend(), CONFIG)
    assert analyst.gather(state()) == {}


# -- eligibility --------------------------------------------------------


def test_a_lull_shorter_than_the_configured_one_is_not_a_lull():
    analyst, _ = analyst_with()

    allowed, reason = analyst.should_speak(3.0, 0.0, 100.0, Event.BUILD_UP)

    assert allowed is False
    assert "lull" in reason


def test_it_will_not_speak_twice_inside_its_own_minimum_gap():
    analyst, _ = analyst_with()

    allowed, reason = analyst.should_speak(20.0, 90.0, 100.0, Event.BUILD_UP)

    assert allowed is False
    assert "own last line" in reason


def test_a_goal_makes_it_eligible_however_recently_it_spoke():
    analyst, _ = analyst_with()

    allowed, reason = analyst.should_speak(1.0, 99.0, 100.0, Event.GOAL)

    assert allowed is True
    assert "goal" in reason


def test_a_real_lull_is_a_yes_and_says_so():
    analyst, _ = analyst_with()

    allowed, reason = analyst.should_speak(9.0, 50.0, 100.0, Event.BUILD_UP)

    assert allowed is True
    assert "lull" in reason


# -- the call -----------------------------------------------------------


async def test_the_model_is_shown_a_wide_window_the_state_and_why_it_is_asking():
    analyst, backend = analyst_with(aside(), match_state=state(last_events=[Event.BUILD_UP]))
    analyst.heard("Saka drives at the full-back and wins the corner.")

    result = await analyst.call(
        buffer_with(), "Arsenal 0-0 Chelsea\n12:04 (first half)", "a lull — 9.0 s"
    )

    assert result is not None and result.speak
    calls = backend.calls_tagged("analyst")
    assert len(calls) == 1
    assert calls[0].images == CONFIG.frames
    assert calls[0].model == ANALYST_MODEL
    assert calls[0].output_format is AnalystLine
    assert analyst.effort == "low"
    assert "Arsenal 0-0 Chelsea" in calls[0].text
    assert "a lull — 9.0 s" in calls[0].text
    assert "Saka drives at the full-back and wins the corner." in calls[0].text
    assert "Arsenal have not beaten Chelsea at home in four attempts" in calls[0].text


async def test_the_system_prompt_is_the_same_object_on_every_call():
    analyst, backend = analyst_with(
        aside("The gap between those midfielders has been there all half."),
        aside("Chelsea have not moved their line up once since the restart."),
    )
    buf = buffer_with()

    await analyst.call(buf, "0-0", "a lull")
    await analyst.call(buf, "0-0", "a lull")

    first, second = backend.calls_tagged("analyst")
    assert first.system == second.system == analyst.system


async def test_a_long_aside_is_trimmed_to_the_word_cap():
    long_line = " ".join(["they sit deep and let the ball come to them"] * 5)
    assert len(long_line.split()) > CONFIG.max_words
    analyst, _ = analyst_with(aside(long_line))

    result = await analyst.call(buffer_with(), "0-0", "a lull")

    assert result is not None
    assert result.speak
    assert len(result.line.split()) == CONFIG.max_words


async def test_low_confidence_forces_silence():
    analyst, _ = analyst_with(aside(confidence=0.2))

    result = await analyst.call(buffer_with(), "0-0", "a lull")

    assert result is not None
    assert result.speak is False
    assert analyst.suppressed["low_confidence"] == 1
    assert "confidence" in analyst.last_reason


async def test_reading_the_scoreboard_back_out_is_not_a_line():
    analyst, _ = analyst_with(
        aside("It is one-nil, and Arsenal look happy enough to sit on that."),
        match_state=state(home_score=1, away_score=0),
    )

    result = await analyst.call(buffer_with(), "Arsenal 1-0 Chelsea", "a lull")

    assert result is not None
    assert result.speak is False
    assert analyst.suppressed["scoreline"] == 1


async def test_it_may_not_paraphrase_what_the_caller_just_said():
    analyst, _ = analyst_with(aside("Saka drives at that full-back again and wins another corner."))
    analyst.heard("Saka drives at the full-back and wins the corner.")

    result = await analyst.call(buffer_with(), "0-0", "a lull")

    assert result is not None
    assert result.speak is False
    assert analyst.suppressed["echo"] == 1
    assert analyst.last_similarity >= ECHO_THRESHOLD


async def test_a_point_the_caller_did_not_make_goes_through():
    analyst, _ = analyst_with(
        aside("Chelsea came into this on one win in six, and it shows in how deep they sit.")
    )
    analyst.heard("Saka drives at the full-back and wins the corner.")

    result = await analyst.call(buffer_with(), "0-0", "a lull")

    assert result is not None
    assert result.speak is True
    assert analyst.suppressed["echo"] == 0


async def test_it_does_not_repeat_itself_twenty_five_seconds_later():
    analyst, _ = analyst_with(
        aside("Chelsea have sat deep since the goal and nobody is breaking that line."),
        aside("Chelsea sitting deep since the goal, and nobody breaks that line."),
    )
    buf = buffer_with()

    first = await analyst.call(buf, "0-0", "a lull")
    second = await analyst.call(buf, "0-0", "a lull")

    assert first is not None and first.speak
    assert second is not None and second.speak is False
    assert analyst.suppressed["repetition"] == 1


async def test_its_own_spoken_line_comes_back_in_the_next_prompt():
    analyst, backend = analyst_with(
        aside("The gap between those midfielders has been there all half."),
        aside("Chelsea have not moved their line up once since the restart."),
    )
    buf = buffer_with()

    await analyst.call(buf, "0-0", "a lull")
    await analyst.call(buf, "0-0", "a lull")

    second = backend.calls_tagged("analyst")[1]
    assert "you: The gap between those midfielders has been there all half." in second.text


async def test_model_silence_is_left_alone_and_not_remembered():
    analyst, _ = analyst_with(aside("", speak=False))

    result = await analyst.call(buffer_with(), "0-0", "a lull")

    assert result is not None
    assert result.speak is False
    assert analyst.gate.recent == []
    assert analyst.last_reason == "the model chose silence"


async def test_an_empty_line_with_speak_set_is_suppressed():
    analyst, _ = analyst_with(aside("   "))

    result = await analyst.call(buffer_with(), "0-0", "a lull")

    assert result is not None
    assert result.speak is False
    assert analyst.suppressed["empty"] == 1


async def test_preamble_and_quotes_are_stripped():
    analyst, _ = analyst_with(aside('Analyst: "That midfield gap has been there all half."'))

    result = await analyst.call(buffer_with(), "0-0", "a lull")

    assert result is not None
    assert result.line == "That midfield gap has been there all half."


async def test_a_failed_model_call_is_a_missed_aside_not_a_crash():
    def explode(_blocks: list[Block], _fmt: type) -> AnalystLine:
        raise LLMError("timed out")

    backend = ScriptedBackend()
    backend.register("analyst", explode)
    analyst = Analyst(backend, CONFIG, MatchTools(state=state(), pack=pack()), pack())

    result = await analyst.call(buffer_with(), "0-0", "a lull")

    assert result is None
    assert analyst.suppressed["llm_error"] == 1
    assert len(backend.calls_tagged("analyst")) == 1


async def test_an_empty_buffer_makes_no_model_call():
    analyst, backend = analyst_with(aside())

    result = await analyst.call(DelayBuffer(fps=15, delay_s=8.0), "0-0", "a lull")

    assert result is None
    assert backend.calls_tagged("analyst") == []
    assert analyst.suppressed["no_frames"] == 1


# -- the scoreline rule on its own --------------------------------------


def test_restates_score_catches_figures_words_and_phrases():
    board = {"home_score": 2, "away_score": 1}

    assert restates_score("Arsenal 2-1 up with ten to go", board)
    assert restates_score("Two-one, and Chelsea need another", board)
    assert restates_score("That is the equaliser", board)
    assert not restates_score("They have gone 4-3-3 since the hour", board)
    assert not restates_score("Chelsea have not moved their line up all half", board)
