"""The gate is the project's central claim, so these are its proof."""

import pytest

from commentary.gate import FactGate, GateStats
from commentary.schemas import (
    CallerLine,
    Event,
    KnowledgePack,
    MatchState,
    Player,
    Scene,
    Side,
    TeamSheet,
)


@pytest.fixture
def pack() -> KnowledgePack:
    """Invented squads. Peñaló is here to prove the accent folding works."""
    return KnowledgePack(
        home=TeamSheet(
            name="Northvale United",
            short="Northvale",
            manager="Lars Hovden",
            starters=[
                Player(name="Tomás Peñaló", number=9),
                Player(name="Errol Kimbanda", number=7),
                Player(name="Wes Dunthorpe", number=4),
            ],
            bench=[Player(name="Petr Salajka", number=18)],
        ),
        away=TeamSheet(
            name="Carrowmere City",
            short="Carrowmere",
            manager="Bea Ollerenshaw",
            starters=[
                Player(name="Ivo Krastanov", number=11),
                Player(name="Marcel Ferreiro", number=3),
                Player(name="Ade Olowokere", number=22),
            ],
        ),
        competition="Coastal Cup",
        venue="Northvale Park",
    )


@pytest.fixture
def state() -> MatchState:
    return MatchState(
        home="Northvale United", away="Carrowmere City", home_score=2, away_score=1, clock="61:20"
    )


def call(text: str, **kwargs: object) -> CallerLine:
    fields: dict[str, object] = {
        "scene": Scene.LIVE_PLAY,
        "event": Event.BUILD_UP,
        "side": Side.HOME,
        "confidence": 0.8,
        "speak": True,
        "line": text,
    }
    fields.update(kwargs)
    return CallerLine.model_validate(fields)


def test_an_invented_name_never_reaches_the_microphone(pack, state):
    gate = FactGate()
    verdict = gate.judge(call("Zaltimore drives at the defence and slides it wide"), state, pack)
    assert "Zaltimore" not in verdict.line
    assert any(r.startswith("name_not_on_roster: Zaltimore") for r in verdict.reasons)


def test_a_roster_name_passes_untouched(pack, state):
    gate = FactGate()
    verdict = gate.judge(call("Ivo Krastanov turns and drives at the back four"), state, pack)
    assert verdict.passed
    assert verdict.line == "Ivo Krastanov turns and drives at the back four"
    assert verdict.reasons == []


def test_a_surname_on_its_own_passes(pack, state):
    gate = FactGate()
    verdict = gate.judge(call("Kimbanda holds it up on the edge of the box"), state, pack)
    assert verdict.passed
    assert verdict.reasons == []


@pytest.mark.parametrize("spelling", ["Peñaló", "Penalo", "PEÑALÓ"])
def test_accents_and_case_fold_away(pack, state, spelling):
    gate = FactGate()
    verdict = gate.judge(call(f"{spelling} comes short for it"), state, pack)
    assert verdict.passed, verdict.reasons
    assert spelling in verdict.line


def test_a_name_read_off_a_graphic_but_on_no_roster_is_rejected(pack, state):
    gate = FactGate()
    verdict = gate.judge(
        call("The substitute is warming up on the touchline", names_read=["Zaltimore"]), state, pack
    )
    assert not verdict.passed
    assert verdict.reasons == ["name_read_not_on_roster: Zaltimore"]


def test_a_shirt_number_that_is_not_in_the_squad_is_rejected(pack, state):
    gate = FactGate()
    verdict = gate.judge(call("The number comes short", names_read=["77"]), state, pack)
    assert not verdict.passed
    assert verdict.reasons == ["number_not_in_squad: 77"]


def test_a_wrong_scoreline_is_rejected(pack, state):
    gate = FactGate()
    verdict = gate.judge(call("Northvale lead three-one with half an hour left"), state, pack)
    assert not verdict.passed
    assert verdict.reasons == ["scoreline_mismatch: said 3-1, board 2-1"]


def test_the_scoreline_on_the_board_passes_in_either_orientation(pack, state):
    gate = FactGate()
    assert gate.judge(call("Northvale lead 2-1 here"), state, pack).passed
    assert gate.judge(call("Carrowmere trail 1-2 with time running out"), state, pack).passed


def test_a_goal_needs_the_board_or_a_celebration(pack, state):
    gate = FactGate()
    goal = call("Krastanov smashes it home from eight yards", event=Event.GOAL)

    unconfirmed = gate.judge(goal, state, pack)
    assert not unconfirmed.passed
    assert unconfirmed.reasons == [
        "unconfirmed_goal: no board change and no celebration in the lookahead"
    ]

    assert gate.judge(goal, state, pack, board_changed=True).passed
    assert gate.judge(goal, state, pack, lookahead_celebration=True).passed


def test_a_goal_claimed_in_prose_is_caught_too(pack, state):
    gate = FactGate()
    verdict = gate.judge(call("Kimbanda finds the net at the near post"), state, pack)
    assert not verdict.passed
    assert verdict.reasons[0].startswith("unconfirmed_goal")


def test_the_word_goal_in_an_innocent_sense_is_not_a_claim(pack, state):
    gate = FactGate()
    assert gate.judge(call("Dunthorpe swings it towards the goal"), state, pack).passed
    assert gate.judge(call("It is a goal kick to Carrowmere"), state, pack).passed


def test_a_replay_never_passes(pack, state):
    gate = FactGate()
    verdict = gate.judge(
        call("Krastanov turns his man beautifully", scene=Scene.REPLAY),
        state,
        pack,
        board_changed=True,
    )
    assert not verdict.passed
    assert verdict.reasons == ["scene_replay: a replay is never called as live"]


def test_trimming_an_unverifiable_name_leaves_a_line_worth_saying(pack, state):
    gate = FactGate()
    verdict = gate.judge(
        call("The cross comes in from Zaltimore and the winger cuts inside"), state, pack
    )
    assert verdict.passed
    assert verdict.line == "The cross comes in and the winger cuts inside"
    assert "trimmed_name: Zaltimore" in verdict.reasons


def test_trimming_down_to_two_words_rejects_instead(pack, state):
    gate = FactGate()
    verdict = gate.judge(call("Zaltimore drives forward"), state, pack)
    assert not verdict.passed
    assert verdict.reasons[-1] == "too_short_after_trim: 2 words left"


def test_silence_from_the_caller_is_respected(pack, state):
    gate = FactGate()
    assert gate.judge(call("", speak=False), state, pack).reasons == [
        "not_speaking: caller chose silence"
    ]
    assert gate.judge(call("   "), state, pack).reasons == ["empty_line: nothing to say"]


def test_with_no_knowledge_pack_every_name_is_unverifiable(state):
    gate = FactGate()
    verdict = gate.judge(call("Krastanov drives at the back four"), state, None)
    assert verdict.passed
    assert "Krastanov" not in verdict.line
    assert any(r.startswith("name_not_on_roster") for r in verdict.reasons)


def test_stats_count_what_the_eval_reports(pack, state):
    gate = FactGate()
    gate.judge(call("Kimbanda holds it up on the edge of the box"), state, pack)
    gate.judge(call("The cross comes in from Zaltimore and the winger cuts inside"), state, pack)
    gate.judge(call("Northvale lead three-one with half an hour left"), state, pack)
    gate.judge(call("Krastanov makes it 4-0", event=Event.GOAL), state, pack)

    stats = gate.stats
    assert (stats.judged, stats.passed, stats.rejected, stats.trimmed) == (4, 2, 2, 1)
    assert stats.by_reason["scoreline_mismatch"] == 2
    assert stats.by_reason["unconfirmed_goal"] == 1
    assert "judged 4" in stats.table()


def test_stats_are_usable_standalone():
    stats = GateStats()
    gate = FactGate()
    verdict = gate.judge(
        call("Zaltimore drives forward"),
        MatchState(home="Northvale United", away="Carrowmere City"),
        None,
    )
    stats.record(verdict)
    assert stats.by_reason["too_short_after_trim"] == 1


# -- names_read, in the form the caller actually writes them -----------------


def argentina() -> tuple[MatchState, KnowledgePack]:
    """The sheet the first real run used, with the names that broke on it."""
    sheet = TeamSheet(
        name="Argentina",
        short="ARG",
        starters=[
            Player(name="Ángel Di María", number=11),
            Player(name="Rodrigo De Paul", number=7),
            Player(name="Alexis Mac Allister", number=20),
        ],
    )
    pack = KnowledgePack(home=sheet, away=TeamSheet(name="France", short="FRA"))
    return MatchState(home="Argentina", away="France"), pack


def sighting_verdict(gate: FactGate, state: MatchState, pack: KnowledgePack, read: str):
    line = CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.BUILD_UP,
        names_read=[read],
        confidence=0.8,
        speak=True,
        line="He drives forward down the left.",
    )
    return gate.judge(line, state, pack)


@pytest.mark.parametrize("read", ["11 Di María", "Di María (11)", "11", "Di María"])
def test_a_sighting_is_parsed_before_it_is_checked(read: str):
    """The caller writes the pairing that justifies the name, so read it that way.

    On the first real run every one of these was rejected
    ``name_read_not_on_roster``, because the whole string was matched against
    the roster instead of being read as a number and a name.
    """
    state, pack = argentina()
    verdict = sighting_verdict(FactGate(), state, pack, read)
    assert verdict.passed, verdict.reasons


def test_a_sighting_whose_name_is_not_on_the_roster_still_fails():
    state, pack = argentina()
    verdict = sighting_verdict(FactGate(), state, pack, "Zaltimore (11)")
    assert not verdict.passed
    assert any("name_read_not_on_roster: Zaltimore" in r for r in verdict.reasons)


def test_a_sighting_whose_number_is_not_in_the_squad_still_fails():
    state, pack = argentina()
    verdict = sighting_verdict(FactGate(), state, pack, "Di María (77)")
    assert not verdict.passed
    assert any("number_not_in_squad: 77" in r for r in verdict.reasons)


# -- compound surnames -------------------------------------------------------


@pytest.mark.parametrize("name", ["Di María", "De Paul", "Mac Allister", "María"])
def test_a_compound_surname_is_on_the_roster(name: str):
    state, pack = argentina()
    verdict = sighting_verdict(FactGate(), state, pack, name)
    assert verdict.passed, verdict.reasons


def test_half_a_compound_surname_is_not_a_name():
    state, pack = argentina()
    verdict = sighting_verdict(FactGate(), state, pack, "Di")
    assert not verdict.passed
    assert any("name_read_not_on_roster: Di" in r for r in verdict.reasons)


# -- what counts as claiming a goal ------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "France scrambling back towards their own goal",
        "Everyone piles back into his goal",
        "The ball is worked towards the French goal",
        "A goal kick to restart it",
    ],
)
def test_the_word_goal_in_passing_is_not_a_goal_claim(text: str):
    """The first real run lost 'towards their own goal' to the bare word.

    A list of innocent uses of "goal" can never be finished, so there is no
    longer a list: the form's event field answers the question outright.
    """
    state, pack = argentina()
    line = CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.BUILD_UP,
        confidence=0.7,
        speak=True,
        line=text,
    )
    verdict = FactGate().judge(line, state, pack, board_changed=False)
    assert verdict.passed, verdict.reasons


def test_the_form_saying_goal_is_still_a_goal_claim():
    state, pack = argentina()
    line = CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.GOAL,
        confidence=0.9,
        speak=True,
        line="He drives it low across the keeper",
    )
    assert not FactGate().judge(line, state, pack, board_changed=False).passed


@pytest.mark.parametrize(
    "text",
    ["He scores from twenty yards", "It is in the back of the net", "That finds the net"],
)
def test_a_line_that_says_it_outright_is_still_a_goal_claim(text: str):
    state, pack = argentina()
    line = CallerLine(
        scene=Scene.LIVE_PLAY, event=Event.SHOT, confidence=0.9, speak=True, line=text
    )
    assert not FactGate().judge(line, state, pack, board_changed=False).passed
