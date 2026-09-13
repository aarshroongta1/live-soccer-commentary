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
    Sighting,
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
    verdict = gate.judge(call("It is Zaltimore driving at the defence and wide"), state, pack)
    assert "Zaltimore" not in verdict.line
    assert any(r.startswith("name_not_on_roster: Zaltimore") for r in verdict.reasons)


def test_the_first_word_of_a_line_is_not_checked_at_all(pack, state):
    """The deliberate trade, and it is a real one: an invented surname put
    first now reaches the microphone.

    Across nine runs on six clips the position-zero trim fired nineteen times
    — Tears, Hands, Arms, Fist, Ice, Thousands, Whole, Pure, Emotion, Sky,
    Restart, Round, Grimacing — and not one was a name the caller invented.
    It cost about two true lines a run to catch nothing. So the front of a
    line is no longer read as a name claim, which also means nothing there is
    verified. Everywhere else in the line the roster check is unchanged.
    """
    gate = FactGate()
    verdict = gate.judge(call("Zaltimore drives at the defence and slides it wide"), state, pack)
    assert verdict.passed
    assert "Zaltimore" in verdict.line
    assert verdict.reasons == []


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
        call(
            "The substitute is warming up on the touchline",
            sightings=[Sighting(name="Zaltimore")],
        ),
        state,
        pack,
    )
    assert not verdict.passed
    assert verdict.reasons == ["name_read_not_on_roster: Zaltimore"]


def test_a_shirt_number_that_is_not_in_the_squad_is_rejected(pack, state):
    gate = FactGate()
    verdict = gate.judge(
        call("The number comes short", sightings=[Sighting(number=77)]), state, pack
    )
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


def test_a_goal_needs_the_board(pack, state):
    gate = FactGate()
    goal = call("Krastanov smashes it home from eight yards", event=Event.GOAL)

    unconfirmed = gate.judge(goal, state, pack)
    assert not unconfirmed.passed
    assert unconfirmed.reasons == ["unconfirmed_goal: no board change, no wire"]

    assert gate.judge(goal, state, pack, board_changed=True).passed
    # The third route exists only when an ablation is running a statistician.
    assert gate.judge(goal, state, pack, wire_confirmed=True).passed


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
    verdict = gate.judge(call("It is Zaltimore"), state, pack)
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
    verdict = gate.judge(call("It is Krastanov driving at the back four"), state, None)
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
        call("It is Zaltimore"),
        MatchState(home="Northvale United", away="Carrowmere City"),
        None,
    )
    stats.record(verdict)
    assert stats.by_reason["too_short_after_trim"] == 1


# -- sightings, the one record of what was read ------------------------------


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


def sighting_verdict(
    gate: FactGate,
    state: MatchState,
    pack: KnowledgePack,
    *,
    number: int | None = None,
    name: str | None = None,
    mark: str | None = None,
):
    line = CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.BUILD_UP,
        sightings=[Sighting(mark=mark, number=number, name=name)],
        confidence=0.8,
        speak=True,
        line="He drives forward down the left.",
    )
    return gate.judge(line, state, pack)


@pytest.mark.parametrize(
    ("number", "name"),
    [(11, "Di María"), (11, None), (None, "Di María"), (11, "Ángel Di María")],
)
def test_a_sighting_of_a_real_player_passes(number, name):
    """Either half on its own, or both, as long as they are that player."""
    state, pack = argentina()
    verdict = sighting_verdict(FactGate(), state, pack, number=number, name=name)
    assert verdict.passed, verdict.reasons


def test_a_sighting_whose_name_is_not_on_the_roster_still_fails():
    state, pack = argentina()
    verdict = sighting_verdict(FactGate(), state, pack, number=11, name="Zaltimore")
    assert not verdict.passed
    assert any("name_read_not_on_roster: Zaltimore" in r for r in verdict.reasons)


def test_a_sighting_whose_number_is_not_in_the_squad_still_fails():
    state, pack = argentina()
    verdict = sighting_verdict(FactGate(), state, pack, number=77, name="Di María")
    assert not verdict.passed
    assert any("number_not_in_squad: 77" in r for r in verdict.reasons)


def test_a_sighting_whose_halves_disagree_is_rejected():
    """Two readings of one shirt that cannot both be right is neither."""
    state, pack = argentina()
    verdict = sighting_verdict(FactGate(), state, pack, number=7, name="Di María")
    assert not verdict.passed
    assert any("sighting_disagrees" in r for r in verdict.reasons)


def test_the_tag_is_never_held_to_the_roster():
    """It is a letter this system drew over a body, not a claim about anyone."""
    state, pack = argentina()
    verdict = sighting_verdict(FactGate(), state, pack, mark="GP", number=11, name="Di María")
    assert verdict.passed, verdict.reasons


# -- compound surnames -------------------------------------------------------


@pytest.mark.parametrize("name", ["Di María", "De Paul", "Mac Allister", "María"])
def test_a_compound_surname_is_on_the_roster(name: str):
    state, pack = argentina()
    verdict = sighting_verdict(FactGate(), state, pack, name=name)
    assert verdict.passed, verdict.reasons


def test_half_a_compound_surname_is_not_a_name():
    state, pack = argentina()
    verdict = sighting_verdict(FactGate(), state, pack, name="Di")
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


def test_a_demonym_is_a_team_word_and_survives_the_trim():
    """'The French lines' was trimmed to 'the lines' on the first real run.

    "French" is capitalised, is on no roster, and is exactly the word a
    commentator reaches for when they cannot name anybody — so it is the last
    word the gate should be taking out.
    """
    sheet = TeamSheet(name="France", short="FRA", demonym="French")
    pack = KnowledgePack(home=TeamSheet(name="Argentina", demonym="Argentine"), away=sheet)
    state = MatchState(home="Argentina", away="France")
    line = CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.BUILD_UP,
        confidence=0.7,
        speak=True,
        line="Argentine pressure, and the French lines drop deeper",
    )
    verdict = FactGate().judge(line, state, pack)
    assert verdict.passed
    assert verdict.line == "Argentine pressure, and the French lines drop deeper"


@pytest.mark.parametrize(
    "text",
    [
        "Blue shirts crowd it out on the edge of the box",
        "Red shirts swarm the ball and boot it away",
        "White sleeves everywhere as the cross comes in",
    ],
)
def test_a_kit_colour_is_not_a_name_to_be_trimmed(text: str):
    """The caller is told to reach for the kit when it cannot read a number.

    A line opening "Blue shirts crowd it out" would otherwise lose the one
    word identifying which team did it, because it is capitalised and on no
    team sheet.
    """
    state, pack = argentina()
    line = CallerLine(
        scene=Scene.LIVE_PLAY, event=Event.BUILD_UP, confidence=0.7, speak=True, line=text
    )
    verdict = FactGate().judge(line, state, pack)
    assert verdict.passed, verdict.reasons
    assert verdict.line == text


# -- sentence openers --------------------------------------------------------


def test_an_ordinary_word_opening_the_line_is_not_a_name():
    """The goal line of the second real run, which the gate beheaded.

    "Round the keeper and rolled in at the far post — Di María finishes off a
    breakaway of real beauty!" went out as "The keeper and rolled in at the far
    post — ...", reason ``name_not_on_roster: Round``: "around" was a stopword
    and "round" was not.
    """
    state, pack = argentina()
    text = (
        "Round the keeper and rolled in at the far post "
        "— Di María finishes off a breakaway of real beauty!"
    )
    line = CallerLine(
        scene=Scene.LIVE_PLAY, event=Event.GOAL, confidence=0.9, speak=True, line=text
    )
    verdict = FactGate().judge(line, state, pack, board_changed=True)
    assert verdict.passed, verdict.reasons
    assert verdict.line == text
    assert verdict.reasons == []


@pytest.mark.parametrize(
    "text",
    [
        "Wide of the far post, and the keeper had it covered anyway",
        "Off the bar and away, France breathe again",
        "Straight at the goalkeeper from eight yards",
        "Through the legs of the full-back and into the box",
    ],
)
def test_an_opener_is_only_an_opener_at_the_start_of_the_line(text: str):
    state, pack = argentina()
    line = CallerLine(
        scene=Scene.LIVE_PLAY, event=Event.BUILD_UP, confidence=0.7, speak=True, line=text
    )
    verdict = FactGate().judge(line, state, pack)
    assert verdict.passed, verdict.reasons
    assert verdict.line == text


def test_a_name_that_opens_the_line_is_no_longer_checked():
    """Was: the rule drops one ordinary word, not the front of every sentence.

    Now: the front of a sentence is not a name claim, so nothing there is
    checked. See test_the_first_word_of_a_line_is_not_checked_at_all for the
    count that decided it.
    """
    state, pack = argentina()
    line = CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.BUILD_UP,
        confidence=0.7,
        speak=True,
        line="Zaltimore turns inside and drives at the back four",
    )
    verdict = FactGate().judge(line, state, pack)
    assert verdict.passed
    assert "Zaltimore" in verdict.line
    assert verdict.reasons == []


def test_a_possessive_is_not_part_of_the_name() -> None:
    """"He strikes it low to De Gea's right" lost the goalkeeper, twice.

    `fold` dropped the apostrophe and glued the s on, so "De Gea's" became
    "de geas", matched nobody on either roster, and the gate trimmed a real
    name out of a penalty being scored past him.
    """
    from commentary.gate import fold

    assert fold("De Gea's right") == "de gea right"
    assert fold("Di María") == "di maria"


def test_the_keepers_name_survives_a_possessive(pack, state) -> None:
    gate = FactGate()
    keeper = pack.away.squad[0].name
    verdict = gate.judge(
        call(f"He strikes it low to {keeper}'s right and it squirms in"), state, pack=pack
    )
    assert verdict.passed
    assert keeper in verdict.line, verdict.reasons


def test_a_nationality_with_a_man_on_the_end_is_still_not_a_name(pack, state) -> None:
    """"the Dutchman steps up" lost its subject twice in one clip.

    The demonym was allowed as a team word and the noun a commentator
    actually reaches for was not, so "Dutchman" read as an invented surname.
    """
    gate = FactGate()
    dutch = pack.model_copy(update={"home": pack.home.model_copy(update={"demonym": "Dutch"})})
    verdict = gate.judge(call("The Dutchman steps up and strikes it low"), state, pack=dutch)
    assert verdict.passed
    assert "Dutchman" in verdict.line, verdict.reasons


def test_a_carried_name_counts_as_verified(pack, state) -> None:
    """The name the last line had on the ball is not an invention.

    Molina was named twice on his run and anonymous when he finished it: the
    number had turned away by the time the shot came, and the gate had no
    reason to believe a name the caller could no longer read.
    """
    gate = FactGate()
    # Mid-line, because the first word of a line is no longer checked at all.
    said = "And it is Vasquez sliding it across the face of goal"
    verdict = gate.judge(call(said), state, pack)
    assert "Vasquez" not in verdict.line

    carried = gate.judge(call(said), state, pack, carried="Ana Vasquez")
    assert carried.passed
    assert "Vasquez" in carried.line


def test_a_name_the_caller_read_and_did_not_say_is_logged(pack, state) -> None:
    """Logged, not rewritten: the gate does not put words in the line.

    On the Messi penalty the caller reported `10 Messi` in the same call that
    wrote "the keeper goes the wrong way", and nothing counted it.
    """
    from commentary.schemas import Sighting

    gate = FactGate()
    player = pack.home.squad[0]
    line = call(
        "The taker steps up and strikes it low",
        sightings=[Sighting(number=player.number, name=player.name)],
    )
    verdict = gate.judge(line, state, pack)
    assert verdict.passed
    assert verdict.line == "The taker steps up and strikes it low"
    assert any(r.startswith("name_withheld:") for r in verdict.reasons)


def test_a_name_the_caller_did_say_is_not_logged_as_withheld(pack, state) -> None:
    from commentary.schemas import Sighting

    gate = FactGate()
    player = pack.home.squad[0]
    surname = player.name.rsplit(" ", 1)[-1]
    line = call(
        f"{surname} steps up and strikes it low past the keeper",
        sightings=[Sighting(number=player.number, name=player.name)],
    )
    verdict = gate.judge(line, state, pack)
    assert not any(r.startswith("name_withheld:") for r in verdict.reasons)


def test_the_long_way_of_saying_it_is_in_is_a_goal_claim() -> None:
    """"Messi steps up ... and it is in" was scored as no goal at all.

    The confirmation rule's pattern is the contraction, `it's in`, and the
    caller writes it out. Unifying the two definitions inherited the gap.
    """
    from commentary.gate import claims_goal

    assert claims_goal("Messi steps up, strikes it low, and it is in")
    assert claims_goal("it's in!")
    assert not claims_goal("the ball is in the corner and he waits")
