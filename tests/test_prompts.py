import numpy as np

from commentary.capture.buffer import Frame
from commentary.config import CallerConfig
from commentary.prompts.caller import CALLER_RULES, caller_blocks, caller_system
from commentary.schemas import KnowledgePack, Player, TeamSheet, Trigger


def make_pack(form_first: str = "Arsenal") -> KnowledgePack:
    """A pack built fresh every time, so equality tests are about the text."""
    home = TeamSheet(
        name="Arsenal",
        short="ARS",
        kit="red shirts, white sleeves",
        formation="4-3-3",
        manager="Mikel Arteta",
        starters=[
            Player(name="David Raya", number=1, position="GK"),
            Player(name="Bukayo Saka", number=7, position="RW"),
        ],
        bench=[Player(name="Kai Havertz", number=29, position="CF")],
    )
    away = TeamSheet(
        name="Chelsea",
        short="CHE",
        kit="blue shirts",
        starters=[Player(name="Robert Sanchez", number=1, position="GK")],
    )
    records = {"Arsenal": "WWDLW", "Chelsea": "LDWWD"}
    form = {form_first: records[form_first]}
    other = "Chelsea" if form_first == "Arsenal" else "Arsenal"
    form[other] = records[other]
    return KnowledgePack(
        home=home,
        away=away,
        competition="Premier League",
        venue="Emirates Stadium",
        storylines=["Arsenal have not lost at home since March."],
        form=form,
    )


def make_frames(count: int, *, start: float = 0.0, step: float = 1.0) -> list[Frame]:
    return [
        Frame(ts=start + i * step, image=np.full((90, 160, 3), 10 * i % 256, dtype=np.uint8))
        for i in range(count)
    ]


def texts(blocks: list[dict[str, object]]) -> list[str]:
    return [str(b["text"]) for b in blocks if b.get("type") == "text"]


def test_system_prompt_is_byte_stable_across_builds():
    assert caller_system(make_pack()) == caller_system(make_pack())


def test_system_prompt_does_not_depend_on_dict_order():
    assert caller_system(make_pack("Arsenal")) == caller_system(make_pack("Chelsea"))


def test_system_prompt_carries_the_roster_with_shirt_numbers():
    system = caller_system(make_pack())
    assert "7 Bukayo Saka (RW)" in system
    assert "1 David Raya (GK)" in system
    assert "29 Kai Havertz (CF)" in system
    assert "red shirts, white sleeves" in system


def test_system_prompt_states_the_word_cap_from_config():
    assert "At most 28 words." in caller_system(None)
    assert "At most 12 words." in caller_system(None, CallerConfig(max_words=12))


def test_system_prompt_carries_the_rules_that_matter():
    system = caller_system(None)
    lowered = system.lower()
    assert "replay" in lowered
    assert "never state it" in lowered
    assert "silence is a real answer" in lowered


def test_system_prompt_without_a_pack_has_no_team_sheets():
    assert "TEAM SHEETS" not in caller_system(None)


def test_blocks_carry_every_frame_as_an_image():
    blocks = caller_blocks(make_frames(4), make_frames(2, start=10.0), "0-0, 12:04", [], [])
    assert sum(1 for b in blocks if b.get("type") == "image") == 6


def test_lookahead_images_are_marked_as_the_future():
    cursor = make_frames(4)
    ahead = make_frames(2, start=11.0, step=2.0)
    blocks = caller_blocks(cursor, ahead, "0-0", [], [])
    joined = "\n".join(texts(blocks))
    assert "SECONDS AFTER" in joined
    assert "Outcome check only." in joined
    assert "Never describe them as if they were happening" in joined

    kinds = [str(b.get("type")) for b in blocks]
    labels = [str(b.get("text", "")) for b in blocks]
    future_at = next(i for i, t in enumerate(labels) if "THE NEAR FUTURE" in t)
    last_cursor_at = max(i for i, t in enumerate(labels) if t.startswith("Frame "))
    assert last_cursor_at < future_at
    # Every image after the future marker is a lookahead image, and there are two.
    assert kinds[future_at:].count("image") == 2


def test_frames_are_labelled_with_their_offset_from_now():
    blocks = caller_blocks(make_frames(4), make_frames(2, start=12.0), "0-0", [], [])
    labels = texts(blocks)
    assert any("3.0 s before now" in t for t in labels)
    assert any("this is now" in t for t in labels)
    assert any("9.0 s after" in t for t in labels)


def test_missing_lookahead_is_said_out_loud():
    blocks = caller_blocks(make_frames(4), [], "0-0", [], [])
    joined = "\n".join(texts(blocks))
    assert sum(1 for b in blocks if b.get("type") == "image") == 4
    assert "no lookahead frames are available" in joined


def test_volatile_content_is_the_last_block():
    blocks = caller_blocks(
        make_frames(4),
        make_frames(2, start=10.0),
        "Arsenal 1-0 Chelsea, 63:20",
        ["Saka drives at the full-back."],
        [Trigger.ROAR, Trigger.WHISTLE],
    )
    tail = blocks[-1]
    assert tail["type"] == "text"
    body = str(tail["text"])
    assert "Arsenal 1-0 Chelsea, 63:20" in body
    assert "Saka drives at the full-back." in body
    assert "roar, whistle" in body


def test_tail_says_when_nothing_has_been_said_yet():
    blocks = caller_blocks(make_frames(4), make_frames(2, start=10.0), "", [], [])
    body = str(blocks[-1]["text"])
    assert "(nothing said yet)" in body
    assert "routine tick" in body
    assert "Not established yet." in body


def test_the_pack_prints_one_player_a_line_with_the_number_first():
    pack = KnowledgePack(
        home=TeamSheet(
            name="Argentina",
            kit="sky blue and white stripes",
            starters=[Player(name="Ángel Di María", number=11, position="LW")],
        ),
        away=TeamSheet(name="France", kit="blue shirts"),
    )
    system = caller_system(pack)
    # A number the caller has just read is looked up down a column, not inside
    # a paragraph of eighteen of them.
    assert "    #11 Ángel Di María (LW)" in system
    # The kit is what tells the caller which of the two sheets a number is on,
    # so both sides carry one.
    assert "kit sky blue and white stripes" in system
    assert "kit blue shirts" in system


def test_the_names_rule_sends_the_caller_looking_for_a_number_first():
    assert "Look before you give up on one" in CALLER_RULES
    assert "A wrong name is the\nworst thing you can do here" in CALLER_RULES
    # The bad example that showed a model naming two players it could not read
    # is gone; what replaces it shows the number that justified the name.
    assert "Odegaard" not in CALLER_RULES
    assert 'sightings carries mark "D",\n        number 11, side home' in CALLER_RULES


def test_a_number_is_useless_without_the_kit_it_was_read_off():
    """Both squads wear a 5, a 7 and an 11, and the caller is the one looking.

    On the penalty clip 18 of 34 sightings were dropped, nearly every one a
    bare number two players in the match wear. The side used to be resolved
    from the kit split, which put an Argentina body on France and named a
    France forward for it.
    """
    assert "Every sighting with a number needs a side" in CALLER_RULES
    assert "leave side unknown and expect the number to be discarded" in CALLER_RULES


def test_the_referees_arm_is_the_penalty_and_the_graphic_is_not_needed():
    """The clip: six `free_kick` lines, then "the graphic tells us the story"."""
    assert "A referee pointing at the penalty spot is a penalty" in CALLER_RULES
    assert "do not wait for a graphic" in CALLER_RULES


def test_a_name_on_a_graphic_is_a_sighting():
    assert "A name on a broadcast graphic is a sighting too" in CALLER_RULES


def test_the_caller_is_told_a_goal_is_a_goal_whatever_put_it_there():
    """Ronaldo's free kick was tagged `free_kick` and died on a camera cut."""
    assert "If your line says the ball went in, the event is goal" in CALLER_RULES


def test_the_caller_is_told_to_say_the_name_it_just_read():
    """The Messi penalty: `10 Messi` reported in the same call as "the taker"."""
    assert "Use the name you have." in CALLER_RULES
    assert "having one and saying \"the striker\" instead" in CALLER_RULES


def test_the_caller_is_told_what_a_name_may_be_carried_through():
    assert "A name stays yours for a few seconds." in CALLER_RULES
    assert "never the taker of a set piece you have not actually read" in CALLER_RULES
