"""The play-by-play prompt: a stable system prefix and a volatile per-call body.

The split is not cosmetic. The system string is long — rules plus two full
team sheets — and it is sent on every call, roughly once every four seconds
for ninety minutes. It earns its keep only if it caches, and it only caches
if it is byte-identical each time, so nothing in here may depend on the
clock, on set iteration order, or on anything else that drifts between
calls. The per-call blocks carry everything that does move: the frames, the
match state, what has just been said, and why the system is asking now.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from commentary.capture.buffer import Frame
from commentary.config import CallerConfig
from commentary.llm.base import Block, encode_frame, image_block, text_block
from commentary.perception.players import Track, mark_of
from commentary.schemas import KnowledgePack, Player, TeamSheet, Trigger

#: Tracks known at a moment of video time. The runtime keeps the store; this
#: module only asks it what was on the pitch when this frame was captured.
TracksFor = Callable[[float], list[Track]]


def no_tracks(ts: float) -> list[Track]:
    """The substitution that turns the marks off, rather than a flag to check."""
    return []

#: The rules. ``{max_words}`` is the only substitution, and it comes from
#: config so a sweep of the word cap changes the prompt with it.
CALLER_RULES = """\
You are the play-by-play commentator on a live football broadcast.

You have the pictures and a team sheet written before kickoff. There may also
be a statistician, and if there is, they speak only through MATCH STATE and
never in your ear. Everything else you say has to be something you can see
happening.

HOW YOU ARE SHOWN THE MATCH

You are given a short burst of frames in order, ending at the moment you are
calling. That last frame is now. The viewer is watching that moment, not a
later one.

You are then given one or two further frames taken a few seconds LATER, from
nearer the live edge. Those are a look at how this move ended. They exist so
that you do not commit to "he is through on goal" half a second before the
keeper smothers it.

Use the later frames to decide what to say. Never describe them as though
they were happening now. If the ball is in the net in the later frames, the
move you are calling is a goal and you may call it as it unfolds. If the
keeper has it, the shot was saved, so do not sell a chance that never was.
Your tense and your subject belong to the earlier moment; the later frames
only tell you which way it goes.

WHAT YOU MAY SAY

Names. Look before you give up on one. On every call, look at the shirt
number of the player on the ball, of the player it goes to, of whoever
shoots, and of the keeper. A number you can actually read is a name you are
allowed to use: match the kit that player is wearing to a team sheet below,
find that number in it, and use that player's surname. A name across the back
of a shirt and a name in a broadcast graphic count the same way.

Players carry a small tag drawn above them in a dark box. A tag with a
surname is a name you may use for that player, and for nobody else on the
pitch — and when that player is involved in what you are describing, use it.
That is what the tags are for. "Tagliafico knocks it infield" is the same
line as "Argentina knock it infield" with the one thing a listener wants
added to it. A tag that is a letter — A, B, ... Z, AA, AB — is a body being
followed that nobody has identified yet. A letter is a label for that body
and nothing else: not a shirt number, not a squad number, not a name. A
player with no tag at all is unidentified, whatever you think you recognise.

Everything you read off the picture goes in sightings, one entry per player.
A shirt number you can read, a name across the shoulders, a name in a
graphic: that is a sighting. Add the letter of the tag over that player
whenever they are wearing one, and leave the mark null when they are not —
a close-up of a player with no tag is still worth reporting, it just cannot
be tied to a body afterwards.

Every sighting with a number needs a side. Both squads wear a 5, a 7, a 10
and an 11, so a number on its own names nobody at all — it is thrown away.
Look at the shirt, match the colours to the kit lines in the team sheets
below, and set side to home or away. If you genuinely cannot tell which kit
it is, leave side unknown and expect the number to be discarded; do not
guess, because a number put on the wrong squad names the wrong man.

Do it every time a number is legible, even for a player you have reported
before and even when the line you are writing does not mention them. The
player tagged D whose shirt reads 11 in the white and blue stripes is one
sighting: mark "D", number 11, side home. A tagged player whose name you can
read across the shoulders is the same sighting with the name filled in too.

A name on a broadcast graphic is a sighting too — a lower third naming the
taker before a penalty, a scorer's caption, a substitution board. Put the
name in, leave the number null, and give it the tag of the body in the
close-up it is over if there is one, so the name goes onto that body.

The tag is what makes a name last. Report a number against a tag and that
tag becomes the player's surname for everybody who looks at the picture
afterwards, including you, through all the shots where the number is turned
away. Leave the tag out and the name dies with your line. And never report
something you cannot actually see: a guess here follows that player around
for the rest of the passage.

MATCH STATE may carry a statistician's lines: "on the ball" with a name,
"from" with the name of whoever passed it, and a "just now" list of things
somebody did — a foul, a card, an offside, a save. Those names may be used as
given, for exactly the thing the statistician says that player did and for
nothing else. They are the only names you may use without a legible number, a
name on a shirt, a graphic or a tag. A foul in the picture with "foul by
Rabiot on Messi" in the state is called with both names; a foul with nothing
in the state is called by kit and role.

If you cannot read a number and there is no surname tag, say the role and the kit
instead: "the left-back in white", "the near-post runner in blue", "the keeper
in green". That is a complete answer and it costs nothing. A wrong name is the
worst thing you can do here. There is no credit for guessing and no penalty
for saying "Arsenal" when you cannot see who it is.

Penalties. A referee pointing at the penalty spot is a penalty. Call it as
one and set event to penalty; do not wait for a graphic to confirm it, and
do not call it a free kick in the meantime. The same goes for the rest of
what the referee's arm says: an arm straight up is an indirect free kick,
an arm pointing to the corner is a corner.

The clock. The clock in MATCH STATE is the match clock, counting up from
zero. A half is 45 minutes and a match is 90. If you talk about time at all,
work it out from that clock — at 35:52 there are nine minutes of the half
left, not half an hour — and if you cannot, do not mention time.

The score. Never state it and never imply it. Do not say "one-nil", "level",
"the equaliser", "ahead", "behind", "back in front", "his second". Someone
else is reading the scoreboard, and if you invent a scoreline you contradict
them. Call the goal, not the arithmetic.

Replays. Broadcasts cut to replays constantly: slow motion, a tight angle, a
missing score bug, a moment you have already called. A replay is not live.
Mark the scene as a replay, and then either stay quiet or name it as a replay
in as many words. Never call a replay as though the move were happening now.

WHEN TO SPEAK

Silence is a real answer and most of the time it is the right one. A voice
that talks over every touch is noise. Set speak to false unless something has
actually changed since the lines you were given below: a shot, a save, a
foul, a card, a chance made or wasted, a real shift of territory. Ordinary
midfield passing does not need a line. Expect to stay quiet more often than
you speak.

Do not repeat the recent lines and do not paraphrase them either. If the only
true thing to say is the thing you have just said, say nothing.

One exception, and it overrides the rest of this paragraph. When the reason
you are being asked includes silence_pressure, nobody has spoken for a long
time, and a broadcast is never silent through half a minute of live
football. Say something true about the shape of play — who is on the ball,
where the ball is, which way the game is being pushed, who has settled into
what — unless the picture is a replay, in which case stay quiet and let it
run. Ordinary passing is worth a line when the alternative is dead air.

THE FORM

Fill every field from the picture, not from the story you would like to tell.
Confidence is your honest read on whether these frames support the claim; low
confidence is not punished, but a confident guess is.

sightings is the record of what was legible, and it is how a name in your
line is justified. One entry per player: the tag letter if they have one,
the number if you can read it, the name if you can read that. Knowing who
usually plays there is not a sighting and does not belong in it.

THE LINE

At most {max_words} words. One sentence. Present tense. No preamble, no sign-off,
no quotation marks, no "we see", no "in this frame". Write what a commentator
says out loud, not what an observer writes down.

  Good: Saka drives at the full-back and wins the corner.
  Good: Cutback from the right, and it is hammered over the bar.
  Good: Long ball forward, and the centre-half in red heads it clear.
  Bad:  In this frame we can see a player in a red shirt. (describing a picture)
  Good: Di María cuts inside and drives it low. (the 11 was legible on the
        player tagged D in the striped kit, so sightings carries mark "D",
        number 11, side home)
  Bad:  That is the equaliser, two apiece. (the score is not yours to give)
  Bad:  The replay shows him clean through. (a replay called as live)\
"""


def caller_system(pack: KnowledgePack | None, config: CallerConfig | None = None) -> str:
    """The cacheable prefix: the rules, then the team sheets.

    Byte-stable by construction for a given pack — no timestamps, no dict
    iteration that is not sorted first — because prompt caching is what makes
    sending two full squads on every call affordable.
    """
    cfg = config or CallerConfig()
    parts = [CALLER_RULES.format(max_words=cfg.max_words)]
    if pack is not None:
        parts.append(_pack_section(pack))
    return "\n\n".join(parts)


#: How far a mark may be from a frame's own timestamp and still be about it.
#: The tracker runs in a loop of its own, as fast as the models allow — three
#: to ten passes a second with real weights — so the nearest tracked frame is
#: a third of a second away at worst. It was 0.2 when detection ran on every
#: other frame, which at three passes a second means no marks at all.
MARK_TOLERANCE_S = 0.5

#: How tall a tag has to be *as the caller receives it*, and the width it is
#: received at. Frames go out through ``encode_frame``, which downscales to
#: 768 px and JPEGs at quality 70, so a tag drawn at cv2's scale 0.4 arrives
#: about six pixels tall — legible to somebody looking for it, which is not
#: the same as legible to somebody reading a football match. Twenty-two
#: pixels is the height of ordinary body text on that image.
MARK_ENCODED_PX = 22
ENCODED_WIDTH = 768

#: Cap height of FONT_HERSHEY_SIMPLEX at scale 1.0, which is what turns a
#: height in pixels into a cv2 font scale.
_SIMPLEX_CAP_PX = 22.0

#: Padding inside the dark box behind the text.
MARK_PAD = 4

#: How many times a tag may be moved up to clear another before it is dropped.
MARK_NUDGES = 2


def mark_font(width: int) -> tuple[float, int]:
    """Font scale and stroke for a tag on a frame this wide.

    Derived rather than fixed, so the tag is the same size in the picture the
    model sees whether the capture is 720p or 1080p — the only thing that
    matters is how many pixels survive the downscale to ``ENCODED_WIDTH``.
    """
    scale = (MARK_ENCODED_PX / _SIMPLEX_CAP_PX) * max(1.0, width / ENCODED_WIDTH)
    return scale, max(2, int(round(scale * 1.5)))


def draw_marks(
    image: np.ndarray, tracks: Sequence[Track], pack: KnowledgePack | None
) -> np.ndarray:
    """A small tag above each body being followed, on a COPY.

    This is the whole point of the vision chain, and what it is for changed
    once it was measured. It was "the local models read the shirt and the
    language model reads the name off the picture"; the local models read no
    shirt in three minutes and the language model read nine. So the tag is a
    handle: ``#4`` says "this body, the one I am following", and the caller
    that can read its shirt reports the pair. From then on the tag is that
    player's surname, and the name stays on the body through the shots where
    the number is turned away.

    Never on a frame in the buffer. The board reader and the analyst get the
    picture as it was broadcast, and a mark drawn over the score bug would be
    a system writing its own evidence.
    """
    import cv2

    marked = image.copy()
    scale, stroke = mark_font(image.shape[1])
    placed: list[tuple[int, int, int, int]] = []
    for track in tracks:
        text = _mark_text(track, pack)
        if text is None:
            continue
        x0, y0, _x1, _y1 = track.box
        (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, stroke)
        box_w, box_h = width + 2 * MARK_PAD, height + 2 * MARK_PAD
        top = y0 - box_h - 2
        # Two tags on top of each other are one unreadable smudge, and a tag
        # that cannot be read is worse than no tag: the caller is asked to
        # copy it exactly. Walk up to clear another tag — but only twice. A
        # tag nudged further than that has left the player it names, and a
        # label floating over somebody else is the one thing worse than none.
        for _ in range(MARK_NUDGES):
            here = (x0, top, x0 + box_w, top + box_h)
            if not any(_overlaps(here, other) for other in placed):
                break
            top -= box_h + 2
        else:
            if any(
                _overlaps((x0, top, x0 + box_w, top + box_h), other) for other in placed
            ):
                continue
        if top < 0:
            # No room above the player, which on a close-up is most of them:
            # the body fills the frame and its box starts off the top of it.
            # Put the tag inside the box at its top instead — still on that
            # body, still readable, which a label clipped by the frame edge
            # is not.
            top = max(0, y0) + 2
        if _overlaps((x0, top, x0 + box_w, top + box_h), _corner(image.shape)):
            # Except in the top-left corner, where the broadcaster keeps the
            # score bug and the simulator keeps the timestamp strip its
            # oracle reads back out of these very frames.
            continue
        placed.append((x0, top, x0 + box_w, top + box_h))
        cv2.rectangle(marked, (x0, top), (x0 + box_w, top + box_h), (20, 20, 20), -1)
        cv2.putText(
            marked,
            text,
            (x0 + MARK_PAD, top + box_h - MARK_PAD),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            stroke,
            cv2.LINE_AA,
        )
    return marked


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _corner(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """The top-left box nothing may be drawn in.

    Wide enough for the simulator's timestamp strip, which is a fraction of
    the frame width, and tall enough for one of its blocks.
    """
    height, width = shape[0], shape[1]
    return (0, 0, int(width * 0.42), int(height * 0.07))


def _mark_text(track: Track, pack: KnowledgePack | None) -> str | None:
    """What to print above a body: who it is, or which body it is.

    An unnamed body gets a letter, which is not a claim about anybody — it is
    a label, so the caller can say "the number I read is on that one" and be
    understood. It is letters rather than the track id printed as "#4"
    because that is what the caller did with a digit: three of six sightings
    on the real clip came back with the mark equal to the number read.

    Every tracked body, including the ones the kit split will not put in a
    team. That rule cost 94% of the close-ups: of 784 bodies tall enough for
    a shirt number to be legible, 44 carried a tag, and the rest were called
    referees because a close-up crop looks nothing like the wide-shot crops
    the split was fitted on. The bodies whose numbers can be read were
    exactly the bodies with nothing to read them against. A letter over the
    referee is a letter over the referee; nobody reports a shirt number off
    him, and if they did the roster would throw it out.
    """
    if track.name is not None:
        return track.name.rsplit(" ", 1)[-1]
    return mark_of(track.id)


def _marked_block(frame: Frame, tracks_for: TracksFor, pack: KnowledgePack | None) -> Block:
    tracks = tracks_for(frame.ts)
    image = draw_marks(frame.image, tracks, pack) if tracks else frame.image
    return image_block(encode_frame(image))


def caller_blocks(
    cursor_frames: Sequence[Frame],
    lookahead_frames: Sequence[Frame],
    state_summary: str,
    recent_lines: Sequence[str],
    triggers: Sequence[Trigger],
    tracks_for: TracksFor = no_tracks,
    pack: KnowledgePack | None = None,
) -> list[Block]:
    """One call's content: the moment, the near future, then the volatile tail.

    Every image gets a text label in front of it, because six unlabelled
    pictures of the same pitch are indistinguishable to the model and the
    whole lookahead idea rests on it knowing which two are the future. The
    state, the recent lines and the triggers go last, after the images, so
    that the part of the message that changes fastest sits furthest from the
    cache breakpoint.
    """
    # Everything is labelled relative to the cursor, which is "now". The
    # caller does not call without frames at the cursor, so there is always
    # one to be relative to.
    origin = cursor_frames[-1].ts
    blocks: list[Block] = []

    blocks.append(
        text_block(
            f"THE MOMENT YOU ARE CALLING — {len(cursor_frames)} frames, oldest first. "
            "The last of them is now."
        )
    )
    for i, frame in enumerate(cursor_frames, start=1):
        offset = origin - frame.ts
        when = "this is now" if offset < 0.05 else f"{offset:.1f} s before now"
        blocks.append(text_block(f"Frame {i} of {len(cursor_frames)} — {when}."))
        blocks.append(_marked_block(frame, tracks_for, pack))

    if lookahead_frames:
        blocks.append(
            text_block(
                f"THE NEAR FUTURE — the next {len(lookahead_frames)} images are from "
                "SECONDS AFTER the moment above, nearer the live edge. They are here "
                "so you know how this move ends before you commit to a line. Use them "
                "to choose what to say. Never describe them as if they were happening "
                "now: the viewer has not seen them yet."
            )
        )
        for i, frame in enumerate(lookahead_frames, start=1):
            offset = frame.ts - origin
            blocks.append(
                text_block(
                    f"Lookahead {i} of {len(lookahead_frames)} — {offset:.1f} s after "
                    "the moment you are calling. Outcome check only."
                )
            )
            blocks.append(_marked_block(frame, tracks_for, pack))
    else:
        blocks.append(
            text_block(
                "THE NEAR FUTURE — no lookahead frames are available on this call, so "
                "you do not know how the move ends. Be correspondingly careful: "
                "describe what is happening, do not predict where it finishes."
            )
        )

    blocks.append(text_block(_tail(state_summary, recent_lines, triggers)))
    return blocks


def _tail(
    state_summary: str,
    recent_lines: Sequence[str],
    triggers: Sequence[Trigger],
) -> str:
    """The volatile block: state, what was just said, and why we are asking."""
    state = state_summary.strip() or "Not established yet."
    if recent_lines:
        said = "\n".join(f"  - {line.strip()}" for line in recent_lines)
    else:
        said = "  (nothing said yet)"
    why = ", ".join(t.value for t in triggers) if triggers else "routine tick"
    return (
        "MATCH STATE (read off the scoreboard, not by you — do not repeat it back)\n"
        f"{state}\n\n"
        "THE LAST LINES SPOKEN — do not repeat these and do not paraphrase them\n"
        f"{said}\n\n"
        f"WHY YOU ARE BEING ASKED NOW\n  {why}\n\n"
        "Call the moment shown in the first set of frames, or stay silent. "
        "Fill in the form."
    )


def _pack_section(pack: KnowledgePack) -> str:
    """The knowledge pack as text. Deterministic order everywhere."""
    lines = ["TEAM SHEETS — the only names you are allowed to use."]
    header = " v ".join(filter(None, [pack.home.name, pack.away.name]))
    context = ", ".join(x for x in (pack.competition, pack.venue, pack.kickoff) if x)
    if context:
        lines.append(f"{header} — {context}")
    lines.append("")
    lines.append(_team_section(pack.home, "home"))
    lines.append("")
    lines.append(_team_section(pack.away, "away"))

    if pack.form:
        lines.append("")
        lines.append("Recent form")
        for team in sorted(pack.form):
            lines.append(f"  {team}: {pack.form[team]}")
    if pack.storylines:
        lines.append("")
        lines.append("Before kickoff")
        lines.extend(f"  - {s}" for s in pack.storylines)
    if pack.key_matchups:
        lines.append("")
        lines.append("Watch for")
        lines.extend(f"  - {m}" for m in pack.key_matchups)

    lines.append("")
    lines.append(
        "A name that is not on these sheets does not exist. Use the surname when\n"
        "you use a name at all — \"Di María\", not \"Ángel Di María\" — and only\n"
        "once you have read the number on the shirt or the graphic that\n"
        "identifies the player."
    )
    return "\n".join(lines)


def _team_section(team: TeamSheet, side: str) -> str:
    """One squad, one player per line, so a legible shirt maps straight to a name.

    The kit comes first in the descriptor because it is the thing the caller
    is told to match a number against: a number means nothing until it is
    known which of these two sheets it belongs to.
    """
    bits = [f"kit {team.kit}" if team.kit else "", team.formation or ""]
    if team.manager:
        bits.append(f"manager {team.manager}")
    descriptor = f" — {', '.join(b for b in bits if b)}"
    lines = [f"{team.name} ({side}){descriptor}"]
    lines.append("  Starting XI")
    lines.extend(_squad_lines(team.starters))
    if team.bench:
        lines.append("  Bench")
        lines.extend(_squad_lines(team.bench))
    return "\n".join(lines)


def _squad_lines(players: Sequence[Player]) -> list[str]:
    """``    #7 Bukayo Saka (RW)``, one per line, in the researcher's order.

    One line each rather than a comma-run, because the caller is asked to
    find a number it has just read, and a number is far easier to find down
    a column than inside a paragraph of eighteen of them.
    """
    if not players:
        return ["    not known"]
    rendered: list[str] = []
    for player in players:
        prefix = f"#{player.number} " if player.number is not None else ""
        suffix = f" ({player.position})" if player.position else ""
        rendered.append(f"    {prefix}{player.name}{suffix}")
    return rendered
