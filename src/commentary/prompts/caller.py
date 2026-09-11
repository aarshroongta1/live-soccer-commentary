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

from collections.abc import Sequence

from commentary.capture.buffer import Frame
from commentary.config import CallerConfig
from commentary.llm.base import Block, encode_frame, image_block, text_block
from commentary.schemas import KnowledgePack, Player, TeamSheet, Trigger

#: The rules. ``{max_words}`` is the only substitution, and it comes from
#: config so a sweep of the word cap changes the prompt with it.
CALLER_RULES = """\
You are the play-by-play commentator on a live football broadcast.

You have the pictures, a team sheet written before kickoff, and nothing else.
No data feed, no statistician, nobody talking in your ear. Everything you say
has to be something you can see happening.

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

Names. Say a player's name only if you can actually read it — a shirt number
you can see, a name across the back, a name in a broadcast graphic. Otherwise
say the team, the position, or the shirt: "the near-post runner", "the
left-back in red", "the man in white". A wrong name is the worst thing you
can do here. There is no credit for guessing and no penalty for saying
"Arsenal" when you cannot see who it is.

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

THE FORM

Fill every field from the picture, not from the story you would like to tell.
Confidence is your honest read on whether these frames support the claim; low
confidence is not punished, but a confident guess is. names_read holds only
what is legible right now — a shirt number counts, a graphic counts, knowing
who usually plays there does not.

THE LINE

At most {max_words} words. One sentence. Present tense. No preamble, no sign-off,
no quotation marks, no "we see", no "in this frame". Write what a commentator
says out loud, not what an observer writes down.

  Good: Saka drives at the full-back and wins the corner.
  Good: Cutback from the right, and it is hammered over the bar.
  Good: Long ball forward, and the centre-half in red heads it clear.
  Bad:  In this frame we can see a player in a red shirt. (describing a picture)
  Bad:  Odegaard picks out Havertz. (names nobody could read off these frames)
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


def caller_blocks(
    cursor_frames: Sequence[Frame],
    lookahead_frames: Sequence[Frame],
    state_summary: str,
    recent_lines: Sequence[str],
    triggers: Sequence[Trigger],
) -> list[Block]:
    """One call's content: the moment, the near future, then the volatile tail.

    Every image gets a text label in front of it, because six unlabelled
    pictures of the same pitch are indistinguishable to the model and the
    whole lookahead idea rests on it knowing which two are the future. The
    state, the recent lines and the triggers go last, after the images, so
    that the part of the message that changes fastest sits furthest from the
    cache breakpoint.
    """
    blocks: list[Block] = []
    origin = _origin_ts(cursor_frames, lookahead_frames)

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
        blocks.append(image_block(encode_frame(frame.image)))

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
            blocks.append(image_block(encode_frame(frame.image)))
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


def _origin_ts(cursor: Sequence[Frame], lookahead: Sequence[Frame]) -> float:
    """The timestamp everything is labelled relative to: the cursor, i.e. now."""
    if cursor:
        return cursor[-1].ts
    if lookahead:
        return lookahead[0].ts
    return 0.0


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
        "you use a name at all, and only once you have read the number on the\n"
        "shirt or the graphic that identifies the player."
    )
    return "\n".join(lines)


def _team_section(team: TeamSheet, side: str) -> str:
    """One squad, numbers first, so a legible shirt maps straight to a name."""
    bits = [b for b in (team.kit, team.formation) if b]
    if team.manager:
        bits.append(f"manager {team.manager}")
    descriptor = f" — {', '.join(bits)}" if bits else ""
    lines = [f"{team.name} ({side}){descriptor}"]
    lines.append(f"  Starting XI: {_squad_line(team.starters)}")
    if team.bench:
        lines.append(f"  Bench: {_squad_line(team.bench)}")
    return "\n".join(lines)


def _squad_line(players: Sequence[Player]) -> str:
    """``7 Bukayo Saka (RW)``, in the order the researcher listed them."""
    rendered: list[str] = []
    for player in players:
        prefix = f"{player.number} " if player.number is not None else ""
        suffix = f" ({player.position})" if player.position else ""
        rendered.append(f"{prefix}{player.name}{suffix}")
    return ", ".join(rendered) if rendered else "not known"
