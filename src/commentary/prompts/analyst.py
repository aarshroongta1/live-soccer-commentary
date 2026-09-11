"""The colour prompt: a stable cacheable prefix and a volatile per-call body.

The analyst's prompt has one job the caller's does not: keeping two voices off
each other's toes. A second voice that describes the picture is not a second
voice, it is an echo, and it is the single most common way two-voice
commentary sounds wrong. So most of the rules below are about what the analyst
may *not* say, and the per-call body ends with what has just gone out on air
so the model can see for itself what has already been covered.

The split follows the caller's for the same reason: the system string carries
the rules and both squads, it is sent on every call, and it only pays for
itself if it is byte-identical each time. Nothing in here may depend on the
clock or on unordered iteration. Everything that moves — the frames, the
state, the facts looked up for this moment, what was just said, why the
analyst is being asked at all — lives in the per-call blocks, after the cache
breakpoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from commentary.capture.buffer import Frame
from commentary.config import AnalystConfig
from commentary.llm.base import Block, encode_frame, image_block, text_block
from commentary.schemas import KnowledgePack, Player, TeamSheet

#: The rules. ``{max_words}`` is the only substitution, and it comes from
#: config so a sweep of the word cap changes the prompt with it.
ANALYST_RULES = """\
You are the analyst on a live football broadcast, working alongside a
play-by-play commentator who is calling the ball.

THE DIVISION OF LABOUR

Your colleague has the ball. You have the match.

They say what is happening. You say what it means, why it keeps happening,
what it costs, and what was true before kickoff that makes this passage worth
watching. Those are two different jobs, and doing theirs is the fastest way to
sound wrong: a second voice that describes the picture is not a second voice,
it is an echo.

So never narrate the frames. Not "he plays it square", not "the cross comes
in", not "Arsenal are on the attack". Lines like that belong to the other
voice, who has better pictures than you do and has probably said them already.
If the only thing you have to offer is what the viewer can already see, say
nothing. The frames you are given are context for the point you are making,
not a moment to call.

WHEN YOU SPEAK

You get your turn in the gaps: a stretch where nobody has said anything, or
the seconds after something big has settled — a goal, a card, a penalty, a
save that mattered. You are told below which of the two it is.

Even then, silence is a real answer and it is the common one. Ninety minutes
has room for perhaps thirty of your lines. Speak when you have something
specific: a pattern that has now repeated, a note from before kickoff that
this moment turns into a fact, a consequence nobody has stated. If what you
have is generic — "it has been a tight game", "they will want to keep this
going" — set speak to false. A gap sounds like a match. Filler sounds like a
machine that has to talk.

WHAT YOU ARE ALLOWED TO KNOW

Three things: the pictures, the notes written before kickoff, and the match
state the system read off the scoreboard and off the pitch. There is no data
feed here and nobody in your ear. A number you did not read below is a number
you made up, so do not reach for a statistic, a date, a fee, a record or a run
of results that is not in front of you.

Use a player's name only if the notes name them and the picture or the state
supports their being involved. Otherwise the team, the position, or the shirt.
A wrong name is worse than no name.

THE SCORE IS FOR THINKING WITH, NOT FOR READING OUT

You are given the scoreline so you can weigh what a moment is worth: a second
goal at one-nil is a different thing from a second goal at three-nil. You are
not given it to say out loud. Someone else is reading the scoreboard. Never
restate it, in figures or in words — no "one-nil", no "two apiece", no "all
square", no "that is the equaliser". If a line survives only because it
announces the score, it is not a line.

THE FORM

angle is the kind of point you are making, so the director can stop you making
the same kind twice in a row.

cites is what you leaned on, named plainly: the storyline, the form line, the
matchup, the run of events, what the state says. Fill it in even when the lean
is light. If you cannot say what you leaned on, you are guessing, and a guess
is not colour — set speak to false instead.

confidence is your honest read on whether the notes and the state support the
claim. It is not how good the line sounds.

THE LINE

At most {max_words} words. One sentence, two at the most — you have more room
than the caller and you should not use all of it every time. Spoken English:
the way someone talks in a gantry, not the way someone writes on a page. No
preamble, no label, no quotation marks, no "in this frame", no sign-off.

  Good: Third time they have gone down that left side in ten minutes, and the
        full-back has had no help all half.
  Good: Chelsea came into this on one win in six, and you can see it in how
        deep they have settled.
  Good: That is the matchup the notes flagged before kickoff, and so far it
        has only gone one way.
  Bad:  Saka has the ball wide on the right. (that is the other voice's job)
  Bad:  It is one-nil with half an hour to go. (reading the scoreboard back)
  Bad:  They will be desperate to get something from this game. (true of
        everyone, leaned on nothing)
  Bad:  Arsenal have not lost here in fourteen months. (a fact nobody gave you)\
"""


def analyst_system(pack: KnowledgePack | None, config: AnalystConfig | None = None) -> str:
    """The cacheable prefix: the rules, then the notes written before kickoff.

    Byte-stable by construction for a given pack — no timestamps, no dict
    iteration that is not sorted first — because this string is sent on every
    call and only earns the squads it carries if it caches.
    """
    cfg = config or AnalystConfig()
    parts = [ANALYST_RULES.format(max_words=cfg.max_words)]
    if pack is not None:
        parts.append(_notes_section(pack))
    return "\n\n".join(parts)


def analyst_blocks(
    frames: Sequence[Frame],
    state_summary: str,
    tool_facts: Mapping[str, Any],
    recent_lines: Sequence[str],
    reason: str,
) -> list[Block]:
    """One call's content: a wide window of pictures, then the volatile tail.

    The window is deliberately sparse — six frames spread over twenty seconds
    rather than four over four — because the analyst is looking for what has
    been true for a while, not for what just happened. Each image is labelled
    with how far back it sits, since six unlabelled pictures of the same pitch
    tell a model nothing about the direction of travel.

    ``recent_lines`` may carry a speaker label ("the caller: ...", "you: ...")
    and is rendered as given: knowing which voice said what is the difference
    between not repeating yourself and not repeating your colleague. It goes
    last, with the state, the looked-up facts and the reason, so that the
    fastest-moving part of the message sits furthest from the cache
    breakpoint.
    """
    blocks: list[Block] = []
    origin = frames[-1].ts if frames else 0.0
    span = origin - frames[0].ts if frames else 0.0

    if frames:
        blocks.append(
            text_block(
                f"THE LAST {span:.0f} SECONDS — {len(frames)} frames, oldest first, spread "
                "wide rather than bunched together. They are here so you can see what has "
                "been true for a while. The other voice is already calling the last one."
            )
        )
        for i, frame in enumerate(frames, start=1):
            offset = origin - frame.ts
            when = "the most recent look" if offset < 0.05 else f"{offset:.1f} s earlier"
            blocks.append(text_block(f"Frame {i} of {len(frames)} — {when}."))
            blocks.append(image_block(encode_frame(frame.image)))
    else:
        blocks.append(
            text_block(
                "NO PICTURES ARE AVAILABLE ON THIS CALL — work from the state and the "
                "notes alone, and say nothing that would need a picture to justify it."
            )
        )

    blocks.append(text_block(_tail(state_summary, tool_facts, recent_lines, reason)))
    return blocks


def _tail(
    state_summary: str,
    tool_facts: Mapping[str, Any],
    recent_lines: Sequence[str],
    reason: str,
) -> str:
    """State, the facts looked up for this moment, what was just said, and why."""
    state = state_summary.strip() or "Not established yet."
    if recent_lines:
        said = "\n".join(f"  - {line.strip()}" for line in recent_lines)
    else:
        said = "  (nothing said yet)"
    why = reason.strip() or "a lull"
    return (
        "MATCH STATE (read off the scoreboard and the pitch — do not read it back out)\n"
        f"{state}\n\n"
        "WHAT YOU LOOKED UP FOR THIS MOMENT — chosen because it bears on what is "
        "happening now, not because it is everything on file\n"
        f"{_render_facts(tool_facts)}\n\n"
        "WHAT HAS JUST GONE OUT ON AIR — do not repeat any of it and do not paraphrase "
        "it either\n"
        f"{said}\n\n"
        f"WHY YOU ARE BEING ASKED NOW\n  {why}\n\n"
        "Say the thing the picture does not show, or stay quiet. Fill in the form."
    )


def _render_facts(facts: Mapping[str, Any]) -> str:
    """The gathered facts as indented text, in the order they were gathered."""
    if not facts:
        return "  (nothing looked up)"
    lines: list[str] = []
    for key, value in facts.items():
        label = key.replace("_", " ")
        rendered = _render_value(value, indent="    ")
        if "\n" in rendered or not rendered:
            lines.append(f"  {label}")
            lines.append(rendered or "    (none)")
        else:
            lines.append(f"  {label}: {rendered.strip()}")
    return "\n".join(lines)


def _render_value(value: Any, indent: str) -> str:
    """Scalars inline, lists as bullets, mappings as key-value lines."""
    if isinstance(value, Mapping):
        parts = [f"{indent}{k}: {_scalar(v)}" for k, v in value.items() if v not in (None, "")]
        return "\n".join(parts)
    if isinstance(value, list | tuple):
        return "\n".join(f"{indent}- {_scalar(v)}" for v in value)
    return _scalar(value)


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "unknown"
    return str(value)


def _notes_section(pack: KnowledgePack) -> str:
    """The knowledge pack as text. Deterministic order everywhere.

    The same pack the caller gets, framed differently: for the caller it is a
    whitelist of names, for the analyst it is the material. Form, storylines
    and matchups are standing notes, so they sit here in the cached prefix;
    which of them matters right now arrives per call, in the looked-up facts.
    """
    lines = ["THE NOTES — written before kickoff, and the only outside information you have."]
    header = " v ".join(filter(None, [pack.home.name, pack.away.name]))
    context = ", ".join(x for x in (pack.competition, pack.venue, pack.kickoff) if x)
    lines.append(f"{header} — {context}" if context else header)
    lines.append("")
    lines.append(_team_section(pack.home, "home"))
    lines.append("")
    lines.append(_team_section(pack.away, "away"))

    if pack.form:
        lines.append("")
        lines.append("Recent form")
        lines.extend(f"  {team}: {pack.form[team]}" for team in sorted(pack.form))
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
        "These notes are the whole of what you know that is not on the screen. A name,\n"
        "a result or a storyline that is not written here is not available to you, and\n"
        "inventing one is the worst thing you can do in this job."
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
