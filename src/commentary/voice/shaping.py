"""Punctuation as a performance direction, not as grammar.

A TTS model reads punctuation. "Mbappé, buried." and "Mbappé! Buried!" are
the same six syllables and two different reads, and the second is what a
commentator actually does at the moment a ball crosses the line: the name is
its own utterance, and what follows is another. The voice settings get the
delivery most of the way there; this closes the last of the gap for free, on
the text, before a byte leaves the process.

It is opt-in (``VOICE_SHAPING=on``) for one reason: it edits the line the
gate approved. The gate checks claims, not commas, so a full stop becoming an
exclamation mark cannot turn a true line false — but it is still the one
place in the pipeline where text changes after the check, and a thing like
that should be switched on deliberately and by somebody who has listened to
both. Words are never added, removed or reordered; only terminal punctuation
and the capital that follows a new sentence break.

Only the caller's lines go through here, and "caller" means the seat, not the
speaker: the analyst is the seat that never shouts, and an analyst line
ending in an exclamation mark is the two voices becoming one. That check
belongs to the speaker, which is the only thing that knows which seat asked.
"""

from __future__ import annotations

import re

#: Above this a line is a goal, a save, or a miss from two yards, and it gets
#: the punctuation of one.
SHOUT_AT = 0.85

#: At or below this it is build-up, and an exclamation mark on build-up is
#: what makes a voice sound like it is shouting at nothing all match.
CALM_AT = 0.3

#: A single capitalised word, then a comma: "Mbappé, buried." The bare name
#: opening is the commonest thing the phraser writes and the one place a
#: sentence break is unambiguously right. Anything with a space in it before
#: the comma is a clause, not a name, and is left alone.
_LEADING_NAME = re.compile(r"^([^\W\d_][\w'’\-]*),\s+(\S.*)$")

#: Ends the line without needing anything doing to it. A trailing ellipsis is
#: a line trailing off, and it is left trailing off.
_ALREADY_EMPHATIC = ("!", "?", "…", "...")


def shape_for_voice(text: str, excitement: float) -> str:
    """Repunctuate one line for how hard it is being said.

    Loud lines get a sentence break after a bare leading name and end on an
    exclamation mark; quiet lines lose a trailing one. Everything in between
    is returned exactly as it came in, as is anything this cannot read
    confidently.
    """
    line = text.strip()
    if not line:
        return text
    if excitement >= SHOUT_AT:
        return _end_with_bang(_split_leading_name(line))
    if excitement <= CALM_AT:
        return line[:-1] + "." if line.endswith("!") else line
    return line


def _split_leading_name(line: str) -> str:
    """"Mbappé, buried." becomes "Mbappé! Buried." — the comma becomes a call."""
    match = _LEADING_NAME.match(line)
    if match is None:
        return line
    name, rest = match.group(1), match.group(2)
    return f"{name}! {rest[0].upper()}{rest[1:]}"


def _end_with_bang(line: str) -> str:
    if line.endswith(_ALREADY_EMPHATIC):
        return line
    if line.endswith("."):
        return line[:-1] + "!"
    return line + "!"


__all__ = ["CALM_AT", "SHOUT_AT", "shape_for_voice"]
