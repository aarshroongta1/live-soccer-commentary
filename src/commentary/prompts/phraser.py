"""The phrasing prompt: how a commentator says a thing the caller already saw.

The split this file exists for is the whole idea. The caller is a vision
model looking at six frames and filling in a form, and on real clips it is
good at that: no wrong player name has reached air in sixty-three runs. What
it is bad at is talking. Asked for a sentence it writes a caption —
"Ronaldo walks back into position, hands on hips, waiting for Portugal to
work something forward in these closing minutes" — which is an accurate and
complete description of a picture and is not what anybody says out loud.
Real commentary, measured off a broadcaster's own captions in
``runs/prompt-name/REAL_COMMENTARY.md``, runs to a median of five words, and
a fifth of live-play utterances are a bare surname.

Three rounds of prompt work on the caller did not move it: with fragment
examples in front of it and a cadence that rewards short lines, the Opus
caller wrote nothing under seven words on the Mbappé clip. So seeing and
speaking are separated. The caller keeps its form, its rules and its record;
this prompt gets that form and writes the line.

The same shape as every other prompt here: a long cacheable system string
with the rules and the examples in it, and a short volatile body carrying the
one moment. The examples are most of the bytes and never change, which is
exactly what prompt caching is for — at a tenth of the input price on a
cache read, ninety real utterances cost about as much as nine.
"""

from __future__ import annotations

from collections.abc import Sequence

from commentary.llm.base import Block, text_block
from commentary.prompts.commentary_examples import EXAMPLES, KINDS
from commentary.schemas import CallerLine, Event, Note, Scene, Side

#: What each kind is called in the prompt. The generated file's keys are
#: identifiers; these are the words a commentator would use.
KIND_LABELS = {
    "build_up": "Build-up, the ball moving between players",
    "pass": "A pass, a carry, a tackle, a clearance",
    "shot": "A shot, a header, a chance",
    "save": "A save",
    "goal": "A goal, and talk about goals",
    "foul": "A foul, a card, an offside, a penalty",
    "dead_ball": "A corner, a free kick, a throw, a restart",
    "aside": "The second voice, between passages",
}

PHRASER_RULES = """\
You are the voice of a live football broadcast. Somebody else is watching the
pictures. They have just told you what they can see, on the form below, and
your only job is to say it the way a commentator says it.

You are not adding anything. You are not checking anything. You are not
deciding whether it is worth saying — that has already been decided, and a
line is going out. What you decide is the words.

WHAT COMMENTARY ACTUALLY SOUNDS LIKE

These numbers are measured off a broadcaster's own captions across a whole
World Cup final, on the utterances that carry a player's name:

  median length              5 words
  four words or fewer        48% of utterances
  two words or fewer         24%
  nothing but a surname      19%
  the longest in half an hour   28 words

So: short. Far shorter than feels natural to write down. Most of what you
send back should be under eight words, and a good proportion of it should be
one name, or two names, and nothing else.

THE RULES

Name first. If the form gives you a player, the line starts with that player.
"Tagliafico knocks it infield", never "Argentina knock it infield" when the
name is there and never "the left-back" when the name is there.

A fragment is a line. "De Paul." is a line. "Messi, Álvarez." is a line.
"Now Di María." is a line. The participle form carries most of the build-up
that is not a bare name: "Played by Molina, collected by Upamecano."

Present tense, always. The ball is moving now.

Never explain. No "which means", no "as they look to", no "in these closing
minutes", no "with the clock running down". A listener watching the match
does not need the situation described to them; they can see it.

Never decorate. No adjectives for atmosphere, no "the crowd rises", no "hands
on hips", no scene-setting, no body language. If it is not the ball, a
player, or what just happened to one of them, it does not go in the line.

A full sentence is for the moment that earned it — a shot, a save, a foul, a
card, a goal — and it is still short: "De Paul strikes." "Messi is offside."
"Save. The deflection off Varane flies wide."

A goal is shouted, not narrated. Three to eight words, then the scorer's
name. "Di María! Glorious goal." "And Messi!" Never the score, never the
arithmetic, never "his second" — somebody else is reading the scoreboard.

YOU COMPRESS. YOU NEVER ADD.

This is the one hard rule and it outranks every line of style above it. You
cannot see the match. The form is the whole of what happened; your job is to
say less of it, never more. Every player, every action and every outcome in
your line has to be one that is already in the form — in the event field, in
the players identified, or in the description of what was seen. Reuse its
nouns and its names. A word you reached for that is not in front of you is a
claim about a football match that nobody made.

THE EVENT FIELD IS BINDING. The form says what happened, in one word, and
your line has to be about that and not about the next thing. Making it bigger
is inventing it:

  - the form says penalty — the kick has not been taken. Not "strikes", not
    "buries", not "scores". "Mbappé steps up." is the line.
  - the form says foul or stoppage — nobody has been booked. Never "in the
    book", "booked", "yellow card", "sent off".
  - the form says shot — the ball is not in the net, and nobody has saved it
    either. "Fires wide." not "Goal!" and not "What a save."
  - the form says build_up, pass, carry or none — nothing has happened yet.
    The line is descriptive: a name, two names, where the ball is. If there
    is not even that, return an empty line.
  - only a form that says goal is a goal.

THE SCORE IS NOT YOURS, AND IT HIDES IN ORDINARY WORDS. These are forbidden
however true they feel: "levels it", "level", "the equaliser", "equalises",
"all square", "ahead", "in front", "behind", "back in it", "his second",
"one-nil", "the winner". Somebody else is reading the scoreboard, and a line
that implies a score contradicts them.

Three rewrites that went out and should not have, so that you can see the
shape of the mistake:

  form: foul. Seen: "Otamendi protests, and the referee is already waving
  him away."
    wrong: Otamendi in the book.        (no card anywhere on the form)
    right: Otamendi protests.

  form: penalty. Seen: "And behind him, Martínez, bouncing on his line,
  daring the kick to come."
    wrong: Mbappé strikes.             (the kick has not happened)
    right: Martínez on his line.

  form: goal. Seen: "Mbappé is already into the net for the ball, hauling it
  back to the centre circle."
    wrong: Mbappé! Levels it!          (a score claim, and it was 2-1)
    right: Mbappé has the ball back.

And the names. You may use any player name that appears on the form, spelled
exactly as it is spelled there, and the two team names. Nothing else — not a
name you think is probably out there, not a player who usually takes those,
not a number, not a time. If the form names nobody at all, name nobody: say
what happened, or say the team, or use the role the description gives.
Inventing a name here would put it on air, and every check that stops that
happens after you.

Do not repeat the last lines you are shown, and do not paraphrase them. If
the form is about the same player doing the same thing, a bare surname is the
honest line.

CONTEXT, AND THE ONE THING YOU MAY ADD

Every rule above says you may not add anything. There is one exception and it
is narrow. Some calls carry a block headed `context:` — one-clause facts that
somebody researched before kickoff and that a deterministic check will verify
after you. On those calls, and only on those, your line MAY carry one of those
clauses, word for word or lightly reworded.

"Tagliafico." becomes "Tagliafico, and Argentina have not lost in thirty-six."
"Mbappé steps up." becomes "Mbappé. Three in the tournament already."

The limits, all of which are checked:

  - One clause. Never two, and never a whole note plus a comment on it.
  - The numbers are the note's numbers. A four where the note says three is
    not a rephrase, it is a different claim, and it is struck out.
  - Only about somebody the line already names. A statistic attached to
    nobody is not commentary.
  - The block is empty on anything big. If there is no `context:` block, or
    it says none, then this section does not apply and you add nothing.
  - It never makes the moment louder. A note is an aside dropped into a lull,
    so the excitement is whatever the play deserves — a note is worth zero.
  - You may ignore it. Most lines should. A bare surname is still the honest
    line, and a commentator who used every fact he had would be unlistenable.

EXCITEMENT

A number from 0 to 1 for how this should be said. Ordinary build-up is 0.1 to
0.3. A chance or a foul is around 0.5. A shot on target, a save, a penalty
given: 0.7 to 0.85. A goal is 1.0. Nothing else reads it yet; it is the
volume knob for the voice that will say this.

THE LINE

At most {max_words} words, and usually a third of that. No quotation marks,
no "commentary:", no stage directions, no explanation of what you did. Write
the words a commentator says out loud and nothing else.

If — and this is rare — the form carries nothing a commentator would say at
all, return an empty line. A line that is going out is better short than
absent, so reach for the bare surname long before you reach for silence.\
"""


def phraser_system(*, max_words: int = 16, examples_per_kind: int = 10) -> str:
    """The cacheable prefix: the rules, then real lines by kind.

    Byte-stable for a given pair of arguments, which is what makes sending
    ninety real utterances on every call cost a tenth of sending them once.
    """
    parts = [PHRASER_RULES.format(max_words=max_words), _examples(examples_per_kind)]
    return "\n\n".join(parts)


def _examples(per_kind: int) -> str:
    """A representative sample per kind, not the whole file.

    Even strides rather than the first few, because the generated file is in
    broadcast order and the head of each list is one passage of play.
    """
    lines = [
        "REAL COMMENTARY, FROM THE CAPTIONS OF A WORLD CUP FINAL",
        "",
        "These are transcripts, not instructions. Nothing in them is about the",
        "match you are calling: do not borrow a name, a score or an incident",
        "from them. What they are for is the shape, the length and the register.",
    ]
    for kind in KINDS:
        sample = _stride(EXAMPLES[kind], per_kind)
        if not sample:
            continue
        lines.append("")
        lines.append(f"{KIND_LABELS.get(kind, kind)}")
        lines.extend(f"  {text}" for text in sample)
    return "\n".join(lines)


def _stride(items: Sequence[str], count: int) -> list[str]:
    if count <= 0:
        return []
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


#: The moments a note may be dropped into. Two groups, and both of them are
#: lulls: play that is going on and has not resolved into anything, and a
#: dead ball that nobody has struck yet. Everything else — a shot, a save, a
#: goal, a foul, a tackle — is the moment itself, and a commentator who
#: reached for a statistic in the middle of one would be talking over it.
#:
#: The dead-ball four are safe here for the reason the rules already give
#: about the form's event field: a form that says `penalty` is a penalty
#: being *stood over*. The kick has not been taken, which is precisely why
#: there is a gap to fill.
QUIET_EVENTS = frozenset(
    {
        Event.BUILD_UP,
        Event.PASS,
        Event.CARRY,
        Event.STOPPAGE,
        Event.KICKOFF,
        Event.FREE_KICK,
        Event.PENALTY,
        Event.CORNER,
        Event.THROW_IN,
    }
)


def notes_allowed(line: CallerLine) -> bool:
    """Is this a moment quiet enough to drop a researched clause into?

    Asked here rather than by the runtime so that the answer is part of the
    prompt the prompt module builds, and so that a test can drive the builder
    with a penalty and a goal and see the difference without a model.
    """
    return line.event in QUIET_EVENTS


def phraser_blocks(
    line: CallerLine,
    state_summary: str,
    recent_lines: Sequence[str],
    *,
    home: str,
    away: str,
    on_the_ball: str | None = None,
    notes: Sequence[Note] = (),
) -> list[Block]:
    """One call's content. Text only, and deliberately small.

    No frames. The phraser is not a second opinion on the picture — it has
    never seen the picture, and giving it one would invite it to describe
    what it saw, which is the failure this whole stage exists to fix. It gets
    the form, the state, and — in a lull — the notes about the people on it.

    ``notes`` are already filtered to the moment by the caller; whether they
    are shown at all is decided here, by the event.
    """
    return [text_block(_body(line, state_summary, recent_lines, home, away, on_the_ball, notes))]


def _body(
    line: CallerLine,
    state_summary: str,
    recent_lines: Sequence[str],
    home: str,
    away: str,
    on_the_ball: str | None,
    notes: Sequence[Note] = (),
) -> str:
    said = (
        "\n".join(f"  - {text.strip()}" for text in recent_lines if text.strip())
        or "  (nothing said yet)"
    )
    state = state_summary.strip() or "Not established yet."
    return (
        f"THE TEAMS\n  {home} (home) v {away} (away)\n\n"
        "MATCH STATE — for context only. Never say the score or the clock.\n"
        f"{state}\n\n"
        "WHAT THE EYES SAW — the form, filled in by whoever is watching\n"
        f"{_form(line, home, away, on_the_ball)}\n\n"
        f"{_context(line, notes)}\n\n"
        "THE LAST LINES SPOKEN — do not repeat or paraphrase these\n"
        f"{said}\n\n"
        "Say it."
    )


def _context(line: CallerLine, notes: Sequence[Note]) -> str:
    """The `context:` block: researched clauses, or an explicit nothing.

    Always printed, even when empty, and that is deliberate. A block that
    appears and disappears teaches a model that its absence means "use your
    own knowledge"; a block that is always there and sometimes says none
    teaches it that none means none.
    """
    if not notes or not notes_allowed(line):
        reason = (
            "too big a moment for an aside"
            if notes
            else "nothing researched about anybody on this form"
        )
        return f"context:\n  (none — {reason})"
    rows = [f"  - {note.about}: {note.text.strip()}  [{note.kind}]" for note in notes]
    return "\n".join(
        [
            "context: verified notes. Your line MAY carry ONE of these clauses,",
            "reworded but not renumbered, about somebody the line names. Or none.",
            *rows,
        ]
    )


def _form(line: CallerLine, home: str, away: str, on_the_ball: str | None) -> str:
    """The caller's form, laid out as facts rather than as prose.

    ``line`` is presented as a description of what was seen and labelled as
    one, because it is the thing the phraser must not copy. Left unlabelled
    it reads as a draft to be lightly edited, and a lightly edited caption is
    still a caption.
    """
    rows = [
        f"  scene: {_scene(line.scene)}",
        # Loud, and second, because it is the field the phrasing has to obey.
        # Every invented outcome the stage produced on its first two traces
        # was the line reaching past this word: a penalty being waited on
        # written as a penalty struck, a foul written as a booking.
        f"  EVENT: {_event(line.event)}  <- your line must be about this and",
        "         nothing later than this",
    ]
    team = line.team or _team_of(line.side, home, away)
    if team:
        rows.append(f"  team in possession: {team}")
    names = _names(line)
    if on_the_ball and on_the_ball not in names:
        names = [on_the_ball, *names]
    if names:
        rows.append(f"  players identified: {', '.join(names)}")
        rows.append("  (these, and any name in the description below, are the only")
        rows.append("   player names you may use)")
    else:
        # Not the same as "name nobody". The caller may name a player it read
        # seconds ago and can no longer see — the carry rule — and that name
        # is in the description with no sighting behind it. Telling the
        # phraser there is nobody would throw it away, and a line that says
        # "Portugal" where the caller said "Ronaldo" is the failure this
        # whole stage is meant to fix in the other direction.
        rows.append("  players identified: none read off the picture this call")
        rows.append("  (you may still use a name that appears in the description below)")
    rows.append(f"  confidence in all of the above: {line.confidence:.2f}")
    rows.append("")
    rows.append("  what was seen, written down as a description. DO NOT say this back.")
    rows.append("  Take the facts out of it and say them the way a commentator would,")
    rows.append(f"  using only what is here and only about a {_event(line.event)}:")
    rows.append(f"    {line.line.strip()}")
    return "\n".join(rows)


def _names(line: CallerLine) -> list[str]:
    """Every name the form carries, in order, without repeats."""
    found: list[str] = []
    for sighting in line.sightings:
        name = (sighting.name or "").strip()
        if name and name not in found:
            found.append(name)
    return found


def _team_of(side: Side, home: str, away: str) -> str:
    if side is Side.HOME:
        return home
    if side is Side.AWAY:
        return away
    return ""


def _scene(scene: Scene) -> str:
    return scene.value.replace("_", " ")


def _event(event: Event) -> str:
    return event.value.replace("_", " ")
