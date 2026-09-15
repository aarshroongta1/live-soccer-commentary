"""The speaking voice: one cheap call that turns a form into six words.

The caller's perception is good and its prose is not. Across sixty-three runs
on real footage it has never put a wrong player name on air, and it has also
never written a line under seven words — with fragment examples in its prompt
and a cadence that pays for short lines. Asked to look and to talk at once it
does the looking well and writes a caption:

    Ronaldo walks back into position, hands on hips, waiting for Portugal to
    work something forward in these closing minutes.

Real commentary is five words. So this module is the second half of that job,
split off: the caller keeps the pictures, the form and the rules about what
may be claimed, and the phraser gets the finished form and writes the line.
It never sees a frame, which is deliberate — a phraser with a picture starts
describing the picture, and that is the failure being fixed.

What it costs. One Haiku call per spoken line, with the register's two
hundred real utterances in a cached system prefix and a few hundred bytes of
form in the body. A tenth of a cent, give or take, against the caller's three
cents. ``PHRASER_MODEL=off`` removes the stage entirely and the runtime is
what it was before this file existed.

What it is not allowed to do. Everything it writes goes through the same fact
gate the caller's line went through — same roster check, same scoreline
arithmetic, same goal rule — so a name it invents is a name that never
reaches the speaker. The prompt tells it this; the gate is what enforces it.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from commentary.agents.caller import clean_line, trim_words
from commentary.agents.colour import mentions, says_a_number
from commentary.config import PHRASER_MODEL, DeadBallConfig, PhraserConfig, SilenceConfig
from commentary.ledger import Fact as LedgerFact
from commentary.llm.base import Block, LLMBackend, LLMError, Parsed, Usage, text_block
from commentary.prompts.phraser import (
    BUILD_UP_FORMS,
    is_long_line,
    phraser_blocks,
    phraser_system,
    strip_replay_marker,
)
from commentary.schemas import (
    CallerLine,
    Event,
    KnowledgePack,
    Note,
    PhrasedLine,
    Scene,
)

#: The beat that is one number about the scorer — see ``GOAL_BEATS[3]`` in
#: ``prompts/phraser.py``. The only beat this module ever needs to
#: distinguish, because it is the only one where a name is mandatory and a
#: number is expected in the same line.
SCORER_BEAT = 3

#: The beats that are *not* the call, and so may not be shouted on a name.
#: Beat 1 is the goal call and "<Scorer>!" is exactly what it should be; beats
#: 2 and 3 are the celebration and the tally, and on
#: ``runs/rephrased/r1-replay/mbappe`` all three opened "Mbappé!" — a listener
#: hearing three goals in twelve seconds. Beat 4 is past tense and has never
#: done it.
UNSHOUTED_BEATS = frozenset({2, 3})

#: The value of ``PHRASER_MODEL`` that means "do not run this stage".
OFF = "off"

#: How many of the most recently spoken lines an opener is checked against.
#: Matches the prompt's own "last two" rule with slack: the prompt asks the
#: model not to open on either of the last two, and real commentary repeats
#: an opener about one line in eight looking back further than that, so
#: checking five catches the failure (35-48% repeats) without punishing the
#: normal, occasional one-in-eight echo from further back.
_OPENER_LOOKBACK = 5

#: And the same window for the last word. The tail is the same tell as the
#: opener and it is the one this voice actually commits: "Through midfield
#: now." / "Wide on the right now." / "Into the corner now." / "Striding out
#: now, France in no hurry to move it on." is four lines in five ending on
#: one word and no two of them opening on one, so the opener check waved
#: every one of them through.
_CLOSER_LOOKBACK = 5

_LEADING_TRAILING_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")
_POSSESSIVE = re.compile(r"['’]s$", re.IGNORECASE)

#: A line that opens by shouting a name: one or two capitalised words and an
#: exclamation mark, which is the goal call's own shape. Whether the words
#: are actually a *name* is decided against the form and the scorer, not by
#: this pattern — "Buried!" and "Save!" open the same way and are not it.
_OPENING_SHOUT = re.compile(r"^\s*([A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*)?)\s*!")


def _opening_word(text: str) -> str:
    """The line's first word, as it would be judged for a repeat: punctuation
    and a trailing possessive stripped, case-folded.
    """
    tokens = text.strip().split()
    if not tokens:
        return ""
    core = _LEADING_TRAILING_PUNCT.sub("", tokens[0])
    core = _POSSESSIVE.sub("", core)
    return core.casefold()


def _closing_word(text: str) -> str:
    """The line's last word, judged the same way the first one is."""
    tokens = text.strip().split()
    if not tokens:
        return ""
    core = _LEADING_TRAILING_PUNCT.sub("", tokens[-1])
    core = _POSSESSIVE.sub("", core)
    return core.casefold()


def _display_word(text: str) -> str:
    """The same word, kept in its written case, for naming in the retry note."""
    tokens = text.strip().split()
    if not tokens:
        return ""
    core = _LEADING_TRAILING_PUNCT.sub("", tokens[0])
    return _POSSESSIVE.sub("", core)


def _display_closer(text: str) -> str:
    """The closing word, kept in its written case, for naming in the note."""
    tokens = text.strip().split()
    if not tokens:
        return ""
    core = _LEADING_TRAILING_PUNCT.sub("", tokens[-1])
    return _POSSESSIVE.sub("", core)


def _is_bare_name(text: str) -> bool:
    """One word and nothing else — a legitimate repeat, not a repeated frame."""
    return len(text.strip().split()) == 1


def _with_note(blocks: Sequence[Block], note: str) -> list[Block]:
    """The same call's body, with a note appended to its last text block."""
    new_blocks = [dict(block) for block in blocks]
    for block in reversed(new_blocks):
        if block.get("type") == "text":
            block["text"] = f"{block['text']}\n\n{note}"
            return new_blocks
    new_blocks.append(text_block(note))
    return new_blocks


def _register_retry_note(opener: str, closer: str) -> str:
    """Name the repeat — the first word, the last word, or both — and re-ask.

    One note and one re-ask for the pair, rather than two calls for one line.
    Both faults are the same fault measured at opposite ends of the sentence,
    the fix for either is to write a different sentence, and a model asked
    twice about one line costs twice and answers worse the second time.
    """
    faults = []
    if opener:
        faults.append(f'it OPENED on "{opener}", which one of the last five lines already used')
    if closer:
        faults.append(f'it ENDED on "{closer}", which one of the last five lines already ended on')
    return (
        "THAT LINE REPEATS ONE OF THE LAST FIVE: "
        + ", and ".join(faults)
        + ".\nSay it differently this time — a different subject, a different verb, or the\n"
        "detail instead — or return an empty line. Do not simply move the word."
    )


def _shout_retry_note(name: str, beat: int) -> str:
    """Say which shape was written, why it is beat 1's and not this one's."""
    return (
        f'THAT OPENED ON "{name}!", WHICH IS THE GOAL CALL\'S OWN SHAPE. The call has '
        "already gone out with the score on the end of it, and a second line in that "
        f"shape is a second goal to whoever is listening. Beat {beat} is not a shout.\n"
        "Write it again without the name-and-exclamation-mark at the front: the name "
        "may be anywhere else in the line."
    )


def opening_shout(text: str, names: Sequence[str]) -> str | None:
    """The name this line opens by shouting, or ``None``.

    A name, not a word: "Save!" and "Buried!" open the same way and neither is
    the fault. So the capitalised opener is matched against the people this
    call actually knows about — the scorer, the names on the form, the two
    team sheets — by surname, the way every other name check in this system
    works, because the form spells a man "Kylian Mbappé" and the line says
    "Mbappé".
    """
    match = _OPENING_SHOUT.match(text)
    if match is None:
        return None
    opener = match.group(1).strip()
    wanted = {opener.casefold(), opener.rsplit(" ", 1)[-1].casefold()}
    for name in names:
        clean = (name or "").strip()
        if not clean:
            continue
        if clean.casefold() in wanted or clean.rsplit(" ", 1)[-1].casefold() in wanted:
            return opener
    return None


def unshout(text: str) -> str:
    """Take a "Name!" off the front of a follow-up beat, leaving a line behind.

    Two shapes, and which one is used is decided by the word after the
    exclamation mark. A lower-case one is a clause the name is the subject or
    the addressee of, and the comma is what a commentator would have written:
    "Mbappé! and away he goes" becomes "Mbappé, and away he goes". An
    upper-case one is a sentence of its own with a shout stuck in front of
    it, so the shout comes off: "Mbappé! The keeper sent the wrong way."
    becomes "The keeper sent the wrong way."

    A line that is *only* the shout is returned untouched. There is nothing
    under it to promote, and an empty line here would drop the beat — which
    is the one thing this must not do, because the beat is what the thirty
    seconds after a goal are made of.
    """
    match = _OPENING_SHOUT.match(text)
    if match is None:
        return text
    rest = text[match.end() :].lstrip()
    if not rest:
        return text
    if rest[0].islower():
        return f"{match.group(1).strip()}, {rest}"
    return rest


def roster_names(pack: KnowledgePack | None) -> list[str]:
    """Every name on either team sheet, for :func:`opening_shout`.

    A goal call's sightings sometimes carry a shirt number and no name at all
    — on the Mbappé penalty every one of them did — so the man the follow-up
    beats are about can be a name that is on the roster and nowhere on the
    form. The shout check needs to recognise him there.
    """
    if pack is None:
        return []
    return [
        player.name
        for sheet in (pack.home, pack.away)
        for player in sheet.squad
        if player.name.strip()
    ]


def nameless_build_up(line: CallerLine, *, on_the_ball: str | None = None) -> bool:
    """Is this form the ball moving between nobody in particular?

    Three conditions, all of them the caller's own record of what it saw: the
    event is one of the words it has for the ball going forward with nothing
    happening to it, no sighting bound to a roster name and no name carried
    on the ball from a few seconds ago, and no ``detail`` — the one concrete
    thing it is asked to pick out of the picture.

    ``docs/research/real-commentary-corpus.md`` section 3.1a: 24% of carries
    and passes in build-up have nothing said within ±3 s of them, and even
    counting generously only 38% of carries have their player named. This is
    the shape of the ones that are not said, and
    :meth:`Phraser.passes_over` is where it is acted on.
    """
    if line.scene is Scene.REPLAY:
        # A replay is on screen to be talked over, and its form carries the
        # event of the incident rather than of the picture.
        return False
    if line.event not in BUILD_UP_FORMS and line.event is not Event.NONE:
        return False
    if (line.detail or "").strip():
        return False
    if (on_the_ball or "").strip():
        return False
    return not any((sighting.name or "").strip() for sighting in line.sightings)


def _name_retry_note(scorer: str) -> str:
    """Two lines: say the number reached nobody, then say who it is about."""
    return (
        "THAT HAD A NUMBER IN IT AND NOBODY'S NAME ON IT — a fact about nobody, which the "
        f"gate refuses. The line must name {scorer}.\n"
        "Keep the number and the fact; add his name, in the same number of words."
    )


@dataclass(frozen=True)
class Said:
    """One line that went out, and the three things about it worth keeping.

    The words are what the prompt shows back. The kind is what lets the model
    tell a second line about the same carry from a new moment. ``nameless``
    and ``ts`` are what :meth:`Phraser.passes_over` reads, and neither can be
    recovered from the words afterwards.
    """

    text: str
    event: Event | None = None
    nameless: bool = False
    ts: float | None = None


def phraser_enabled(model: str = PHRASER_MODEL) -> bool:
    """Is the phrasing stage switched on at all?"""
    return model.strip().lower() not in (OFF, "", "none")


class Phraser:
    """A form in, a line a commentator would say out.

    Holds its own memory of what has been said, like the caller does and for
    the same reason: the prompt shows the last few lines back so the voice
    does not repeat itself. It is a separate memory from the caller's because
    the two are looking at different text — the caller sees its own
    descriptions, the phraser sees what actually went out.
    """

    def __init__(
        self,
        backend: LLMBackend,
        config: PhraserConfig | None = None,
        *,
        model: str | None = None,
        home: str = "Home",
        away: str = "Away",
        silence: SilenceConfig | None = None,
        dead_ball: DeadBallConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or PhraserConfig()
        #: When a form is passed over without a model call at all. See
        #: :meth:`passes_over`.
        self.silence = silence or SilenceConfig()
        #: The restart slot: what the body block asks for and how far the
        #: word cap is lifted for it. See
        #: :func:`commentary.prompts.phraser.is_long_line`.
        self.dead_ball = dead_ball or DeadBallConfig()
        self.model = model if model is not None else self.config.model
        self.home = home
        self.away = away
        #: Built once. Identical bytes every call is what makes two hundred
        #: real utterances affordable to send on each line.
        self.system = phraser_system(
            max_words=self.config.max_words,
            examples_per_kind=self.config.examples_per_kind,
        )
        #: What was said, and what each line was about. The kind is carried
        #: beside the words because the two rules that need it — do not open
        #: the same way twice, and say nothing when the last line was this
        #: same moment about this same man — cannot be checked from the text
        #: alone. A model shown "Upamecano works it forward." has no way to
        #: know whether that was a carry or a tackle.
        #:
        #: The third field is whether the line went out over a form with
        #: nobody on it — see :func:`nameless_build_up`. It cannot be read
        #: back off the words either: "Through midfield now." names nobody
        #: and neither does "Away.", and only one of the two came off a form
        #: with a bound name.
        self._recent: deque[Said] = deque(maxlen=max(1, self.config.recent_lines))
        #: Why the last call produced nothing, in words, for the error row.
        self.last_reason = ""
        #: Did the last call choose to say nothing? A model that returns an
        #: empty line has decided this moment is one of the ones real
        #: commentary passes over — 43% of goal kicks, 37% of throw-in
        #: deliveries, 33% of free-kick deliveries, 31% of kickoffs and a
        #: quarter of all build-up touches
        #: (``docs/research/real-commentary-corpus.md`` section 3). That is a
        #: different event from a call that failed or a line that was
        #: nothing but a label, and the two used to be one flag: both came
        #: back as an empty line and both fell back to the caller's words,
        #: so the phraser could never choose silence. Read it beside
        #: :meth:`phrase` returning ``None``, which is still a failure.
        self.chose_silence = False
        #: What the last call cost, so a rephrase can price itself per line.
        self.last_usage = Usage()

    @property
    def enabled(self) -> bool:
        return phraser_enabled(self.model)

    @property
    def recent(self) -> list[str]:
        """What has actually been said, oldest first, each tagged with its kind.

        The tag is what the prompt reads as "a pass, 2 lines ago": without it
        the model cannot tell a second line about the same carry from a
        genuinely new moment, and the corpus says the second one is silence
        a quarter of the time.
        """
        # The kind goes after the words, not in front of them. In front, the
        # model read the tag as the opener and stopped obeying the rule about
        # not opening two lines the same way: two adjacent lines both
        # starting "Argentina" went out in the round that tried it.
        return [
            f"{said.text}   ({said.event.value if said.event else 'no kind'})"
            for said in self._recent
        ]

    def accept(
        self,
        line: str,
        event: Event | None = None,
        *,
        ts: float | None = None,
        nameless: bool = False,
    ) -> None:
        """Record a line as spoken. Only call this when it really is going out.

        ``ts`` and ``nameless`` are what :meth:`passes_over` reads back: when
        the last thing said was a nameless build-up line, and how long ago.
        Both default to the answer every caller of this gave before the
        silence rule existed — no time, and named — so an older caller keeps
        asking the model about every form, which is what it used to do.
        """
        text = line.strip()
        if text:
            self._recent.append(Said(text=text, event=event, nameless=nameless, ts=ts))

    def passes_over(
        self,
        line: CallerLine,
        *,
        ts: float,
        on_the_ball: str | None = None,
    ) -> str | None:
        """The reason to say nothing about this form, or ``None`` to ask the model.

        The one decision this stage makes without spending a call. The corpus
        (``docs/research/real-commentary-corpus.md`` section 3.1a) is that
        24% of carries and passes in build-up pass with nothing said and only
        38% of carries name anybody; the prompt has said so for four rounds
        and the measured result was **one chosen silence in 35 calls**, with
        "Through midfield now. / Wide on the right now. / Into the corner
        now." going out instead. A model asked to choose silence about a
        picture somebody has just handed it will write something.

        So the narrowest version of the corpus's own rule is decided here:
        this form has nobody on it and nothing on it
        (:func:`nameless_build_up`), and the last thing that went out was the
        same, recently. Never the first of a pair — the first nameless
        build-up line is ordinary commentary — and never when a name is bound,
        a name is carried, or the eyes picked out a detail.
        """
        if not self.silence.enabled or not nameless_build_up(line, on_the_ball=on_the_ball):
            return None
        recent = list(self._recent)[-max(1, self.silence.after_nameless) :]
        if len(recent) < self.silence.after_nameless:
            return None
        if not all(said.nameless for said in recent):
            return None
        if any(
            said.ts is None or not 0.0 <= ts - said.ts <= self.silence.within_s for said in recent
        ):
            # Older than the window, or from a caller that does not stamp
            # what it says. A quiet passage answered with more quiet is how a
            # system goes mute, and the corpus's longest silences are bounded.
            return None
        return "silence: nameless build-up after nameless build-up"

    async def phrase(
        self,
        line: CallerLine,
        state_summary: str,
        *,
        on_the_ball: str | None = None,
        notes: Sequence[Note] = (),
        callbacks: Sequence[bool] = (),
        ledger: Sequence[LedgerFact] = (),
        followup: str = "",
        goal_beat: int | None = None,
        scorer: str | None = None,
        roster: Sequence[str] = (),
        replay_first: bool = True,
    ) -> PhrasedLine | None:
        """Rewrite one caller line, or return ``None`` if the call failed.

        ``None`` is a failure and the caller's own words go out instead: a
        line the gate was about to pass is worth more spoken badly than
        lost. An empty ``line`` on the returned object is the other thing —
        the phraser deciding this is a moment to say nothing — and
        :attr:`chose_silence` tells the two apart.

        ``notes`` are the pack's clauses about the people on this form, and
        they are the only outside information this stage has ever been given.
        The prompt decides whether to show them — a goal is no moment for a
        statistic — and the fact gate checks whatever comes back against the
        same notes, so a figure the model adjusts on its way out is a line
        that never reaches the speaker.

        ``ledger`` is the same offer made out of this match's own counts —
        a fourth corner, a second foul — written by
        :class:`commentary.ledger.Ledger` rather than researched, and checked
        afterwards by the gate's ``ledger_claim`` exactly as a note is checked
        by ``note_claim``.

        ``callbacks`` marks which of those clauses have already been said
        once in this match, one flag a note, so the prompt can ask for a new
        form of an old fact rather than the same words again. It is
        :class:`commentary.threads.Threads` that knows, and empty is the
        answer everywhere nothing has been said twice yet.

        ``followup`` is the block :class:`commentary.goalfollow.GoalFollowup`
        writes in the thirty seconds after a goal, naming which of the corpus's
        beats is due — the moment again, the scorer's tally, the move rebuilt
        in past tense. Empty everywhere else, which is most of a match.

        ``goal_beat`` and ``scorer`` are what :class:`~commentary.goalfollow.
        GoalFollowup` already knows and ``followup`` only says in prose: which
        beat this call is under and who scored. Beat 3 is one number about the
        scorer, and the prompt already says "Name him" — about half the time
        the first answer does not, and comes back a fact about nobody, which
        ``note_claim`` in the gate refuses outright. So this stage checks its
        own beat-3 answers the same way it checks a repeated opener: once,
        same call, and whatever comes back is what goes out.

        ``roster`` is every name this match can legitimately carry, and it is
        used for one thing only: telling a line that opens by shouting a name
        from a line that opens "Save!" or "Buried!". Beats 2 and 3 may not be
        shouted on a name — see :data:`UNSHOUTED_BEATS` — and the check needs
        to know which capitalised words are people. Empty is safe; the scorer
        and the names on the form are checked either way.

        ``replay_first`` matters only on a replay form and is the one thing
        the model cannot see for itself: whether an earlier line in this same
        replay sequence has already named it as a replay. Whoever is counting
        the sequence — the runtime, or the rephrase — says so here. See
        :func:`commentary.prompts.phraser.replay_block`. When it is false, a
        leading "as we see it again" is taken off the answer in code: the
        model was told the sequence had already been named and named it again
        in two lines of three.
        """
        self.last_reason = ""
        self.chose_silence = False
        self.last_usage = Usage()
        blocks = phraser_blocks(
            line,
            state_summary,
            self.recent,
            home=self.home,
            away=self.away,
            on_the_ball=on_the_ball,
            notes=notes,
            callbacks=callbacks,
            ledger=ledger,
            last_event=self._recent[-1].event if self._recent else None,
            followup=followup,
            replay_first=replay_first,
            dead_ball=self.dead_ball,
        )
        try:
            parsed = await self.backend.parse(
                model=self.model,
                system=self.system,
                blocks=blocks,
                output_format=PhrasedLine,
                max_tokens=self.config.max_tokens,
                effort="low",
                cache_system=True,
                tag="phraser",
            )
        except LLMError as exc:
            self.last_reason = f"model call failed: {exc}"
            return None

        usage = parsed.usage
        proposed = parsed.value

        # The register repeat, both ends of the sentence, one re-ask. Asked
        # once: whatever comes back — even the same word again — is what goes
        # out, and a second re-ask is never made.
        opener = self._repeated_opener(proposed)
        closer = self._repeated_closer(proposed)
        opener_retry = False
        closer_retry = False
        if opener is not None or closer is not None:
            retry = await self._reask(blocks, _register_retry_note(opener or "", closer or ""))
            if retry is not None:
                usage = usage + retry.usage
                proposed = retry.value
                opener_retry = opener is not None
                closer_retry = closer is not None

        name_retry = False
        if goal_beat == SCORER_BEAT and scorer and self._needs_a_name(proposed, scorer):
            retry = await self._reask(blocks, _name_retry_note(scorer))
            if retry is not None:
                usage = usage + retry.usage
                proposed = retry.value
                name_retry = True
                # Asked once. Whatever came back — even nameless again — is
                # what goes out; the gate is the backstop from here.

        # The goal call's own shape, on a line that is not the call. Unlike
        # every other check here this one does not stop at a re-ask: a beat
        # that comes back shouting the name again is rewritten rather than
        # dropped, because a dropped beat is a hole in the thirty seconds
        # after a goal and the shout is the one thing wrong with the line.
        shout_retry = False
        shout_rewritten = False
        if goal_beat in UNSHOUTED_BEATS:
            names = self._names_here(line, scorer, roster)
            shouted = opening_shout(proposed.line, names)
            if shouted is not None:
                retry = await self._reask(blocks, _shout_retry_note(shouted, goal_beat))
                if retry is not None:
                    usage = usage + retry.usage
                    proposed = retry.value
                    shout_retry = True
                if opening_shout(proposed.line, names) is not None:
                    proposed = proposed.model_copy(update={"line": unshout(proposed.line)})
                    shout_rewritten = True

        # The replay is named once a sequence, and this is the line after
        # the one that named it.
        marker = ""
        if line.scene is Scene.REPLAY and not replay_first:
            without, marker = strip_replay_marker(proposed.line)
            if marker:
                proposed = proposed.model_copy(update={"line": without})

        self.last_usage = usage
        settled = self._settle(proposed, self._word_cap(line, goal_beat))
        return settled.model_copy(
            update={
                "opener_retry": opener_retry,
                "closer_retry": closer_retry,
                "name_retry": name_retry,
                "shout_retry": shout_retry,
                "shout_rewritten": shout_rewritten,
                "replay_marker_stripped": marker[:64],
            }
        )

    async def _reask(self, blocks: Sequence[Block], note: str) -> Parsed[PhrasedLine] | None:
        """The same call again with a note on the end of it, or ``None``.

        ``None`` is the re-ask itself failing to come back, and the answer
        already in hand is kept rather than the line lost over it. Every
        check in :meth:`phrase` re-asks exactly once and through here, so
        one line can never cost more than one extra call per fault.
        """
        try:
            return await self.backend.parse(
                model=self.model,
                system=self.system,
                blocks=_with_note(blocks, note),
                output_format=PhrasedLine,
                max_tokens=self.config.max_tokens,
                effort="low",
                cache_system=True,
                tag="phraser",
            )
        except LLMError:
            return None

    @staticmethod
    def _names_here(
        line: CallerLine, scorer: str | None, roster: Sequence[str]
    ) -> list[str]:
        """Everybody this call could legitimately be shouting at."""
        names = [scorer or "", *((s.name or "") for s in line.sightings), *roster]
        return [name for name in names if name.strip()]

    def _word_cap(self, line: CallerLine, goal_beat: int | None) -> int:
        """The hard trim for this moment, which is not one number any more.

        28 is the backstop everywhere the ball is live: study section 1 puts
        club football's 95th percentile at 22-27 words. At a restart and in
        the thirty seconds after a goal the corpus goes past it — section 2.3
        has one restart line in five over sixteen words, section 2.4 has 60
        words in 30 seconds — and a line asked for at 12 to 22 words that is
        trimmed at 28 has the trim deciding how it ends. So those kinds get
        :attr:`~commentary.config.DeadBallConfig.max_words` instead, and the
        trim is still there.
        """
        if is_long_line(line, followup_beat=goal_beat):
            return max(self.config.max_words, self.dead_ball.max_words)
        return self.config.max_words

    @staticmethod
    def _needs_a_name(proposed: PhrasedLine, scorer: str) -> bool:
        """A number with nobody's name on it — beat 3's own failure mode.

        Checked against the model's raw answer, not the trimmed one: cleaning
        never adds or removes a name, so there is nothing the settle step
        could change this by, and checking the raw answer means the retry
        fires on exactly what the gate is about to see.
        """
        text = proposed.line.strip()
        if not text or not says_a_number(text):
            return False
        return not mentions(text, scorer)

    def _repeated_opener(self, proposed: PhrasedLine) -> str | None:
        """The offending opener word if this line needs a re-ask, else ``None``.

        Stacked repetition is how a goal sounds (excitement >= 0.9), and a
        bare surname is a legitimate repeat rather than a repeated frame, so
        both are waved through without a retry.
        """
        if proposed.excitement >= 0.9:
            return None
        text = proposed.line.strip()
        if not text or _is_bare_name(text):
            return None
        word = _opening_word(text)
        if not word:
            return None
        recent = list(self._recent)[-_OPENER_LOOKBACK:]
        if any(_opening_word(said.text) == word for said in recent):
            return _display_word(text)
        return None

    def _repeated_closer(self, proposed: PhrasedLine) -> str | None:
        """The offending closing word if this line needs a re-ask, else ``None``.

        The same rule as :meth:`_repeated_opener` at the other end of the
        sentence, and the same two exemptions for the same reasons: a goal is
        shouted in repeated fragments, and a bare surname is a whole line
        that happens to be one word.

        What it is for is the tail this voice actually writes. On
        ``runs/rephrased/r1-replay/mbappe``: "Through the middle at speed." /
        "Through midfield now." / "Wide on the right now." / "Into the corner
        now." / "Striding out now, France in no hurry to move it on." Four of
        the five end on "now" and no two of them open on the same word, so
        the opener check passed every one. The register judge counts a closer
        repeat the same way it counts an opener repeat.
        """
        if proposed.excitement >= 0.9:
            return None
        text = proposed.line.strip()
        if not text or _is_bare_name(text):
            return None
        word = _closing_word(text)
        if not word:
            return None
        recent = list(self._recent)[-_CLOSER_LOOKBACK:]
        if any(_closing_word(said.text) == word for said in recent):
            return _display_closer(text)
        return None

    def _settle(self, proposed: PhrasedLine, max_words: int) -> PhrasedLine:
        """The same two post-conditions the caller applies, for the same reasons.

        A "Commentary:" label or a pair of quotation marks is harmless on a
        page and ruinous through a speech synthesiser, and a model that
        ignores a word cap must not be able to hold the voice channel while
        the next chance goes past.

        ``max_words`` is the moment's cap rather than the config's, because
        the cap is per kind now: see :meth:`_word_cap`.
        """
        text = trim_words(clean_line(proposed.line), max_words)
        if not text:
            # An empty answer is a choice; an answer that was only a label
            # or a pair of quotation marks is a failed one. The difference
            # decides whether the moment passes in silence or the caller's
            # own line goes out, so it is drawn on what the model actually
            # returned rather than on what survived the cleaning.
            self.chose_silence = not proposed.line.strip()
            self.last_reason = (
                "the phraser chose silence"
                if self.chose_silence
                else "the phraser wrote nothing sayable"
            )
        return proposed.model_copy(update={"line": text})
