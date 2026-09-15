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

from commentary.agents.caller import clean_line, trim_words
from commentary.agents.colour import mentions, says_a_number
from commentary.config import PHRASER_MODEL, PhraserConfig
from commentary.ledger import Fact as LedgerFact
from commentary.llm.base import Block, LLMBackend, LLMError, Usage, text_block
from commentary.prompts.phraser import phraser_blocks, phraser_system
from commentary.schemas import CallerLine, Event, Note, PhrasedLine

#: The beat that is one number about the scorer — see ``GOAL_BEATS[3]`` in
#: ``prompts/phraser.py``. The only beat this module ever needs to
#: distinguish, because it is the only one where a name is mandatory and a
#: number is expected in the same line.
SCORER_BEAT = 3

#: The value of ``PHRASER_MODEL`` that means "do not run this stage".
OFF = "off"

#: How many of the most recently spoken lines an opener is checked against.
#: Matches the prompt's own "last two" rule with slack: the prompt asks the
#: model not to open on either of the last two, and real commentary repeats
#: an opener about one line in eight looking back further than that, so
#: checking five catches the failure (35-48% repeats) without punishing the
#: normal, occasional one-in-eight echo from further back.
_OPENER_LOOKBACK = 5

_LEADING_TRAILING_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")
_POSSESSIVE = re.compile(r"['’]s$", re.IGNORECASE)


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


def _display_word(text: str) -> str:
    """The same word, kept in its written case, for naming in the retry note."""
    tokens = text.strip().split()
    if not tokens:
        return ""
    core = _LEADING_TRAILING_PUNCT.sub("", tokens[0])
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


def _opener_retry_note(word: str) -> str:
    """Two lines: name the repeated opener, then say what to do about it."""
    return (
        f'THAT OPENED ON "{word}" AGAIN, which one of the last five lines already used.\n'
        "Open differently this time, or return an empty line instead."
    )


def _name_retry_note(scorer: str) -> str:
    """Two lines: say the number reached nobody, then say who it is about."""
    return (
        "THAT HAD A NUMBER IN IT AND NOBODY'S NAME ON IT — a fact about nobody, which the "
        f"gate refuses. The line must name {scorer}.\n"
        "Keep the number and the fact; add his name, in the same number of words."
    )


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
    ) -> None:
        self.backend = backend
        self.config = config or PhraserConfig()
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
        self._recent: deque[tuple[str, Event | None]] = deque(
            maxlen=max(1, self.config.recent_lines)
        )
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
            f"{text}   ({event.value if event else 'no kind'})" for text, event in self._recent
        ]

    def accept(self, line: str, event: Event | None = None) -> None:
        """Record a line as spoken. Only call this when it really is going out."""
        text = line.strip()
        if text:
            self._recent.append((text, event))

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

        ``replay_first`` matters only on a replay form and is the one thing
        the model cannot see for itself: whether an earlier line in this same
        replay sequence has already named it as a replay. Whoever is counting
        the sequence — the runtime, or the rephrase — says so here. See
        :func:`commentary.prompts.phraser.replay_block`.
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
            last_event=self._recent[-1][1] if self._recent else None,
            followup=followup,
            replay_first=replay_first,
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
        opener_retry = False
        offending = self._repeated_opener(proposed)
        if offending is not None:
            retry_blocks = _with_note(blocks, _opener_retry_note(offending))
            try:
                retry = await self.backend.parse(
                    model=self.model,
                    system=self.system,
                    blocks=retry_blocks,
                    output_format=PhrasedLine,
                    max_tokens=self.config.max_tokens,
                    effort="low",
                    cache_system=True,
                    tag="phraser",
                )
            except LLMError:
                # The re-ask itself failed to come back — keep the first
                # attempt rather than lose the line over it.
                pass
            else:
                usage = usage + retry.usage
                proposed = retry.value
                opener_retry = True
                # Asked once. Whatever came back — even the same opener
                # again — is what goes out; a second re-ask is not made.

        name_retry = False
        if goal_beat == SCORER_BEAT and scorer and self._needs_a_name(proposed, scorer):
            retry_blocks = _with_note(blocks, _name_retry_note(scorer))
            try:
                retry = await self.backend.parse(
                    model=self.model,
                    system=self.system,
                    blocks=retry_blocks,
                    output_format=PhrasedLine,
                    max_tokens=self.config.max_tokens,
                    effort="low",
                    cache_system=True,
                    tag="phraser",
                )
            except LLMError:
                # The re-ask itself failed to come back — keep whatever the
                # first attempt was rather than lose the line over it.
                pass
            else:
                usage = usage + retry.usage
                proposed = retry.value
                name_retry = True
                # Asked once. Whatever came back — even nameless again — is
                # what goes out; the gate is the backstop from here.

        self.last_usage = usage
        settled = self._settle(proposed)
        return settled.model_copy(update={"opener_retry": opener_retry, "name_retry": name_retry})

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
        if any(_opening_word(spoken) == word for spoken, _event in recent):
            return _display_word(text)
        return None

    def _settle(self, proposed: PhrasedLine) -> PhrasedLine:
        """The same two post-conditions the caller applies, for the same reasons.

        A "Commentary:" label or a pair of quotation marks is harmless on a
        page and ruinous through a speech synthesiser, and a model that
        ignores a word cap must not be able to hold the voice channel while
        the next chance goes past.
        """
        text = trim_words(clean_line(proposed.line), self.config.max_words)
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
