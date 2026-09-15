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

from collections import deque
from collections.abc import Sequence

from commentary.agents.caller import clean_line, trim_words
from commentary.config import PHRASER_MODEL, PhraserConfig
from commentary.llm.base import LLMBackend, LLMError, Usage
from commentary.prompts.phraser import phraser_blocks, phraser_system
from commentary.schemas import CallerLine, Note, PhrasedLine

#: The value of ``PHRASER_MODEL`` that means "do not run this stage".
OFF = "off"


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
        self._recent: deque[str] = deque(maxlen=max(1, self.config.recent_lines))
        #: Why the last call produced nothing, in words, for the error row.
        self.last_reason = ""
        #: What the last call cost, so a rephrase can price itself per line.
        self.last_usage = Usage()

    @property
    def enabled(self) -> bool:
        return phraser_enabled(self.model)

    @property
    def recent(self) -> list[str]:
        """What has actually been said, oldest first."""
        return list(self._recent)

    def accept(self, line: str) -> None:
        """Record a line as spoken. Only call this when it really is going out."""
        text = line.strip()
        if text:
            self._recent.append(text)

    async def phrase(
        self,
        line: CallerLine,
        state_summary: str,
        *,
        on_the_ball: str | None = None,
        notes: Sequence[Note] = (),
    ) -> PhrasedLine | None:
        """Rewrite one caller line, or return ``None`` if the call failed.

        ``None`` and an empty line mean the same thing to the runtime, which
        falls back to the caller's own words either way rather than losing a
        line the gate was about to pass. They are kept apart here so the
        error row can say which happened.

        ``notes`` are the pack's clauses about the people on this form, and
        they are the only outside information this stage has ever been given.
        The prompt decides whether to show them — a goal is no moment for a
        statistic — and the fact gate checks whatever comes back against the
        same notes, so a figure the model adjusts on its way out is a line
        that never reaches the speaker.
        """
        self.last_reason = ""
        self.last_usage = Usage()
        blocks = phraser_blocks(
            line,
            state_summary,
            self.recent,
            home=self.home,
            away=self.away,
            on_the_ball=on_the_ball,
            notes=notes,
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

        self.last_usage = parsed.usage
        return self._settle(parsed.value)

    def _settle(self, proposed: PhrasedLine) -> PhrasedLine:
        """The same two post-conditions the caller applies, for the same reasons.

        A "Commentary:" label or a pair of quotation marks is harmless on a
        page and ruinous through a speech synthesiser, and a model that
        ignores a word cap must not be able to hold the voice channel while
        the next chance goes past.
        """
        text = trim_words(clean_line(proposed.line), self.config.max_words)
        if not text:
            self.last_reason = "the phraser wrote nothing"
        return proposed.model_copy(update={"line": text})
