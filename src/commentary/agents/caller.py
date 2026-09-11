"""The play-by-play voice: one vision call, then the rules it has to obey.

The model is asked for a form, not a sentence, and what comes back is treated
as a proposal. Everything after the call is this module deciding whether the
proposal is fit to say out loud: short enough, confident enough, not a replay,
not something the same voice said twelve seconds ago. A model will do all four
of those wrong at some point in ninety minutes, and none of them should reach
a speaker.

Suppression is recorded rather than hidden. A line held back for repetition is
a fact about the system worth counting — the eval reports it — so the line
still comes back with ``speak`` false and the reason on the agent, instead of
disappearing into a ``return None``.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter, deque

from commentary.capture.buffer import DelayBuffer
from commentary.config import CALLER_MODEL, CallerConfig
from commentary.llm.base import LLMBackend, LLMError
from commentary.prompts.caller import caller_blocks, caller_system
from commentary.schemas import CallerLine, KnowledgePack, Scene, Trigger

#: Openers a model reaches for when it forgets it is talking, not writing.
_PREAMBLE = re.compile(
    r"^\s*(?:commentary|caller|line|play[- ]by[- ]play|pbp|commentator)\s*[:\-–—]\s*",
    re.IGNORECASE,
)
_QUOTES = "\"'“”‘’`"
_DANGLING = " ,;:-–—"

#: Words that carry no information about what happened, so two lines that
#: differ only in these are the same line.
_STOPWORD_TEXT = """
a an the and or but so as at by for from in into of off on onto out over to up down
with without across against along around through under is are was were be been being
it its he she they him her them his their this that these those there here now
just still again very really quite got get gets has have had do does did no not
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def similarity(left: str, right: str) -> float:
    """How close two lines are, 0 to 1.

    The failure mode that matters is not the verbatim repeat — those are rare
    — but the near miss: "Arsenal push forward down the left" arriving twelve
    seconds after "Arsenal pushing forward on the left". Word overlap catches
    that once tense and prepositions are stripped out, so it carries most of
    the weight. A character ratio is mixed in underneath it to catch the
    reshuffles that survive a bag of words, but only at a third of the weight,
    because on its own it flags any two lines with the same shape — and
    "Arsenal break down the right" and "Chelsea break down the left" are two
    different things happening.
    """
    left_tokens, right_tokens = _content(left), _content(right)
    if left_tokens and right_tokens:
        overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    else:
        overlap = 0.0
    ratio = difflib.SequenceMatcher(None, _flatten(left), _flatten(right)).ratio()
    if not left_tokens or not right_tokens:
        return ratio
    return 0.7 * overlap + 0.3 * ratio


def _flatten(text: str) -> str:
    """Lower case, letters and digits only. Punctuation is not a difference."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _content(text: str) -> frozenset[str]:
    """The words that carry the meaning, stemmed so tense does not hide a repeat."""
    return frozenset(_stem(word) for word in _flatten(text).split() if word not in _STOPWORDS)


def _stem(word: str) -> str:
    """A crude suffix trim. Enough for push/pushing/pushes, and nothing clever.

    A real stemmer would be a dependency and a thing to tune; all that is
    needed here is that a commentator's two tenses of the same verb collapse
    onto one token.
    """
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]
    return word


def clean_line(text: str) -> str:
    """Strip the wrapper a model sometimes puts around the sentence.

    Quotation marks and a "Commentary:" label are harmless on a page and
    ruinous through a speech synthesiser, which reads them as pauses or, worse,
    out loud.
    """
    cleaned = text.strip()
    if cleaned and cleaned[0] in _QUOTES:
        cleaned = cleaned[1:]
    if cleaned and cleaned[-1] in _QUOTES:
        cleaned = cleaned[:-1]
    cleaned = _PREAMBLE.sub("", cleaned).strip()
    if cleaned and cleaned[0] in _QUOTES:
        cleaned = cleaned[1:]
    return cleaned.strip()


def trim_words(text: str, max_words: int) -> str:
    """Hard cap on length. A blunt instrument, and deliberately the last one.

    The prompt is what should keep lines short; this only exists so that a
    model that ignores it cannot hold the voice channel for nine seconds while
    the next chance goes begging.
    """
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).rstrip(_DANGLING)


class RepetitionGate:
    """Remembers the last few spoken lines and refuses anything too close.

    Inherited from worldcupvoice, where it is the single thing that stops a
    fixed-cadence caller from saying the same sentence about the same passage
    of play four times in a row. Kept explicit here because the eval counts how
    often it fires: a high rate means the caller is being asked to speak when
    nothing has changed, which is a speak-predictor problem, not a prompt one.
    """

    def __init__(self, config: CallerConfig | None = None) -> None:
        cfg = config or CallerConfig()
        self.threshold = cfg.repetition_threshold
        self._recent: deque[str] = deque(maxlen=max(1, cfg.recent_lines))

    @property
    def recent(self) -> list[str]:
        """The remembered lines, oldest first. This is what the prompt shows."""
        return list(self._recent)

    def judge(self, line: str) -> tuple[bool, float]:
        """``(may_speak, closest match)`` against everything remembered.

        The score comes back either way so that a rejection can say how close
        it was, and so a threshold sweep has something to sweep over.
        """
        text = line.strip()
        if not text:
            return False, 0.0
        worst = max((similarity(text, prev) for prev in self._recent), default=0.0)
        return worst < self.threshold, worst

    def accept(self, line: str) -> None:
        """Record a line as spoken. Only call this when it really is going out."""
        text = line.strip()
        if text:
            self._recent.append(text)

    def reset(self) -> None:
        """Half time, or a new match. Nothing said yet."""
        self._recent.clear()


class Caller:
    """Frames in, one short sentence out — or, more often, nothing.

    Holds its own :class:`RepetitionGate`, which doubles as the memory of what
    has been said: the same deque feeds the prompt's "last lines spoken" and
    the similarity check, so the model and the gate can never disagree about
    what the voice has already covered.

    One wrinkle worth knowing: a line is recorded as spoken the moment this
    agent approves it, which is before the fact gate downstream has had its
    say. A line the fact gate then kills still occupies a slot in the gate's
    memory. That is the cheap direction to be wrong in — it suppresses a near
    repeat of something never said, rather than letting a real repeat through
    — but if it starts mattering, hand ``gate.accept`` to the director instead.
    """

    def __init__(
        self,
        backend: LLMBackend,
        config: CallerConfig | None = None,
        pack: KnowledgePack | None = None,
        *,
        model: str = CALLER_MODEL,
        max_tokens: int = 512,
        effort: str | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or CallerConfig()
        self.pack = pack
        self.model = model
        #: Small on purpose. The answer is a form and one sentence; the only
        #: reason it is not smaller is that adaptive thinking spends from the
        #: same budget, and a truncated response is a dropped line.
        self.max_tokens = max_tokens
        self.effort = effort
        #: Built once, never rebuilt. Identical bytes on every call is what
        #: makes the cache hit that pays for sending two squads each time.
        self.system = caller_system(self.pack, self.config)
        self.gate = RepetitionGate(self.config)
        #: Why the last call did not produce speech, in words, for the log.
        self.last_reason = ""
        #: How close the last candidate came to a recent line.
        self.last_similarity = 0.0
        #: Suppressions by cause, for the eval's silence breakdown.
        self.suppressed: Counter[str] = Counter()

    async def call(
        self,
        buffer: DelayBuffer,
        state_summary: str,
        triggers: list[Trigger],
    ) -> CallerLine | None:
        """Look at the cursor and the near future, and decide whether to speak.

        Returns ``None`` only when there was nothing to look at or the model
        call failed. A dropped call is a missed line, not a crashed match, so
        the error is swallowed here and left on ``last_reason`` — the director
        will be back in four seconds either way.
        """
        cursor = buffer.at_cursor(self.config.frames_at_cursor, self.config.cursor_spacing_s)
        if not cursor:
            self.last_reason = "no frames at the cursor"
            self.suppressed["no_frames"] += 1
            return None
        lookahead = buffer.lookahead(self.config.frames_lookahead)
        blocks = caller_blocks(cursor, lookahead, state_summary, self.gate.recent, triggers)

        try:
            parsed = await self.backend.parse(
                model=self.model,
                system=self.system,
                blocks=blocks,
                output_format=CallerLine,
                max_tokens=self.max_tokens,
                effort=self.effort,
                cache_system=True,
                tag="caller",
            )
        except LLMError as exc:
            self.last_reason = f"model call failed: {exc}"
            self.suppressed["llm_error"] += 1
            return None

        return self._settle(parsed.value)

    def _settle(self, proposed: CallerLine) -> CallerLine:
        """Apply the post-conditions and record why, before anyone sees the line."""
        text = trim_words(clean_line(proposed.line), self.config.max_words)
        self.last_similarity = 0.0

        if not proposed.speak:
            self.last_reason = "the model chose silence"
            return proposed.model_copy(update={"line": text, "speak": False})

        code, reason, score = self._veto(proposed, text)
        self.last_similarity = score
        if code:
            self.last_reason = reason
            self.suppressed[code] += 1
            return proposed.model_copy(update={"line": text, "speak": False})

        self.last_reason = ""
        self.gate.accept(text)
        return proposed.model_copy(update={"line": text, "speak": True})

    def _veto(self, proposed: CallerLine, text: str) -> tuple[str, str, float]:
        """``(code, reason, similarity)``; an empty code means the line may go."""
        if not text:
            return "empty", "the model set speak but wrote nothing", 0.0
        if proposed.scene is Scene.REPLAY:
            return "replay", "the scene is a replay, which is never called as live", 0.0
        if proposed.confidence < self.config.min_confidence:
            return (
                "low_confidence",
                f"confidence {proposed.confidence:.2f} is under {self.config.min_confidence:.2f}",
                0.0,
            )
        allowed, score = self.gate.judge(text)
        if not allowed:
            return "repetition", f"too close to a recent line ({score:.2f})", score
        return "", "", score
