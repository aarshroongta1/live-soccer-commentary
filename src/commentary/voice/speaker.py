"""Turning a line into sound, in a way that can be stopped mid-word.

Every speaker here is cancellable at a granularity finer than a sentence.
That is not a nicety: the moment a goal goes in, whatever the analyst was
halfway through saying is wrong to keep saying, and a human producer would
cut the mic. A speaker that can only be cancelled between lines would make
the director's preemption meaningless.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Protocol

from commentary.schemas import Beat

#: Roughly how fast an excited play-by-play voice actually talks.
WORDS_PER_SECOND = 3.2


@dataclass(frozen=True)
class Utterance:
    """What actually came out, which is not always what was asked for."""

    beat: Beat
    spoken: str
    seconds: float
    completed: bool

    @property
    def cut_off(self) -> bool:
        return not self.completed


class Speaker(Protocol):
    """Anything that can say a beat out loud and stop when told."""

    async def say(self, beat: Beat, cancel: asyncio.Event) -> Utterance: ...

    async def aclose(self) -> None: ...


@dataclass
class LogSpeaker:
    """Speech without audio: paces a line at a human rate and can be cut.

    This is what the whole pipeline runs against in tests and in any run that
    does not ask for sound. Because it consumes wall-clock time at the same
    rate a real voice does, the director's gaps, queue depth, and preemption
    behave exactly as they will with sound.
    """

    words_per_second: float = WORDS_PER_SECOND
    echo: bool = False
    said: list[Utterance] = field(default_factory=list)

    async def say(self, beat: Beat, cancel: asyncio.Event) -> Utterance:
        started = time.monotonic()
        words = beat.text.split()
        out: list[str] = []
        per_word = 1.0 / self.words_per_second if self.words_per_second > 0 else 0.0
        for word in words:
            if cancel.is_set():
                break
            out.append(word)
            if per_word:
                try:
                    await asyncio.wait_for(cancel.wait(), timeout=per_word)
                    break
                except TimeoutError:
                    pass
        utterance = Utterance(
            beat=beat,
            spoken=" ".join(out),
            seconds=time.monotonic() - started,
            completed=len(out) == len(words) and not cancel.is_set(),
        )
        self.said.append(utterance)
        if self.echo:
            mark = "" if utterance.completed else " [cut]"
            print(f"[{beat.voice.value}] {utterance.spoken}{mark}", flush=True)
        return utterance

    async def aclose(self) -> None:
        return None
