"""macOS's built-in ``say``, for testing sound without spending a key.

``ElevenLabsSpeaker`` is the real voice: two distinct, well-acted voices,
streamed so a goal can cut them off inside a word. That is worth a key. But
proving the director's preemption and pacing actually sound right does not
need acting quality, and paying ElevenLabs to check whether cancellation
still works is a bad trade. ``say`` is free, already on every Mac, and — like
the real speaker — a subprocess that can be killed rather than asked
politely, so the same mid-word cut this project cares about is exercisable
here for nothing.

There is no streaming here: ``say`` renders and speaks in one step, so a cut
lands wherever the process happens to be when it is killed, not at a chunk
boundary. That is coarser than ElevenLabs but exactly as coarse as
``LogSpeaker``, which this exists alongside as a step up from — text plus
actual sound, still free.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from dataclasses import dataclass, field

from commentary.config import VoiceConfig
from commentary.schemas import Beat, Voice
from commentary.voice.elevenlabs import VoiceUnavailable
from commentary.voice.speaker import WORDS_PER_SECOND, Utterance

#: Stock macOS voices, picked the same way the ElevenLabs pair was: different
#: enough to tell apart instantly. Both overridable from the environment.
CALLER_VOICE = "Daniel"
ANALYST_VOICE = "Samantha"

#: ``say``'s own unit is words per minute; 190 is the default it ships with
#: and lands close to the 3.2 words/second the rest of the project assumes.
#: It is now the middle of a range rather than the whole story — see
#: :attr:`~commentary.config.VoiceConfig.say_rate_low` — and is what a line
#: gets when ``SAY_RATE`` pins the rate or ``VOICE_CURVE=off``.
RATE_WPM = 190


@dataclass
class SaySpeaker:
    """Speaks through macOS's ``say`` binary. Cancellable, but not streamed.

    ``executable`` exists so tests can point this at a fake script instead of
    the real ``say`` — the same seam :class:`~commentary.voice.elevenlabs.ElevenLabsSpeaker`
    gives the stream and the sink.
    """

    caller_voice: str = ""
    analyst_voice: str = ""
    #: Pins the rate and ignores the excitement entirely. Zero means the
    #: curve decides, which is the default; ``SAY_RATE`` sets it too.
    rate: int = 0
    executable: str = ""
    words_per_second: float = WORDS_PER_SECOND
    curve: VoiceConfig | None = None
    said: list[Utterance] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.caller_voice = self.caller_voice or os.getenv("SAY_CALLER_VOICE") or CALLER_VOICE
        self.analyst_voice = self.analyst_voice or os.getenv("SAY_ANALYST_VOICE") or ANALYST_VOICE
        self.rate = self.rate or int(os.getenv("SAY_RATE") or 0)
        if self.curve is None:
            self.curve = VoiceConfig()
        found = shutil.which(self.executable or "say")
        if found is None:
            raise VoiceUnavailable(
                "no `say` on PATH: this speaker is macOS only, run with --voice log"
            )
        self.executable = found

    def voice_name(self, voice: Voice) -> str:
        return self.analyst_voice if voice is Voice.ANALYST else self.caller_voice

    def rate_for(self, excitement: float) -> int:
        """Words per minute for one beat.

        ``say`` has one dial and it is speed, so the whole of the excitement
        curve arrives here as a number between roughly 170 and 210. It is not
        the delivery ElevenLabs gives — nothing here breaks pitch — but a
        goal called faster than the build-up is audibly the right shape, and
        it costs nothing to check the plumbing with.
        """
        curve = self.curve
        if self.rate or curve is None or not curve.on:
            return self.rate or RATE_WPM
        return curve.say_rate(excitement)

    async def say(self, beat: Beat, cancel: asyncio.Event) -> Utterance:
        started = time.monotonic()
        rate = self.rate_for(beat.excitement)
        proc = await asyncio.create_subprocess_exec(
            self.executable,
            "-v",
            self.voice_name(beat.voice),
            "-r",
            str(rate),
            beat.text,
        )
        cancel_wait = asyncio.ensure_future(cancel.wait())
        proc_wait = asyncio.ensure_future(proc.wait())
        try:
            done, _ = await asyncio.wait(
                {cancel_wait, proc_wait}, return_when=asyncio.FIRST_COMPLETED
            )
            completed = proc_wait in done and not cancel.is_set()
            if not completed:
                await _kill(proc)
        finally:
            cancel_wait.cancel()
            if not proc_wait.done():
                proc_wait.cancel()

        elapsed = time.monotonic() - started
        utterance = Utterance(
            beat=beat,
            spoken=beat.text if completed else self._estimate(beat.text, elapsed),
            seconds=elapsed,
            completed=completed,
            # ``say`` renders and plays behind its own back: nothing here can
            # see the moment sound starts, and guessing at it would put a
            # made-up number next to two measured ones.
            first_audio_s=None,
            # One dial, so one number — but recorded in the same place as
            # ElevenLabs' six, so the trace answers "how was this said?" in
            # the same way whichever voice said it.
            voice_settings={"rate_wpm": rate},
        )
        self.said.append(utterance)
        return utterance

    def _estimate(self, text: str, elapsed: float) -> str:
        words = text.split()
        heard = round(elapsed * self.words_per_second)
        return " ".join(words[: max(0, heard)])

    async def aclose(self) -> None:
        return None


async def _kill(proc: asyncio.subprocess.Process) -> None:
    """Stop speech mid-word: ask first, then insist.

    ``say`` does not react to being asked over stdin the way a media player
    might, so terminate/kill is the whole story — the same treatment
    ElevenLabs' player gets, for the same reason: a cancelled line is not a
    request, it is a fact.
    """
    if proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=0.5)
    except TimeoutError:
        proc.kill()
        await proc.wait()


__all__ = ["ANALYST_VOICE", "CALLER_VOICE", "RATE_WPM", "SaySpeaker"]
