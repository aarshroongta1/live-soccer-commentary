"""What the human commentators actually said, for comparison only.

The feed says what happened; it says nothing about how commentary should
sound. For that the recording carries its own answer — two professionals
calling the same pictures this system is calling — and transcribing them
gives the pairwise judge something to compare against that is not another
language model's idea of good commentary.

Transcription is local, with ``mlx-whisper``, which matters for two reasons.
The broadcast audio is not ours to upload, and this is the only part of the
eval that would otherwise send match content anywhere. It is also an
optional extra: the runtime never needs it, and a machine without it should
say so in a sentence rather than in an ImportError traceback.

Ninety minutes is a slow transcription, so the result is saved and reloaded.
Segment times share the capture's clock with ``video_ts`` everywhere else,
because they come from the same recording.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from commentary.schemas import KnowledgePack

#: Whisper large-v3-turbo, as the plan specifies: near large-v3 accuracy on
#: broadcast speech at a fraction of the time on Apple silicon.
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

#: Whisper truncates its prompt at 224 tokens and quietly ignores the rest,
#: so the roster hint is cut well short of that rather than half-fed.
PROMPT_LIMIT = 600


class TranscriptionUnavailable(RuntimeError):
    """``mlx-whisper`` is not installed, or will not run on this machine."""


@dataclass(frozen=True)
class Segment:
    """One utterance, on the same clock as everything else in a run."""

    start: float
    end: float
    text: str

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass
class Transcript:
    """A whole broadcast's commentary, timestamped."""

    segments: list[Segment] = field(default_factory=list)
    source: str = ""
    model: str = DEFAULT_MODEL
    language: str = "en"

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments).strip()

    @property
    def duration_s(self) -> float:
        return max((s.end for s in self.segments), default=0.0)

    def __len__(self) -> int:
        return len(self.segments)


def roster_prompt(roster: KnowledgePack | Sequence[str] | None) -> str:
    """The names, as a hint prompt, so Whisper spells the players right.

    Left to itself Whisper renders ``Ødegaard`` half a dozen ways and the
    pairwise judge then compares our line against a transcript that has the
    wrong surname in it. Priming with the squad fixes most of that for the
    price of one string.
    """
    if roster is None:
        return ""
    if isinstance(roster, KnowledgePack):
        names = [p.name for sheet in (roster.home, roster.away) for p in sheet.squad]
        teams = [s.name for s in (roster.home, roster.away) if s.name]
        parts = []
        if teams:
            parts.append(" versus ".join(teams) + ".")
        if names:
            parts.append("Players: " + ", ".join(names) + ".")
        hint = " ".join(parts)
    else:
        hint = "Players: " + ", ".join(roster) + "."
    return hint[:PROMPT_LIMIT].rstrip(", ")


def transcribe(
    path: Path,
    *,
    roster: KnowledgePack | Sequence[str] | None = None,
    model: str = DEFAULT_MODEL,
    language: str = "en",
) -> Transcript:
    """Transcribe captured broadcast audio locally.

    Imported lazily: ``mlx-whisper`` is an Apple-silicon-only extra that CI
    and most contributors will not have, and nothing else in this module
    needs it — a saved transcript loads fine without it.
    """
    try:
        import mlx_whisper
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise TranscriptionUnavailable(
            "mlx-whisper is not installed, so broadcast audio cannot be transcribed here. "
            "Install it with `uv sync --extra eval` on Apple silicon, or load a transcript "
            "somebody already saved with transcripts.load()."
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"no audio at {path}")

    result: Any = mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=model,
        initial_prompt=roster_prompt(roster) or None,
        language=language,
    )
    return Transcript(
        segments=_segments(result.get("segments", [])),
        source=str(path),
        model=model,
        language=str(result.get("language", language)),
    )


# -- on disk -----------------------------------------------------------


def save(transcript: Transcript, path: Path) -> None:
    """Write a transcript as JSON::

    {"source": "...", "model": "...", "language": "en",
     "segments": [{"start": 12.4, "end": 15.1, "text": "..."}]}
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "source": transcript.source,
        "model": transcript.model,
        "language": transcript.language,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text} for s in transcript.segments
        ],
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")


def load(path: Path) -> Transcript:
    """Read one back, without needing ``mlx-whisper`` present."""
    document = json.loads(path.read_text(encoding="utf-8"))
    return Transcript(
        segments=_segments(document.get("segments", [])),
        source=str(document.get("source", "")),
        model=str(document.get("model", DEFAULT_MODEL)),
        language=str(document.get("language", "en")),
    )


def _segments(rows: Any) -> list[Segment]:
    out: list[Segment] = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        out.append(Segment(start=float(row["start"]), end=float(row["end"]), text=text))
    out.sort(key=lambda s: s.start)
    return out


# -- what was said around a moment -------------------------------------


def segments_near(transcript: Transcript, video_ts: float, window_s: float = 5.0) -> list[Segment]:
    """Every segment that overlaps the window around ``video_ts``.

    Overlap rather than containment: a commentator's sentence about a goal
    routinely starts before the moment and runs six seconds past it, and a
    containment test would return nothing for exactly the events that matter
    most.
    """
    low, high = video_ts - window_s, video_ts + window_s
    return [s for s in transcript.segments if s.end >= low and s.start <= high]


def said_near(transcript: Transcript, video_ts: float, window_s: float = 5.0) -> str:
    """The same thing as one string, which is what the judge is handed."""
    return " ".join(s.text for s in segments_near(transcript, video_ts, window_s)).strip()
