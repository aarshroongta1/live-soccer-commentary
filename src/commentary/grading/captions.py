"""The broadcast's own captions, read as the human commentator's transcript.

yt-dlp writes these beside the video when asked for them, so on real footage
there is a transcript without a transcription: no model, no minutes of
Whisper, no audio leaving the machine. One flag on the download that already
had to happen.

What they are good for is the whole point. These captions are ground truth
for *timing and naturalness* — when a professional spoke, for how long, how
long the gaps between were, how a sentence is shaped — and never for facts.
Auto-captions mangle surnames constantly ("Di Maria" comes out four ways in
a half), and that costs nothing here, because no claim about the match is
ever read out of them. When this system's factual accuracy is measured it is
measured against the feed, which is a different file with a different job.

Two properties follow from where they come from:

* They are already on the video clock, stamped from the start of the file,
  so nothing has to be aligned. The feed counts in match time and needs
  :func:`commentary.grading.feed.align` to reach video time; captions do not.
* A gap in the captions is ambiguous. It means either that the commentator
  went quiet or that the ASR gave up in crowd noise, and nothing in the file
  distinguishes the two. So a silence ratio taken from captions is a ceiling
  on the real one, and the comparison in the results table has to be read
  that way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commentary.grading.transcripts import Segment, Transcript

#: What the model field says for a transcript nobody transcribed.
MODEL = "youtube-auto"


def load_json3(path: Path) -> Transcript:
    """Read a ``.en.json3`` caption file into a :class:`Transcript`.

    The shape yt-dlp writes is one ``events`` list, each event carrying
    ``tStartMs``, ``dDurationMs`` and a ``segs`` list whose ``utf8`` pieces
    join to the line. Events with no text — the window definitions, and the
    bare newlines that separate the rolling caption from the next one — are
    dropped rather than kept as empty segments that would read as speech.
    """
    document: Any = json.loads(path.read_text(encoding="utf-8"))
    segments: list[Segment] = []
    for event in document.get("events", []):
        text = "".join(str(seg.get("utf8", "")) for seg in event.get("segs", []))
        if not text.strip():
            continue
        start = float(event.get("tStartMs", 0)) / 1000.0
        segments.append(
            Segment(
                start=start,
                end=start + float(event.get("dDurationMs", 0)) / 1000.0,
                text=" ".join(text.split()),
            )
        )
    segments.sort(key=lambda s: s.start)
    return Transcript(segments=segments, source=str(path), model=MODEL)
