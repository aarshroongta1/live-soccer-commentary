"""The same lines at different voice settings, side by side, in a browser.

The excitement curve in :class:`~commentary.config.VoiceConfig` is six
numbers a seat, and not one of them has been chosen by ear — they are a
plausible starting shape and nothing more. There is no way to pick them from
a spec sheet: "stability 0.2" means whatever it means to a listener, and it
means something different at speed 1.15 than at 1.0. So this renders a grid
of them over a handful of real lines, writes an ``index.html`` beside the
clips, and leaves somebody to sit and listen.

It spends money. Every clip is a separate synthesis, the grid multiplies, and
a careless run is a few thousand characters — which is why it prints the
character count and the credit estimate first and does nothing at all unless
``--yes`` says to. It is also deliberately not the streaming path the match
uses: this wants files on disk to compare, not the lowest possible latency,
so it is one plain REST call a clip.

    uv run python scripts/voice_sweep.py --voice-id JBFqnCBsd6RMkjVDRZzb --yes

Clips land in ``runs/voice/sweep/<timestamp>/``, named for the line and the
settings that made them, and the index puts every setting for one line in a
row so the comparison is one click apart.
"""

from __future__ import annotations

import argparse
import html
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from commentary.voice.elevenlabs import MODEL_ID

#: Non-streaming, because the point is a file to play twice.
API_ROOT = "https://api.elevenlabs.io/v1"

#: mp3 rather than the runtime's raw PCM: a browser plays it, and the sweep
#: is about delivery, not about the sink.
OUTPUT_FORMAT = "mp3_44100_128"

REQUEST_TIMEOUT_S = 60.0

#: Where the clips go. One directory a run, because a second run with a
#: different grid is a different comparison, not more of the same one.
OUT_ROOT = Path("runs/voice/sweep")


@dataclass(frozen=True)
class Line:
    """One thing to say, and how excited the phraser said it was."""

    slug: str
    text: str
    excitement: float


#: The five lines the default grid is rendered over. Four of them are the
#: rephrased Mbappé run, kept exactly as the phraser wrote them so the sweep
#: is judged on lines the system actually produces rather than on lines
#: written to flatter it; the fifth is an analyst line, which is the seat
#: whose settings are easiest to get wrong because nothing about it shouts.
DEFAULT_LINES = (
    Line("buildup", "France push forward down the left.", 0.2),
    Line("spot", "Mbappé on the spot.", 0.75),
    Line("goal", "Mbappé! Buried!", 1.0),
    Line("back", "Mbappé has the ball back.", 0.5),
    Line(
        "analyst",
        "That's the run Scaloni has wanted all night, "
        "and nobody in a blue shirt goes with him.",
        0.3,
    ),
)

#: The axes of the default grid, and the whole reason this script exists.
#: Three stabilities because that is the setting with the most audible range,
#: two styles and two speeds because past a couple of points each they stop
#: being distinguishable and start being a bill.
DEFAULT_STABILITIES = (0.20, 0.35, 0.55)
DEFAULT_STYLES = (0.15, 0.60)
DEFAULT_SPEEDS = (1.00, 1.15)

#: Held fixed across the grid, matching the curve's own: a voice that stops
#: sounding like itself is a different voice, not a setting.
SIMILARITY_BOOST = 0.8
USE_SPEAKER_BOOST = True


@dataclass(frozen=True)
class Setting:
    """One point in the grid."""

    stability: float
    style: float
    speed: float

    @property
    def slug(self) -> str:
        return f"st{self.stability:.2f}-sy{self.style:.2f}-sp{self.speed:.2f}".replace(".", "")

    @property
    def label(self) -> str:
        return f"stability {self.stability:.2f} / style {self.style:.2f} / speed {self.speed:.2f}"

    def as_settings(self) -> dict[str, float | bool]:
        return {
            "stability": self.stability,
            "style": self.style,
            "speed": self.speed,
            "similarity_boost": SIMILARITY_BOOST,
            "use_speaker_boost": USE_SPEAKER_BOOST,
        }


@dataclass(frozen=True)
class Clip:
    """One line, at one setting, in one voice: a single file to be rendered."""

    voice_id: str
    line: Line
    setting: Setting

    @property
    def filename(self) -> str:
        return f"{self.voice_id[:8]}-{self.line.slug}-{self.setting.slug}.mp3"

    def body(self, model_id: str) -> dict[str, object]:
        return {
            "text": self.line.text,
            "model_id": model_id,
            "voice_settings": self.setting.as_settings(),
        }


@dataclass(frozen=True)
class Estimate:
    """What a run would cost, before it costs it."""

    clips: int
    characters: int
    credits: float

    def __str__(self) -> str:
        return (
            f"{self.clips} clips, {self.characters} characters, "
            f"about {self.credits:.0f} credits"
        )


def grid(
    stabilities: Sequence[float], styles: Sequence[float], speeds: Sequence[float]
) -> list[Setting]:
    return [Setting(a, b, c) for a, b, c in product(stabilities, styles, speeds)]


def plan(
    voice_ids: Sequence[str], lines: Sequence[Line], settings: Sequence[Setting]
) -> list[Clip]:
    """Every clip the run would render, in the order it would render them."""
    return [
        Clip(voice_id=voice_id, line=line, setting=setting)
        for voice_id in voice_ids
        for line in lines
        for setting in settings
    ]


def credits_per_character(model_id: str) -> float:
    """Flash and Turbo bill at half a credit a character; the rest at one.

    A guess would be the wrong thing to print next to a spend, so anything
    unrecognised is charged at the higher rate: an estimate that comes in
    under is a nicer surprise than one that comes in over.
    """
    return 0.5 if "flash" in model_id or "turbo" in model_id else 1.0


def estimate(clips: Sequence[Clip], model_id: str) -> Estimate:
    characters = sum(len(clip.line.text) for clip in clips)
    return Estimate(
        clips=len(clips),
        characters=characters,
        credits=characters * credits_per_character(model_id),
    )


#: Given a voice id, a request body and a key, the mp3 bytes. Injected so the
#: tests can exercise the planning, the writing and the refusal without a key
#: and without spending anything.
Fetch = Callable[[str, dict[str, object], str], bytes]


def post_mp3(voice_id: str, body: dict[str, object], api_key: str) -> bytes:
    """One plain REST call. Imported inside, so the extra stays optional."""
    import httpx

    response = httpx.post(
        f"{API_ROOT}/text-to-speech/{voice_id}",
        params={"output_format": OUTPUT_FORMAT},
        headers={"xi-api-key": api_key, "content-type": "application/json"},
        json=body,
        timeout=REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    return bytes(response.content)


def render(
    clips: Sequence[Clip],
    out_dir: Path,
    *,
    api_key: str,
    model_id: str,
    fetch: Fetch = post_mp3,
    log: Callable[[str], None] = print,
) -> list[Path]:
    """Render every clip to ``out_dir``, skipping ones already there.

    A failed clip is logged and stepped over rather than raised: half a grid
    is still a comparison, and a run that dies on clip forty has spent the
    money for the first thirty-nine either way.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, clip in enumerate(clips, start=1):
        path = out_dir / clip.filename
        if path.exists():
            written.append(path)
            continue
        try:
            path.write_bytes(fetch(clip.voice_id, clip.body(model_id), api_key))
        except Exception as exc:  # noqa: BLE001 - one bad clip is not the run
            log(f"  [{index}/{len(clips)}] failed {clip.filename}: {exc}")
            continue
        written.append(path)
        log(f"  [{index}/{len(clips)}] {clip.filename}")
    return written


_STYLE = """
body { font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 2rem auto;
       max-width: 60rem; padding: 0 1rem; color: #16161d; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.05rem; margin-top: 2.5rem; }
h3 { font-size: 1rem; font-weight: 600; margin: 1.75rem 0 .4rem; }
p.meta { color: #6b6b76; }
blockquote { margin: 0 0 .6rem; padding-left: .8rem; border-left: 3px solid #d7d7de;
             font-size: 1.05rem; }
table { border-collapse: collapse; width: 100%; }
td, th { text-align: left; padding: .35rem .6rem .35rem 0; border-bottom: 1px solid #ededf2;
         font-variant-numeric: tabular-nums; white-space: nowrap; }
audio { height: 2rem; vertical-align: middle; }
"""


def index_html(clips: Sequence[Clip], *, model_id: str, cost: Estimate) -> str:
    """The page to listen on: every setting for one line, one click apart."""
    out: list[str] = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Voice sweep</title>",
        f"<style>{_STYLE}</style>",
        "<h1>Voice sweep</h1>",
        f"<p class='meta'>{html.escape(model_id)} &middot; {html.escape(str(cost))}</p>",
    ]
    for voice_id in dict.fromkeys(clip.voice_id for clip in clips):
        out.append(f"<h2>voice {html.escape(voice_id)}</h2>")
        for line in dict.fromkeys(
            clip.line for clip in clips if clip.voice_id == voice_id
        ):
            out.append(f"<h3>{html.escape(line.slug)} &middot; excitement {line.excitement:g}</h3>")
            out.append(f"<blockquote>{html.escape(line.text)}</blockquote>")
            out.append("<table><tr><th>stability<th>style<th>speed<th>clip")
            for clip in clips:
                if clip.voice_id != voice_id or clip.line != line:
                    continue
                setting = clip.setting
                src = html.escape(clip.filename)
                out.append(
                    f"<tr><td>{setting.stability:.2f}<td>{setting.style:.2f}"
                    f"<td>{setting.speed:.2f}"
                    f"<td><audio controls preload='none' src='{src}'></audio>"
                )
            out.append("</table>")
    return "\n".join(out) + "\n"


def write_index(clips: Sequence[Clip], out_dir: Path, *, model_id: str, cost: Estimate) -> Path:
    path = out_dir / "index.html"
    path.write_text(index_html(clips, model_id=model_id, cost=cost), encoding="utf-8")
    return path


def _floats(raw: str) -> tuple[float, ...]:
    return tuple(float(part) for part in raw.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a grid of voice settings over a few lines, to listen to.",
    )
    parser.add_argument(
        "--voice-id",
        action="append",
        required=True,
        metavar="ID",
        help="ElevenLabs voice to render. Repeat for more than one.",
    )
    parser.add_argument("--model-id", default=MODEL_ID, help=f"default {MODEL_ID}")
    parser.add_argument(
        "--stability", type=_floats, default=DEFAULT_STABILITIES, metavar="A,B,C"
    )
    parser.add_argument("--style", type=_floats, default=DEFAULT_STYLES, metavar="A,B")
    parser.add_argument("--speed", type=_floats, default=DEFAULT_SPEEDS, metavar="A,B")
    parser.add_argument(
        "--out", type=Path, default=None, help=f"default {OUT_ROOT}/<timestamp>"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="actually spend the credits. Without it this prints the estimate and stops.",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, fetch: Fetch = post_mp3) -> int:
    args = build_parser().parse_args(argv)
    clips = plan(args.voice_id, DEFAULT_LINES, grid(args.stability, args.style, args.speed))
    cost = estimate(clips, args.model_id)
    rate = credits_per_character(args.model_id)
    print(f"{cost} at {rate:g} a character on {args.model_id}")

    if not args.yes:
        print("refusing to spend that without --yes. Re-run with --yes to render.")
        return 2

    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key:
        print("no ELEVENLABS_API_KEY: nothing to render with.")
        return 1

    out_dir = args.out or OUT_ROOT / time.strftime("%Y%m%d-%H%M%S")
    render(clips, out_dir, api_key=api_key, model_id=args.model_id, fetch=fetch)
    index = write_index(clips, out_dir, model_id=args.model_id, cost=cost)
    print(f"open {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
