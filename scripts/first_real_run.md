# First run on real footage

The order matters: every step before the last one is free, and the last one
spends money at roughly $0.25 a minute.

## 1. Keys

`.env` in the repo root:

```
ANTHROPIC_API_KEY=sk-ant-...
CALLER_MODEL=claude-haiku-4-5
MAX_USD_PER_MATCH=5
```

`claude-haiku-4-5` for the first run because the point of it is to find the
crop, the clock and the roster, not to hear good commentary. Move the caller
to `claude-opus-5` once the run completes without the gate rejecting
everything.

## 2. The video and the captions, in one command

Downloading is your call; the repo has no downloader.

```
brew install yt-dlp ffmpeg
yt-dlp -f "bv*[height<=720]+ba" --merge-output-format mp4 \
       --write-auto-subs --sub-lang en --sub-format json3 <url>
```

That leaves `<name>.mp4` and `<name>.en.json3` side by side. The captions are
the human commentator's transcript — ground truth for timing and naturalness,
never for facts:

```
uv run python -m commentary captions "<name>.en.json3" --out transcript.json
```

## 3. The crop

The score bug sits in a different corner on every broadcast, and a box that
is half a bug reads as an unreadable board for ninety minutes with nothing in
the trace to say so. Check it by eye, at a second when the bug is up:

```
uv run python -m commentary crop --path "<name>.mp4" --at 300 \
       --crop 0,0,0.42,0.16 --out /tmp/bug.png
```

The left half is the frame with the box drawn on it; the right half is what
the board reader will actually be handed. Adjust the four fractions until the
score and the clock fill it, then keep them for `run --crop`.

## 4. The pack

Either research it:

```
uv run python -m commentary research "Argentina" "France" \
       --competition "World Cup final" --when "2022-12-18"
```

or write the JSON by hand from the team sheet — shirt numbers matter more
than anything else in it, and for a StatsBomb fixture the lineups file has
them. Set each team's `kit` and `demonym` by hand either way.

## 4b. Names from the picture

Nothing to install. The caller reads shirt numbers and graphics itself and
checks them against the team sheet, so the `kit` strings in the pack are what
let it say which side a number is on. There used to be a local tracker
drawing tags on the frames; it was removed after a real-clip ablation (see
`docs/HANDOFF.md`, section 7).

## 5. Ten minutes first

Never start on ninety. Cut ten minutes that contain something:

```
ffmpeg -ss 00:20:00 -t 00:10:00 -i "<name>.mp4" -c copy cut.mp4
```

```
uv run python -m commentary run --source file --path cut.mp4 \
       --backend anthropic --pack packs/argentina-france.json \
       --crop 0,0,0.42,0.16 --seconds 600 --delay 8
```

`--seconds` is wall-clock and the file plays at real speed, so ten minutes of
video is ten minutes of run. `MAX_USD_PER_MATCH` stops the model calls dead
when the run passes the cap, so the number in `.env` is the real limit, not
the intention.

## 6. Grade it

```
uv run python -m commentary feed statsbomb-events.json \
       --home Argentina --away France --out feed.json
uv run python -m commentary grade runs/file-*.jsonl
```

Read the alignment line before any recall number: an alignment four seconds
out produces a complete, plausible table in which every line has missed its
event.
