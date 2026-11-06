# Recorded commentary

Generate a script from your own soccer clip, review it, then render speech and
captions. Run these commands from the repository root after the
[setup steps](../README.md#generate-your-own-commentary). Rendering needs FFmpeg with subtitle
support and ffprobe; the `demo` extra supplies a compatible FFmpeg binary.

## Generate a script

Place a video of at least 40 seconds at `clips/match.mp4`, then inspect a dry run:

```bash
uv run python scripts/generate_demo.py \
  --clip clips/match.mp4 --duration 40 --out runs/demo/plan.json
```

Add `--spend` to generate the script using your `OPENAI_API_KEY`. Defaults are
GPT-6 Astra for vision and GPT-5.6 Terra for writing and research.

Use `--pack path/to/roster.json` to supply teams, kit colours, and player identities.
For pre-match research, add `--fixture "Home vs Away, competition and date"`
and `--as-of YYYY-MM-DD`. A saved brief can be supplied with `--research` instead.
See `--help` for all options.

Review `runs/demo/plan.json` and the stage outputs in `runs/demo/plan.json.run/`.
Names and descriptions can be wrong even when validation passes. To revise a
script without repeating vision, pass
`--reuse-observations runs/demo/plan.json.run/observe.json` and a new output path.

## Render

On macOS, the renderer uses the local Daniel and Samantha voices:

```bash
uv run python scripts/render_demo.py \
  --plan runs/demo/plan.json --clip clips/match.mp4 \
  --out runs/demo/preview.mp4
```

The renderer replaces the original audio, adds captions, and writes a timing
report beside the MP4. Use `--dry-run` to inspect estimated timing without
creating media. Rendering itself makes no model or ElevenLabs calls.

For ElevenLabs speech, set `ELEVENLABS_API_KEY` in `.env` and run:

```bash
uv run python scripts/synthesize_demo.py \
  --plan runs/demo/plan.json --out-dir runs/demo/audio
```

This is also a dry run until `--spend` is added. Pass the resulting
`--manifest runs/demo/audio/manifest.json` to the renderer to use that speech.
On other platforms, use this option or supply your own manifest mapping line IDs
to audio file paths. Existing audio can be reused without new synthesis.

## View the result

The player at `/demo` uses `web/data/demo-clips.json` to pair each video with
its own transcript. To add different footage, add a catalog entry and its MP4
under `web/public/soccer-commentary/`. Transcript timestamps must match the
rendered audio placement, including any delays reported by the renderer.

The five featured clips can also use remote URLs through the variables in
`web/.env.example`. Their [edited showcase plans](../examples/demo-showcase/README.md)
retain the words, evidence, source windows, and review notes for each render.

Media is git-ignored and is not supplied with a fresh clone. See the
[portfolio guide](PORTFOLIO.md) for deployment and missing-video behavior.

The [example directory](../examples/recorded-demo/) contains a reviewed script,
observations, and research for inspection without model calls. Rendering that
example requires its matching source clip.
