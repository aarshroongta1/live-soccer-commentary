# Recorded commentary pipeline

- `commentary-pipeline.mmd`: Mermaid source.
- `commentary-pipeline.excalidraw`: editable scene; open with File → Open at excalidraw.com.
- `commentary-pipeline.svg`: vector export.
- `commentary-pipeline.png`: raster export.

The flow follows the recorded workflow in `src/commentary/recorded_demo.py`,
then validation and manual review, ElevenLabs synthesis, FFmpeg rendering,
and saved playback. The graph's research, observation, and joint writer stages
are summarized in one node to keep the overview simple.

The demo pages render the same six stages as responsive HTML: a connected
two-row flow on desktop and a vertical flow on phones. The standalone
component is `web/app/demo/architecture.tsx`; the portfolio copy lives in
`personal-website/src/components/CommentaryArchitecture.tsx`.

The diagram describes recorded generation, rather than the local live-text
experiment. Published showcase scripts are manually edited before voicing,
and playback makes no model calls.
