# Portfolio demo

Public page: https://www.aarshroongta.com/projects/soccer-commentary

The portfolio hosts a minimal player with a match switcher, three sentences
about the tech, and an expandable, seekable transcript. Back to projects and
source links remain visible. The project entry links directly to this page.

Five recorded examples are available, all with the same ElevenLabs caller
and analyst:

- Di María vs France, 2022 World Cup final — 28 seconds.
- Messi vs Bayern, 2015 Champions League — 34 seconds.
- Arsenal vs Norwich, 2013 Premier League — 28 seconds.
- Ronaldo vs Spain, 2018 World Cup — 32 seconds.
- Ferran Torres vs Betis, 2025 La Liga — 32 seconds.

These scripts were manually reviewed and edited for the showcase. They are
not unedited system output. The [retained plans and review notes](../examples/demo-showcase/README.md)
record source footage, observations, exact words, and editorial decisions.

## Source and media

The pipeline and standalone player live in this repository. The integrated
portfolio page lives in `aarshroongta1/personal-website`, under
`src/pages/SoccerCommentary.tsx`. Its video files are served from the existing
`soccer-commentary-demo.vercel.app` deployment, so ordinary Git deployments of
the portfolio do not depend on ignored local media.

The standalone player uses `web/data/demo-clips.json` as its catalog. Each
entry pairs a video with its own timestamped caller and analyst transcript.
Switching clips stops the previous video and starts the new selection paused
at the beginning. Playback and transcript inspection make no model calls.

Local video files are deliberately excluded from Git:

- `web/public/soccer-commentary/argentina-france-edited.mp4`
- `web/public/soccer-commentary/barcelona-bayern-edited.mp4`
- `web/public/soccer-commentary/arsenal-norwich-edited.mp4`
- `web/public/soccer-commentary/portugal-spain-edited.mp4`
- `web/public/soccer-commentary/betis-barcelona-edited.mp4`

Alternatively set the matching video URL variables documented in
`web/.env.example`, in `web/.env.local` or the hosting build environment.
The page exposes the matching transcript if a video is absent or fails to load.

## Run and deploy the player

```bash
cd web
npm ci
npm run dev
```

To update the media deployment, keep the matching video files in place and
run `vercel --prod` from `web`. The `.vercelignore` uploads these files from the
local machine without putting them in Git. A GitHub-only deployment needs the
remote video environment variables instead; otherwise it will have no media.

Deploy portfolio page changes from the `personal-website` repository. Keep the
source catalog and portfolio catalog aligned when replacing a clip or script.

## Verification

Check all five clips, switch while playback is active, and confirm the old video
stops. Expand the transcript, verify each match has the correct lines, and seek
using a timestamp. Check Back to projects, the source link, direct navigation to
the demo route, and the portfolio's demo link. Test mobile and desktop widths.

The source repository's web CI job checks lint and production builds. Python
pipeline tests run separately.
