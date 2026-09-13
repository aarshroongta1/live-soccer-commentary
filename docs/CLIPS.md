# Clips: what happens on footage the system was never tuned on

Everything before this was one three-minute clip — Argentina v France 2022,
video 36:00 to 39:00, Di María's goal — run and re-run until it produced good
commentary. Every rule in the gate, every number in `perception`, and the
whole definition of done were fitted to those three minutes. What none of it
had been asked is whether any of it is about football or only about that clip.

So: five more clips off the same broadcast, each carrying an event type the
system has never seen — a penalty, two substitutions, a card, a second-half
comeback, a shootout — run once each with the same command, and graded by the
same machine rather than by eye:

```
uv run python -m commentary run --source file --path clips/<clip>.mp4 \
    --backend anthropic --pack clips/pack-argfra-2022.json \
    --seconds 195 --delay 8 --marks --out runs/<clip>
uv run python -m commentary grade runs/<clip>/*.jsonl \
    --pack clips/pack-argfra-2022.json \
    --statsbomb clips/statsbomb-events-3869685.json \
    --lineups clips/statsbomb-lineups-3869685.json
```

## The video clock

The clips come out of one FIFA upload
(`https://www.youtube.com/watch?v=RgqKdplLIk4`, 2:37:00 long), and the offset
between the match clock and that upload's clock is different in every period,
because the upload cuts the intervals down. Each one was measured by pulling a
three-second probe and reading the score bug off a frame:

| period | probe | bug read | offset (video − match) |
|---|---|---|---|
| 1 first half | 36:00 | 34:24 | **+96 s** |
| 2 second half | 1:00:00 | 50:01 | **+599 s** |
| 2 (check) | 1:30:00 | 80:01 | +599 s |
| 3 extra time 1 | 2:05:00 | 104:54 | **+1206 s** |
| 4 extra time 2 | 2:22:00 | 120:00 +0:04 | **+1320 s** |
| 5 shootout | 2:29:30 | *no clock* — tally 1-1 | **≈ +1645 s** |

The shootout has no clock to measure against: StatsBomb stamps the kicks
120:13 to 125:57 and the broadcast shows a tally with no time on it at all.
The offset above is fitted from one frame (the tally at 1-1 with Coman already
saved, so between 121:42 and 122:27) and is good to about twenty seconds.

## Budget

$10, hard, across every run on this page.

| run | cost | running total |
|---|---:|---:|

## Per-clip reports

*(filled in as the runs land)*

## Needs a decision

*(design changes a clip exposed, which are not this session's to make)*
