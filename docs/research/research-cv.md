# Soccer Broadcast CV: Open-Source Landscape Survey

Prepared as reference for a video-first live commentary system needing a perception layer:
detect players and ball, track them, map pixels to pitch coordinates, identify teams and
ideally players, and derive events in near real time.

Survey date: 2026-09-10. Star counts and last-push dates pulled live from the GitHub API.

---

## 1. Repos and tools

### 1.1 End-to-end pipelines

| Repo | Stars | Last push | License |
|---|---|---|---|
| [roboflow/sports](https://github.com/roboflow/sports) | 5,350 | 2026-08-28 | MIT |

Player / goalkeeper / referee / ball detection, pitch keypoint detection, ByteTrack tracking
via `supervision`, SigLIP + UMAP + KMeans unsupervised team clustering, and a top-down radar
view. Six modes: `PITCH_DETECTION`, `PLAYER_DETECTION`, `BALL_DETECTION`, `PLAYER_TRACKING`,
`TEAM_CLASSIFICATION`, `RADAR`.

- **Models:** YOLOv8 (players), YOLOv8 (pitch keypoints), SigLIP (crop embeddings), UMAP, KMeans.
- **Weights:** Yes. Roboflow Universe datasets plus fine-tuned checkpoints via download badges
  in the README. `--device mps` is supported, so Apple Silicon works out of the box.
- **Hardware / FPS:** No published benchmarks. Python >= 3.8.
- **Biggest limitation:** These are example scripts, not a library. Batch processing over a
  file, no streaming loop, no persistent player identity, and ball recall is the weak point.

| Repo | Stars | Last push | License |
|---|---|---|---|
| [SoccerNet/sn-gamestate](https://github.com/SoccerNet/sn-gamestate) | 452 | 2026-05-02 | GPL-3.0 |

CVPRW'24. Full Game State Reconstruction: tracking and identifying every player from a single
moving broadcast camera onto a video-game-style minimap, with roles (player / goalkeeper /
referee), jersey numbers, and team affiliation.

- **Models:** YOLOv11 detection, StrongSORT tracking, PRTReid or BPBreID for re-id and team
  affiliation (multi-task), MMOCR jersey numbers, TVCalib / PnLCalib / No-Bells-Just-Whistles
  for calibration.
- **Weights:** Yes, fully automatic. TrackLab downloads the SoccerNet-gamestate dataset and all
  model weights on first baseline run.
- **Hardware / FPS:** No FPS numbers published. Docs advise lowering `batch_size` on OOM, so a
  real GPU is assumed. No real-time claims anywhere in the docs; workflow is batch.
- **Biggest limitation:** Offline by design, and the heaviest stack surveyed. GPL-3.0 is viral,
  which matters if the commentary system ever becomes a product.

| Repo | Stars | Last push | License |
|---|---|---|---|
| [TrackingLaboratory/tracklab](https://github.com/TrackingLaboratory/tracklab) | 249 | 2026-05-01 | MIT |

The modular end-to-end tracking framework that sn-gamestate is built on. Swappable
detect / track / re-id / callback modules under Hydra config.

- **Biggest limitation:** Framework overhead and a Hydra config maze. Built for research
  benchmarking, not for latency.

| Repo | Stars | Last push | License |
|---|---|---|---|
| [abdullahtarek/football_analysis](https://github.com/abdullahtarek/football_analysis) | 1,008 | 2024-04-23 | None stated |

The widely-followed tutorial repo. YOLO detection, ByteTrack, KMeans pixel segmentation for
team colors, optical flow for camera motion compensation, perspective transform for pitch
coordinates, plus speed and distance estimation.

- **Weights:** A trained checkpoint via download link.
- **Biggest limitation:** Tutorial code with no license file, hardcoded to one clip, and the
  homography comes from four manually picked points. Not a foundation to build on.

| Repo | Stars | Last push | License |
|---|---|---|---|
| [AtomScott/SportsLabKit](https://github.com/AtomScott/SportsLabKit) | 322 | 2023-12-19 | GPL-3.0 |

Python package turning sports video into CSV. Ships the SoccerTrack dataset (drone and fisheye
fixed cameras).

- **Biggest limitation:** Abandoned since late 2023, and its dataset is fixed-camera rather
  than broadcast, so the domain gap is large.

| Repo | Stars | Last push | License |
|---|---|---|---|
| [DonsetPG/narya](https://github.com/DonsetPG/narya) | 179 | 2023-03-25 | MIT |

Historic reference. TensorFlow-era tracking plus homography plus an expected-value model.
Worth reading for the pitch-registration approach, not worth running.

### 1.2 Pitch calibration / homography

| Repo | Stars | Last push | License |
|---|---|---|---|
| [mguti97/No-Bells-Just-Whistles](https://github.com/mguti97/No-Bells-Just-Whistles) (PnLCalib) | 60 | 2024-10-25 | Custom (NOASSERTION) |

CVPRW 2024. Sports field registration using both keypoint and line correspondences against a
3D soccer field model, solved with DLT. The strongest open calibration available.

- **Model:** Modified HRNetV2-w48 encoder-decoder for keypoint and line detection.
- **Weights:** Released.
- **Reported accuracy** ([paper](https://arxiv.org/html/2404.08401v4)):

  | Benchmark | Completeness | JaC@5 | Final |
  |---|---|---|---|
  | SN22-test-center | 97.7% | 80.6% | 78.7% |
  | SN23-test (full multi-view) | 79.5% | 76.7% | 60.9% |
  | WorldCup 2014 | 100.0% | 85.2% | IoU_part 98.6 / IoU_whole 96.3 |

- **Reported speed:** 7 Hz base, 4 Hz with the refinement module.
- **Biggest limitation:** At 4-7 Hz this is the single slowest link in any real-time chain.
  Note the completeness collapse from 97.7% to 79.5% once non-center camera views are included.

| Repo | Stars | Last push | License |
|---|---|---|---|
| [SoccerNet/sn-calibration](https://github.com/SoccerNet/sn-calibration) | 109 | 2024-06-18 | None stated |

Camera calibration challenge devkit and benchmark baselines. Useful mainly for the annotation
format and the evaluation code.

### 1.3 Action / event spotting

| Repo | Stars | Last push | License |
|---|---|---|---|
| [arturxe2/T-DEED](https://github.com/arturxe2/T-DEED) | 37 | 2026-01-07 | GPL-3.0 |

CVPRW 2024. Temporal-Discriminability Enhancer Encoder-Decoder for precise event spotting.
Won the SoccerNet 2024 Ball Action Spotting Challenge at **73.39 mAP@1** against a 56.15
baseline. Current SOTA baseline for the family.

- **Model:** RegNetY backbone with GSF temporal modules, encoder-decoder head.
- **Reported on other benchmarks:** FigureSkating FS-Comp 85.15, FS-Perf 88.17, FineDiving 73.23 (δ=1).
- **Biggest limitation:** Bidirectional temporal context, so it is non-causal. Cannot be used
  as-is for streaming. No published inference speed.

| Repo | Stars | Last push | License |
|---|---|---|---|
| [SoccerNet/sn-teamspotting](https://github.com/SoccerNet/sn-teamspotting) | 19 | 2025-08-26 | GPL-3.0 |

DevKit for the 2025 Team Action Spotting Challenge. T-DEED adapted to also predict which team
performed the action. Baseline 51.72 team-mAP@1; 2025 winner reached 60.03.

- **Biggest limitation:** Same non-causal issue, and team attribution is the harder half.

| Repo | Stars | Last push | License |
|---|---|---|---|
| [JeremieOchin/FOOTPASS](https://github.com/JeremieOchin/FOOTPASS) | 26 | 2026-06-17 | Apache-2.0 |

Baselines for the SoccerNet 2026 Player-Centric Ball-Action Spotting Challenge. First
player-centric, multi-modal, multi-agent dataset for play-by-play action spotting over
full-length broadcasts. This is the closest thing in the ecosystem to "commentary-grade" events.

- **Biggest limitation:** Very new, and it assumes tracking and player identity are already
  solved upstream, which is exactly the hard part.

| Repo | Stars | Last push | License |
|---|---|---|---|
| [lRomul/ball-action-spotting](https://github.com/lRomul/ball-action-spotting) | 137 | 2023-07-02 | MIT |

First place, Ball Action Spotting Challenge 2023. The cleanest production-grade training
pipeline in the whole ecosystem, worth reading even if not used.

- **Biggest limitation:** Stale since mid-2023, tuned to passes and drives only.

| Repo | Stars | Last push | License |
|---|---|---|---|
| [SoccerNet/sn-spotting](https://github.com/SoccerNet/sn-spotting) | 104 | 2024-02-07 | None stated |

Action spotting devkit with NetVLAD++ and CALF baselines running over precomputed features at
2 fps across 17 action classes and 550 matches.

- **Weights:** The precomputed features are the real asset here.
- **Biggest limitation:** Feature-based at 2 fps means coarse temporal localization. No license file.

| Repo | Stars | Last push | License |
|---|---|---|---|
| [int8/dude.k](https://github.com/int8/dude.k) | 15 | 2025-03-14 | None stated |

Refined T-DEED for the 2025 Ball Action Spotting Challenge. The author's writeup at
https://int8.io/team-ball-action-spotting-challenge-2025 is a practical account of what
training this class of model actually costs.

### 1.4 Ball detection specialists

| Repo | Stars | Last push | License |
|---|---|---|---|
| [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) | 247 | 2024-03-17 | None stated |

Unofficial PyTorch TrackNet. Consumes multiple consecutive frames and regresses a Gaussian
heatmap centered on the ball, learning trajectory as well as appearance. The canonical
architecture for small, fast, blurry sports balls.

- **Original paper results:** 99.7 precision / 97.3 recall / 98.5 F1 on tennis
  ([arXiv 1907.03698](https://arxiv.org/pdf/1907.03698)). Soccer is materially harder: larger
  pitch, wider shots, more occlusion, ball frequently off-frame.
- **Biggest limitation:** Trained for tennis. Soccer needs retraining, and no license is stated.

Also relevant: `DeepBall` ([arXiv 1902.07304](https://arxiv.org/pdf/1902.07304)) as the
soccer-specific ball-detector architecture reference, and the practical writeup at
https://blog.roboflow.com/tracking-ball-sports-computer-vision/ on image slicing / tiling for
small-object recall.

### 1.5 Tracking and re-identification

| Repo | Stars | Last push | License |
|---|---|---|---|
| [hsiangwei0903/Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU) | 74 | 2024-08-22 | None stated |
| [MCG-NJU/MixSort](https://github.com/MCG-NJU/MixSort) | 99 | 2023-08-21 | MIT |
| [MCG-NJU/SportsMOT](https://github.com/MCG-NJU/SportsMOT) | 227 | 2023-07-24 | None stated |
| [SoccerNet/sn-tracking](https://github.com/SoccerNet/sn-tracking) | 110 | 2023-07-07 | None stated |
| [SoccerNet/sn-reid](https://github.com/SoccerNet/sn-reid) | 90 | 2023-07-07 | MIT |

Deep-EIoU uses expanded-IoU association and is strong on SportsMOT, where players move
erratically and appearance is nearly uniform within a team. MixSort blends appearance into
ByteTrack/OC-SORT association. SportsMOT is the ICCV 2023 dataset (240 clips, three sports).
sn-tracking and sn-reid are the SoccerNet challenge devkits.

- **Common limitation:** All five are stale (2023-2024). None handles camera cuts, and
  broadcast re-id across shot boundaries remains poor.

### 1.6 Jersey number recognition

| Repo | Stars | Last push | License |
|---|---|---|---|
| [mkoshkina/jersey-number-pipeline](https://github.com/mkoshkina/jersey-number-pipeline) | 67 | 2024-10-07 | Custom (NOASSERTION) |
| [SoccerNet/sn-jersey](https://github.com/SoccerNet/sn-jersey) | 30 | 2024-07-02 | None stated |

The Koshkina pipeline is the general framework: a legibility classifier filters frames, pose
guides the crop to the torso, PARSeq scene-text recognition reads the digits, and votes are
aggregated over the whole tracklet. sn-jersey is the challenge devkit, operating on short
player tracklets.

- **Biggest limitation:** Tracklet-level voting means it cannot report an identity until it has
  accumulated a tracklet, which inherently lags live play. Numbers are legible in only a
  minority of broadcast frames.

### 1.7 Commentary generation (directly on-target)

| Repo | Stars | Last push | License |
|---|---|---|---|
| [haolinyang-hlyang/SoccerMaster](https://github.com/haolinyang-hlyang/SoccerMaster) | 90 | 2026-08-22 | None stated |

**CVPR 2026 Oral.** A unified soccer-specific vision foundation model spanning commentary
generation, detection, tracking, and classification.

- **Architecture:** SigLIP2-large (`google/siglip2-large-patch16-512`) backbone with spatial and
  temporal attention modules, integrating YOLO detection, SAM2 segmentation, and Qwen2.5-VL.
- **Weights:** Released on Hugging Face at `xleprime/SoccerMaster`.
- **Dataset:** "Soccer Factory", 7,000 videos with per-frame bounding boxes, roles, jersey
  numbers, and camera parameters. Roughly 127 GB.
- **Hardware:** CUDA 12.1 / PyTorch 2.4.1. Multi-GPU for some components. No memory floor stated.
- **Biggest limitation:** No license file, no latency numbers, CUDA-only (no Apple Silicon
  path), and the dataset download is enormous.

| Repo | Stars | Last push | License |
|---|---|---|---|
| [AntoineBohin/soccer-ai-commentator](https://github.com/AntoineBohin/soccer-ai-commentator) | 11 | 2025-04-28 | MIT |

The closest existing analog to the target system, and the single most useful repo to read for
wiring.

| Stage | Model |
|---|---|
| Action spotting | EfficientNetV2-B0 + 3D CNN (2023 SoccerNet challenge model) |
| Tracking | YOLOv8 + ByteTrack, adapted from roboflow/sports |
| Visual description | Qwen2.5-VL-7B (selected after benchmarking six VLMs) |
| Commentary text | Gemma 3-4B Instruct |
| Speech | Zonos / Coqui-TTS with voice cloning and emotion embeddings |

- **Weights:** Download scripts for every stage (`download_vlm_weights.py`,
  `download_action_data.py`, `download_ball_data.py`).
- **Hardware:** Minimum 24 GB VRAM.
- **Latency:** Under ten seconds per clip. Clip-wise, not live. No real-time claims.
- **Biggest limitation:** Clip-batch architecture. Latency is an order of magnitude off live.

| Repo | Stars | Last push | License |
|---|---|---|---|
| [jyrao/MatchTime](https://github.com/jyrao/MatchTime) | 105 | 2025-01-02 | None stated |

**EMNLP 2024 Oral.** The key contribution is temporal re-alignment: SoccerNet-Caption
timestamps are systematically misaligned with the video, and MatchTime corrects them at scale.
Result is 422 aligned videos (373 train / 49 val) and 29,476 video-text pairs. The MatchVoice
model encodes frames with a pretrained visual encoder, aggregates temporally, and projects into
LLM prefix tokens via a trainable MLP.

- **Biggest limitation:** Commentary is generated from visual features alone, with no explicit
  event or player grounding. It produces plausible commentary, not verified commentary.

| Repo | Stars | Last push | License |
|---|---|---|---|
| [SoccerNet/sn-caption](https://github.com/SoccerNet/sn-caption) | 39 | 2024-04-12 | None stated |
| [SoccerNet/sn-echoes](https://github.com/SoccerNet/sn-echoes) | 20 | 2025-05-29 | None stated |

sn-caption is the Dense Video Captioning devkit and labels, i.e. the corpus MatchTime exists to
fix. sn-echoes is ASR-transcribed real broadcast audio commentary, notably released CC BY 4.0.

### 1.8 Detector / infra building blocks

| Repo | Stars | Last push | License | Note |
|---|---|---|---|---|
| [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | 9,428 | 2026-09-10 | Apache-2.0 | Real-time DETR, SOTA on COCO, edge-latency oriented |
| [roboflow/supervision](https://github.com/roboflow/supervision) | 49,951 | 2026-09-10 | MIT | Annotation, tracking wrappers, ByteTrack |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | 61,477 | 2026-09-10 | **AGPL-3.0** | YOLO26 / YOLO11 / YOLOv8 |
| [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) | 7,890 | 2025-08-04 | Apache-2.0 | Pose, needed for torso-guided jersey crops |
| [open-mmlab/mmtracking](https://github.com/open-mmlab/mmtracking) | 3,901 | 2023-09-19 | Apache-2.0 | Stale, superseded |

**Licensing trap:** Ultralytics YOLO is AGPL-3.0, which infects any network-hosted service
built on it. RF-DETR at Apache-2.0 is the clean alternative and is also faster at comparable
accuracy. Make this swap early, not late.

---

## 2. Datasets and what may legally be shown

### SoccerNet (the center of gravity)
https://www.soccer-net.org/data

- **Contents:** 550 broadcast games. 500 + 50 videos at 25 fps in 720p and 224p. Labels across
  action spotting (17 classes), ball action (12 classes), replay grounding, calibration,
  re-identification, tracking, jersey numbers, dense video captioning, and monocular depth.
  Precomputed video features extracted at 2 fps.
- **Access:** **An NDA is required to download any video data.** Features, labels, and images
  do not carry the same explicit NDA gate.
- **Redistribution:** Videos may not be redistributed, publicly hosted, reconstructed, or used
  commercially. Annotations and metadata derived from SoccerNet content must not be used to
  reconstruct or redistribute the original videos.
- **Exception:** SoccerNet-Echoes (ASR commentary transcripts) is CC BY 4.0.

### DFL Bundesliga Data Shootout
https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout

- **Contents:** 12 training videos of 60 minutes each with CSV event labels, plus short clips.
  Three event classes: Play (pass), Challenge (tackle), Throw-In.
- **Terms:** Kaggle competition rules. Treat as research-only and non-redistributable.

### Others
- **SportsMOT** (ICCV 2023): 240 tracking clips across three sports including soccer. No license file.
- **SoccerTrack** (via SportsLabKit): drone and fisheye fixed cameras, non-commercial.
- **Soccer Factory** (SoccerMaster): 7,000 videos, per-frame boxes / roles / jersey numbers /
  camera parameters, ~127 GB.
- **FOOTPASS** (Footovision): player-centric play-by-play over full broadcasts.
- **WorldCup 2014 / TS-WorldCup:** the standard homography-estimation benchmarks.

### Licensing implication for a public demo

Blunt version: **every match-footage dataset in this space is research-only.** A public demo
can ship model weights trained on them. It cannot ship the footage.

Practical plan: train and evaluate against SoccerNet, but demo on video the student actually
owns (a self-filmed amateur or academy match) or on a clip with unambiguous rights. Keep
SoccerNet strictly on the training side of the line. Also avoid AGPL detector weights in
anything hosted.

---

## 3. Accuracy and latency to actually expect

### Challenge results, baseline vs. winner

| Task | Metric | Baseline | Winner | Source |
|---|---|---|---|---|
| Ball Action Spotting 2024 (12 classes) | mAP@1 | 56.15 | **73.39** (T-DEED) | [SoccerNet 2024](https://arxiv.org/pdf/2409.10587) |
| Team Ball Action Spotting 2025 | team-mAP@1 | 51.72 | **60.03** (dudekTBAS-1) | [SoccerNet 2025](https://arxiv.org/pdf/2508.19182) |
| Game State Reconstruction 2025 | GS-HOTA | 29.01 | **63.90** (KIST-GSR) | SoccerNet 2025 |
| Multi-View Foul Recognition 2025 | Balanced Acc. | 36.99% | **52.22%** | SoccerNet 2025 |
| Monocular Depth 2025 | RMSE (x1e-3) | 3.757 | **2.418** | SoccerNet 2025 |

Read the GS-HOTA number carefully. Even the challenge winner, with unlimited offline compute,
lands near 64 on a HOTA-style scale for the combined task of knowing who is where on the pitch.
**That is the hard ceiling on grounded, player-named commentary today.**

### Component-level expectations

| Component | Realistic quality | Realistic speed |
|---|---|---|
| Player detection, main tactical camera | Effectively solved. Fine-tuned YOLO or RF-DETR is reliable | 60+ FPS on RTX 3080-class with FP16/TensorRT. ~10-20 FPS at 640px on M-series MPS |
| Player detection, replays / close-ups / crowd | Degrades sharply; every camera cut breaks every track | Same |
| Ball detection | **The weak link.** Small, motion-blurred, occluded, frequently off-frame. Needs tiled or high-res inference | Tiling multiplies cost several-fold |
| Tracking identity | ByteTrack holds identity for seconds, not minutes. Cuts reset it | Negligible |
| Pitch homography (PnLCalib) | 97.7% completeness / 80.6 JaC@5 center-cam; 79.5% completeness full multi-view | **7 Hz base, 4 Hz refined** |
| Ball action spotting | 73.39 mAP@1 (2024 winner) | Offline, non-causal |
| Team-attributed ball action | 60.03 team-mAP@1 (2025 winner) | Offline |
| Full game state reconstruction | GS-HOTA 63.90 (2025 winner) | Offline, minutes per clip |
| Commentary LLM turnaround | See below | Detection fed at 5 FPS, LLM at 1-3 FPS |

### Real-time commentary latency, measured

From GetStream's build log (https://getstream.io/blog/ai-football-commentator-lessons/), which
is the only public measurement of this exact architecture:

- **Architecture:** RF-DETR detection at 5 FPS, annotated video streamed into a realtime LLM at 1-3 FPS.
- **Time to first audio:** OpenAI Realtime 0.39s mean (0.31-0.72s range). Gemini Live 3.06s
  mean (1.52-5.05s range). Roughly 8x difference, and OpenAI far more consistent.
- **Accuracy:** Both models "got match events right by sheer luck, but it was wrong more than
  half the time."
- **Diagnosed failure modes:** models reason over a handful of frames rather than a temporal
  sequence; fast action is where they fall apart, especially when ball detection drops out.
- **Stated requirements for viability:** 5-10 second video context windows, native
  video-triggered response rather than text prompting, and sub-1-second latency.

The load-bearing lesson: **a raw VLM fed video is wrong more than half the time on event
identification.** The CV layer is not optional garnish; it is what makes commentary factual.

### Other reported figures for context

- YOLOv5 + OpenCV real-time football system: 28.5 FPS ([IEEE](https://ieeexplore.ieee.org/abstract/document/10934234)).
- NVIDIA DeepStream + TensorRT FP16 tracking: up to 80 FPS on an RTX 3080Ti laptop GPU.
- YOLOv8x beats YOLOv12x on F1 for football detection despite YOLOv12x having fewer parameters
  and lower inference time.

---

## 4. Recommended minimal perception stack

**Start from [roboflow/sports](https://github.com/roboflow/sports).** It is MIT-licensed,
actively maintained through August 2026, already contains soccer-tuned detection, pitch
keypoints, team clustering, and radar in one place, and already accepts `--device mps`.
Nothing else in the survey combines those four properties.

Then make five specific changes:

1. **Swap the detector to RF-DETR**, fine-tuned on the Roboflow `football-players-detection`
   dataset. This escapes the Ultralytics AGPL and buys latency headroom.
2. **Wrap the per-frame code in a streaming loop** with a ring buffer. The shipped examples are
   file-batch scripts; this is the largest single piece of engineering work.
3. **Freeze the team clustering.** Compute SigLIP embeddings over a few hundred crops at
   kickoff, fit UMAP + KMeans once, then classify by nearest centroid per frame. Re-clustering
   every frame is both slow and unstable across shots.
4. **Decimate calibration.** Run pitch keypoints every 5th-10th frame and smooth the homography
   between solves with an exponential filter. This matches PnLCalib's real 4-7 Hz ceiling
   instead of fighting it.
5. **Derive v1 events from geometry, not a learned spotter.** Possession = nearest player to
   ball with hysteresis. Pass = possession change within a team. Shot = ball velocity vector
   aimed at the goal mouth. Goal = ball position crossing the line plus a sustained possession
   reset. Geometry rules are debuggable, causal, and free; a learned spotter is none of those.

**Add later, in this order:** T-DEED as a second pass over a trailing few-second window
(accepting the lag) for events geometry cannot catch; jersey numbers via the Koshkina pipeline
once tracklets are stable; player identity last, if at all.

**Read for wiring, don't fork:** `soccer-ai-commentator` for the end-to-end VLM/LLM/TTS chain,
`MatchTime` for how commentary text gets time-aligned to video, `lRomul/ball-action-spotting`
for what disciplined training code looks like in this domain.

**Explicitly out of scope for v1:** player identity. Use team plus shirt-number-when-legible.

---

## 5. What is still research-hard

**Player identity from broadcast.** Jersey numbers are legible in only a minority of frames, so
every working pipeline votes across a whole tracklet and therefore lags. Combined with identity
breaking at every camera cut, naming individual players live is not reliably solvable today.
The GS-HOTA 63.90 figure is the honest ceiling.

**The ball when it is not visible.** Off-screen, occluded, and motion-blurred balls are common,
and possession logic built on ball position fails silently during exactly the moments
commentary matters most. That the 2026 literature is now pursuing ball-free event inference
from player trajectories alone (e.g. PathCRF, [arXiv 2602.12080](https://arxiv.org/pdf/2602.12080))
is direct evidence the frontal approach has not closed.

**Goal detection.** The temporal anchor is genuinely ambiguous, and broadcast directors cut to
replays and celebrations precisely when the visual evidence would be clearest. Treat it as the
one event class that needs a deliberate confirmation delay rather than an instant call.

**Causal (streaming) action spotting.** Every strong model on every leaderboard consumes future
context. Converting one to past-only streaming is an open engineering problem, and the accuracy
cost of doing so is unmeasured in the public literature. Budget for it to be worse than the
published numbers by an unknown margin.

**Multi-frame temporal reasoning in VLMs.** Per GetStream's measurements, current realtime
models excel at static frame analysis but cannot synthesize multi-frame sequences into coherent
narrative. This is why the CV layer must supply the facts and the LLM must only supply the prose.

---

## Sources

- https://github.com/roboflow/sports
- https://github.com/roboflow/sports/blob/main/examples/soccer/README.md
- https://github.com/SoccerNet/sn-gamestate
- https://github.com/TrackingLaboratory/tracklab
- https://github.com/SoccerNet/sn-tracking
- https://github.com/SoccerNet/sn-calibration
- https://github.com/SoccerNet/sn-spotting
- https://github.com/SoccerNet/sn-teamspotting
- https://github.com/SoccerNet/sn-reid
- https://github.com/SoccerNet/sn-jersey
- https://github.com/SoccerNet/sn-caption
- https://github.com/SoccerNet/sn-echoes
- https://github.com/mguti97/No-Bells-Just-Whistles
- https://arxiv.org/html/2404.08401v4 (PnLCalib)
- https://github.com/arturxe2/T-DEED
- https://arxiv.org/html/2404.05392v2 (T-DEED paper)
- https://github.com/JeremieOchin/FOOTPASS
- https://github.com/lRomul/ball-action-spotting
- https://github.com/int8/dude.k
- https://int8.io/team-ball-action-spotting-challenge-2025
- https://github.com/mkoshkina/jersey-number-pipeline
- https://github.com/haolinyang-hlyang/SoccerMaster
- https://github.com/AntoineBohin/soccer-ai-commentator
- https://github.com/jyrao/MatchTime
- https://arxiv.org/pdf/2406.18530 (MatchTime)
- https://github.com/AtomScott/SportsLabKit
- https://github.com/DonsetPG/narya
- https://github.com/yastrebksv/TrackNet
- https://arxiv.org/pdf/1907.03698 (TrackNet)
- https://arxiv.org/pdf/1902.07304 (DeepBall)
- https://github.com/hsiangwei0903/Deep-EIoU
- https://github.com/MCG-NJU/MixSort
- https://github.com/MCG-NJU/SportsMOT
- https://github.com/abdullahtarek/football_analysis
- https://github.com/roboflow/rf-detr
- https://github.com/roboflow/supervision
- https://arxiv.org/pdf/2508.19182 (SoccerNet 2025 Challenges Results)
- https://arxiv.org/pdf/2409.10587 (SoccerNet 2024 Challenges Results)
- https://arxiv.org/pdf/2504.06357 (From Broadcast to Minimap)
- https://arxiv.org/pdf/2602.12080 (PathCRF, ball-free event detection)
- https://www.soccer-net.org/data
- https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout
- https://getstream.io/blog/ai-football-commentator-lessons/
- https://blog.roboflow.com/tracking-ball-sports-computer-vision/
- https://ieeexplore.ieee.org/abstract/document/10934234
