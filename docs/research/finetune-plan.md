# Fine-tuning the phraser: a plan for tonight

## 1. Recommendation in five lines

1. **Data**: SoccerNet-Echoes (real Whisper ASR of broadcast audio, 550 games, CC BY 4.0, ungated) joined on game+half to SoccerNet action-spotting `Labels-v2.json` (17 classes, no NDA) for the event field; GOAL (MIT, human-proofread ASR, names intact) as clean eval seed; our 228 YouTube utterances as register anchor.
2. **Do not train on SoccerNet-Caption or MatchTime.** Their text is scraped from flashscore.com, written past-tense match-report prose, not speech. It would undo the register work.
3. **Base model**: `mlx-community/Qwen3-1.7B-4bit` (968 MB, Apache-2.0). Fallback `Qwen3-0.6B-4bit` (335 MB).
4. **Train and serve on the Mac.** 8 GB rules out 8B and makes the CUDA-to-MLX export path cost more than it saves. `mlx_lm.lora`, then `mlx_lm.server` with prefix caching.
5. **End to end: 7-8 hours, ~$15 of API spend, $0 of GPU rental.**

## 2. Schedule

| # | Step | Hours | $ |
|---|---|---|---|
| 1 | Pull SN-Echoes (`whisper_v3`, 566 MB), `Labels-v2.json`, GOAL | 0.5 | 0 |
| 2 | *Parallel*: install mlx-lm, pull the model, measure real prefill/decode | 0.5 | 0 |
| 3 | Join segments to events by time; roster string-match; filter under 12 words | 1.5 | 0 |
| 4 | Haiku back-fill of residual form fields, in background | 1.0 | ~$15 |
| 5 | *Parallel*: MLX backend behind `commentary.llm.base` (the server is OpenAI-shaped) | 1.0 | 0 |
| 6 | `mlx_lm.lora`, 2000-3000 iters | 2.0 | 0 |
| 7 | `mlx_lm.fuse`, serve, measure latency | 0.5 | 0 |
| 8 | Eval via `rephrase` and `replay` over existing traces | 1.0 | ~$1 |

Steps 2 and 5 run against 1, 3 and 4. Start step 4 as soon as step 3 has 30k rows, and write the trainer while it runs.

## 3. Data

| Source | Utterances | Real speech | Names | Event labels | Licence | Friction |
|---|---|---|---|---|---|---|
| **SN-Echoes** | 4,387,121 segs, 550 games | Yes (Whisper) | Yes, garbled | No, join | CC BY 4.0 | None |
| **Labels-v2** | 500 games, 17 classes | n/a | n/a | Yes | Research | No NDA |
| **GOAL** | ~22k sentences | Yes, proofread | Yes | Clip-level | MIT | None |
| SN-Caption | 36.9k, 471 games | **No, flashscore** | `description` | Often empty | Unclear | None |
| MatchTime | 26k realigned | **No, same text** | Yes | Yes | CC BY-SA 4.0 | None |
| LFC (INLG 25) | 12,440, 40 matches | Yes | Unstated | Tracking | CC BY 4.0 | Paper |
| YouTube json3 | ~1-2k/match | Yes | Garbled | No | ToS | `yt-dlp` |

SoccerReplay-1988 is larger still (~150k lines) and also flashscore prose, masked to `[PLAYER]`. SN-Caption's `description` does carry real names, so anonymisation is not the blocker. The register is.

**Excitement labels.** Audio energy, the method Rui et al. used for baseball in 2000: six statistics over a 0.5 s window, max/mean/range of pitch and energy, reaching 75% agreement with human highlight marking. `yt-dlp -f bestaudio` pulls audio without video, ~144 MB for two hours. Map caption times with `librosa.time_to_frames`, take `librosa.feature.rms`, normalise per match to 0..1. YouTube slice only; for SN-Echoes we hold no audio, so derive excitement from the event class.

## 4. Models

| Model | 4-bit size | Est. M1 decode | Licence | Fit |
|---|---|---|---|---|
| Qwen3-0.6B-4bit | 335 MB | ~160 t/s | Apache-2.0 | Fallback; hits 0.3 s cold |
| **Qwen3-1.7B-4bit** | **968 MB** | **~58 t/s** | **Apache-2.0** | **Pick.** ~0.31 s warm |
| Llama-3.2-1B-4bit | 695 MB | ~90 t/s | Llama 3.2 Community | Derivative must be named `Llama-*` |
| Gemma-3-1b-qat-4bit | 733 MB | ~90 t/s | Gemma custom | Viable, worse licence |
| Qwen3-4B-4bit | 2.26 GB | ~25 t/s | Apache-2.0 | Too slow, ~0.72 s |
| Qwen3-8B-4bit | 4.61 GB | — | Apache-2.0 | **Out.** 5.61 GB peak vs ~5.3 GB set |

Decode figures are extrapolated from measured M1 8-core llama.cpp numbers (117.96 t/s prefill, 14.15 t/s decode on 7B Q4_0), bracketed by measured M1/8 GB readings of 23-45 t/s at 2-4B. No published MLX benchmark exists for a base M1, so **measure in step 2 before committing.**

Latency only works with a cached prefix. `mlx_lm.server` keeps the model warm and reuses the longest cached KV prefix via `--prompt-cache-dir`. Put a long static system block first and a short volatile match-state block last, so prefill covers only changed tokens. Do not pass `--kv-bits`, which disables batching. Disable Qwen3 thinking mode, or a stray `<think>` block alone blows the budget. Format compliance is settled at this size: a 2026 LoRA study shows 100% parse rate at 0.8B and above, collapsing only at 270M.

## 5. Training pair design

Input, plain text so the prefix caches:

```
event: goal | side: home | team: Argentina
scene: live_play | score: 1-0 | clock: 23:14
players: Messi, Di Maria
recent: "Messi again." / "Up by Tagliafico."
```

Output: `Messi! || 0.95`, the utterance and the excitement float.

Use the `completions` schema with `--mask-prompt`, so loss falls on the 15-token output not the 100-token input. Set `--max-seq-length 256`; the 2048 default wastes ~90% of each step on padding.

**Back-filling.** Event comes from the join and names from roster match, so only scene, side and score need a model. One Haiku pass over 30k lines at ~150 in / ~80 out tokens is ~$15, halved by the batch API. If the clock is tight, derive side and score from `Labels-v2`, skip scene, pay $0.

**Holdout**: split by match, never by line. Reserve 20 Echoes matches plus all of GOAL. Splitting by line leaks a match's naming habits and flatters every metric.

## 6. Evaluation

Run `commentary rephrase` against the local server over an existing trace, then `replay`. Four bars, against the Haiku phraser on the same traces:

- **Register**: median 3-5 words, fragment rate above 70%, name-first rate above 50%. Fail past a median of 6.
- **Fidelity**: gate rejection rate no worse than Haiku's; `gate_rejection_rate` and `gate_table` already report it. Zero invented goals is absolute.
- **Cliché**: `repetition_rate` at threshold 0.62 over a 5-line window, under 0.10 across a match. This is the metric the fine-tune should move most.
- **Latency**: p50 under 0.35 s, p95 under 0.5 s, warm through the server.

Ship only if fidelity holds and cliché improves. Register parity is not enough; Haiku already has that.

## 7. Risks, ranked

1. **ASR too garbled.** Keep only utterances whose every capitalised token matches the team sheet, as `build_commentary_examples.py` does. Even 5% of 4.4M segments is 200k lines.
2. **The event join is loose**, since Echoes times and `Labels-v2` positions are independently derived. Widen to ±15 s and keep only segments with exactly one candidate.
3. **Latency misses.** Drop to Qwen3-0.6B-4bit, which clears 0.5 s cold. Failing that, shorten the volatile suffix.
4. **MLX training memory creeps**, a known issue on far larger machines. Use `--num-layers 4 --batch-size 1 --grad-checkpoint`.
5. **1.7B loses the register.** Keep Haiku behind a flag and route by event class, local for routine touches.
6. **MLX training fails outright.** RunPod RTX 4090 at $0.34/hr, or Together managed LoRA at a $4.00 minimum. Confirm the vendor returns adapter weights first; PEFT to MLX needs a merge, a transpose, and a `scale = lora_alpha / r` correction.

## Sources

Datasets: [SN-Echoes](https://arxiv.org/abs/2405.07354), [HF](https://huggingface.co/datasets/SoccerNet/SN-echoes) · [SoccerNet data/NDA table](https://www.soccer-net.org/data) · [SN-Caption](https://arxiv.org/abs/2304.04565) · [MatchTime](https://arxiv.org/abs/2406.18530), [HF mirror](https://huggingface.co/datasets/Homie0609/MatchTime) · [GOAL](https://arxiv.org/abs/2303.14655), [repo](https://github.com/THU-KEG/goal) · [SoccerReplay-1988](https://arxiv.org/abs/2412.01820) · [LFC](https://aclanthology.org/2025.inlg-main.13/) · [SoccerNet 2026](https://www.soccer-net.org/challenges/2026).

Models and speed: [Qwen3-1.7B-4bit](https://huggingface.co/mlx-community/Qwen3-1.7B-4bit) · [Apple MLX memory study](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) · [M1 llama.cpp benchmarks](https://github.com/ggml-org/llama.cpp/discussions/4167) · [M1 8 GB readings](https://llmcheck.net/benchmarks) · [format compliance at small sizes](https://arxiv.org/html/2606.08051) · [Qwen3 thinking leak](https://www.distillabs.ai/learn/qwen3-1-7b-fine-tuning-guide/).

Training and serving: [LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) · [SERVER.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md) · [prompt-cache-dir PR](https://github.com/ml-explore/mlx-lm/pull/1405) · [memory creep](https://github.com/ml-explore/mlx-lm/issues/828) · [PEFT-to-MLX PR](https://github.com/ml-explore/mlx-lm/pull/1120) · [RunPod pricing](https://www.runpod.io/pricing) · [Together pricing](https://www.together.ai/pricing).

Excitement: [Rui et al. 2000](https://dl.acm.org/doi/10.1145/354384.354443) · [Boril et al. 2010](https://www.isca-archive.org/interspeech_2010/boril10b_interspeech.html) · [audio-visual spotting](https://arxiv.org/abs/2011.04258) · [librosa RMS](https://librosa.org/doc/0.10.2/generated/librosa.feature.rms.html) · [yt-dlp](https://github.com/yt-dlp/yt-dlp).
