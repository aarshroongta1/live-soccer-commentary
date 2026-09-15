# Open-weights emotional TTS for live soccer commentary

## 1. Candidates

| Model | Emotion control | Streaming / TTFA | Apple Silicon | Size | License | Quality (AA Elo) |
|---|---|---|---|---|---|---|
| **Chatterbox Turbo** (Resemble) | `exaggeration` scalar 0–1 plus tags `[gasp] [laugh] [whisper]` | Vendor claims 75 ms, 6x realtime on one GPU; upstream repo documents **no** streaming. MIT fork adds `generate_stream()`, 0.47 s first chunk, RTF 0.499 on RTX 4090 (CUDA only) | `mps` device in repo; mlx-audio ships Chatterbox but lists streaming = No; community MPS port reports 2–3x speedup, MPS tensor bugs push many to CPU | 350M (Nano 110M, base 500M) | MIT | 1020 (base Chatterbox) |
| **Qwen3-TTS** (Alibaba) | Natural-language instruction per line (CustomVoice), free-form VoiceDesign | Native streaming, claims 97 ms first packet | mlx-audio supports it **with streaming**; no Mac latency published | 0.6B / 1.7B | Apache 2.0 | closed Qwen-Audio-3.0-TTS-Plus 1235; open weights unrated |
| **Kyutai Pocket TTS** | None documented | Streaming, ~200 ms first chunk | **~6x realtime on MacBook Air M4 CPU, 2 cores**; MLX port exists | 100M | MIT | not rated |
| **Kyutai TTS 2B** | None documented | Text-in + audio-out streaming, 220 ms GPU | MLX impl; 1B tested on iPhone 16 Pro | 2B | CC-BY-4.0 weights | not rated |
| **IndexTTS 2.5** (bilibili) | Best in class: 8-float emotion vector, `emo_alpha`, emotion-from-text | No streaming documented; RTF 0.21 on 4090 | Not supported | 0.8B | bilibili license, **written authorization for commercial use** | not rated |
| **Higgs Audio v2 / v3** | Inline emotion/style/prosody; 75.7% win vs gpt-4o-mini-tts on EmergentTTS Emotions | Streaming experimental; 1.3x realtime on 4090 | mlx-audio has v3, streaming = No | 3–4B, 5.8 GB fp16 | Apache 2.0 (v2) | 1038 (v3) |
| **CosyVoice 2 / 3** | Instruct mode: emotion, speed, volume | Bi-streaming, 150 ms first packet | None | 0.5B | Apache 2.0 | not rated |
| **Orpheus 3B** | Tags `<laugh> <sigh> <gasp>` | ~200 ms, ~100 ms with input streaming | No native path; GGUF in LM Studio on Metal | 3B | Apache 2.0 | not rated |
| **Breeze TTS 2** | Voice design + `(laugh)` vocal events | <40 ms TTFA on H100 | Not found | 3B | Code Apache 2.0, **weights research/non-commercial** | **1203, top open-weights** |
| **Fish OpenAudio S1** | Emotion/tone markers | Streaming | Not found | S1-mini | **CC-BY-NC-SA-4.0** | 1042 mini |
| **Kokoro 82M** (baseline) | None, speed only | No streaming in mlx-audio | 178 ms for 4 words on M4 Max (MetalRT), 493 ms via mlx-audio; CoreML 12–79x realtime | 82M | Apache 2.0 | 1061 |
| **VibeVoice** | None explicit | Realtime 0.5B ~300 ms first audio | mlx-audio listed | 0.5B / 1.5B | MIT, but Microsoft pulled TTS code from the repo | not rated |

## 2. Shortlist

**1. Qwen3-TTS-12Hz-0.6B-CustomVoice.** The only candidate that pairs a clean Apache 2.0 license with native streaming *and* per-line natural-language style control, which maps directly onto a director that wants "shout this one". mlx-audio supports it with streaming on Apple Silicon, and nine built-in speakers give you the caller and the analyst without cloning. Apple Silicon latency: **not found**, only Alibaba's 97 ms end-to-end claim. Risk: that claim is a GPU number, and the MLX path may be several times slower.

**2. Chatterbox Turbo.** The exaggeration scalar is the closest thing in open weights to a goal-excitement knob, it is MIT, and 350M is small enough to be plausible on an M-series chip. Risk is the biggest of the three: upstream ships no streaming, the streaming fork is CUDA-tested, and mlx-audio's table says Chatterbox does not stream. Without streaming you cannot cut the analyst off mid-word. Apple Silicon latency: **not found**.

**3. Kyutai Pocket TTS.** The only model with a verified Mac number: about 6x realtime on a MacBook Air M4 using two CPU cores, roughly 200 ms to first chunk, MIT, 26 voices plus cloning. Risk: no emotion control at all, so a goal would sound the same as a throw-in. It is a much better `say`, not an excited caller.

## 3. Recommendation

Try **Qwen3-TTS 0.6B CustomVoice through mlx-audio** first on this Mac. It is the only option that satisfies streaming, per-line steerability, and a permissive license at once, and measuring its real time-to-first-audio on your hardware is a half-day experiment that settles the whole question. Keep **Chatterbox Turbo** as the excitement specialist to A/B on goal lines only, where a 400 ms non-streaming burst is tolerable because the director is cutting everything else off anyway. If a cloud GPU is acceptable, run **Chatterbox Turbo on a 4090 or L4** with the streaming fork, or **CosyVoice 3** if you want instruction control with a documented 150 ms first packet. On goal excitement specifically, nothing here clearly beats ElevenLabs: Eleven v3 sits at 1168 Elo and v3 Conversational at 1197, while the best commercially usable open weights are Fish S2 Pro at 1121 and Step-Audio-EditX at 1100. The one open model that does beat Eleven v3, Breeze TTS 2 at 1203, has non-commercial weights that block a public demo. Expect to trade some vocal quality for control, cost, and local execution, not to gain quality.

## 4. Sources

- https://github.com/resemble-ai/chatterbox
- https://www.resemble.ai/learn/models/chatterbox-turbo
- https://github.com/davidbrowne17/chatterbox-streaming
- https://huggingface.co/Jimmi42/chatterbox-tts-apple-silicon-code
- https://github.com/QwenLM/Qwen3-TTS
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
- https://github.com/Blaizzy/mlx-audio
- https://blaizzy.github.io/mlx-audio/models/tts/
- https://github.com/kyutai-labs/pocket-tts
- https://github.com/kyutai-labs/delayed-streams-modeling/
- https://kyutai.org/tts/
- https://github.com/index-tts/index-tts
- https://github.com/index-tts/index-tts/issues/228
- https://github.com/boson-ai/higgs-audio
- https://github.com/FunAudioLLM/CosyVoice
- https://github.com/canopyai/Orpheus-TTS
- https://github.com/isaiahbjork/orpheus-tts-local
- https://github.com/stepfun-ai/Step-Audio-EditX
- https://huggingface.co/BreezeBlue/Breeze-TTS-2
- https://www.stork.ai/blog/this-ai-beats-elevenlabs-dont-use-it
- https://huggingface.co/fishaudio/openaudio-s1-mini
- https://github.com/microsoft/VibeVoice
- https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice
- https://www.runanywhere.ai/blog/metalrt-speech-fastest-stt-tts-apple-silicon
- https://github.com/5uck1ess/tts-bench
