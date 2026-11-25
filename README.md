# ⚡ A 120B MODEL IN 11 GB OF RAM ON A 24 GB LAPTOP, EXACT OUTPUT. 35B MOE AT 22 TOK/S. ⚡

**ExpertStream** runs big mixture-of-experts models (Gemma 4 26B, Qwen 3.5 35B, gpt-oss 20B) on a 24 GB MacBook
with a **fraction of the RAM**, **no swap**, and **no SSD writes**.

## New: arena3 now runs every MoE model exactly, including gpt-oss 120B

arena3 now supports experts with bias terms (gpt-oss), so all of these run on one engine. Measured on a 24 GB MacBook (Apple M5), teacher-forced token agreement against the unmodified model, 2 prompts x 200 tokens:

| Model | RAM (peak) | Agreement | Speed |
|---|---|---|---|
| gpt-oss 120B (3-bit, 51 GB on disk) | 11.3 GB | identical (exact mode) | 6.5-8.6 tok/s |
| Gemma 4 26B (4-bit) | 7.7 GB | 397/400 (99.3%) | 11 tok/s (Low Power Mode) |
| gpt-oss 20B | 7.0 GB | 400/400 | 14 tok/s (Low Power Mode) |
| Qwen 3.5 35B-A3B | ~10 GB | 160/160 | 22 tok/s |

**A 120B-parameter model in 11 GB of RAM, with exact output, on a laptop.**

### Lesson: never silently drop experts
The first arena skipped any expert that wasn't already loaded. On Gemma 4 26B that matched the real model only **92.5%** of the time, about one wrong token in 13, and in real chats it showed up as stutters and broken words ("the the", "nons-ensical"). arena3 always waits for important experts and only skips ones with a routing weight under 3% (`ES_A3_TAU=0.03`); that is what brings agreement to 99-100%.

### What limits gpt-oss 120B
Its experts are ~10 MB each and ~57 are missing per token, so each token waits ~100 ms for the SSD at its full ~5.6 GB/s. Guessing the next layer's experts and reading them early made it **slower** (wrong guesses waste the same bandwidth). Skipping experts above a 3% weight broke the output (loops). It is bandwidth-bound, honestly.

## New: arena3, the MoE model is 2x faster at the same accuracy

Qwen 3.5 35B-A3B, with its experts streamed from the SSD and RAM capped at 10 GB, on a MacBook:

| Engine | tok/s | Agreement with the full model | Peak RAM |
|---|---|---|---|
| Streamed engine | 10.6-11.0 | 159/160 | 9.2 GB |
| **arena3** | **20.7-22.4** | **160/160** | 10.0 GB |

Agreement is teacher-forced over 160 tokens. Swap did not grow.

How it works (`arena3.py`):
- **One short GPU stop per layer** to learn the routed experts. It costs little: with every expert present it still runs at 41 tok/s.
- **Experts already in the arena** run in place as 3 `gather_qmm` kernels instead of 24 small ones.
- **Missing experts with real weight** are read from the SSD straight into a GPU slot, 9 tensor parts in parallel (0.46 ms each), and the engine waits for them.
- **Only low-weight misses are skipped** (share < 0.15, at most 25% of a layer). They are loaded the second time they are wanted. A high-weight expert is never dropped.

Knobs: `ES_A3_TAU`, `ES_A3_BG`, `ES_A3_MAXDROP`. Usage: `arena3.arena_model(model, 7600, model_dir)`.

Tried and dropped:
- **A fused Metal kernel for all 8 experts:** only 1.5-1.8x at full accuracy; the kernel was never the bottleneck.
- **Background or look-ahead installs:** 75% of the next layer's experts guessed right, but the speed stalls for seconds at a time.
- **Smarter slot replacement** (frequency plus recency): about 0.5 points fewer misses than LRU.

## New: the zero-copy expert arena (2.4x faster)

The old streaming engine stopped the GPU **30-40 times per word**: the GPU picks experts, the CPU fetches them, the GPU continues.

On Apple Silicon the CPU and GPU share one memory. So `arena.py` gives each layer a block of **expert slots that live in GPU memory**,
and the SSD reads experts **straight into those slots** (`os.preadv` into a numpy view of an MLX array, no copies).
The GPU looks up which slot holds each routed expert **itself**, so a whole word is **one uninterrupted GPU job**.
An expert that isn't in a slot yet is left out of that word (weights renormalised) and loaded before the next.

| Gemma 4 26B (4-bit), M5 MacBook 24 GB | old streaming | arena |
|---|---|---|
| Speed (standalone) | ~10-12 tok/s | **26.5 tok/s** |
| Speed (real chat app, words/s) | ~7 | **~16-23** |
| Next-token agreement with the full model | 98% | 98% |
| RAM | ~4.4-11 GB | 8-9 GB |

## What we measured that did NOT help (so you don't have to try)

- **Storing each expert contiguously** on disk: scattered reads already hit **5.6 GB/s**; contiguous was 0.8x.
  Once the cache is big, the SSD is not the bottleneck; the per-layer CPU<->GPU stop is.
- **Bigger expert cache alone**: 4x the cache gave no speed-up for Gemma (the stops dominate).
- **Speculative / prompt-lookup decoding**: verifying 8 guessed tokens costs **~14x one token** on a streamed MoE
  (each extra token pulls different experts). A small draft model made it 2.6x *slower*.
- **Low-rank "ghost" experts** for cache misses: experts are near full-rank (rank 256 keeps only 79%). Useless.
- `mx.compile` of the per-expert maths: +3%.

## Still works

- Cache-aware routing: skip a not-in-RAM expert with a small routing share (top 2 always kept).
- Expert-by-expert prompt reading in big chunks (each expert read once per chunk).
- Adaptive RAM: the expert cache follows a live budget file, so "macOS + model" can stay under a fixed total.
- `serve.py`: OpenAI-compatible server; set `ES_ARENA=1` for the arena (Gemma-style experts).

## How it works

Gemma 4 26B is a mixture-of-experts model: 30 layers × 128 experts, and each token only uses 8 experts per layer.

1. **Keep only the always-used weights in RAM** (attention, routers, dense MLP, embeddings: ~2.2 GB).
2. **Read experts straight from the original `.safetensors` files** with `pread` + `F_NOCACHE` (no page-cache bloat), 32 reads in parallel. Reads don't wear an SSD; swap writes do.
3. **LRU expert cache** (~2.3 GB) so recently used experts stay in RAM.
4. **Cache-aware routing (the speed trick):** if an expert the router picked is *not* in RAM and has a small share of the routing weight (< 15%), skip it and renormalise the rest. The top 2 experts are always used. This cut SSD reads ~35% and doubled speed, at ~98% top-1 agreement with the full model.
5. **Expert-by-expert prompt reading:** for long prompts, each expert is read once per 1,024-token chunk instead of once per tiny chunk (16× faster prefill).

It's in the same family as Apple's "LLM in a flash" and MoE-offloading work; the contribution here is a small, practical drop-in for **mlx-lm** with an OpenAI-compatible server.

## Run it

Needs Apple Silicon, `mlx` and `mlx-lm` with Gemma 4 support, and an MLX 4-bit Gemma 4 26B folder.

```bash
python expertstream.py --cache-mb 2300 "Explain why the sky is blue"
python serve.py   # OpenAI-compatible server on 127.0.0.1:1235
```

Knobs: `PI5_CACHE_MB` (RAM vs speed), `PI5_SKIP_TAU` (0 = exact, 0.15 = default).

## Limits (honest)

- Speed is ~10 tok/s, not the 40+ you get with everything in RAM.
- Text only (vision tower dropped to save RAM).
- Skipping experts is an approximation: measured ~98% next-token agreement on a coding prompt, not a full benchmark suite.
- Only tested on Gemma 4 26B-A4B on one M5 MacBook.

MIT licence. Model weights are not included.
