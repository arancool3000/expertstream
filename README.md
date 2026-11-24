# ⚡ 26B MODEL, 26 TOK/S, UNDER 10 GB OF RAM, ON A LAPTOP. ⚡

**ExpertStream** runs big mixture-of-experts models (Gemma 4 26B, Qwen 3.5 35B, gpt-oss 20B) on a 24 GB MacBook
with a **fraction of the RAM**, **no swap**, and **no SSD writes**.

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
