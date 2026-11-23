# 🚀 A 26B MODEL IN UNDER 5 GB OF RAM. ZERO SWAP. ON A LAPTOP. 🚀

**ExpertStream** runs Gemma 4 26B-A4B (4-bit MLX, ~15.6 GB of weights) on a 24 GB MacBook while the process uses **~4.7 GB of RAM**, **adds no swap**, and **writes nothing to the SSD**.

| | Stock (LM Studio / mlx-lm) | ExpertStream |
|---|---|---|
| RAM while running | ~16 GB + 8–10 GB swap | **4.7 GB** |
| SSD writes | GBs of swap | **0** |
| Prompt reading | fast | ~185 tok/s |
| Writing | fast | ~9–10 tok/s (M5, 24 GB) |
| Next-token agreement with full model | 100% | ~98% |

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
