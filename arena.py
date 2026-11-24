"""Zero-copy expert arena: one uninterrupted GPU job per word.

Each MoE layer owns S expert slots that live in GPU memory. The SSD reads experts *straight into*
those slots (Apple's CPU and GPU share memory, so a numpy view of an MLX array is the GPU buffer).
The GPU looks up which slot holds each routed expert itself, so a whole word is computed without
the 30-40 CPU<->GPU stops of the streamed engine. Experts that aren't in a slot yet are left out of
that word (weights renormalised) and loaded before the next word.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import expertstream as SP
from expertstream import PROJ, DT, POOL


class Arena:
    def __init__(self, store, layers, budget_mb):
        self.s, self.layers = store, layers
        L0 = layers[0]
        per = sum(store.loc[(L0, p, q)][2] for p in PROJ for q in store.parts[(L0, p)])
        self.S = max(8, min(store.n_exp, int(budget_mb * 2**20 / len(layers) / per)))
        self.buf, self.view = {}, {}
        for L in layers:
            for p in PROJ:
                for q in store.parts[(L, p)]:
                    _, _, stride, shape, dt = store.loc[(L, p, q)]
                    a = mx.zeros([self.S] + list(shape), DT[dt][1] if DT[dt][1] is not mx.bfloat16 else mx.uint16)
                    mx.eval(a)
                    self.buf[(L, p, q)] = a if DT[dt][1] is not mx.bfloat16 else a      # bf16 stored as raw uint16
                    self.view[(L, p, q)] = np.array(a, copy=False).reshape(self.S, -1).view(np.uint8)
        self.slot_np = {L: np.full(store.n_exp, -1, np.int32) for L in layers}
        self.slot_mx = {L: mx.array(self.slot_np[L]) for L in layers}
        for L in layers:
            mx.eval(self.slot_mx[L]); self.slot_np[L] = np.array(self.slot_mx[L], copy=False)
        self.owner = {L: np.full(self.S, -1, np.int64) for L in layers}
        self.last = {L: np.zeros(self.S) for L in layers}
        self.rec, self.step = [], 0
        self.used = self.dropped = self.loads = 0
        print(f"[arena] {self.S} expert slots per layer x {len(layers)} layers ({self.S * per * len(layers) / 2**30:.1f} GB)", flush=True)

    def _load(self, L, e, s):
        """pread one expert straight into GPU slot s of layer L."""
        for p in PROJ:
            for q in self.s.parts[(L, p)]:
                fd, base, stride, shape, dt = self.s.loc[(L, p, q)]
                os.preadv(fd, [memoryview(self.view[(L, p, q)][s])], base + e * stride)

    def install(self, todo):
        """todo: list of (L, e). Pick LRU victims, read all in parallel, then publish the slot table."""
        jobs = []
        for L, e in todo:
            if self.slot_np[L][e] >= 0: continue
            s = int(np.argmin(self.last[L]))
            old = self.owner[L][s]
            if old >= 0: self.slot_np[L][old] = -1
            self.owner[L][s] = e; self.last[L][s] = self.step + 0.5
            jobs.append((L, e, s))
        list(POOL.map(lambda j: self._load(*j), jobs))
        for L, e, s in jobs: self.slot_np[L][e] = s          # publish only after the bytes are in place
        self.loads += len(jobs)

    def after_word(self):
        self.step += 1
        need = []
        for L, idx in self.rec:
            ids = np.array(idx).reshape(-1)
            sl = self.slot_np[L][ids]
            for e, s in zip(ids, sl):
                if s >= 0: self.last[L][s] = self.step; self.used += 1
                else: self.dropped += 1; need.append((L, int(e)))
        self.rec = []
        if need: self.install(list(dict.fromkeys(need)))


class ArenaExperts(nn.Module):
    def __init__(self, L, arena, streamed):
        super().__init__()
        self._L, self._a, self._st = L, arena, streamed

    def __call__(self, x, idx, w):
        a, L = self._a, self._L
        if x.shape[0] * x.shape[1] != 1:                     # prompt reading: old streaming path, then warm the arena
            a.rec.append((L, idx[:, -1:]))
            return self._st(x, idx, w)
        s = a.slot_mx[L][idx]
        ok = s >= 0
        ww = mx.where(ok, w, 0)
        ww = ww * (w.sum(-1, keepdims=True) / mx.maximum(ww.sum(-1, keepdims=True), 1e-6))
        si = mx.maximum(s, 0).astype(mx.uint32)
        a.rec.append((L, idx))

        def lin(h, p):
            bits, gs = a.s.bits[(L, p)]
            sc, bi = a.buf[(L, p, "scales")], a.buf[(L, p, "biases")]
            if sc.dtype == mx.uint16: sc, bi = sc.view(mx.bfloat16), bi.view(mx.bfloat16)
            return mx.gather_qmm(h, a.buf[(L, p, "weight")], sc, bi, rhs_indices=si, transpose=True, group_size=gs, bits=bits)

        h = mx.expand_dims(x, (-2, -3))
        y = lin(self._st._act(lin(h, "up_proj"), lin(h, "gate_proj")), "down_proj").squeeze(-2)
        return (mx.expand_dims(ww, -1) * y).sum(-2)


def arena_model(model, budget_mb, model_dir=None, prompt_cache_mb=300):
    store = SP.stream_experts(model, prompt_cache_mb, model_dir, prefetch=False)
    layers = SP._layers(model)
    moe = [i for i, l in enumerate(layers) if isinstance(getattr(l, "experts", None), SP.StreamedExperts)]
    if not moe: raise RuntimeError("arena: only Gemma-style experts supported for now")
    ar = Arena(store, moe, budget_mb)
    for i in moe: layers[i].experts = ArenaExperts(i, ar, layers[i].experts)
    cls = type(model); base = cls.__call__
    def call(self, inputs, *a, **k):
        out = base(self, inputs, *a, **k)
        mx.eval(out)
        ar.after_word()
        return out
    cls.__call__ = call
    return store, ar
