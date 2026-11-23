"""Pi 5 with experts streamed from SSD (reads only, no writes).

Only the always-used weights (attention, dense MLP, embeddings, routers)
stay in RAM. Each token the router picks 8 of 128 experts per layer; those
are read straight out of the original safetensors files with F_NOCACHE
(no page-cache bloat) into a small LRU cache.

usage: stream_pi5.py [--cache-mb N] [--tokens N] "prompt"
"""
import argparse, json, os, struct, sys, time, fcntl, subprocess
from collections import OrderedDict
from pathlib import Path

import numpy as np
from concurrent.futures import ThreadPoolExecutor
POOL = ThreadPoolExecutor(32)
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load
from mlx_lm.generate import stream_generate

MODEL = Path(os.environ.get("EXPERTSTREAM_MODEL", Path.home() / ".lmstudio/models/aran/1-Apex-Gemma4-26B"))
PROJ = ("gate_proj", "up_proj", "down_proj")
PART = ("weight", "scales", "biases")
DT = {"U32": (np.uint32, mx.uint32), "BF16": (np.uint16, mx.bfloat16)}


class Store:
    """Byte offsets of every expert slice, async SSD reads, LRU of loaded experts."""

    def __init__(self, cache_mb, model_dir=None):
        self.dir = Path(model_dir or MODEL)
        cfgq = json.load(open(self.dir / "config.json")).get("quantization", {})
        idx = json.load(open(self.dir / "model.safetensors.index.json"))["weight_map"]
        self.fds, hdrs = {}, {}
        for fn in set(idx.values()):
            fd = os.open(self.dir / fn, os.O_RDONLY)
            fcntl.fcntl(fd, 48, 1)  # F_NOCACHE: don't fill the page cache
            n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            hdrs[fn] = (json.loads(os.pread(fd, n, 8)), 8 + n)
            self.fds[fn] = fd
        self.loc, self.bits = {}, {}
        for k, fn in idx.items():
            if ".experts.switch_glu." not in k:
                continue
            L = int(k.split(".layers.")[1].split(".")[0])
            proj, part = k.split(".")[-2:]
            h, off0 = hdrs[fn]
            m = h[k]
            a, b = m["data_offsets"]
            self.loc[(L, proj, part)] = (self.fds[fn], off0 + a, (b - a) // m["shape"][0],
                                         m["shape"][1:], m["dtype"])
            q = cfgq.get(k.rsplit(".", 1)[0], cfgq)
            self.bits[(L, proj)] = (q.get("bits", 4), q.get("group_size", 64))
        self.cache = OrderedDict()
        self.fut = {}
        self.budget = cache_mb << 20
        self.used = 0
        self.hits = self.miss = self.pre = 0
        self.read_bytes = 0
        self.skip_tau, self.keep_top, self.skipped = 0.0, 2, 0

    def _read(self, L, e):
        out = []
        for p in PROJ:
            for q in PART:
                fd, base, stride, shape, dt = self.loc[(L, p, q)]
                out.append(os.pread(fd, stride, base + e * stride))
        return out

    def prefetch(self, L, es):
        for e in es:
            k = (L, e)
            if k not in self.cache and k not in self.fut:
                self.fut[k] = POOL.submit(self._read, L, e)

    def get(self, L, e):
        key = (L, e)
        c = self.cache.get(key)
        if c is not None:
            self.cache.move_to_end(key)
            self.hits += 1
            return c[0]
        f = self.fut.pop(key, None)
        if f is not None:
            self.pre += 1 if f.done() else 0
        else:
            f = POOL.submit(self._read, L, e)
        self.miss += 1
        bufs, out, size, i = f.result(), {}, 0, 0
        for p in PROJ:
            for q in PART:
                _, _, stride, shape, dt = self.loc[(L, p, q)]
                npdt, mxdt = DT[dt]
                a = mx.array(np.frombuffer(bufs[i], npdt).reshape(shape))
                if mxdt is mx.bfloat16:
                    a = a.view(mx.bfloat16)
                out[(p, q)] = a
                size += stride; i += 1
        self.read_bytes += size
        self.cache[key] = (out, size)
        self.used += size
        while self.used > self.budget:
            _, (_, s) = self.cache.popitem(last=False)
            self.used -= s
        return out


class PrefetchRouter(nn.Module):
    """Runs this layer's router and, on the same input, the NEXT layer's router,
    then starts reading the next layer's likely experts from SSD in the background."""

    def __init__(self, router, nxt, L, store):
        super().__init__()
        self.r, self._n, self._L, self._s = router, nxt, L, store

    def __call__(self, h):
        idx, w = self.r(h)
        if self._n is not None and h.shape[0] * h.shape[1] == 1:
            nidx, nw = self._n(h)
            mx.async_eval(idx, w)
            ni = np.array(nidx).reshape(-1); nwv = np.array(nw.astype(mx.float32)).reshape(-1)
            share = nwv / nwv.sum(); order = np.argsort(-share)
            st = self._s
            pick = [int(ni[k]) for r, k in enumerate(order) if r < st.keep_top or share[k] >= max(st.skip_tau, 1e-9)]
            st.prefetch(self._L + 1, pick)
        return idx, w


class StreamedExperts(nn.Module):
    def __init__(self, layer, store, act):
        super().__init__()
        self._L, self._s, self._act = layer, store, act

    def _qmm(self, h, W, p):
        bits, gs = self._s.bits[(self._L, p)]
        return mx.quantized_matmul(h, W[(p, "weight")], W[(p, "scales")], W[(p, "biases")],
                                   transpose=True, group_size=gs, bits=bits)

    def __call__(self, x, top_k_indices, top_k_weights):
        mx.eval(top_k_indices)
        ids = np.array(top_k_indices)
        uniq = np.unique(ids)
        if ids.shape[0] * ids.shape[1] == 1:            # decode: one token, 8 experts, no stacking
            wv = np.array(top_k_weights.astype(mx.float32)).reshape(-1)
            ev = ids.reshape(-1)
            keep = list(range(len(ev)))
            tau = self._s.skip_tau
            if tau > 0:   # cache-aware routing: skip a not-in-RAM expert if its share is small
                share = wv / wv.sum()
                order = np.argsort(-share)
                keep = [k for r, k in enumerate(order) if r < self._s.keep_top or (self._L, int(ev[k])) in self._s.cache or share[k] >= tau]
                self._s.skipped += len(ev) - len(keep)
            self._s.prefetch(self._L, [int(ev[k]) for k in keep])   # SSD reads in parallel
            scale = float(wv.sum() / wv[keep].sum())
            want = {int(ev[k]) for k in keep}
            for key in [k for k in self._s.fut if k[0] == self._L and k[1] not in want]:
                self._s.fut.pop(key).cancel()
            y = None
            for k in keep:
                e = ev[k]
                W = self._s.get(self._L, int(e))
                o = self._qmm(self._act(self._qmm(x, W, "up_proj"), self._qmm(x, W, "gate_proj")), W, "down_proj")
                o = o * (top_k_weights[..., k:k + 1] * scale)
                y = o if y is None else y + o
            return y
        # prompt reading: go expert by expert over the WHOLE chunk, so each expert
        # is read from SSD once per chunk and only one expert is in flight at a time
        shp = x.shape
        xf = x.reshape(-1, shp[-1])
        idf = ids.reshape(-1, ids.shape[-1])
        wf = top_k_weights.reshape(-1, ids.shape[-1])
        self._s.prefetch(self._L, [int(e) for e in uniq[:16]])
        out = mx.zeros(xf.shape, xf.dtype)
        for j, e in enumerate(uniq):
            if j + 16 < len(uniq):
                self._s.prefetch(self._L, [int(uniq[j + 16])])
            rows, ks = np.nonzero(idf == e)
            r = mx.array(rows.astype(np.uint32))
            W = self._s.get(self._L, int(e))
            xe = xf[r]
            ye = self._qmm(self._act(self._qmm(xe, W, "up_proj"), self._qmm(xe, W, "gate_proj")), W, "down_proj")
            out = out.at[r].add(ye * wf[r, mx.array(ks.astype(np.uint32))][:, None])
            if j % 8 == 7:
                mx.eval(out)
        return out.reshape(shp)


def stream_experts(model, cache_mb, model_dir=None, prefetch=True):
    """Swap every MoE layer's experts for SSD-streamed ones; returns the Store."""
    store = Store(cache_mb, model_dir)
    layers = model.language_model.model.layers
    routers = [getattr(l, "router", None) for l in layers]
    for i, layer in enumerate(layers):
        if getattr(layer, "enable_moe", False):
            layer.experts = StreamedExperts(i, store, layer.experts.switch_glu.activation)
            if prefetch:
                nxt = routers[i + 1] if i + 1 < len(layers) else None
                layer.router = PrefetchRouter(routers[i], nxt, i, store)
    if hasattr(model, "vision_tower"):
        model.vision_tower = None
    mx.eval(model.parameters())
    return store


def footprint_mb():
    out = subprocess.run(["footprint", str(os.getpid())], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "phys_footprint:" in line:
            v, unit = line.split(":")[1].split()[:2]
            return float(v) * (1024 if unit.startswith("G") else 1)
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-mb", type=int, default=1200)
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("prompt")
    a = ap.parse_args()

    mx.set_cache_limit(256 << 20)
    model, tok = load(str(MODEL), lazy=True)
    store = stream_experts(model, a.cache_mb)
    print(f"[resident weights loaded] footprint {footprint_mb():.0f} MB", flush=True)

    msgs = [{"role": "user", "content": a.prompt}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True)
    t0, n, peak = time.time(), 0, 0
    for r in stream_generate(model, tok, p, max_tokens=a.tokens, prefill_step_size=32):
        sys.stdout.write(r.text); sys.stdout.flush()
        n += 1
        if n % 20 == 0:
            peak = max(peak, footprint_mb())
    dt = time.time() - t0
    peak = max(peak, footprint_mb())
    hr = store.hits / max(1, store.hits + store.miss)
    print(f"\n\n[stats] {n} tok in {dt:.1f}s = {n/dt:.2f} tok/s | peak footprint {peak:.0f} MB"
          f" | mlx peak {mx.get_peak_memory()/2**20:.0f} MB | cache hit {hr:.0%}"
          f" | SSD read {store.read_bytes/2**30:.1f} GB (reads only)")


if __name__ == "__main__":
    main()
