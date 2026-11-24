"""Pi 5 with experts streamed from SSD (reads only, no writes).

Only the always-used weights (attention, dense MLP, embeddings, routers)
stay in RAM. Each token the router picks 8 of 128 experts per layer; those
are read straight out of the original safetensors files with F_NOCACHE
(no page-cache bloat) into a small LRU cache.

usage: expertstream.py [--cache-mb N] [--tokens N] "prompt"
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
STACK_DECODE = os.environ.get("ES_STACK", "0") == "1"
COMPILE = os.environ.get("ES_COMPILE", "1") == "1"      # one pre-built GPU program per layer instead of ~24 small jobs
_COMPILED = {}
PART = ("weight", "scales", "biases")
DT = {"U32": (np.uint32, mx.uint32), "BF16": (np.uint16, mx.bfloat16)}


class Store:
    """Byte offsets of every expert slice, async SSD reads, LRU of loaded experts."""

    def __init__(self, cache_mb, model_dir=None):
        self.dir = Path(model_dir or MODEL)
        cfgq = json.load(open(self.dir / "config.json")).get("quantization", {})
        ip = self.dir / "model.safetensors.index.json"
        if ip.exists():
            idx = json.load(open(ip))["weight_map"]
        else:                                   # no index: read every shard's header
            idx = {}
            for fp in sorted(self.dir.glob("*.safetensors")):
                with open(fp, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    idx.update({k: fp.name for k in json.loads(f.read(n)) if k != "__metadata__"})
        self.fds, hdrs = {}, {}
        for fn in set(idx.values()):
            fd = os.open(self.dir / fn, os.O_RDONLY)
            fcntl.fcntl(fd, 48, 1)  # F_NOCACHE: don't fill the page cache
            n = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            hdrs[fn] = (json.loads(os.pread(fd, n, 8)), 8 + n)
            self.fds[fn] = fd
        self.loc, self.bits, self.parts = {}, {}, {}
        for k, fn in idx.items():
            if ".experts.switch_glu." not in k and ".mlp.experts." not in k and ".switch_mlp." not in k:
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
            self.parts.setdefault((L, proj), [])
            if part not in self.parts[(L, proj)]:
                self.parts[(L, proj)].append(part)
            self.n_exp = m["shape"][0]
        self.cache = OrderedDict()
        self.fut = {}
        self.budget = cache_mb << 20
        self.used = 0
        self.hits = self.miss = self.pre = 0
        self.read_bytes = 0
        self.skip_tau, self.keep_top, self.skipped = 0.0, 2, 0
        self.ghost_dir = os.environ.get("ES_GHOSTS", "")
        bf = os.environ.get("ES_BUDGET_FILE")          # Rivet writes the live expert-cache budget (MB) here; we follow it
        if bf and os.environ.get("ES_ARENA") != "1":   # in arena mode this cache only serves prompt reading: keep it small
            import threading
            def _follow():
                while True:
                    try:
                        mb = int(float(open(bf).read().strip()))
                        if mb > 0 and abs((mb << 20) - self.budget) > (64 << 20):
                            self.budget = mb << 20
                            while self.used > self.budget and self.cache:
                                _, (_, sz) = self.cache.popitem(last=False); self.used -= sz
                    except Exception: pass
                    time.sleep(2)
            threading.Thread(target=_follow, daemon=True).start()
        self.ghosts, self.ghosted = {}, 0

    def _read(self, L, e):
        out = []
        for p in PROJ:
            for q in self.parts[(L, p)]:
                fd, base, stride, shape, dt = self.loc[(L, p, q)]
                out.append(os.pread(fd, stride, base + e * stride))
        return out

    def ghost(self, L):
        """rank-R sketches of every expert in layer L, kept on the GPU (fp16)."""
        g = self.ghosts.get(L)
        if g is None and self.ghost_dir:
            f = os.path.join(self.ghost_dir, f"L{L:02d}.npz")
            if os.path.exists(f):
                z = np.load(f)
                g = {p: tuple(mx.array(z[f"{p}.{n}"].astype(np.float16) * z[f"{p}.{n}s"]) for n in "AB") for p in PROJ}
                mx.eval([a for t in g.values() for a in t])
                self.ghosts[L] = g
        return g

    def prefetch(self, L, es):
        for e in es:
            k = (L, e)
            if k not in self.cache and k not in self.fut:
                self.fut[k] = POOL.submit(self._read, L, e)

    def settle(self):
        """move finished background reads into the cache (used with ghosts)."""
        for key, f in list(self.fut.items()):
            if f.done():
                self.get(*key); self.hits -= 0

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
            for q in self.parts[(L, p)]:
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
        y = mx.quantized_matmul(h, W[(p, "weight")], W[(p, "scales")], W[(p, "biases")],
                                transpose=True, group_size=gs, bits=bits)
        b = W.get((p, "bias"))
        return y if b is None else y + b

    def __call__(self, x, top_k_indices, top_k_weights):
        mx.eval(top_k_indices)
        ids = np.array(top_k_indices)
        uniq = np.unique(ids)
        if ids.shape[0] * ids.shape[1] == 1:            # decode: one token, 8 experts, no stacking
            wv = np.array(top_k_weights.astype(mx.float32)).reshape(-1)
            ev = ids.reshape(-1)
            keep = list(range(len(ev)))
            G = self._s.ghost(self._L)
            if G is not None:
                self._s.settle()   # ghost experts: a miss is answered from the in-RAM sketch now; the real one loads for later
                self._s.prefetch(self._L, [int(e) for e in ev])
                y = None
                for k, e in enumerate(ev):
                    e = int(e)
                    if (self._L, e) in self._s.cache:
                        W = self._s.get(self._L, e)
                        o = self._qmm(self._act(self._qmm(x, W, "up_proj"), self._qmm(x, W, "gate_proj")), W, "down_proj")
                    else:
                        self._s.ghosted += 1
                        gl = lambda h, p: (h @ G[p][1][e].T) @ G[p][0][e].T
                        o = gl(self._act(gl(x, "up_proj"), gl(x, "gate_proj")), "down_proj").astype(x.dtype)
                    o = o * top_k_weights[..., k:k + 1]
                    y = o if y is None else y + o
                return y
            tau = self._s.skip_tau
            if tau > 0:   # cache-aware routing: skip a not-in-RAM expert if its share is small
                share = wv / wv.sum()
                order = np.argsort(-share)
                keep = [k for r, k in enumerate(order) if r < self._s.keep_top or (self._L, int(ev[k])) in self._s.cache or share[k] >= tau]
                self._s.skipped += len(ev) - len(keep)
            self._s.prefetch(self._L, [int(ev[k]) for k in keep])   # SSD reads in parallel
            if STACK_DECODE:                                 # one batched GPU job per projection instead of 3 per expert
                Ws = [self._s.get(self._L, int(ev[k])) for k in keep]
                W = {key: mx.stack([w[key] for w in Ws]) for key in Ws[0]}
                idx = mx.arange(len(keep), dtype=mx.uint32).reshape(1, 1, -1)
                wk = top_k_weights[..., mx.array([int(k) for k in keep])] * float(wv.sum() / wv[keep].sum())
                def lin(h, p):
                    bits, gs = self._s.bits[(self._L, p)]
                    y = mx.gather_qmm(h, W[(p, "weight")], W[(p, "scales")], W[(p, "biases")],
                                      rhs_indices=idx, transpose=True, group_size=gs, bits=bits)
                    b = W.get((p, "bias"))
                    return y if b is None else y + mx.expand_dims(b, -2)
                h = mx.expand_dims(x, (-2, -3))
                y = lin(self._act(lin(h, "up_proj"), lin(h, "gate_proj")), "down_proj").squeeze(-2)
                return (mx.expand_dims(wk, -1) * y).sum(-2)
            scale = float(wv.sum() / wv[keep].sum())
            want = {int(ev[k]) for k in keep}
            for key in [k for k in self._s.fut if k[0] == self._L and k[1] not in want]:
                self._s.fut.pop(key).cancel()
            y = None
            if COMPILE and "bias" not in str(self._s.parts.get((self._L, "up_proj"), [])):
                Ws = [self._s.get(self._L, int(ev[k])) for k in keep]
                flat = []
                for W in Ws:
                    for pj in PROJ: flat += [W[(pj, "weight")], W[(pj, "scales")], W[(pj, "biases")]]
                wk = top_k_weights.reshape(-1)[mx.array([int(k) for k in keep])] * scale
                key = (len(keep), self._s.bits[(self._L, "up_proj")], self._s.bits[(self._L, "down_proj")], id(self._act))
                fn = _COMPILED.get(key)
                if fn is None:
                    bu, gu = self._s.bits[(self._L, "up_proj")]; bd, gd = self._s.bits[(self._L, "down_proj")]; act = self._act; n = len(keep)
                    def f(x, wk, *w):
                        y = None
                        for i in range(n):
                            g = w[9 * i: 9 * i + 9]
                            q = lambda h, a, b, c, bb, gg: mx.quantized_matmul(h, a, b, c, transpose=True, group_size=gg, bits=bb)
                            hg = q(x, g[0], g[1], g[2], bu, gu); hu = q(x, g[3], g[4], g[5], bu, gu)
                            o = q(act(hu, hg), g[6], g[7], g[8], bd, gd) * wk[i]
                            y = o if y is None else y + o
                        return y
                    fn = _COMPILED[key] = mx.compile(f)
                return fn(x, wk, *flat)
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


class StreamedOssMLP(nn.Module):
    """gpt-oss MLP block: router -> top-k -> softmax -> streamed experts."""

    def __init__(self, mlp, L, store):
        super().__init__()
        self.router, self._k = mlp.router, mlp.num_experts_per_tok
        self.experts = StreamedExperts(L, store, mlp.experts.activation)

    def __call__(self, x):
        g = self.router(x)
        inds = mx.argpartition(-g, kth=self._k - 1, axis=-1)[..., :self._k]
        vals = mx.take_along_axis(g, inds, axis=-1)
        w = mx.softmax(vals, axis=-1, precise=True)
        return self.experts(x, inds, w)


class StreamedQwenMoE(nn.Module):
    """Qwen 3.5 / Qwen3-Next sparse MoE: softmax router -> top-k -> streamed experts + resident shared expert."""

    def __init__(self, mlp, L, store):
        super().__init__()
        self.gate = mlp.gate
        self._shared = hasattr(mlp, "shared_expert")            # Qwen 3.5 has one; Qwen3 / Qwen3-Coder do not
        if self._shared: self.shared_expert, self.shared_expert_gate = mlp.shared_expert, mlp.shared_expert_gate
        self._k, self._norm = mlp.top_k, mlp.norm_topk_prob
        self.experts = StreamedExperts(L, store, mlp.switch_mlp.activation)

    def __call__(self, x):
        g = mx.softmax(self.gate(x), axis=-1, precise=True)
        inds = mx.argpartition(g, kth=-self._k, axis=-1)[..., -self._k:]
        sc = mx.take_along_axis(g, inds, axis=-1)
        if self._norm:
            sc = sc / sc.sum(axis=-1, keepdims=True)
        y = self.experts(x, inds, sc)
        return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x) if self._shared else y


def _layers(model):
    for path in ("language_model.model.layers", "model.layers", "language_model.layers"):
        o = model
        try:
            for p in path.split("."): o = getattr(o, p)
            return o
        except AttributeError:
            continue
    raise RuntimeError("can't find decoder layers")


def stream_experts(model, cache_mb, model_dir=None, prefetch=True):
    """Swap every MoE layer's experts for SSD-streamed ones (Gemma 4, gpt-oss, Qwen 3.5); returns the Store."""
    store = Store(cache_mb, model_dir)
    layers = _layers(model)
    routers = [getattr(l, "router", None) for l in layers]
    for i, layer in enumerate(layers):
        if getattr(layer, "enable_moe", False):                          # Gemma 4
            layer.experts = StreamedExperts(i, store, layer.experts.switch_glu.activation)
            if prefetch:
                nxt = routers[i + 1] if i + 1 < len(layers) else None
                layer.router = PrefetchRouter(routers[i], nxt, i, store)
        elif hasattr(getattr(layer, "mlp", None), "switch_mlp"):           # Qwen 3.5 MoE
            layer.mlp = StreamedQwenMoE(layer.mlp, i, store)
        elif hasattr(getattr(layer, "mlp", None), "experts"):              # gpt-oss
            layer.mlp = StreamedOssMLP(layer.mlp, i, store)
    for attr in ("vision_tower", "visual"):
        if hasattr(model, attr): setattr(model, attr, None)
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
