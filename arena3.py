"""Arena v3 (agent A): zero-copy GPU expert arena for a big MoE that does NOT fit in RAM (Jetson: 256 experts/layer).

What it does, per decoded word and per layer:
  1. ONE small CPU<->GPU stop: learn which 8 experts this layer's router picked, and their weights.
  2. Experts already in a GPU slot: nothing to do.  A missing expert with a meaningful share of the layer's weight
     is read from SSD straight into a GPU slot (numpy view of the MLX buffer = same unified memory; its 9 tensor
     parts on 9 readers at once, one of them the waiting thread itself) and the layer waits for it.
  3. A missing expert with a small share (< ES_A3_TAU, and never more than ES_A3_MAXDROP of the layer in total) is
     left out of THIS word (the others are re-weighted).  What happens next is ES_A3_BG:
       4 (default)  the second time it is wanted it is loaded and waited for like any other  -> ~7 % of routed experts skipped
       0            it stays out until its share reaches tau                                 -> ~17 % skipped, fastest
       1            it is loaded in the background during the word (fast in bursts, but background SSD reads make
                    the decode thread + GPU stops 3x slower for seconds at a time; not recommended)
       2 / 3        loaded right after the word while the GPU is idle (3: only if skipped twice)
     ES_A3_TAU=0 skips nothing: output identical to the exact streaming engine (160/160 on the reference).
  4. The 8 experts run as 3 gather_qmm kernels that read the slots in place (no copies, not 24 small kernels).

Why writes into the slots are safe: a slot is unmapped from the table the moment it is chosen as a victim and is
only re-published once all its bytes are in; a layer's graph is built from the table after that layer's own stop,
using only published slots, so the GPU never reads a slot that is being written.

Knobs (env):  ES_A3_TAU (0.15; 0 = exact) · ES_A3_MAXDROP (0.25) · ES_A3_BG (4) · ES_A3_USPLIT (9 parallel part reads)
  ES_A3_K1/K2 (0) speculative loads guessed from the next layers' routers (75 % recall at top-8) - background reads, see BG=1
  ES_A3_THREADS / ES_A3_UTHREADS (16 / 12) background / urgent reader threads · ES_A3_BOOST (1) foreground + user-interactive QoS
"""
import os, sys, time, ctypes, threading, queue
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import expertstream as SP
from expertstream import PROJ, DT, POOL

_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
def fast_footprint_mb():
    """phys_footprint via proc_pid_rusage (microseconds; the `footprint` tool costs ~40 ms)."""
    buf = (ctypes.c_uint64 * 64)()
    if _libproc.proc_pid_rusage(os.getpid(), 2, ctypes.byref(buf)) != 0: return SP.footprint_mb()
    return buf[9] / 2**20

def boost_process():
    """Ask macOS to treat this process as a foreground app and the calling (decode) thread as user-interactive.
    The arena stops ~40x per word, so it is sensitive to the decode thread landing on slow cores."""
    rc = []
    try:
        lib = ctypes.CDLL(None)
        task = ctypes.c_uint.in_dll(lib, "mach_task_self_").value
        role = (ctypes.c_int * 1)(1)                                   # TASK_FOREGROUND_APPLICATION
        rc.append(lib.task_policy_set(task, 1, role, 1))               # TASK_CATEGORY_POLICY
        qos = (ctypes.c_int * 2)(0xFF01, 0xFE01)                       # LATENCY_QOS_TIER_0, THROUGHPUT_QOS_TIER_0
        rc.append(lib.task_policy_set(task, 8, qos, 2))                # TASK_BASE_QOS_POLICY
        rc.append(lib.pthread_set_qos_class_self_np(0x21, 0))          # calling thread: user-interactive
        if os.environ.get("ES_A3_RT", "0") == "1":                     # experimental: real-time class for the calling thread
            lib.mach_thread_self.restype = ctypes.c_uint
            pol = (ctypes.c_uint32 * 4)(24000, 12000, 24000, 1)        # period 1 ms, computation 0.5 ms, constraint 1 ms, preemptible
            rc.append(lib.thread_policy_set(lib.mach_thread_self(), 2, pol, 4))
    except Exception as e: rc.append(repr(e))
    return rc

BIG = 1e18


class Pend:
    __slots__ = ("L", "e", "s", "done", "ev", "val", "n", "mine")
    def __init__(self, L, e, s, val, n):
        self.L, self.e, self.s, self.val, self.n, self.mine = L, e, s, val, n, None
        self.done, self.ev = [], threading.Event()


class Arena:
    def __init__(self, store, layers, budget_mb, k1=None, k2=None, age=None, threads=None, tau=None, maxdrop=None, bg=None):
        self.s, self.layers = store, layers
        self.k1 = int(os.environ.get("ES_A3_K1", 0)) if k1 is None else k1
        self.k2 = int(os.environ.get("ES_A3_K2", 0)) if k2 is None else k2
        self.age = float(os.environ.get("ES_A3_AGE", 5)) if age is None else age
        self.tau = float(os.environ.get("ES_A3_TAU", 0.15)) if tau is None else tau           # 0 = exact
        self.maxdrop = float(os.environ.get("ES_A3_MAXDROP", 0.25)) if maxdrop is None else maxdrop
        self.bg = int(os.environ.get("ES_A3_BG", 4)) if bg is None else bg                    # what happens to skipped experts (see top)
        self.dropped = self.bgl = 0
        self.deferred = []
        self.seen = {L: {} for L in layers}
        nthr = int(os.environ.get("ES_A3_THREADS", 16)) if threads is None else threads
        L0 = layers[0]
        self.per = per = sum(store.loc[(L0, p, q)][2] for p in PROJ for q in store.parts[(L0, p)])
        self.S = max(16, min(store.n_exp, int(budget_mb * 2**20 / len(layers) / per)))
        self.gbuf, self.jobs = {}, {}
        for L in layers:
            self.jobs[L] = []
            for p in PROJ:
                grp = []
                for q in store.parts[(L, p)]:
                    fd, base, stride, shape, dt = store.loc[(L, p, q)]
                    a = mx.zeros([self.S] + list(shape), DT[dt][1] if DT[dt][1] is not mx.bfloat16 else mx.uint16)
                    mx.eval(a)
                    view = np.array(a, copy=False).reshape(self.S, -1).view(np.uint8)   # the GPU buffer itself
                    self.gbuf[(L, p, q)] = a.view(mx.bfloat16) if a.dtype == mx.uint16 else a
                    grp.append((fd, base, stride, view))
                self.jobs[L].append(grp)                      # one read job per projection (weight+scales+biases)
            self.jobs1 = getattr(self, "jobs1", {}); self.jobs1[L] = [[t for g in self.jobs[L] for t in g]]
            self.jobs9 = getattr(self, "jobs9", {}); self.jobs9[L] = sorted(([t] for g in self.jobs[L] for t in g), key=lambda g: -g[0][2])
        self.slot = {L: np.full(store.n_exp, -1, np.int32) for L in layers}     # expert -> ready slot
        self.owner = {L: np.full(self.S, -1, np.int64) for L in layers}
        self.last = {L: np.zeros(self.S) for L in layers}
        self.pending = {L: {} for L in layers}
        self.step = 0
        self.hits = self.miss = self.inflight = self.spec = self.loads = 0
        self.t_wait = self.t_eval = self.t_book = self.t_flush = 0.0
        self.spec_jobs = int(os.environ.get("ES_A3_SPECJOBS", 3))   # read jobs per speculative expert (1 = fewer thread wake-ups)
        self.urgent = os.environ.get("ES_A3_URGENT", "1") == "1"
        self.inline = os.environ.get("ES_A3_INLINE", "1") == "1"
        self.usplit = int(os.environ.get("ES_A3_USPLIT", 9))
        # experimental: busy-wait this long at each layer stop before sleeping. Measured: does NOT cure the slow phases.
        self.spin = float(os.environ.get("ES_A3_SPIN_US", 0)) * 1e-6
        self.q, self.uq = queue.SimpleQueue(), queue.SimpleQueue()      # speculative loads / loads a layer is waiting for
        nu = int(os.environ.get("ES_A3_UTHREADS", 12))
        for _ in range(nthr): threading.Thread(target=self._worker, args=(self.q,), daemon=True).start()
        for _ in range(nu): threading.Thread(target=self._worker, args=(self.uq,), daemon=True).start()
        print(f"[arena3] {self.S} expert slots/layer x {len(layers)} layers ({self.S * per * len(layers) / 2**30:.2f} GB), "
              f"guess next layer top-{self.k1}, +2 top-{self.k2}", flush=True)

    def _worker(self, q):
        preadv = os.preadv
        bg = q is self.q
        if bg and os.environ.get("ES_A3_QOS", "default") == "utility":
            try: ctypes.CDLL(None).pthread_set_qos_class_self_np(0x11, 0)     # background readers: stay off the performance cores
            except Exception: pass
        stage = bg and os.environ.get("ES_A3_ALIGN", "1") == "1"
        if stage:                                             # page-aligned direct reads disturb the GPU about half as much
            import mmap
            PG = 16384; mm = mmap.mmap(-1, 1 << 20); mv = memoryview(mm); scr = np.frombuffer(mm, np.uint8)
        while True:
            grp, e, s, pend = q.get()
            try:
                if stage:
                    for fd, base, stride, view in grp:
                        off = base + e * stride; a0 = off // PG * PG; ln = -(-(off + stride - a0) // PG) * PG
                        preadv(fd, [mv[:ln]], a0); view[s][:] = scr[off - a0: off - a0 + stride]
                else:
                    for fd, base, stride, view in grp: preadv(fd, [memoryview(view[s])], base + e * stride)
            finally:
                pend.done.append(1)
                if len(pend.done) >= pend.n: pend.ev.set()

    def start(self, L, e, val, urgent=False):
        """begin loading expert e of layer L into the least valuable free slot (non-blocking)."""
        last = self.last[L]
        s = int(np.argmin(last))
        old = self.owner[L][s]
        if old >= 0: self.slot[L][old] = -1                   # unmap the victim before a single byte is written
        self.owner[L][s] = e; last[s] = BIG                   # in flight: never a victim
        jobs = self.jobs[L] if (urgent or self.spec_jobs != 1) else self.jobs1[L]
        if urgent and self.usplit == 9: jobs = self.jobs9[L]  # waited-for load: every tensor part on its own reader
        p = Pend(L, e, s, val, len(jobs))
        self.pending[L][e] = p
        q = self.uq if (urgent and self.urgent) else self.q
        if urgent and self.inline:                            # the waiting thread reads one part itself (no wake-up latency)
            p.mine = jobs[0]; jobs = jobs[1:]
        for grp in jobs: q.put((grp, e, s, p))
        self.loads += 1
        return p

    def publish(self, p):
        self.slot[p.L][p.e] = p.s; self.last[p.L][p.s] = p.val; del self.pending[p.L][p.e]

    def harvest(self, L):
        pd = self.pending[L]
        if pd:
            for p in [p for p in pd.values() if p.ev.is_set()]: self.publish(p)

    def need(self, L, ids, share=None):
        """make sure experts ids of layer L are in slots. Returns (slots, loads to wait for, keep mask or None).
        With tau > 0 a missing expert whose routing share is < tau is left out of this word (the rest are
        re-weighted) as long as the layer loses <= maxdrop in total; it is loaded in the background for later."""
        self.harvest(L)
        now = self.step + 1
        sl = self.slot[L][ids]
        hit = sl >= 0
        self.last[L][sl[hit]] = now
        nh = int(hit.sum()); self.hits += nh
        if nh == len(ids): return sl, None, None
        wait, keep, lost = [], None, 0.0
        pd = self.pending[L]
        order = np.nonzero(~hit)[0]
        if share is not None and self.tau > 0: order = order[np.argsort(share[order])]     # least important first
        for k in order:
            e = int(ids[k]); p = pd.get(e)
            if share is not None and share[k] < self.tau and lost + share[k] <= self.maxdrop \
                    and not (self.bg == 4 and self.seen[L].pop(e, 0)):       # bg=4: second time it is wanted -> load it now
                lost += share[k]; self.dropped += 1
                if keep is None: keep = np.ones(len(ids), np.float32)
                keep[k] = 0.0
                if self.bg == 4: self.seen[L][e] = 1
                elif p is None and self.bg >= 2:              # load after this word (bg=3: only on the 2nd skip)
                    c = self.seen[L][e] = self.seen[L].get(e, 0) + 1
                    if self.bg == 2 or c >= 2: self.deferred.append((L, e)); self.bgl += 1; self.seen[L].pop(e, None)
                elif p is None and self.bg: self.start(L, e, now - 0.5); self.bgl += 1
                elif p is not None: p.val = max(p.val, now - 0.5)
                continue
            if p is None: p = self.start(L, e, now, True); self.miss += 1
            else: p.val = now; self.inflight += 1
            wait.append(p)
        if keep is not None: keep *= 1.0 / (1.0 - lost)
        return sl, wait, keep

    def finish(self, L, ids, wait):
        t0 = time.perf_counter()
        for p in wait:
            if p.mine is not None:
                for fd, base, stride, view in p.mine: os.preadv(fd, [memoryview(view[p.s])], base + p.e * stride)
                p.mine = None; p.done.append(1)
                if len(p.done) >= p.n: p.ev.set()
        for p in wait: p.ev.wait(); self.publish(p)
        self.t_wait += time.perf_counter() - t0
        return self.slot[L][ids]

    def guess(self, L, es):
        """speculatively load experts es of a later layer L (its graph does not exist yet)."""
        self.harvest(L)
        val = self.step + 1 - self.age
        sl = self.slot[L][es]; last = self.last[L]; pd = self.pending[L]
        h = sl[sl >= 0]
        last[h] = np.maximum(last[h], val)                    # don't evict one guess to make room for another
        for e in es[sl < 0]:
            e = int(e)
            if e not in pd: self.start(L, e, val); self.spec += 1

    def reset(self):
        """forget everything in the slots (benchmarks: cold start without reloading the model)."""
        for L in self.layers:
            for p in list(self.pending[L].values()): p.ev.wait(); self.publish(p)
            self.slot[L][:] = -1; self.owner[L][:] = -1; self.last[L][:] = 0
        self.hits = self.miss = self.inflight = self.spec = self.loads = self.dropped = self.bgl = 0; self.t_wait = 0.0
        self.seen = {L: {} for L in self.layers}; self.deferred = []

    def flush_deferred(self):
        """bg=2: read every expert skipped during this word now, all in parallel, while the GPU is idle."""
        if not self.deferred: return
        t0 = time.perf_counter(); ps = []
        for L, e in self.deferred:
            if self.slot[L][e] < 0 and e not in self.pending[L]: ps.append(self.start(L, e, self.step + 0.5, True))
        self.deferred = []
        for p in ps:
            if p.mine is not None: self.uq.put((p.mine, p.e, p.s, p)); p.mine = None
        for p in ps: p.ev.wait(); self.publish(p)
        self.t_flush += time.perf_counter() - t0

    def install_now(self, L, es):
        ids = np.array(es, np.int64)
        if len(ids) == 0: return
        sl, wait, _ = self.need(L, ids)
        if wait: self.finish(L, ids, wait)


class ArenaExperts(nn.Module):
    def __init__(self, L, arena, streamed, gates):
        super().__init__()
        self._L, self._a, self._st = L, arena, streamed
        self._bits = {p: arena.s.bits[(L, p)] for p in PROJ}
        self._g = {p: tuple(arena.gbuf[(L, p, q)] for q in ("weight", "scales", "biases")) for p in PROJ}
        self._gates = gates

    def _mlp(self, x, si):
        def lin(h, p):
            bits, gs = self._bits[p]; W, sc, bi = self._g[p]
            return mx.gather_qmm(h, W, sc, bi, rhs_indices=si, transpose=True, group_size=gs, bits=bits)
        h = mx.expand_dims(x, (-2, -3))
        return lin(self._st._act(lin(h, "up_proj"), lin(h, "gate_proj")), "down_proj").squeeze(-2)

    def __call__(self, x, idx, w):
        a, L = self._a, self._L
        if x.shape[0] * x.shape[1] != 1:                     # prompt reading: streamed path; remember last word's experts
            a.rec.append((L, idx[:, -1:]))
            return self._st(x, idx, w)
        ng = [(L + d, self._gates[L + d], k) for d, k in ((1, a.k1), (2, a.k2)) if k and (L + d) in self._gates]
        extra = [g(x).reshape(-1).astype(mx.float32) for (_, g, _) in ng]     # next layers' router scores for this input
        if a.tau > 0: extra.append(w.reshape(-1).astype(mx.float32))
        t0 = time.perf_counter()
        if a.spin > 0:                                       # stay awake while the GPU works (see Arena.spin)
            mx.async_eval(idx, w, x, *extra)
            pc = time.perf_counter; end = t0 + a.spin
            while pc() < end: pass
        mx.eval(idx, w, x, *extra)                           # the one stop for this layer
        t1 = time.perf_counter()
        ids = np.array(idx).reshape(-1)
        share = None
        if a.tau > 0:
            share = np.array(extra[-1]); share = share / share.sum()
        sl, wait, keep = a.need(L, ids, share)               # real misses start loading first
        for (Ln, _, k), g in zip(ng, extra): a.guess(Ln, np.argpartition(-np.array(g), k - 1)[:k])
        t2 = time.perf_counter()
        if wait: sl = a.finish(L, ids, wait)
        a.t_eval += t1 - t0; a.t_book += t2 - t1
        if keep is not None:                                  # some low-share experts left out of this word
            sl = np.where(keep > 0, sl, sl.max()); w = w * mx.array(keep).reshape(w.shape).astype(w.dtype)
        si = mx.array(sl.astype(np.uint32)).reshape(idx.shape)
        return (mx.expand_dims(w, -1) * self._mlp(x, si)).sum(-2)


def arena_model(model, budget_mb, model_dir=None, prompt_cache_mb=300, **kw):
    """Swap the model's experts for the exact arena. budget_mb = RAM for expert slots. Returns (store, arena)."""
    store = SP.stream_experts(model, prompt_cache_mb, model_dir, prefetch=False)
    layers = SP._layers(model)
    holders = {}
    for i, l in enumerate(layers):
        if isinstance(getattr(l, "experts", None), SP.StreamedExperts): holders[i] = l
        elif isinstance(getattr(getattr(l, "mlp", None), "experts", None), SP.StreamedExperts): holders[i] = l.mlp
    moe = sorted(holders)
    if not moe: raise RuntimeError("arena: no streamed experts found")
    if any("bias" in store.parts[(moe[0], p)] for p in PROJ): raise RuntimeError("arena: experts with bias (gpt-oss) not supported yet")
    gates = {i: getattr(holders[i], "gate", None) for i in moe}          # Qwen routers; (Gemma: no guessing)
    gates = {i: g for i, g in gates.items() if g is not None}
    ar = Arena(store, moe, budget_mb, **kw); ar.rec = []
    if os.environ.get("ES_A3_BOOST", "1") == "1": boost_process()
    for i in moe: holders[i].experts = ArenaExperts(i, ar, holders[i].experts, gates)
    cls = type(model); base = cls.__call__
    def call(self, inputs, *a, **k):
        out = base(self, inputs, *a, **k)
        mx.eval(out)
        ar.flush_deferred()
        ar.step += 1
        if ar.rec:                                            # a prompt was just read: warm the arena with its last word
            for L, idx in ar.rec: ar.install_now(L, np.unique(np.array(idx).reshape(-1)))
            ar.rec = []
        return out
    cls.__call__ = call
    return store, ar
