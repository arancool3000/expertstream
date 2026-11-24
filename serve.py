"""Rivet's Pi 5 engine: mlx_lm's OpenAI server with experts streamed from SSD.
Listens on localhost:1235. Peak RAM ~5 GB, zero swap, zero SSD writes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import importlib.util, mlx_lm
# patch mlx_lm.server so tool-call text streams out as it is generated ("tool_progress"), for live code
_p = os.path.join(os.path.dirname(mlx_lm.__file__), "server.py")
_src = open(_p).read()
_old = """                elif current_state == "tool":
                    tool_text += clean_text"""
_new = """                elif current_state == "tool":
                    tool_text += clean_text
                    if self.stream and clean_text:
                        self.wfile.write(("data: " + json.dumps({"tool_progress": clean_text}) + "\\n\\n").encode()); self.wfile.flush()"""
assert _old in _src, "mlx_lm server layout changed"
_spec = importlib.util.spec_from_loader("mlx_lm.server", loader=None, origin=_p)
S = importlib.util.module_from_spec(_spec); S.__file__ = _p; S.__package__ = "mlx_lm"
sys.modules["mlx_lm.server"] = S
exec(compile(_src.replace(_old, _new), _p, "exec"), S.__dict__)
from mlx_lm.utils import load as _load
from expertstream import MODEL as _M, stream_experts
from pathlib import Path
MODEL = Path(os.environ.get("ES_MODEL", _M))
PORT = os.environ.get("ES_PORT", "1235")
CONC = os.environ.get("ES_CONC", "2")

CACHE_MB = int(os.environ.get("PI5_CACHE_MB", "2300"))
SKIP_TAU = float(os.environ.get("PI5_SKIP_TAU", "0.15"))

DRAFT = os.environ.get("ES_DRAFT", "")   # small same-family model for speculative decoding (lossless)
NDRAFT = os.environ.get("ES_NDRAFT", "4")

def load(path, **kw):
    if DRAFT and os.path.abspath(str(path)) == os.path.abspath(DRAFT):
        return _load(path, **kw)                 # the helper stays fully in RAM (small)
    base = os.environ.get("ES_BASE")             # shared parts live in another model's files (Pi 4 borrows Pi 5's)
    model, tok = _load(base or path, lazy=True, **kw)
    if os.environ.get("ES_ARENA") == "1":           # zero-copy expert arena: one GPU job per word (Gemma models)
        import arena
        try:
            arena.arena_model(model, max(1500, CACHE_MB - 2600), MODEL)   # leave room for the prompt-reading spike
            return model, tok
        except RuntimeError: pass
    st = stream_experts(model, CACHE_MB, MODEL)
    st.skip_tau = SKIP_TAU
    return model, tok

S.load = load
mx.set_cache_limit(256 << 20)
sys.argv = [sys.argv[0], "--model", str(MODEL), "--host", "127.0.0.1", "--port", PORT,
            "--prompt-cache-size", "6", "--prompt-cache-bytes", str(int(os.environ.get("ES_PCACHE_MB", "600")) << 20),
            "--prefill-step-size", "4096", "--max-tokens", "65536", "--decode-concurrency", CONC, "--prompt-concurrency", "1"]
if os.environ.get("ES_ARENA") == "1":               # arena: smaller prompt chunks (smaller RAM spike), one reply at a time
    sys.argv = [("2048" if a == "4096" else a) for a in sys.argv]
    i = sys.argv.index("--decode-concurrency"); sys.argv[i + 1] = "1"
if DRAFT and os.path.isdir(DRAFT):
    sys.argv += ["--draft-model", DRAFT, "--num-draft-tokens", NDRAFT]
S.main()
