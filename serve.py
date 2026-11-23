"""Rivet's Pi 5 engine: mlx_lm's OpenAI server with experts streamed from SSD.
Listens on localhost:1235. Peak RAM ~5 GB, zero swap, zero SSD writes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import mlx_lm.server as S
from mlx_lm.utils import load as _load
from expertstream import MODEL, stream_experts

CACHE_MB = int(os.environ.get("PI5_CACHE_MB", "2300"))
SKIP_TAU = float(os.environ.get("PI5_SKIP_TAU", "0.15"))

def load(path, **kw):
    model, tok = _load(path, lazy=True, **kw)
    st = stream_experts(model, CACHE_MB)
    st.skip_tau = SKIP_TAU
    return model, tok

S.load = load
mx.set_cache_limit(256 << 20)
sys.argv = [sys.argv[0], "--model", str(MODEL), "--host", "127.0.0.1", "--port", "1235",
            "--prompt-cache-size", "1", "--prompt-cache-bytes", str(300 << 20),
            "--prefill-step-size", "1024", "--decode-concurrency", "1", "--prompt-concurrency", "1"]
S.main()
