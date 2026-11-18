import time, mlx.core as mx
from pathlib import Path
from mlx_lm.utils import load_model
import sys; sys.path.insert(0, "/home/alimu")
from ring_gen import run_layers
mx.set_default_device(mx.cpu)
model, _ = load_model(Path("/home/alimu/models/Qwen3.5-9B-4bit"), lazy=True, strict=False)
lm = model.language_model; inner = lm.model
inner.layers = inner.layers[:4]; mx.eval(inner.layers)
cache = lm.make_cache(); x = mx.zeros((1, 1, 4096), dtype=inner.norm.weight.dtype)
for i in range(4):
    s = time.perf_counter(); mx.eval(run_layers(inner, x, cache)); print(f"decode 4 layers: {1000*(time.perf_counter()-s):.0f} ms", flush=True)
