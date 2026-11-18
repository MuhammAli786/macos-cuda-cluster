"""Laptop-only: N layers with CPU experts; time the GPU part vs CPU part per layer."""
import sys, time
from pathlib import Path
import numpy as np, mlx.core as mx
sys.path.insert(0, "/home/alimu")
import ring_gen
from ring_gen import attach_cpu_experts, run_layers
from mlx_lm.utils import load_model
model, _ = load_model(Path("/home/alimu/models/Qwen3.6-35B-A3B-4bit"), lazy=True, strict=False)
lm = model.language_model; inner = lm.model
n = int(sys.argv[1]); inner.layers = inner.layers[:n]; lm.lm_head = None
inner.start_idx = 0; inner.end_idx = n
stats = attach_cpu_experts(inner, int(__import__("os").environ.get("MOE_T", 6)), print, "/home/alimu/models/Qwen3.6-35B-A3B-4bit/.expert_pack")
mx.eval(inner.parameters())
hid = lm.args.hidden_size
xs = [(mx.random.normal((1, 1, hid)) * 0.5).astype(mx.bfloat16) for _ in range(60)]
def bench(label):
    cache = lm.make_cache()
    for x in xs[:10]: mx.eval(run_layers(inner, x, cache))
    for k in stats: stats[k] = 0 if isinstance(stats[k], int) else 0.0
    s = time.perf_counter()
    for x in xs[10:]: mx.eval(run_layers(inner, x, cache))
    d = (time.perf_counter() - s) / 50
    c = max(stats["calls"], 1)
    print(f"{label}: {1000*d:.1f} ms/token = {1000*d/n:.2f} ms/layer (sync {1000*stats['t_sync']/c:.2f}, cpu {1000*stats['t_cpu']/c:.2f})")
bench("baseline")
