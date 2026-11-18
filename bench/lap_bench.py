"""Laptop-only: 19 MoE layers with the expert cache, random inputs, profiled."""
import sys, time, cProfile, pstats
from pathlib import Path
import numpy as np, mlx.core as mx
sys.path.insert(0, "/home/alimu")
from ring_gen import attach_expert_cache, run_layers
from mlx_lm.utils import load_model
model, _ = load_model(Path("/home/alimu/models/Qwen3.6-35B-A3B-4bit"), lazy=True, strict=False)
lm = model.language_model; inner = lm.model
n = int(sys.argv[1]) if len(sys.argv) > 1 else 19
inner.layers = inner.layers[:n]
lm.lm_head = None
stats = attach_expert_cache(inner, 48, print, "/home/alimu/models/Qwen3.6-35B-A3B-4bit/.expert_pack")
mx.eval(inner.parameters())
cache = lm.make_cache()
hid = lm.args.hidden_size
xs = [mx.random.normal((1, 1, hid)).astype(mx.bfloat16) * 0.5 for _ in range(60)]
for x in xs[:10]: mx.eval(run_layers(inner, x, cache))
for k in stats: stats[k] = 0 if isinstance(stats[k], int) else 0.0
pr = cProfile.Profile(); pr.enable()
s = time.perf_counter()
for x in xs[10:]: mx.eval(run_layers(inner, x, cache))
d = (time.perf_counter() - s) / 50
pr.disable()
print(f"{n} layers: {1000*d:.1f} ms/token = {1000*d/n:.2f} ms/layer; stats {stats}")
pstats.Stats(pr).sort_stats("tottime").print_stats(12)
