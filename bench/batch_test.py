"""moe_batch (prompt chunks) vs moe_run per token: same answer? how fast?"""
import sys, time
from pathlib import Path
import numpy as np, mlx.core as mx
sys.path.insert(0, "/home/alimu")
import ring_gen
from ring_gen import attach_cpu_experts, CpuMoE
from mlx_lm.utils import load_model
M = "/home/alimu/models/Qwen3.6-35B-A3B-4bit"
model, _ = load_model(Path(M), lazy=True, strict=False)
lm = model.language_model; inner = lm.model
n = 4
inner.layers = inner.layers[:n]; lm.lm_head = None; inner.start_idx = 0; inner.end_idx = n
attach_cpu_experts(inner, 6, print, f"{M}/.expert_pack")
mx.eval(inner.parameters())
blk = inner.layers[0].mlp
for L in (16, 256, 2048):
    x = (mx.random.normal((1, L, 2048)) * 0.5).astype(mx.bfloat16); mx.eval(x)
    CpuMoE.BATCH = True
    blk(x); s = time.perf_counter(); yb = blk(x); mx.eval(yb); tb = time.perf_counter() - s
    CpuMoE.BATCH = False; g = CpuMoE.GPU_MIN_TOKENS; CpuMoE.GPU_MIN_TOKENS = 10**9
    s = time.perf_counter(); yt = blk(x); mx.eval(yt); tt = time.perf_counter() - s
    CpuMoE.GPU_MIN_TOKENS = g
    a, b = np.array(yb.astype(mx.float32)), np.array(yt.astype(mx.float32))
    print(f"L={L:5d}: batch {1000*tb:7.1f} ms, per-token {1000*tt:7.1f} ms; max abs diff {np.abs(a-b).max():.4f} (|y| max {np.abs(b).max():.3f})", flush=True)
