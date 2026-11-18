"""Laptop-only: one prompt chunk of L tokens through N layers, timed per layer."""
import sys, time
from pathlib import Path
import numpy as np, mlx.core as mx
sys.path.insert(0, "/home/alimu")
import gdn_cuda; gdn_cuda.install()
import ring_gen
from ring_gen import attach_cpu_experts
from mlx_lm.utils import load_model
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
ring_gen.install_blocked_attention(128)
model, _ = load_model(Path("/home/alimu/models/Qwen3.6-35B-A3B-4bit"), lazy=True, strict=False)
lm = model.language_model; inner = lm.model
n, L = int(sys.argv[1]), int(sys.argv[2])
inner.layers = inner.layers[:n]; lm.lm_head = None; inner.start_idx = 0; inner.end_idx = n
stats = attach_cpu_experts(inner, 6, print, "/home/alimu/models/Qwen3.6-35B-A3B-4bit/.expert_pack")
mx.eval(inner.parameters())
for l in inner.layers: l.mlp.debug = True
cache = lm.make_cache()
h = (mx.random.normal((1, L, 2048)) * 0.5).astype(mx.bfloat16); mx.eval(h)
for rep in range(2):
    cache = lm.make_cache()
    x = h
    fa = create_attention_mask(x, cache[inner.fa_idx]); ssm = create_ssm_mask(x, cache[inner.ssm_idx])
    for i, (layer, c) in enumerate(zip(inner.pipeline_layers, cache)):
        s = time.perf_counter()
        r = layer.linear_attn(layer.input_layernorm(x), ssm, c) if layer.is_linear else layer.self_attn(layer.input_layernorm(x), fa, c)
        hh = x + r; mx.eval(hh); t1 = time.perf_counter()
        x = hh + layer.mlp(layer.post_attention_layernorm(hh)); mx.eval(x); t2 = time.perf_counter()
        if rep == 1 or i < 3:
            print(f"rep {rep} layer {i:2d} {'lin' if layer.is_linear else 'att'}: attn {1000*(t1-s):7.1f} ms, moe {1000*(t2-t1):7.1f} ms, GPU peak {mx.get_peak_memory()/1e9:.2f} GB", flush=True)
