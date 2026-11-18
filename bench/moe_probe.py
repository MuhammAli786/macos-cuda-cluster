import time, sys, numpy as np, mlx.core as mx, mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU
def t(name, f, n=2):
    f(); s = time.perf_counter()
    for _ in range(n): f()
    print(f"{name}: {1000*(time.perf_counter()-s)/n:.1f} ms", flush=True)
sw = SwitchGLU(2048, 512, 256); nn.quantize(sw, group_size=64, bits=4); mx.eval(sw.parameters())
for L in (256, 512, 1024):
    h = mx.random.normal((1, L, 2048)).astype(mx.bfloat16)
    inds = mx.random.randint(0, 256, (1, L, 8)).astype(mx.uint32)
    t(f"gather_qmm SwitchGLU L={L}", lambda: mx.eval(sw(h, inds)))
    # alternative: per-expert dense quantized matmuls over the tokens routed to each expert
    idx = np.array(inds).reshape(-1); tok = np.repeat(np.arange(L), 8)
    order = np.argsort(idx, kind="stable"); cnt = np.bincount(idx, minlength=256); starts = np.concatenate([[0], np.cumsum(cnt)])
    tok_sorted = mx.array(tok[order].astype(np.uint32))
    g, u, d = sw.gate_proj, sw.up_proj, sw.down_proj
    def per_expert():
        xs = h[0][tok_sorted]  # (L*8, 2048) rows grouped by expert
        outs = []
        for e in range(256):
            a, b = int(starts[e]), int(starts[e + 1])
            if a == b: continue
            xe = xs[a:b]
            q = lambda p, v: mx.quantized_matmul(v, p["weight"][e], p["scales"][e], p["biases"][e], transpose=True, group_size=64, bits=4)
            outs.append(q(d, nn.silu(q(g, xe)) * q(u, xe)))
        mx.eval(mx.concatenate(outs))
    t(f"per-expert qmm L={L}", per_expert)
