import time, mlx.core as mx
from mlx_lm.models.base import quantized_scaled_dot_product_attention as qsdpa
for bits in (8, 4):
    for S in (32768, 131072):
        q = mx.random.normal((1, 16, 1, 256)).astype(mx.bfloat16)
        k = mx.random.normal((1, 2, S, 256)).astype(mx.bfloat16); v = mx.random.normal((1, 2, S, 256)).astype(mx.bfloat16)
        qk = mx.quantize(k, group_size=64, bits=bits); qv = mx.quantize(v, group_size=64, bits=bits); mx.eval(q, qk, qv)
        f = lambda: mx.eval(qsdpa(q, qk, qv, scale=0.0625, mask=None, group_size=64, bits=bits))
        for _ in range(3): f()
        s = time.perf_counter()
        for _ in range(10): f()
        print(f"{bits}-bit KV decode S={S:6d}: {100*(time.perf_counter()-s):.2f} ms per layer")
