import time, mlx.core as mx
for S in (8192, 32768, 131072):
    q = mx.random.normal((1, 16, 1, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, S, 256)).astype(mx.bfloat16); v = mx.random.normal((1, 2, S, 256)).astype(mx.bfloat16); mx.eval(q, k, v)
    f = lambda: mx.eval(mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625))
    for _ in range(3): f()
    s = time.perf_counter()
    for _ in range(10): f()
    print(f"decode S={S:6d}: {100*(time.perf_counter()-s):.2f} ms per layer")
q = mx.random.normal((1, 16, 256, 256)).astype(mx.bfloat16)
k = mx.random.normal((1, 2, 131072, 256)).astype(mx.bfloat16); v = mx.random.normal((1, 2, 131072, 256)).astype(mx.bfloat16); mx.eval(q, k, v)
f = lambda: mx.eval(mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal"))
f(); s = time.perf_counter(); f(); f(); print(f"prefill chunk 256 at S=128k: {500*(time.perf_counter()-s):.0f} ms per layer")
