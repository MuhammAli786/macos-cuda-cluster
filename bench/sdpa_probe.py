import sys, time, mlx.core as mx
for L, S in [(1, 131072), (256, 32768), (512, 131072), (2048, 131072)]:
    mx.reset_peak_memory(); mx.clear_cache()
    q = mx.random.normal((1, 16, L, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 2, S, 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, S, 256)).astype(mx.bfloat16)
    mx.eval(q, k, v); base = mx.get_peak_memory()
    try:
        s = time.perf_counter()
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.0625, mask="causal" if L > 1 else None)
        mx.eval(o); d = time.perf_counter() - s
        print(f"L={L:5d} S={S:6d}: {1000*d:8.1f} ms, extra peak {(mx.get_peak_memory()-base)/1e9:.2f} GB")
    except Exception as e:
        print(f"L={L} S={S}: FAILED {str(e)[:120]}")
    del q, k, v
