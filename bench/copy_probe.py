import time, numpy as np, mlx.core as mx
# One layer of 8 experts at 4-bit is roughly 15-40 MB; test a few sizes.
for mb in (4, 16, 64):
    host = np.random.randint(0, 2**31, size=(mb * 2**20 // 4,), dtype=np.uint32)
    for _ in range(2): mx.eval(mx.array(host) + 0)   # warm-up
    s = time.perf_counter()
    for _ in range(10):
        a = mx.array(host); mx.eval(a + 0)          # host RAM -> GPU, then touch it on GPU
    d = (time.perf_counter() - s) / 10
    print(f"{mb:3d} MB: {1000*d:6.2f} ms  ->  {mb/1024/d:5.2f} GB/s", flush=True)
# Copy into an existing GPU buffer slot (what a cache would do)
slots = mx.zeros((64, 2**20 // 4), dtype=mx.uint32); mx.eval(slots)
host = np.random.randint(0, 2**31, size=(8, 2**20 // 4), dtype=np.uint32)  # 8 experts x 1 MB
idx = mx.array([3, 9, 12, 20, 33, 41, 50, 63])
for _ in range(3): slots[idx] = mx.array(host); mx.eval(slots)
s = time.perf_counter()
for _ in range(50): slots[idx] = mx.array(host); mx.eval(slots)
d = (time.perf_counter() - s) / 50
print(f"scatter 8x1MB into GPU slots: {1000*d:.2f} ms", flush=True)
