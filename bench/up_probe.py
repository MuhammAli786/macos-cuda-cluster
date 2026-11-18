import time, numpy as np, mlx.core as mx
def t(f, n=30):
    for _ in range(3): f()
    s = time.perf_counter()
    for _ in range(n): f()
    return 1000 * (time.perf_counter() - s) / n
x = mx.ones((4,)); mx.eval(x)
print(f"empty GPU sync:                {t(lambda: mx.eval(x + 1)):.2f} ms")
host = np.random.randint(0, 255, size=(256, 1856000), dtype=np.uint8)   # 256 experts x 1.77 MB
rows = np.array([5, 77, 140])
print(f"numpy gather 3 experts:        {t(lambda: host[rows]):.2f} ms")
sub = np.ascontiguousarray(host[rows])
print(f"mx.array(3 experts), no eval:  {t(lambda: mx.array(sub)):.2f} ms")
print(f"mx.array(3 experts) + eval:    {t(lambda: mx.eval(mx.array(sub) + 0)):.2f} ms")
one = np.ascontiguousarray(host[:1])
print(f"mx.array(1 expert) + eval:     {t(lambda: mx.eval(mx.array(one) + 0)):.2f} ms")
slots = mx.zeros((48, 1856000), dtype=mx.uint8); mx.eval(slots)
vic = mx.array(np.array([1, 2, 3], dtype=np.uint32))
def scat():
    slots[vic] = mx.array(sub); mx.eval(slots)
print(f"upload 3 + scatter into slots: {t(scat):.2f} ms")
