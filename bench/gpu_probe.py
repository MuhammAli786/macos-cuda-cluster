import time, numpy as np, mlx.core as mx, mlx.nn as nn
def t(f, n=200):
    for _ in range(20): f()
    s = time.perf_counter()
    for _ in range(n): f()
    return 1000 * (time.perf_counter() - s) / n
a = mx.ones((1, 2048), dtype=mx.bfloat16); mx.eval(a)
print(f"eval(a+1): {t(lambda: mx.eval(a + 1)):.3f} ms")
print(f"np.array(a+1): {t(lambda: np.array((a + 1).astype(mx.float32))):.3f} ms")
h = mx.array(np.ones((1, 2048), np.float32))
print(f"mx.array(host 8KB)+eval: {t(lambda: mx.eval(mx.array(np.ones((1, 2048), np.float32)) + a)):.3f} ms")
q = nn.QuantizedLinear(2048, 12288, bias=False, group_size=64, bits=4); mx.eval(q.parameters())
print(f"1 qmv 2048->12288 eval: {t(lambda: mx.eval(q(a))):.3f} ms")
def chain10():
    y = a
    for _ in range(10): y = q(y)[:, :2048]
    mx.eval(y)
print(f"10 chained qmv in one eval: {t(chain10) / 10:.3f} ms each")
def ew50():
    y = a
    for _ in range(50): y = y * 1.0001 + 0.0001
    mx.eval(y)
print(f"50 elementwise ops in one eval: {t(ew50):.3f} ms total")
