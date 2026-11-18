import time, mlx.core as mx, mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU
def t(name, f, n=3):
    f(); s = time.perf_counter()
    for _ in range(n): f()
    print(f"{name}: {1000*(time.perf_counter()-s)/n:.1f} ms", flush=True)
L = int(__import__("sys").argv[1])
conv = nn.Conv1d(8192, 8192, kernel_size=4, groups=8192, bias=False, padding=0); mx.eval(conv.parameters())
x = mx.random.normal((1, L + 3, 8192)).astype(mx.bfloat16)
t(f"depthwise conv1d L={L}", lambda: mx.eval(conv(x)))
w = conv.weight  # (C, K, 1)
def manual():
    y = sum(x[:, j:j + L] * w[:, j, 0] for j in range(4))
    mx.eval(y)
t(f"manual shifted-sum conv L={L}", manual)
sw = SwitchGLU(2048, 512, 256); nn.quantize(sw, group_size=64, bits=4); mx.eval(sw.parameters())
h = mx.random.normal((1, L, 2048)).astype(mx.bfloat16)
inds = mx.random.randint(0, 256, (1, L, 8)).astype(mx.uint32)
t(f"SwitchGLU 256 experts L={L}", lambda: mx.eval(sw(h, inds)))
