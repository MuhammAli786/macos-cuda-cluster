"""Check moe_cpu.so against MLX's own MoE block on layer 0, then time it."""
import ctypes, sys, time
from pathlib import Path
import numpy as np, mlx.core as mx
from mlx_lm.utils import load_model

M = Path("/home/alimu/models/Qwen3.6-35B-A3B-4bit")
model, _ = load_model(M, lazy=True, strict=False)
blk = model.language_model.model.layers[0].mlp
sw = blk.switch_mlp
offs, off, dts = [], 0, []
for p in (sw.gate_proj, sw.up_proj, sw.down_proj):
    for nm in ("weight", "scales", "biases"):
        a = p[nm]; offs.append(off); dts.append(a.dtype)
        off += int(np.prod(a.shape[1:])) * (2 if a.dtype in (mx.bfloat16, mx.float16) else a.dtype.size)
E, row = sw.gate_proj["weight"].shape[0], off
D, F = sw.gate_proj["weight"].shape[2] * 8, sw.gate_proj["weight"].shape[1]
print("E", E, "row", row, "D", D, "F", F, "scales", dts[1])
host = np.memmap(M / ".expert_pack/layer000.bin", dtype=np.uint8, mode="r", shape=(E, row))
lib = ctypes.CDLL("/home/alimu/moe_cpu.so")
T = int(sys.argv[1]) if len(sys.argv) > 1 else 6
assert lib.moe_init(D, F, (ctypes.c_long * 9)(*offs), T, int(dts[1] == mx.bfloat16)) == 0
mx.eval(blk.parameters())
x = (mx.random.normal((1, 1, D)) * 0.5).astype(mx.bfloat16)
gates = mx.softmax(blk.gate(x), axis=-1, precise=True)
inds = mx.argpartition(gates, kth=-8, axis=-1)[..., -8:]
sc = mx.take_along_axis(gates, inds, -1); sc = sc / sc.sum(-1, keepdims=True)
ref = (sw(x, inds) * sc[..., None]).sum(-2)
ref = np.array(ref.astype(mx.float32)).ravel()
xi = np.ascontiguousarray(np.array(x.astype(mx.float32)).ravel())
ii = np.array(inds).ravel().astype(np.int32); ss = np.array(sc.astype(mx.float32)).ravel()
out = np.zeros(D, np.float32)
P = lambda a: a.ctypes.data_as(ctypes.c_void_p)
run = lambda n, i: lib.moe_run(ctypes.c_void_p(host.ctypes.data), ctypes.c_long(row), n, P(i), P(ss), P(xi), P(out))
run(8, ii)
err = np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9)
print(f"max rel err vs MLX: {err:.4f}  (ref |max| {np.abs(ref).max():.3f})")
for _ in range(20): run(8, ii)
s = time.perf_counter(); N = 200
for k in range(N): run(8, np.random.choice(E, 8, replace=False).astype(np.int32))
print(f"threads {T}: {1000 * (time.perf_counter() - s) / N:.3f} ms per token-layer (8 experts, random)")
s = time.perf_counter()
for k in range(N): run(8, ii)
print(f"same 8 experts (L3-hot): {1000 * (time.perf_counter() - s) / N:.3f} ms")
one = ii[:1].copy()
s = time.perf_counter()
for k in range(N): run(1, one)
print(f"1 expert: {1000 * (time.perf_counter() - s) / N:.3f} ms")
big = np.ones(1 << 28, np.uint8)  # 256 MB in anonymous RAM
s = time.perf_counter(); big.view(np.uint64).sum(); d = time.perf_counter() - s
print(f"numpy RAM read: {big.nbytes / d / 1e9:.1f} GB/s (1 thread)")
s = time.perf_counter(); np.asarray(host[:64]).view(np.uint64).sum(); d = time.perf_counter() - s
print(f"memmap read: {64 * row / d / 1e9:.1f} GB/s (1 thread)")
