import sys, time
exec(open("/home/alimu/lap_bench2.py").read().split('bench("baseline")')[0])
from ring_gen import CpuMoE
import ctypes
pass
x = xs[0]
T = {}
def tic(k, s): T[k] = T.get(k, 0) + time.perf_counter() - s
for it in range(440):
    if it == 40: T.clear()
    blk = inner.layers[it % n].mlp; x = xs[it % 60]
    s = time.perf_counter()
    gates = mx.softmax(blk.gate(x), axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-8, axis=-1)[..., -8:]
    sc = mx.take_along_axis(gates, inds, axis=-1).astype(mx.float32)
    shared = mx.sigmoid(blk.shared_expert_gate(x)) * blk.shared_expert(x)
    xf = x.astype(mx.float32)
    mx.eval(inds, sc, xf); tic("eval", s); s = time.perf_counter()
    mx.async_eval(shared); tic("async_eval", s); s = time.perf_counter()
    idx = np.array(inds).reshape(-1, 8).astype(np.int32); tic("np inds", s); s = time.perf_counter()
    scn = np.array(sc).reshape(-1, 8); tic("np sc", s); s = time.perf_counter()
    xn = np.ascontiguousarray(np.array(xf).reshape(-1, 2048)); tic("np x", s); s = time.perf_counter()
    out = np.empty_like(xn)
    CpuMoE.lib.moe_run(CpuMoE.vp(blk.host.ctypes.data), CpuMoE.cl(blk.row), 8, CpuMoE.vp(idx[0].ctypes.data), CpuMoE.vp(scn[0].ctypes.data), CpuMoE.vp(xn[0].ctypes.data), CpuMoE.vp(out[0].ctypes.data)); tic("moe_run", s); s = time.perf_counter()
    y = mx.array(out).reshape(x.shape).astype(x.dtype) + shared
    mx.eval(y); tic("upload+add eval", s)
print({k: round(1000 * v / 400, 3) for k, v in T.items()})
