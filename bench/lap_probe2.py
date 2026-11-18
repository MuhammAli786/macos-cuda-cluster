import sys, time
exec(open("/home/alimu/lap_bench2.py").read().split('bench("baseline")')[0])
from ring_gen import CpuMoE
L = CpuMoE.lib; vp, cl = CpuMoE.vp, CpuMoE.cl
xn = (np.random.randn(2048) * 0.5).astype(np.float32); scn = np.full(8, .125, np.float32); out = np.zeros(2048, np.float32)
def run(blk):
    idx = np.random.choice(256, 8, replace=False).astype(np.int32)
    s = time.perf_counter()
    L.moe_run(vp(blk.host.ctypes.data), cl(blk.row), 8, vp(idx.ctypes.data), vp(scn.ctypes.data), vp(xn.ctypes.data), vp(out.ctypes.data))
    return time.perf_counter() - s
x = xs[0]
for mode in ("A alone", "B after async GPU work", "C after GPU sync", "A alone again"):
    tot = 0
    for it in range(400):
        blk = inner.layers[it % n].mlp
        if mode.startswith("B"):
            sh = blk.shared_expert(x); mx.async_eval(sh)
        if mode.startswith("C"):
            sh = blk.shared_expert(x); mx.eval(sh)
        tot += run(blk)
    print(f"{mode}: moe_run {1000 * tot / 400:.3f} ms")
