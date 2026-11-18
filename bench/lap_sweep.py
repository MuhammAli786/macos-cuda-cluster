import sys, time, os
exec(open("/home/alimu/lap_bench2.py").read().split('def bench(')[0])
from ring_gen import CpuMoE
def bench():
    cache = lm.make_cache()
    for x in xs[:5]: mx.eval(run_layers(inner, x, cache))
    for k in stats: stats[k] = 0 if isinstance(stats[k], int) else 0.0
    s = time.perf_counter()
    for x in xs[5:35]: mx.eval(run_layers(inner, x, cache))
    c = max(stats["calls"], 1)
    return 1000 * (time.perf_counter() - s) / 30 / n, 1000 * stats["t_cpu"] / c
cfgs = [(0, 20), (200, 20), (2000, 20), (20000, 20), (2000, 100), (0, 100)]
res = {c: [] for c in cfgs}
for rep in range(4):
    for c in cfgs:
        CpuMoE.lib.moe_set_wait(*c)
        res[c].append(bench())
for c, v in res.items():
    a = np.array(v)
    print(f"T={os.environ.get('MOE_T')} spin {c[0]:>5}us nap {c[1]:>3}us: layer {np.median(a[:,0]):.2f} ms (min {a[:,0].min():.2f}), cpu {np.median(a[:,1]):.2f}")
