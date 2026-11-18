import sys, time, os
exec(open("/home/alimu/lap_bench2.py").read().split('def bench(')[0])
import ring_gen
def bench():
    cache = lm.make_cache()
    mx.eval(run_layers(inner, xs[0][:, :1].repeat(4, axis=1) if False else mx.concatenate([xs[0], xs[1]], axis=1), cache))  # prompt step sets states
    for x in xs[2:8]: mx.eval(run_layers(inner, x, cache))
    for k in stats: stats[k] = 0 if isinstance(stats[k], int) else 0.0
    s = time.perf_counter()
    for x in xs[8:38]: mx.eval(run_layers(inner, x, cache))
    c = max(stats["calls"], 1)
    return 1000 * (time.perf_counter() - s) / 30 / n, 1000 * stats["t_sync"] / c, 1000 * stats["t_cpu"] / c
saved = {}
for l in inner.pipeline_layers:
    if l.is_linear: saved[id(l)] = None
plain = [bench() for _ in range(3)]
ring_gen.compile_layers(inner, print)
comp = [bench() for _ in range(3)]
plain2 = []
for l in inner.pipeline_layers:
    if l.is_linear: l.fast_attn = None; l.mlp.gpu_fast = None
plain2 = [bench() for _ in range(3)]
for name, r in (("plain", plain), ("compiled", comp), ("plain again", plain2)):
    a = np.array(r); print(f"{name:12s} layer {np.median(a[:,0]):.2f} ms, sync {np.median(a[:,1]):.2f}, cpu {np.median(a[:,2]):.2f}")
# correctness: compiled vs plain on the same input
cache1, cache2 = lm.make_cache(), lm.make_cache()
p = mx.concatenate([xs[0], xs[1]], axis=1)
mx.eval(run_layers(inner, p, cache1)); mx.eval(run_layers(inner, p, cache2))
y1 = run_layers(inner, xs[3], cache1)
ring_gen.compile_layers(inner, lambda *a: None)
y2 = run_layers(inner, xs[3], cache2)
print("max abs diff compiled vs plain:", float(mx.abs(y1.astype(mx.float32) - y2.astype(mx.float32)).max()), "scale", float(mx.abs(y1).max()))
