"""Time N layers of a model on the CPU (or GPU) with MLX, no network."""
import sys, time
import mlx.core as mx
from mlx_lm.utils import load_model
sys.path.insert(0, "/home/alimu")
from ring_gen import run_layers

path, dev, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
mx.set_default_device(mx.cpu if dev == "cpu" else mx.gpu)
from pathlib import Path
model, _ = load_model(Path(path), lazy=True, strict=False)
lm = getattr(model, "language_model", model); inner = lm.model
inner.layers = inner.layers[:n]
mx.eval(inner.layers)
hid = lm.args.hidden_size; dt = inner.norm.weight.dtype
cache = lm.make_cache()
x = mx.zeros((1, 64, hid), dtype=dt)
mx.eval(run_layers(inner, x, cache))           # warm-up prompt shape
s = time.perf_counter(); mx.eval(run_layers(inner, x, lm.make_cache())); p = time.perf_counter() - s
x1 = mx.zeros((1, 1, hid), dtype=dt)
for _ in range(3): mx.eval(run_layers(inner, x1, cache))
s = time.perf_counter()
for _ in range(20): mx.eval(run_layers(inner, x1, cache))
d = (time.perf_counter() - s) / 20
print(f"{dev}: {n} layers, decode {1000*d:.2f} ms/token = {1000*d/n:.2f} ms/layer; "
      f"64-token prompt {1000*p/n:.1f} ms/layer; mem {mx.get_peak_memory()/1e9:.2f} GB")
