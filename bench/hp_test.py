import mmap, sys, time, ctypes, numpy as np
exec(open("/home/alimu/cpu_test.py").read().split("lib = ctypes")[0].replace("model, _ = load_model", "model, _ = load_model"))
lib = ctypes.CDLL("/home/alimu/moe_cpu.so")
T = int(sys.argv[1]); assert lib.moe_init(D, F, (ctypes.c_long * 9)(*offs), T, 1) == 0
m = mmap.mmap(-1, E * row, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
m.madvise(mmap.MADV_HUGEPAGE)
anon = np.frombuffer(m, dtype=np.uint8).reshape(E, row); anon[:] = host
xi = (np.random.randn(D) * 0.5).astype(np.float32); ss = np.full(8, 0.125, np.float32); out = np.zeros(D, np.float32)
P = lambda a: a.ctypes.data_as(ctypes.c_void_p)
for name, arr in (("memmap", host), ("anon+hugepage", anon)):
    s = time.perf_counter(); N = 300
    for k in range(N):
        i = np.random.choice(E, 8, replace=False).astype(np.int32)
        lib.moe_run(ctypes.c_void_p(arr.ctypes.data), ctypes.c_long(row), 8, P(i), P(ss), P(xi), P(out))
    print(f"T={T} {name}: {1000 * (time.perf_counter() - s) / N:.3f} ms")
