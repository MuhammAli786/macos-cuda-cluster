"""All 19 laptop layers, CPU experts only, random experts: does the page cache hold?"""
import ctypes, sys, time, resource, numpy as np
lib = ctypes.CDLL("/home/alimu/moe_cpu.so")
E, row, D, F = 256, 1769472, 2048, 512
offs = [0, 524288, 557056, 589824, 1114112, 1146880, 1179648, 1703936, 1736704]
assert lib.moe_init(D, F, (ctypes.c_long * 9)(*offs), int(sys.argv[1]), 1) == 0
import mmap, os
def popmap(p):
    fd = os.open(p, os.O_RDONLY)
    m = mmap.mmap(fd, 0, flags=mmap.MAP_SHARED | mmap.MAP_POPULATE, prot=mmap.PROT_READ)
    os.close(fd)
    return np.frombuffer(m, dtype=np.uint8).reshape(E, row)
t0 = time.perf_counter()
hosts = [popmap(f"/home/alimu/models/Qwen3.6-35B-A3B-4bit/.expert_pack/layer{i:03d}.bin") for i in range(19)]
print(f"populate {time.perf_counter() - t0:.1f}s")
x = (np.random.randn(D) * 0.5).astype(np.float32); s8 = np.full(8, .125, np.float32); out = np.zeros(D, np.float32)
P = lambda a: ctypes.c_void_p(a.ctypes.data)
for rep in range(3):
    f0 = resource.getrusage(resource.RUSAGE_SELF).ru_majflt; t = time.perf_counter(); N = 40
    for k in range(N):
        for h in hosts:
            i = np.random.choice(E, 8, replace=False).astype(np.int32)
            lib.moe_run(P(h), ctypes.c_long(row), 8, P(i), P(s8), P(x), P(out))
    print(f"rep {rep}: {1000 * (time.perf_counter() - t) / N / 19:.3f} ms/layer, major faults {resource.getrusage(resource.RUSAGE_SELF).ru_majflt - f0}")
