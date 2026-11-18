# bench

Standalone probes, each written to measure one suspected bottleneck. They
were run by hand during development and still contain the original
hard-coded paths (`/home/alimu`, model folders). Edit those before running.

| Script | Measures |
|---|---|
| `bw.c` | Raw RAM read bandwidth (sets the decode floor) |
| `cpu1.py`, `cpu_probe.py` | MLX's own CPU path on a few layers (the ~1.5 s/layer result) |
| `cpu_test.py` | `moe_cpu.so` accuracy vs MLX's MoE block on layer 0, then its speed |
| `hp_test.py` | `moe_cpu.so` reading from a memmap vs anonymous huge-page memory |
| `cpu19.py` | All 19 layers' expert files pre-faulted: does the page cache hold them? |
| `lap_bench.py` | Laptop-only layers with the GPU expert cache, profiled |
| `lap_bench2.py`, `lap_probe.py`, `lap_cmp.py` | Laptop-only layers with CPU experts: GPU part vs CPU part, compiled vs plain |
| `lap_probe2.py` | `moe_run` alone vs right after GPU work (thread wake-up cost) |
| `lap_sweep.py` | Thread spin/nap timing sweep (env `MOE_T` = threads) |
| `lap_chunk.py` | One long prompt chunk through N laptop layers, attention vs MoE time (N >= 4) |
| `gpu_probe.py`, `copy_probe.py`, `up_probe.py` | GPU op overhead and RAM-to-GPU upload speed under WSL |
| `moe_probe.py` | Batched expert matmul (`gather_qmm`) vs one matmul per expert |
| `piece_probe.py` | Individual prefill pieces (e.g. depthwise conv1d) at long chunk lengths |
| `sdpa_probe.py`, `sdpa_probe2.py`, `qsdpa_probe.py` | Attention time and memory at long context, fp16 vs quantized KV |
