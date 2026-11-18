# How it works

Engineering log for running Qwen3.6-35B-A3B across two machines that are each
too small for it. Each bottleneck is recorded as **symptom, measurement,
cause, fix, result**.

---

## 1. Hardware and model

| | Mac (rank 0, "tail") | Gaming laptop (rank 1, "head") |
|---|---|---|
| Chip | Apple M1 Pro | AMD Ryzen 5 5600H + NVIDIA RTX 3050 Laptop |
| Memory | 16 GB unified | 4 GB VRAM + ~12 GB RAM visible inside WSL2 |
| Runs | layers 21 to 39 (19 layers) | layers 0 to 20 (21 layers) |
| Software | MLX 0.32.4 (Metal), oMLX's bundled Python 3.11 | MLX 0.32.4 (`mlx-cuda-13`), Python 3.14, WSL2 |

Direct Ethernet cable (`192.168.99.2` and `192.168.99.1`), MLX's `ring`
distributed backend.

**Qwen3.6-35B-A3B, 4-bit** (`mlx-community/Qwen3.6-35B-A3B-4bit`, ~20 GB):

- 40 layers, hidden size 2048.
- MoE: 256 experts per layer, 8 routed per token (~3B active parameters).
- One expert (gate + up + down, 4-bit, with scales) is 1.77 MB.
- Hybrid attention: every 4th layer (3, 7, 11, ...) is full attention with a
  KV cache (10 layers). The other 30 are Gated DeltaNet linear-attention
  layers with a fixed-size state, so only 10 layers grow with context.

### Token flow (pipeline parallelism)

```
laptop: embed -> layers 0..20 --(hidden state, 4 KB, over Ethernet)--> Mac: layers 21..39 -> pick token
   ^                                                                                         |
   +----------------------------------(token id, 4 bytes)------------------------------------+
```

Each machine holds only its own layers. During decode one machine works at a
time: **per-token time = laptop time + Mac time + network**.

---

## 2. Decode floor

Decode is memory-bandwidth bound:

```
time per token >= bytes read per token / memory bandwidth
```

Laptop experts: 21 layers x 8 experts x 1.77 MB = **297 MB per token**.
Measured RAM read bandwidth (`bench/bw.c`): **18 to 19 GB/s** (consistent with
single-channel RAM). Floor for the experts alone: **~15.7 ms per token**.

| Part | ms per token |
|---|---|
| Laptop experts in RAM (bandwidth limit) | ~15.7 |
| Laptop GPU work (router, shared expert, attention), ~20 µs per GPU op | ~24 |
| Mac (19 layers, compiled) | ~15.3 |
| Network | ~2 |
| **Total** | **~57 ms, i.e. ~17.5 tok/s** |

The original target was 30 tok/s. The floor rules that out for plain
decoding on this hardware; it would need speculative decoding (not built).

---

## 3. Bottlenecks, in the order they were hit

### 3.1 First run on a small model (Qwen3.5-9B)

- **Symptom:** prompt processing at 4.5 tok/s.
- **Cause:** the CUDA backend JIT-compiles a kernel for each new tensor shape;
  every prompt length was a new shape.
- **Fix:** split prompts into fixed chunk sizes (`BUCKETS`: 2048, 256, 64, 16,
  4, 1) and warm each up at startup. Also: send a 4-byte token id back instead
  of a hidden state, load only each rank's weights, stream prompt chunks
  from the head without waiting.
- **Result:** prompt 4.5 to 33.5 tok/s, decode 23.9 to 25.9 tok/s.

### 3.2 35B, attempt 1: GPU expert cache

- **Problem:** the laptop's 21 layers of experts are ~9.5 GB; the GPU has 4 GB.
- **Approach:** experts in RAM, copied on demand into a fixed number of GPU
  slots per layer (`archive/ring_gen_v2_gpucache.py`).
- **Result:** 40 slots: 2.1 tok/s. 48 slots with one packed upload per call:
  **3.3 tok/s** (71% hit rate, laptop 281 ms per token). Memory-mapped
  variant: 1.0 tok/s.
- **Why it lost:** every layer has to sync the router result back to the CPU,
  then upload. Those per-layer round trips through WSL's GPU driver dominated.

### 3.3 Attempt 2: experts on the laptop CPU

- **First try:** MLX's CPU backend. **~1.5 s per layer per token** (~1,600x
  slower than GPU): its 4-bit CPU path is single-threaded and unvectorized.
- **Fix:** `cluster/moe_cpu.c`.
  - Reads MLX's 4-bit format directly (w = scale x q + bias, groups of 64).
  - Activations quantized to int8 so the inner loop uses AVX2
    `_mm256_maddubs_epi16` (32 byte-pairs per instruction).
  - 6-thread pool. Threads spin briefly between layers instead of sleeping,
    because WSL wakes a sleeping thread ~1 ms late.
  - One file per layer (`.expert_pack/layerNNN.bin`), one row per expert, so
    each expert is one contiguous read.
  - Accuracy vs MLX: ~2.5% max relative error per output (from int8 activations).
- **Result:** ~1 ms per layer. Decode 3.3 to 9.7 tok/s.

### 3.4 First-touch page faults

- **Symptom:** CPU path 6.2 ms in real runs vs 1 ms in the benchmark.
- **Cause:** the memory-mapped expert file was faulted in lazily, mid-token.
- **Fix:** `MAP_POPULATE` plus a touch pass after load.

### 3.5 Reverted experiments

- **Hot experts pinned on GPU** (56 per layer, 3.6 GB): overflowed the 4 GB
  card, Windows paged VRAM. **2.1 tok/s.**
- **Longer thread spin** (0.5 s): starved the Windows host, CPU path went to
  17 ms. Back to 200 µs.
- **21 layers on the Mac:** swapped. 19 layers (9.6 GB peak) is the limit.

### 3.6 Mac: `mx.compile`

- Compiling the single-token linear-attention path: **19 to 15.3 ms per token**.
- Decode 10.4 tok/s.

### 3.7 Long prompts on the laptop GPU

For 128k context, prefill runs in 2048-token chunks. Per-token CPU experts are
too slow there, so chunks of 192+ tokens upload every expert the chunk uses
(one transfer) and run on the GPU.

- **Symptom:** **15.5 s per layer** in the MoE step for a 2048-token chunk.
- **Measurement:** routing 0.1 to 0.3 s, upload 0.05 to 0.7 s, **compute
  15.5 s**. The same op in isolation: 0.49 s.
- **Cause:** MLX CUDA `gather_qmm` degrades badly under uneven routing.
- **Fix:** sort tokens by expert on the CPU, one `quantized_matmul` per expert,
  un-sort.
- **Result:** **15.5 s to ~0.5 s per layer** (~30x).

### 3.8 Gated DeltaNet on CUDA

- **Symptom:** MLX-LM's fast Gated DeltaNet kernel is Metal-only; CUDA falls
  back to a Python loop over tokens.
- **Fix:** `cluster/gdn_cuda.py`, a kernel via `mx.fast.cuda_kernel` (warp
  shuffles, one block per head).
- **Result:** 256 tokens: **117 ms to 2.2 ms**. 2048 tokens: 17 ms.
- Single-token decode keeps the stock path (already fast, compiles).

### 3.9 Memory pressure: RAM

- **Symptom:** after adding 2048-token chunks, decode fell from 10.4 to
  6.0 tok/s; 13,740 major page faults on the laptop.
- **Cause:** 9.5 GB of expert files in ~12 GB of WSL RAM. Larger warm-up left
  MLX buffers and model-file pages resident, evicting expert pages.
- **Fix:** after warm-up and after each prompt, `mx.clear_cache()` and
  re-fault expert pages (`madvise(MADV_WILLNEED)` plus a touch pass).
- **Result:** **15.1 tok/s steady decode** (laptop timer). The Mac's timer
  shows 12.1 because it includes a one-off ~5 s refresh after the prompt.

### 3.10 Memory pressure: VRAM

- **Symptom:** a 30k prompt stalled. GPU 0%, laptop main thread 100% CPU,
  VRAM 3,950 / 4,096 MiB.
- **Diagnosis:** per-thread CPU time from `/proc/<pid>/task/*/stat`. Only the
  main thread was busy with the GPU idle: blocked in the driver.
- **Cause:** the card is shared with the Windows desktop. Past ~3.5 GB, WDDM
  pages GPU memory to system RAM.
- **Fix:** MLX cache limit 256 MB, memory target 2.8 GB, free each chunk's
  uploaded experts as soon as the layer finishes.
- **Result:** VRAM ~3.0 to 3.6 GB, no stall.

### 3.11 128k context

- **KV cache:** 10 full-attention layers x 2 KV heads x 256 dims = 20 KB per
  token in fp16, 2.6 GB at 128k (1.3 GB per machine). 8-bit halves it.
- **8-bit KV on CUDA:** no quantized matmul for these attention shapes in
  MLX's CUDA backend. Fix: store 8-bit, dequantize to fp16 right before
  attention.
- **Attention scores:** head dim 256 is unfused in MLX, so it materializes the
  full score matrix: 1.7 GB for 512 queries x 128k keys. Fix: block the
  queries and shrink the block as context grows, keeping scores near 128 MB
  (128 queries per block at 16k keys, 16 at 128k).
- **Mac `Killed: 9` at 40k tokens:** MLX's freed-buffer cache grew without
  bound because every attention block at a new context length has a new size.
  Fix: `mx.set_cache_limit` (512 MB on the Mac) and memory in each progress
  line. The rerun held flat at ~10 GB.

---

## 4. Experimental: batched CPU kernel for prefill

`moe_batch` in `cluster/moe_cpu.c` runs a whole prompt chunk through the CPU
expert kernel, reading each expert from RAM once per chunk instead of
uploading to the GPU. In isolation (`bench/batch_test.py`) it matched the
per-token kernel's output. It was **not benchmarked end to end** and is off by
default; enable with `MOE_CPU_BATCH=1`.

## 5. Results

See [RESULTS.md](RESULTS.md).
