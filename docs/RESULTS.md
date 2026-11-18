# Results

All runs: Qwen3.6-35B-A3B-4bit, split 19 layers (Mac) / 21 layers (laptop),
measured 2026-11-11. "tok/s" is tokens per second.

## Decode speed over time (short prompt, 161 tokens)

| Stage | Decode tok/s | What changed |
|---|---|---|
| GPU expert cache, 40 slots | 2.1 | experts copied into GPU slots per layer |
| GPU expert cache, 48 slots, packed upload | 3.3 | one upload per layer instead of many |
| CPU experts (`moe_cpu.c`) | 9.7 | experts computed on the laptop CPU from RAM |
| + `mx.compile` on the Mac | 10.4 | Mac 19 to 15.3 ms per token |
| + 2048-token chunks, CUDA GDN kernel | 6.0 | regression: expert pages pushed out of RAM |
| + cache clearing and page refresh | **15.1** steady (12.1 incl. a one-off 5 s pause after the prompt) | laptop 48 ms, Mac 16.5 ms per token |

Theoretical floor for plain decoding on this hardware: ~17.5 tok/s
(see [HOW-IT-WORKS.md](HOW-IT-WORKS.md), section 2).

## Long-context needle tests

A code ("4826-TANGO") is hidden 60% of the way through thousands of
synthetic log lines; the model is asked for it at the end.

| Prompt | Found the code? | Prompt time | Prompt tok/s | Mac peak memory |
|---|---|---|---|---|
| 8,095 tokens | yes, answered `4826-TANGO` | 309 s | 26.2 | 10.73 GB |
| 29,918 tokens | yes, answered `4826-TANGO` | 589 s | 50.8 | 10.85 GB |
| 122,882 tokens | not completed: run stopped partway through the prompt | | | |

## Prefill pieces (laptop, one 2048-token chunk, per layer)

| Piece | Before | After |
|---|---|---|
| MoE (MLX `gather_qmm`) | 15.5 s | 0.5 s (one matmul per expert) |
| Gated DeltaNet, 256 tokens | 117 ms (Python loop) | 2.2 ms (CUDA kernel) |
| Gated DeltaNet, 2048 tokens | | 17 ms |
