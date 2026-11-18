# macos-cuda-cluster

Run **Qwen3.6-35B-A3B (4-bit, ~20 GB)** split across a 16 GB MacBook Pro and a
gaming laptop with a 4 GB GPU, joined by one Ethernet cable. Neither machine
can hold the model alone. Together they decode at about **15 tokens/sec** and
are built for prompts up to 128k tokens.

Built on [MLX](https://github.com/ml-explore/mlx) and
[MLX-LM](https://github.com/ml-explore/mlx-lm), with three custom pieces:

| Piece | What it does |
|---|---|
| `cluster/moe_cpu.c` | AVX2 C kernel that runs 4-bit MoE experts on the laptop's CPU, straight from RAM |
| `cluster/gdn_cuda.py` | CUDA kernel for the Gated DeltaNet linear-attention layers (MLX-LM only ships a Metal one) |
| `cluster/ring_gen.py` | The pipeline: splits layers across machines, streams prompt chunks, handles memory limits |

[docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md) logs every bottleneck hit along the
way and how each one was measured and fixed.
Numbers are in [docs/RESULTS.md](docs/RESULTS.md).

## Hardware this was built for

| | Mac (rank 0) | Laptop (rank 1) |
|---|---|---|
| Machine | Apple M1 Pro, 16 GB | Ryzen 5 5600H, RTX 3050 Laptop 4 GB, WSL2 (~12 GB RAM) |
| Layers | 21 to 39 | 0 to 20 |
| Experts run on | Mac GPU | Laptop CPU (`moe_cpu.c`) |
| Address | `192.168.99.2` | `192.168.99.1` |

## Setup

### Laptop (WSL2, Linux)

1. Python venv with `mlx-cuda-13==0.32.4` and `mlx-lm` (tested with
   `0.32.3.dev8`). This setup used `/opt/omlx-cluster-worker/venv`.
2. Build the expert kernel (no gcc needed, `pip install ziglang` provides a compiler):
   ```bash
   python -m ziglang cc -O3 -mavx2 -mfma -mf16c -shared -fPIC \
       -o moe_cpu.so cluster/moe_cpu.c -lpthread
   ```
3. Copy `moe_cpu.so`, `cluster/rank1.sh` and `cluster/hosts.json` to the
   laptop's home directory, and the model to `~/models/Qwen3.6-35B-A3B-4bit`.
4. First run builds `.expert_pack/layerNNN.bin` inside the model folder
   (~9.5 GB, once).

### Mac

1. MLX 0.32.4 and MLX-LM (this setup used the Python bundled with the oMLX
   app, `~/.omlx/bin/omlx-cluster-python`).
2. Model at `~/.omlx/models/mlx-community/Qwen3.6-35B-A3B-4bit`.
3. Passwordless SSH from the Mac to the laptop.

### Network

Direct Ethernet cable, static IPs `192.168.99.2` (Mac) and `192.168.99.1`
(laptop). `cluster/hosts.json` lists them in rank order.

## Run

From the Mac:

```bash
cd cluster
MODEL=Qwen3.6-35B-A3B-4bit \
LAPTOP_ARGS="--cpu-experts 6 --kv-bits 8" \
MAC_ARGS="--kv-bits 8" \
./run.sh 19,21 --max-tokens 300 --compile --prompt-file ../prompts/calib_prompt.txt
```

- `19,21` = 19 layers on the Mac, 21 on the laptop.
- `--cpu-experts 6` = run laptop experts on 6 CPU threads.
- `--kv-bits 8` = 8-bit KV cache (needed for 128k).
- `--compile` = compile single-token layers with `mx.compile`.
- Logs: `cluster/rank0.log` (Mac), `~/ring.log` (laptop).

`run.sh` reads `LAPTOP` (default `alimu@192.168.99.1`) and `MAC_PYTHON` to
point at your own machines.

### Long-context test

```bash
# needle_30k.txt and needle_120k.txt are included; to make another size:
# python prompts/make_needle.py <approx_tokens> <out_file>
./run.sh 19,21 --max-tokens 64 --compile --prompt-file ../prompts/needle_30k.txt
```

The model should answer `4826-TANGO`, a code hidden in the middle of the
prompt among thousands of distractor lines.

## Layout

```
cluster/   the code that runs (ring_gen.py, moe_cpu.c, gdn_cuda.py, run.sh, rank1.sh)
bench/     standalone probes used to measure each bottleneck
prompts/   test prompts and the needle-prompt generator
archive/   earlier versions of ring_gen.py, kept for comparison
docs/      HOW-IT-WORKS.md (engineering log), RESULTS.md (numbers)
```

## Limits

- Plain decoding tops out near **17.5 tok/s** on this hardware: the laptop's
  RAM bandwidth (~18 to 19 GB/s) sets the floor. Faster would need
  speculative decoding (MTP), which is not implemented.
- The laptop GPU is shared with the Windows desktop. Past ~3.5 GB, Windows
  pages VRAM and everything stalls; the code caps MLX at 2.8 GB.
- Gaming on the laptop while this runs will not work.
- 128k is supported by the memory design (section 3.11 of HOW-IT-WORKS), but
  the longest prompt verified end to end is ~30k tokens.
- `moe_batch` (batched CPU prefill) is experimental and off by default
  (`MOE_CPU_BATCH=1` to enable). It was not benchmarked end to end.

## License

MIT. See [LICENSE](LICENSE).
