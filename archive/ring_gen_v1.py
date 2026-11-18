"""Run one MLX-LM model split across machines (pipeline parallel, ring backend).

Built on the same MLX + MLX-LM that oMLX bundles. Every rank runs this same
script with MLX_RANK and MLX_HOSTFILE set. Rank 0 runs the LAST layers and
prints the output; the highest rank runs the FIRST layers.

Example (2 ranks, Mac = rank 0, laptop = rank 1):
  --split 25,7   -> rank 0 gets 25 layers, rank 1 gets 7 layers
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load_model, load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local model directory on THIS machine")
    ap.add_argument("--split", default=None, help="layers per rank, rank 0 first, e.g. 25,7")
    ap.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    group = mx.distributed.init(backend="ring", strict=True)
    rank, size = group.rank(), group.size()
    log = lambda *a: print(f"[rank {rank}/{size} {mx.default_device()}]", *a, flush=True)
    log("connected")

    path = Path(args.model)
    t0 = time.time()
    model, config = load_model(path, lazy=True, strict=False)
    split = [int(x) for x in args.split.split(",")] if args.split else None
    model.model.pipeline(group, split=split)
    log(f"layers {model.model.start_idx}..{model.model.end_idx - 1} of {len(model.model.layers)}")
    mx.eval(model.parameters())

    # Warm-up: run this rank's layers locally once (network steps disabled by
    # pretending to be a 1-rank pipeline) so CUDA kernel compilation happens
    # now, not while the Mac's GPU is waiting on us (macOS kills GPU waits
    # that run too long).
    inner = model.model
    saved = (inner.pipeline_rank, inner.pipeline_size)
    inner.pipeline_rank, inner.pipeline_size = 0, 1
    t1 = time.time()
    for n in (64, 1):
        cache = model.make_cache()
        mx.eval(model(mx.zeros((1, n), dtype=mx.int32), cache=cache))
        mx.eval(model(mx.zeros((1, 1), dtype=mx.int32), cache=cache))
    inner.pipeline_rank, inner.pipeline_size = saved
    log(f"warm-up done in {time.time() - t1:.1f}s")

    # Barrier so no rank starts generating while another is still loading.
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))
    log(f"loaded in {time.time() - t0:.1f}s, peak memory {mx.get_peak_memory() / 1e9:.2f} GB")

    tokenizer = load_tokenizer(path, eos_token_ids=config.get("eos_token_id"))
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Greedy sampling so every rank picks the same token in lockstep.
    last = None
    for resp in stream_generate(
        model, tokenizer, prompt, max_tokens=args.max_tokens, sampler=make_sampler(temp=0.0)
    ):
        last = resp
        if rank == 0:
            print(resp.text, end="", flush=True)
    if rank == 0:
        print()
    if last is not None:
        log(
            f"prompt {last.prompt_tokens} tok @ {last.prompt_tps:.1f} tok/s, "
            f"gen {last.generation_tokens} tok @ {last.generation_tps:.1f} tok/s, "
            f"peak {last.peak_memory:.2f} GB"
        )


if __name__ == "__main__":
    main()
