"""Run one MLX-LM model split across machines (pipeline, ring backend). v2.

Every rank runs this script with MLX_RANK and MLX_HOSTFILE set.
The highest rank ("head") embeds tokens and runs the FIRST layers.
Rank 0 ("tail") runs the LAST layers, picks the next token and prints.

v2 changes vs v1 (ring_gen_v1.py):
  1. Prompts are processed in fixed chunk sizes (BUCKETS) and every size is
     warmed up at start, so CUDA never compiles kernels mid-request.
  2. Per-rank timers (compute / waiting / sending) to rebalance --split.
  3. Only the tail computes the vocabulary step (norm + lm_head); the head
     gets back a 4-byte token id instead of the full hidden state.
     Unused weights are never loaded: lm_head off the head, embeddings off
     the tail, which frees GPU memory for more layers.
  4. Per token: hidden state goes head -> tail, token id goes tail -> head.
     No broadcast. During the prompt the head streams chunks without
     waiting, so both machines work at the same time.

Example: --split 25,7 -> rank 0 (Mac) 25 layers, rank 1 (laptop) 7 layers.
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import mlx.nn as nn
from mlx_lm.generate import wired_limit
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.utils import load_model, load_tokenizer

# Prompt chunk sizes. Any prompt length is cut into these, largest first,
# so only these shapes are ever compiled. 1 is the decode shape.
BUCKETS = (256, 64, 16, 4, 1)


def chunk_sizes(n):
    """Cut n tokens into bucket-sized chunks, e.g. 23 -> [16, 4, 1, 1, 1]."""
    out = []
    for b in BUCKETS:
        while n >= b:
            out.append(b)
            n -= b
    return out


class ExpertCache(nn.Module):
    """MoE experts kept in host RAM; a fixed number of GPU slots per layer.

    Wraps one SwitchGLU. Each expert's 9 tensors (3 projections x weight,
    scales, biases) are packed into ONE byte row in RAM, so a cache miss is a
    single host->GPU transfer. Before each call it reads which experts the
    router picked, uploads the missing ones into least-recently-used slots
    and remaps expert ids to slot ids. Upload counts are padded to fixed
    sizes so CUDA never compiles new kernels mid-run. If one call needs more
    experts than there are slots (long prompt chunks), they go into a
    temporary buffer instead.
    """

    PAD_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    NAMES = ("weight", "scales", "biases")

    def __init__(self, switch, slots, stats, pack_file=None):
        super().__init__()
        self.inner = switch
        self.stats = stats
        self.projs = [switch.gate_proj, switch.up_proj, switch.down_proj]
        # Layout of one packed expert row, from shapes only (nothing loaded yet).
        lazy, self.layout, off = [], [], 0
        for pi, p in enumerate(self.projs):
            for name in self.NAMES:
                if name not in p:
                    continue
                a = p[name]
                raw_dt = mx.uint16 if a.dtype in (mx.bfloat16, mx.float16) else a.dtype
                n = int(np.prod(a.shape[1:])) * raw_dt.size
                self.layout.append((pi, name, off, n, a.shape[1:], raw_dt, a.dtype))
                lazy.append((a, raw_dt))
                off += n
        E, row = lazy[0][0].shape[0], off
        if pack_file is None or not (Path(pack_file).exists()
                                     and Path(pack_file).stat().st_size == E * row):
            # Build the packed rows once (loads each tensor once).
            host = np.empty((E, row), dtype=np.uint8)
            for (a, raw_dt), (_, _, o, n, _, _, _) in zip(lazy, self.layout):
                v = mx.view(a, raw_dt) if raw_dt != a.dtype else a
                host[:, o:o + n] = np.array(v).reshape(E, -1).view(np.uint8)
                mx.clear_cache()
            if pack_file is not None:
                Path(pack_file).parent.mkdir(parents=True, exist_ok=True)
                host.tofile(pack_file)
                del host
        if pack_file is not None:
            # File-backed: Linux keeps hot experts cached and drops cold ones
            # under memory pressure instead of swapping.
            host = np.memmap(pack_file, dtype=np.uint8, mode="r", shape=(E, row))
        self.host = host
        del lazy
        self.E = self.host.shape[0]
        self.S = min(slots, self.E)
        self.slot_buf = {}
        for pi, name, _, _, tail, _, dt in self.layout:
            buf = mx.zeros((self.S,) + tail, dtype=dt)
            self.slot_buf[(pi, name)] = buf
            setattr(self.projs[pi], name, buf)
        self.slot_of = np.full(self.E, -1, dtype=np.int64)
        self.owner = np.full(self.S, -1, dtype=np.int64)
        self.last = np.zeros(self.S, dtype=np.int64)
        self.tick = 0

    def bytes_per_expert(self):
        return self.host.shape[1]

    def _upload(self, experts):
        """ONE host->GPU transfer, then split on the GPU into the 9 tensors."""
        k = len(experts)
        size = next((s for s in self.PAD_SIZES if s >= k), k)
        pad = np.concatenate([experts, np.full(size - k, experts[0])]) if size > k else experts
        packed = mx.array(self.host[pad])  # (size, bytes) uint8
        out = {}
        for pi, name, off, n, tail, raw_dt, dt in self.layout:
            a = mx.view(packed[:, off:off + n], raw_dt).reshape((size,) + tail)
            out[(pi, name)] = mx.view(a, dt) if raw_dt != dt else a
        return out, pad

    def __call__(self, x, indices):
        st = self.stats
        s0 = time.perf_counter()
        idx = np.array(indices)  # sync: we need the router's choice on the host
        s1 = time.perf_counter()
        st["t_sync"] += s1 - s0
        need = np.unique(idx)
        self.tick += 1
        if len(need) <= self.S:
            missing = need[self.slot_of[need] < 0]
            st["hit"] += len(need) - len(missing)
            st["miss"] += len(missing)
            if len(missing):
                busy = np.zeros(self.S, dtype=bool)
                busy[self.slot_of[need[self.slot_of[need] >= 0]]] = True
                order = np.argsort(np.where(busy, np.iinfo(np.int64).max, self.last), kind="stable")
                victims = order[: len(missing)]
                old = self.owner[victims]
                self.slot_of[old[old >= 0]] = -1
                self.owner[victims] = missing
                self.slot_of[missing] = victims
                g, pad = self._upload(missing)
                # padded rows repeat the first expert into its own slot: harmless
                vic = np.concatenate([victims, np.full(len(pad) - len(victims), victims[0])])
                vic = mx.array(vic.astype(np.uint32))
                for key, buf in self.slot_buf.items():
                    buf[vic] = g[key]
                st["t_upload"] += time.perf_counter() - s1
            self.last[self.slot_of[need]] = self.tick
            mapped = mx.array(self.slot_of[idx].astype(np.uint32))
            return self.inner(x, mapped)
        # Too many distinct experts for the slots: temporary buffer.
        st["temp"] += len(need)
        g, pad = self._upload(need)
        for (pi, name), a in g.items():
            setattr(self.projs[pi], name, a)
        remap = np.zeros(self.E, dtype=np.uint32)
        remap[need] = np.arange(len(need), dtype=np.uint32)
        y = self.inner(x, mx.array(remap[idx]))
        mx.eval(y)
        for (pi, name), buf in self.slot_buf.items():
            setattr(self.projs[pi], name, buf)
        st["t_upload"] += time.perf_counter() - s1
        return y


def attach_expert_cache(inner, slots, log, pack_dir=None):
    """Swap every MoE layer's experts on this rank for an ExpertCache."""
    stats = {"hit": 0, "miss": 0, "temp": 0, "t_sync": 0.0, "t_upload": 0.0}
    n, per = 0, 0
    for i, layer in enumerate(inner.pipeline_layers):
        block = getattr(layer, "mlp", None)
        if block is not None and hasattr(block, "switch_mlp"):
            pf = f"{pack_dir}/layer{inner.start_idx + i:03d}.bin" if pack_dir else None
            block.switch_mlp = ExpertCache(block.switch_mlp, slots, stats, pf)
            per = block.switch_mlp.bytes_per_expert()
            n += 1
    if n:
        e = block.switch_mlp
        log(f"expert cache: {n} layers x {e.E} experts in RAM/SSD file ({n * e.E * per / 1e9:.2f} GB), "
            f"{e.S} GPU slots/layer ({n * e.S * per / 1e9:.2f} GB), {per / 1e6:.2f} MB/expert")
    return stats


def run_layers(inner, h, cache):
    """Run this rank's layers. No network calls in here."""
    fa = create_attention_mask(h, cache[inner.fa_idx]) if inner.fa_idx is not None else None
    ssm = create_ssm_mask(h, cache[inner.ssm_idx]) if inner.ssm_idx is not None else None
    for layer, c in zip(inner.pipeline_layers, cache):
        h = layer(h, mask=ssm if layer.is_linear else fa, cache=c)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local model directory on THIS machine")
    ap.add_argument("--split", default=None, help="layers per rank, rank 0 first, e.g. 25,7")
    ap.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--expert-slots", type=int, default=0,
                    help="MoE: keep experts in RAM with this many GPU slots per layer (0 = off)")
    args = ap.parse_args()

    group = mx.distributed.init(backend="ring", strict=True)
    rank, size = group.rank(), group.size()
    head, tail = rank == size - 1, rank == 0
    log = lambda *a: print(f"[rank {rank}/{size} {mx.default_device()}]", *a, flush=True)
    log("connected")

    # ---- load only the weights this rank needs -------------------------
    path = Path(args.model)
    t0 = time.time()
    model, config = load_model(path, lazy=True, strict=False)
    lm = getattr(model, "language_model", model)  # VLM wrappers keep the text model here
    inner = lm.model
    split = [int(x) for x in args.split.split(",")] if args.split else None
    inner.pipeline(group, split=split)
    tied = getattr(lm.args, "tie_word_embeddings", False)
    if not tail and hasattr(lm, "lm_head"):
        lm.lm_head = nn.Identity()  # vocab step happens on the tail only
    if not head and not tied:
        inner.embed_tokens = nn.Identity()  # tail receives hidden states instead
    ex_stats = attach_expert_cache(inner, args.expert_slots, log, str(path / ".expert_pack")) if args.expert_slots else None
    mx.eval(model.parameters())
    log(f"layers {inner.start_idx}..{inner.end_idx - 1}, weights {mx.get_active_memory() / 1e9:.2f} GB")

    hidden = lm.args.hidden_size
    dtype = inner.norm.weight.dtype
    cpu = mx.cpu  # network ops on the CPU stream, so the GPU never waits on the cable

    def vocab_step(h):
        logits = lm.lm_head(inner.norm(h[:, -1:])) if not tied else \
            inner.embed_tokens.as_linear(inner.norm(h[:, -1:]))
        return mx.argmax(logits[0, -1]).astype(mx.int32)

    # ---- fix 1: warm up every chunk shape, locally, before connecting ---
    t1 = time.time()
    for b in BUCKETS:
        cache = lm.make_cache()
        for n in (b, 1):
            x = inner.embed_tokens(mx.zeros((1, n), dtype=mx.int32)) if head \
                else mx.zeros((1, n, hidden), dtype=dtype)
            if head:
                assert x.dtype == dtype, f"embedding dtype {x.dtype} != {dtype}"
            h = run_layers(inner, x, cache)
            mx.eval(vocab_step(h) if tail else h)
    log(f"warm-up of {len(BUCKETS)} shapes done in {time.time() - t1:.1f}s")
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=cpu))  # everyone ready
    log(f"ready after {time.time() - t0:.1f}s, peak memory {mx.get_peak_memory() / 1e9:.2f} GB")

    tokenizer = load_tokenizer(path, eos_token_ids=config.get("eos_token_id"))
    text = open(args.prompt_file).read() if args.prompt_file else args.prompt
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False
    )
    eos = set(tokenizer.eos_token_ids)
    cache = lm.make_cache()
    t = {"compute": 0.0, "wait": 0.0, "send": 0.0}

    def step(L, tokens=None, need_token=False):
        """One forward pass of this rank's layers for L tokens."""
        if head:
            h = inner.embed_tokens(tokens)
        else:
            s = time.perf_counter()
            h = mx.distributed.recv((1, L, hidden), dtype, rank + 1, group=group, stream=cpu)
            mx.eval(h)
            t["wait"] += time.perf_counter() - s
        s = time.perf_counter()
        h = run_layers(inner, h, cache)
        out = vocab_step(h) if need_token else h
        mx.eval(out)
        t["compute"] += time.perf_counter() - s
        if not tail:
            s = time.perf_counter()
            mx.eval(mx.distributed.send(h, rank - 1, group=group, stream=cpu))
            t["send"] += time.perf_counter() - s
        return out

    def send_token(tok):
        s = time.perf_counter()
        mx.eval(mx.distributed.send(tok.reshape(1, 1), size - 1, group=group, stream=cpu))
        t["send"] += time.perf_counter() - s

    def recv_token():
        s = time.perf_counter()
        tok = mx.distributed.recv((1, 1), mx.int32, 0, group=group, stream=cpu)
        mx.eval(tok)
        t["wait"] += time.perf_counter() - s
        return int(tok.item())

    with wired_limit(model):
        # ---- prompt ------------------------------------------------------
        sizes = chunk_sizes(len(prompt))
        t_start = time.perf_counter()
        pos = 0
        tok = None
        for i, L in enumerate(sizes):
            last = i == len(sizes) - 1
            ids = mx.array([prompt[pos:pos + L]], dtype=mx.int32) if head else None
            out = step(L, ids, need_token=tail and last)
            pos += L
        prompt_time = time.perf_counter() - t_start
        prompt_stats = dict(t)
        for k in t:
            t[k] = 0.0

        # ---- generation --------------------------------------------------
        t_gen = time.perf_counter()
        generated = []
        if tail:
            tok = int(out.item())
            generated.append(tok)
            send_token(out)
            detok = tokenizer.detokenizer
            detok.reset()
            detok.add_token(tok)
            print(detok.last_segment, end="", flush=True)
            while len(generated) < args.max_tokens and tok not in eos:
                out = step(1, need_token=True)
                tok = int(out.item())
                generated.append(tok)
                send_token(out)
                detok.add_token(tok)
                print(detok.last_segment, end="", flush=True)
            detok.finalize()
            print(detok.last_segment, flush=True)
        else:
            if head:
                while True:
                    tok = recv_token()
                    generated.append(tok)
                    if tok in eos or len(generated) >= args.max_tokens:
                        break
                    step(1, mx.array([[tok]], dtype=mx.int32))
            else:  # middle ranks: forward until the head stops sending
                raise SystemExit("more than 2 ranks not wired up yet")
        gen_time = time.perf_counter() - t_gen

    n = len(generated)
    ms = lambda v, k: f"{1000 * v / max(k, 1):.1f}"
    log(f"prompt {len(prompt)} tok in {sizes and len(sizes)} chunks: {len(prompt) / prompt_time:.1f} tok/s "
        f"(compute {prompt_stats['compute']:.2f}s, wait {prompt_stats['wait']:.2f}s, send {prompt_stats['send']:.2f}s)")
    log(f"gen {n} tok: {n / gen_time:.1f} tok/s; per token ms: compute {ms(t['compute'], n)}, "
        f"wait {ms(t['wait'], n)}, send {ms(t['send'], n)}; peak {mx.get_peak_memory() / 1e9:.2f} GB")
    if ex_stats:
        tot = ex_stats["hit"] + ex_stats["miss"]
        log(f"expert cache: hit rate {ex_stats['hit'] / max(tot, 1):.0%} "
            f"({ex_stats['hit']} hits, {ex_stats['miss']} uploads, {ex_stats['temp']} via temp buffer); "
            f"total router-sync {ex_stats['t_sync']:.2f}s, upload {ex_stats['t_upload']:.2f}s")


if __name__ == "__main__":
    main()
