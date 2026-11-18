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
import os
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


def chunk_sizes(n, buckets=BUCKETS):
    """Cut n tokens into bucket-sized chunks, e.g. 23 -> [16, 4, 1, 1, 1]."""
    out = []
    for b in buckets:
        while n >= b:
            out.append(b)
            n -= b
    return out


def install_blocked_attention(qblock):
    """Long prompts: run attention in query blocks so the score matrix stays
    small. MLX materialises (heads x queries x keys) scores for head_dim 256,
    which is 1.7 GB for 512 queries against 128k keys."""
    from mlx.utils import tree_map
    from mlx_lm.models import base, qwen3_next
    orig = base.scaled_dot_product_attention

    deq_on_gpu = not mx.metal.is_available()

    def blocked(queries, keys, values, cache, scale, mask, sinks=None):
        if deq_on_gpu and isinstance(keys, tuple):
            # MLX's CUDA backend has no quantized_matmul for these attention
            # shapes: store KV in n bits, but expand to fp16 for the math.
            dq = lambda t: mx.dequantize(*t, group_size=cache.group_size, bits=cache.bits).astype(queries.dtype)
            keys, values, cache = dq(keys), dq(values), None
        L = queries.shape[2]
        S = (keys[0] if isinstance(keys, tuple) else keys).shape[-2]
        # Shrink the block as the context grows: scores are heads x qb x S
        # floats, so cap them at ~128 MB (qb=128 at 16k keys, 16 at 128k).
        qb = max(8, min(qblock, (128 << 20) // (4 * queries.shape[1] * S)))
        if L <= qb or L * S <= qb * 16384 or not (mask is None or isinstance(mask, str)):
            return orig(queries, keys, values, cache, scale, mask, sinks)
        off, outs = S - L, []
        for i in range(0, L, qb):
            j = min(i + qb, L)
            cut = lambda a, e=off + j: a[..., :e, :]
            k = tree_map(cut, keys) if isinstance(keys, tuple) else cut(keys)
            v = tree_map(cut, values) if isinstance(values, tuple) else cut(values)
            o = orig(queries[:, :, i:j], k, v, cache, scale, "causal", sinks)
            mx.eval(o)  # free this block's scores before the next one
            outs.append(o)
        return mx.concatenate(outs, axis=2)

    qwen3_next.scaled_dot_product_attention = blocked


def make_cache(lm, kv_bits):
    """Model cache; full-attention layers get an n-bit quantized KV cache if kv_bits."""
    from mlx_lm.models.cache import KVCache, QuantizedKVCache
    cache = lm.make_cache()
    if kv_bits:
        cache = [QuantizedKVCache(group_size=64, bits=kv_bits) if type(c) is KVCache else c for c in cache]
    return cache


class _Hold:
    """Keeps a module out of MLX's parameter tree (so it is never loaded)."""

    def __init__(self, m):
        self.m = m


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


def expert_layout(switch):
    """Byte layout of one packed expert row: (proj, name, offset, nbytes, shape, raw dtype, dtype)."""
    layout, off = [], 0
    for pi, p in enumerate([switch.gate_proj, switch.up_proj, switch.down_proj]):
        for name in ("weight", "scales", "biases"):
            if name not in p:
                continue
            a = p[name]
            raw_dt = mx.uint16 if a.dtype in (mx.bfloat16, mx.float16) else a.dtype
            n = int(np.prod(a.shape[1:])) * raw_dt.size
            layout.append((pi, name, off, n, a.shape[1:], raw_dt, a.dtype))
            off += n
    return layout, off


class CpuMoE(nn.Module):
    """Replaces a Qwen3-Next MoE block on this rank.

    Router and shared expert run on the GPU. The 8 routed experts per token
    run on the CPU (moe_cpu.so, int8 AVX2) straight from the packed expert
    file in RAM, so expert weights never cross to the GPU at all.
    """

    lib = None

    def __init__(self, block, pack_file, stats, threads, gpu_slots=0):
        super().__init__()
        self.gate = block.gate
        self.shared_expert = block.shared_expert
        self.shared_expert_gate = block.shared_expert_gate
        self.k = block.top_k
        self.norm_topk = block.norm_topk_prob
        self.stats = stats
        sw = block.switch_mlp
        layout, row = expert_layout(sw)
        assert [(l[0], l[1]) for l in layout] == [(p, n) for p in range(3) for n in ("weight", "scales", "biases")]
        E = sw.gate_proj["weight"].shape[0]
        self.D = sw.gate_proj["weight"].shape[2] * 8
        F = sw.gate_proj["weight"].shape[1]
        pf = Path(pack_file)
        if not (pf.exists() and pf.stat().st_size == E * row):
            host = np.empty((E, row), dtype=np.uint8)
            for pi, name, o, n, _, raw_dt, dt in layout:
                a = [sw.gate_proj, sw.up_proj, sw.down_proj][pi][name]
                v = mx.view(a, raw_dt) if raw_dt != dt else a
                host[:, o:o + n] = np.array(v).reshape(E, -1).view(np.uint8)
                mx.clear_cache()
            pf.parent.mkdir(parents=True, exist_ok=True)
            host.tofile(pf)
            del host
        # Map the file with every page pre-faulted (MAP_POPULATE): touching a
        # page for the first time mid-token costs far more than reading it.
        import mmap, os
        fd = os.open(pf, os.O_RDONLY)
        self._map = mmap.mmap(fd, 0, flags=mmap.MAP_SHARED | mmap.MAP_POPULATE, prot=mmap.PROT_READ)
        os.close(fd)
        self.host = np.frombuffer(self._map, dtype=np.uint8).reshape(E, row)
        self.host.reshape(-1)[::4096].sum()  # touch every page so Windows pages the VM memory in now
        self.row = row
        # Prompt chunks: experts are uploaded to the GPU for the chunk instead.
        self.layout = layout
        self.E = E
        self.dummy = {}
        for pi, name, o, n, tail, raw_dt, dt in layout:
            p = [sw.gate_proj, sw.up_proj, sw.down_proj][pi]
            self.dummy[(pi, name)] = mx.zeros((1,) + tuple(tail), dtype=dt)
            setattr(p, name, self.dummy[(pi, name)])  # drop the full on-disk tensors
        self._sw = _Hold(sw)
        # Expert usage, saved after each run, picks which experts live on the GPU.
        self.usage_file = pf.with_suffix(".usage.npy")
        self.counts = np.zeros(E, dtype=np.int64)
        self.S = 0
        if gpu_slots and self.usage_file.exists():
            usage = np.load(self.usage_file)
            hot = np.argsort(-usage, kind="stable")[:gpu_slots]
            self.S = len(hot)
            packed = mx.array(self.host[hot])  # one upload at load time
            for pi, name, o, n, tail, raw_dt, dt in layout:
                a = mx.view(packed[:, o:o + n], raw_dt).reshape((self.S,) + tail)
                setattr([sw.gate_proj, sw.up_proj, sw.down_proj][pi], name,
                        mx.contiguous(mx.view(a, dt) if raw_dt != dt else a))
            self.switch = sw  # now holds only the hot experts
            self._sw = None
            slot = np.full(E, self.S, dtype=np.int32)  # S = "not on GPU"
            slot[hot] = np.arange(self.S, dtype=np.int32)
            self.slot_np = slot
            self.slot_mx = mx.array(np.minimum(slot, self.S - 1).astype(np.uint32))
            self.on_gpu = mx.array(slot < self.S)
        if CpuMoE.lib is None:
            import ctypes
            lib = ctypes.CDLL(str(Path(__file__).with_name("moe_cpu.so")))
            offs = (ctypes.c_long * 9)(*[l[2] for l in layout])
            assert lib.moe_init(self.D, F, offs, threads, int(layout[1][6] == mx.bfloat16)) == 0
            CpuMoE.lib = lib
            CpuMoE.vp = ctypes.c_void_p
            CpuMoE.cl = ctypes.c_long

    def refresh(self):
        """Pull any evicted expert pages back into RAM (cheap if all resident)."""
        import mmap
        self._map.madvise(mmap.MADV_WILLNEED)
        int(self.host.reshape(-1)[::4096].sum())

    def save_usage(self):
        old = np.load(self.usage_file) if self.usage_file.exists() else 0
        np.save(self.usage_file, old + self.counts)
        self.counts[:] = 0

    def gpu_part(self, x):
        """Everything before the host sync: router, shared expert, hot GPU experts."""
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-self.k, axis=-1)[..., -self.k:]
        sc = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk:
            sc = sc / sc.sum(axis=-1, keepdims=True)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        if self.S:
            hot = self.on_gpu[inds]
            shared = shared + (self.switch(x, self.slot_mx[inds]) * mx.where(hot, sc, 0)[..., None]).sum(axis=-2)
        return inds, sc.astype(mx.float32), x.astype(mx.float32), shared

    # Experimental, off by default: batched CPU kernel for prompt chunks (moe_batch).
    # Matches the per-token kernel in isolation but was not benchmarked end to end.
    # Set MOE_CPU_BATCH=1 to try it; the default path is the one in docs/RESULTS.md.
    BATCH = os.environ.get("MOE_CPU_BATCH") == "1"
    GPU_MIN_TOKENS = 192  # below this, per-token CPU is cheaper than uploading ~all experts

    def gpu_chunk(self, x, inds, sc, shared):
        """Prompt chunk: upload every expert the chunk uses (one transfer) and
        run them on the GPU, then put the placeholders back."""
        sw = self._sw.m
        t0 = time.perf_counter()
        idx = np.array(inds)
        need = np.unique(idx)
        rows = self.host[need]
        t1 = time.perf_counter()
        packed = mx.array(rows)
        mx.eval(packed)
        t2 = time.perf_counter()
        projs = [sw.gate_proj, sw.up_proj, sw.down_proj]
        for pi, name, o, n, tail, raw_dt, dt in self.layout:
            a = mx.view(packed[:, o:o + n], raw_dt).reshape((len(need),) + tuple(tail))
            setattr(projs[pi], name, mx.view(a, dt) if raw_dt != dt else a)
        # One dense quantized matmul per expert over the tokens routed to it.
        # (MLX's CUDA gather_qmm takes ~15 s per layer when routing is uneven.)
        L, k = idx.shape[-2], idx.shape[-1]
        flat = idx.reshape(-1)
        order = np.argsort(flat, kind="stable")
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        counts = np.bincount(flat, minlength=self.E)
        xs = x.reshape(L, -1)[mx.array((order // k).astype(np.uint32))]
        W = {(pi, name): getattr(projs[pi], name) for pi, name, *_ in self.layout}
        qm = lambda pi, v, j: mx.quantized_matmul(v, W[(pi, "weight")][j], W[(pi, "scales")][j], W[(pi, "biases")][j],
                                                  transpose=True, group_size=64, bits=4)
        outs, pos = [], 0
        for j, e in enumerate(need):
            n = int(counts[e])
            xe = xs[pos:pos + n]
            outs.append(qm(2, nn.silu(qm(0, xe, j)) * qm(1, xe, j), j))
            pos += n
        ys = mx.concatenate(outs)[mx.array(inv.astype(np.uint32))].reshape(1, L, k, -1)
        y = (ys * sc.astype(x.dtype)[..., None]).sum(axis=-2) + shared
        mx.eval(y)
        t3 = time.perf_counter()
        for pi, name, *_ in self.layout:
            setattr(projs[pi], name, self.dummy[(pi, name)])
        del packed, W, outs, ys, xs
        mx.clear_cache()  # the 4 GB card has no room for ~0.5 GB of stale uploads per layer
        if getattr(self, "debug", False):
            print(f"  gpu_chunk L={x.shape[1]}: router+gather {1000*(t1-t0):.0f} ms, upload {1000*(t2-t1):.0f} ms, compute {1000*(t3-t2):.0f} ms", flush=True)
        self.stats["gpu_chunk_experts"] += len(need)
        return y

    def __call__(self, x):
        st = self.stats
        s0 = time.perf_counter()
        if x.shape[1] > 1 and not self.S and self.BATCH:
            # Prompt chunk: batched CPU kernel, each expert read from RAM once.
            inds, sc, xf, shared = self.gpu_part(x)
            mx.eval(inds, sc, xf, shared)
            s1 = time.perf_counter()
            idx = np.ascontiguousarray(np.array(inds).reshape(-1, self.k), dtype=np.int32)
            scn = np.ascontiguousarray(np.array(sc).reshape(-1, self.k), dtype=np.float32)
            xn = np.ascontiguousarray(np.array(xf).reshape(-1, self.D))
            np.add.at(self.counts, idx.ravel(), 1)
            out = np.empty_like(xn)
            vp, cl = CpuMoE.vp, CpuMoE.cl
            rc = CpuMoE.lib.moe_batch(vp(self.host.ctypes.data), cl(self.row), self.E, xn.shape[0], self.k,
                                      vp(idx.ctypes.data), vp(scn.ctypes.data), vp(xn.ctypes.data), vp(out.ctypes.data))
            assert rc == 0, f"moe_batch failed: {rc}"
            st["t_sync"] += s1 - s0
            st["t_batch"] += time.perf_counter() - s1
            st["batch_tokens"] += xn.shape[0]
            st["calls"] += 1
            return mx.array(out).reshape(x.shape).astype(x.dtype) + shared
        if x.shape[1] >= self.GPU_MIN_TOKENS and self._sw is not None:
            inds, sc, xf, shared = self.gpu_part(x)
            np.add.at(self.counts, np.array(inds).ravel(), 1)
            y = self.gpu_chunk(x, inds, sc, shared)
            st["t_chunk"] += time.perf_counter() - s0
            return y
        fn = self.gpu_fast if (x.shape[1] == 1 and getattr(self, "gpu_fast", None)) else self.gpu_part
        inds, sc, xf, shared = fn(x)
        mx.eval(inds, sc, xf, shared)  # the one sync per layer
        s1 = time.perf_counter()
        idx = np.ascontiguousarray(np.array(inds).reshape(-1, self.k), dtype=np.int32)
        scn = np.ascontiguousarray(np.array(sc).reshape(-1, self.k), dtype=np.float32)
        xn = np.ascontiguousarray(np.array(xf).reshape(-1, self.D))
        np.add.at(self.counts, idx.ravel(), 1)
        if self.S:  # CPU gets only experts not on the GPU
            cold = self.slot_np[idx] >= self.S
            st["gpu_hits"] += int((~cold).sum())
            st["cpu_experts"] += int(cold.sum())
        else:
            cold = np.ones_like(idx, dtype=bool)
            st["cpu_experts"] += idx.size
        out = np.empty_like(xn)
        s2 = time.perf_counter()
        st["t_np"] += s2 - s1
        vp, cl, base = CpuMoE.vp, CpuMoE.cl, CpuMoE.vp(self.host.ctypes.data)
        for t in range(xn.shape[0]):
            c = cold[t]
            if not c.any():
                out[t] = 0
                continue
            it = np.ascontiguousarray(idx[t][c]); stt = np.ascontiguousarray(scn[t][c])
            CpuMoE.lib.moe_run(base, cl(self.row), len(it), vp(it.ctypes.data), vp(stt.ctypes.data),
                               vp(xn[t].ctypes.data), vp(out[t].ctypes.data))
        st["t_sync"] += s1 - s0
        st["t_cpu"] += time.perf_counter() - s1
        st["calls"] += 1
        return mx.array(out).reshape(x.shape).astype(x.dtype) + shared


def refresh_experts(inner):
    """Free MLX's cached buffers, then re-fault expert pages they pushed out."""
    mx.clear_cache()
    for layer in inner.pipeline_layers:
        if isinstance(getattr(layer, "mlp", None), CpuMoE):
            layer.mlp.refresh()


def attach_cpu_experts(inner, threads, log, pack_dir, gpu_slots=0):
    stats = {"t_sync": 0.0, "t_cpu": 0.0, "t_np": 0.0, "calls": 0, "gpu_hits": 0, "cpu_experts": 0,
             "t_chunk": 0.0, "gpu_chunk_experts": 0, "t_batch": 0.0, "batch_tokens": 0}
    n = 0
    for i, layer in enumerate(inner.pipeline_layers):
        block = getattr(layer, "mlp", None)
        if block is not None and hasattr(block, "switch_mlp"):
            layer.mlp = CpuMoE(block, f"{pack_dir}/layer{inner.start_idx + i:03d}.bin", stats, threads, gpu_slots)
            n += 1
    S = layer.mlp.S if n else 0
    log(f"cpu experts: {n} layers, {threads} threads, {S} hot experts/layer on GPU")
    return stats


class _StateCache:
    """Stand-in for ArraysCache inside a compiled function."""
    lengths = None

    def __init__(self, a, b):
        self.s = [a, b]

    def __getitem__(self, i):
        return self.s[i]

    def __setitem__(self, i, v):
        self.s[i] = v

    def __bool__(self):
        return True

    def advance(self, n):
        pass


def compile_layers(inner, log):
    """mx.compile the single-token path of every linear-attention layer.

    Fewer, fused GPU kernels per token. Full-attention layers (growing KV
    cache) stay uncompiled. A CpuMoE block gets its pre-sync GPU part
    compiled; a normal MoE block is compiled with the layer.
    """
    n = 0
    for layer in inner.pipeline_layers:
        if not layer.is_linear:
            continue
        cpu = isinstance(layer.mlp, CpuMoE)

        def attn(x, s0, s1, layer=layer, cpu=cpu):
            c = _StateCache(s0, s1)
            h = x + layer.linear_attn(layer.input_layernorm(x), None, c)
            if cpu:
                return h, layer.post_attention_layernorm(h), c[0], c[1]
            return h + layer.mlp(layer.post_attention_layernorm(h)), None, c[0], c[1]

        layer.fast_attn = mx.compile(attn)
        if cpu:
            layer.mlp.gpu_fast = mx.compile(layer.mlp.gpu_part)
        n += 1
    log(f"compiled {n} linear-attention layers")


def run_layer_fast(layer, h, c):
    if h.shape[1] != 1 or not getattr(layer, "fast_attn", None) or c[0] is None:
        return None
    h, moe_in, c[0], c[1] = layer.fast_attn(h, c[0], c[1])
    c.advance(1)
    return h if moe_in is None else h + layer.mlp(moe_in)


def run_layers(inner, h, cache):
    """Run this rank's layers. No network calls in here."""
    fa = create_attention_mask(h, cache[inner.fa_idx]) if inner.fa_idx is not None else None
    ssm = create_ssm_mask(h, cache[inner.ssm_idx]) if inner.ssm_idx is not None else None
    for layer, c in zip(inner.pipeline_layers, cache):
        out = run_layer_fast(layer, h, c) if layer.is_linear else None
        h = out if out is not None else layer(h, mask=ssm if layer.is_linear else fa, cache=c)
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
    ap.add_argument("--kv-bits", type=int, default=0, help="quantize this rank's KV cache (8 or 4; 0 = 16-bit)")
    ap.add_argument("--prefill-chunk", type=int, default=2048, help="prompt tokens per pipeline step")
    ap.add_argument("--attn-block", type=int, default=128, help="query block for long-prompt attention")
    ap.add_argument("--compile", action="store_true", help="mx.compile single-token linear layers")
    ap.add_argument("--cpu-experts", type=int, default=0,
                    help="MoE: run routed experts on the CPU with this many threads (0 = off)")
    args = ap.parse_args()

    install_blocked_attention(args.attn_block)
    try:
        import gdn_cuda  # CUDA gated-delta kernel (no-op on Metal)
        cuda_gdn = gdn_cuda.install()
        if cuda_gdn:
            # 4 GB card shared with the Windows desktop: past ~3.5 GB, WDDM pages
            # VRAM to system RAM and the GPU stalls. Keep MLX well under that.
            mx.set_cache_limit(256 << 20)
            mx.set_memory_limit(int(2.8e9))
    except ImportError:
        cuda_gdn = False
    if not cuda_gdn:
        # Long prompts make attention buffers of a new size every chunk; MLX
        # keeps freed buffers for reuse, which grew the Mac until macOS killed it.
        mx.set_cache_limit(512 << 20)
    buckets = tuple(sorted({args.prefill_chunk, *BUCKETS}, reverse=True))

    group = mx.distributed.init(backend="ring", strict=True)
    rank, size = group.rank(), group.size()
    head, tail = rank == size - 1, rank == 0
    log = lambda *a: print(f"[rank {rank}/{size} {mx.default_device()}]", *a, flush=True)
    log(f"connected; CUDA gated-delta kernel: {cuda_gdn}; KV cache bits: {args.kv_bits or 16}; buckets {buckets}")

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
    ex_stats = attach_expert_cache(inner, args.expert_slots, log, str(path / ".expert_pack")) \
        if args.expert_slots and not args.cpu_experts else None
    cpu_stats = attach_cpu_experts(inner, args.cpu_experts, log, str(path / ".expert_pack"),
                                   args.expert_slots) if args.cpu_experts else None
    if args.cpu_experts:
        ex_stats = None
    mx.eval(model.parameters())
    if args.compile:
        compile_layers(inner, log)
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
    for b in buckets:
        cache = make_cache(lm, args.kv_bits)
        for n in (b, 1):
            x = inner.embed_tokens(mx.zeros((1, n), dtype=mx.int32)) if head \
                else mx.zeros((1, n, hidden), dtype=dtype)
            if head:
                assert x.dtype == dtype, f"embedding dtype {x.dtype} != {dtype}"
            h = run_layers(inner, x, cache)
            mx.eval(vocab_step(h) if tail else h)
    log(f"warm-up of {len(buckets)} shapes done in {time.time() - t1:.1f}s")
    if cpu_stats:
        t2 = time.time()
        refresh_experts(inner)
        log(f"expert pages refreshed in {time.time() - t2:.1f}s")
    for st in (ex_stats, cpu_stats):  # count only the real request
        for k in st or {}:
            st[k] = 0 if isinstance(st[k], int) else 0.0
    mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=cpu))  # everyone ready
    log(f"ready after {time.time() - t0:.1f}s, peak memory {mx.get_peak_memory() / 1e9:.2f} GB")

    tokenizer = load_tokenizer(path, eos_token_ids=config.get("eos_token_id"))
    text = open(args.prompt_file).read() if args.prompt_file else args.prompt
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False
    )
    eos = set(tokenizer.eos_token_ids)
    cache = make_cache(lm, args.kv_bits)
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
        sizes = chunk_sizes(len(prompt), buckets)
        t_start = time.perf_counter()
        pos = 0
        tok = None
        for i, L in enumerate(sizes):
            last = i == len(sizes) - 1
            ids = mx.array([prompt[pos:pos + L]], dtype=mx.int32) if head else None
            out = step(L, ids, need_token=tail and last)
            pos += L
            if tail and len(sizes) > 8 and (i % 4 == 3 or last):
                mx.eval(out) if last else None
                log(f"  prompt {pos}/{len(prompt)} tok, {time.perf_counter() - t_start:.0f}s; "
                    f"memory {mx.get_active_memory() / 1e9:.2f} GB + cache {mx.get_cache_memory() / 1e9:.2f} GB")
        prompt_time = time.perf_counter() - t_start
        prompt_stats = dict(t)
        for k in t:
            t[k] = 0.0

        # ---- generation --------------------------------------------------
        import resource
        if cpu_stats:
            t2 = time.perf_counter()
            refresh_experts(inner)
            log(f"after prompt: expert pages refreshed in {time.perf_counter() - t2:.2f}s")
            cpu_prompt = dict(cpu_stats)
            for k in cpu_stats:
                cpu_stats[k] = 0 if isinstance(cpu_stats[k], int) else 0.0
        r0 = resource.getrusage(resource.RUSAGE_SELF)
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
    if cpu_stats:
        c = max(cpu_stats["calls"], 1)
        log(f"cpu experts: {cpu_stats['calls']} layer calls, per call ms: sync {1000 * cpu_stats['t_sync'] / c:.2f}, "
            f"cpu {1000 * cpu_stats['t_cpu'] / c:.2f} (of which numpy {1000 * cpu_stats['t_np'] / c:.2f})")
        r = resource.getrusage(resource.RUSAGE_SELF)
        log(f"prompt: {cpu_prompt['calls']} layer calls, cpu {cpu_prompt['t_cpu']:.1f}s, sync {cpu_prompt['t_sync']:.1f}s, "
            f"GPU chunks {cpu_prompt['t_chunk']:.1f}s, {cpu_prompt['gpu_chunk_experts']} expert uploads, "
            f"CPU batch {cpu_prompt['t_batch']:.1f}s for {cpu_prompt['batch_tokens']} token-layers")
        g, cx = cpu_stats["gpu_hits"], cpu_stats["cpu_experts"]
        log(f"routed experts: {g / max(g + cx, 1):.0%} on GPU, {cx / c:.1f} per layer call on CPU")
        for layer in inner.pipeline_layers:
            if isinstance(getattr(layer, "mlp", None), CpuMoE):
                layer.mlp.save_usage()
        log(f"page faults: major {r.ru_majflt} ({r.ru_majflt - r0.ru_majflt} during gen), minor {r.ru_minflt}; "
            f"ctx switches vol {r.ru_nvcsw} invol {r.ru_nivcsw}")


if __name__ == "__main__":
    main()
