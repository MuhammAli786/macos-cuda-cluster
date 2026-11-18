"""CUDA port of mlx-lm's Metal gated-delta kernel (Qwen3.5/3.6 linear attention).

On CUDA, mlx-lm falls back to a Python loop with ~8 ops per token per layer,
which is very slow for long prompts. This runs the whole recurrence in one
kernel: one warp per (batch, value head, value row), 4 state values per lane.
install() patches qwen3_5 to use it when there is no padding mask.
"""
import mlx.core as mx

_SRC = r"""
    constexpr int n_per_t = Dk / 32;
    const int dk_idx = threadIdx.x;
    const int dv_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z;
    const int b_idx = n / Hv;
    const int hv_idx = n % Hv;
    const int hk_idx = hv_idx / (Hv / Hk);

    const InT* q_ = q + (size_t)b_idx * T * Hk * Dk + hk_idx * Dk;
    const InT* k_ = k + (size_t)b_idx * T * Hk * Dk + hk_idx * Dk;
    const InT* v_ = v + (size_t)b_idx * T * Hv * Dv + hv_idx * Dv;
    InT* y_ = y + (size_t)b_idx * T * Hv * Dv + hv_idx * Dv;
    const float* g_ = g + (size_t)b_idx * T * Hv;
    const InT* beta_ = beta + (size_t)b_idx * T * Hv;

    const StT* i_state = state_in + ((size_t)n * Dv + dv_idx) * Dk;
    StT* o_state = state_out + ((size_t)n * Dv + dv_idx) * Dk;

    float st[n_per_t];
    #pragma unroll
    for (int i = 0; i < n_per_t; ++i) st[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);

    for (int t = 0; t < T; ++t) {
        const float gg = g_[hv_idx];
        float kv = 0.f;
        float kk[n_per_t];
        #pragma unroll
        for (int i = 0; i < n_per_t; ++i) {
            kk[i] = static_cast<float>(k_[n_per_t * dk_idx + i]);
            st[i] *= gg;
            kv += st[i] * kk[i];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) kv += __shfl_xor_sync(0xffffffff, kv, o);
        const float delta = (static_cast<float>(v_[dv_idx]) - kv) * static_cast<float>(beta_[hv_idx]);
        float out = 0.f;
        #pragma unroll
        for (int i = 0; i < n_per_t; ++i) {
            st[i] += kk[i] * delta;
            out += st[i] * static_cast<float>(q_[n_per_t * dk_idx + i]);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) out += __shfl_xor_sync(0xffffffff, out, o);
        if (dk_idx == 0) y_[dv_idx] = static_cast<InT>(out);
        q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; y_ += Hv * Dv; g_ += Hv; beta_ += Hv;
    }
    #pragma unroll
    for (int i = 0; i < n_per_t; ++i) o_state[n_per_t * dk_idx + i] = static_cast<StT>(st[i]);
"""

_kernel = None


def gated_delta_cuda(q, k, v, g, beta, state):
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.cuda_kernel(
            name="gated_delta_cuda", input_names=["q", "k", "v", "g", "beta", "state_in"],
            output_names=["y", "state_out"], source=_SRC)
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    rows = 4
    return _kernel(
        inputs=[q, k, v, g.astype(mx.float32), beta, state],
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv),
                  ("Hk", Hk), ("Hv", Hv), ("T", T)],
        grid=(32, Dv, B * Hv), threadgroup=(32, rows, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape], output_dtypes=[q.dtype, state.dtype])


def install():
    """Route qwen3_5's gated_delta_update through the CUDA kernel."""
    if mx.metal.is_available() or not mx.cuda.is_available():
        return False
    from mlx_lm.models import gated_delta, qwen3_5
    orig = gated_delta.gated_delta_update

    def patched(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, *, use_kernel=True,
                lower_bound=None, allow_neg_eigval=False):
        # T == 1 (decode) keeps the stock path: it is already fast and compiles.
        if mask is not None or lower_bound is not None or allow_neg_eigval or q.shape[-1] % 32 or q.shape[1] == 1:
            return orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel,
                        lower_bound=lower_bound, allow_neg_eigval=allow_neg_eigval)
        beta = mx.sigmoid(b)
        g = gated_delta.compute_g(A_log, a, dt_bias)
        if state is None:
            B, _, Hk, Dk = q.shape
            Hv, Dv = v.shape[-2:]
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        return gated_delta_cuda(q, k, v, g, beta, state)

    gated_delta.gated_delta_update = patched
    qwen3_5.gated_delta_update = patched
    return True


if __name__ == "__main__":
    import time
    from mlx_lm.models.gated_delta import gated_delta_ops, compute_g
    mx.random.seed(0)
    B, Hk, Hv, Dk, Dv = 1, 16, 32, 128, 128
    for T in (1, 256, 2048):
        q = (mx.random.normal((B, T, Hk, Dk)) * 0.1).astype(mx.bfloat16)
        k = (mx.random.normal((B, T, Hk, Dk)) * 0.1).astype(mx.bfloat16)
        v = mx.random.normal((B, T, Hv, Dv)).astype(mx.bfloat16)
        g = mx.random.uniform(0.8, 1.0, (B, T, Hv))
        beta = mx.random.uniform(0, 1, (B, T, Hv)).astype(mx.bfloat16)
        s0 = (mx.random.normal((B, Hv, Dv, Dk)) * 0.1)
        y1, st1 = gated_delta_cuda(q, k, v, g, beta, s0); mx.eval(y1, st1)
        s = time.perf_counter()
        for _ in range(5): mx.eval(gated_delta_cuda(q, k, v, g, beta, s0))
        tk = (time.perf_counter() - s) / 5
        if T <= 256:
            s = time.perf_counter()
            y2, st2 = gated_delta_ops(q, k, v, g, beta, s0); mx.eval(y2, st2)
            to = time.perf_counter() - s
            err = float(mx.abs(y1.astype(mx.float32) - y2.astype(mx.float32)).max())
            serr = float(mx.abs(st1 - st2).max())
            print(f"T={T}: kernel {1000*tk:.2f} ms, ops loop {1000*to:.1f} ms, max err y {err:.4f} state {serr:.5f}")
        else:
            print(f"T={T}: kernel {1000*tk:.2f} ms")
