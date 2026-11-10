"""Tests for SM120 MXFP8 GEMM (issue #2728)."""

import pytest
import torch
import torch.nn.functional as F

from flashinfer import mm_mxfp8, SfLayout
from flashinfer.fp8_quantization import mxfp8_quantize
from flashinfer.utils import get_compute_capability


def _is_sm120_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        cc = get_compute_capability(torch.device("cuda"))
        return cc[0] == 12
    except RuntimeError:
        return False


def _skip_if_not_sm120():
    if not _is_sm120_available():
        pytest.skip("Requires SM12x GPU")


def _prepare_mxfp8(a_bf16, b_bf16, swizzled: bool):
    sflayout = SfLayout.layout_128x4 if swizzled else SfLayout.layout_linear
    a_fp8, a_sf = mxfp8_quantize(a_bf16, sf_swizzle_layout=sflayout)
    b_fp8, b_sf = mxfp8_quantize(b_bf16, sf_swizzle_layout=sflayout)
    if not swizzled:
        m, k = a_bf16.shape
        n = b_bf16.shape[0]
        a_sf = a_sf.view(m, k // 32)
        b_sf = b_sf.view(n, k // 32).t()
    return a_fp8, b_fp8, a_sf, b_sf


# Swizzled (layout_128x4) scale: mxfp8_quantize pads the scale buffer to pad_up(M, 128)
# rows internally, so arbitrary M is supported.
@pytest.mark.parametrize("m", [1, 17, 100, 128, 256, 512, 1024])
@pytest.mark.parametrize("n", [128, 256, 512, 1024])
@pytest.mark.parametrize("k", [128, 256, 512, 1024])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mm_mxfp8_sm120_swizzled(m, n, k, out_dtype):
    _skip_if_not_sm120()

    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)

    a_fp8, b_fp8, a_sf, b_sf = _prepare_mxfp8(a, b, swizzled=True)
    reference = torch.mm(a, b.T)

    result = mm_mxfp8(
        a_fp8, b_fp8.T, a_sf, b_sf, out_dtype=out_dtype, backend="cutlass"
    )

    assert result.shape == (m, n)
    assert result.dtype == out_dtype
    assert torch.isfinite(result).all(), "Output contains NaN/Inf"

    cos_sim = F.cosine_similarity(
        reference.reshape(-1).float(), result.reshape(-1).float(), dim=0
    ).item()
    assert cos_sim > 0.99, f"cos_sim={cos_sim:.4f} < 0.99 for M={m},N={n},K={k}"


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mm_mxfp8_sm120_all_tactics(out_dtype):
    """Every advertised tactic must run and match the reference.

    No hardcoded tactic count or index list: whatever the module advertises
    has to work on the device the test runs on.
    """
    _skip_if_not_sm120()
    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8

    m, n, k = 512, 512, 512
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_fp8, b_fp8, a_sf, b_sf = _prepare_mxfp8(a, b, swizzled=True)
    reference = torch.mm(a, b.T)

    module = gen_gemm_sm120_module_cutlass_mxfp8().build_and_load()
    num_tactics = module.mxfp8_gemm_tactic_num()
    assert num_tactics > 0
    workspace = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for tactic in range(num_tactics):
        out = torch.empty(m, n, dtype=out_dtype, device="cuda")
        module.mxfp8_gemm(a_fp8, b_fp8, a_sf, b_sf, out, workspace, tactic)
        assert torch.isfinite(out).all(), f"tactic {tactic}: output contains NaN/Inf"
        cos_sim = F.cosine_similarity(
            reference.reshape(-1).float(), out.reshape(-1).float(), dim=0
        ).item()
        assert cos_sim > 0.99, f"tactic {tactic}: cos_sim={cos_sim:.4f}"


def test_mm_mxfp8_sm120_auto_tactic():
    """Verify SM120 MXFP8 produces correct results (tactic auto-selected)."""
    _skip_if_not_sm120()
    from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8

    m, n, k = 256, 256, 256
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_fp8, b_fp8, a_sf, b_sf = _prepare_mxfp8(a, b, swizzled=True)
    reference = torch.mm(a, b.T)

    module = gen_gemm_sm120_module_cutlass_mxfp8().build_and_load()
    num_tactics = module.mxfp8_gemm_tactic_num()
    assert num_tactics > 0

    result = mm_mxfp8(
        a_fp8,
        b_fp8.T,
        a_sf,
        b_sf,
        out_dtype=torch.bfloat16,
        backend="cutlass",
    )
    assert result.shape == (m, n)
    assert torch.isfinite(result).all()
    cos_sim = F.cosine_similarity(
        reference.reshape(-1).float(), result.reshape(-1).float(), dim=0
    ).item()
    assert cos_sim > 0.98, f"cos_sim={cos_sim:.4f}"


def test_mm_mxfp8_sm120_rejects_linear_scales():
    """SM120 CUTLASS MXFP8 must reject non-swizzled (2D) scale tensors."""
    _skip_if_not_sm120()

    m, n, k = 128, 128, 128
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_fp8, b_fp8, a_sf, b_sf = _prepare_mxfp8(a, b, swizzled=False)

    with pytest.raises((RuntimeError, ValueError)):
        mm_mxfp8(
            a_fp8, b_fp8.T, a_sf, b_sf, out_dtype=torch.bfloat16, backend="cutlass"
        )


def _concurrent_replay_mismatches(launch, outs, replays=60):
    """Replay `launch(out)` for every out in a CUDA graph beside a memory-bound
    branch on a second stream; count replays whose output differs bitwise from
    the eager result. Outputs are NaN-poisoned before every replay."""
    launch(outs[0])
    torch.cuda.synchronize()
    expected = outs[0].clone()
    noise = torch.ones(64 * 2**20, device="cuda")  # 256 MB
    side = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        main = torch.cuda.current_stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            for _ in range(4):
                noise.mul_(1.0)
        for out in outs:
            launch(out)
        main.wait_stream(side)
    bad = 0
    for _ in range(replays):
        for out in outs:
            out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        bad += any(not torch.equal(out, expected) for out in outs)
    return bad


@pytest.mark.parametrize("m", [1, 4])
def test_mm_mxfp8_b12x_concurrent_graph_replays_are_exact(m):
    """A released mainloop smem stage must not be overwritten by the next TMA
    while the MMA warps' generic-proxy reads are still in flight (needs
    fence.proxy.async before consumer_release). The race shows up when another
    stream's DRAM traffic overlaps the GEMM."""
    _skip_if_not_sm120()
    from flashinfer.gemm.gemm_mm_mxfp8_cute_dsl import _b12x_mxfp8_dsl_supported

    if not _b12x_mxfp8_dsl_supported():
        pytest.skip("b12x mm_mxfp8 requires nvidia-cutlass-dsl >= 4.6.0")
    n, k = 4096, 2048  # DeepSeek-V4.1-Flash attention O-projection per TP4 rank
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_fp8, b_fp8, a_sf, b_sf = _prepare_mxfp8(a, b, swizzled=True)
    outs = [torch.empty([m, n], device="cuda", dtype=torch.bfloat16) for _ in range(8)]

    def launch(out):
        mm_mxfp8(a_fp8, b_fp8.T, a_sf, b_sf, out=out, backend="b12x")

    bad = _concurrent_replay_mismatches(launch, outs)
    assert bad == 0, f"{bad}/60 concurrent graph replays differ from eager (M={m})"
