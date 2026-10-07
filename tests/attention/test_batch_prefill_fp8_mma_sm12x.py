"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# FP8-KV prefill (paged or ragged) on SM12x runs with FP8 math (Q and P rounded to e4m3, FP8 mma.sync) when
# the batch is long enough. Two kinds of inputs:
# * exact: Q is on the e4m3 grid with row amax 448, K rows are one-hot with value 1 and sm_scale makes
#   every log2-domain score an integer, so every FP8 rounding in the kernel is exact and the output must
#   match FP64 attention up to 16-bit output rounding. Any layout, paging, masking or merge bug shows.
# * random: Gaussian inputs; the error vs exact attention stays at the FP8 rounding level.

import math

import pytest
import torch

import flashinfer

LOG2E = 1.4426950408889634
E4M3 = torch.float8_e4m3fn


def _sm12x():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


pytestmark = pytest.mark.skipif(
    not _sm12x(), reason="FP8-MMA prefill runs on SM12x only"
)


def _make(
    q_lens,
    kv_lens,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    layout,
    q_dtype,
    exact,
    gen,
    kv_dtype=E4M3,
):
    dev = "cuda"
    pages_per = [math.ceil(n / page_size) for n in kv_lens]
    total_pages = sum(pages_per) + 3
    perm = torch.randperm(total_pages, generator=gen).tolist()  # scattered pages
    indices, indptr = [], [0]
    for n in pages_per:
        indices += perm[indptr[-1] : indptr[-1] + n]
        indptr.append(indptr[-1] + n)
    last = [n - (p - 1) * page_size for n, p in zip(kv_lens, pages_per, strict=True)]
    shape = (total_pages, 2, num_kv_heads, page_size, head_dim)  # HND per-page layout
    if exact:
        kv = torch.zeros(shape)
        dims = torch.randint(
            1, head_dim, (total_pages, num_kv_heads, page_size), generator=gen
        )
        kv[:, 0].scatter_(-1, dims.unsqueeze(-1), 1.0)
        kv[:, 1] = torch.randn(
            total_pages, num_kv_heads, page_size, head_dim, generator=gen
        )
        q = (
            -16.0
            * torch.randint(
                0, 13, (sum(q_lens), num_qo_heads, head_dim), generator=gen
            ).float()
        )
        q[..., 0] = 448.0
    else:
        kv = torch.randn(shape, generator=gen)
        q = torch.randn(sum(q_lens), num_qo_heads, head_dim, generator=gen)
    kv8 = kv.to(kv_dtype)
    # Garbage past each sequence end in its last page (e4m3 NaN), which must not leak into the output.
    raw = kv8.view(torch.uint8)
    for r, n in enumerate(kv_lens):
        if n % page_size:
            raw[indices[indptr[r + 1] - 1], :, :, n % page_size :] = 0x7F
    if layout == "NHD":
        kv8 = kv8.permute(0, 1, 3, 2, 4).contiguous()
    i32 = dict(dtype=torch.int32, device=dev)
    qo = [0]
    for n in q_lens:
        qo.append(qo[-1] + n)
    return (
        q.to(q_dtype).to(dev),
        kv8.to(dev),
        torch.tensor(qo, **i32),
        torch.tensor(indptr, **i32),
        torch.tensor(indices, **i32),
        torch.tensor(last, **i32),
    )


def _attention(
    qt, K, V, q_pos, group, sm_scale, causal, window_left, logits_soft_cap, keep=None
):
    """FP64 attention of rows qt [t, H, D] at positions q_pos over K/V [H_kv, n, D]; LSE in log2."""
    Kh = K.repeat_interleave(group, dim=0)
    Vh = V.repeat_interleave(group, dim=0)
    s = torch.einsum("thd,hld->thl", qt, Kh) * sm_scale
    if logits_soft_cap > 0:
        s = logits_soft_cap * torch.tanh(s / logits_soft_cap)
    pos = torch.arange(K.shape[1], device=qt.device)
    if causal:
        s = s.masked_fill(pos[None, None, :] > q_pos[:, None, None], float("-inf"))
    if window_left >= 0:
        s = s.masked_fill(
            pos[None, None, :] < (q_pos - window_left)[:, None, None], float("-inf")
        )
    if keep is not None:
        s = s.masked_fill(~keep[:, None, :], float("-inf"))
    return torch.einsum("thl,hld->thd", torch.softmax(s, dim=-1), Vh), torch.logsumexp(
        s, dim=-1
    ) * LOG2E


def _sample_rows(n_q, rows_per_req, gen):
    tok = sorted(
        {0, 1, 7, 8, 15, 16, n_q // 2, n_q - 2, n_q - 1}
        | set(torch.randint(0, n_q, (rows_per_req,), generator=gen).tolist())
    )
    return [t for t in tok if 0 <= t < n_q]


def _reference(
    q,
    kv8,
    layout,
    qo,
    kv_indptr,
    kv_indices,
    kv_lens,
    page_size,
    num_kv_heads,
    sm_scale,
    causal,
    k_scale,
    v_scale,
    rows_per_req=24,
    gen=None,
    window_left=-1,
    logits_soft_cap=0.0,
    keep_fn=None,
):
    """FP64 attention on sampled query tokens (all heads) of every request; also returns LSE in log2.
    keep_fn(request, query tokens, kv_len) gives an extra [tokens, kv_len] keep mask."""
    outs, lses, picks = [], [], []
    num_qo_heads = q.shape[1]
    group = num_qo_heads // num_kv_heads
    kvd = kv8.double() if layout == "HND" else kv8.permute(0, 1, 3, 2, 4).double()
    for r, n_kv in enumerate(kv_lens):
        q0, q1 = int(qo[r]), int(qo[r + 1])
        n_q = q1 - q0
        pages = kv_indices[int(kv_indptr[r]) : int(kv_indptr[r + 1])].long()
        kvr = kvd[pages]  # [p, 2, H, page, D]
        K = (
            kvr[:, 0]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, q.shape[2])[:, :n_kv]
            * k_scale
        )
        V = (
            kvr[:, 1]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, q.shape[2])[:, :n_kv]
            * v_scale
        )
        tok = _sample_rows(n_q, rows_per_req, gen)
        idx = q0 + torch.tensor(tok, device=q.device)
        q_pos = (n_kv - n_q) + torch.tensor(tok, device=q.device)
        keep = keep_fn(r, torch.tensor(tok, device=q.device), n_kv) if keep_fn else None
        out, lse = _attention(
            q[idx].double(),
            K,
            V,
            q_pos,
            group,
            sm_scale,
            causal,
            window_left,
            logits_soft_cap,
            keep,
        )
        outs.append(out)
        lses.append(lse)
        picks.append(idx)
    return torch.cat(picks), torch.cat(outs), torch.cat(lses)


def _check(out, lse, rows, ref, ref_lse, exact):
    assert torch.isfinite(out).all()
    got = out[rows].double()
    rel = ((got - ref).norm() / ref.norm()).item()
    lse_err = (lse[rows].double() - ref_lse).abs().max().item()
    if exact:
        assert rel < 8e-3, rel  # 16-bit output (and split-KV partial) rounding only
        assert lse_err < 2e-3, lse_err
    else:
        assert rel < 6e-2, rel  # FP8 rounding of Q and P
        assert lse_err < 0.25, lse_err


def _run(
    q_lens,
    kv_lens,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    layout,
    causal,
    q_dtype,
    exact,
    k_scale=1.0,
    v_scale=1.0,
    use_cuda_graph=False,
    expect_fp8_mma=True,
    seed=0,
    window_left=-1,
    logits_soft_cap=0.0,
    kv_dtype=E4M3,
    custom_mask_fn=None,
    multi_item=None,
):
    """custom_mask_fn(request, qo_len, kv_len) -> [qo_len, kv_len] bool mask. multi_item = (prefix lens,
    per-request token-position-in-item lists) for multi-item scoring."""
    gen = torch.Generator().manual_seed(seed)
    q, kv8, qo, kv_indptr, kv_indices, last = _make(
        q_lens,
        kv_lens,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        layout,
        q_dtype,
        exact,
        gen,
        kv_dtype,
    )
    sm_scale = (1.0 / 16 / k_scale / LOG2E) if exact else head_dim**-0.5
    ws = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    kwargs, plan_kwargs, keep_fn = {}, {}, None
    if use_cuda_graph:
        kwargs = dict(
            use_cuda_graph=True,
            qo_indptr_buf=qo.clone(),
            paged_kv_indptr_buf=kv_indptr.clone(),
            paged_kv_indices_buf=kv_indices.clone(),
            paged_kv_last_page_len_buf=last.clone(),
        )
    if custom_mask_fn is not None:
        masks = [
            custom_mask_fn(r, n_q, n_kv).cuda()
            for r, (n_q, n_kv) in enumerate(zip(q_lens, kv_lens, strict=True))
        ]
        plan_kwargs["custom_mask"] = torch.cat([m.flatten() for m in masks])
        keep_fn = lambda r, tok, n_kv: masks[r][tok]  # noqa: E731
    if multi_item is not None:
        prefix_lens, item_pos = multi_item
        pos_len = max(len(p) for p in item_pos)
        item_pos_buf = torch.zeros(len(q_lens), pos_len, dtype=torch.int32)
        for r, p in enumerate(item_pos):
            item_pos_buf[r, : len(p)] = torch.tensor(p)
        plan_kwargs.update(
            prefix_len_ptr=torch.tensor(prefix_lens, dtype=torch.uint32, device="cuda"),
            token_pos_in_items_ptr=item_pos_buf.flatten().to(torch.uint16).cuda(),
            token_pos_in_items_len=pos_len,
            max_item_len_ptr=torch.tensor(
                [max(p) for p in item_pos], dtype=torch.uint16, device="cuda"
            ),
        )

        def keep_fn(r, tok, n_kv):
            # FA2's logits_mask_multi_item_scoring: a row inside an item sees the prefix and its own item.
            q_pos = (n_kv - q_lens[r]) + tok
            pos = torch.arange(n_kv, device=tok.device)
            prefix = prefix_lens[r]
            in_item = q_pos >= prefix
            tpi = item_pos_buf[r].to(tok.device)[(q_pos - prefix).clamp(min=0)]
            own = q_pos[:, None] < pos[None, :] + tpi[:, None]
            return ~in_item[:, None] | (pos[None, :] < prefix) | own

    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, layout, backend="fa2", **kwargs
    )
    w.plan(
        qo,
        kv_indptr,
        kv_indices,
        last,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        sm_scale=sm_scale,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
        o_data_type=q_dtype,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap or None,
        **plan_kwargs,
    )
    assert bool(w._plan_info[15]) == expect_fp8_mma, "unexpected kernel choice"
    out, lse = w.run(q, kv8, return_lse=True, k_scale=k_scale, v_scale=v_scale)
    rows, ref, ref_lse = _reference(
        q,
        kv8,
        layout,
        qo.cpu(),
        kv_indptr.cpu(),
        kv_indices.cpu(),
        kv_lens,
        page_size,
        num_kv_heads,
        sm_scale,
        causal or multi_item is not None,
        k_scale,
        v_scale,
        gen=gen,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        keep_fn=keep_fn,
    )
    _check(out, lse, rows, ref, ref_lse, exact)


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("group", [1, 2, 5, 8])
@pytest.mark.parametrize("causal", [True, False])
def test_shapes(exact, head_dim, group, causal):
    # 300 query tokens x group packed rows >= 256 for every group size, so the FP8-MMA path is taken.
    _run(
        [300, 513],
        [2048, 1100],
        2 * group,
        2,
        head_dim,
        32,
        "HND",
        causal,
        torch.bfloat16,
        exact,
    )


@pytest.mark.parametrize("page_size", [2, 4, 8, 16, 64, 128, 256])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("head_dim", [64, 256])
def test_page_sizes_and_layouts(page_size, layout, head_dim):
    _run(
        [257, 64, 1000],
        [3000, 64, 1500],
        16,
        2,
        head_dim,
        page_size,
        layout,
        True,
        torch.bfloat16,
        True,
    )


@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
def test_dtypes_and_scales(q_dtype):
    _run(
        [400],
        [5000],
        16,
        2,
        128,
        16,
        "HND",
        True,
        q_dtype,
        True,
        k_scale=0.5,
        v_scale=2.0,
    )
    _run(
        [400],
        [5000],
        16,
        2,
        128,
        16,
        "HND",
        True,
        q_dtype,
        False,
        k_scale=0.5,
        v_scale=2.0,
    )


def test_long_split_kv():
    # One long request: few CTAs, so the plan splits the KV range and merges partials.
    _run([2048], [34816], 16, 2, 256, 32, "HND", True, torch.bfloat16, True)
    _run([2048], [34816], 16, 2, 256, 32, "HND", True, torch.bfloat16, False)
    _run(
        [96], [34816], 16, 2, 256, 32, "HND", False, torch.bfloat16, True
    )  # cascade-level-0 shape


@pytest.mark.parametrize("window_left", [0, 37, 511, 3000])
def test_sliding_window(window_left):
    _run(
        [300, 2048],
        [2048, 34816],
        16,
        2,
        256,
        32,
        "HND",
        True,
        torch.bfloat16,
        True,
        window_left=window_left,
    )
    _run(
        [700],
        [9000],
        8,
        2,
        128,
        16,
        "NHD",
        True,
        torch.bfloat16,
        True,
        window_left=window_left,
    )


def test_non_causal_sliding_window_keeps_fa2():
    # The plan sizes window ranges for causal windows, so non-causal windows stay on FA2.
    _run(
        [300],
        [2048],
        16,
        2,
        256,
        32,
        "HND",
        False,
        torch.bfloat16,
        True,
        window_left=4096,
        expect_fp8_mma=False,
    )


@pytest.mark.parametrize("logits_soft_cap", [30.0, 50.0])
@pytest.mark.parametrize("window_left", [-1, 100])
def test_logits_soft_cap(logits_soft_cap, window_left):
    _run(
        [300, 2048],
        [2048, 34816],
        16,
        2,
        256,
        32,
        "HND",
        True,
        torch.bfloat16,
        False,
        logits_soft_cap=logits_soft_cap,
        window_left=window_left,
    )


@pytest.mark.parametrize("exact", [True, False])
def test_e5m2_kv(exact):
    _run(
        [300, 2048],
        [2048, 34816],
        16,
        2,
        256,
        32,
        "HND",
        True,
        torch.bfloat16,
        exact,
        kv_dtype=torch.float8_e5m2,
    )


def test_cuda_graph_plan():
    _run(
        [600, 600],
        [4096, 2000],
        16,
        2,
        256,
        32,
        "HND",
        True,
        torch.bfloat16,
        True,
        use_cuda_graph=True,
    )


@pytest.mark.parametrize("page_size", [1, 2])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_tiny_pages(page_size, layout, head_dim):
    # Pages of 1 or 2 rows are loaded four rows per TMA copy (gather4).
    _run(
        [300, 64],
        [2000, 100],
        16,
        2,
        head_dim,
        page_size,
        layout,
        True,
        torch.bfloat16,
        True,
    )
    _run([300], [2000], 8, 2, head_dim, page_size, layout, False, torch.bfloat16, False)


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_custom_mask(exact, head_dim):
    def mask_fn(r, n_q, n_kv):
        # Random keep pattern; the diagonal stays visible so no row is empty.
        g = torch.Generator().manual_seed(100 + r)
        m = torch.rand(n_q, n_kv, generator=g) < 0.6
        m[torch.arange(n_q), torch.arange(n_q) + (n_kv - n_q)] = True
        return m

    _run(
        [300, 513],
        [2048, 1100],
        16,
        2,
        head_dim,
        16,
        "HND",
        False,
        torch.bfloat16,
        exact,
        custom_mask_fn=mask_fn,
    )


def test_custom_mask_split_kv_and_window():
    def causal_band(r, n_q, n_kv):
        pos = torch.arange(n_kv)
        q_pos = torch.arange(n_q) + (n_kv - n_q)
        return (pos[None, :] <= q_pos[:, None]) & (pos[None, :] > q_pos[:, None] - 700)

    _run(
        [2048],
        [34816],
        16,
        2,
        256,
        32,
        "HND",
        False,
        torch.bfloat16,
        True,
        custom_mask_fn=causal_band,
    )
    _run(
        [300],
        [4000],
        16,
        2,
        128,
        16,
        "NHD",
        True,
        torch.bfloat16,
        True,
        custom_mask_fn=causal_band,
        window_left=300,
    )


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_multi_item_scoring(exact, head_dim):
    # Request 0: a 200-token prefix (the last 50 tokens of it are new) and items of 37, 80 and 13 tokens,
    # each led by a delimiter (position 0 in its item). Request 1: a cached 400-token prefix and 3 items.
    def items(lengths):
        return [p for n in lengths for p in [0] + list(range(1, n))]

    pos0 = items([37, 80, 13]) + [0]
    pos1 = items([100, 157, 50])
    q_lens = [50 + len(pos0), len(pos1)]
    kv_lens = [200 + len(pos0), 400 + len(pos1)]
    _run(
        q_lens,
        kv_lens,
        16,
        2,
        head_dim,
        16,
        "HND",
        True,
        torch.bfloat16,
        exact,
        multi_item=([200, 400], [pos0, pos1]),
    )


def test_short_batches_keep_fa2():
    # Decode-like batches (few packed rows per request) stay on the FA2 path.
    _run(
        [1] * 16,
        [1000] * 16,
        16,
        2,
        256,
        32,
        "HND",
        True,
        torch.bfloat16,
        True,
        expect_fp8_mma=False,
    )
    _run(
        [20],
        [1000],
        16,
        2,
        256,
        32,
        "HND",
        True,
        torch.bfloat16,
        True,
        expect_fp8_mma=False,
    )


def _run_ragged(
    q_lens,
    kv_lens,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    layout,
    causal,
    exact,
    seed=0,
    expect_fp8_mma=True,
    window_left=-1,
):
    gen = torch.Generator().manual_seed(seed)
    total_kv, pad = sum(kv_lens), 37
    if exact:
        k = torch.zeros(total_kv + pad, num_kv_heads, head_dim)
        k.scatter_(
            -1,
            torch.randint(
                1, head_dim, (total_kv + pad, num_kv_heads, 1), generator=gen
            ),
            1.0,
        )
        q = (
            -16.0
            * torch.randint(
                0, 13, (sum(q_lens), num_qo_heads, head_dim), generator=gen
            ).float()
        )
        q[..., 0] = 448.0
    else:
        k = torch.randn(total_kv + pad, num_kv_heads, head_dim, generator=gen)
        q = torch.randn(sum(q_lens), num_qo_heads, head_dim, generator=gen)
    v = torch.randn(total_kv + pad, num_kv_heads, head_dim, generator=gen)
    k8, v8 = k.to(E4M3).cuda(), v.to(E4M3).cuda()
    # Rows past the last request (NaN) are read by its tail tile and must not leak into the output.
    k8.view(torch.uint8)[total_kv:] = 0x7F
    v8.view(torch.uint8)[total_kv:] = 0x7F
    k8, v8 = k8[: total_kv + pad], v8[: total_kv + pad]
    if layout == "HND":
        k8, v8 = k8.transpose(0, 1).contiguous(), v8.transpose(0, 1).contiguous()
    q = q.to(torch.bfloat16).cuda()
    i32 = dict(dtype=torch.int32, device="cuda")
    qo = torch.tensor([0] + torch.tensor(q_lens).cumsum(0).tolist(), **i32)
    kv_indptr = torch.tensor([0] + torch.tensor(kv_lens).cumsum(0).tolist(), **i32)
    sm_scale = (1.0 / 16 / LOG2E) if exact else head_dim**-0.5
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(256 << 20, dtype=torch.uint8, device="cuda"), layout, backend="fa2"
    )
    w.plan(
        qo,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        sm_scale=sm_scale,
        q_data_type=torch.bfloat16,
        kv_data_type=E4M3,
        window_left=window_left,
    )
    assert bool(w._plan_info[15]) == expect_fp8_mma, "unexpected kernel choice"
    out, lse = w.run(q, k8, v8, return_lse=True)
    kd = (k8 if layout == "HND" else k8.transpose(0, 1)).double()
    vd = (v8 if layout == "HND" else v8.transpose(0, 1)).double()
    picks, outs, lses = [], [], []
    for r, (n_q, n_kv) in enumerate(zip(q_lens, kv_lens, strict=True)):
        k0 = int(kv_indptr[r])
        tok = _sample_rows(n_q, 24, gen)
        idx = int(qo[r]) + torch.tensor(tok, device="cuda")
        q_pos = (n_kv - n_q) + torch.tensor(tok, device="cuda")
        o_r, l_r = _attention(
            q[idx].double(),
            kd[:, k0 : k0 + n_kv],
            vd[:, k0 : k0 + n_kv],
            q_pos,
            num_qo_heads // num_kv_heads,
            sm_scale,
            causal,
            window_left,
            0.0,
        )
        picks.append(idx)
        outs.append(o_r)
        lses.append(l_r)
    _check(out, lse, torch.cat(picks), torch.cat(outs), torch.cat(lses), exact)


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("causal", [True, False])
def test_ragged(exact, head_dim, layout, causal):
    _run_ragged(
        [300, 70, 1000], [2048, 70, 1500], 16, 2, head_dim, layout, causal, exact
    )


def test_ragged_split_kv_and_window():
    _run_ragged([2048], [34816], 16, 2, 256, "NHD", True, True)
    _run_ragged([700, 300], [9000, 300], 8, 2, 128, "NHD", True, True, window_left=257)
