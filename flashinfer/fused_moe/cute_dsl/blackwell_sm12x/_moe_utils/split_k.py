"""Device-side dynamic split-K for the SM12x decode fc1_act_q1 swap-AB tile.

At decode a rank has only D = 1..4 experts with tokens, so fc1 has T = 18*D (8,128) tiles on
188 persistent CTAs, and each CTA streams its 655 KB of weights through a 2-stage TMA ring at
~28 GB/s (the per-CTA rate holds even with the weights already in L2).  The scheduler warp
counts T on device (no host sync, CUDA-graph safe) and splits every tile's K into
S = min(grid // T, S_max, k_cycles) slices, one CTA each:

  * S == 1: the original schedule, loads and math (bitwise = the unsplit kernel).
  * S > 1:  CTA u = bidx works on tile t = u // S, K-cycles [s*C/S, (s+1)*C/S) with
            s = u % S.  Every slice writes its FP32 (up, gate) accumulators to scratch slot u,
            fences, and bumps counter[t]; the slice that arrives last resets counter[t] to 0
            and sums the S slots in slice order 0..S-1 (a fixed order, so the result is
            run-to-run deterministic), then runs the unchanged SwiGLU + MXFP8 epilogue.

``MAX_SPLITS`` (4) caps S and ``MIN_SPLITS`` (4) is the smallest S used (below it the tile
stays whole).  The choice depends only on the live tile count, so it follows the per-rank
shape (e.g. 18 fc1 tiles/expert at EP4 inter 2304, 9 at EP2xTP2 inter 1152).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Float32, Int32, Int64, T, dsl_user_op

#: Largest and smallest split count S (S below MIN_SPLITS keeps the tile whole).
MAX_SPLITS = 4
MIN_SPLITS = 4
#: FP32 accumulator values per thread a slot can hold (fc1 swap (128 x 8) tile: 2 x 4).
SLOT_VALUES_PER_THREAD = 16
SLOT_THREADS = 256


_WS: dict = {}


def workspace(device, grid_x: int):
    """(scratch fp32 [grid * slot], counters int32 [grid]); counters start at 0 and every
    split tile's last CTA puts its counter back to 0.  Allocated once per device, before any
    graph capture (the decode leaf warms every batch size first)."""
    k = (torch.device(device), grid_x)
    ws = _WS.get(k)
    if ws is None:
        scratch = torch.zeros(
            grid_x * SLOT_VALUES_PER_THREAD * SLOT_THREADS,
            dtype=torch.float32,
            device=device,
        )
        counters = torch.zeros(grid_x, dtype=torch.int32, device=device)
        ws = (scratch, counters)
        _WS[k] = ws
    return ws


@dsl_user_op
def _addr(tensor: cute.Tensor, *, loc=None, ip=None) -> Int64:
    return Int64(llvm.ptrtoint(T.i64(), tensor.iterator.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def st_f32(addr: Int64, v: Float32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_cg_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.cg.f32 $0, [$1];",
            "=f,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def atom_add_i32(addr: Int64, v: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
            "atom.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def atom_add_acqrel_i32(addr: Int64, v: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_i32(addr: Int64, v: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
        "st.global.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def membar_gl(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "membar.gl;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _bar_red_or_asm(bar_id, nthreads):
    return (
        "{ .reg .pred q, r; setp.ne.s32 q, $1, 0; "
        f"bar.red.or.pred r, {bar_id}, {nthreads}, q; "
        "selp.s32 $0, 1, 0, r; }"
    )


@dsl_user_op
def bar_red_or(flag: Int32, bar_id: int, nthreads: int, *, loc=None, ip=None) -> Int32:
    """Named barrier over nthreads threads; returns 1 on every thread if any flag != 0."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(flag).ir_value(loc=loc, ip=ip)],
            _bar_red_or_asm(bar_id, nthreads),
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


_UNROLL_WINDOWS = 4  # groups 0..127 are counted with independent loads (one round trip)


def count_loads(offsets: cute.Tensor, num_groups):
    """Issue the m-block count's loads (no use yet), so they are in flight together with
    the scheduler's own first lookup.  Finish with count_m_blocks_lane(*loads, ...)."""
    lane = cute.arch.lane_idx()
    ng = Int32(num_groups)
    vals = []
    for w in range(_UNROLL_WINDOWS):
        g = lane + Int32(32 * w)
        vals.append(offsets[cute.min(g, ng)])
        vals.append(offsets[cute.min(g + Int32(1), ng)])
    return vals


@cute.jit
def count_m_blocks_lane(
    l0, h0, l1, h1, l2, h2, l3, h3, offsets: cute.Tensor, num_groups, pm
):
    """Per-lane part of the m-block count from count_loads' values (+ any groups >= 128);
    reduce with warp_total."""
    lane = cute.arch.lane_idx()
    ng = Int32(num_groups)
    p = Int32(pm - 1)
    total = (h0 - l0 + p) // Int32(pm) + (h1 - l1 + p) // Int32(pm)
    total = total + (h2 - l2 + p) // Int32(pm) + (h3 - l3 + p) // Int32(pm)
    base = Int32(32 * _UNROLL_WINDOWS)
    while base < ng:
        g = base + lane
        lo = offsets[cute.min(g, ng)]
        hi = offsets[cute.min(g + Int32(1), ng)]
        total = total + (hi - lo + Int32(pm - 1)) // Int32(pm)
        base = base + Int32(32)
    return total


@dsl_user_op
def warp_total(v: Int32, *, loc=None, ip=None) -> Int32:
    """Sum over the full warp in one instruction (redux.sync, SM80+)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(v).ir_value(loc=loc, ip=ip)],
            "redux.sync.add.s32 $0, $1, 0xffffffff;",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def splits_of(valid):
    """S of a published work tile (0 for an unsplit tile)."""
    return (valid >> Int32(16)) & Int32(0xFF)


def cycle_range(valid, bidx, cycles):
    """[lo, hi) K-cycles of this CTA's slice; (0, cycles) for an unsplit tile."""
    S = cute.max(splits_of(valid), Int32(1))
    s = bidx % S
    return (s * cycles) // S, ((s + Int32(1)) * cycles) // S


def choose_splits(T_tiles, grid_x: int, s_max: int, cycles, s_min: int):
    """S = min(grid // T, s_max, cycles), or 1 when that is below s_min (measured: at S = 2
    the partial write + ordered reduction costs about what the split saves)."""
    # Division-free (it sits on the first tile's critical path): the largest
    # c in [s_min, s_max] with c * T <= grid and c <= cycles, else 1.
    S = Int32(1)
    for c in range(s_min, s_max + 1):
        fits = (Int32(grid_x) - Int32(c) * T_tiles) | (Int32(cycles) - Int32(c))
        ok = Int32(1) + (fits >> Int32(31))  # 1 if both >= 0 else 0
        S = S + (Int32(c) - S) * ok
    return S


@cute.jit
def reduce_partials(
    acc_u,
    acc_g,
    gScratch: cute.Tensor,
    gCnt: cute.Tensor,
    bidx,
    S,
    tidx,
    nthreads: cutlass.Constexpr,
    bar_id: cutlass.Constexpr,
    use_acqrel: cutlass.Constexpr = False,
):
    """All nthreads math threads of a split CTA.  Returns 1 on the CTA that must finish the
    tile (with acc_u / acc_g holding the slice-ordered sum), 0 elsewhere.

    use_acqrel: bar.sync -> thread 0 atom.acq_rel.gpu -> bar.red.or, the CUTLASS split-K
    semaphore pattern (release of the CTA's partial stores by cumulativity, acquire of the
    other slices' partials), instead of per-thread membar.gl before the count and again before
    the slot loads (two dependent L2 round trips less on the last slice).  Same values, same
    slice order: the sum is bitwise unchanged."""
    nv = cute.size(acc_u)
    slot_bytes = 2 * nv * nthreads * 4
    base = _addr(gScratch)
    my = base + Int64(bidx) * Int64(slot_bytes) + Int64(tidx) * Int64(4)
    for i in cutlass.range_constexpr(nv):
        st_f32(my + Int64(i * nthreads * 4), acc_u[i])
        st_f32(my + Int64((nv + i) * nthreads * 4), acc_g[i])
    if cutlass.const_expr(not use_acqrel):
        membar_gl()
    cute.arch.barrier(barrier_id=bar_id, number_of_threads=nthreads)
    t = bidx // S
    flag = Int32(0)
    if tidx == Int32(0):
        caddr = _addr(gCnt) + Int64(t) * Int64(4)
        old = Int32(0)
        if cutlass.const_expr(use_acqrel):
            old = atom_add_acqrel_i32(caddr, Int32(1))
        else:
            old = atom_add_i32(caddr, Int32(1))
        if old == S - Int32(1):
            st_i32(caddr, Int32(0))
            flag = Int32(1)
    last = bar_red_or(flag, bar_id, nthreads)
    if last != Int32(0):
        if cutlass.const_expr(not use_acqrel):
            membar_gl()
        first = base + Int64(t * S) * Int64(slot_bytes) + Int64(tidx) * Int64(4)
        for i in cutlass.range_constexpr(nv):
            acc_u[i] = ld_cg_f32(first + Int64(i * nthreads * 4))
            acc_g[i] = ld_cg_f32(first + Int64((nv + i) * nthreads * 4))
        j = Int32(1)
        while j < S:
            p = first + Int64(j) * Int64(slot_bytes)
            for i in cutlass.range_constexpr(nv):
                acc_u[i] = acc_u[i] + ld_cg_f32(p + Int64(i * nthreads * 4))
                acc_g[i] = acc_g[i] + ld_cg_f32(p + Int64((nv + i) * nthreads * 4))
            j = j + Int32(1)
    return last
