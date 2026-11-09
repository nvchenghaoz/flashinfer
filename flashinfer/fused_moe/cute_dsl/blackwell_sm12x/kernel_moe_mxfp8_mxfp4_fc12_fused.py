"""SM12x decode MoE: fc1_act_q1 (swap-AB (8,128) tile, device split-K) and fc2_finalize
((32,128) tile, routing-weight scatter) in ONE persistent kernel.

Why: at decode a rank has D = 1..12 active experts, and the two-kernel chain pays a whole
second kernel (launch, prologue, the PDL wait for fc1's full completion, teardown: 2.6 us at
D = 0 on an EP4 rank) and starts fc2's weight stream only when fc1's last mainloop ends.

How (one grid of num_SMs CTAs x 384 threads, 1 CTA/SM, warp roles as in both kernels):
  * ONE atomic work counter.  Units [0, U1) are fc1 units (T1 tiles x S split-K slices,
    S chosen exactly as fc1_act_q1 does), [U1, U1 + T2) are fc2 tiles.  The scheduler warp of
    a resident CTA claims units one at a time, so every fc1 unit is held by a running CTA
    before any fc2 unit is handed out: an fc2 waiter never waits on an unstarted fc1 tile.
  * fc1 units run fc1_act_q1's code unchanged (loads, mma_swap, split-K reduce, SwiGLU, MXFP8
    q1/sf1 epilogue), so q1/sf1 are bitwise those of fc1_act_q1.  The CTA that writes a tile's
    q1/sf1 then releases it: per-thread membar.gl + fence.proxy.async.global, a math-warp
    barrier, and one atom.release.gpu add on that expert's done counter.
  * fc2 tiles run fc2_finalize's code unchanged.  Their WEIGHT loader starts at once; only the
    q1 loader acquire-waits (ld.acquire.gpu) until its expert's counter reaches the expert's
    fc1 tile count, then fence.proxy.async.global, then TMA.
  * Both phases share one smem pool (fc1 and fc2 each fill the SM at their own ab_stage); a
    CTA changes phase once, behind a named barrier its math warps arrive at after their last
    fc1 epilogue (fence.proxy.async.shared::cta first).
  * Counters are reset in-kernel by the last CTA to exit; no memset node.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import (
    Int32,
    Int64,
    T,
    dsl_user_op,
    extract_mlir_values,
    new_from_mlir_values,
)

from ....utils import ceil_div
from . import kernel_moe_mxfp8_mxfp4_fc1_act_q1 as fc1k
from . import kernel_moe_mxfp8_mxfp4_fc2_finalize as fc2k
from ._moe_utils import moe_epilogue, moe_scheduler, split_k
from ._moe_utils.sm12x_blockscaled_layout import SF_M_ALIGN, UE8M0_PACK_NUM

PHASE2_BIT = 24
PREFETCH_BIT = 25
TRANS_BAR_ID = 5
SYNC_HEADER = (
    4  # [0] claim counter, [1] exit counter, [2..3] unused; [4 + e] expert e done
)
MAX_GROUPS = 1024
# [KDONE_BASE + e * KCOLS_MAX + n]: fc1 m-blocks of expert e whose column tile n is released
KCOLS_MAX = 32
KDONE_BASE = SYNC_HEADER + 2 * MAX_GROUPS
SYNC_WORDS = KDONE_BASE + MAX_GROUPS * KCOLS_MAX
#: KACQ (per-SF-cycle fc2 acquire): fc1 releases per (expert, column tile); fc2's q1 loader
#: waits per SF cycle only for the column tiles that cycle reads (its SFA pack spans several),
#: and in no-prefetch launches the weight loader waits only for cycle 0.
#: EARLY_W2: the fc2 weight loader of a CTA that ran an fc1 unit starts its first fc2 tile's
#: weights at once (into the fc1 stages its mainloop has drained), without the expert wait and
#: without the phase barrier; sB/sSFB must not overlap the fc1 epilogue's sC.
#: The split-K slice hand-off uses the acq_rel semaphore form (split_k.reduce_partials
#: use_acqrel=True).
#: nanosleep (ns) between acquire polls.
SLEEP_NS = 32
#: fc2 tiles stream their weights before the acquire only when fc1 has at least this many
#: tiles: at fewer (EP4 D <= 4), that early stream competes with fc1's own weight stream for
#: DRAM and costs more than it saves (measured on an 8-layer A/B).
PREFETCH_MIN_T1 = 81


# --------------------------------------------------------------------------- PTX helpers
def _asm(ret, args, text, cons, loc, ip):
    return llvm.inline_asm(
        ret,
        args,
        text,
        cons,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_acquire_i32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.acquire.gpu.s32 $0, [$1];",
            "=r,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def atom_add_release_i32(addr: Int64, v: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
            "atom.release.gpu.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def atom_add_relaxed_i32(addr: Int64, v: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), Int32(v).ir_value(loc=loc, ip=ip)],
            "atom.relaxed.gpu.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def fence_proxy_async_shared(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.shared::cta;", "", loc, ip)


@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.global;", "", loc, ip)


@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.gpu;", "", loc, ip)


@dsl_user_op
def nanosleep(ns: int, *, loc=None, ip=None):
    _asm(None, [], f"nanosleep.u32 {int(ns)};", "", loc, ip)


@dsl_user_op
def bar_arrive(bar_id: int, nthreads: int, *, loc=None, ip=None):
    _asm(None, [], f"bar.arrive {int(bar_id)}, {int(nthreads)};", "", loc, ip)


@dsl_user_op
def bar_sync(bar_id: int, nthreads: int, *, loc=None, ip=None):
    _asm(None, [], f"bar.sync {int(bar_id)}, {int(nthreads)};", "", loc, ip)


@dsl_user_op
def vote_all(flag: Int32, *, loc=None, ip=None) -> Int32:
    """1 on every lane if flag != 0 on every lane of the (full) warp."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(flag).ir_value(loc=loc, ip=ip)],
            "{ .reg .pred p, q; setp.ne.s32 p, $1, 0; "
            "vote.sync.all.pred q, p, 0xffffffff; selp.s32 $0, 1, 0, q; }",
            "=r,r",
            loc,
            ip,
        )
    )


def is_fc1(valid):
    return (valid & Int32(1)) * (Int32(1) - ((valid >> Int32(PHASE2_BIT)) & Int32(1)))


@cute.jit
def acquire_cols(
    kbase: Int64, c0, ncols: cutlass.Constexpr, nnb, need, sleep_ns: cutlass.Constexpr
):
    """Whole warp: wait until every column tile in [c0, min(c0 + ncols, nnb)) of one expert
    has `need` released fc1 m-blocks (lane l polls column c0 + l with ld.acquire.gpu), then
    bar.warp.sync so the elected lane's TMA is ordered after every lane's acquire."""
    lane = cute.arch.lane_idx()
    col = c0 + lane
    live = Int32(0)
    if lane < Int32(ncols):
        if col < nnb:
            live = Int32(1)
    ok = Int32(1)
    if live != Int32(0):
        seen = ld_acquire_i32(kbase + Int64(col) * Int64(4))
        if seen < need:
            ok = Int32(0)
    allok = vote_all(ok)
    while allok == Int32(0):
        nanosleep(sleep_ns)
        if ok == Int32(0):
            seen2 = ld_acquire_i32(kbase + Int64(col) * Int64(4))
            if seen2 >= need:
                ok = Int32(1)
        allok = vote_all(ok)
    cute.arch.sync_warp()


@cute.jit
def load_a_sfa_kacq(
    tma_atom_a,
    tma_atom_sfa,
    tma_tensor_a,
    tma_tensor_sfa,
    sA,
    sSFA,
    tile_mnk,
    tile,
    a_full,
    a_empty,
    a_bytes,
    sfa_bytes,
    num_sf_cycles,
    sf_stages: cutlass.Constexpr,
    kt_per_pack: cutlass.Constexpr,
    ab_stages: cutlass.Constexpr,
    a_phase,
    kbase: Int64,
    need,
    nnb1,
    cols_per_cycle: cutlass.Constexpr,
    sleep_ns: cutlass.Constexpr,
):
    """fc2_finalize's load_a_sfa (same TMA copies, same order) with a per-SF-cycle acquire:
    cycle c's q1 K-tiles and SFA pack read fc1 column tiles [c*cols, (c+1)*cols) only."""
    i32 = cutlass.Int32
    cluster = cute.make_layout((1, 1, 1))
    multicast = cute.make_layout(cute.slice_(cluster, (0, None, 0)).shape)
    mA = cute.domain_offset((tile.m_offset, 0), tma_tensor_a)
    gA_mkl = cute.local_tile(mA, cute.slice_(tile_mnk, (None, 0, None)), (None, None))
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        i32(0),
        multicast,
        cute.group_modes(sA, 0, 2),
        cute.group_modes(gA_mkl, 0, 2),
    )
    sf_m_off = (tile.m_offset + tile.group * i32(SF_M_ALIGN - 1)) & i32(-SF_M_ALIGN)
    mSFA = cute.domain_offset((sf_m_off, 0, 0), tma_tensor_sfa)
    gSFA_ml = cute.local_tile(mSFA, (tile_mnk[0], 1), (None, None, None))
    tAsSFA, tAgSFA = cpasync.tma_partition(
        tma_atom_sfa,
        i32(0),
        multicast,
        cute.group_modes(sSFA, 0, 2),
        cute.group_modes(gSFA_ml, 0, 2),
    )
    for sf_cycle in cutlass.range(num_sf_cycles):
        acquire_cols(
            kbase, sf_cycle * i32(cols_per_cycle), cols_per_cycle, nnb1, need, sleep_ns
        )
        fence_proxy_async_global()
        for sf_stage in cutlass.range_constexpr(sf_stages):
            global_sf_stage = sf_cycle * sf_stages + sf_stage
            for k_in_sf in cutlass.range_constexpr(kt_per_pack):
                local_k_tile = sf_stage * kt_per_pack + k_in_sf
                global_k_tile = sf_cycle * sf_stages * kt_per_pack + local_k_tile
                stage = local_k_tile & (ab_stages - 1)
                cute.arch.mbarrier_wait(a_empty + stage, a_phase)
                tx_bytes = a_bytes
                if cutlass.const_expr(k_in_sf == 0):
                    tx_bytes += sfa_bytes
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(a_full + stage, tx_bytes)
                if cutlass.const_expr(k_in_sf == 0):
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA[(None, tile.m_block, global_sf_stage, 0)],
                        tAsSFA[(None, sf_stage)],
                        tma_bar_ptr=a_full + stage,
                    )
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, tile.m_block, global_k_tile)],
                    tAsA[(None, stage)],
                    tma_bar_ptr=a_full + stage,
                )
                if cutlass.const_expr(stage == ab_stages - 1):
                    a_phase ^= 1
    return a_phase


# --------------------------------------------------------------------- work tile + pipe
class FusedWorkTile:
    FIELDS = 7
    VALID = 5

    def __init__(self, m_block, n_block, group, m_offset, m_boundary, valid, unit):
        self.m_block, self.n_block, self.group = m_block, n_block, group
        self.m_offset, self.m_boundary, self.valid = m_offset, m_boundary, valid
        self.unit = unit

    def _fields(self):
        return (
            self.m_block,
            self.n_block,
            self.group,
            self.m_offset,
            self.m_boundary,
            self.valid,
            self.unit,
        )

    def store(self, work, stage):
        for i, v in enumerate(self._fields()):
            work[stage, i] = v

    @staticmethod
    def load(work, stage):
        return FusedWorkTile(*(work[stage, i] for i in range(FusedWorkTile.FIELDS)))

    def __extract_mlir_values__(self):
        vals = []
        for f in self._fields():
            vals.extend(extract_mlir_values(f))
        return vals

    def __new_from_mlir_values__(self, values):
        new_fields, i = [], 0
        for f in self._fields():
            w = len(extract_mlir_values(f))
            new_fields.append(new_from_mlir_values(f, values[i : i + w]))
            i += w
        return FusedWorkTile(*new_fields)


class FusedSchedProducer:
    def __init__(self, stages, sched_iter):
        self.stages, self.sched_iter = stages, sched_iter

    @staticmethod
    def create(stages):
        return FusedSchedProducer(stages, Int32(0))

    def __extract_mlir_values__(self):
        return extract_mlir_values(self.sched_iter)

    def __new_from_mlir_values__(self, values):
        return FusedSchedProducer(
            self.stages, new_from_mlir_values(self.sched_iter, values)
        )

    @dsl_user_op
    @cute.jit
    def publish(self, work, full, empty, tile, *, loc=None, ip=None):
        stage = self.sched_iter % self.stages
        cute.arch.mbarrier_wait(
            empty + stage, ((self.sched_iter // self.stages) & 1) ^ 1
        )
        with cute.arch.elect_one():
            tile.store(work, stage)
            cute.arch.mbarrier_arrive(full + stage)
        self.sched_iter = self.sched_iter + Int32(1)

    @dsl_user_op
    @cute.jit
    def publish_sentinel(self, work, full, empty, *, loc=None, ip=None):
        stage = self.sched_iter % self.stages
        cute.arch.mbarrier_wait(
            empty + stage, ((self.sched_iter // self.stages) & 1) ^ 1
        )
        with cute.arch.elect_one():
            work[stage, FusedWorkTile.VALID] = Int32(0)
            cute.arch.mbarrier_arrive(full + stage)


class FusedSchedConsumer:
    def __init__(self, stages, sched_iter):
        self.stages, self.sched_iter = stages, sched_iter

    @staticmethod
    def create(stages):
        return FusedSchedConsumer(stages, Int32(0))

    def __extract_mlir_values__(self):
        return extract_mlir_values(self.sched_iter)

    def __new_from_mlir_values__(self, values):
        return FusedSchedConsumer(
            self.stages, new_from_mlir_values(self.sched_iter, values)
        )

    @dsl_user_op
    @cute.jit
    def get_next_tile(self, work, full, empty, *, loc=None, ip=None) -> FusedWorkTile:
        stage = self.sched_iter % self.stages
        cute.arch.mbarrier_wait(full + stage, (self.sched_iter // self.stages) & 1)
        tile = FusedWorkTile.load(work, stage)
        cute.arch.sync_warp()
        if tile.valid != Int32(0):
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(empty + stage)
            self.sched_iter = self.sched_iter + Int32(1)
        return tile


def _align(n, a=128):
    return (n + a - 1) // a * a


# ------------------------------------------------------------------------------- kernel
class CuteDslSm120MoeMxfp8Mxfp4Fc12Fused:
    def __init__(self, cfg1, cfg2, grid_x):
        self.cfg1, self.cfg2, self.grid_x = cfg1, cfg2, grid_x
        assert cfg1.mma.swap_ab, "fused kernel: fc1 must use the swap-AB decode tile"
        assert cfg1.threads == cfg2.threads
        assert cfg1.num_sched_consumers == cfg2.num_sched_consumers
        assert cfg1.sched_stages == cfg2.sched_stages
        assert cfg1.mma_threads == cfg2.mma_threads
        assert cfg2.union_smem
        self.sleep_ns = SLEEP_NS
        self.prefetch_min_t1 = PREFETCH_MIN_T1
        self.split_max = split_k.MAX_SPLITS
        self.split_min = split_k.MIN_SPLITS
        self.qcfg = fc1k._DEFAULT_QCFG
        self.sf1, self.sf2 = cfg1.load_sf, cfg2.load_sf
        bk1 = cfg1.TILE[2]
        assert cfg1.TILE[0] == fc1k.GRANK_A
        assert self.sf1.k_tiles_per_pack_a(bk1) == 1
        kt1 = self.sf1.sfb_stages(cfg1.ab_stage, bk1) * self.sf1.k_tiles_per_pack_b(bk1)
        assert kt1 % cfg1.ab_stage == 0
        bk2 = cfg2.TILE[2]
        assert self.sf2.k_tiles_per_pack_b(bk2) == 1
        kt2 = self.sf2.sfa_stages(cfg2.ab_stage, bk2) * self.sf2.k_tiles_per_pack_a(bk2)
        assert kt2 % cfg2.ab_stage == 0

    @cute.jit
    def __call__(
        self,
        gA1: cute.Tensor,
        gB1_u8: cute.Tensor,
        gSFA1_u8: cute.Tensor,
        gSFB1_u8: cute.Tensor,
        gQ1: cute.Tensor,
        gSFQ_u8: cute.Tensor,
        offsets: cute.Tensor,
        gSplit: cute.Tensor,
        gCnt: cute.Tensor,
        gA2: cute.Tensor,
        gB2_u8: cute.Tensor,
        gSFA2_u8: cute.Tensor,
        gSFB2_u8: cute.Tensor,
        gOut: cute.Tensor,
        gTok: cute.Tensor,
        gWt: cute.Tensor,
        gSync: cute.Tensor,
        stream,
    ):
        c1, c2 = self.cfg1, self.cfg2
        sf_dtype = c1.load_sf.sf_dtype
        # ---------------- fc1 (swap-AB): A = weights (U/G halves), B = activations
        tiledmma1 = c1.mma.make_tiled_mma(c1.TILE)
        tiled_r2s1 = c1.epi.make_tiled_r2s(tiledmma1)
        gB1_e = cute.recast_tensor(gB1_u8, cutlass.Float4E2M1FN)
        E1, N21, K1 = (
            cute.size(gB1_e, mode=[0]),
            cute.size(gB1_e, mode=[1]),
            cute.size(gB1_e, mode=[2]),
        )
        gW1 = cute.make_tensor(
            gB1_e.iterator, cute.make_layout((N21, K1, E1), stride=(K1, 1, N21 * K1))
        )
        m_padded_sf1 = cute.size(gSFA1_u8, mode=[1])
        tok_ptr = cute.recast_ptr(gSFA1_u8.iterator, dtype=sf_dtype)
        wgt_ptr = cute.recast_ptr(gSFB1_u8.iterator, dtype=sf_dtype)
        gSFA1 = cute.make_tensor(wgt_ptr, self.sf1.deduce_sfa_layout(N21, K1, E1))
        gSFB1 = cute.make_tensor(
            tok_ptr, self.sf1.deduce_sfb_layout(m_padded_sf1, K1, 1)
        )
        a_smem1 = c1.load_ab.make_smem_layout_a()
        b_smem1 = c1.load_ab.make_smem_layout_b()
        epi_smem1 = c1.epi.make_smem_layout(c1.TILE)
        bm1, bn1, bk1 = c1.TILE
        sfa_stages1 = c1.load_sf.sfa_stages(c1.ab_stage, bk1)
        sfb_stages1 = c1.load_sf.sfb_stages(c1.ab_stage, bk1)
        sfa_smem1 = c1.load_sf.make_smem_layout_sfa(bm1, sfa_stages1)
        sfb_smem1 = c1.load_sf.make_smem_layout_sfb(bn1, sfb_stages1)
        tma_atom_a1, tma_tensor_a1 = c1.load_ab.make_tma_atom_a(gW1, a_smem1)
        tma_atom_b1, tma_tensor_b1 = c1.load_ab.make_tma_atom_b(gA1, b_smem1)
        tma_atom_sfa1, tma_tensor_sfa1 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gSFA1,
            cute.slice_(sfa_smem1, (None, None, 0)),
            (bm1, 1),
            num_multicast=1,
        )
        tma_atom_sfb1, tma_tensor_sfb1 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gSFB1,
            cute.slice_(sfb_smem1, (None, None, 0)),
            (bn1, 1),
            num_multicast=1,
        )
        gate_sf_bytes = cute.size_in_bytes(
            sf_dtype, cute.slice_(sfa_smem1, (None, None, 0))
        )
        coarse_sf_bytes = cute.size_in_bytes(
            sf_dtype, cute.slice_(sfb_smem1, (None, None, 0))
        )
        self.a_bytes1 = 2 * c1.load_ab.tma_bytes_a
        self.sfa_bytes1 = 2 * gate_sf_bytes
        self.b_bytes1 = c1.load_ab.tma_bytes_b
        self.sfb_bytes1 = coarse_sf_bytes

        # ---------------- fc2: A = q1, B = weights
        tiledmma2 = c2.mma.make_tiled_mma(c2.TILE)
        tiled_r2s2 = c2.epi.make_tiled_r2s(tiledmma2)
        tiled_s2r2 = c2.epi.make_tiled_s2r(c2.TILE)
        gB2_e = cute.recast_tensor(gB2_u8, cutlass.Float4E2M1FN)
        E2, N2, K2 = (
            cute.size(gB2_e, mode=[0]),
            cute.size(gB2_e, mode=[1]),
            cute.size(gB2_e, mode=[2]),
        )
        gW2 = cute.make_tensor(
            gB2_e.iterator, cute.make_layout((N2, K2, E2), stride=(K2, 1, N2 * K2))
        )
        m_padded_sf2 = cute.size(gSFA2_u8, mode=[1])
        gSFA2 = cute.make_tensor(
            cute.recast_ptr(gSFA2_u8.iterator, dtype=sf_dtype),
            self.sf2.deduce_sfa_layout(m_padded_sf2, K2, 1),
        )
        gSFB2 = cute.make_tensor(
            cute.recast_ptr(gSFB2_u8.iterator, dtype=sf_dtype),
            self.sf2.deduce_sfb_layout(N2, K2, E2),
        )
        a_layout2 = cutlass.utils.LayoutEnum.from_tensor(gA2)
        self.a_is_m_major2 = a_layout2.is_m_major_a()
        a_smem2 = c2.load_ab.make_smem_layout_a()
        b_smem2 = c2.load_ab.make_smem_layout_b()
        epi_smem2 = c2.epi.make_smem_layout(c2.TILE)
        bm2, bn2, bk2 = c2.TILE
        sfa_stages2 = c2.load_sf.sfa_stages(c2.ab_stage, bk2)
        sfb_stages2 = c2.load_sf.sfb_stages(c2.ab_stage, bk2)
        sfa_smem2 = c2.load_sf.make_smem_layout_sfa(bm2, sfa_stages2)
        sfb_smem2 = c2.load_sf.make_smem_layout_sfb(bn2, sfb_stages2)
        tma_atom_a2, tma_tensor_a2 = c2.load_ab.make_tma_atom_a(gA2, a_smem2)
        tma_atom_b2, tma_tensor_b2 = c2.load_ab.make_tma_atom_b(gW2, b_smem2)
        tma_atom_sfa2, tma_tensor_sfa2 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gSFA2,
            cute.slice_(sfa_smem2, (None, None, 0)),
            (bm2, 1),
            num_multicast=1,
        )
        tma_atom_sfb2, tma_tensor_sfb2 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gSFB2,
            cute.slice_(sfb_smem2, (None, None, 0)),
            (bn2, 1),
            num_multicast=1,
        )
        self.a_bytes2 = c2.load_ab.tma_bytes_a
        self.sfa_bytes2 = cute.size_in_bytes(
            sf_dtype, cute.slice_(sfa_smem2, (None, None, 0))
        )
        self.b_bytes2 = c2.load_ab.tma_bytes_b
        self.sfb_bytes2 = cute.size_in_bytes(
            sf_dtype, cute.slice_(sfb_smem2, (None, None, 0))
        )

        # ---------------- one smem pool, fc1 and fc2 views alias it
        sfw = sf_dtype.width
        offs1 = {}
        cur = 0
        regions1 = [
            ("sA", cute.cosize(b_smem1) * c1.load_ab.b_smem_dtype.width // 8),
            ("sU", cute.cosize(a_smem1) * c1.load_ab.a_smem_dtype.width // 8),
            ("sG", cute.cosize(a_smem1) * c1.load_ab.a_smem_dtype.width // 8),
            ("sSFU", cute.cosize(sfa_smem1) * sfw // 8),
            ("sSFG", cute.cosize(sfa_smem1) * sfw // 8),
            ("sSFA", cute.cosize(sfb_smem1) * sfw // 8),
            (
                "sC",
                cute.cosize(epi_smem1) * c1.epi.out_dtype.width // 8
                if c1.owns_epi_smem
                else 0,
            ),
        ]
        # fc1's epilogue sC first: it then lies under fc2's sA (q1), which the q1 loader
        # fills only after the phase barrier, and clear of sB/sSFB that EARLY_W2 fills
        # during that epilogue.  Every region stays 1024-aligned.
        regions1 = regions1[-1:] + regions1[:-1]
        for name, nbytes in regions1:
            cur = _align(cur, 1024)
            offs1[name] = cur
            cur += nbytes
        end1 = cur
        offs2 = {}
        cur = 0
        for name, nbytes in (
            ("sA", cute.cosize(a_smem2) * c2.load_ab.a_smem_dtype.width // 8),
            ("sB", cute.cosize(b_smem2) * c2.load_ab.b_smem_dtype.width // 8),
            ("sSFB", cute.cosize(sfb_smem2) * sfw // 8),
            ("sSFA", cute.cosize(sfa_smem2) * sfw // 8),
        ):
            # sC + scatter meta overlay sA and may run into sB: keep sA, sB contiguous.
            cur = _align(cur, 1024) if name != "sB" else _align(cur, 128)
            offs2[name] = cur
            cur += nbytes
        end2 = cur
        union_bytes2 = c2.epi.smem_bytes(c2.TILE) + c2.epi.aux_smem_bytes(c2.TILE)
        assert union_bytes2 <= offs2["sSFB"], "fc2 sC + meta must stay inside sA/sB"
        self.offs1, self.offs2 = offs1, offs2
        # EARLY_W2 writes sB/sSFB (fc2) while the math warps still run the fc1 epilogue,
        # which touches only sC (fc1): the byte ranges must be disjoint.
        sz2 = {
            "sB": cute.cosize(b_smem2) * c2.load_ab.b_smem_dtype.width // 8,
            "sSFB": cute.cosize(sfb_smem2) * sfw // 8,
        }
        sc1 = (offs1["sC"], offs1["sC"] + dict(regions1)["sC"])
        rng2 = {k: (offs2[k], offs2[k] + v) for k, v in sz2.items()}
        overlap = any(lo < sc1[1] and sc1[0] < hi for lo, hi in rng2.values())
        print(
            f"[fc12_fused] smem pool: sC1=[{sc1[0]},{sc1[1]}) sB2=[{rng2['sB'][0]},"
            f"{rng2['sB'][1]}) sSFB2=[{rng2['sSFB'][0]},{rng2['sSFB'][1]}) "
            f"overlap={overlap}",
            flush=True,
        )
        assert not overlap, "EARLY_W2 needs sB/sSFB (fc2) disjoint from sC (fc1)"
        assert c1.owns_epi_smem, "EARLY_W2 needs the fc1 epilogue's own sC region"
        pool_bytes = _align(max(end1, end2), 1024)
        epi_empty_bars2 = c2.epi.num_empty_barriers
        assert c1.epi.num_empty_barriers == 0

        @cute.struct
        class SharedStorage:
            a_full1: cute.struct.MemRange[c1.I64, c1.ab_stage]
            a_empty1: cute.struct.MemRange[c1.I64, c1.ab_stage]
            b_full1: cute.struct.MemRange[c1.I64, c1.ab_stage]
            b_empty1: cute.struct.MemRange[c1.I64, c1.ab_stage]
            a_full2: cute.struct.MemRange[c2.I64, c2.ab_stage]
            a_empty2: cute.struct.MemRange[c2.I64, c2.ab_stage]
            b_full2: cute.struct.MemRange[c2.I64, c2.ab_stage]
            b_empty2: cute.struct.MemRange[c2.I64, c2.ab_stage]
            epi_empty2: cute.struct.MemRange[c2.I64, epi_empty_bars2]
            sfull: cute.struct.MemRange[c1.I64, c1.sched_stages]
            sempty: cute.struct.MemRange[c1.I64, c1.sched_stages]
            work: cute.struct.MemRange[c1.I32, c1.sched_stages * FusedWorkTile.FIELDS]
            tdone: cute.struct.MemRange[c1.I64, 1]
            pool: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, pool_bytes], 1024
            ]

        smem_size = SharedStorage.size_in_bytes()  # type: ignore[attr-defined]
        cap = cutlass.utils.SmemAllocator.capacity_in_bytes()
        assert smem_size <= cap, f"fused smem {smem_size} B > {cap} B"
        self.storage = SharedStorage

        self.kernel(
            tiledmma1,
            tiled_r2s1,
            tiledmma2,
            tiled_r2s2,
            tiled_s2r2,
            tma_atom_a1,
            tma_atom_b1,
            tma_atom_sfa1,
            tma_atom_sfb1,
            tma_tensor_a1,
            tma_tensor_b1,
            tma_tensor_sfa1,
            tma_tensor_sfb1,
            tma_atom_a2,
            tma_atom_b2,
            tma_atom_sfa2,
            tma_atom_sfb2,
            tma_tensor_a2,
            tma_tensor_b2,
            tma_tensor_sfa2,
            tma_tensor_sfb2,
            gQ1,
            gSFQ_u8,
            offsets,
            gSplit,
            gCnt,
            gOut,
            gTok,
            gWt,
            gSync,
            a_smem1,
            b_smem1,
            sfa_smem1,
            sfb_smem1,
            epi_smem1,
            a_smem2,
            b_smem2,
            sfa_smem2,
            sfb_smem2,
            epi_smem2,
        ).launch(
            grid=[self.grid_x, 1, 1],
            block=[c1.threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=c1.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiledmma1,
        tiled_r2s1,
        tiledmma2,
        tiled_r2s2,
        tiled_s2r2,
        tma_atom_a1,
        tma_atom_b1,
        tma_atom_sfa1,
        tma_atom_sfb1,
        tma_tensor_a1: cute.Tensor,
        tma_tensor_b1: cute.Tensor,
        tma_tensor_sfa1: cute.Tensor,
        tma_tensor_sfb1: cute.Tensor,
        tma_atom_a2,
        tma_atom_b2,
        tma_atom_sfa2,
        tma_atom_sfb2,
        tma_tensor_a2: cute.Tensor,
        tma_tensor_b2: cute.Tensor,
        tma_tensor_sfa2: cute.Tensor,
        tma_tensor_sfb2: cute.Tensor,
        gQ1: cute.Tensor,
        gSFQ_u8: cute.Tensor,
        offsets: cute.Tensor,
        gSplit: cute.Tensor,
        gCnt: cute.Tensor,
        gOut: cute.Tensor,
        gTok: cute.Tensor,
        gWt: cute.Tensor,
        gSync: cute.Tensor,
        a_smem1,
        b_smem1,
        sfa_smem1,
        sfb_smem1,
        epi_smem1,
        a_smem2,
        b_smem2,
        sfa_smem2,
        sfb_smem2,
        epi_smem2,
    ):
        c1 = self.cfg1
        c2 = self.cfg2
        i32 = cutlass.Int32
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        bm1, bn1, bk1 = c1.TILE
        bm2, bn2, bk2 = c2.TILE
        pm1, pn1 = bn1, bm1  # swap-AB: tokens are the tile's N
        pm2, pn2 = bm2, bn2
        # fc1 geometry (as fc1_act_q1's kernel)
        M1, N1 = cute.size(gQ1, mode=[0]), cute.size(gQ1, mode=[1])
        sfq_m_padded = cute.size(gSFQ_u8, mode=[1]) // UE8M0_PACK_NUM
        gSFQ = cute.make_tensor(
            gSFQ_u8.iterator,
            cute.make_layout(
                (cute.size(gSFQ_u8, mode=[0]), sfq_m_padded, UE8M0_PACK_NUM),
                stride=(UE8M0_PACK_NUM * sfq_m_padded, UE8M0_PACK_NUM, 1),
            ),
        )
        gQ1_t = cute.make_tensor(
            gQ1.iterator, cute.make_layout((N1, M1), stride=(1, N1))
        )
        K1 = cute.size(tma_tensor_b1, mode=[1])
        num_groups = cute.size(offsets, mode=[0]) - 1
        nnb1 = ceil_div(N1, pn1)
        kt_per_pack1 = self.sf1.k_tiles_per_pack_b(bk1)
        sf_stages1 = self.sf1.sfb_stages(c1.ab_stage, bk1)
        ncyc1 = ceil_div(K1, self.sf1.grank_b * self.sf1.PACK_NSF * sf_stages1)
        # fc2 geometry (as fc2_finalize's kernel)
        N2 = cute.size(gOut, mode=[1])
        K2 = cute.size(tma_tensor_b2, mode=[1])
        nnb2 = ceil_div(N2, pn2)
        kt_per_pack2 = self.sf2.k_tiles_per_pack_a(bk2)
        sf_stages2 = self.sf2.sfa_stages(c2.ab_stage, bk2)
        ncyc2 = ceil_div(K2, self.sf2.grank_a * self.sf2.PACK_NSF * sf_stages2)
        # fc1 column tiles (pn1 wide) one fc2 SF cycle reads (its K-tiles and its SFA packs)
        kcols2 = sf_stages2 * kt_per_pack2 * bk2 // pn1
        assert kcols2 * pn1 == sf_stages2 * kt_per_pack2 * bk2 and 1 <= kcols2 <= 32
        assert self.sf2.grank_a * self.sf2.PACK_NSF * sf_stages2 == kcols2 * pn1

        smem = cutlass.utils.SmemAllocator()
        stg = smem.allocate(self.storage)
        pool = stg.pool.data_ptr()
        o1, o2 = self.offs1, self.offs2
        # fc1 views
        sA1 = cute.make_tensor(
            cute.recast_ptr(
                pool + o1["sA"], b_smem1.inner, dtype=c1.load_ab.b_smem_dtype
            ),
            b_smem1.outer,
        )
        sU1 = cute.make_tensor(
            cute.recast_ptr(
                pool + o1["sU"], a_smem1.inner, dtype=c1.load_ab.a_smem_dtype
            ),
            a_smem1.outer,
        )
        sG1 = cute.make_tensor(
            cute.recast_ptr(
                pool + o1["sG"], a_smem1.inner, dtype=c1.load_ab.a_smem_dtype
            ),
            a_smem1.outer,
        )
        sSFU1 = cute.make_tensor(
            cute.recast_ptr(pool + o1["sSFU"], dtype=c1.load_sf.sf_dtype), sfa_smem1
        )
        sSFG1 = cute.make_tensor(
            cute.recast_ptr(pool + o1["sSFG"], dtype=c1.load_sf.sf_dtype), sfa_smem1
        )
        sSFA1 = cute.make_tensor(
            cute.recast_ptr(pool + o1["sSFA"], dtype=c1.load_sf.sf_dtype), sfb_smem1
        )
        sC1_stages = cute.make_tensor(
            cute.recast_ptr(pool + o1["sC"], epi_smem1.inner, dtype=c1.epi.out_dtype),
            epi_smem1.outer,
        )
        sC1 = cute.slice_(sC1_stages, (None, None, 0))
        # fc2 views
        sA2 = cute.make_tensor(
            cute.recast_ptr(
                pool + o2["sA"], a_smem2.inner, dtype=c2.load_ab.a_smem_dtype
            ),
            a_smem2.outer,
        )
        sB2 = cute.make_tensor(
            cute.recast_ptr(
                pool + o2["sB"], b_smem2.inner, dtype=c2.load_ab.b_smem_dtype
            ),
            b_smem2.outer,
        )
        sSFB2 = cute.make_tensor(
            cute.recast_ptr(pool + o2["sSFB"], dtype=c2.load_sf.sf_dtype), sfb_smem2
        )
        sSFA2 = cute.make_tensor(
            cute.recast_ptr(pool + o2["sSFA"], dtype=c2.load_sf.sf_dtype), sfa_smem2
        )
        sC2_stages = cute.make_tensor(
            cute.recast_ptr(pool + o2["sA"], dtype=c2.epi.out_dtype), epi_smem2
        )
        sC2 = cute.slice_(sC2_stages, (None, None, 0))
        sTok_ptr = cute.recast_ptr(
            sC2_stages.iterator + cute.cosize(epi_smem2), dtype=c2.I32
        )
        sTok = cute.make_tensor(sTok_ptr, cute.make_layout(bm2))
        sWt = cute.make_tensor(
            cute.recast_ptr(sTok_ptr + bm2, dtype=cutlass.Float32),
            cute.make_layout(bm2),
        )

        a_full1, a_empty1 = stg.a_full1.data_ptr(), stg.a_empty1.data_ptr()
        b_full1, b_empty1 = stg.b_full1.data_ptr(), stg.b_empty1.data_ptr()
        a_full2, a_empty2 = stg.a_full2.data_ptr(), stg.a_empty2.data_ptr()
        b_full2, b_empty2 = stg.b_full2.data_ptr(), stg.b_empty2.data_ptr()
        epi_empty2 = stg.epi_empty2.data_ptr()
        sfull, sempty = stg.sfull.data_ptr(), stg.sempty.data_ptr()
        sWork = stg.work.get_tensor(
            cute.make_layout((c1.sched_stages, FusedWorkTile.FIELDS))
        )
        tdone = stg.tdone.data_ptr()
        sync_base = split_k._addr(gSync)

        if warp_idx == 0:
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(c1.ab_stage):
                    cute.arch.mbarrier_init(a_full1 + s, 1)
                    cute.arch.mbarrier_init(a_empty1 + s, c1.mma_threads)
                    cute.arch.mbarrier_init(b_full1 + s, 1)
                    cute.arch.mbarrier_init(b_empty1 + s, c1.mma_threads)
                for s in cutlass.range_constexpr(c2.ab_stage):
                    cute.arch.mbarrier_init(a_full2 + s, 1)
                    cute.arch.mbarrier_init(a_empty2 + s, c2.mma_threads)
                    cute.arch.mbarrier_init(b_full2 + s, 1)
                    cute.arch.mbarrier_init(b_empty2 + s, c2.mma_threads)
                for s in cutlass.range_constexpr(c2.epi.num_empty_barriers):
                    cute.arch.mbarrier_init(
                        epi_empty2 + s, c2.epi.empty_barrier_arrivals
                    )
                for s in cutlass.range_constexpr(c1.sched_stages):
                    cute.arch.mbarrier_init(sfull + s, 1)
                    cute.arch.mbarrier_init(sempty + s, c1.num_sched_consumers)
                cute.arch.mbarrier_init(tdone, c1.num_math_warps)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # The phase barrier: math warps + the q1 loader warp (EARLY_W2: the weight loader
        # does not take it).
        trans_threads = c1.mma_threads + 32
        # Everything that calls into the configs is hoisted out of the runtime `if`.
        tiled_s2r1 = c1.epi.make_tiled_s2r(c1.TILE)
        swap_threads_m = c1.epi.s2g_threads_n(c1.TILE)
        swap_n_store = c1.epi.num_store_threads
        split_act = c1.activation
        is_prod = c1.is_prod_wg(warp_idx)

        if is_prod:
            cute.arch.setmaxregister_decrease(c1.reg_prod)

            if warp_idx == c1.sched_warp:
                prod = FusedSchedProducer.create(c1.sched_stages)
                # First claim before the tile counts: its round trip overlaps their loads.
                old = i32(0)
                if lane == i32(0):
                    old = atom_add_relaxed_i32(sync_base, i32(1))
                cl = split_k.count_loads(offsets, num_groups)
                mb1 = split_k.count_m_blocks_lane(
                    cl[0],
                    cl[1],
                    cl[2],
                    cl[3],
                    cl[4],
                    cl[5],
                    cl[6],
                    cl[7],
                    offsets,
                    num_groups,
                    pm1,
                )
                mb2 = split_k.count_m_blocks_lane(
                    cl[0],
                    cl[1],
                    cl[2],
                    cl[3],
                    cl[4],
                    cl[5],
                    cl[6],
                    cl[7],
                    offsets,
                    num_groups,
                    pm2,
                )
                T1 = split_k.warp_total(mb1) * nnb1
                T2 = split_k.warp_total(mb2) * nnb2
                S = split_k.choose_splits(
                    T1, self.grid_x, self.split_max, ncyc1, self.split_min
                )
                U1 = T1 * S
                U = U1 + T2
                fc2_flags = i32(1) + (i32(1) << i32(PHASE2_BIT))
                if i32(self.prefetch_min_t1) <= T1:
                    fc2_flags = fc2_flags + (i32(1) << i32(PREFETCH_BIT))
                # Atomic claims return v; the unit is v.
                claim_off = i32(0)
                last_v = U + i32(self.grid_x - 1)
                did_claim = i32(1)
                u = cute.arch.shuffle_sync(old, i32(0))
                v = u
                tpar = i32(0)
                while u < U:
                    if u < U1:
                        t = u // S
                        sch = moe_scheduler.MoeTileScheduler.create(
                            pm1, num_groups, nnb1, 1, t, offsets
                        )
                        has, e, m_tile, n_tile, m_off, m_bnd = sch.get_next_block(
                            offsets
                        )
                        valid = i32(1)
                        if i32(1) < S:
                            valid = i32(1) + (S << i32(16))
                        prod.publish(
                            sWork,
                            sfull,
                            sempty,
                            FusedWorkTile(n_tile, m_tile, e, m_off, m_bnd, valid, u),
                        )
                    else:
                        sch2 = moe_scheduler.MoeTileScheduler.create(
                            pm2, num_groups, nnb2, 1, u - U1, offsets
                        )
                        has2, e2, m_tile2, n_tile2, m_off2, m_bnd2 = (
                            sch2.get_next_block(offsets)
                        )
                        prod.publish(
                            sWork,
                            sfull,
                            sempty,
                            FusedWorkTile(
                                m_tile2, n_tile2, e2, m_off2, m_bnd2, fc2_flags, u
                            ),
                        )
                    # Claim the next unit only once this one's mainloop is done: a CTA busy
                    # with fc1 must not hoard fc2 tiles that idle CTAs could start (and
                    # prefetch weights for) right away.
                    cute.arch.mbarrier_wait(tdone, tpar)
                    tpar = tpar ^ i32(1)
                    old = i32(0)
                    if lane == i32(0):
                        old = atom_add_relaxed_i32(sync_base, i32(1))
                    v = cute.arch.shuffle_sync(old, i32(0))
                    u = v + claim_off
                    did_claim = i32(1)
                # Every claiming CTA ends with exactly one failing claim, so the claim values
                # run 0..last_v: whoever draws last_v is the last claimer of this launch.
                if did_claim != i32(0):
                    if v == last_v:
                        if lane == i32(0):
                            split_k.st_i32(sync_base, i32(0))
                prod.publish_sentinel(sWork, sfull, sempty)

            if warp_idx == c1.load_warp_0:
                # fc1: weights (U/G halves); fc2: q1 rows (after the expert's release)
                cons = FusedSchedConsumer.create(c1.sched_stages)
                tile = cons.get_next_tile(sWork, sfull, sempty)
                a_phase = i32(1)
                go = is_fc1(tile.valid)
                while go != i32(0):
                    c_lo, c_hi = split_k.cycle_range(tile.valid, tile.unit, ncyc1)
                    a_phase = fc1k.load_a_sfa_swap(
                        tma_atom_a1,
                        tma_atom_sfa1,
                        tma_tensor_a1,
                        tma_tensor_sfa1,
                        sU1,
                        sG1,
                        sSFU1,
                        sSFG1,
                        c1.TILE,
                        tile,
                        N1,
                        a_full1,
                        a_empty1,
                        self.a_bytes1,
                        self.sfa_bytes1,
                        c_hi,
                        sf_stages1,
                        kt_per_pack1,
                        c1.ab_stage,
                        a_phase,
                        c_lo,
                    )
                    tile = cons.get_next_tile(sWork, sfull, sempty)
                    go = is_fc1(tile.valid)
                bar_sync(TRANS_BAR_ID, trans_threads)
                fence_proxy_async_shared()
                a_phase2, epi_phase = i32(1), i32(1)
                while tile.valid != i32(0):
                    kbase = sync_base + Int64(
                        KDONE_BASE + tile.group * KCOLS_MAX
                    ) * Int64(4)
                    need_c = ceil_div(tile.m_boundary - tile.m_offset, pm1)
                    cute.arch.mbarrier_wait(epi_empty2, epi_phase)
                    epi_phase ^= 1
                    a_phase2 = load_a_sfa_kacq(
                        tma_atom_a2,
                        tma_atom_sfa2,
                        tma_tensor_a2,
                        tma_tensor_sfa2,
                        sA2,
                        sSFA2,
                        c2.TILE,
                        tile,
                        a_full2,
                        a_empty2,
                        self.a_bytes2,
                        self.sfa_bytes2,
                        ncyc2,
                        sf_stages2,
                        kt_per_pack2,
                        c2.ab_stage,
                        a_phase2,
                        kbase,
                        need_c,
                        nnb1,
                        kcols2,
                        self.sleep_ns,
                    )
                    # The expert's last fc2 tile past its acquires puts the counters back.
                    rd = i32(0)
                    if lane == i32(0):
                        raddr = sync_base + Int64(
                            SYNC_HEADER + MAX_GROUPS + tile.group
                        ) * Int64(4)
                        rd = atom_add_relaxed_i32(raddr, i32(1))
                    rd = cute.arch.shuffle_sync(rd, i32(0))
                    t2e = ceil_div(tile.m_boundary - tile.m_offset, pm2) * nnb2
                    if rd == t2e - i32(1):
                        if lane < nnb1:
                            split_k.st_i32(kbase + Int64(lane) * Int64(4), i32(0))
                        if lane == i32(0):
                            raddr0 = sync_base + Int64(
                                SYNC_HEADER + MAX_GROUPS + tile.group
                            ) * Int64(4)
                            split_k.st_i32(raddr0, i32(0))
                    tile = cons.get_next_tile(sWork, sfull, sempty)

            if warp_idx == c1.load_warp_1:
                # fc1: activations (after the PDL wait); fc2: weights (no wait at all)
                cons = FusedSchedConsumer.create(c1.sched_stages)
                tile = cons.get_next_tile(sWork, sfull, sempty)
                b_phase = i32(1)
                go = is_fc1(tile.valid)
                ran_fc1 = go
                if go != i32(0):
                    if cutlass.const_expr(c1.enable_pdl):
                        cute.arch.griddepcontrol_wait()
                while go != i32(0):
                    c_lo, c_hi = split_k.cycle_range(tile.valid, tile.unit, ncyc1)
                    b_phase = fc1k.load_b_sfb_swap(
                        tma_atom_b1,
                        tma_atom_sfb1,
                        tma_tensor_b1,
                        tma_tensor_sfb1,
                        sA1,
                        sSFA1,
                        c1.TILE,
                        tile,
                        b_full1,
                        b_empty1,
                        self.b_bytes1,
                        self.sfb_bytes1,
                        c_hi,
                        sf_stages1,
                        kt_per_pack1,
                        c1.ab_stage,
                        b_phase,
                        c_lo,
                    )
                    tile = cons.get_next_tile(sWork, sfull, sempty)
                    go = is_fc1(tile.valid)
                # EARLY_W2 (no phase barrier here): sB/sSFB are disjoint from the fc1
                # epilogue's sC, and every fc1 stage this CTA loaded was consumed before its
                # mainloop arrived on tdone (which the scheduler waited on before publishing
                # this fc2 tile).
                b_phase2, epi_phase = i32(1), i32(1)
                first_w2 = ran_fc1
                while tile.valid != i32(0):
                    wait_b = i32(1) - ((tile.valid >> i32(PREFETCH_BIT)) & i32(1))
                    if first_w2 != i32(0):
                        wait_b = i32(0)
                    first_w2 = i32(0)
                    if wait_b != i32(0):
                        # No early weight stream for this launch: wait for the expert too.
                        kbase_b = sync_base + Int64(
                            KDONE_BASE + tile.group * KCOLS_MAX
                        ) * Int64(4)
                        acquire_cols(
                            kbase_b,
                            i32(0),
                            kcols2,
                            nnb1,
                            ceil_div(tile.m_boundary - tile.m_offset, pm1),
                            self.sleep_ns,
                        )
                    cute.arch.mbarrier_wait(epi_empty2, epi_phase)
                    epi_phase ^= 1
                    b_phase2 = fc2k.load_b_sfb(
                        tma_atom_b2,
                        tma_atom_sfb2,
                        tma_tensor_b2,
                        tma_tensor_sfb2,
                        sB2,
                        sSFB2,
                        c2.TILE,
                        tile,
                        b_full2,
                        b_empty2,
                        self.b_bytes2,
                        self.sfb_bytes2,
                        ncyc2,
                        sf_stages2,
                        kt_per_pack2,
                        c2.ab_stage,
                        b_phase2,
                    )
                    tile = cons.get_next_tile(sWork, sfull, sempty)

        else:
            cute.arch.setmaxregister_increase(c1.reg_math)
            thr_r2s1 = tiled_r2s1.get_slice(tidx)
            thr_s2r1 = tiled_s2r1.get_slice(tidx)
            thr_r2s2 = tiled_r2s2.get_slice(tidx)
            thr_s2r2 = tiled_s2r2.get_slice(tidx)
            cons = FusedSchedConsumer.create(c1.sched_stages)
            tile = cons.get_next_tile(sWork, sfull, sempty)
            go = is_fc1(tile.valid)
            if go != i32(0):
                if cutlass.const_expr(c1.enable_pdl):
                    cute.arch.griddepcontrol_wait()
            a_phase, b_phase = i32(0), i32(0)
            while go != i32(0):
                c_lo, c_hi = split_k.cycle_range(tile.valid, tile.unit, ncyc1)
                acc, acc_g, a_phase, b_phase = fc1k.mma_swap(
                    tiledmma1,
                    c1.mma,
                    self.sf1,
                    c1.activation,
                    sU1,
                    sG1,
                    sA1,
                    sSFU1,
                    sSFG1,
                    sSFA1,
                    (bm1, bn1),
                    bk1,
                    c1.ACC,
                    c1.load_ab.a_smem_dtype,
                    c1.load_ab.b_dtype,
                    c1.load_ab.a_unpack_bits,
                    tidx,
                    a_full1,
                    a_empty1,
                    b_full1,
                    b_empty1,
                    c_hi - c_lo,
                    sf_stages1,
                    kt_per_pack1,
                    c1.ab_stage,
                    a_phase,
                    b_phase,
                    True,
                )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(tdone)
                fin = i32(1)
                S_t = split_k.splits_of(tile.valid)
                if S_t > i32(1):
                    fin = split_k.reduce_partials(
                        acc,
                        acc_g,
                        gSplit,
                        gCnt,
                        tile.unit,
                        S_t,
                        tidx,
                        c1.mma_threads,
                        c1.epi_bar_id,
                        True,
                    )
                split_act(acc, acc_g)
                if fin != i32(0):
                    moe_epilogue.store_wg_q1_swap(
                        acc,
                        thr_r2s1,
                        thr_s2r1,
                        sC1,
                        gQ1_t,
                        gSFQ,
                        tile,
                        (bm1, bn1),
                        c1.epi.out_dtype,
                        self.qcfg,
                        c1.epi_bar_id,
                        c1.mma_threads,
                        SF_M_ALIGN,
                        UE8M0_PACK_NUM,
                        tidx,
                        swap_threads_m,
                        swap_n_store,
                    )
                    # release this tile's q1/sf1 to the fc2 tiles of its expert (thread 0's
                    # atom.release after the CTA barrier covers every math thread's q1/sf1
                    # stores by cumulativity, as CUTLASS's semaphore)
                    fence_proxy_async_global()
                    cute.arch.barrier(
                        barrier_id=c1.epi_bar_id, number_of_threads=c1.mma_threads
                    )
                    if tidx == i32(0):
                        atom_add_release_i32(
                            sync_base
                            + Int64(KDONE_BASE + tile.group * KCOLS_MAX + tile.m_block)
                            * Int64(4),
                            i32(1),
                        )
                tile = cons.get_next_tile(sWork, sfull, sempty)
                go = is_fc1(tile.valid)
            fence_proxy_async_shared()
            bar_arrive(TRANS_BAR_ID, trans_threads)
            if cutlass.const_expr(c1.enable_pdl):
                cute.arch.griddepcontrol_wait()
            a_phase2, b_phase2 = i32(0), i32(0)
            while tile.valid != i32(0):
                acc2, a_phase2, b_phase2 = fc2k.mma(
                    tiledmma2,
                    c2.mma,
                    self.sf2,
                    sA2,
                    sB2,
                    sSFA2,
                    sSFB2,
                    (bm2, bn2),
                    bk2,
                    c2.ACC,
                    c2.load_ab.a_dtype,
                    c2.load_ab.b_smem_dtype,
                    self.a_is_m_major2,
                    c2.load_ab.b_unpack_bits,
                    tidx,
                    a_full2,
                    a_empty2,
                    b_full2,
                    b_empty2,
                    ncyc2,
                    sf_stages2,
                    kt_per_pack2,
                    c2.ab_stage,
                    a_phase2,
                    b_phase2,
                )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(tdone)
                moe_epilogue.store_wg_scatter(
                    acc2,
                    thr_r2s2,
                    thr_s2r2,
                    sC2,
                    sTok,
                    sWt,
                    gTok,
                    gWt,
                    gOut,
                    tile,
                    (bm2, bn2),
                    N2,
                    c2.epi.out_dtype,
                    epi_empty2,
                    c2.epi_bar_id,
                    c2.mma_threads,
                    tidx,
                )
                tile = cons.get_next_tile(sWork, sfull, sempty)

        # No exit protocol: the last claimer resets the claim counter, and the last fc2 tile
        # of each expert (after its acquire) resets that expert's done/read counters.


# --------------------------------------------------------------------------- host side
_SYNC: dict = {}


def sync_workspace(device):
    """int32 [SYNC_HEADER + MAX_GROUPS] zeros, once per device (before any capture)."""
    key = torch.device(device)
    ws = _SYNC.get(key)
    if ws is None:
        ws = torch.zeros(SYNC_WORDS, dtype=torch.int32, device=device)
        _SYNC[key] = ws
    return ws


def _stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


_COMPILED: dict = {}


def cute_dsl_sm12x_fc12_fused_mxfp8_mxfp4(
    a_q,
    a_scale,
    w1,
    w1_scale,
    m_indptr,
    w2,
    w2_scale,
    src_token,
    pair_scales,
    num_tokens: int,
    *,
    out_q: torch.Tensor,
    out_sf: torch.Tensor,
    out: torch.Tensor,
    swiglu_limit=None,
    enable_pdl: bool = False,
):
    """fc1_act_q1 + fc2_finalize in one launch (decode shapes: <= 32 rows per expert).

    Same contract as calling cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(a_q, a_scale, w1, w1_scale,
    m_indptr, out_q=out_q, out_sf=out_sf, activation=Swiglu, swiglu_limit=...) and then
    cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4(out_q, out_sf, w2, w2_scale, m_indptr, src_token,
    pair_scales, num_tokens, out=out): out_q/out_sf/out must start zeroed as there."""
    from ....tllm_enums import ActivationType
    from . import moe_mxfp8_mxfp4_fc1_act_q1 as fc1h
    from . import moe_mxfp8_mxfp4_fc2_finalize as fc2h
    from ._moe_utils.moe_epilogue import EpiMethod

    swiglu_limit = fc1h._normalize_swiglu_limit(ActivationType.Swiglu, swiglu_limit)
    m, n1, k1 = int(a_q.shape[0]), int(w1.shape[1]) // 2, int(w1.shape[2]) * 2
    n2, k2 = int(w2.shape[1]), int(w2.shape[2]) * 2
    num_experts = int(w1.shape[0])
    assert num_experts <= MAX_GROUPS
    props = torch.cuda.get_device_properties(a_q.device)
    grid_x, sm_version = props.multi_processor_count, f"sm_{props.major}{props.minor}"
    tile1 = fc1h.select_tile(
        total_rows=m, n=n1, k=k1, num_experts=num_experts, num_sms=grid_x
    )
    tile2 = fc2h.select_tile(
        total_rows=m, n=n2, num_experts=num_experts, num_sms=grid_x
    )
    assert tuple(tile1[:2]) == (8, 128) and tuple(tile2[:2]) == (32, 128), (
        tile1,
        tile2,
    )
    op1 = fc1h._op(
        n1,
        k1,
        tuple(tile1),
        torch.float8_e4m3fn,
        ActivationType.Swiglu,
        EpiMethod.WG_QUANT_SWAP,
        enable_pdl,
        swiglu_limit,
    )
    op2 = fc2h.CuteDslSm120GroupedMxfp8Mxfp4Fc2FinalizeOp(
        n2, k2, tuple(tile2), EpiMethod.WG_SCATTER, enable_pdl
    )
    indptr = m_indptr.to(torch.int32)
    args1 = fc1k.make_args(
        a_q,
        a_scale,
        w1,
        w1_scale,
        out_q,
        out_sf.view(torch.uint8),
        indptr,
        op1.cfg.epi,
        op1.cfg.TILE,
    )
    args2 = fc2k.make_args(
        out_q,
        out_sf,
        w2,
        w2_scale,
        out,
        src_token.to(torch.int32),
        pair_scales.to(torch.float32),
        indptr,
        op2.cfg.epi,
        op2.cfg.TILE,
    )
    sync = from_dlpack(
        sync_workspace(a_q.device), assumed_align=16
    ).mark_layout_dynamic()
    args = (*args1[:9], *args2[:7], sync, args1[9])
    assert ceil_div(n1, 128) <= KCOLS_MAX, "KACQ: more fc1 column tiles than KCOLS_MAX"
    key = (op1.tactic, op2.tactic, enable_pdl, swiglu_limit, grid_x, sm_version)
    hit = _COMPILED.get(key)
    if hit is None:
        hit = cute.compile(
            CuteDslSm120MoeMxfp8Mxfp4Fc12Fused(op1.cfg, op2.cfg, grid_x), *args
        )
        _COMPILED[key] = hit
    hit(*args)
    return out
