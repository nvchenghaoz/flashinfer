/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_ATTENTION_PREFILL_FP8_MMA_SM12X_CUH_
#define FLASHINFER_ATTENTION_PREFILL_FP8_MMA_SM12X_CUH_

// Batch prefill with FP8 (e4m3 or e5m2) K/V on SM12x where both matmuls run in FP8, for a paged KV
// cache or ragged K/V tensors.
//
// The FA2 path upcasts FP8 K/V to 16 bit and runs 16-bit mma.sync. On SM12x prefill is compute
// bound, so that path can at best match a 16-bit KV cache. This path instead rounds Q (per row) and
// P to e4m3 and runs S = Q K^T and O += P V as mma.sync.m16n8k32.e4m3 with FP32 accumulation, which
// has twice the 16-bit tensor throughput.
//
// Structure (one CTA = 8 warps = 128 packed query rows of one KV head, FlashInfer's work items):
//  * K/V tiles of 64 tokens are stored in TMA's 128B swizzle. They arrive by TMA from tensor maps
//    built from the K/V strides (HND or NHD): one box per column block for ragged K/V, one box per
//    page and column block for a paged cache with pages of 8 rows or more. Smaller pages
//    (power-of-two sizes) are copied by all threads with 16-byte cp.async into the same layout,
//    since each TMA copy has a fixed cost.
//  * Two groups of four warps take turns on the tensor pipe (named barriers 1 and 2): while one
//    group issues P V of tile j - 1 and Q K^T of tile j, the other runs its softmax. Warp w and
//    w + 4 share a sub-partition. The hand-off happens one k-step before the end of Q K^T so the
//    pipe never drains.
//  * No CTA-wide barrier per tile: with TMA the last warp to finish with a pipeline stage refills
//    it; with cp.async every thread copies its share once all warps have released the stage
//    (mbarrier).
//  * Q is rounded to e4m3 with one scale per row (folded into the softmax scale).
//    P = 2^(s - m + 8) is rounded to e4m3 in registers; the S accumulator is reused as the P
//    A-fragment through a token permutation inside each 32-token step, and V is read with
//    ldmatrix.trans in the same order.
//  * Split-KV partials use FA2's tmp_v / tmp_s layout and are merged by VariableLengthMergeStates.
//  * Masks follow FA2: none, causal, custom (bit mask) and multi-item scoring, each optionally
//    with a sliding window and a logits soft cap.

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "../exception.h"
#include "../fastdiv.cuh"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "cascade.cuh"
#include "mask.cuh"
#include "variants.cuh"

namespace flashinfer {
namespace fp8_mma_sm12x {

constexpr uint32_t kNumWarps = 8;
constexpr uint32_t kNumThreads = kNumWarps * 32;
constexpr uint32_t kCtaTileQ = kNumWarps * 16;  // packed (token, head-in-group) rows per CTA
constexpr uint32_t kKVTile = 64;                // KV tokens per pipeline stage
constexpr uint32_t kRowBytes = 128;             // TMA box / swizzle row width

template <uint32_t HEAD_DIM>
struct Traits {
  static_assert(HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256,
                "FP8-MMA prefill supports head_dim 64, 128 and 256");
  static constexpr uint32_t kKSteps = HEAD_DIM / 32;  // k32 steps of Q K^T
  static constexpr uint32_t kChunks = HEAD_DIM / 16;  // 16-dim chunks of V / O
  // Tiles are stored as column blocks of up to 128 bytes per row (the TMA swizzle width).
  static constexpr uint32_t kBlockBytes = HEAD_DIM < 128 ? HEAD_DIM : 128;
  static constexpr uint32_t kColBlocks = HEAD_DIM / kBlockBytes;
  // Pipeline stages of 64-token K and V tiles: three at head_dim 256 fill the 99 KB of shared
  // memory; smaller head dims take a fourth for more latency hiding.
  static constexpr uint32_t kStages = HEAD_DIM == 256 ? 3 : 4;
  static constexpr uint32_t kTileBytes = kKVTile * HEAD_DIM;
  static constexpr uint32_t kStageBytes = 2 * kTileBytes;  // K then V
  static constexpr uint32_t kEarlyKs = kKSteps - 1;
  // head_dim >= 128: the two warp groups alternate on the tensor pipe (one CTA per SM). At head_dim
  // 64 a tile's MMAs are too short to hide the other group's softmax, so warps run freely instead
  // and two CTAs share an SM (registers allow it).
  static constexpr bool kPingPong = HEAD_DIM >= 128;
  static constexpr uint32_t kCtasPerSm = kPingPong ? 1 : 2;
  // stages + 1 KB for 1024-byte alignment of the swizzled tiles + full / empty mbarriers and stage
  // counters
  static constexpr uint32_t kSmemBytes = kStages * kStageBytes + 1024 + kStages * 32;
};

/*!
 * \brief Whether a prefill module can take this path (compile-time part). The runtime part (SM12x,
 *   page size, workload) is decided by the plan.
 */
template <typename DTypeQ, typename DTypeKV, typename DTypeO, uint32_t HEAD_DIM_QK,
          uint32_t HEAD_DIM_VO, PosEncodingMode POS_ENCODING_MODE, bool USE_SLIDING_WINDOW,
          bool USE_LOGITS_SOFT_CAP, bool USE_FP16_QK_REDUCTION>
constexpr bool kEligible =
    (std::is_same_v<DTypeKV, __nv_fp8_e4m3> || std::is_same_v<DTypeKV, __nv_fp8_e5m2>) &&
    (std::is_same_v<DTypeQ, half> || std::is_same_v<DTypeQ, nv_bfloat16>) &&
    (std::is_same_v<DTypeO, half> || std::is_same_v<DTypeO, nv_bfloat16>) &&
    HEAD_DIM_QK == HEAD_DIM_VO && (HEAD_DIM_QK == 64 || HEAD_DIM_QK == 128 || HEAD_DIM_QK == 256) &&
    POS_ENCODING_MODE == PosEncodingMode::kNone && !USE_FP16_QK_REDUCTION;

/*!
 * \brief Whether this path runs a prefill module's attention variant for MASK_MODE: FlashInfer's
 * default variant on an eligible module (custom-variant modules keep the FA2 path).
 */
template <typename DTypeQ, typename DTypeKV, typename DTypeO, uint32_t HEAD_DIM_QK,
          uint32_t HEAD_DIM_VO, PosEncodingMode POS_ENCODING_MODE, bool USE_SLIDING_WINDOW,
          bool USE_LOGITS_SOFT_CAP, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant>
constexpr bool kSupportsVariant =
    kEligible<DTypeQ, DTypeKV, DTypeO, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
              USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, USE_FP16_QK_REDUCTION> &&
    std::is_same_v<AttentionVariant,
                   DefaultAttention<MASK_MODE == MaskMode::kCustom, USE_SLIDING_WINDOW,
                                    USE_LOGITS_SOFT_CAP, /*use_alibi=*/false>>;

/*!
 * \brief Runtime part of the eligibility check that does not depend on the batch. Ragged K/V plans
 *   with page size 1.
 */
inline bool RuntimeSupported(uint32_t page_size) {
  int dev_id = 0, major = 0;
  if (cudaGetDevice(&dev_id) != cudaSuccess) return false;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev_id) != cudaSuccess) {
    return false;
  }
  // Power-of-two pages up to 256 rows (TMA box limit): page arithmetic is shifts and masks, and a
  // 64-token tile either covers whole pages or lies inside one.
  const bool page_ok = page_size >= 1 && page_size <= 256 && (page_size & (page_size - 1)) == 0;
  return major == 12 && page_ok;
}

DEFINE_HAS_MEMBER(paged_kv)
DEFINE_HAS_MEMBER(maybe_prefix_len_ptr)
DEFINE_HAS_MEMBER(maybe_token_pos_in_items_ptr)
DEFINE_HAS_MEMBER(token_pos_in_items_len)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define FLASHINFER_FP8_MMA_SM12X_DEVICE 1
#endif
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Tile layout: [column block][64 rows][block bytes] in TMA's swizzle, which is a function of the
// shared-memory address: 128-byte rows XOR the 16-byte chunk index with (row % 8), 64-byte rows
// (head_dim 64) with ((row / 2) % 4). Either way 8 consecutive rows of one chunk hit 8 distinct
// bank groups, so ldmatrix over 8 rows is conflict free.
template <uint32_t HEAD_DIM>
__device__ __forceinline__ uint32_t swz(uint32_t row, uint32_t chunk) {
  if constexpr (HEAD_DIM >= 128) {
    return (chunk >> 3) * (kKVTile * kRowBytes) + row * kRowBytes +
           (((chunk & 7) ^ (row & 7)) << 4);
  } else {
    return row * HEAD_DIM + ((chunk ^ ((row >> 1) & 3)) << 4);
  }
}

#ifdef FLASHINFER_FP8_MMA_SM12X_DEVICE
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t* r) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_trans(uint32_t addr, uint32_t* r) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
// D += A * B with A (Q or P) in e4m3 and B (K or V) in the KV cache's FP8 format.
template <typename DTypeKV>
__device__ __forceinline__ void mma_fp8(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  if constexpr (std::is_same_v<DTypeKV, __nv_fp8_e5m2>) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
}
// Byte i of the result holds x_i rounded to e4m3.
__device__ __forceinline__ uint32_t pack_e4m3x4(float x0, float x1, float x2, float x3) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(x1), "f"(x0));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(x3), "f"(x2));
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float tanh_approx(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
// Arrives on bar once all of this thread's earlier cp.async copies have landed.
__device__ __forceinline__ void cp_async_mbar_arrive(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t phase) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n"
      "}\n" ::"r"(bar),
      "r"(phase)
      : "memory");
}
// One box: 128 bytes x `rows` rows of one page, one column block, one KV head.
__device__ __forceinline__ void tma_load_box(uint32_t dst, const CUtensorMap* map,
                                             uint32_t row_in_page, uint32_t col_block,
                                             uint32_t kv_head, uint32_t page, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, "
      "%3, "
      "%4, %5, %6}], [%7];\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(0), "r"(row_in_page), "r"(col_block), "r"(kv_head),
      "r"(page), "r"(bar)
      : "memory");
}
// Ragged K/V: one box of 64 rows x 128 bytes of one column block, one KV head.
__device__ __forceinline__ void tma_load_rows(uint32_t dst, const CUtensorMap* map, uint32_t row,
                                              uint32_t col_block, uint32_t kv_head, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, "
      "%3, "
      "%4, %5}], [%6];\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(0), "r"(row), "r"(col_block), "r"(kv_head), "r"(bar)
      : "memory");
}
__device__ __forceinline__ void named_sync(uint32_t id, uint32_t count) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(count) : "memory");
}
__device__ __forceinline__ void named_arrive(uint32_t id, uint32_t count) {
  asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(count) : "memory");
}

template <typename T>
__device__ __forceinline__ float2 to_float2(uint32_t v) {
  if constexpr (std::is_same_v<T, half>) {
    return __half22float2(*reinterpret_cast<const __half2*>(&v));
  } else {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
  }
}
template <typename T>
__device__ __forceinline__ uint32_t from_float2(float a, float b) {
  if constexpr (std::is_same_v<T, half>) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
  } else {
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
  }
}

struct RowState {
  float m0, m1, l0, l1;
};

// S = Q K^T for one tile. After EARLY_KS k-steps the pipe is handed to the other warp group.
template <uint32_t HEAD_DIM, typename DTypeKV>
__device__ __forceinline__ void compute_qk(const uint8_t* sk,
                                           const uint32_t (&qf)[HEAD_DIM / 32][4],
                                           float (&s)[kKVTile / 8][4], uint32_t lane,
                                           uint32_t handoff_bar) {
  constexpr bool kHandoff = Traits<HEAD_DIM>::kPingPong;
  using T = Traits<HEAD_DIM>;
  constexpr uint32_t NB = kKVTile / 8;
  const uint32_t ld_m = lane >> 3, ld_r = lane & 7;
#pragma unroll
  for (uint32_t nb = 0; nb < NB; ++nb)
#pragma unroll
    for (uint32_t e = 0; e < 4; ++e) s[nb][e] = 0.f;
#pragma unroll
  for (uint32_t ks = 0; ks < T::kKSteps; ++ks) {
    if (kHandoff && ks == T::kEarlyKs) named_arrive(handoff_bar, kNumThreads);
#pragma unroll
    for (uint32_t jj = 0; jj < NB / 2; ++jj) {
      uint32_t b[4];
      const uint32_t row = (2 * jj + (ld_m >> 1)) * 8 + ld_r;
      ldsm_x4(smem_u32(sk + swz<HEAD_DIM>(row, 2 * ks + (ld_m & 1))), b);
      mma_fp8<DTypeKV>(s[2 * jj], qf[ks], b[0], b[1]);
      mma_fp8<DTypeKV>(s[2 * jj + 1], qf[ks], b[2], b[3]);
    }
  }
}

// Online softmax of one tile (log2 domain; scores are scaled inside the exp). Rescales O when a
// row's max moves and packs P (x2^8, e4m3) into P V A-fragments. Within each 32-token step logical
// k = 4t + i maps to token {2t, 2t + 1, 8 + 2t, 9 + 2t}[i] (+16 for the upper half): what this
// thread holds in S and what ldmatrix.trans + byte permutes deliver for V. Row r keeps KV positions
// lo[r] <= pos <= hi[r] for which keep(r, pos) holds when MASK. With SOFT_CAP the raw scores go
// through tanh(s * pre[r]) first (FlashInfer's logits soft cap; pre folds the row's Q scale), and
// sc is cap * log2e.
template <uint32_t HEAD_DIM, bool MASK, bool SOFT_CAP, typename KeepFn>
__device__ __forceinline__ void softmax_to_p(float (&s)[kKVTile / 8][4],
                                             uint32_t (&pa)[kKVTile / 32][4],
                                             float (&o)[HEAD_DIM / 16][2][4], RowState& st,
                                             float sc0, float sc1, float pre0, float pre1,
                                             int kv_base, const int (&lo)[2], const int (&hi)[2],
                                             const KeepFn& keep, uint32_t lane) {
  constexpr uint32_t NB = kKVTile / 8;
  const float kNegInf = -INFINITY;
  const int t = lane & 3;
  float mx0 = kNegInf, mx1 = kNegInf;
#pragma unroll
  for (uint32_t nb = 0; nb < NB; ++nb) {
#pragma unroll
    for (uint32_t e = 0; e < 2; ++e) {
      if constexpr (SOFT_CAP) {
        s[nb][e] = tanh_approx(s[nb][e] * pre0);
        s[nb][2 + e] = tanh_approx(s[nb][2 + e] * pre1);
      }
      if constexpr (MASK) {
        const int pos = kv_base + static_cast<int>(nb * 8) + 2 * t + static_cast<int>(e);
        if (pos < lo[0] || pos > hi[0] || !keep(0, pos)) s[nb][e] = kNegInf;
        if (pos < lo[1] || pos > hi[1] || !keep(1, pos)) s[nb][2 + e] = kNegInf;
      }
      mx0 = fmaxf(mx0, s[nb][e]);
      mx1 = fmaxf(mx1, s[nb][2 + e]);
    }
  }
  mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
  mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
  mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
  mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
  const float mn0 = fmaxf(st.m0, mx0 * sc0), mn1 = fmaxf(st.m1, mx1 * sc1);
  const float base0 = mn0 == kNegInf ? 0.f : mn0;
  const float base1 = mn1 == kNegInf ? 0.f : mn1;
  const float corr0 = ex2(st.m0 - base0), corr1 = ex2(st.m1 - base1);
  st.m0 = mn0;
  st.m1 = mn1;
  const float pb0 = base0 - 8.f,
              pb1 = base1 - 8.f;  // P comes out scaled by 2^8; l uses the same units
  float l0 = 0.f, l1 = 0.f;
#pragma unroll
  for (uint32_t nb = 0; nb < NB; ++nb) {
#pragma unroll
    for (uint32_t e = 0; e < 2; ++e) {
      s[nb][e] = ex2(fmaf(s[nb][e], sc0, -pb0));
      s[nb][2 + e] = ex2(fmaf(s[nb][2 + e], sc1, -pb1));
      l0 += s[nb][e];
      l1 += s[nb][2 + e];
    }
  }
  st.l0 = st.l0 * corr0 + l0;
  st.l1 = st.l1 * corr1 + l1;
#pragma unroll
  for (uint32_t kk = 0; kk < kKVTile / 32; ++kk) {
    pa[kk][0] = pack_e4m3x4(s[4 * kk][0], s[4 * kk][1], s[4 * kk + 1][0], s[4 * kk + 1][1]);
    pa[kk][1] = pack_e4m3x4(s[4 * kk][2], s[4 * kk][3], s[4 * kk + 1][2], s[4 * kk + 1][3]);
    pa[kk][2] = pack_e4m3x4(s[4 * kk + 2][0], s[4 * kk + 2][1], s[4 * kk + 3][0], s[4 * kk + 3][1]);
    pa[kk][3] = pack_e4m3x4(s[4 * kk + 2][2], s[4 * kk + 2][3], s[4 * kk + 3][2], s[4 * kk + 3][3]);
  }
  if (__any_sync(0xffffffffu, corr0 != 1.f || corr1 != 1.f)) {
#pragma unroll
    for (uint32_t c = 0; c < HEAD_DIM / 16; ++c)
#pragma unroll
      for (uint32_t par = 0; par < 2; ++par) {
        o[c][par][0] *= corr0;
        o[c][par][1] *= corr0;
        o[c][par][2] *= corr1;
        o[c][par][3] *= corr1;
      }
  }
}

// O += P V. TAIL clears V bytes of tokens >= valid_rows (pages can hold NaN past the sequence end
// and 0 * NaN = NaN; P is already 0 there).
template <uint32_t HEAD_DIM, bool TAIL, typename DTypeKV>
__device__ __forceinline__ void compute_pv(const uint8_t* sv, const uint32_t (&pa)[kKVTile / 32][4],
                                           float (&o)[HEAD_DIM / 16][2][4], int valid_rows,
                                           uint32_t lane) {
  const uint32_t ld_m = lane >> 3, ld_r = lane & 7, t = lane & 3;
#pragma unroll
  for (uint32_t kk = 0; kk < kKVTile / 32; ++kk) {
    uint32_t keep0 = 0xffffffffu, keep1 = 0xffffffffu;
    if constexpr (TAIL) {
      const int b = static_cast<int>(kk * 32);
      const int tok[4] = {static_cast<int>(2 * t), static_cast<int>(2 * t + 1),
                          static_cast<int>(8 + 2 * t), static_cast<int>(9 + 2 * t)};
      keep0 = keep1 = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        keep0 |= (b + tok[i] < valid_rows ? 0xffu : 0u) << (8 * i);
        keep1 |= (b + 16 + tok[i] < valid_rows ? 0xffu : 0u) << (8 * i);
      }
    }
    const uint32_t row = kk * 32 + ld_m * 8 + ld_r;
#pragma unroll
    for (uint32_t c = 0; c < HEAD_DIM / 16; ++c) {
      uint32_t v[4];
      ldsm_x4_trans(smem_u32(sv + swz<HEAD_DIM>(row, c)), v);
      // v[i] bytes: V[2t][2g], V[2t][2g+1], V[2t+1][2g], V[2t+1][2g+1] of token rows 8i..8i+7.
      mma_fp8<DTypeKV>(o[c][0], pa[kk], __byte_perm(v[0], v[1], 0x6420) & keep0,
                       __byte_perm(v[2], v[3], 0x6420) & keep1);
      mma_fp8<DTypeKV>(o[c][1], pa[kk], __byte_perm(v[0], v[1], 0x7531) & keep0,
                       __byte_perm(v[2], v[3], 0x7531) & keep1);
    }
  }
}
#endif  // FLASHINFER_FP8_MMA_SM12X_DEVICE

template <uint32_t HEAD_DIM, MaskMode MASK_MODE, bool SLIDING_WINDOW, bool SOFT_CAP,
          typename Params>
__global__ void __launch_bounds__(kNumThreads, Traits<HEAD_DIM>::kCtasPerSm)
    BatchPrefillFP8MMAKernel(const __grid_constant__ Params params,
                             const __grid_constant__ CUtensorMap k_map,
                             const __grid_constant__ CUtensorMap v_map) {
#ifdef FLASHINFER_FP8_MMA_SM12X_DEVICE
  using T = Traits<HEAD_DIM>;
  using DTypeQ = typename Params::DTypeQ;
  using DTypeO = typename Params::DTypeO;
  using DTypeKV = typename Params::DTypeKV;
  using IdType = typename Params::IdType;
  constexpr uint32_t NB = kKVTile / 8;
  constexpr uint32_t kStages = T::kStages;
  constexpr bool kPaged = has_paged_kv_v<Params>;
  constexpr bool kCustom = MASK_MODE == MaskMode::kCustom;
  constexpr bool kMultiItem = MASK_MODE == MaskMode::kMultiItemScoring;
  // Multi-item scoring is causal plus, per query row inside an item, a masked gap between the
  // shared prefix and the row's own item. The custom mask replaces the causal mask.
  constexpr bool kCausal = MASK_MODE == MaskMode::kCausal || kMultiItem;

  // Work items come in increasing query-tile order, which under a causal mask means increasing KV
  // length; launching them in reverse puts the longest first and shortens the tail of the last
  // wave.
  const uint32_t bx = gridDim.x - 1 - blockIdx.x, kvh = blockIdx.z;
  if (params.block_valid_mask && !params.block_valid_mask[bx]) return;

  extern __shared__ __align__(128) uint8_t smem_raw[];
  uint8_t* smem = smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);
  const uint32_t full_bar = smem_u32(smem + kStages * T::kStageBytes);
  const uint32_t empty_bar = full_bar + 8 * kStages;
  int* stage_done = reinterpret_cast<int*>(smem + kStages * T::kStageBytes + 16 * kStages);
  const uint32_t tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const uint32_t g = lane / 4, t = lane % 4;

#if (__CUDACC_VER_MAJOR__ >= 12)
  asm volatile("griddepcontrol.wait;");
#endif

  const uint32_t request_idx = params.request_indices[bx];
  const uint32_t qo_tile_idx = params.qo_tile_indices[bx];
  const uint32_t kv_tile_idx = params.kv_tile_indices[bx];
  const uint32_t qo_len = params.get_qo_len(request_idx);
  const uint32_t kv_len = params.get_kv_len(request_idx);
  const uint_fastdiv group_size = params.group_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const bool partition_kv = params.partition_kv;
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  const uint32_t kv_len_safe = kv_len > 0 ? kv_len : 1;
  // FA2's window and chunk bounds (identical formulas, so partial rows line up with FA2's merge).
  const uint32_t window_left =
      (SLIDING_WINDOW && params.window_left >= 0) ? params.window_left : kv_len;
  const uint32_t packed_base = qo_tile_idx * kCtaTileQ;
  const uint32_t tok_min = packed_base / group_size;
  const uint32_t kv_start_idx = sub_if_greater_or_zero(kv_len + tok_min, qo_len + window_left);
  const uint32_t chunk_start =
      partition_kv ? min(kv_tile_idx * kv_chunk_size + kv_start_idx, kv_len) : kv_start_idx;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * kv_chunk_size + kv_start_idx, kv_len) : kv_len;
  const uint32_t num_kv_chunks =
      partition_kv ? ceil_div(min(kv_len_safe, window_left + kCtaTileQ), kv_chunk_size) : 1;
  const IdType o_row_base = params.o_indptr[request_idx];
  // Paged: this request's page table (power-of-two pages, see RuntimeSupported). Ragged: its first
  // row.
  const IdType* pages = nullptr;
  uint32_t num_pages = 0, page_size = kKVTile, kv_row_base = 0;
  if constexpr (kPaged) {
    const auto& paged_kv = params.paged_kv;
    pages = paged_kv.indices + paged_kv.indptr[request_idx];
    num_pages = paged_kv.indptr[request_idx + 1] - paged_kv.indptr[request_idx];
    page_size = paged_kv.page_size;
  } else {
    kv_row_base = params.kv_indptr[request_idx];
  }
  const uint32_t page_shift = __ffs(page_size) - 1;
  const int prefix = static_cast<int>(kv_len) - static_cast<int>(qo_len);

  // This thread's two packed rows (warp rows g and g + 8) and the KV positions each keeps:
  // lo = max(chunk start, q_pos - window_left), hi = min(chunk end - 1, q_pos if causal).
  uint32_t tok[2], head[2];
  int lo[2], hi[2];
#pragma unroll
  for (uint32_t rr = 0; rr < 2; ++rr) {
    uint32_t q, r;
    group_size.divmod(packed_base + warp * 16 + g + rr * 8, q, r);
    tok[rr] = q;
    head[rr] = kvh * group_size + r;
    const int q_pos = prefix + static_cast<int>(q);
    lo[rr] = max(static_cast<int>(chunk_start),
                 SLIDING_WINDOW ? q_pos - static_cast<int>(window_left) : 0);
    hi[rr] = min(static_cast<int>(chunk_end) - 1, kCausal ? q_pos : static_cast<int>(kv_len) - 1);
  }
  // Tiles start 64-aligned (so they are page aligned); [unmasked_begin, unmasked_end) need no mask.
  const uint32_t tok_max = min(qo_len - 1, (packed_base + kCtaTileQ - 1) / group_size);
  const int tile_base = static_cast<int>(chunk_start) & ~static_cast<int>(kKVTile - 1);
  const int cta_end = kCausal
                          ? min(static_cast<int>(chunk_end), prefix + static_cast<int>(tok_max) + 1)
                          : static_cast<int>(chunk_end);
  const int num_tiles =
      cta_end > static_cast<int>(chunk_start)
          ? (cta_end - tile_base + static_cast<int>(kKVTile) - 1) / static_cast<int>(kKVTile)
          : 0;
  const int all_lo =
      max(static_cast<int>(chunk_start),
          SLIDING_WINDOW ? prefix + static_cast<int>(tok_max) - static_cast<int>(window_left) : 0);
  const int all_hi =
      min(static_cast<int>(chunk_end) - 1,
          kCausal ? prefix + static_cast<int>(tok_min) : static_cast<int>(kv_len) - 1);
  int unmasked_begin =
      (max(0, all_lo - tile_base) + static_cast<int>(kKVTile) - 1) / static_cast<int>(kKVTile);
  int unmasked_end =
      max(unmasked_begin, min(num_tiles, (all_hi - tile_base + 1) / static_cast<int>(kKVTile)));

  // Masks beyond [lo, hi]. Custom: bit (q, pos) of the request's row-major qo_len x kv_len bit
  // mask. Multi-item scoring: a row at q_pos >= prefix_len (inside an item) does not see the
  // earlier items, [prefix_len, q_pos - token_pos_in_item], as in FA2's
  // logits_mask_multi_item_scoring.
  const uint8_t* custom_mask = nullptr;
  uint64_t mask_row_bit[2] = {0, 0};
  int gap_lo[2] = {0, 0}, gap_hi[2] = {0, 0};
  if constexpr (kCustom) {
    custom_mask = params.maybe_custom_mask + params.maybe_mask_indptr[request_idx];
#pragma unroll
    for (uint32_t rr = 0; rr < 2; ++rr) {
      mask_row_bit[rr] = static_cast<uint64_t>(min(tok[rr], qo_len - 1)) * kv_len;
    }
    unmasked_end = unmasked_begin;
  } else if constexpr (kMultiItem) {
    static_assert(has_maybe_prefix_len_ptr_v<Params> &&
                  has_maybe_token_pos_in_items_ptr_v<Params> &&
                  has_token_pos_in_items_len_v<Params>);
    const int prefix_len = static_cast<int>(params.maybe_prefix_len_ptr[request_idx]);
    const uint16_t* item_pos = params.maybe_token_pos_in_items_ptr +
                               request_idx * static_cast<int64_t>(params.token_pos_in_items_len);
#pragma unroll
    for (uint32_t rr = 0; rr < 2; ++rr) {
      const int q_pos = prefix + static_cast<int>(tok[rr]);
      if (tok[rr] < qo_len && q_pos >= prefix_len) {
        gap_lo[rr] = prefix_len;
        gap_hi[rr] = q_pos - static_cast<int>(item_pos[q_pos - prefix_len]) + 1;
      }
    }
    // Tiles entirely inside the shared prefix see no gap.
    unmasked_end = max(unmasked_begin,
                       min(unmasked_end, (prefix_len - tile_base) / static_cast<int>(kKVTile)));
  }
  auto keep = [&](uint32_t rr, int pos) -> bool {
    if constexpr (kCustom) {
      const uint64_t bit = mask_row_bit[rr] + static_cast<uint64_t>(pos);
      return (custom_mask[bit >> 3] >> (bit & 7)) & 1;
    } else if constexpr (kMultiItem) {
      return pos < gap_lo[rr] || pos >= gap_hi[rr];
    } else {
      return true;
    }
  };

  // ---- TMA producer (ragged K/V and pages of 8 rows or more): any warp issues a tile's boxes, one
  // per lane. A paged tile is min(page, 64)-row boxes per page, column block and tensor: at most 32
  // copies.
  const uint32_t rows_per_box = min(page_size, kKVTile);
  const uint32_t boxes_per_tile = kKVTile / rows_per_box;  // pages (or one page part) per tile
  // This lane's page id for tile `tile` (box j = lane; lanes past the tile's boxes load a clamped,
  // valid index and are ignored when issuing).
  int page_id = 0;
  auto fetch_pages = [&](int tile) {
    if constexpr (kPaged) {
      const uint32_t pos = tile_base + tile * kKVTile + lane * rows_per_box;
      page_id = pages[min(pos >> page_shift, num_pages - 1)];
    }
  };
  // Called by all 32 lanes of one warp.
  auto issue_tile = [&](int tile) {
    const uint32_t stage = tile % kStages;
    const uint32_t dst = smem_u32(smem + stage * T::kStageBytes);
    if (lane == 0) mbar_expect_tx(full_bar + 8 * stage, T::kStageBytes);
    __syncwarp();
    const uint32_t tile_pos = tile_base + tile * kKVTile;
    if constexpr (!kPaged) {
      // Rows past this request belong to the next one (or are zero-filled past the tensor); they
      // are masked like the tail of a page.
      if (lane < T::kColBlocks * 2) {
        const uint32_t cb = lane >> 1;
        const bool is_v = lane & 1;
        tma_load_rows(dst + cb * (kKVTile * T::kBlockBytes) + (is_v ? T::kTileBytes : 0),
                      is_v ? &v_map : &k_map, kv_row_base + tile_pos, cb, kvh,
                      full_bar + 8 * stage);
      }
    } else {
      const uint32_t j = lane / (T::kColBlocks * 2);  // page (or page part) within the tile
      const uint32_t cb = (lane >> 1) % T::kColBlocks;
      const bool is_v = lane & 1;
      const int pid = __shfl_sync(0xffffffffu, page_id, j);
      if (j < boxes_per_tile) {
        const uint32_t row_in_page = (tile_pos + j * rows_per_box) & (page_size - 1);
        const uint32_t off = cb * (kKVTile * T::kBlockBytes) + j * rows_per_box * T::kBlockBytes +
                             (is_v ? T::kTileBytes : 0);
        tma_load_box(dst + off, is_v ? &v_map : &k_map, row_in_page, cb, kvh, pid,
                     full_bar + 8 * stage);
      }
    }
  };

  // ---- cp.async producer: every thread copies its share of each tile in 16-byte pieces, pieces
  // c = tid + 256 i of K, then the same pieces of V; completion arrives on the stage's full
  // barrier, and a stage is refilled once all warps have arrived on its empty barrier after their
  // P V. Used at head_dim <= 128, where it beats TMA (one warp issuing a tile's boxes), and for
  // pages of fewer than 8 rows, where TMA's fixed per-copy cost limits the load rate.
  const bool ldgsts = HEAD_DIM <= 128 || (kPaged && page_size < 8);
  constexpr uint32_t kRowPieces = HEAD_DIM / 16;
  constexpr uint32_t kPiecesPerThread =
      kKVTile * kRowPieces / kNumThreads;  // per tensor: 1, 2 or 4
  int ld_page[kPiecesPerThread];  // paged: page ids of this thread's rows of the next tile to load
  auto ld_fetch = [&](int tile) {
    if constexpr (kPaged) {
#pragma unroll
      for (uint32_t i = 0; i < kPiecesPerThread; ++i) {
        const uint32_t pos = tile_base + tile * kKVTile + (tid + kNumThreads * i) / kRowPieces;
        ld_page[i] = pages[min(pos >> page_shift, num_pages - 1)];
      }
    }
  };
  auto ld_issue = [&](int tile) {
    const uint32_t stage = tile % kStages;
    uint8_t* dst = smem + stage * T::kStageBytes;
#pragma unroll
    for (uint32_t i = 0; i < kPiecesPerThread; ++i) {
      const uint32_t piece = tid + kNumThreads * i;
      const uint32_t row = piece / kRowPieces, col = piece % kRowPieces;
      const uint32_t pos = tile_base + tile * kKVTile + row;
      const uint8_t *k_src, *v_src;
      if constexpr (kPaged) {
        const auto& kv = params.paged_kv;
        const uint32_t in_page = pos & (page_size - 1);
        const size_t page = static_cast<size_t>(ld_page[i]);
        k_src = reinterpret_cast<const uint8_t*>(kv.k_data) + page * kv.stride_page +
                kvh * kv.stride_h + in_page * kv.stride_n;
        v_src = reinterpret_cast<const uint8_t*>(kv.v_data) + page * kv.v_stride_page +
                kvh * kv.v_stride_h + in_page * kv.v_stride_n;
      } else {
        // Rows past this request repeat its last row (masked like the tail of a page) and stay in
        // bounds.
        const size_t r = kv_row_base + min(pos, kv_len - 1);
        k_src = reinterpret_cast<const uint8_t*>(params.k) + r * params.k_stride_n +
                kvh * params.k_stride_h;
        v_src = reinterpret_cast<const uint8_t*>(params.v) + r * params.v_stride_n +
                kvh * params.v_stride_h;
      }
      const uint32_t off = swz<HEAD_DIM>(row, col);
      cp_async_16(smem_u32(dst + off), k_src + 16 * col);
      cp_async_16(smem_u32(dst + T::kTileBytes + off), v_src + 16 * col);
    }
    cp_async_mbar_arrive(full_bar + 8 * stage);
  };

  if (tid == 0) {
    for (uint32_t s = 0; s < kStages; ++s) {
      mbar_init(full_bar + 8 * s, ldgsts ? kNumThreads : 1);
      mbar_init(empty_bar + 8 * s, kNumWarps);
      stage_done[s] = 0;
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (ldgsts) {
    for (int s = 0; s < static_cast<int>(kStages) && s < num_tiles; ++s) {
      ld_fetch(s);
      ld_issue(s);
    }
    if (static_cast<int>(kStages) < num_tiles) ld_fetch(kStages);
  } else if (warp == 0) {
    for (int s = 0; s < static_cast<int>(kStages) && s < num_tiles; ++s) {
      fetch_pages(s);
      issue_tile(s);
    }
  }

  // ---- Q -> e4m3 A fragments, one scale per row.
  uint32_t qf[T::kKSteps][4];
  float qscale[2];
#pragma unroll
  for (uint32_t rr = 0; rr < 2; ++rr) {
    const uint32_t q_tok = min(tok[rr], qo_len - 1);
    const DTypeQ* qrow =
        params.q + static_cast<size_t>(params.q_indptr[request_idx] + q_tok) * params.q_stride_n +
        static_cast<size_t>(head[rr]) * params.q_stride_h;
    float amax = 0.f;
#pragma unroll
    for (uint32_t ks = 0; ks < T::kKSteps; ++ks) {
#pragma unroll
      for (uint32_t h = 0; h < 2; ++h) {
        const uint2 raw = *reinterpret_cast<const uint2*>(qrow + ks * 32 + h * 16 + 4 * t);
        const float2 a = to_float2<DTypeQ>(raw.x), b = to_float2<DTypeQ>(raw.y);
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y))));
      }
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float scale = amax > 0.f ? amax / 448.f : 1.f;
    const float inv = 1.f / scale;
    qscale[rr] = scale;
#pragma unroll
    for (uint32_t ks = 0; ks < T::kKSteps; ++ks) {
#pragma unroll
      for (uint32_t h = 0; h < 2; ++h) {
        const uint2 raw = *reinterpret_cast<const uint2*>(qrow + ks * 32 + h * 16 + 4 * t);
        const float2 a = to_float2<DTypeQ>(raw.x), b = to_float2<DTypeQ>(raw.y);
        qf[ks][rr + 2 * h] = pack_e4m3x4(a.x * inv, a.y * inv, b.x * inv, b.y * inv);
      }
    }
  }
  float sc0, sc1, pre0 = 0.f, pre1 = 0.f;
  if constexpr (SOFT_CAP) {  // s = cap * log2e * tanh(qk * sm_scale / cap)
    const float cap = params.logits_soft_cap;
    pre0 = qscale[0] * params.sm_scale / cap;
    pre1 = qscale[1] * params.sm_scale / cap;
    sc0 = sc1 = cap * math::log2e;
  } else {
    const float sm_scale_log2 = params.sm_scale * math::log2e;
    sc0 = qscale[0] * sm_scale_log2;
    sc1 = qscale[1] * sm_scale_log2;
  }

  RowState st{-INFINITY, -INFINITY, 0.f, 0.f};
  float o[T::kChunks][2][4];
#pragma unroll
  for (uint32_t c = 0; c < T::kChunks; ++c)
#pragma unroll
    for (uint32_t par = 0; par < 2; ++par)
#pragma unroll
      for (uint32_t e = 0; e < 4; ++e) o[c][par][e] = 0.f;

  if (num_tiles > 0) {
    const uint32_t grp = warp / 4;
    const uint32_t my_bar = 1 + grp, other_bar = 2 - grp;
    auto tile_ptr = [&](int j) { return smem + (j % kStages) * T::kStageBytes; };
    auto wait_tile = [&](int j) { mbar_wait(full_bar + 8 * (j % kStages), (j / kStages) & 1); };
    // The last warp done with tile j refills its stage with tile j + kStages. It issues the copies
    // at the top of its next iteration, where it would otherwise wait for the tensor pipe, not
    // between its Q K^T and softmax: with many boxes per tile (small pages) issuing takes a while.
    int refill = -1;
    auto release = [&](int j) {
      const uint32_t stage = j % kStages;
      __syncwarp();
      int last = 0;
      if (lane == 0) {
        __threadfence_block();
        last = atomicAdd(stage_done + stage, 1) == static_cast<int>(kNumWarps) - 1;
        if (last) stage_done[stage] = 0;
      }
      last = __shfl_sync(0xffffffffu, last, 0);
      if (last && j + static_cast<int>(kStages) < num_tiles) refill = j + kStages;
    };
    float s[NB][4];
    uint32_t pa[kKVTile / 32][4];
    constexpr bool kPingPong = T::kPingPong;
    if (kPingPong && grp == 1) named_arrive(1, kNumThreads);  // group 0 takes the pipe first
    for (int j = 0; j < num_tiles; ++j) {
      if (ldgsts) {
        // Refill the stage of tile j - 2 (all warps finished its P V in their previous iteration).
        const int t = j - 2 + static_cast<int>(kStages);
        if (j >= 2 && t < num_tiles) {
          mbar_wait(empty_bar + 8 * ((j - 2) % kStages), ((j - 2) / kStages) & 1);
          ld_issue(t);
          if (t + 1 < num_tiles) ld_fetch(t + 1);
        }
      } else {
        if (refill >= 0) {
          asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
          issue_tile(refill);
          refill = -1;
        }
        // This warp may refill stage (j - 1) with tile j - 1 + kStages after release(j - 1) below;
        // fetch those page ids now so the global load overlaps this iteration's MMAs.
        if (j > 0 && j - 1 + static_cast<int>(kStages) < num_tiles) fetch_pages(j - 1 + kStages);
      }
      wait_tile(j);
      if (kPingPong) named_sync(my_bar, kNumThreads);
      if (j > 0) {
        compute_pv<HEAD_DIM, false, DTypeKV>(tile_ptr(j - 1) + T::kTileBytes, pa, o, kKVTile, lane);
        if (ldgsts) {
          __syncwarp();
          if (lane == 0) mbar_arrive(empty_bar + 8 * ((j - 1) % kStages));
        }
      }
      compute_qk<HEAD_DIM, DTypeKV>(tile_ptr(j), qf, s, lane, other_bar);
      if (j > 0 && !ldgsts) release(j - 1);
      const int kv_base = tile_base + j * static_cast<int>(kKVTile);
      if (j >= unmasked_begin && j < unmasked_end) {
        softmax_to_p<HEAD_DIM, false, SOFT_CAP>(s, pa, o, st, sc0, sc1, pre0, pre1, kv_base, lo, hi,
                                                keep, lane);
      } else {
        softmax_to_p<HEAD_DIM, true, SOFT_CAP>(s, pa, o, st, sc0, sc1, pre0, pre1, kv_base, lo, hi,
                                               keep, lane);
      }
    }
    const int last = num_tiles - 1;
    const int valid_rows =
        static_cast<int>(kv_len) - (tile_base + last * static_cast<int>(kKVTile));
    if (kPingPong) named_sync(my_bar, kNumThreads);
    if (valid_rows < static_cast<int>(kKVTile)) {
      compute_pv<HEAD_DIM, true, DTypeKV>(tile_ptr(last) + T::kTileBytes, pa, o, valid_rows, lane);
    } else {
      compute_pv<HEAD_DIM, false, DTypeKV>(tile_ptr(last) + T::kTileBytes, pa, o, kKVTile, lane);
    }
    if (kPingPong) {
      named_arrive(other_bar, kNumThreads);
      if (grp == 0) named_sync(my_bar, kNumThreads);  // consume group 1's last arrive
    }
  }

  // ---- Epilogue: normalized O and log2-domain LSE (FA2 conventions).
  float l[2] = {st.l0, st.l1};
#pragma unroll
  for (uint32_t rr = 0; rr < 2; ++rr) {
    l[rr] += __shfl_xor_sync(0xffffffffu, l[rr], 1);
    l[rr] += __shfl_xor_sync(0xffffffffu, l[rr], 2);
  }
  const float m[2] = {st.m0, st.m1};
  const uint32_t o_stride_n = num_qo_heads * HEAD_DIM;
#pragma unroll
  for (uint32_t rr = 0; rr < 2; ++rr) {
    if (tok[rr] >= qo_len) continue;
    const float f = l[rr] > 0.f ? 1.f / l[rr] : 0.f;  // l is in P units (x2^8), as is O
    const size_t row = partition_kv ? static_cast<size_t>(o_row_base) +
                                          static_cast<size_t>(tok[rr]) * num_kv_chunks + kv_tile_idx
                                    : static_cast<size_t>(o_row_base) + tok[rr];
    if (params.lse != nullptr && t == 0) {
      params.lse[row * num_qo_heads + head[rr]] =
          l[rr] > 0.f ? m[rr] + math::ptx_log2(l[rr]) - 8.f : -INFINITY;
    }
    DTypeO* orow = params.o + row * o_stride_n + static_cast<size_t>(head[rr]) * HEAD_DIM;
#pragma unroll
    for (uint32_t c = 0; c < T::kChunks; ++c) {
      // Even-dim tile holds dims 16c + 4t + {0, 2}, odd-dim tile 16c + 4t + {1, 3}.
      uint2 packed;
      packed.x = from_float2<DTypeO>(o[c][0][2 * rr] * f, o[c][1][2 * rr] * f);
      packed.y = from_float2<DTypeO>(o[c][0][2 * rr + 1] * f, o[c][1][2 * rr + 1] * f);
      *reinterpret_cast<uint2*>(orow + 16 * c + 4 * t) = packed;
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12)
  asm volatile("griddepcontrol.launch_dependents;");
#endif
#endif  // FLASHINFER_FP8_MMA_SM12X_DEVICE
}

inline cudaError_t EncodeTensorMap(CUtensorMap* map, const void* base, uint32_t rank,
                                   const cuuint64_t* dims, const cuuint64_t* strides,
                                   const cuuint32_t* box, uint32_t block_bytes) {
  static PFN_cuTensorMapEncodeTiled_v12000 encode = nullptr;
  if (encode == nullptr) {
    cudaDriverEntryPointQueryResult status;
    FLASHINFER_CUDA_CALL(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",
                                                          reinterpret_cast<void**>(&encode), 12000,
                                                          cudaEnableDefault, &status));
    if (status != cudaDriverEntryPointSuccess || encode == nullptr) return cudaErrorNotSupported;
  }
  const cuuint32_t elem_strides[5] = {1, 1, 1, 1, 1};
  const CUresult r =
      encode(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, const_cast<void*>(base), dims, strides, box,
             elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
             block_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B,
             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

/*!
 * \brief Tensor map for K or V of a paged cache: dims (innermost first) column-block bytes, page
 * row, column block, KV head, page; a box is min(page, 64) rows of one column block of one head.
 */
template <uint32_t HEAD_DIM>
inline cudaError_t MakePagedKVTensorMap(CUtensorMap* map, const void* base, int64_t num_pages,
                                        uint32_t page_size, uint32_t num_kv_heads,
                                        int64_t stride_page, int64_t stride_n, int64_t stride_h) {
  constexpr uint32_t kBlock = Traits<HEAD_DIM>::kBlockBytes;
  const cuuint64_t dims[5] = {kBlock, page_size, Traits<HEAD_DIM>::kColBlocks, num_kv_heads,
                              static_cast<cuuint64_t>(num_pages)};
  const cuuint64_t strides[4] = {static_cast<cuuint64_t>(stride_n), kBlock,
                                 static_cast<cuuint64_t>(stride_h),
                                 static_cast<cuuint64_t>(stride_page)};
  const cuuint32_t box[5] = {kBlock, page_size < kKVTile ? page_size : kKVTile, 1, 1, 1};
  return EncodeTensorMap(map, base, 5, dims, strides, box, kBlock);
}

/*!
 * \brief Tensor map for ragged K or V: dims column-block bytes, row, column block, KV head; a box
 * is 64 rows of one column block of one head.
 */
template <uint32_t HEAD_DIM>
inline cudaError_t MakeRaggedKVTensorMap(CUtensorMap* map, const void* base, int64_t num_rows,
                                         uint32_t num_kv_heads, int64_t stride_n,
                                         int64_t stride_h) {
  constexpr uint32_t kBlock = Traits<HEAD_DIM>::kBlockBytes;
  const cuuint64_t dims[4] = {kBlock, static_cast<cuuint64_t>(num_rows),
                              Traits<HEAD_DIM>::kColBlocks, num_kv_heads};
  const cuuint64_t strides[3] = {static_cast<cuuint64_t>(stride_n), kBlock,
                                 static_cast<cuuint64_t>(stride_h)};
  const cuuint32_t box[4] = {kBlock, kKVTile, 1, 1};
  return EncodeTensorMap(map, base, 4, dims, strides, box, kBlock);
}

template <uint32_t HEAD_DIM, MaskMode MASK_MODE, bool SLIDING_WINDOW, bool SOFT_CAP,
          typename Params>
cudaError_t LaunchFP8MMA(Params params, const CUtensorMap& k_map, const CUtensorMap& v_map,
                         uint32_t num_kv_heads, typename Params::DTypeO* tmp_v, float* tmp_s,
                         bool enable_pdl, cudaStream_t stream) {
  constexpr uint32_t smem_size = Traits<HEAD_DIM>::kSmemBytes;
  auto kernel = BatchPrefillFP8MMAKernel<HEAD_DIM, MASK_MODE, SLIDING_WINDOW, SOFT_CAP, Params>;
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  typename Params::DTypeO* o = params.o;
  float* lse = params.lse;
  if (tmp_v != nullptr) {
    params.partition_kv = true;
    params.o = tmp_v;
    params.lse = tmp_s;
  } else {
    params.partition_kv = false;
  }
  cudaLaunchAttribute attribute[1];
  cudaLaunchConfig_t config;
  config.gridDim = dim3(params.padded_batch_size, 1, num_kv_heads);
  config.blockDim = dim3(kNumThreads);
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = enable_pdl ? 1 : 0;
  config.attrs = attribute;
  config.numAttrs = 1;
  FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params, k_map, v_map));
  if (tmp_v != nullptr) {
    FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
        tmp_v, tmp_s, params.merge_indptr, o, lse, params.max_total_num_rows, params.total_num_rows,
        params.num_qo_heads, HEAD_DIM, enable_pdl, stream));
  }
  return cudaSuccess;
}

}  // namespace fp8_mma_sm12x

/*!
 * \brief FP8-MMA paged prefill on SM12x (see prefill_fp8_mma_sm12x.cuh). Same contract as
 *   BatchPrefillWithPagedKVCacheDispatched for CTA_TILE_Q = 128 with the default attention variant.
 */
template <uint32_t HEAD_DIM, MaskMode MASK_MODE, bool SLIDING_WINDOW, bool SOFT_CAP,
          typename Params>
cudaError_t BatchPrefillWithPagedKVCacheFP8MMADispatched(Params params,
                                                         typename Params::DTypeO* tmp_v,
                                                         float* tmp_s, int64_t num_pages,
                                                         bool enable_pdl, cudaStream_t stream) {
  using namespace fp8_mma_sm12x;
  if (params.padded_batch_size == 0) return cudaSuccess;
  const auto& kv = params.paged_kv;
  CUtensorMap k_map, v_map;
  FLASHINFER_CUDA_CALL(MakePagedKVTensorMap<HEAD_DIM>(&k_map, kv.k_data, num_pages, kv.page_size,
                                                      kv.num_heads, kv.stride_page, kv.stride_n,
                                                      kv.stride_h));
  FLASHINFER_CUDA_CALL(MakePagedKVTensorMap<HEAD_DIM>(&v_map, kv.v_data, num_pages, kv.page_size,
                                                      kv.num_heads, kv.v_stride_page, kv.v_stride_n,
                                                      kv.v_stride_h));
  return LaunchFP8MMA<HEAD_DIM, MASK_MODE, SLIDING_WINDOW, SOFT_CAP>(
      params, k_map, v_map, kv.num_heads, tmp_v, tmp_s, enable_pdl, stream);
}

/*!
 * \brief FP8-MMA ragged prefill on SM12x. Same contract as BatchPrefillWithRaggedKVCacheDispatched
 * for CTA_TILE_Q = 128 with the default attention variant; num_kv_rows is the row count of K and V.
 */
template <uint32_t HEAD_DIM, MaskMode MASK_MODE, bool SLIDING_WINDOW, bool SOFT_CAP,
          typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheFP8MMADispatched(Params params,
                                                          typename Params::DTypeO* tmp_v,
                                                          float* tmp_s, int64_t num_kv_rows,
                                                          bool enable_pdl, cudaStream_t stream) {
  using namespace fp8_mma_sm12x;
  if (params.padded_batch_size == 0) return cudaSuccess;
  CUtensorMap k_map, v_map;
  FLASHINFER_CUDA_CALL(MakeRaggedKVTensorMap<HEAD_DIM>(
      &k_map, params.k, num_kv_rows, params.num_kv_heads, params.k_stride_n, params.k_stride_h));
  FLASHINFER_CUDA_CALL(MakeRaggedKVTensorMap<HEAD_DIM>(
      &v_map, params.v, num_kv_rows, params.num_kv_heads, params.v_stride_n, params.v_stride_h));
  return LaunchFP8MMA<HEAD_DIM, MASK_MODE, SLIDING_WINDOW, SOFT_CAP>(
      params, k_map, v_map, params.num_kv_heads, tmp_v, tmp_s, enable_pdl, stream);
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_PREFILL_FP8_MMA_SM12X_CUH_
