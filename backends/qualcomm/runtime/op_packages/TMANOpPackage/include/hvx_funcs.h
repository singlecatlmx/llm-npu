#pragma once

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/intrinsics.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

#include <hexagon_types.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#define UNUSED(x) (void)(x)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifndef _HVX_INTERNAL_H
#define _HVX_INTERNAL_H

#define VLEN 128
#define vmem(A)     *((HVX_Vector *)(A))
#define HVX_INLINE_ALWAYS inline __attribute__((unused,always_inline))
static HVX_INLINE_ALWAYS void l2fetch(const void *p, uint32_t stride,
                                      uint32_t width, uint32_t height,
                                      uint32_t dir)
{
#ifdef __hexagon__
    uint64_t control = HEXAGON_V64_CREATE_H(dir, stride, width, height);
    __asm__ __volatile__ (" l2fetch(%0,%1) " : :"r"(p),"r"(control));
#endif
}

#endif /* _HVX_INTERNAL_H */

static inline int32_t _fp32_to_bits(float x)
{
  union {
    float f;
    int32_t i;
  } u;
  u.f = x;
  return u.i;
}

static inline int16_t _fp16_to_bits(const __fp16 *x)
{
  union {
    __fp16 f;
    int16_t i;
  } u;
  u.f = *x;
  return u.i;
}

// ---- TMAN W4A8 helpers ----
// 这一段 helper 是当前 A8 路径真正的“执行语义中心”。
//
// 设计思路分三层：
// 1. 先用显式的 layout / offset / block 常量把数据组织固定下来；
// 2. 再用 scalar helper 把数学路径和 scratch 语义跑通；
// 3. 在不改 op 接口的前提下，逐步把热点路径替换成更实质的 HVX intrinsic。
//
// 当前它已经不是最早的纯 baseline：
// - `LinearA8` 已经引入目标专用 block staging；
// - `FinalizeA8` 已经引入一段更实质的 HVX round/clamp/s16-sat 路径。
//
// 但它也还没有完全达到旧 `W4A16 hvx_tbl(...)` 的成熟程度，
// 因此这里仍然保留了：
// - scalar fallback
// - target-specific helper
// - 部分 HVX 化 helper
// 三者并存的结构。

constexpr int32_t TMAN_A8_LUT_GROUP = 4;
constexpr int32_t TMAN_A8_LUT_SIZE = 16;
constexpr int32_t TMAN_A8_BITS = 4;
constexpr int32_t TMAN_A8_TILE_ROWS = 4;
constexpr int32_t TMAN_A8_TILE_M = 8;
constexpr int32_t TMAN_A8_QGROUPS_PER_VECTOR = VLEN / (TMAN_A8_LUT_SIZE * sizeof(int16_t));
constexpr int32_t TMAN_A8_LUT_VALUES_PER_VECTOR_BLOCK =
    TMAN_A8_QGROUPS_PER_VECTOR * TMAN_A8_LUT_SIZE;
constexpr int32_t TMAN_A8_WEIGHT_IDX_VALUES_PER_VECTOR_BLOCK =
    TMAN_A8_QGROUPS_PER_VECTOR * TMAN_A8_BITS;
constexpr int32_t TMAN_A8_OUTPUT_VALUES_PER_VECTOR = 128;
constexpr int32_t TMAN_A8_TARGET_GROUP_SIZE = 128;
constexpr bool TMAN_A8_TARGET_SYMMETRIC = true;
constexpr int32_t TMAN_A8_TARGET_QGROUPS_PER_GROUP =
    TMAN_A8_TARGET_GROUP_SIZE / TMAN_A8_LUT_GROUP;
constexpr int32_t TMAN_A8_TARGET_QGROUP_VECTOR_BLOCKS =
    TMAN_A8_TARGET_QGROUPS_PER_GROUP / TMAN_A8_QGROUPS_PER_VECTOR;

static_assert(TMAN_A8_QGROUPS_PER_VECTOR == 4, "Unexpected TMAN A8 LUT vector packing");

struct alignas(128) TmanA8LutBlock {
  int16_t values[TMAN_A8_QGROUPS_PER_VECTOR * TMAN_A8_LUT_SIZE];
};

static_assert(sizeof(TmanA8LutBlock) == 128, "TMAN A8 LUT block must be exactly one HVX vector line");

struct TmanA8IntTile {
  // group 内整数乘加结果先落到这个 tile。
  // 之所以单独保留 int32 tile，而不是直接边乘 scale 边写 float，
  // 是为了更贴近最终“整数累加 -> group scale 融合”的 kernel 结构。
  int32_t values[TMAN_A8_TILE_ROWS][TMAN_A8_TILE_M];
};

struct TmanA8FloatTile {
  // 一个 `row_tile x m_tile` 的 float 累加 tile。
  // 当前 `LinearA8` 先把每个 group 的 int32 部分和累加出来，
  // 再乘上 group-wise effective_scale，把结果累到这个 float tile 里。
  float values[TMAN_A8_TILE_ROWS][TMAN_A8_TILE_M];
};

struct alignas(128) TmanA8OutputBlock {
  // 最终输出按 128B block 暂存，便于：
  // 1. scalar fallback 直接写 block；
  // 2. HVX 路径在 block 粒度做 pack/store。
  int8_t values[TMAN_A8_OUTPUT_VALUES_PER_VECTOR];
};

static_assert(
    sizeof(TmanA8OutputBlock) == TMAN_A8_OUTPUT_VALUES_PER_VECTOR,
    "TMAN A8 output block must match one vector-line worth of int8 values");

struct TmanA8Layout {
  int32_t gemm_n;
  int32_t gemm_m;
  int32_t gemm_k;
  int32_t num_groups;
  int32_t qgroups_per_group;
  int32_t qgroup_vector_blocks;
  int32_t scale_row_stride;
};

inline TmanA8Layout tman_a8_make_layout(
    int32_t gemm_n,
    int32_t gemm_m,
    int32_t group_size) {
  TmanA8Layout layout;
  layout.gemm_n = gemm_n;
  layout.gemm_m = gemm_m;
  layout.gemm_k = layout.num_groups = layout.qgroups_per_group = layout.qgroup_vector_blocks =
      layout.scale_row_stride = 0;
  if (group_size <= 0 || (group_size % TMAN_A8_LUT_GROUP) != 0) {
    return layout;
  }
  layout.qgroups_per_group = group_size / TMAN_A8_LUT_GROUP;
  if ((layout.qgroups_per_group % TMAN_A8_QGROUPS_PER_VECTOR) != 0) {
    return layout;
  }
  layout.qgroup_vector_blocks = layout.qgroups_per_group / TMAN_A8_QGROUPS_PER_VECTOR;
  return layout;
}

inline bool tman_a8_populate_layout(
    TmanA8Layout* layout,
    int32_t gemm_n,
    int32_t gemm_m,
    int32_t gemm_k,
    int32_t group_size) {
  if (layout == nullptr) {
    return false;
  }
  *layout = tman_a8_make_layout(gemm_n, gemm_m, group_size);
  if (group_size <= 0 || gemm_k <= 0 || (gemm_k % group_size) != 0) {
    return false;
  }
  if (layout->qgroup_vector_blocks <= 0) {
    return false;
  }
  layout->gemm_k = gemm_k;
  layout->num_groups = gemm_k / group_size;
  layout->scale_row_stride = layout->num_groups;
  return true;
}

inline bool tman_a8_is_target_config(
    int32_t bits,
    int32_t group_size,
    bool symmetric) {
  return bits == TMAN_A8_BITS &&
      group_size == TMAN_A8_TARGET_GROUP_SIZE &&
      symmetric == TMAN_A8_TARGET_SYMMETRIC;
}

inline bool tman_a8_is_target_layout(const TmanA8Layout& layout) {
  return layout.qgroups_per_group == TMAN_A8_TARGET_QGROUPS_PER_GROUP &&
      layout.qgroup_vector_blocks == TMAN_A8_TARGET_QGROUP_VECTOR_BLOCKS;
}

inline bool tman_a8_is_vector_aligned(const void* ptr) {
  return ((((uintptr_t)ptr) & (VLEN - 1)) == 0);
}

inline size_t tman_a8_lut_offset(
    const TmanA8Layout& layout,
    int32_t row,
    int32_t group_idx,
    int32_t qgroup_in_group,
    int32_t lut_idx) {
  return ((((size_t)row * layout.num_groups) + group_idx) * layout.qgroups_per_group +
          qgroup_in_group) *
             TMAN_A8_LUT_SIZE +
         lut_idx;
}

inline size_t tman_a8_weight_idx_offset(
    const TmanA8Layout& layout,
    int32_t out_idx,
    int32_t group_idx,
    int32_t qgroup_in_group,
    int32_t bit_idx) {
  return ((((size_t)out_idx * layout.num_groups) + group_idx) * layout.qgroups_per_group +
          qgroup_in_group) *
             TMAN_A8_BITS +
         bit_idx;
}

inline size_t tman_a8_scale_offset(
    const TmanA8Layout& layout,
    int32_t out_idx,
    int32_t group_idx) {
  return ((size_t)out_idx * (size_t)layout.scale_row_stride) + group_idx;
}

inline size_t tman_a8_lut_block_offset(
    const TmanA8Layout& layout,
    int32_t row,
    int32_t group_idx,
    int32_t qgroup_block) {
  return ((((size_t)row * layout.num_groups) + group_idx) * layout.qgroup_vector_blocks +
          qgroup_block) *
         TMAN_A8_LUT_VALUES_PER_VECTOR_BLOCK;
}

inline const int16_t* tman_a8_lut_block_ptr(
    const TmanA8Layout& layout,
    const int16_t* l_ptr,
    int32_t row,
    int32_t group_idx,
    int32_t qgroup_block) {
  return l_ptr + tman_a8_lut_block_offset(layout, row, group_idx, qgroup_block);
}

inline const uint8_t* tman_a8_weight_idx_block_ptr(
    const TmanA8Layout& layout,
    const uint8_t* idx_ptr,
    int32_t out_idx,
    int32_t group_idx,
    int32_t qgroup_block) {
  const int32_t qg_base = qgroup_block * TMAN_A8_QGROUPS_PER_VECTOR;
  return idx_ptr + tman_a8_weight_idx_offset(layout, out_idx, group_idx, qg_base, 0);
}

inline int32_t tman_a8_group_dot_from_lut(
    const int16_t* lut_base,
    const uint8_t* idx_base,
    int32_t bits) {
  constexpr int32_t zero_bias = 1 << (TMAN_A8_BITS - 1);
  int32_t acc_q = -zero_bias * (int32_t)lut_base[TMAN_A8_LUT_SIZE - 1];
  for (int32_t bit = 0; bit < bits; ++bit) {
    const int32_t lut_idx = (int32_t)idx_base[bit];
    acc_q += (1 << bit) * (int32_t)lut_base[lut_idx];
  }
  return acc_q;
}

inline void tman_a8_build_single_lut_scalar(
    int32_t x0,
    int32_t x1,
    int32_t x2,
    int32_t x3,
    int16_t* lut_base) {
  for (int32_t idx = 0; idx < TMAN_A8_LUT_SIZE; ++idx) {
    int32_t acc = 0;
    if (idx & 0x1) acc += x0;
    if (idx & 0x2) acc += x1;
    if (idx & 0x4) acc += x2;
    if (idx & 0x8) acc += x3;
    lut_base[idx] = (int16_t)acc;
  }
}

inline void tman_a8_build_lut_block_scalar(
    const int8_t* x_base,
    TmanA8LutBlock* block) {
  // 一个 block 内包含 4 个 qgroup，对应 4 * 16 * sizeof(int16_t) = 128B。
  // 这个 helper 的组织方式已经和“一个 HVX vector line 一个 block”一致。
  for (int32_t lane = 0; lane < TMAN_A8_QGROUPS_PER_VECTOR; ++lane) {
    const int32_t x0 = (int32_t)x_base[lane * TMAN_A8_LUT_GROUP + 0];
    const int32_t x1 = (int32_t)x_base[lane * TMAN_A8_LUT_GROUP + 1];
    const int32_t x2 = (int32_t)x_base[lane * TMAN_A8_LUT_GROUP + 2];
    const int32_t x3 = (int32_t)x_base[lane * TMAN_A8_LUT_GROUP + 3];
    tman_a8_build_single_lut_scalar(
        x0,
        x1,
        x2,
        x3,
        block->values + lane * TMAN_A8_LUT_SIZE);
  }
}

inline void tman_a8_store_lut_block(
    int16_t* dst,
    const TmanA8LutBlock& block) {
#ifdef __hexagon__
  if (tman_a8_is_vector_aligned(dst) && tman_a8_is_vector_aligned(block.values)) {
    *reinterpret_cast<HVX_Vector*>(dst + 0) =
        *reinterpret_cast<const HVX_Vector*>(block.values + 0);
    return;
  }
#endif
  std::memcpy(dst, block.values, sizeof(TmanA8LutBlock));
}

inline void tman_a8_load_lut_block_vectorized(
    const int16_t* src,
    TmanA8LutBlock* block) {
  // 这里做的不是数学变换，而是“载入方式”的向量化：
  // - source/destination 都是一整条 128B LUT block；
  // - 在 Hexagon 上直接按 HVX vector line 载入；
  // - 在其它平台保留 memcpy fallback。
  //
  // 这一步让 `LinearA8` 的目标专用 helper 已经开始具备“按向量寄存器 staging LUT block”
  // 的风格，而不再是纯标量逐元素读取。
#ifdef __hexagon__
  if (tman_a8_is_vector_aligned(src) && tman_a8_is_vector_aligned(block->values)) {
    *reinterpret_cast<HVX_Vector*>(block->values) =
        *reinterpret_cast<const HVX_Vector*>(src);
    return;
  }
#endif
  std::memcpy(block->values, src, sizeof(TmanA8LutBlock));
}

inline void tman_a8_zero_int_tile(TmanA8IntTile* tile) {
  std::memset(tile->values, 0, sizeof(tile->values));
}

inline void tman_a8_zero_float_tile(TmanA8FloatTile* tile) {
  std::memset(tile->values, 0, sizeof(tile->values));
}

inline int32_t tman_a8_group_dot_block_scalar(
    const int16_t* block_base,
    const uint8_t* idx_block_base,
    int32_t bits);

inline void tman_a8_accumulate_group_tile_scalar(
    const TmanA8Layout& layout,
    int32_t row_tile_start,
    int32_t row_tile_size,
    int32_t m_tile_start,
    int32_t m_tile_size,
    int32_t group_idx,
    int32_t bits,
    const int16_t* l_ptr,
    const uint8_t* idx_ptr,
    TmanA8IntTile* group_acc_tile) {
  // 通用版本：按 layout 描述的 qgroup_vector_blocks 逐块消费 LUT 和 weight index。
  // 这个版本的价值在于“保留一份语义清晰、配置更泛化的实现”，
  // 方便 target-specific helper 之外的 reference / fallback 继续存在。
  for (int32_t qgb = 0; qgb < layout.qgroup_vector_blocks; ++qgb) {
    const int16_t* lut_blocks[TMAN_A8_TILE_ROWS] = {nullptr};
    for (int32_t row_idx = 0; row_idx < row_tile_size; ++row_idx) {
      lut_blocks[row_idx] = tman_a8_lut_block_ptr(
          layout,
          l_ptr,
          row_tile_start + row_idx,
          group_idx,
          qgb);
    }
    for (int32_t m_idx = 0; m_idx < m_tile_size; ++m_idx) {
      const int32_t out_idx = m_tile_start + m_idx;
      const uint8_t* idx_block = tman_a8_weight_idx_block_ptr(
          layout,
          idx_ptr,
          out_idx,
          group_idx,
          qgb);
      for (int32_t row_idx = 0; row_idx < row_tile_size; ++row_idx) {
        group_acc_tile->values[row_idx][m_idx] += tman_a8_group_dot_block_scalar(
            lut_blocks[row_idx],
            idx_block,
            bits);
      }
    }
  }
}

inline void tman_a8_accumulate_group_tile_w4g128_sym_scalar(
    const TmanA8Layout& layout,
    int32_t row_tile_start,
    int32_t row_tile_size,
    int32_t m_tile_start,
    int32_t m_tile_size,
    int32_t group_idx,
    const int16_t* l_ptr,
    const uint8_t* idx_ptr,
    TmanA8IntTile* group_acc_tile) {
  // 目标专用版本：当前项目真实目标就是 `W4G128 sym`，
  // 因此这里把 group 内 block 数直接固定成 8。
  //
  // 这样做的好处是：
  // - helper 内循环边界更稳定；
  // - 更接近后续直接写成固定展开 / intrinsic 化版本的形态；
  // - 便于和“当前项目不是泛化实现，而是目标专用实现”的工程判断保持一致。
  static_assert(
      TMAN_A8_TARGET_QGROUP_VECTOR_BLOCKS == 8,
      "W4G128 symmetric target is expected to have exactly 8 vector blocks");
#pragma unroll
  for (int32_t qgb = 0; qgb < TMAN_A8_TARGET_QGROUP_VECTOR_BLOCKS; ++qgb) {
    TmanA8LutBlock staged_lut_blocks[TMAN_A8_TILE_ROWS];
    const int16_t* lut_blocks[TMAN_A8_TILE_ROWS] = {nullptr};
    for (int32_t row_idx = 0; row_idx < row_tile_size; ++row_idx) {
      const int16_t* lut_block_src = tman_a8_lut_block_ptr(
          layout, l_ptr, row_tile_start + row_idx, group_idx, qgb);
      // 先把一个完整 LUT block 载入本地 staging buffer。
      // 后续即使把 `tman_a8_group_dot_block_scalar` 替成更激进的 vector-register 消费版本，
      // 这个“先整块载入、再整块消费”的结构也可以继续复用。
      tman_a8_load_lut_block_vectorized(lut_block_src, &staged_lut_blocks[row_idx]);
      lut_blocks[row_idx] = staged_lut_blocks[row_idx].values;
    }
    for (int32_t m_idx = 0; m_idx < m_tile_size; ++m_idx) {
      const int32_t out_idx = m_tile_start + m_idx;
      const uint8_t* idx_block = tman_a8_weight_idx_block_ptr(
          layout,
          idx_ptr,
          out_idx,
          group_idx,
          qgb);
      for (int32_t row_idx = 0; row_idx < row_tile_size; ++row_idx) {
        group_acc_tile->values[row_idx][m_idx] += tman_a8_group_dot_block_scalar(
            lut_blocks[row_idx],
            idx_block,
            TMAN_A8_BITS);
      }
    }
  }
}

inline void tman_a8_apply_scales_to_group_tile_scalar(
    const TmanA8Layout& layout,
    int32_t row_tile_size,
    int32_t m_tile_start,
    int32_t m_tile_size,
    int32_t group_idx,
    const float* s_ptr,
    const TmanA8IntTile& group_acc_tile,
    TmanA8FloatTile* acc_tile) {
  // 当前 A8 路径把：
  //   weight_scale * input_scale / output_scale
  // 预折叠成 effective_scale。
  //
  // 因此 linear helper 在这一层只需要把：
  //   group 内整数和 * group-wise effective_scale
  // 累到 float tile 里即可。
  for (int32_t m_idx = 0; m_idx < m_tile_size; ++m_idx) {
    const int32_t out_idx = m_tile_start + m_idx;
    const float scale = s_ptr[tman_a8_scale_offset(layout, out_idx, group_idx)];
    for (int32_t row_idx = 0; row_idx < row_tile_size; ++row_idx) {
      acc_tile->values[row_idx][m_idx] +=
          (float)group_acc_tile.values[row_idx][m_idx] * scale;
    }
  }
}

inline void tman_a8_store_accum_tile_scalar(
    const TmanA8Layout& layout,
    int32_t row_tile_start,
    int32_t row_tile_size,
    int32_t m_tile_start,
    int32_t m_tile_size,
    const TmanA8FloatTile& acc_tile,
    float* c_ptr) {
  // 把一个完整 `row_tile x m_tile` 累加 tile 回写到 c。
  // 这里保留 tile 级 store，而不是在更内层逐元素写回，
  // 目的是让 helper 结构更接近真正的寄存器/向量 tile 实现。
  for (int32_t row_idx = 0; row_idx < row_tile_size; ++row_idx) {
    float* c_row = c_ptr + ((row_tile_start + row_idx) * layout.gemm_m) + m_tile_start;
    std::memcpy(c_row, acc_tile.values[row_idx], sizeof(float) * m_tile_size);
  }
}

inline int8_t tman_a8_round_clamp_to_int8_scalar(float value) {
  value = std::round(value);
  if (value > 127.0f) {
    value = 127.0f;
  } else if (value < -127.0f) {
    value = -127.0f;
  }
  return (int8_t)value;
}

inline void tman_a8_pack_output_block_scalar(
    const float* c_row,
    int32_t cols_in_block,
    TmanA8OutputBlock* block) {
  // scalar fallback：
  // - 逐元素 round/clamp 到 int8
  // - 对 tail block 做 zero-fill
  //
  // 它保证即使 HVX 路径没有命中，`FinalizeA8` 的语义仍然完全可用。
  for (int32_t col = 0; col < cols_in_block; ++col) {
    block->values[col] = tman_a8_round_clamp_to_int8_scalar(c_row[col]);
  }
  if (cols_in_block < TMAN_A8_OUTPUT_VALUES_PER_VECTOR) {
    std::memset(
        block->values + cols_in_block,
        0,
        sizeof(int8_t) * (TMAN_A8_OUTPUT_VALUES_PER_VECTOR - cols_in_block));
  }
}

inline bool tman_a8_pack_output_block_hvx(
    const float* c_row,
    int32_t cols_in_block,
    TmanA8OutputBlock* block) {
#ifdef __hexagon__
  if (cols_in_block != TMAN_A8_OUTPUT_VALUES_PER_VECTOR || block == nullptr) {
    return false;
  }
  if (!tman_a8_is_vector_aligned(c_row) || !tman_a8_is_vector_aligned(block->values)) {
    return false;
  }

  // 当前这条 HVX 路径做的是“交付前最稳妥的一步 intrinsic 化”：
  // 1. 4 条 HVX vector line 载入 128 个 float accumulator；
  // 2. 向量化 round 到 s32；
  // 3. 向量化 clamp 到 [-127, 127]；
  // 4. 向量化收成 s16；
  // 5. 最后一步 s16 -> s8 仍保留标量压缩。
  //
  // 这样做的原因是：
  // - 已经能尽早暴露 HVX intrinsic / 编译器兼容问题；
  // - 实现风险明显低于“一次性把最后一段 signed int8 pack 也写成全向量版”；
  // - 保留 scalar fallback，便于设备侧 smoke test 优先打通。
  const HVX_Vector clamp_min = Q6_V_vsplat_R(-127);
  const HVX_Vector clamp_max = Q6_V_vsplat_R(127);

  HVX_Vector s32_0 = convert_sf_to_s32_rnd(vmem(c_row + 0));
  HVX_Vector s32_1 = convert_sf_to_s32_rnd(vmem(c_row + 32));
  HVX_Vector s32_2 = convert_sf_to_s32_rnd(vmem(c_row + 64));
  HVX_Vector s32_3 = convert_sf_to_s32_rnd(vmem(c_row + 96));

  s32_0 = Q6_Vw_vmax_VwVw(s32_0, clamp_min);
  s32_1 = Q6_Vw_vmax_VwVw(s32_1, clamp_min);
  s32_2 = Q6_Vw_vmax_VwVw(s32_2, clamp_min);
  s32_3 = Q6_Vw_vmax_VwVw(s32_3, clamp_min);

  s32_0 = Q6_Vw_vmin_VwVw(s32_0, clamp_max);
  s32_1 = Q6_Vw_vmin_VwVw(s32_1, clamp_max);
  s32_2 = Q6_Vw_vmin_VwVw(s32_2, clamp_max);
  s32_3 = Q6_Vw_vmin_VwVw(s32_3, clamp_max);

  const HVX_Vector s16_lo = Q6_Vh_vsat_VwVw(s32_1, s32_0);
  const HVX_Vector s16_hi = Q6_Vh_vsat_VwVw(s32_3, s32_2);

  alignas(128) int16_t packed_s16[TMAN_A8_OUTPUT_VALUES_PER_VECTOR];
  vmem(packed_s16 + 0) = s16_lo;
  vmem(packed_s16 + (TMAN_A8_OUTPUT_VALUES_PER_VECTOR / 2)) = s16_hi;

  for (int32_t i = 0; i < TMAN_A8_OUTPUT_VALUES_PER_VECTOR; ++i) {
    block->values[i] = (int8_t)packed_s16[i];
  }
  return true;
#else
  UNUSED(c_row);
  UNUSED(cols_in_block);
  UNUSED(block);
  return false;
#endif
}

inline void tman_a8_store_output_block(
    int8_t* y_row,
    int32_t cols_in_block,
    const TmanA8OutputBlock& block) {
  // store 阶段按 block 粒度处理：
  // - 若是完整 128B block 且地址满足要求，在 Hexagon 上直接走 HVX vector store；
  // - 否则回退到 memcpy，兼顾 tail block 与非 Hexagon 环境。
#ifdef __hexagon__
  if (cols_in_block == TMAN_A8_OUTPUT_VALUES_PER_VECTOR &&
      ((((uintptr_t)y_row) & (TMAN_A8_OUTPUT_VALUES_PER_VECTOR - 1)) == 0)) {
    *reinterpret_cast<HVX_Vector*>(y_row) =
        *reinterpret_cast<const HVX_Vector*>(block.values);
    return;
  }
#endif
  std::memcpy(y_row, block.values, sizeof(int8_t) * cols_in_block);
}

template <typename TensorType>
inline int32_t tman_a8_num_rows(const TensorType& tensor) {
  auto dims = tensor.dims();
  int32_t rows = 1;
  for (size_t i = 0; i + 1 < dims.size(); ++i) {
    rows *= (int32_t)dims[i];
  }
  return rows;
}

inline void tman_a8_precompute_lut_scalar(
    const TmanA8Layout& layout,
    const int8_t* x_ptr,
    int16_t* l_ptr) {
  // `PrecomputeA8` 的语义是：
  // - 输入已经是 INT8 激活；
  // - 每 4 个激活构造一个 16 项 LUT；
  // - 每 4 个 qgroup 的 LUT 又组成一个 128B block。
  //
  // 因此这里的循环组织本质上是在做：
  //   row -> group -> qgroup_vector_block
  // 三层 block 构造。
  for (int32_t row_tile = 0; row_tile < layout.gemm_n; row_tile += TMAN_A8_TILE_ROWS) {
    const int32_t row_tile_end = MIN(row_tile + TMAN_A8_TILE_ROWS, layout.gemm_n);
    for (int32_t row = row_tile; row < row_tile_end; ++row) {
      const int32_t row_offset = row * layout.gemm_k;
      for (int32_t g = 0; g < layout.num_groups; ++g) {
        for (int32_t qgb = 0; qgb < layout.qgroup_vector_blocks; ++qgb) {
          int16_t* block_dst =
              l_ptr + tman_a8_lut_block_offset(layout, row, g, qgb);
          const int32_t qg_base =
              g * layout.qgroups_per_group + qgb * TMAN_A8_QGROUPS_PER_VECTOR;
          const int8_t* x_block_base =
              x_ptr + row_offset + qg_base * TMAN_A8_LUT_GROUP;
          TmanA8LutBlock lut_block;
          tman_a8_build_lut_block_scalar(x_block_base, &lut_block);
          tman_a8_store_lut_block(block_dst, lut_block);
        }
      }
    }
  }
}

inline int32_t tman_a8_group_dot_block_scalar(
    const int16_t* block_base,
    const uint8_t* idx_block_base,
    int32_t bits) {
  int32_t acc = 0;
  for (int32_t lane = 0; lane < TMAN_A8_QGROUPS_PER_VECTOR; ++lane) {
    acc += tman_a8_group_dot_from_lut(
        block_base + lane * TMAN_A8_LUT_SIZE,
        idx_block_base + lane * bits,
        bits);
  }
  return acc;
}

inline void tman_a8_linear_scalar(
    const TmanA8Layout& layout,
    int32_t bits,
    const int16_t* l_ptr,
    const uint8_t* idx_ptr,
    const float* s_ptr,
    float* c_ptr) {
  // 当前 `LinearA8` 的整体策略是：
  // 1. 以 `row_tile x m_tile` 为外层处理单元；
  // 2. 对每个 weight group，先求出 group 内整数部分和；
  // 3. 再乘上 group-wise effective_scale；
  // 4. 最后把结果写回 float accumulator tile。
  //
  // 这与旧 `W4A16 hvx_tbl(...)` 的“查表 -> group scale 融合 -> 累加”思路是一致的，
  // 但激活侧已经改造成 INT8 + folded scale 的新语义。
  for (int32_t row_tile = 0; row_tile < layout.gemm_n; row_tile += TMAN_A8_TILE_ROWS) {
    const int32_t row_tile_size = MIN(TMAN_A8_TILE_ROWS, layout.gemm_n - row_tile);
    for (int32_t m_tile = 0; m_tile < layout.gemm_m; m_tile += TMAN_A8_TILE_M) {
      const int32_t m_tile_size = MIN(TMAN_A8_TILE_M, layout.gemm_m - m_tile);
      TmanA8FloatTile acc_tile;
      tman_a8_zero_float_tile(&acc_tile);
      for (int32_t g = 0; g < layout.num_groups; ++g) {
        TmanA8IntTile group_acc_tile;
        tman_a8_zero_int_tile(&group_acc_tile);
        if (bits == TMAN_A8_BITS && tman_a8_is_target_layout(layout)) {
          tman_a8_accumulate_group_tile_w4g128_sym_scalar(
              layout,
              row_tile,
              row_tile_size,
              m_tile,
              m_tile_size,
              g,
              l_ptr,
              idx_ptr,
              &group_acc_tile);
        } else {
          tman_a8_accumulate_group_tile_scalar(
              layout,
              row_tile,
              row_tile_size,
              m_tile,
              m_tile_size,
              g,
              bits,
              l_ptr,
              idx_ptr,
              &group_acc_tile);
        }
        tman_a8_apply_scales_to_group_tile_scalar(
            layout,
            row_tile_size,
            m_tile,
            m_tile_size,
            g,
            s_ptr,
            group_acc_tile,
            &acc_tile);
      }
      tman_a8_store_accum_tile_scalar(
          layout,
          row_tile,
          row_tile_size,
          m_tile,
          m_tile_size,
          acc_tile,
          c_ptr);
    }
  }
}

inline void tman_a8_finalize_scalar(
    int32_t gemm_n,
    int32_t gemm_m,
    const float* c_ptr,
    int8_t* y_ptr) {
  // `FinalizeA8` 按 output block 处理：
  // - 优先尝试命中 HVX 路径；
  // - 未命中时回退到 scalar；
  // - 最终统一走 block store。
  //
  // 这样既能保证当前交付版本具备更实质的 HVX 痕迹，
  // 又能保证在非 Hexagon 或 tail block 情况下仍有稳定 fallback。
  for (int32_t n = 0; n < gemm_n; ++n) {
    const float* c_row = c_ptr + (n * gemm_m);
    int8_t* y_row = y_ptr + (n * gemm_m);
    for (int32_t m = 0; m < gemm_m; m += TMAN_A8_OUTPUT_VALUES_PER_VECTOR) {
      const int32_t cols_in_block = MIN(TMAN_A8_OUTPUT_VALUES_PER_VECTOR, gemm_m - m);
      TmanA8OutputBlock packed_block;
      if (!tman_a8_pack_output_block_hvx(c_row + m, cols_in_block, &packed_block)) {
        tman_a8_pack_output_block_scalar(c_row + m, cols_in_block, &packed_block);
      }
      tman_a8_store_output_block(y_row + m, cols_in_block, packed_block);
    }
  }
}

template <typename LType,
          typename XType,
          int ActGroupSize = -1,
          int GroupSize = 0,
          bool ZeroPoint = false,
          int g = 4>
inline typename std::enable_if_t<std::is_same<LType, int16_t>::value && std::is_same<XType, __fp16>::value, int>
hvx_lut_ctor(int32_t GemmK, int32_t GemmN, const XType *x, LType *l, float *ls, float *lb)
{
  UNUSED(GemmN);

  const int32_t Q = GemmK / g;

  const int32_t q_act_group_size = (ActGroupSize < 0) ? (Q / -ActGroupSize) : (ActGroupSize / g);
  const int32_t q_group_size     = (GroupSize == 0) ? Q : (GroupSize / g);

  constexpr int32_t lut_size = 16;
  constexpr float max_int16 = 32767.0f;

  constexpr int32_t VecQ = VLEN / sizeof(XType);

  const HVX_Vector zero_vec = Q6_V_vzero();
  const HVX_Vector ones_vec = Q6_Vh_vsplat_R(0x3C00);  // 1.0f
  const HVX_Vector abs_mask = Q6_Vh_vsplat_R(0x7FFF);

  // lut_bias is stored in fp16 if ZeroPoint is true to avoid conversion during hvx_tbl
  using lb_t = typename std::conditional<ZeroPoint, __fp16, float>::type;
  lb_t *lb_p = reinterpret_cast<lb_t *>(lb);

  XType __attribute__((aligned(VLEN))) tmp_buf[VLEN / sizeof(XType)];

  HVX_Vector lb_val_vec = zero_vec;
  for (int32_t group_q = 0; group_q < Q; group_q += q_act_group_size)
  {
    // Compute LUT scales
    HVX_Vector ls_val_vec = zero_vec;
    for (int32_t q = 0; q < q_act_group_size; q += VecQ)
    {
      const XType *x_base = x + (group_q + q) * g;

      HVX_Vector x0 = vmem(x_base);
      HVX_Vector x1 = vmem(x_base + VecQ);
      HVX_Vector x2 = vmem(x_base + VecQ * 2);
      HVX_Vector x3 = vmem(x_base + VecQ * 3);

      // Transpose (64, 4) -> (4, 64)
      // 16-bit
      HVX_VectorPair x01 = Q6_W_vdeal_VVR(x1, x0, -2);
      HVX_VectorPair x23 = Q6_W_vdeal_VVR(x3, x2, -2);
      // 32-bit
      HVX_VectorPair x02 = Q6_W_vdeal_VVR(Q6_V_lo_W(x23), Q6_V_lo_W(x01), -2);
      HVX_VectorPair x13 = Q6_W_vdeal_VVR(Q6_V_hi_W(x23), Q6_V_hi_W(x01), -2);

      // abs
      // Vhf_vabs_Vhf works on simulator, but not on device (test on 8 gen 3)
      HVX_Vector x0_abs = Q6_V_vand_VV(Q6_V_lo_W(x02), abs_mask);
      HVX_Vector x1_abs = Q6_V_vand_VV(Q6_V_lo_W(x13), abs_mask);
      HVX_Vector x2_abs = Q6_V_vand_VV(Q6_V_hi_W(x02), abs_mask);
      HVX_Vector x3_abs = Q6_V_vand_VV(Q6_V_hi_W(x13), abs_mask);

      // sum
      HVX_Vector x01_abs = Q6_Vqf16_vadd_VhfVhf(x0_abs, x1_abs);
      HVX_Vector x23_abs = Q6_Vqf16_vadd_VhfVhf(x2_abs, x3_abs);
      HVX_Vector sum_abs = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(x01_abs, x23_abs));

      ls_val_vec = Q6_Vhf_vmax_VhfVhf(ls_val_vec, sum_abs);
    }

    // self_max
    for (int32_t i = VLEN / 2; i >= 2; i >>= 1)
    {
      ls_val_vec = Q6_Vhf_vmax_VhfVhf(ls_val_vec, Q6_V_vlalign_VVR(ls_val_vec, zero_vec, i));
    }
    vmem(tmp_buf) = ls_val_vec;
    float ls_val = (float)tmp_buf[VLEN / 2 - 1] / max_int16;
    ls[group_q / q_act_group_size] = ls_val;

    float tls_val = ls_val ? 1.0f / ls_val : 0.0f;
    HVX_Vector tls_val_qf32 = Q6_Vqf32_vadd_VsfVsf(Q6_V_vsplat_R(_fp32_to_bits(tls_val)), zero_vec);

    // Construct LUT
    // qf16 is not enough for accumulation
    for (int32_t q = 0; q < q_act_group_size; q += VecQ)
    {
      const XType *x_base = x + (group_q + q) * g;

      HVX_Vector x0 = vmem(x_base);
      HVX_Vector x1 = vmem(x_base + VecQ);
      HVX_Vector x2 = vmem(x_base + VecQ * 2);
      HVX_Vector x3 = vmem(x_base + VecQ * 3);

      // Transpose (64, 4) -> (4, 64)
      HVX_VectorPair x01 = Q6_W_vdeal_VVR(x1, x0, -2);
      HVX_VectorPair x23 = Q6_W_vdeal_VVR(x3, x2, -2);
      HVX_VectorPair x02 = Q6_W_vdeal_VVR(Q6_V_lo_W(x23), Q6_V_lo_W(x01), -2);
      HVX_VectorPair x13 = Q6_W_vdeal_VVR(Q6_V_hi_W(x23), Q6_V_hi_W(x01), -2);

      // Instead of add zero, multiply by one is more accurate
      HVX_VectorPair x0_qf32 = Q6_Wqf32_vmpy_VhfVhf(Q6_V_lo_W(x02), ones_vec);
      HVX_VectorPair x1_qf32 = Q6_Wqf32_vmpy_VhfVhf(Q6_V_lo_W(x13), ones_vec);
      HVX_VectorPair x2_qf32 = Q6_Wqf32_vmpy_VhfVhf(Q6_V_hi_W(x02), ones_vec);
      HVX_VectorPair x3_qf32 = Q6_Wqf32_vmpy_VhfVhf(Q6_V_hi_W(x13), ones_vec);

      HVX_Vector l_tmp_lo_qf32[lut_size];
      HVX_Vector l_tmp_hi_qf32[lut_size];
#pragma unroll
      for (int32_t i = 1; i < lut_size; i += 2)
      {
        if (i & 0b0010) {
          l_tmp_lo_qf32[i] = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(x0_qf32), Q6_V_lo_W(x1_qf32));
          l_tmp_hi_qf32[i] = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(x0_qf32), Q6_V_hi_W(x1_qf32));
        } else {
          l_tmp_lo_qf32[i] = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(x0_qf32), Q6_V_lo_W(x1_qf32));
          l_tmp_hi_qf32[i] = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_hi_W(x0_qf32), Q6_V_hi_W(x1_qf32));
        }
        if (i & 0b0100) {
          l_tmp_lo_qf32[i] = Q6_Vqf32_vadd_Vqf32Vqf32(l_tmp_lo_qf32[i], Q6_V_lo_W(x2_qf32));
          l_tmp_hi_qf32[i] = Q6_Vqf32_vadd_Vqf32Vqf32(l_tmp_hi_qf32[i], Q6_V_hi_W(x2_qf32));
        } else {
          l_tmp_lo_qf32[i] = Q6_Vqf32_vsub_Vqf32Vqf32(l_tmp_lo_qf32[i], Q6_V_lo_W(x2_qf32));
          l_tmp_hi_qf32[i] = Q6_Vqf32_vsub_Vqf32Vqf32(l_tmp_hi_qf32[i], Q6_V_hi_W(x2_qf32));
        }
        if (i & 0b1000) {
          l_tmp_lo_qf32[i] = Q6_Vqf32_vadd_Vqf32Vqf32(l_tmp_lo_qf32[i], Q6_V_lo_W(x3_qf32));
          l_tmp_hi_qf32[i] = Q6_Vqf32_vadd_Vqf32Vqf32(l_tmp_hi_qf32[i], Q6_V_hi_W(x3_qf32));
        } else {
          l_tmp_lo_qf32[i] = Q6_Vqf32_vsub_Vqf32Vqf32(l_tmp_lo_qf32[i], Q6_V_lo_W(x3_qf32));
          l_tmp_hi_qf32[i] = Q6_Vqf32_vsub_Vqf32Vqf32(l_tmp_hi_qf32[i], Q6_V_hi_W(x3_qf32));
        }
      }

      // Mirror consolidation
#pragma unroll
      for (int32_t i = 0; i < lut_size; i += 2)
      {
        // NOT the sign bit won't work
        l_tmp_lo_qf32[i] = Q6_Vqf32_vsub_Vqf32Vqf32(zero_vec, l_tmp_lo_qf32[lut_size - 1 - i]);
        l_tmp_hi_qf32[i] = Q6_Vqf32_vsub_Vqf32Vqf32(zero_vec, l_tmp_hi_qf32[lut_size - 1 - i]);
      }

      lb_val_vec = Q6_Vqf32_vadd_Vqf32Vqf32(lb_val_vec, l_tmp_lo_qf32[lut_size - 1]);
      lb_val_vec = Q6_Vqf32_vadd_Vqf32Vqf32(lb_val_vec, l_tmp_hi_qf32[lut_size - 1]);

      // Quant LUT
      HVX_Vector l_tmp[lut_size];
#pragma unroll
      for (int32_t i = 0; i < lut_size; i += 1)
      {
        HVX_Vector l_tmp_lo = Q6_Vqf32_vmpy_Vqf32Vqf32(l_tmp_lo_qf32[i], tls_val_qf32);
        HVX_Vector l_tmp_hi = Q6_Vqf32_vmpy_Vqf32Vqf32(l_tmp_hi_qf32[i], tls_val_qf32);
        l_tmp_lo = Q6_Vw_equals_Vsf(Q6_Vsf_equals_Vqf32(l_tmp_lo));
        l_tmp_hi = Q6_Vw_equals_Vsf(Q6_Vsf_equals_Vqf32(l_tmp_hi));
        l_tmp[i] = Q6_Vh_vsat_VwVw(l_tmp_hi, l_tmp_lo);
      }

      // Shuffle and store:
      // only need to shuffle to 32-bit,
      // as even and odd LUTs are interleaved
      HVX_VectorPair l_pa[lut_size / 2];
      HVX_VectorPair l_pb[lut_size / 2];

      // 32-bit, interval=1
#pragma unroll
      for (int32_t i = 0; i < lut_size; i += 2)
      {
        l_pa[i / 2] = Q6_W_vshuff_VVR(l_tmp[i + 1], l_tmp[i], -4);
      }
      // 64-bit, interval=2
#pragma unroll
      for (int32_t i = 0; i < lut_size / 2; i += 2)
      {
        l_pb[i + 0] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[i + 1]), Q6_V_lo_W(l_pa[i + 0]), -8);
        l_pb[i + 1] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[i + 1]), Q6_V_hi_W(l_pa[i + 0]), -8);
      }
      // 128-bit, interval=4
#pragma unroll
      for (int32_t i = 0; i < lut_size / 2; i += 4)
      {
        l_pa[i + 0] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pb[i + 2]), Q6_V_lo_W(l_pb[i + 0]), -16);
        l_pa[i + 1] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pb[i + 2]), Q6_V_hi_W(l_pb[i + 0]), -16);
        l_pa[i + 2] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pb[i + 3]), Q6_V_lo_W(l_pb[i + 1]), -16);
        l_pa[i + 3] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pb[i + 3]), Q6_V_hi_W(l_pb[i + 1]), -16);
      }
      // 256-bit, interval=8
#pragma unroll
      for (int32_t i = 0; i < lut_size / 2; i += 8)
      {
        l_pb[i + 0] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[i + 4]), Q6_V_lo_W(l_pa[i + 0]), -32);
        l_pb[i + 1] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[i + 4]), Q6_V_hi_W(l_pa[i + 0]), -32);
        l_pb[i + 2] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[i + 5]), Q6_V_lo_W(l_pa[i + 1]), -32);
        l_pb[i + 3] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[i + 5]), Q6_V_hi_W(l_pa[i + 1]), -32);
        l_pb[i + 4] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[i + 6]), Q6_V_lo_W(l_pa[i + 2]), -32);
        l_pb[i + 5] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[i + 6]), Q6_V_hi_W(l_pa[i + 2]), -32);
        l_pb[i + 6] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[i + 7]), Q6_V_lo_W(l_pa[i + 3]), -32);
        l_pb[i + 7] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[i + 7]), Q6_V_hi_W(l_pa[i + 3]), -32);
      }
      // write back
      LType *l_base = l + (group_q + q) * lut_size;
#pragma unroll
      for (int32_t i = 0; i < lut_size / 2; i += 1)
      {
        vmem(l_base + (i * 2)     * VLEN / sizeof(LType)) = Q6_V_lo_W(l_pb[i]);
        vmem(l_base + (i * 2 + 1) * VLEN / sizeof(LType)) = Q6_V_hi_W(l_pb[i]);
      }
      if ((q_group_size >= VecQ) && ((group_q + q) % q_group_size == (q_group_size - VecQ)))
      {
        // self_sum
        for (int32_t i = VLEN / 2; i >= 4; i >>= 1)
        {
          lb_val_vec = Q6_Vqf32_vadd_Vqf32Vqf32(lb_val_vec, Q6_V_vlalign_VVR(lb_val_vec, zero_vec, i));
        }
        vmem(tmp_buf) = Q6_Vsf_equals_Vqf32(lb_val_vec);
        lb_p[(group_q + q) / q_group_size] = -((const float *)tmp_buf)[VLEN / 4 - 1] * 0.5f;
        lb_val_vec = zero_vec;
      }
      if (q_group_size < VecQ)
      {
        // self_sum with VecQ/q_group_size groups
        const int32_t sum_len = VLEN / (VecQ / q_group_size);
        for (int32_t i = sum_len / 2; i >= 4; i >>= 1)
        {
          lb_val_vec = Q6_Vqf32_vadd_Vqf32Vqf32(lb_val_vec, Q6_V_vlalign_VVR(lb_val_vec, zero_vec, i));
        }
        vmem(tmp_buf) = Q6_Vsf_equals_Vqf32(lb_val_vec);
        for (int32_t i = VLEN / 4 - 1; i >= 0; i -= sum_len / 4)
        {
          lb_p[(group_q + q) / q_group_size + i / (sum_len / 4)] = -((const float *)tmp_buf)[i] * 0.5f;
        }
        lb_val_vec = zero_vec;
      }
    }  // q_act_group_size
  }

  return 0;
}

// For fine-grained group-wise quantization (GPTQ)
template <typename LType = int16_t,
          typename XType = __fp16,
          typename CType = float,  // use for aggregation
          int ActGroupSize = 256,  // 256 should be enough for int16_t quantization
          int GroupSize = 128,
          bool ZeroPoint = false,
          int Bits = 2,
          int TileK = 256,
          int g = 4,
          bool WeightsInVTCM = false>
inline typename std::enable_if_t<std::is_same<LType, int16_t>::value && std::is_same<XType, __fp16>::value && std::is_same<CType, float>::value && (GroupSize > 0), int>
hvx_tbl(int32_t GemmM, int32_t GemmK, int32_t GemmN, const LType *l, const float *ls, const float *lb, const uint8_t *w, const XType *s, CType *c)
{
  UNUSED(GemmN);

  // Number of elements in a single 4bit pack
  constexpr int8_t mask_4bit = 0b1111;
  constexpr int8_t shift_len = 4;

  const HVX_Vector mask_vec = Q6_Vb_vsplat_R(mask_4bit);
  const HVX_Vector ones_vec = Q6_Vh_vsplat_R(0x3C00);  // 1.0f

  constexpr int32_t lut_size = 16;
  constexpr int32_t lut_bytes = lut_size * sizeof(LType);
  // K, M -> Q, P lookup Q tables with P indices
  // Q = K / g, P = M * Bits
  // x_shape: (Q / TileQ, TileQ / VecQ, VecQ, lut_size) = (Q, lut_size), elem_size = 2 bytes
  // w_shape: (P / TileP, Q / TileQ, TileP / VecP, TileQ / VecQ, VecQ, VecP) indices, elem_size = g / 8 = 0.5 bytes
  // indices of two VecQ are zipped into one Vector
  const int32_t Q = GemmK / g;
  const int32_t P = GemmM * Bits;

  constexpr int32_t q_group_size     = GroupSize / g;
  constexpr int32_t q_act_group_size = ActGroupSize / g;
  // compute block size
  constexpr int32_t cmp_blk_size = MIN(GroupSize / g, ActGroupSize / g);

  constexpr int32_t VecQ = VLEN / lut_bytes;
  constexpr int32_t VecP = VLEN / sizeof(uint8_t);

  constexpr int32_t TileQ = TileK / g;
  // TileP = ThreadP
  const int32_t TileP = P;

  // In practice, for int16_t activation, group size < act group size (not required)
  static_assert((ActGroupSize % GroupSize == 0) || (GroupSize % ActGroupSize) == 0, "ActGroupSize or GroupSize must be divisible by the other");
  // Implies that GroupSize % 16 == 0
  static_assert((cmp_blk_size % VecQ == 0), "cmp_blk_size must be divisible by VecQ");
  static_assert((TileQ % cmp_blk_size == 0), "TileQ must be divisible by cmp_blk_size");  // this requirement is unnecessary. however, i enforce it to simplify the code
  static_assert((Bits <= 4 && Bits >= 2), "2 <= Bits <= 4 is required");  // Bits == 1 also works. Just need to multiply lb by 2

  // Step.1: TABLE TOOKUP
  HVX_Vector lvec_arr[TileQ / VecQ];

  memset(c, 0, sizeof(CType) * TileP);

  for (int32_t tile_q = 0; tile_q < Q; tile_q += TileQ)
  {
#pragma unroll
    for (int32_t vec_q = 0; vec_q < TileQ; vec_q += VecQ)
    {
      lvec_arr[vec_q / VecQ] = vmem(l + (tile_q + vec_q) * lut_size);
    }

    // we can't prefetch scales here, as the size is too large
    // e.g., 64KB for m=4096, group_size=64, TileQ=64, float16, ZeroPoint=true
    // prefetch size = s_l2fetch_p / Bits * sizeof(XType) * TileQ / q_group_size * (1 + ZeroPoint)
    // e.g., 2KB for the same parameters above
    constexpr int32_t s_l2fetch_p    = VecP * Bits;
    constexpr int32_t s_l2fetch_size = (s_l2fetch_p / Bits) * (TileQ / q_group_size) * (1 + ZeroPoint);
    constexpr int32_t s_l2fetch_one  = (s_l2fetch_p / Bits) * (1 + ZeroPoint);

    const uint8_t *w_tile_base = w + tile_q * TileP * g / 8;
    const XType   *s_tile_base = s + (tile_q / q_group_size) * (TileP / Bits) * (1 + ZeroPoint);

    if (!WeightsInVTCM) {
      l2fetch(s_tile_base + s_l2fetch_one, VLEN, VLEN, (s_l2fetch_size - s_l2fetch_one) * sizeof(XType) / VLEN, 0);
    }

    if (tile_q + TileQ < VecQ)
    {
      constexpr int32_t l1cache_line = 64;
      for (int i = (tile_q + TileQ) / q_act_group_size; i < (tile_q + TileQ * 2) / q_act_group_size; i += l1cache_line / sizeof(float))
      {
        Q6_dcfetch_A((void *)(ls + i));
      }
      for (int i = (tile_q + TileQ) / q_group_size; i < (tile_q + TileQ * 2) / q_group_size; i += l1cache_line / sizeof(float))
      {
        Q6_dcfetch_A((void *)(lb + i));
      }
    }

#pragma unroll(Bits)
    for (int32_t vec_p = 0; vec_p < TileP; vec_p += VecP)
    {
      // qf32
      // we should guarantee all these belong to the same bits during preprocessing
      // i.e., VecBits = VecP = VecC * 4
      HVX_Vector c_vec_0 = vmem(c + (vec_p +  0));
      HVX_Vector c_vec_1 = vmem(c + (vec_p + 32));
      HVX_Vector c_vec_2 = vmem(c + (vec_p + 64));
      HVX_Vector c_vec_3 = vmem(c + (vec_p + 96));

      // int32_t
      HVX_VectorPair c_vec_lo;
      HVX_VectorPair c_vec_hi;

      const uint8_t *w_base = w_tile_base + vec_p * TileQ * g / 8;
      const XType   *s_base = s_tile_base + vec_p / (VecP * Bits) * (TileQ / q_group_size) * VecP * (1 + ZeroPoint);
      if (!WeightsInVTCM)
      {
        if (vec_p + VecP < TileP)
        {
          l2fetch(w_base + VecP * TileQ * g / 8, VecP, VecP, TileQ * g / 8, 0);
          if (vec_p % s_l2fetch_p == 0)
          {
            l2fetch(s_base + s_l2fetch_size, VLEN, VLEN, s_l2fetch_size * sizeof(XType) / VLEN, 0);
          }
        }
      }

#pragma unroll
      for (int32_t vec_q = 0; vec_q < TileQ; vec_q += VecQ)
      {
        HVX_Vector w_vec_lo = vmem(w_base + vec_q * VecP * g / 8 + 0);
        HVX_Vector w_vec_hi = vmem(w_base + vec_q * VecP * g / 8 + VLEN);

        HVX_Vector w_vec_lo_bo = Q6_V_vand_VV(w_vec_lo, mask_vec);     // Q = 0
        HVX_Vector w_vec_hi_bo = Q6_V_vand_VV(w_vec_hi, mask_vec);     // Q = 2
        HVX_Vector w_vec_lo_to = Q6_Vh_vasr_VhR(w_vec_lo, shift_len);  // Q = 1
        HVX_Vector w_vec_hi_to = Q6_Vh_vasr_VhR(w_vec_hi, shift_len);  // Q = 3

        // int16_t
        // c_vec_lo_bo_lo: even bytes of w_vec_lo_bo, c_vec_lo_bo_hi: odd bytes of w_vec_lo_bo
        HVX_VectorPair c_vec_lo_bo = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_lo_bo, lvec_arr[vec_q / VecQ], 0);  // Q = 0, even lo
        HVX_VectorPair c_vec_hi_bo = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_hi_bo, lvec_arr[vec_q / VecQ], 1);  // Q = 2, even hi
        HVX_VectorPair c_vec_lo_to = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_lo_to, lvec_arr[vec_q / VecQ], 2);  // Q = 1, odd lo
        HVX_VectorPair c_vec_hi_to = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_hi_to, lvec_arr[vec_q / VecQ], 3);  // Q = 3, odd hi

        // After unroll, the boolean variables should be broadcasted to constexpr and the branches will be expanded
        const bool cmp_blk_head  = (vec_q % cmp_blk_size == 0);
        const bool cmp_blk_tail  = (vec_q % cmp_blk_size == (cmp_blk_size - VecQ));
        const bool q_group_tail  = (vec_q % q_group_size == (q_group_size - VecQ));

        // int32_t
        // c_vec_lo: even bytes of w_vec
        // c_vec_hi:  odd bytes of w_vec
        // TAG0: Here widening add will perform a 2x64 transpose
        if (cmp_blk_head)
        {
          // reset int32_t sum
          c_vec_lo = Q6_Ww_vadd_VhVh(Q6_V_lo_W(c_vec_lo_bo), Q6_V_lo_W(c_vec_hi_bo));
          c_vec_hi = Q6_Ww_vadd_VhVh(Q6_V_hi_W(c_vec_lo_bo), Q6_V_hi_W(c_vec_hi_bo));
        }
        else
        {
          c_vec_lo = Q6_Ww_vaddacc_WwVhVh(c_vec_lo, Q6_V_lo_W(c_vec_lo_bo), Q6_V_lo_W(c_vec_hi_bo));
          c_vec_hi = Q6_Ww_vaddacc_WwVhVh(c_vec_hi, Q6_V_hi_W(c_vec_lo_bo), Q6_V_hi_W(c_vec_hi_bo));
        }
        c_vec_lo = Q6_Ww_vaddacc_WwVhVh(c_vec_lo, Q6_V_lo_W(c_vec_lo_to), Q6_V_lo_W(c_vec_hi_to));
        c_vec_hi = Q6_Ww_vaddacc_WwVhVh(c_vec_hi, Q6_V_hi_W(c_vec_lo_to), Q6_V_hi_W(c_vec_hi_to));

        // qf32
        if (cmp_blk_tail)
        {
          const XType *s_ptr = s_base + (vec_q / q_group_size) * VecP * (1 + ZeroPoint);
          // for fp16 scales, 64 elements per vector
          HVX_Vector s_vec_lo_fp16 = vmem(s_ptr);
          HVX_Vector s_vec_hi_fp16 = vmem(s_ptr + VLEN / sizeof(XType));

          HVX_VectorPair s_vec_lo = Q6_Wqf32_vmpy_VhfVhf(s_vec_lo_fp16, ones_vec);
          HVX_VectorPair s_vec_hi = Q6_Wqf32_vmpy_VhfVhf(s_vec_hi_fp16, ones_vec);

          HVX_Vector ls_vec = Q6_V_vsplat_R(_fp32_to_bits(ls[(tile_q + vec_q) / q_act_group_size]));
          HVX_Vector lb_vec;
          if (ZeroPoint) {
            lb_vec = Q6_Vh_vsplat_R(_fp16_to_bits(reinterpret_cast<const __fp16 *>(lb) + (tile_q + vec_q) / q_group_size));
          } else {
            lb_vec = Q6_V_vsplat_R(_fp32_to_bits(lb[(tile_q + vec_q) / q_group_size]));
          }

          // int32_t -> fp32
          // TODO: consider reordering for understanding: 0, 1, 2, 3 -> lo_W(lo), hi_W(lo), lo_W(hi), hi_W(hi)
          HVX_Vector c_vec_0_sf = Q6_Vsf_equals_Vw(Q6_V_lo_W(c_vec_lo));
          HVX_Vector c_vec_1_sf = Q6_Vsf_equals_Vw(Q6_V_lo_W(c_vec_hi));
          HVX_Vector c_vec_2_sf = Q6_Vsf_equals_Vw(Q6_V_hi_W(c_vec_lo));
          HVX_Vector c_vec_3_sf = Q6_Vsf_equals_Vw(Q6_V_hi_W(c_vec_hi));

          // * ls
          HVX_Vector c_vec_0_qf32 = Q6_Vqf32_vmpy_VsfVsf(c_vec_0_sf, ls_vec);
          HVX_Vector c_vec_1_qf32 = Q6_Vqf32_vmpy_VsfVsf(c_vec_1_sf, ls_vec);
          HVX_Vector c_vec_2_qf32 = Q6_Vqf32_vmpy_VsfVsf(c_vec_2_sf, ls_vec);
          HVX_Vector c_vec_3_qf32 = Q6_Vqf32_vmpy_VsfVsf(c_vec_3_sf, ls_vec);

          // + lb (lb = -1/2 partial sum)
          // only add to b=1, and once for each weights quantization group
          if (q_group_tail && (vec_p % (VecP * Bits) == VecP))
          {
            // (c * ls + lb) * s + z * s * lb * 2
            // = (c * ls + lb + z * lb * 2) * s
            // = (c * ls + (z * 2 + 1) * lb) * s
            if (ZeroPoint)
            {
              HVX_Vector z_vec_lo_fp16 = vmem(s_ptr + VecP);
              HVX_Vector z_vec_hi_fp16 = vmem(s_ptr + VecP + VLEN / sizeof(XType));

              HVX_VectorPair zlb_vec_lo = Q6_Wqf32_vmpy_VhfVhf(z_vec_lo_fp16, lb_vec);
              HVX_VectorPair zlb_vec_hi = Q6_Wqf32_vmpy_VhfVhf(z_vec_hi_fp16, lb_vec);

              c_vec_0_qf32 = Q6_Vqf32_vadd_Vqf32Vqf32(c_vec_0_qf32, Q6_V_lo_W(zlb_vec_lo));
              c_vec_1_qf32 = Q6_Vqf32_vadd_Vqf32Vqf32(c_vec_1_qf32, Q6_V_lo_W(zlb_vec_hi));
              c_vec_2_qf32 = Q6_Vqf32_vadd_Vqf32Vqf32(c_vec_2_qf32, Q6_V_hi_W(zlb_vec_lo));
              c_vec_3_qf32 = Q6_Vqf32_vadd_Vqf32Vqf32(c_vec_3_qf32, Q6_V_hi_W(zlb_vec_hi));
            }
            else
            {
              c_vec_0_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(c_vec_0_qf32, lb_vec);
              c_vec_1_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(c_vec_1_qf32, lb_vec);
              c_vec_2_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(c_vec_2_qf32, lb_vec);
              c_vec_3_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(c_vec_3_qf32, lb_vec);
            }
          }

          // * s
          c_vec_0_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(c_vec_0_qf32, Q6_V_lo_W(s_vec_lo));
          c_vec_1_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(c_vec_1_qf32, Q6_V_lo_W(s_vec_hi));
          c_vec_2_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(c_vec_2_qf32, Q6_V_hi_W(s_vec_lo));
          c_vec_3_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(c_vec_3_qf32, Q6_V_hi_W(s_vec_hi));

          c_vec_0 = Q6_Vqf32_vadd_Vqf32Vqf32(c_vec_0, c_vec_0_qf32);
          c_vec_1 = Q6_Vqf32_vadd_Vqf32Vqf32(c_vec_1, c_vec_1_qf32);
          c_vec_2 = Q6_Vqf32_vadd_Vqf32Vqf32(c_vec_2, c_vec_2_qf32);
          c_vec_3 = Q6_Vqf32_vadd_Vqf32Vqf32(c_vec_3, c_vec_3_qf32);
        }
      }

      vmem(c + (vec_p +  0)) = c_vec_0;
      vmem(c + (vec_p + 32)) = c_vec_1;
      vmem(c + (vec_p + 64)) = c_vec_2;
      vmem(c + (vec_p + 96)) = c_vec_3;
    }
  }

  return 0;
}

// For BitNet
template <typename LType = int16_t,
          typename XType = __fp16,
          typename CType = float,  // use for aggregation
          int ActGroupSize = 256,  // 256 should be enough for int16_t quantization
          int GroupSize = 128,
          bool ZeroPoint = false,
          int Bits = 2,
          int TileK = 256,
          int g = 4,
          bool WeightsInVTCM = false>
inline typename std::enable_if_t<std::is_same<LType, int16_t>::value && std::is_same<XType, __fp16>::value && std::is_same<CType, float>::value && GroupSize == 0, int>
hvx_tbl(int32_t GemmM, int32_t GemmK, int32_t GemmN, const LType *l, const float *ls, const float *lb, const uint8_t *w, const XType *s, CType *c)
{
  UNUSED(GemmN);

  // Number of elements in a single 4bit pack
  constexpr int8_t mask_4bit = 0b1111;
  constexpr int8_t shift_len = 4;

  const HVX_Vector mask_vec = Q6_Vb_vsplat_R(mask_4bit);
  const HVX_Vector ones_vec = Q6_Vh_vsplat_R(0x3C00);  // 1.0f

  constexpr int32_t lut_size = 16;
  constexpr int32_t lut_bytes = lut_size * sizeof(LType);
  // K, M -> Q, P lookup Q tables with P indices
  // Q = K / g, P = M * Bits
  // x_shape: (Q / TileQ, TileQ / VecQ, VecQ, lut_size) = (Q, lut_size), elem_size = 2 bytes
  // w_shape: (P / TileP, Q / TileQ, TileP / VecP, TileQ / VecQ, VecQ, VecP) indices, elem_size = g / 8 = 0.5 bytes
  // indices of two VecQ are zipped into one Vector
  const int32_t Q = GemmK / g;
  const int32_t P = GemmM * Bits;

  constexpr int32_t VecQ = VLEN / lut_bytes;
  constexpr int32_t VecP = VLEN / sizeof(uint8_t);

  constexpr int32_t TileQ = TileK / g;
  // TileP = ThreadP
  const int32_t TileP = P;

  // In practice, for int16_t activation, group size < act group size (not required)
  static_assert((ActGroupSize == -1), "For BitNet model, only per-tensor quantization is supported");
  static_assert(!ZeroPoint, "For BitNet model, the quantization should be symmetric");
  // Implies that GroupSize % 16 == 0
  static_assert((Bits <= 4 && Bits >= 2), "2 <= Bits <= 4 is required");  // Bits == 1 also works. Just need to multiply lb by 2

  // Step.1: TABLE TOOKUP
  HVX_Vector lvec_arr[TileQ / VecQ];

  memset(c, 0, sizeof(CType) * TileP);

  for (int32_t tile_q = 0; tile_q < Q; tile_q += TileQ)
  {
#pragma unroll
    for (int32_t vec_q = 0; vec_q < TileQ; vec_q += VecQ)
    {
      lvec_arr[vec_q / VecQ] = vmem(l + (tile_q + vec_q) * lut_size);
    }

    const uint8_t *w_tile_base = w + tile_q * TileP * g / 8;

#pragma unroll(Bits)
    for (int32_t vec_p = 0; vec_p < TileP; vec_p += VecP)
    {
      // qf32
      // we should guarantee all these belong to the same bits during preprocessing
      // i.e., VecBits = VecP = VecC * 4
      HVX_Vector c_vec_0 = vmem(c + (vec_p +  0));
      HVX_Vector c_vec_1 = vmem(c + (vec_p + 32));
      HVX_Vector c_vec_2 = vmem(c + (vec_p + 64));
      HVX_Vector c_vec_3 = vmem(c + (vec_p + 96));

      // int32_t
      HVX_VectorPair c_vec_lo = Q6_W_vcombine_VV(c_vec_2, c_vec_0);
      HVX_VectorPair c_vec_hi = Q6_W_vcombine_VV(c_vec_3, c_vec_1);

      const uint8_t *w_base = w_tile_base + vec_p * TileQ * g / 8;
      if (!WeightsInVTCM)
      {
        if (vec_p + VecP < TileP)
        {
          l2fetch(w_base + VecP * TileQ * g / 8, VecP, VecP, TileQ * g / 8, 0);
        }
      }

#pragma unroll
      for (int32_t vec_q = 0; vec_q < TileQ; vec_q += VecQ)
      {
        HVX_Vector w_vec_lo = vmem(w_base + vec_q * VecP * g / 8 + 0);
        HVX_Vector w_vec_hi = vmem(w_base + vec_q * VecP * g / 8 + VLEN);

        HVX_Vector w_vec_lo_bo = Q6_V_vand_VV(w_vec_lo, mask_vec);     // Q = 0
        HVX_Vector w_vec_hi_bo = Q6_V_vand_VV(w_vec_hi, mask_vec);     // Q = 2
        HVX_Vector w_vec_lo_to = Q6_Vh_vasr_VhR(w_vec_lo, shift_len);  // Q = 1
        HVX_Vector w_vec_hi_to = Q6_Vh_vasr_VhR(w_vec_hi, shift_len);  // Q = 3

        // int16_t
        // c_vec_lo_bo_lo: even bytes of w_vec_lo_bo, c_vec_lo_bo_hi: odd bytes of w_vec_lo_bo
        HVX_VectorPair c_vec_lo_bo = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_lo_bo, lvec_arr[vec_q / VecQ], 0);  // Q = 0, even lo
        HVX_VectorPair c_vec_hi_bo = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_hi_bo, lvec_arr[vec_q / VecQ], 1);  // Q = 2, even hi
        HVX_VectorPair c_vec_lo_to = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_lo_to, lvec_arr[vec_q / VecQ], 2);  // Q = 1, odd lo
        HVX_VectorPair c_vec_hi_to = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_hi_to, lvec_arr[vec_q / VecQ], 3);  // Q = 3, odd hi

        // int32_t
        // c_vec_lo: even bytes of w_vec
        // c_vec_hi:  odd bytes of w_vec
        // TAG0: Here widening add will perform a 2x64 transpose
        c_vec_lo = Q6_Ww_vaddacc_WwVhVh(c_vec_lo, Q6_V_lo_W(c_vec_lo_bo), Q6_V_lo_W(c_vec_hi_bo));
        c_vec_hi = Q6_Ww_vaddacc_WwVhVh(c_vec_hi, Q6_V_hi_W(c_vec_lo_bo), Q6_V_hi_W(c_vec_hi_bo));
        c_vec_lo = Q6_Ww_vaddacc_WwVhVh(c_vec_lo, Q6_V_lo_W(c_vec_lo_to), Q6_V_lo_W(c_vec_hi_to));
        c_vec_hi = Q6_Ww_vaddacc_WwVhVh(c_vec_hi, Q6_V_hi_W(c_vec_lo_to), Q6_V_hi_W(c_vec_hi_to));
      }

      vmem(c + (vec_p +  0)) = Q6_V_lo_W(c_vec_lo);
      vmem(c + (vec_p + 32)) = Q6_V_lo_W(c_vec_hi);
      vmem(c + (vec_p + 64)) = Q6_V_hi_W(c_vec_lo);
      vmem(c + (vec_p + 96)) = Q6_V_hi_W(c_vec_hi);
    }
  }

  HVX_Vector ls_vec = Q6_V_vsplat_R(_fp32_to_bits(ls[0]));
  HVX_Vector lb_vec = Q6_V_vsplat_R(_fp32_to_bits(lb[0]));
  HVX_Vector s_vec = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vsplat_R(_fp16_to_bits(s)), ones_vec));
#pragma unroll(Bits)
  for (int32_t vec_p = 0; vec_p < TileP; vec_p += VecP)
  {
#pragma unroll
    for (int32_t vec_c = vec_p; vec_c < vec_p + VecP; vec_c += VecP / sizeof(CType))
    {
      HVX_Vector c_vec = vmem(c + vec_c);
      c_vec = Q6_Vsf_equals_Vw(c_vec);
      c_vec = Q6_Vqf32_vmpy_VsfVsf(c_vec, ls_vec);     // * ls
      if (vec_p % (VecP * Bits) == VecP)
      {
        c_vec = Q6_Vqf32_vadd_Vqf32Vsf(c_vec, lb_vec); // + lb
      }
      c_vec = Q6_Vqf32_vmpy_Vqf32Vqf32(c_vec, s_vec);  // * s
      vmem(c + vec_c) = c_vec;
    }
  }

  return 0;
}

template <typename XType = __fp16,
          typename CType = float,  // use for aggregation
          int Bits = 2>
inline typename std::enable_if_t<std::is_same<XType, __fp16>::value && std::is_same<CType, float>::value, int>
hvx_bit_serial(int32_t GemmM, int32_t GemmN, const CType *c, XType *y)
{
  UNUSED(GemmN);

  const int32_t P = GemmM * Bits;

  constexpr int32_t VecP = VLEN / sizeof(uint8_t);
  // TileP = ThreadP
  const int32_t TileP = P;

  static_assert((Bits <= 4 && Bits >= 2), "2 <= Bits <= 4 is required");  // Bits == 1 also works. Just need to multiply lb by 2

  // Step.2: BIT-SERIAL SUM
  const HVX_Vector f0_5_vec = Q6_V_vsplat_R(0x4000007e);  // 0.5f
  const HVX_Vector f2_0_vec = Q6_V_vsplat_R(0x40000080);  // 2.0f
  const HVX_Vector f4_0_vec = Q6_V_vsplat_R(0x40000081);  // 4.0f

  for (int32_t vec_p = 0; vec_p < TileP; vec_p += VecP * Bits)
  {
    // VecP / VecC = 4
    HVX_Vector c_bits[Bits * 4];
#pragma unroll
    for (int32_t b = 0; b < Bits * 4; b++)
    {
      c_bits[b] = vmem(c + (vec_p + b * 32));
    }

#pragma unroll
    for (int32_t i = 0; i < 4; i++)
    {
      c_bits[i] = Q6_Vqf32_vmpy_Vqf32Vqf32(c_bits[i], f0_5_vec);
    }
    if (Bits >= 2)
    {
#pragma unroll
      for (int32_t i = 0; i < 4; i++)
      {
        c_bits[i] = Q6_Vqf32_vadd_Vqf32Vqf32(c_bits[i], c_bits[i + 4]);
      }
    }
    if (Bits >= 3)
    {
#pragma unroll
      for (int32_t i = 0; i < 4; i++)
      {
        c_bits[i + 8] = Q6_Vqf32_vmpy_Vqf32Vqf32(c_bits[i + 8], f2_0_vec);
        c_bits[i] = Q6_Vqf32_vadd_Vqf32Vqf32(c_bits[i], c_bits[i + 8]);
      }
    }
    if (Bits == 4)
    {
#pragma unroll
      for (int32_t i = 0; i < 4; i++)
      {
        c_bits[i + 12] = Q6_Vqf32_vmpy_Vqf32Vqf32(c_bits[i + 12], f4_0_vec);
        c_bits[i] = Q6_Vqf32_vadd_Vqf32Vqf32(c_bits[i], c_bits[i + 12]);
      }
    }
    // TAG1: here narrowing performs a 64x2 transpose to restore TAG0
    HVX_Vector c_bitsum_lo = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(c_bits[2], c_bits[0]));
    HVX_Vector c_bitsum_hi = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(c_bits[3], c_bits[1]));
    vmem(y + vec_p / Bits +  0) = c_bitsum_lo;
    vmem(y + vec_p / Bits + 64) = c_bitsum_hi;
  }

  return 0;
}
