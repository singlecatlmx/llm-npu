#pragma once

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/intrinsics.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <hexagon_types.h>
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

// -----------------------------------------------------------------------------
// Minimal W4A8 scalar helpers
//
// 这部分只服务一个目标：
// - 在 `TMANOpPackage-dev` 中先实现一条语义最干净的 per-tensor activation
//   int8 线性层路径
// - 不追求当前阶段的 HVX 优化
// - 优先保证：
//   1. 数据布局清晰
//   2. 数学语义容易验证
//   3. 后续便于替换成更高性能实现
// -----------------------------------------------------------------------------

constexpr int32_t TMAN_A8_LUT_GROUP = 4;
constexpr int32_t TMAN_A8_LUT_SIZE = 16;
constexpr int32_t TMAN_A8_BITS = 4;
constexpr int32_t TMAN_A8_QGROUPS_PER_VECTOR = 4;
constexpr int32_t TMAN_A8_LUT_VALUES_PER_VECTOR_BLOCK =
    TMAN_A8_QGROUPS_PER_VECTOR * TMAN_A8_LUT_SIZE;
constexpr int32_t TMAN_A8_WEIGHT_IDX_VALUES_PER_VECTOR_BLOCK =
    TMAN_A8_QGROUPS_PER_VECTOR * TMAN_A8_BITS;
constexpr int32_t TMAN_A8_PACKED_BYTES_PER_QGROUP = TMAN_A8_LUT_GROUP / 2;
constexpr int32_t TMAN_A8_ROW_TILE = 4;
constexpr int32_t TMAN_A8_M_TILE = 8;
constexpr int32_t TMAN_A8_M_TILE_HVX = 32;
constexpr int32_t TMAN_A8_TARGET_GROUP_SIZE = 128;
constexpr int32_t TMAN_A8_TARGET_QGROUPS_PER_GROUP = TMAN_A8_TARGET_GROUP_SIZE / TMAN_A8_LUT_GROUP;
constexpr int32_t TMAN_A8_TARGET_QGROUP_VECTOR_BLOCKS =
    TMAN_A8_TARGET_QGROUPS_PER_GROUP / TMAN_A8_QGROUPS_PER_VECTOR;
constexpr int32_t TMAN_A8_OUTPUT_VALUES_PER_VECTOR = 128;

struct TmanA8Layout {
  int32_t gemm_n;
  int32_t gemm_m;
  int32_t gemm_k;
  int32_t num_groups;
  int32_t qgroups_per_group;
  int32_t qgroup_vector_blocks;
  int32_t scale_row_stride;
};

struct alignas(128) TmanA8OutputBlock {
  uint8_t values[TMAN_A8_OUTPUT_VALUES_PER_VECTOR];
};

inline bool tman_a8_is_vector_aligned(const void* ptr) {
  return ((((uintptr_t)ptr) & (128 - 1)) == 0);
}

inline bool tman_a8_populate_layout(
    TmanA8Layout* layout,
    int32_t gemm_n,
    int32_t gemm_m,
    int32_t gemm_k,
    int32_t group_size) {
  if (layout == nullptr || group_size <= 0 || (group_size % TMAN_A8_LUT_GROUP) != 0 ||
      gemm_k <= 0 || (gemm_k % group_size) != 0) {
    return false;
  }

  layout->gemm_n = gemm_n;
  layout->gemm_m = gemm_m;
  layout->gemm_k = gemm_k;
  layout->num_groups = gemm_k / group_size;
  layout->qgroups_per_group = group_size / TMAN_A8_LUT_GROUP;
  if ((layout->qgroups_per_group % TMAN_A8_QGROUPS_PER_VECTOR) != 0) {
    return false;
  }
  layout->qgroup_vector_blocks = layout->qgroups_per_group / TMAN_A8_QGROUPS_PER_VECTOR;
  layout->scale_row_stride = layout->num_groups;
  return true;
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

inline size_t tman_a8_weight_value_offset(
    const TmanA8Layout& layout,
    int32_t out_idx,
    int32_t group_idx,
    int32_t qgroup_in_group,
    int32_t lane) {
  return ((((size_t)out_idx * layout.num_groups) + group_idx) * layout.qgroups_per_group +
          qgroup_in_group) *
          TMAN_A8_LUT_GROUP +
      lane;
}

inline size_t tman_a8_weight_packed_offset(
    const TmanA8Layout& layout,
    int32_t out_idx,
    int32_t group_idx,
    int32_t qgroup_in_group) {
  return ((((size_t)out_idx * layout.num_groups) + group_idx) * layout.qgroups_per_group +
          qgroup_in_group) *
      TMAN_A8_PACKED_BYTES_PER_QGROUP;
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
    lut_base[idx] = static_cast<int16_t>(acc);
  }
}

inline int32_t tman_a8_center_qnn_int8_byte(uint8_t raw) {
  return static_cast<int32_t>(static_cast<int8_t>(raw ^ 0x80u));
}

inline void tman_a8_build_single_lut_from_qnn_int8_scalar(
    const uint8_t* x_base,
    int16_t* lut_base) {
  tman_a8_build_single_lut_scalar(
      tman_a8_center_qnn_int8_byte(x_base[0]),
      tman_a8_center_qnn_int8_byte(x_base[1]),
      tman_a8_center_qnn_int8_byte(x_base[2]),
      tman_a8_center_qnn_int8_byte(x_base[3]),
      lut_base);
}

inline void tman_a8_precompute_lut_scalar(
    const TmanA8Layout& layout,
    const int8_t* x_ptr,
    int16_t* l_ptr) {
  for (int32_t row = 0; row < layout.gemm_n; ++row) {
    const int32_t row_offset = row * layout.gemm_k;
    for (int32_t g = 0; g < layout.num_groups; ++g) {
      for (int32_t qgb = 0; qgb < layout.qgroup_vector_blocks; ++qgb) {
        int16_t* block_dst = l_ptr + tman_a8_lut_block_offset(layout, row, g, qgb);
        for (int32_t lane = 0; lane < TMAN_A8_QGROUPS_PER_VECTOR; ++lane) {
          const int32_t qgroup = qgb * TMAN_A8_QGROUPS_PER_VECTOR + lane;
          const int32_t base = row_offset + g * layout.qgroups_per_group * TMAN_A8_LUT_GROUP +
              qgroup * TMAN_A8_LUT_GROUP;
          tman_a8_build_single_lut_scalar(
              static_cast<int32_t>(x_ptr[base + 0]),
              static_cast<int32_t>(x_ptr[base + 1]),
              static_cast<int32_t>(x_ptr[base + 2]),
              static_cast<int32_t>(x_ptr[base + 3]),
              block_dst + lane * TMAN_A8_LUT_SIZE);
        }
      }
    }
  }
}

inline void tman_a8_precompute_lut_from_qnn_int8_scalar(
    const TmanA8Layout& layout,
    const uint8_t* x_raw,
    int16_t* l_ptr) {
  for (int32_t row = 0; row < layout.gemm_n; ++row) {
    const int32_t row_offset = row * layout.gemm_k;
    for (int32_t g = 0; g < layout.num_groups; ++g) {
      for (int32_t qgb = 0; qgb < layout.qgroup_vector_blocks; ++qgb) {
        int16_t* block_dst = l_ptr + tman_a8_lut_block_offset(layout, row, g, qgb);
        for (int32_t lane = 0; lane < TMAN_A8_QGROUPS_PER_VECTOR; ++lane) {
          const int32_t qgroup = qgb * TMAN_A8_QGROUPS_PER_VECTOR + lane;
          const int32_t base = row_offset + g * layout.qgroups_per_group * TMAN_A8_LUT_GROUP +
              qgroup * TMAN_A8_LUT_GROUP;
          tman_a8_build_single_lut_from_qnn_int8_scalar(
              x_raw + base,
              block_dst + lane * TMAN_A8_LUT_SIZE);
        }
      }
    }
  }
}

inline int32_t tman_a8_group_dot_from_lut(
    const int16_t* lut_base,
    const uint8_t* idx_base,
    int32_t bits) {
  constexpr int32_t zero_bias = 1 << (TMAN_A8_BITS - 1);
  int32_t acc_q = -zero_bias * static_cast<int32_t>(lut_base[TMAN_A8_LUT_SIZE - 1]);
  for (int32_t bit = 0; bit < bits; ++bit) {
    const int32_t lut_idx = static_cast<int32_t>(idx_base[bit]);
    acc_q += (1 << bit) * static_cast<int32_t>(lut_base[lut_idx]);
  }
  return acc_q;
}

inline int32_t tman_a8_group_dot_from_int4(
    const int16_t* lut_base,
    const int8_t* weight_base) {
  uint8_t idx_base[TMAN_A8_BITS] = {0, 0, 0, 0};
  for (int32_t lane = 0; lane < TMAN_A8_LUT_GROUP; ++lane) {
    const int32_t weight = static_cast<int32_t>(weight_base[lane]);
    if (weight < -8 || weight > 7) {
      return 0;
    }
    const uint8_t code = static_cast<uint8_t>(weight + 8);
    for (int32_t bit = 0; bit < TMAN_A8_BITS; ++bit) {
      if ((code >> bit) & 0x1u) {
        idx_base[bit] |= static_cast<uint8_t>(1u << lane);
      }
    }
  }
  return tman_a8_group_dot_from_lut(lut_base, idx_base, TMAN_A8_BITS);
}

inline int32_t tman_a8_group_dot_from_qnn_int4(
    const int16_t* lut_base,
    const uint8_t* weight_raw_base) {
  uint8_t idx_base[TMAN_A8_BITS] = {0, 0, 0, 0};
  for (int32_t lane = 0; lane < TMAN_A8_LUT_GROUP; ++lane) {
    const int32_t weight =
        static_cast<int32_t>(static_cast<int8_t>(weight_raw_base[lane] ^ 0x80u));
    if (weight < -8 || weight > 7) {
      return 0;
    }
    const uint8_t code = static_cast<uint8_t>(weight + 8);
    for (int32_t bit = 0; bit < TMAN_A8_BITS; ++bit) {
      if ((code >> bit) & 0x1u) {
        idx_base[bit] |= static_cast<uint8_t>(1u << lane);
      }
    }
  }
  return tman_a8_group_dot_from_lut(lut_base, idx_base, TMAN_A8_BITS);
}

inline int32_t tman_a8_group_dot_from_packed_int4(
    const int16_t* lut_base,
    const uint8_t* packed_weight_base) {
  uint8_t idx_base[TMAN_A8_BITS] = {0, 0, 0, 0};
  for (int32_t lane = 0; lane < TMAN_A8_LUT_GROUP; ++lane) {
    const uint8_t packed_byte = packed_weight_base[lane / 2];
    const uint8_t code =
        (lane & 1) == 0 ? (packed_byte & 0x0Fu) : ((packed_byte >> 4) & 0x0Fu);
    for (int32_t bit = 0; bit < TMAN_A8_BITS; ++bit) {
      if ((code >> bit) & 0x1u) {
        idx_base[bit] |= static_cast<uint8_t>(1u << lane);
      }
    }
  }
  return tman_a8_group_dot_from_lut(lut_base, idx_base, TMAN_A8_BITS);
}

inline uint32_t tman_a8_packed_byte_to_idx01(uint8_t packed_byte) {
  static const auto table = []() {
    std::array<uint32_t, 256> t{};
    for (int32_t value = 0; value < 256; ++value) {
      uint32_t packed = 0;
      for (int32_t bit = 0; bit < TMAN_A8_BITS; ++bit) {
        const uint8_t lane0 = static_cast<uint8_t>((value >> bit) & 0x1u);
        const uint8_t lane1 = static_cast<uint8_t>(((value >> (bit + 4)) & 0x1u) << 1);
        packed |= static_cast<uint32_t>(lane0 | lane1) << (bit * 8);
      }
      t[static_cast<size_t>(value)] = packed;
    }
    return t;
  }();
  return table[packed_byte];
}

inline int32_t tman_a8_group_dot_from_packed_int4_fast(
    const int16_t* lut_base,
    const uint8_t* packed_weight_base) {
  const uint32_t idx_pack =
      tman_a8_packed_byte_to_idx01(packed_weight_base[0]) |
      (tman_a8_packed_byte_to_idx01(packed_weight_base[1]) << 2);
  int32_t acc_q = -(1 << (TMAN_A8_BITS - 1)) * static_cast<int32_t>(lut_base[TMAN_A8_LUT_SIZE - 1]);
  acc_q += static_cast<int32_t>(lut_base[(idx_pack >> 0) & 0xFF]);
  acc_q += 2 * static_cast<int32_t>(lut_base[(idx_pack >> 8) & 0xFF]);
  acc_q += 4 * static_cast<int32_t>(lut_base[(idx_pack >> 16) & 0xFF]);
  acc_q += 8 * static_cast<int32_t>(lut_base[(idx_pack >> 24) & 0xFF]);
  return acc_q;
}

inline void tman_a8_linear_scalar(
    const TmanA8Layout& layout,
    int32_t bits,
    const int16_t* l_ptr,
    const uint8_t* idx_ptr,
    const float* s_ptr,
    float* c_ptr) {
  for (int32_t row = 0; row < layout.gemm_n; ++row) {
    float* c_row = c_ptr + row * layout.gemm_m;
    for (int32_t m = 0; m < layout.gemm_m; ++m) {
      float acc = 0.0f;
      for (int32_t g = 0; g < layout.num_groups; ++g) {
        int32_t group_acc = 0;
        for (int32_t qgb = 0; qgb < layout.qgroup_vector_blocks; ++qgb) {
          const int16_t* lut_block =
              l_ptr + tman_a8_lut_block_offset(layout, row, g, qgb);
          const uint8_t* idx_block =
              idx_ptr + tman_a8_weight_idx_offset(
                            layout, m, g, qgb * TMAN_A8_QGROUPS_PER_VECTOR, 0);
          for (int32_t lane = 0; lane < TMAN_A8_QGROUPS_PER_VECTOR; ++lane) {
            group_acc += tman_a8_group_dot_from_lut(
                lut_block + lane * TMAN_A8_LUT_SIZE,
                idx_block + lane * bits,
                bits);
          }
        }
        acc += static_cast<float>(group_acc) *
            s_ptr[tman_a8_scale_offset(layout, m, g)];
      }
      c_row[m] = acc;
    }
  }
}

inline void tman_a8_linear_from_int4_scalar(
    const TmanA8Layout& layout,
    const int16_t* l_ptr,
    const int8_t* w_ptr,
    const float* s_ptr,
    float* c_ptr) {
  for (int32_t row = 0; row < layout.gemm_n; ++row) {
    float* c_row = c_ptr + row * layout.gemm_m;
    for (int32_t m = 0; m < layout.gemm_m; ++m) {
      float acc = 0.0f;
      for (int32_t g = 0; g < layout.num_groups; ++g) {
        int32_t group_acc = 0;
        for (int32_t qgb = 0; qgb < layout.qgroup_vector_blocks; ++qgb) {
          const int16_t* lut_block =
              l_ptr + tman_a8_lut_block_offset(layout, row, g, qgb);
          const int32_t qgroup_base = qgb * TMAN_A8_QGROUPS_PER_VECTOR;
          for (int32_t lane = 0; lane < TMAN_A8_QGROUPS_PER_VECTOR; ++lane) {
            const int8_t* weight_group =
                w_ptr + tman_a8_weight_value_offset(layout, m, g, qgroup_base + lane, 0);
            group_acc += tman_a8_group_dot_from_int4(
                lut_block + lane * TMAN_A8_LUT_SIZE,
                weight_group);
          }
        }
        acc += static_cast<float>(group_acc) *
            s_ptr[tman_a8_scale_offset(layout, m, g)];
      }
      c_row[m] = acc;
    }
  }
}

inline void tman_a8_linear_from_qnn_int4_tiled_scalar(
    const TmanA8Layout& layout,
    const int16_t* l_ptr,
    const uint8_t* w_raw_ptr,
    const float* s_ptr,
    float* c_ptr) {
  for (int32_t row_tile = 0; row_tile < layout.gemm_n; row_tile += TMAN_A8_ROW_TILE) {
    const int32_t row_tile_size = MIN(TMAN_A8_ROW_TILE, layout.gemm_n - row_tile);
    for (int32_t m_tile = 0; m_tile < layout.gemm_m; m_tile += TMAN_A8_M_TILE) {
      const int32_t m_tile_size = MIN(TMAN_A8_M_TILE, layout.gemm_m - m_tile);
      float acc_tile[TMAN_A8_ROW_TILE][TMAN_A8_M_TILE] = {};

      for (int32_t g = 0; g < layout.num_groups; ++g) {
        int32_t group_acc_tile[TMAN_A8_ROW_TILE][TMAN_A8_M_TILE] = {};
        for (int32_t qgb = 0; qgb < layout.qgroup_vector_blocks; ++qgb) {
          const int32_t qgroup_base = qgb * TMAN_A8_QGROUPS_PER_VECTOR;
          for (int32_t lane = 0; lane < TMAN_A8_QGROUPS_PER_VECTOR; ++lane) {
            for (int32_t rt = 0; rt < row_tile_size; ++rt) {
              const int16_t* lut_ptr =
                  l_ptr + tman_a8_lut_block_offset(layout, row_tile + rt, g, qgb) +
                  lane * TMAN_A8_LUT_SIZE;
              for (int32_t mt = 0; mt < m_tile_size; ++mt) {
                const uint8_t* w_ptr =
                    w_raw_ptr + tman_a8_weight_value_offset(
                                    layout, m_tile + mt, g, qgroup_base + lane, 0);
                group_acc_tile[rt][mt] += tman_a8_group_dot_from_qnn_int4(lut_ptr, w_ptr);
              }
            }
          }
        }

        for (int32_t rt = 0; rt < row_tile_size; ++rt) {
          for (int32_t mt = 0; mt < m_tile_size; ++mt) {
            acc_tile[rt][mt] += static_cast<float>(group_acc_tile[rt][mt]) *
                s_ptr[tman_a8_scale_offset(layout, m_tile + mt, g)];
          }
        }
      }

      for (int32_t rt = 0; rt < row_tile_size; ++rt) {
        float* c_row = c_ptr + (row_tile + rt) * layout.gemm_m + m_tile;
        for (int32_t mt = 0; mt < m_tile_size; ++mt) {
          c_row[mt] = acc_tile[rt][mt];
        }
      }
    }
  }
}

inline void tman_a8_linear_from_packed_int4_tiled_scalar(
    const TmanA8Layout& layout,
    const int16_t* l_ptr,
    const uint8_t* w_packed_ptr,
    const float* s_ptr,
    float* c_ptr) {
  for (int32_t row_tile = 0; row_tile < layout.gemm_n; row_tile += TMAN_A8_ROW_TILE) {
    const int32_t row_tile_size = MIN(TMAN_A8_ROW_TILE, layout.gemm_n - row_tile);
    for (int32_t m_tile = 0; m_tile < layout.gemm_m; m_tile += TMAN_A8_M_TILE) {
      const int32_t m_tile_size = MIN(TMAN_A8_M_TILE, layout.gemm_m - m_tile);
      float acc_tile[TMAN_A8_ROW_TILE][TMAN_A8_M_TILE] = {};

      for (int32_t g = 0; g < layout.num_groups; ++g) {
        int32_t group_acc_tile[TMAN_A8_ROW_TILE][TMAN_A8_M_TILE] = {};
        for (int32_t qgb = 0; qgb < layout.qgroup_vector_blocks; ++qgb) {
          const int32_t qgroup_base = qgb * TMAN_A8_QGROUPS_PER_VECTOR;
          for (int32_t lane = 0; lane < TMAN_A8_QGROUPS_PER_VECTOR; ++lane) {
            for (int32_t rt = 0; rt < row_tile_size; ++rt) {
              const int16_t* lut_ptr =
                  l_ptr + tman_a8_lut_block_offset(layout, row_tile + rt, g, qgb) +
                  lane * TMAN_A8_LUT_SIZE;
              for (int32_t mt = 0; mt < m_tile_size; ++mt) {
                const uint8_t* w_ptr =
                    w_packed_ptr +
                    tman_a8_weight_packed_offset(layout, m_tile + mt, g, qgroup_base + lane);
                group_acc_tile[rt][mt] +=
                    tman_a8_group_dot_from_packed_int4_fast(lut_ptr, w_ptr);
              }
            }
          }
        }

        for (int32_t rt = 0; rt < row_tile_size; ++rt) {
          for (int32_t mt = 0; mt < m_tile_size; ++mt) {
            acc_tile[rt][mt] += static_cast<float>(group_acc_tile[rt][mt]) *
                s_ptr[tman_a8_scale_offset(layout, m_tile + mt, g)];
          }
        }
      }

      for (int32_t rt = 0; rt < row_tile_size; ++rt) {
        float* c_row = c_ptr + (row_tile + rt) * layout.gemm_m + m_tile;
        for (int32_t mt = 0; mt < m_tile_size; ++mt) {
          c_row[mt] = acc_tile[rt][mt];
        }
      }
    }
  }
}

inline void tman_a8_linear_from_packed_int4_tiled_scalar(
    const TmanA8Layout& layout,
    const int16_t* l_ptr,
    const uint8_t* w_packed_ptr,
    const __fp16* s_ptr,
    float* c_ptr) {
  for (int32_t row_tile = 0; row_tile < layout.gemm_n; row_tile += TMAN_A8_ROW_TILE) {
    const int32_t row_tile_size = MIN(TMAN_A8_ROW_TILE, layout.gemm_n - row_tile);
    for (int32_t m_tile = 0; m_tile < layout.gemm_m; m_tile += TMAN_A8_M_TILE) {
      const int32_t m_tile_size = MIN(TMAN_A8_M_TILE, layout.gemm_m - m_tile);
      float acc_tile[TMAN_A8_ROW_TILE][TMAN_A8_M_TILE] = {};

      for (int32_t g = 0; g < layout.num_groups; ++g) {
        int32_t group_acc_tile[TMAN_A8_ROW_TILE][TMAN_A8_M_TILE] = {};
        for (int32_t qgb = 0; qgb < layout.qgroup_vector_blocks; ++qgb) {
          const int32_t qgroup_base = qgb * TMAN_A8_QGROUPS_PER_VECTOR;
          for (int32_t lane = 0; lane < TMAN_A8_QGROUPS_PER_VECTOR; ++lane) {
            for (int32_t rt = 0; rt < row_tile_size; ++rt) {
              const int16_t* lut_ptr =
                  l_ptr + tman_a8_lut_block_offset(layout, row_tile + rt, g, qgb) +
                  lane * TMAN_A8_LUT_SIZE;
              for (int32_t mt = 0; mt < m_tile_size; ++mt) {
                const uint8_t* w_ptr =
                    w_packed_ptr +
                    tman_a8_weight_packed_offset(layout, m_tile + mt, g, qgroup_base + lane);
                group_acc_tile[rt][mt] +=
                    tman_a8_group_dot_from_packed_int4(lut_ptr, w_ptr);
              }
            }
          }
        }

        for (int32_t rt = 0; rt < row_tile_size; ++rt) {
          for (int32_t mt = 0; mt < m_tile_size; ++mt) {
            acc_tile[rt][mt] += static_cast<float>(group_acc_tile[rt][mt]) *
                static_cast<float>(s_ptr[tman_a8_scale_offset(layout, m_tile + mt, g)]);
          }
        }
      }

      for (int32_t rt = 0; rt < row_tile_size; ++rt) {
        float* c_row = c_ptr + (row_tile + rt) * layout.gemm_m + m_tile;
        for (int32_t mt = 0; mt < m_tile_size; ++mt) {
          c_row[mt] = acc_tile[rt][mt];
        }
      }
    }
  }
}

inline bool tman_a8_is_target_w4g128_layout(const TmanA8Layout& layout) {
  return layout.qgroups_per_group == TMAN_A8_TARGET_QGROUPS_PER_GROUP &&
      layout.qgroup_vector_blocks == TMAN_A8_TARGET_QGROUP_VECTOR_BLOCKS;
}

inline int32_t tman_a8_group_acc_w4g128_fast(
    const int16_t* lut_group_base,
    const uint8_t* weight_group_base) {
  int32_t acc = 0;
#pragma unroll
  for (int32_t qgb = 0; qgb < TMAN_A8_TARGET_QGROUP_VECTOR_BLOCKS; ++qgb) {
    const int16_t* lut_block = lut_group_base + qgb * TMAN_A8_LUT_VALUES_PER_VECTOR_BLOCK;
    const uint8_t* weight_block =
        weight_group_base + qgb * TMAN_A8_QGROUPS_PER_VECTOR * TMAN_A8_PACKED_BYTES_PER_QGROUP;
#pragma unroll
    for (int32_t lane = 0; lane < TMAN_A8_QGROUPS_PER_VECTOR; ++lane) {
      acc += tman_a8_group_dot_from_packed_int4_fast(
          lut_block + lane * TMAN_A8_LUT_SIZE,
          weight_block + lane * TMAN_A8_PACKED_BYTES_PER_QGROUP);
    }
  }
  return acc;
}

inline void tman_a8_accumulate_group_w4g128_4x8_fast(
    const int16_t* const lut_group_base[TMAN_A8_ROW_TILE],
    const uint8_t* const weight_group_base[TMAN_A8_M_TILE],
    int32_t group_acc_tile[TMAN_A8_ROW_TILE][TMAN_A8_M_TILE]) {
#pragma unroll
  for (int32_t qgb = 0; qgb < TMAN_A8_TARGET_QGROUP_VECTOR_BLOCKS; ++qgb) {
#pragma unroll
    for (int32_t lane = 0; lane < TMAN_A8_QGROUPS_PER_VECTOR; ++lane) {
#pragma unroll
      for (int32_t rt = 0; rt < TMAN_A8_ROW_TILE; ++rt) {
        const int16_t* lut_ptr =
            lut_group_base[rt] +
            qgb * TMAN_A8_LUT_VALUES_PER_VECTOR_BLOCK +
            lane * TMAN_A8_LUT_SIZE;
#pragma unroll
        for (int32_t mt = 0; mt < TMAN_A8_M_TILE; ++mt) {
          const uint8_t* w_ptr =
              weight_group_base[mt] +
              qgb * TMAN_A8_QGROUPS_PER_VECTOR * TMAN_A8_PACKED_BYTES_PER_QGROUP +
              lane * TMAN_A8_PACKED_BYTES_PER_QGROUP;
          group_acc_tile[rt][mt] +=
              tman_a8_group_dot_from_packed_int4_fast(lut_ptr, w_ptr);
        }
      }
    }
  }
}

inline void tman_a8_accumulate_group_w4g128_4x32_fast(
    const int16_t* const lut_group_base[TMAN_A8_ROW_TILE],
    const uint8_t* const weight_group_base[TMAN_A8_M_TILE_HVX],
    int32_t group_acc_tile[TMAN_A8_ROW_TILE][TMAN_A8_M_TILE_HVX]) {
  for (int32_t qgb = 0; qgb < TMAN_A8_TARGET_QGROUP_VECTOR_BLOCKS; ++qgb) {
    for (int32_t lane = 0; lane < TMAN_A8_QGROUPS_PER_VECTOR; ++lane) {
      for (int32_t rt = 0; rt < TMAN_A8_ROW_TILE; ++rt) {
        const int16_t* lut_ptr =
            lut_group_base[rt] +
            qgb * TMAN_A8_LUT_VALUES_PER_VECTOR_BLOCK +
            lane * TMAN_A8_LUT_SIZE;
        for (int32_t mt = 0; mt < TMAN_A8_M_TILE_HVX; ++mt) {
          const uint8_t* w_ptr =
              weight_group_base[mt] +
              qgb * TMAN_A8_QGROUPS_PER_VECTOR * TMAN_A8_PACKED_BYTES_PER_QGROUP +
              lane * TMAN_A8_PACKED_BYTES_PER_QGROUP;
          group_acc_tile[rt][mt] +=
              tman_a8_group_dot_from_packed_int4_fast(lut_ptr, w_ptr);
        }
      }
    }
  }
}

inline void tman_a8_apply_scales_w4g128_fp16_hvx_4x32(
    const TmanA8Layout& layout,
    int32_t m_tile_start,
    int32_t group_idx,
    const __fp16* s_ptr,
    const int32_t group_acc_tile[TMAN_A8_ROW_TILE][TMAN_A8_M_TILE_HVX],
    float acc_tile[TMAN_A8_ROW_TILE][TMAN_A8_M_TILE_HVX]) {
  alignas(128) float scale_tile[TMAN_A8_M_TILE_HVX];
#pragma unroll
  for (int32_t mt = 0; mt < TMAN_A8_M_TILE_HVX; ++mt) {
    scale_tile[mt] =
        static_cast<float>(s_ptr[tman_a8_scale_offset(layout, m_tile_start + mt, group_idx)]);
  }
#ifdef __hexagon__
  const HVX_Vector scale_vec = vmem(scale_tile);
#pragma unroll
  for (int32_t rt = 0; rt < TMAN_A8_ROW_TILE; ++rt) {
    HVX_Vector acc_vec = vmem(acc_tile[rt]);
    const HVX_Vector group_acc_vec = vmem(group_acc_tile[rt]);
    const HVX_Vector group_acc_fp = int32_to_fp32(group_acc_vec);
    const HVX_Vector prod = Q6_Vqf32_vmpy_Vqf32Vqf32(group_acc_fp, scale_vec);
    acc_vec = Q6_Vqf32_vadd_Vqf32Vqf32(acc_vec, prod);
    vmem(acc_tile[rt]) = acc_vec;
  }
#else
#pragma unroll
  for (int32_t rt = 0; rt < TMAN_A8_ROW_TILE; ++rt) {
#pragma unroll
    for (int32_t mt = 0; mt < TMAN_A8_M_TILE_HVX; ++mt) {
      acc_tile[rt][mt] +=
          static_cast<float>(group_acc_tile[rt][mt]) * scale_tile[mt];
    }
  }
#endif
}

inline void tman_a8_linear_from_packed_int4_w4g128_fp16_fast(
    const TmanA8Layout& layout,
    const int16_t* l_ptr,
    const uint8_t* w_packed_ptr,
    const __fp16* s_ptr,
    float* c_ptr) {
  for (int32_t row_tile = 0; row_tile < layout.gemm_n; row_tile += TMAN_A8_ROW_TILE) {
    const int32_t row_tile_size = MIN(TMAN_A8_ROW_TILE, layout.gemm_n - row_tile);
    for (int32_t m_tile = 0; m_tile < layout.gemm_m;) {
      const int32_t preferred_tile =
          (layout.gemm_m - m_tile >= TMAN_A8_M_TILE_HVX) ? TMAN_A8_M_TILE_HVX : TMAN_A8_M_TILE;
      const int32_t m_tile_size = MIN(preferred_tile, layout.gemm_m - m_tile);
      alignas(128) float acc_tile_hvx[TMAN_A8_ROW_TILE][TMAN_A8_M_TILE_HVX] = {};
      float(*acc_tile_8)[TMAN_A8_M_TILE] =
          reinterpret_cast<float(*)[TMAN_A8_M_TILE]>(acc_tile_hvx);

      if (row_tile_size == TMAN_A8_ROW_TILE && m_tile_size == TMAN_A8_M_TILE_HVX) {
        for (int32_t g = 0; g < layout.num_groups; ++g) {
          const int16_t* lut_group_base[TMAN_A8_ROW_TILE];
          const uint8_t* weight_group_base[TMAN_A8_M_TILE_HVX];
          alignas(128) int32_t group_acc_tile[TMAN_A8_ROW_TILE][TMAN_A8_M_TILE_HVX] = {};

#pragma unroll
          for (int32_t rt = 0; rt < TMAN_A8_ROW_TILE; ++rt) {
            lut_group_base[rt] =
                l_ptr + tman_a8_lut_block_offset(layout, row_tile + rt, g, 0);
          }
#pragma unroll
          for (int32_t mt = 0; mt < TMAN_A8_M_TILE_HVX; ++mt) {
            weight_group_base[mt] =
                w_packed_ptr + tman_a8_weight_packed_offset(layout, m_tile + mt, g, 0);
          }

          tman_a8_accumulate_group_w4g128_4x32_fast(
              lut_group_base,
              weight_group_base,
              group_acc_tile);
          tman_a8_apply_scales_w4g128_fp16_hvx_4x32(
              layout,
              m_tile,
              g,
              s_ptr,
              group_acc_tile,
              acc_tile_hvx);
        }
      } else if (row_tile_size == TMAN_A8_ROW_TILE && m_tile_size == TMAN_A8_M_TILE) {
        for (int32_t g = 0; g < layout.num_groups; ++g) {
          const int16_t* lut_group_base[TMAN_A8_ROW_TILE];
          const uint8_t* weight_group_base[TMAN_A8_M_TILE];
          float scale_tile[TMAN_A8_M_TILE];
          int32_t group_acc_tile[TMAN_A8_ROW_TILE][TMAN_A8_M_TILE] = {};

#pragma unroll
          for (int32_t rt = 0; rt < TMAN_A8_ROW_TILE; ++rt) {
            lut_group_base[rt] =
                l_ptr + tman_a8_lut_block_offset(layout, row_tile + rt, g, 0);
          }
#pragma unroll
          for (int32_t mt = 0; mt < TMAN_A8_M_TILE; ++mt) {
            weight_group_base[mt] =
                w_packed_ptr + tman_a8_weight_packed_offset(layout, m_tile + mt, g, 0);
            scale_tile[mt] =
                static_cast<float>(s_ptr[tman_a8_scale_offset(layout, m_tile + mt, g)]);
          }

          tman_a8_accumulate_group_w4g128_4x8_fast(
              lut_group_base,
              weight_group_base,
              group_acc_tile);

#pragma unroll
          for (int32_t rt = 0; rt < TMAN_A8_ROW_TILE; ++rt) {
#pragma unroll
            for (int32_t mt = 0; mt < TMAN_A8_M_TILE; ++mt) {
              acc_tile_8[rt][mt] += static_cast<float>(group_acc_tile[rt][mt]) *
                  scale_tile[mt];
            }
          }
        }
      } else {
        for (int32_t g = 0; g < layout.num_groups; ++g) {
          for (int32_t rt = 0; rt < row_tile_size; ++rt) {
            const int16_t* lut_group_base =
                l_ptr + tman_a8_lut_block_offset(layout, row_tile + rt, g, 0);
            for (int32_t mt = 0; mt < m_tile_size; ++mt) {
              const uint8_t* weight_group_base =
                  w_packed_ptr + tman_a8_weight_packed_offset(layout, m_tile + mt, g, 0);
              acc_tile_hvx[rt][mt] +=
                  static_cast<float>(tman_a8_group_acc_w4g128_fast(lut_group_base, weight_group_base)) *
                  static_cast<float>(s_ptr[tman_a8_scale_offset(layout, m_tile + mt, g)]);
            }
          }
        }
      }

      for (int32_t rt = 0; rt < row_tile_size; ++rt) {
        float* c_row = c_ptr + (row_tile + rt) * layout.gemm_m + m_tile;
        for (int32_t mt = 0; mt < m_tile_size; ++mt) {
          c_row[mt] = acc_tile_hvx[rt][mt];
        }
      }
      m_tile += preferred_tile;
    }
  }
}

inline int8_t tman_a8_round_clamp_to_int8_scalar(float value) {
  value = std::round(value);
  if (value > 127.0f) value = 127.0f;
  if (value < -127.0f) value = -127.0f;
  return static_cast<int8_t>(value);
}

inline void tman_a8_finalize_scalar(
    int32_t gemm_n,
    int32_t gemm_m,
    const float* c_ptr,
    int8_t* y_ptr) {
  for (int32_t n = 0; n < gemm_n; ++n) {
    const float* c_row = c_ptr + n * gemm_m;
    int8_t* y_row = y_ptr + n * gemm_m;
    for (int32_t m = 0; m < gemm_m; ++m) {
      y_row[m] = tman_a8_round_clamp_to_int8_scalar(c_row[m]);
    }
  }
}

inline void tman_a8_finalize_to_qnn_int8_scalar(
    int32_t gemm_n,
    int32_t gemm_m,
    const float* c_ptr,
    uint8_t* y_raw_ptr) {
  for (int32_t n = 0; n < gemm_n; ++n) {
    const float* c_row = c_ptr + n * gemm_m;
    uint8_t* y_row = y_raw_ptr + n * gemm_m;
    for (int32_t m = 0; m < gemm_m; ++m) {
      const int8_t y_signed = tman_a8_round_clamp_to_int8_scalar(c_row[m]);
      y_row[m] = static_cast<uint8_t>(static_cast<uint8_t>(y_signed) ^ 0x80u);
    }
  }
}

inline void tman_a8_pack_output_block_scalar(
    const float* c_row,
    int32_t cols_in_block,
    TmanA8OutputBlock* block) {
  for (int32_t col = 0; col < cols_in_block; ++col) {
    const int8_t y_signed = tman_a8_round_clamp_to_int8_scalar(c_row[col]);
    block->values[col] = static_cast<uint8_t>(static_cast<uint8_t>(y_signed) ^ 0x80u);
  }
  if (cols_in_block < TMAN_A8_OUTPUT_VALUES_PER_VECTOR) {
    std::memset(
        block->values + cols_in_block,
        0,
        sizeof(uint8_t) * (TMAN_A8_OUTPUT_VALUES_PER_VECTOR - cols_in_block));
  }
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

inline void tman_a8_finalize_to_qnn_int8_hvx(
    int32_t gemm_n,
    int32_t gemm_m,
    const float* c_ptr,
    uint8_t* y_raw_ptr) {
  for (int32_t n = 0; n < gemm_n; ++n) {
    const float* c_row = c_ptr + n * gemm_m;
    uint8_t* y_row = y_raw_ptr + n * gemm_m;
    for (int32_t m = 0; m < gemm_m; m += TMAN_A8_OUTPUT_VALUES_PER_VECTOR) {
      const int32_t cols_in_block =
          MIN(TMAN_A8_OUTPUT_VALUES_PER_VECTOR, gemm_m - m);
      TmanA8OutputBlock packed_block;
      tman_a8_pack_output_block_scalar(c_row + m, cols_in_block, &packed_block);
#ifdef __hexagon__
      if (cols_in_block == TMAN_A8_OUTPUT_VALUES_PER_VECTOR &&
          tman_a8_is_vector_aligned(y_row + m) &&
          tman_a8_is_vector_aligned(packed_block.values)) {
        vmem(y_row + m) = *reinterpret_cast<const HVX_Vector*>(packed_block.values);
      } else {
        std::memcpy(y_row + m, packed_block.values, sizeof(uint8_t) * cols_in_block);
      }
#else
      std::memcpy(y_row + m, packed_block.values, sizeof(uint8_t) * cols_in_block);
#endif
    }
  }
}

inline int32_t tman_w4a8_packed_lut_size(int32_t gemm_k) {
  return (gemm_k / 4) * 16;
}

inline int32_t tman_w4a8_packed_ls_size(int32_t gemm_k) {
  return gemm_k / 256;
}

inline int32_t tman_w4a8_packed_lb_size(int32_t gemm_k, int32_t group_size) {
  return gemm_k / group_size;
}

inline int32_t tman_w4a8_packed_buffer_bytes(int32_t gemm_k, int32_t group_size) {
  const int32_t l_size = tman_w4a8_packed_lut_size(gemm_k);
  const int32_t ls_size = tman_w4a8_packed_ls_size(gemm_k);
  const int32_t lb_size = tman_w4a8_packed_lb_size(gemm_k, group_size);
  return l_size * static_cast<int32_t>(sizeof(int16_t)) +
      MAX(ls_size * static_cast<int32_t>(sizeof(float)), 128) +
      MAX(lb_size * static_cast<int32_t>(sizeof(float)), 128);
}

inline int8_t tman_w4a8_round_clip_int8(float value) {
  value = std::round(value);
  if (value > 127.0f) value = 127.0f;
  if (value < -127.0f) value = -127.0f;
  return static_cast<int8_t>(value);
}

inline void tman_w4a8_build_signed_lut_from_int8(
    int32_t x0,
    int32_t x1,
    int32_t x2,
    int32_t x3,
    int16_t* lut_base) {
  for (int32_t idx = 0; idx < 16; ++idx) {
    int32_t acc = 0;
    acc += (idx & 0x1) ? x0 : -x0;
    acc += (idx & 0x2) ? x1 : -x1;
    acc += (idx & 0x4) ? x2 : -x2;
    acc += (idx & 0x8) ? x3 : -x3;
    lut_base[idx] = static_cast<int16_t>(acc);
  }
}

inline bool tman_w4a8_precompute_fp16_to_packed_sym_g128(
    int32_t gemm_k,
    int32_t gemm_n,
    const __fp16* x_ptr,
    int16_t* l_ptr,
    float* ls_ptr,
    float* lb_ptr) {
  if (x_ptr == nullptr || l_ptr == nullptr || ls_ptr == nullptr || lb_ptr == nullptr) {
    return false;
  }
  if (gemm_n != 1 || gemm_k <= 0 || (gemm_k % 256) != 0) {
    return false;
  }

  float amax = 0.0f;
  for (int32_t k = 0; k < gemm_k; ++k) {
    const float x_val = static_cast<float>(x_ptr[k]);
    amax = MAX(amax, std::fabs(x_val));
  }
  const float s_a = (amax == 0.0f) ? 1.0f : (amax / 127.0f);
  const float inv_s_a = (s_a == 0.0f) ? 0.0f : (1.0f / s_a);

  const int32_t ls_size = tman_w4a8_packed_ls_size(gemm_k);
  const int32_t num_groups = gemm_k / 128;
  for (int32_t i = 0; i < ls_size; ++i) {
    ls_ptr[i] = s_a;
  }

  int32_t group_sum = 0;
  int32_t group_idx = 0;
  for (int32_t q = 0; q < gemm_k / 4; ++q) {
    const int32_t base = q * 4;
    const int32_t q0 = static_cast<int32_t>(
        tman_w4a8_round_clip_int8(static_cast<float>(x_ptr[base + 0]) * inv_s_a));
    const int32_t q1 = static_cast<int32_t>(
        tman_w4a8_round_clip_int8(static_cast<float>(x_ptr[base + 1]) * inv_s_a));
    const int32_t q2 = static_cast<int32_t>(
        tman_w4a8_round_clip_int8(static_cast<float>(x_ptr[base + 2]) * inv_s_a));
    const int32_t q3 = static_cast<int32_t>(
        tman_w4a8_round_clip_int8(static_cast<float>(x_ptr[base + 3]) * inv_s_a));

    tman_w4a8_build_signed_lut_from_int8(q0, q1, q2, q3, l_ptr + q * 16);
    group_sum += q0 + q1 + q2 + q3;

    if (((base + 4) % 128) == 0) {
      lb_ptr[group_idx++] = -0.5f * static_cast<float>(group_sum) * s_a;
      group_sum = 0;
    }
  }

  return group_idx == num_groups;
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
