#pragma once

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <hexagon_types.h>
#include <limits>
#include <type_traits>

#define UNUSED(x) (void)(x)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

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

// -----------------------------------------------------------------------------
// W4A8 整数主路径的总体约定
//
// 1. 该路径只处理 8bit 量化激活输入，输入 dtype 允许为：
//    - QInt8  : 有符号 int8 存储
//    - QUInt8 : 无符号 uint8 存储，内部会先减去 128 再进入计算
//
// 2. 当前这一版 kernel 不额外接收激活 per-channel scale 张量。
//    约定上游 builder/预处理阶段已经把：
//    - 激活 per-channel 对称 scale
//    - 权重 group scale
//    折叠进了预处理后的 `scales` 张量中。
//
// 3. 为了在不修改旧 scratch buffer 布局的前提下接入新路径：
//    - Precompute 在 ls[0] 写入 sentinel，告诉后续这是 W4A8 路径
//    - Linear 在 c 尾部写 magic，告诉 Finalize 当前 c 中保存的是
//      W4A8 路径产生的浮点输出结果
//
// 4. 这样设计的目标是：
//    - 保留原有 W4A16(fp16 内部路径) 不受影响
//    - 先把 W4A8 的整数主路径在 kernel 侧打通
//    - 后续 builder 只需要按约定把数据喂进来即可
// -----------------------------------------------------------------------------
static constexpr float TMAN_W4A8_SENTINEL = -1.0f;
static constexpr int32_t TMAN_W4A8_MAGIC0 = 0x544D414E;  // "TMAN"
static constexpr int32_t TMAN_W4A8_MAGIC1 = 0x57344138;  // "W4A8"
static constexpr int32_t TMAN_W4A8_SCALE_FRAC_BITS = 20;
static constexpr int32_t TMAN_W4A8_BITS = 4;
static constexpr int32_t TMAN_W4A8_G = 4;
static constexpr int32_t TMAN_W4A8_LUT_SIZE = 16;
static constexpr int32_t TMAN_W4A8_TILE_Q = 64;
static constexpr int32_t TMAN_W4A8_VEC_P = 128;
static constexpr int32_t TMAN_W4A8_VEC_Q = 4;

static inline bool tman_is_q8_dtype(DType dtype)
{
  // 仅把 QInt8 / QUInt8 视为 W4A8 新路径的输入/输出类型。
  return dtype == DType::QInt8 || dtype == DType::QUInt8;
}

static inline int32_t tman_clamp_int32(int64_t value)
{
  if (value > std::numeric_limits<int32_t>::max()) return std::numeric_limits<int32_t>::max();
  if (value < std::numeric_limits<int32_t>::min()) return std::numeric_limits<int32_t>::min();
  return static_cast<int32_t>(value);
}

static inline int16_t tman_clamp_int16(int32_t value)
{
  if (value > std::numeric_limits<int16_t>::max()) return std::numeric_limits<int16_t>::max();
  if (value < std::numeric_limits<int16_t>::min()) return std::numeric_limits<int16_t>::min();
  return static_cast<int16_t>(value);
}

static inline int32_t tman_centered_q8_value(const void *data, DType dtype, int32_t index)
{
  // W4A8 主路径内部统一使用“以 0 为中心”的激活值：
  // - QUInt8: 需要减去存储零点 128
  // - QInt8 : 直接读取
  if (dtype == DType::QUInt8)
  {
    const uint8_t *u8 = reinterpret_cast<const uint8_t *>(data);
    return static_cast<int32_t>(u8[index]) - 128;
  }

  const int8_t *s8 = reinterpret_cast<const int8_t *>(data);
  return static_cast<int32_t>(s8[index]);
}

static inline int32_t tman_q8_storage_zero_point(DType dtype)
{
  // Finalize 回写时需要恢复到存储域。
  return dtype == DType::QUInt8 ? 128 : 0;
}

static inline int32_t tman_apply_fixed_point_scale(int32_t value, float scale)
{
  // 这里使用一个简单的定点乘法近似，把 float scale 转成
  // Q(TMAN_W4A8_SCALE_FRAC_BITS) 定点乘数，再做带符号 rounding shift。
  //
  // 注意：
  // - 这不是最终性能/精度最优实现
  // - 但可以先把“整数主路径”打通，便于后续继续优化
  const int64_t multiplier = static_cast<int64_t>(std::llround(static_cast<double>(scale) * (1LL << TMAN_W4A8_SCALE_FRAC_BITS)));
  const int64_t product = static_cast<int64_t>(value) * multiplier;
  const int64_t rounding = (product >= 0) ? (1LL << (TMAN_W4A8_SCALE_FRAC_BITS - 1))
                                          : -(1LL << (TMAN_W4A8_SCALE_FRAC_BITS - 1));
  return tman_clamp_int32((product + rounding) >> TMAN_W4A8_SCALE_FRAC_BITS);
}

static inline int32_t tman_w4a8_local_p(int32_t bit, int32_t m_in_vec)
{
  // 该函数把 “输出通道内的 bit-plane 索引” 映射回 hvx_preprocess_weights
  // 打包后的 tile_p 局部布局。
  //
  // 这是通过对 Python 侧 hvx_preprocess_weights 的 reshape/transpose
  // 过程做逆向推导得到的，用来确保 kernel 能正确读取已经预处理好的
  // qweight 字节流，而不需要改动 builder 的现有打包格式。
  const int32_t r3 = m_in_vec % 32;
  const int32_t t = m_in_vec / 32;
  const int32_t r2 = t % 2;
  const int32_t r1 = t / 2;
  return bit * TMAN_W4A8_VEC_P + r2 * 64 + r3 * 2 + r1;
}

static inline uint8_t tman_w4a8_read_packed_index(
    const uint8_t *w_bytes,
    int32_t gemm_m,
    int32_t gemm_k,
    int32_t bits,
    int32_t tile_p,
    int32_t m,
    int32_t bit,
    int32_t q)
{
  UNUSED(gemm_m);

  // 读取 builder 预处理后的 qweight 中，某个 (m, bit, q) 位置对应的
  // 4bit 查表索引。
  //
  // 这里返回的是 unsigned 0..15 编码值：
  // - 对称 4bit 权重时，后面会再减去 8 还原到有符号域
  // - 非对称 4bit 权重时，后面会减去 group zero-point
  const int32_t q_tiles = (gemm_k / TMAN_W4A8_G) / TMAN_W4A8_TILE_Q;
  const int32_t outputs_per_tile = tile_p / bits;
  const int32_t m_tile = m / outputs_per_tile;
  const int32_t m_in_tile = m % outputs_per_tile;
  const int32_t out_block = m_in_tile / TMAN_W4A8_VEC_P;
  const int32_t m_in_vec = m_in_tile % TMAN_W4A8_VEC_P;

  const int32_t p_in_tile = out_block * bits * TMAN_W4A8_VEC_P + tman_w4a8_local_p(bit, m_in_vec);
  const int32_t tpv = p_in_tile / TMAN_W4A8_VEC_P;
  const int32_t pv = p_in_tile % TMAN_W4A8_VEC_P;

  const int32_t q_tile = q / TMAN_W4A8_TILE_Q;
  const int32_t q_in_tile = q % TMAN_W4A8_TILE_Q;
  const int32_t tqv = q_in_tile / TMAN_W4A8_VEC_Q;
  const int32_t pair = (q_in_tile % TMAN_W4A8_VEC_Q) / 2;
  const int32_t half = (q_in_tile % TMAN_W4A8_VEC_Q) % 2;

  const size_t idx = (((((static_cast<size_t>(m_tile) * q_tiles + q_tile) * (tile_p / TMAN_W4A8_VEC_P) + tpv)
                      * (TMAN_W4A8_TILE_Q / TMAN_W4A8_VEC_Q) + tqv)
                      * (TMAN_W4A8_VEC_Q / 2) + pair)
                      * TMAN_W4A8_VEC_P) + pv;

  const uint8_t packed = w_bytes[idx];
  return (packed >> (half * TMAN_W4A8_G)) & 0x0F;
}

static inline float tman_w4a8_read_group_scale(
    const __fp16 *scales,
    int32_t gemm_k,
    int32_t group_size,
    int32_t bits,
    int32_t tile_p,
    int32_t m,
    int32_t q)
{
  // 读取某个 (m, q_group) 对应的 group scale。
  //
  // 这里读取的是 builder 预处理后的“权重 group scale”。
  // 对当前 W4A8 浮点输出路径，不再额外折叠 output_scale，
  // 因此这里读出的就是纯粹的 weight_group_scale。
  const int32_t q_tiles = (gemm_k / TMAN_W4A8_G) / TMAN_W4A8_TILE_Q;
  const int32_t q_group_size = group_size / TMAN_W4A8_G;
  const int32_t outputs_per_tile = tile_p / bits;
  const int32_t out_blocks_per_tile = outputs_per_tile / TMAN_W4A8_VEC_P;
  const int32_t groups_per_tile = TMAN_W4A8_TILE_Q / q_group_size;

  const int32_t m_tile = m / outputs_per_tile;
  const int32_t m_in_tile = m % outputs_per_tile;
  const int32_t out_block = m_in_tile / TMAN_W4A8_VEC_P;
  const int32_t m_in_vec = m_in_tile % TMAN_W4A8_VEC_P;

  const int32_t q_tile = q / TMAN_W4A8_TILE_Q;
  const int32_t group_in_tile = (q % TMAN_W4A8_TILE_Q) / q_group_size;

  const size_t idx = (((((static_cast<size_t>(m_tile) * q_tiles + q_tile) * out_blocks_per_tile + out_block)
                      * groups_per_tile + group_in_tile)
                      * TMAN_W4A8_VEC_P) + m_in_vec);
  return static_cast<float>(scales[idx]);
}

static inline int32_t tman_w4a8_read_zero_unsigned(
    const __fp16 *scales,
    int32_t gemm_k,
    int32_t group_size,
    int32_t bits,
    int32_t tile_p,
    int32_t m,
    int32_t q)
{
  // 非对称 GPTQ 权重会把 zero-point 也一起打包进 scales 张量。
  // hvx_preprocess_weights 对 zero 做了 (zero * 2 + 1) 的编码，
  // 这里需要做逆变换还原出 unsigned zero，再转到实际计算所需的域。
  const int32_t q_tiles = (gemm_k / TMAN_W4A8_G) / TMAN_W4A8_TILE_Q;
  const int32_t q_group_size = group_size / TMAN_W4A8_G;
  const int32_t outputs_per_tile = tile_p / bits;
  const int32_t out_blocks_per_tile = outputs_per_tile / TMAN_W4A8_VEC_P;
  const int32_t groups_per_tile = TMAN_W4A8_TILE_Q / q_group_size;

  const int32_t m_tile = m / outputs_per_tile;
  const int32_t m_in_tile = m % outputs_per_tile;
  const int32_t out_block = m_in_tile / TMAN_W4A8_VEC_P;
  const int32_t m_in_vec = m_in_tile % TMAN_W4A8_VEC_P;

  const int32_t q_tile = q / TMAN_W4A8_TILE_Q;
  const int32_t group_in_tile = (q % TMAN_W4A8_TILE_Q) / q_group_size;

  const size_t idx = ((((((static_cast<size_t>(m_tile) * q_tiles + q_tile) * out_blocks_per_tile + out_block)
                      * groups_per_tile + group_in_tile)
                      * 2 + 1)
                      * TMAN_W4A8_VEC_P) + m_in_vec);
  const float shifted_zero = (static_cast<float>(scales[idx]) - 1.0f) * 0.5f;
  return static_cast<int32_t>(std::lrintf(shifted_zero + static_cast<float>(1 << (bits - 1))));
}

static inline int tman_w4a8_precompute_from_q8(
    int32_t gemm_k,
    int32_t group_size,
    DType x_dtype,
    const void *x,
    const float *input_scale,
    int16_t *l,
    float *ls,
    float *lb)
{
  if (!tman_is_q8_dtype(x_dtype) || gemm_k % TMAN_W4A8_G != 0 || group_size <= 0) return -1;

  // 对于 W4A8 路径，Precompute 的职责改为：
  // - 从 q8 激活和显式输入 per-channel scale 构造 LUT
  // - 每个 weight group 单独生成一份 ls，用于把带 scale 的 LUT 值压到 int16
  // - 不使用旧 W4A16 路径里的 lb 语义，仅借用 lb[0] 存 sentinel
  //
  // LUT 的定义与旧路径保持一致：
  // 每 4 个激活值组成一组，对应 16 种 +/- 组合。
  const int32_t q_count = gemm_k / TMAN_W4A8_G;
  const int32_t q_group_size = group_size / TMAN_W4A8_G;
  const int32_t group_count = q_count / q_group_size;

  for (int32_t group_idx = 0; group_idx < group_count; ++group_idx)
  {
    const int32_t q_begin = group_idx * q_group_size;

    float max_abs = 0.0f;
    for (int32_t q = q_begin; q < q_begin + q_group_size; ++q)
    {
      float xv[TMAN_W4A8_G];
      for (int32_t ig = 0; ig < TMAN_W4A8_G; ++ig)
      {
        const int32_t ch = q * TMAN_W4A8_G + ig;
        xv[ig] = static_cast<float>(tman_centered_q8_value(x, x_dtype, ch)) * input_scale[ch];
      }

      for (int32_t idx = 0; idx < TMAN_W4A8_LUT_SIZE; ++idx)
      {
        float value = 0.0f;
        for (int32_t ig = 0; ig < TMAN_W4A8_G; ++ig)
        {
          value += ((idx >> ig) & 0x1) ? xv[ig] : -xv[ig];
        }
        max_abs = std::max(max_abs, std::fabs(value));
      }
    }

    const float ls_value = max_abs > 0.0f
        ? max_abs / static_cast<float>(std::numeric_limits<int16_t>::max())
        : 1.0f;
    ls[group_idx] = ls_value;
    const float inv_ls = (max_abs > 0.0f) ? (1.0f / ls_value) : 0.0f;

    for (int32_t q = q_begin; q < q_begin + q_group_size; ++q)
    {
      float xv[TMAN_W4A8_G];
      for (int32_t ig = 0; ig < TMAN_W4A8_G; ++ig)
      {
        const int32_t ch = q * TMAN_W4A8_G + ig;
        xv[ig] = static_cast<float>(tman_centered_q8_value(x, x_dtype, ch)) * input_scale[ch];
      }

      int16_t *lut = l + q * TMAN_W4A8_LUT_SIZE;
      for (int32_t idx = 0; idx < TMAN_W4A8_LUT_SIZE; ++idx)
      {
        float value = 0.0f;
        for (int32_t ig = 0; ig < TMAN_W4A8_G; ++ig)
        {
          value += ((idx >> ig) & 0x1) ? xv[ig] : -xv[ig];
        }
        const int32_t q_value = static_cast<int32_t>(std::lrintf(value * inv_ls));
        lut[idx] = tman_clamp_int16(q_value);
      }
    }
  }

  // W4A8 路径借用 lb[0] 作为 sentinel，后续 Linear 通过它判断是否走
  // 新的整数主路径。其余 lb 位置不使用。
  lb[0] = TMAN_W4A8_SENTINEL;
  return 0;
}

static inline int tman_w4a8_linear_from_lut(
    int32_t gemm_m,
    int32_t gemm_k,
    int32_t group_size,
    bool zero_point,
    int32_t bits,
    int32_t tile_p,
    const int16_t *l,
    const float *ls,
    const float *lb,
    const uint8_t *w_bytes,
    const __fp16 *scales,
    int32_t *c,
    int32_t c_elements)
{
  if (bits != TMAN_W4A8_BITS || group_size <= 0) return -1;
  if (gemm_k % TMAN_W4A8_G != 0 || (gemm_k / TMAN_W4A8_G) % TMAN_W4A8_TILE_Q != 0) return -1;
  if (group_size % TMAN_W4A8_G != 0 || TMAN_W4A8_TILE_Q % (group_size / TMAN_W4A8_G) != 0) return -1;
  if (tile_p <= 0 || (tile_p % (bits * TMAN_W4A8_VEC_P)) != 0) return -1;
  if (c_elements < gemm_m + 2) return -1;
  if (lb[0] != TMAN_W4A8_SENTINEL) return -1;

  // 该函数是新的 W4A8 激活路径核心：
  // - 输入：int16 LUT + 预处理后的 4bit weight + group scale
  // - 中间：权重点积仍以整数方式构造，但最终按浮点 scale 恢复
  // - 输出：浮点域结果，存放到 c 的前 gemm_m 个 float 位置
  //
  // 这样做的目的不是追求最终性能最优，而是先把“q8 输入 + 浮点输出”
  // 这条更稳的 Transformer 集成路径打通，避免后续 add/rms_norm/cat
  // 等子图继续承受 q8 中间张量的兼容性压力。
  const int32_t q_group_size = group_size / TMAN_W4A8_G;
  const int32_t q_count = gemm_k / TMAN_W4A8_G;
  const int32_t group_count = q_count / q_group_size;

  std::fill(c, c + c_elements, 0);

  float *c_fp = reinterpret_cast<float *>(c);
  for (int32_t m = 0; m < gemm_m; ++m)
  {
    float accum = 0.0f;
    for (int32_t group_idx = 0; group_idx < group_count; ++group_idx)
    {
      // 每个 group 都单独读取一份折叠后的 scale。
      const int32_t q_begin = group_idx * q_group_size;
      const float act_group_scale = ls[group_idx];
      const float weight_group_scale = tman_w4a8_read_group_scale(scales, gemm_k, group_size, bits, tile_p, m, q_begin);
      const int32_t zero_unsigned = zero_point
          ? tman_w4a8_read_zero_unsigned(scales, gemm_k, group_size, bits, tile_p, m, q_begin)
          : (1 << (bits - 1));

      int32_t group_dot = 0;
      for (int32_t q = q_begin; q < q_begin + q_group_size; ++q)
      {
        const int16_t *lut = l + q * TMAN_W4A8_LUT_SIZE;
        const int32_t sum_x = static_cast<int32_t>(lut[TMAN_W4A8_LUT_SIZE - 1]);

        // 关键恒等式：
        // 对 unsigned 4bit 索引 idx 和 LUT 末项 sum_x，有
        //   dot_unsigned(idx) = (lut[idx] + sum_x) / 2
        //
        // 它对应的是“把每一位映射成 0/1 后与 4 个激活求点积”。
        // 这样可以直接用现有 16 项 LUT 恢复每个 bit-plane 的无符号点积。
        int32_t dot_unsigned = 0;
        for (int32_t bit = 0; bit < bits; ++bit)
        {
          const uint8_t idx = tman_w4a8_read_packed_index(w_bytes, gemm_m, gemm_k, bits, tile_p, m, bit, q);
          const int32_t bit_dot = (static_cast<int32_t>(lut[idx]) + sum_x) >> 1;
          dot_unsigned += bit_dot << bit;
        }

        // 把 unsigned 编码恢复为真实权值域：
        // - 对称 4bit: zero_unsigned = 8
        // - 非对称 4bit: zero_unsigned = group zero-point
        //
        // 由于真实权值 = unsigned_value - zero_unsigned，
        // 所以需要额外减去 zero_unsigned * sum_x。
        dot_unsigned -= zero_unsigned * sum_x;

        group_dot += dot_unsigned;
      }

      // 先恢复 LUT 的真实数值尺度，再乘以 builder 侧预处理好的
      // weight_group_scale，直接在浮点域累加。
      accum += static_cast<float>(group_dot) * act_group_scale * weight_group_scale;
    }

    c_fp[m] = accum;
  }

  // 在 c 末尾写 magic，供 Finalize 判断当前 c 是否来自 W4A8 浮点输出路径。
  c[c_elements - 2] = TMAN_W4A8_MAGIC0;
  c[c_elements - 1] = TMAN_W4A8_MAGIC1;
  return 0;
}

static inline bool tman_w4a8_has_magic(const int32_t *c, int32_t c_elements)
{
  // 简单的轻量标记，不引入新的 scratch tensor。
  return c_elements >= 2 && c[c_elements - 2] == TMAN_W4A8_MAGIC0 && c[c_elements - 1] == TMAN_W4A8_MAGIC1;
}

static inline int tman_w4a8_finalize_to_fp16(
    int32_t gemm_m,
    const int32_t *c,
    int32_t c_elements,
    void *y)
{
  if (!tman_w4a8_has_magic(c, c_elements)) return -1;
  const float *c_fp = reinterpret_cast<const float *>(c);
  __fp16 *y_fp16 = reinterpret_cast<__fp16 *>(y);
  for (int32_t m = 0; m < gemm_m; ++m)
  {
    y_fp16[m] = static_cast<__fp16>(c_fp[m]);
  }
  return 0;
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
