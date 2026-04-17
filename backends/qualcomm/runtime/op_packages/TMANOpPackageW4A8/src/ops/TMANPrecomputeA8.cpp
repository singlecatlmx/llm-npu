//==============================================================================
// W4A8 TMAN precompute path aligned with the FP16 TMAN pipeline
// - input  : fp16 activation
// - output : packed [lut:int16][ls:float][lb:float]
// - scope  : W4G128 symmetric, decoder-style N=1
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

#include <vector>

#include "hvx_funcs.h"

BEGIN_PKG_OP_DEFINITION(PKG_TMANPrecomputeA8);

static Qnn_Scalar_t sg_opDefaultGroup_SizeScalar = {
    .dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
    .int32Value = 128};
static Qnn_Param_t sg_opDefaultGroup_Size = {
    .paramType = QNN_PARAMTYPE_SCALAR,
    .scalarParam = sg_opDefaultGroup_SizeScalar};
static Qnn_Scalar_t sg_opDefaultBitsScalar = {
    .dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
    .int32Value = 4};
static Qnn_Param_t sg_opDefaultBits = {
    .paramType = QNN_PARAMTYPE_SCALAR,
    .scalarParam = sg_opDefaultBitsScalar};
static Qnn_Scalar_t sg_opDefaultSymmetricScalar = {
    .dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
    .int32Value = 1};
static Qnn_Param_t sg_opDefaultSymmetric = {
    .paramType = QNN_PARAMTYPE_SCALAR,
    .scalarParam = sg_opDefaultSymmetricScalar};

template <typename TensorType>
GraphStatus tmanprecomputea8Impl(
    TensorType& l,
    const TensorType& x,
    const TensorType& input_scale,
    const Int32Tensor& t_group_size,
    const Int32Tensor& t_bits,
    const Int32Tensor& t_symmetric);

static float tmanprecomputea8CostFunc(const Op* op);

DEF_PACKAGE_OP((tmanprecomputea8Impl<Tensor>), "TMANPrecomputeA8")

DEF_TENSOR_PROPERTIES(
    Op("TMANPrecomputeA8", "x", "input_scale", "group_size", "bits", "symmetric"),
    Flat("*", "x", "input_scale"),
    MainMemory("group_size", "bits", "symmetric"),
    Tcm("*", "x"))

DEF_PACKAGE_PARAM_ORDER(
    "TMANPrecomputeA8",
    "group_size",
    false,
    &sg_opDefaultGroup_Size,
    "bits",
    false,
    &sg_opDefaultBits,
    "symmetric",
    false,
    &sg_opDefaultSymmetric)

template <typename TensorType>
GraphStatus tmanprecomputea8Impl(
    TensorType& l,
    const TensorType& x,
    const TensorType& input_scale,
    const Int32Tensor& t_group_size,
    const Int32Tensor& t_bits,
    const Int32Tensor& t_symmetric) {
  using LType = int16_t;
  using XType = __fp16;

  // 这一版的正式 W4A8 路径先只收敛到 README 明确推荐的最小可用配置：
  // W4G128 + symmetric + 解码阶段 N=1。
  // 这样可以最大化复用旧 TMAN 的 LUT/bit-serial 主干，减少功能回归风险。
  const int32_t group_size = ((const int32_t*)t_group_size.raw_data_const())[0];
  const int32_t bits = ((const int32_t*)t_bits.raw_data_const())[0];
  const bool symmetric = ((const int32_t*)t_symmetric.raw_data_const())[0] != 0;
  if (group_size != 128 || bits != 4 || !symmetric) {
    return GraphStatus::ErrorDimensions;
  }

  auto x_dims = x.dims();
  auto l_dims = l.dims();
  auto input_scale_dims = input_scale.dims();
  if (x_dims.size() != 4 || l_dims.size() != 4 || input_scale_dims.size() != 4) {
    return GraphStatus::ErrorDimensions;
  }

  const int32_t gemm_n = static_cast<int32_t>(x_dims[2]);
  const int32_t gemm_k = static_cast<int32_t>(x_dims[3]);
  if (gemm_n != 1 || gemm_k <= 0 || (gemm_k % group_size) != 0 ||
      (gemm_k % 256) != 0) {
    return GraphStatus::ErrorDimensions;
  }

  // PrecomputeA8 的输出仍然保持旧 TMAN 的单一 packed buffer 语义：
  // [lut:int16][ls:float][lb:float]
  // 这样下游 LinearA8 可以直接复用旧 hvx_tbl 的寻址方式。
  const int32_t expected_l_bytes = tman_w4a8_packed_buffer_bytes(gemm_k, group_size);
  if (static_cast<int32_t>(l_dims[0]) != 1 || static_cast<int32_t>(l_dims[1]) != 1 ||
      static_cast<int32_t>(l_dims[2]) != 1 ||
      static_cast<int32_t>(l_dims[3]) != expected_l_bytes) {
    return GraphStatus::ErrorDimensions;
  }

  const int32_t l_size = tman_w4a8_packed_lut_size(gemm_k);
  const int32_t ls_size = tman_w4a8_packed_ls_size(gemm_k);

  const XType* x_ptr = (const XType*)x.raw_data_const();
  const float* input_scale_ptr = (const float*)input_scale.raw_data_const();
  LType* l_ptr = (LType*)l.raw_data();
  float* ls_ptr = (float*)(l_ptr + l_size);
  float* lb_ptr = ls_ptr + MAX(ls_size, 128 / sizeof(float));
  if (x_ptr == nullptr || input_scale_ptr == nullptr || l_ptr == nullptr || ls_ptr == nullptr ||
      lb_ptr == nullptr) {
    return GraphStatus::ErrorFatal;
  }

  // 正式路径里不再在 runtime 内部动态估计 activation scale，
  // 而是严格使用外部导入的 per-tensor input_scale。
  // 这样模型导出侧和 runtime 侧对激活量化语义是一致的，也便于离线标定。
  const float s_a = input_scale_ptr[0];
  if (!(s_a > 0.0f)) {
    return GraphStatus::ErrorDimensions;
  }
  if (static_cast<int32_t>(input_scale_dims[0]) != 1 || static_cast<int32_t>(input_scale_dims[1]) != 1 ||
      static_cast<int32_t>(input_scale_dims[2]) != 1 || static_cast<int32_t>(input_scale_dims[3]) != 1) {
    return GraphStatus::ErrorDimensions;
  }
  // 恢复回与 hvx_tbl(...) 契约一致的 hvx_lut_ctor 路线：
  // 1. 仍严格使用外部 input_scale；
  // 2. 先按 input_scale 量化到 A8；
  // 3. 再反量化回 fp16；
  // 4. 由 hvx_lut_ctor 统一生成 [lut][ls][lb]。
  //
  // 这样可以最大程度保证 PrecomputeA8 的输出与 LinearA8/hvx_tbl 的
  // 既有消费语义一致。
  const float inv_s_a = (s_a == 0.0f) ? 0.0f : (1.0f / s_a);
  std::vector<XType> x_quantized(
      static_cast<size_t>(gemm_k),
      static_cast<XType>(0.0f));
  for (int32_t k = 0; k < gemm_k; ++k) {
    const int8_t q_a =
        tman_w4a8_round_clip_int8(static_cast<float>(x_ptr[k]) * inv_s_a);
    x_quantized[static_cast<size_t>(k)] =
        static_cast<XType>(static_cast<float>(q_a) * s_a);
  }

  constexpr int32_t ACT_GROUP_SIZE = 256;
  constexpr int32_t LUT_G = 4;
  hvx_lut_ctor<LType, XType, ACT_GROUP_SIZE, 128, false, LUT_G>(
      gemm_k, gemm_n, x_quantized.data(), l_ptr, ls_ptr, lb_ptr);
  return GraphStatus::Success;
}

__attribute__((unused)) static float tmanprecomputea8CostFunc(const Op* op) {
  float cost = 0.0f;
  return cost;
}

END_PKG_OP_DEFINITION(PKG_TMANPrecomputeA8);
