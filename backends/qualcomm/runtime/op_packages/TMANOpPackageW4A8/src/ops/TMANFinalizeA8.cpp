//==============================================================================
// W4A8 TMAN finalize path for fp16 output
// - input  : float scratch buffer from TMANLinearA8
// - output : fp16 tensor after output activation quantize-dequant
// - scope  : W4G128 symmetric, decoder-style N=1
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

#include "hvx_funcs.h"

BEGIN_PKG_OP_DEFINITION(PKG_TMANFinalizeA8);

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
GraphStatus tmanfinalizea8Impl(
    TensorType& y,
    const TensorType& c,
    const TensorType& output_scale,
    const Int32Tensor& t_group_size,
    const Int32Tensor& t_bits,
    const Int32Tensor& t_symmetric);

static float tmanfinalizea8CostFunc(const Op* op);

DEF_PACKAGE_OP((tmanfinalizea8Impl<Tensor>), "TMANFinalizeA8")

DEF_TENSOR_PROPERTIES(
    Op("TMANFinalizeA8", "c", "output_scale", "group_size", "bits", "symmetric"),
    Flat("*", "c", "output_scale"),
    MainMemory("*", "group_size", "bits", "symmetric"),
    Tcm("c"))

DEF_PACKAGE_PARAM_ORDER(
    "TMANFinalizeA8",
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
GraphStatus tmanfinalizea8Impl(
    TensorType& y,
    const TensorType& c,
    const TensorType& output_scale,
    const Int32Tensor& t_group_size,
    const Int32Tensor& t_bits,
    const Int32Tensor& t_symmetric) {
  const int32_t group_size = ((const int32_t*)t_group_size.raw_data_const())[0];
  const int32_t bits = ((const int32_t*)t_bits.raw_data_const())[0];
  const bool symmetric = ((const int32_t*)t_symmetric.raw_data_const())[0] != 0;
  if (group_size != 128 || bits != 4 || !symmetric) {
    return GraphStatus::ErrorDimensions;
  }

  // 这里新增的 FinalizeA8 是这次正式 W4A8 路径的关键补充：
  // - 先复用旧 TMANFinalize 的 bit-serial 合成语义；
  // - 再按外部 output_scale 做一次对称量化与反量化；
  // - 最终仍然输出 fp16，方便继续衔接后续 fp16 算子。
  //
  // 换句话说，它是一个“保持层间 fp16 接口不变”的轻量 output-dequant 变体。
  auto y_dims = y.dims();
  auto c_dims = c.dims();
  auto output_scale_dims = output_scale.dims();
  if (y_dims.size() != 4 || c_dims.size() != 4 || output_scale_dims.size() != 4) {
    return GraphStatus::ErrorDimensions;
  }

  const int32_t gemm_n = static_cast<int32_t>(y_dims[2]);
  const int32_t gemm_m = static_cast<int32_t>(y_dims[3]);
  if (static_cast<int32_t>(y_dims[0]) != 1 || static_cast<int32_t>(y_dims[1]) != 1 ||
      gemm_n != 1 || static_cast<int32_t>(c_dims[0]) != 1 ||
      static_cast<int32_t>(c_dims[1]) != 1 || static_cast<int32_t>(c_dims[2]) != 1 ||
      static_cast<int32_t>(c_dims[3]) != gemm_m * bits * static_cast<int32_t>(sizeof(float))) {
    return GraphStatus::ErrorDimensions;
  }
  if (static_cast<int32_t>(output_scale_dims[0]) != 1 ||
      static_cast<int32_t>(output_scale_dims[1]) != 1 ||
      static_cast<int32_t>(output_scale_dims[2]) != 1 ||
      static_cast<int32_t>(output_scale_dims[3]) != 1) {
    return GraphStatus::ErrorDimensions;
  }

  const float* c_ptr = (const float*)c.raw_data_const();
  const float* output_scale_ptr = (const float*)output_scale.raw_data_const();
  __fp16* y_ptr = (__fp16*)y.raw_data();
  if (c_ptr == nullptr || output_scale_ptr == nullptr || y_ptr == nullptr) {
    return GraphStatus::ErrorFatal;
  }
  if (!(output_scale_ptr[0] > 0.0f)) {
    return GraphStatus::ErrorDimensions;
  }

  tman_w4a8_finalize_fp16_with_output_scale(
      gemm_n, gemm_m, c_ptr, output_scale_ptr[0], y_ptr);
  return GraphStatus::Success;
}

__attribute__((unused)) static float tmanfinalizea8CostFunc(const Op* op) {
  float cost = 0.0f;
  return cost;
}

END_PKG_OP_DEFINITION(PKG_TMANFinalizeA8);
