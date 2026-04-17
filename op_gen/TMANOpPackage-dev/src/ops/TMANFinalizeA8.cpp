//==============================================================================
// Minimal W4A8 TMAN finalize path
// - input  : float32 accumulators already scaled to output quantized domain
// - output : int8 activation
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
    const Int32Tensor& t_group_size,
    const Int32Tensor& t_bits,
    const Int32Tensor& t_symmetric);

static float tmanfinalizea8CostFunc(const Op* op);

DEF_PACKAGE_OP((tmanfinalizea8Impl<Tensor>), "TMANFinalizeA8")

DEF_TENSOR_PROPERTIES(
    Op("TMANFinalizeA8", "c", "group_size", "bits", "symmetric"),
    Flat("*", "c"),
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
    const Int32Tensor& t_group_size,
    const Int32Tensor& t_bits,
    const Int32Tensor& t_symmetric) {
  const int32_t bits = ((const int32_t*)t_bits.raw_data_const())[0];
  const bool symmetric = ((const int32_t*)t_symmetric.raw_data_const())[0] != 0;
  UNUSED(t_group_size);
  if (bits != 4 || !symmetric) {
    return GraphStatus::ErrorDimensions;
  }

  auto y_dims = y.dims();
  auto c_dims = c.dims();
  if (y_dims.size() != 4 || c_dims.size() != 4) {
    return GraphStatus::ErrorDimensions;
  }
  if ((int32_t)y_dims[0] != 1 || (int32_t)y_dims[1] != 1 ||
      (int32_t)c_dims[0] != 1 || (int32_t)c_dims[1] != 1 ||
      (int32_t)y_dims[2] != (int32_t)c_dims[2] ||
      (int32_t)y_dims[3] != (int32_t)c_dims[3]) {
    return GraphStatus::ErrorDimensions;
  }

  const int32_t gemm_n = (int32_t)y_dims[2];
  const int32_t gemm_m = (int32_t)y_dims[3];
  const float* c_ptr = (const float*)c.raw_data_const();
  uint8_t* y_raw = (uint8_t*)y.raw_data();
  if (c_ptr == nullptr || y_raw == nullptr) {
    return GraphStatus::ErrorFatal;
  }

  tman_a8_finalize_to_qnn_int8_hvx(gemm_n, gemm_m, c_ptr, y_raw);
  return GraphStatus::Success;
}

__attribute__((unused)) static float tmanfinalizea8CostFunc(const Op* op) {
  float cost = 0.0f;
  return cost;
}

END_PKG_OP_DEFINITION(PKG_TMANFinalizeA8);
