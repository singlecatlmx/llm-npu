//==============================================================================
// W4A8 TMAN linear path aligned with the FP16 TMAN pipeline
// - l       : packed [lut:int16][ls:float][lb:float]
// - qweight : old TMAN packed int32 weight layout
// - scales  : old TMAN packed int32 scale layout
// - c       : float scratch buffer consumed by TMANFinalize
// - scope   : W4G128 symmetric, decoder-style N=1
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

#include "hvx_funcs.h"

#ifndef PREPARE_DISABLED
API_EXPORT QuickShape simpledim_chunk1_4d_split_start_a8(
    Replacement& rpx,
    Split_Context const& splitinfo,
    OpRef const& orig,
    int dim) {
  size_t dims[4] = {0, 0, 0, 0};
  dims[dim] = splitinfo.start / splitinfo.size;
  return QuickShape(dims[0], dims[1], dims[2], dims[3]);
}

API_EXPORT QuickShape simpledim_chunk1_4d_split_size_a8(
    Replacement& rpx,
    Split_Context const& splitinfo,
    OpRef const& orig,
    int dim) {
  size_t dims[4] = {
      orig.dim(rpx.graph(), 0),
      orig.dim(rpx.graph(), 1),
      orig.dim(rpx.graph(), 2),
      orig.dim(rpx.graph(), 3)};
  dims[dim] = 1;
  return QuickShape(dims[0], dims[1], dims[2], dims[3]);
}
#endif

BEGIN_PKG_OP_DEFINITION(PKG_TMANLinearA8);

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
GraphStatus tmanlineara8Impl(
    TensorType& c,
    const TensorType& l,
    const TensorType& qweight,
    const TensorType& scales,
    const Int32Tensor& t_group_size,
    const Int32Tensor& t_bits,
    const Int32Tensor& t_symmetric);

static float tmanlineara8CostFunc(const Op* op);

DEF_PACKAGE_OP((tmanlineara8Impl<Tensor>), "TMANLinearA8")

DEF_TENSOR_PROPERTIES(
    Op("TMANLinearA8", "l", "qweight", "scales", "group_size", "bits", "symmetric"),
    Flat("*", "l", "qweight", "scales"),
    MainMemory("group_size", "bits", "symmetric"),
    Tcm("*", "l", "qweight", "scales"))

#define SIZE_OF(TENSOR)       \
  MUL(ELEMENTSIZE_OF(TENSOR), \
      DIM_OF(TENSOR, 0),      \
      DIM_OF(TENSOR, 1),      \
      DIM_OF(TENSOR, 2),      \
      DIM_OF(TENSOR, 3))

DEF_PACKAGE_OPTIMIZATION(
    EARLY,
    Op("TMANLinearA8", "l", "qweight", "scales", "group_size", "bits", "symmetric"),
    AND(GT(DIM_OF("qweight", 2), 1), GT(SIZE_OF("scales"), 128)),
    AUTOSPLIT(
        3,
        "I",
        DIV(DIM_OF("*", 3), DIM_OF("qweight", 2)),
        Op(
            "TMANLinearA8",
            "l",
            AUTOSPLIT_SLICE(
                "qweight",
                AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_start_a8, "I", "qweight", 2),
                AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_size_a8, "I", "qweight", 2)),
            AUTOSPLIT_SLICE(
                "scales",
                AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_start_a8, "I", "scales", 2),
                AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_size_a8, "I", "scales", 2)),
            "group_size",
            "bits",
            "symmetric")))

DEF_PACKAGE_PARAM_ORDER(
    "TMANLinearA8",
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
GraphStatus tmanlineara8Impl(
    TensorType& c,
    const TensorType& l,
    const TensorType& qweight,
    const TensorType& scales,
    const Int32Tensor& t_group_size,
    const Int32Tensor& t_bits,
    const Int32Tensor& t_symmetric) {
  using LType = int16_t;
  using XType = __fp16;
  using CType = float;

  constexpr int32_t ACT_GROUP_SIZE = 256;
  constexpr int32_t LUT_G = 4;
  constexpr int32_t TILE_K = 256;

  const int32_t group_size = ((const int32_t*)t_group_size.raw_data_const())[0];
  const int32_t bits = ((const int32_t*)t_bits.raw_data_const())[0];
  const bool symmetric = ((const int32_t*)t_symmetric.raw_data_const())[0] != 0;
  if (group_size != 128 || bits != 4 || !symmetric) {
    return GraphStatus::ErrorDimensions;
  }
  if (l.dims().size() != 4 || qweight.dims().size() != 4 || scales.dims().size() != 4 ||
      c.dims().size() != 4) {
    return GraphStatus::ErrorDimensions;
  }

  const int32_t gemm_n = static_cast<int32_t>(c.dims()[2]);
  if (gemm_n != 1) {
    return GraphStatus::ErrorDimensions;
  }

  const int32_t gemm_m = static_cast<int32_t>(c.dims()[3]) / sizeof(float) / bits;
  const int32_t gemm_k =
      static_cast<int32_t>(qweight.dims()[2]) * static_cast<int32_t>(qweight.dims()[3]) *
      32 / bits / gemm_m;
  if (gemm_k <= 0 || (gemm_k % group_size) != 0 || (gemm_k % TILE_K) != 0) {
    return GraphStatus::ErrorDimensions;
  }

  const int32_t l_size = tman_w4a8_packed_lut_size(gemm_k);
  const int32_t ls_size = tman_w4a8_packed_ls_size(gemm_k);
  const int32_t expected_l_bytes = tman_w4a8_packed_buffer_bytes(gemm_k, group_size);
  const int32_t expected_c_bytes =
      gemm_m * bits * static_cast<int32_t>(sizeof(float));
  if (static_cast<int32_t>(l.dims()[0]) != 1 || static_cast<int32_t>(l.dims()[1]) != 1 ||
      static_cast<int32_t>(l.dims()[2]) != 1 ||
      static_cast<int32_t>(l.dims()[3]) != expected_l_bytes) {
    return GraphStatus::ErrorDimensions;
  }
  if (static_cast<int32_t>(c.dims()[0]) != 1 || static_cast<int32_t>(c.dims()[1]) != 1 ||
      static_cast<int32_t>(c.dims()[2]) != 1 ||
      static_cast<int32_t>(c.dims()[3]) != expected_c_bytes) {
    return GraphStatus::ErrorDimensions;
  }

  const LType* l_ptr = (const LType*)l.raw_data_const();
  const float* ls_ptr = (const float*)(l_ptr + l_size);
  const float* lb_ptr = ls_ptr + MAX(ls_size, 128 / sizeof(float));
  const uint8_t* w_ptr = (const uint8_t*)qweight.raw_data_const();
  const XType* s_ptr = (const XType*)scales.raw_data_const();
  CType* c_ptr = (CType*)c.raw_data();
  if (l_ptr == nullptr || ls_ptr == nullptr || lb_ptr == nullptr || w_ptr == nullptr ||
      s_ptr == nullptr || c_ptr == nullptr) {
    return GraphStatus::ErrorFatal;
  }

  hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 128, false, 4, TILE_K, LUT_G, true>(
      gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, c_ptr);
  return GraphStatus::Success;
}

__attribute__((unused)) static float tmanlineara8CostFunc(const Op* op) {
  float cost = 0.0f;
  return cost;
}

END_PKG_OP_DEFINITION(PKG_TMANLinearA8);
