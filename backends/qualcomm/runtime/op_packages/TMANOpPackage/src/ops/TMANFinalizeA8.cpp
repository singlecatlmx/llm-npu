//==============================================================================
// Auto Generated Code for TMANOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

#include "hvx_funcs.h"

BEGIN_PKG_OP_DEFINITION(PKG_TMANFinalizeA8);

static Qnn_Scalar_t sg_opDefaultGroup_SizeScalar = {.dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
                                                   .int32Value = 128};
static Qnn_Param_t sg_opDefaultGroup_Size = {.paramType = QNN_PARAMTYPE_SCALAR,
                                            .scalarParam = sg_opDefaultGroup_SizeScalar};
static Qnn_Scalar_t sg_opDefaultBitsScalar = {.dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
                                             .int32Value = 4};
static Qnn_Param_t sg_opDefaultBits = {.paramType = QNN_PARAMTYPE_SCALAR,
                                      .scalarParam = sg_opDefaultBitsScalar};
static Qnn_Scalar_t sg_opDefaultSymmetricScalar = {.dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
                                                  .int32Value = 1};
static Qnn_Param_t sg_opDefaultSymmetric = {.paramType = QNN_PARAMTYPE_SCALAR,
                                           .scalarParam = sg_opDefaultSymmetricScalar};

template<typename TensorType>
GraphStatus tmanfinalizea8Impl(TensorType& y,
                               const TensorType& c,
                               const Int32Tensor& t_group_size,
                               const Int32Tensor& t_bits,
                               const Int32Tensor& t_symmetric);

static float tmanfinalizea8CostFunc(const Op *op);

DEF_PACKAGE_OP((tmanfinalizea8Impl<Tensor>), "TMANFinalizeA8")

DEF_TENSOR_PROPERTIES(Op("TMANFinalizeA8", "c", "group_size", "bits", "symmetric"),
                      Flat("*", "c"),
                      MainMemory("*", "group_size", "bits", "symmetric"),
                      Tcm("c"))

DEF_PACKAGE_PARAM_ORDER("TMANFinalizeA8",
                        "group_size",
                        false,
                        &sg_opDefaultGroup_Size,
                        "bits",
                        false,
                        &sg_opDefaultBits,
                        "symmetric",
                        false,
                        &sg_opDefaultSymmetric)

template<typename TensorType>
GraphStatus tmanfinalizea8Impl(TensorType& y,
                               const TensorType& c,
                               const Int32Tensor& t_group_size,
                               const Int32Tensor& t_bits,
                               const Int32Tensor& t_symmetric)
{
  // `FinalizeA8` 的职责是把 linear 阶段留下的 float accumulator
  // 收敛成最终的 INT8 输出。
  //
  // 当前 helper 内部已经支持：
  // - scalar fallback
  // - 一段更实质的 HVX round/clamp/s16-sat 路径
  // 这里仍然只负责做接口层校验，并把具体实现下沉到 hvx_funcs.h。
  const int32_t group_size = ((const int32_t*)t_group_size.raw_data_const())[0];
  const int32_t bits = ((const int32_t*)t_bits.raw_data_const())[0];
  const bool symmetric = ((const int32_t*)t_symmetric.raw_data_const())[0] != 0;
  if (!tman_a8_is_target_config(bits, group_size, symmetric)) {
    return GraphStatus::ErrorFatal;
  }

  const int32_t gemm_n = tman_a8_num_rows(y);
  auto y_dims = y.dims();
  const int32_t gemm_m = (int32_t)y_dims[y_dims.size() - 1];
  auto c_dims = c.dims();
  if (c_dims.size() != y_dims.size()) {
    return GraphStatus::ErrorFatal;
  }
  for (size_t i = 0; i < y_dims.size(); ++i) {
    if ((int32_t)c_dims[i] != (int32_t)y_dims[i]) {
      return GraphStatus::ErrorFatal;
    }
  }
  const float* c_ptr = (const float*)c.raw_data_const();
  int8_t* y_ptr = (int8_t*)y.raw_data();
  tman_a8_finalize_scalar(gemm_n, gemm_m, c_ptr, y_ptr);

  return GraphStatus::Success;
}

__attribute__((unused)) static float tmanfinalizea8CostFunc(const Op *op)
{
  float cost = 0.0;
  return cost;
}

END_PKG_OP_DEFINITION(PKG_TMANFinalizeA8);
