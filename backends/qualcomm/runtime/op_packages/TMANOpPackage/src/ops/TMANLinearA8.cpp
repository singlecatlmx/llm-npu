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

BEGIN_PKG_OP_DEFINITION(PKG_TMANLinearA8);

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
GraphStatus tmanlineara8Impl(TensorType& c,
                             const TensorType& l,
                             const TensorType& qweight,
                             const TensorType& scales,
                             const Int32Tensor& t_group_size,
                             const Int32Tensor& t_bits,
                             const Int32Tensor& t_symmetric);

static float tmanlineara8CostFunc(const Op *op);

DEF_PACKAGE_OP((tmanlineara8Impl<Tensor>), "TMANLinearA8")

DEF_TENSOR_PROPERTIES(
  Op("TMANLinearA8", "l", "qweight", "scales", "group_size", "bits", "symmetric"),
  Flat("*", "l", "qweight", "scales"),
  MainMemory("group_size", "bits", "symmetric"),
  Tcm("*", "l", "qweight", "scales"))

DEF_PACKAGE_PARAM_ORDER("TMANLinearA8",
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
GraphStatus tmanlineara8Impl(TensorType& c,
                             const TensorType& l,
                             const TensorType& qweight,
                             const TensorType& scales,
                             const Int32Tensor& t_group_size,
                             const Int32Tensor& t_bits,
                             const Int32Tensor& t_symmetric)
{
  // `TMANLinearA8` 是 A8 三段式里的核心算子，但这里依旧只做：
  // 1. 参数 / tensor rank / packed ABI 校验；
  // 2. 把 packed 4D 物理形态恢复成 helper 可理解的 layout；
  // 3. 调用 hvx_funcs.h 中已经收敛好的 A8 helper。
  //
  // 这样做的原因是：
  // - 便于把复杂 kernel 逻辑集中在一个地方维护；
  // - 便于后续 device 编译问题先定位“接口层”还是“helper 内核层”。
  const int32_t group_size = ((const int32_t*)t_group_size.raw_data_const())[0];
  const int32_t bits = ((const int32_t*)t_bits.raw_data_const())[0];
  const bool symmetric = ((const int32_t*)t_symmetric.raw_data_const())[0] != 0;
  if (!tman_a8_is_target_config(bits, group_size, symmetric)) {
    return GraphStatus::ErrorFatal;
  }

  const int32_t gemm_n = tman_a8_num_rows(c);
  auto c_dims = c.dims();
  const int32_t gemm_m = (int32_t)c_dims[c_dims.size() - 1];

  auto qweight_dims = qweight.dims();
  // 当前 qweight 对外 ABI 已压回 4D packed：
  // [1, M, G, qgroup_vector_blocks * (4 * bits)]
  // 这里先从最后一维恢复出 qgroup_vector_blocks，再继续做 layout 推导。
  if (qweight_dims.size() != 4 ||
      (int32_t)qweight_dims[0] != 1 ||
      (int32_t)qweight_dims[1] != gemm_m) {
    return GraphStatus::ErrorFatal;
  }
  const int32_t num_groups = (int32_t)qweight_dims[2];
  const int32_t packed_weight_values = (int32_t)qweight_dims[3];
  if (packed_weight_values % TMAN_A8_WEIGHT_IDX_VALUES_PER_VECTOR_BLOCK != 0) {
    return GraphStatus::ErrorFatal;
  }
  const int32_t qgroup_vector_blocks =
      packed_weight_values / TMAN_A8_WEIGHT_IDX_VALUES_PER_VECTOR_BLOCK;
  const int32_t qgroups_per_group = qgroup_vector_blocks * TMAN_A8_QGROUPS_PER_VECTOR;
  const int32_t gemm_k = num_groups * qgroups_per_group * 4;
  if (gemm_k % 4 != 0 || group_size % 4 != 0) {
    return GraphStatus::ErrorFatal;
  }
  TmanA8Layout layout;
  if (!tman_a8_populate_layout(&layout, gemm_n, gemm_m, gemm_k, group_size)) {
    return GraphStatus::ErrorFatal;
  }
  if (layout.num_groups != num_groups ||
      layout.qgroup_vector_blocks != qgroup_vector_blocks ||
      layout.qgroups_per_group != qgroups_per_group) {
    return GraphStatus::ErrorFatal;
  }
  if (!tman_a8_is_target_layout(layout)) {
    return GraphStatus::ErrorFatal;
  }
  auto l_dims = l.dims();
  // LUT scratch 对外 ABI 也是 4D packed：
  // [1, row, group, qgroup_vector_blocks * (4 * 16)]
  if (l_dims.size() != 4 ||
      (int32_t)l_dims[0] != 1 ||
      (int32_t)l_dims[1] != gemm_n ||
      (int32_t)l_dims[2] != num_groups ||
      (int32_t)l_dims[3] !=
          qgroup_vector_blocks * TMAN_A8_LUT_VALUES_PER_VECTOR_BLOCK) {
    return GraphStatus::ErrorFatal;
  }
  auto scales_dims = scales.dims();
  if (scales_dims.size() != 4 ||
      (int32_t)scales_dims[0] != 1 ||
      (int32_t)scales_dims[1] != gemm_m ||
      (int32_t)scales_dims[2] != num_groups ||
      (int32_t)scales_dims[3] != 1) {
    return GraphStatus::ErrorFatal;
  }

  const int16_t* l_ptr = (const int16_t*)l.raw_data_const();
  const uint8_t* idx_ptr = (const uint8_t*)qweight.raw_data_const();
  const float* s_ptr = (const float*)scales.raw_data_const();
  float* c_ptr = (float*)c.raw_data();
  tman_a8_linear_scalar(
      layout,
      bits,
      l_ptr,
      idx_ptr,
      s_ptr,
      c_ptr);

  return GraphStatus::Success;
}

__attribute__((unused)) static float tmanlineara8CostFunc(const Op *op)
{
  float cost = 0.0;
  return cost;
}

END_PKG_OP_DEFINITION(PKG_TMANLinearA8);
