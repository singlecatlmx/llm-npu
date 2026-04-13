#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

#include "hvx_funcs.h"

BEGIN_PKG_OP_DEFINITION(PKG_TMANPrecomputeA8);

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
GraphStatus tmanprecomputea8Impl(TensorType& l,
                                 const TensorType& x,
                                 const Int32Tensor& t_group_size,
                                 const Int32Tensor& t_bits,
                                 const Int32Tensor& t_symmetric);

static float tmanprecomputea8CostFunc(const Op *op);

DEF_PACKAGE_OP((tmanprecomputea8Impl<Tensor>), "TMANPrecomputeA8")

DEF_TENSOR_PROPERTIES(Op("TMANPrecomputeA8", "x", "group_size", "bits", "symmetric"),
                      Flat("*", "x"),
                      MainMemory("group_size", "bits", "symmetric"),
                      Tcm("*", "x"))

DEF_PACKAGE_PARAM_ORDER("TMANPrecomputeA8",
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
GraphStatus tmanprecomputea8Impl(TensorType& l,
                                 const TensorType& x,
                                 const Int32Tensor& t_group_size,
                                 const Int32Tensor& t_bits,
                                 const Int32Tensor& t_symmetric)
{
  // 这个 op 的职责刻意保持得很“薄”：
  // 1. 检查当前节点确实处于项目支持的目标配置；
  // 2. 把 tensor shape 解释成 helper 所需的 layout；
  // 3. 调用 hvx_funcs.h 中的 A8 helper 真正构造 LUT。
  //
  // 这样后续如果把内部实现继续替换成更激进的 HVX intrinsic 版本，
  // 这里的接口层不需要继续承载复杂分支。
  const int32_t group_size = ((const int32_t*)t_group_size.raw_data_const())[0];
  const int32_t bits = ((const int32_t*)t_bits.raw_data_const())[0];
  const bool symmetric = ((const int32_t*)t_symmetric.raw_data_const())[0] != 0;
  if (!tman_a8_is_target_config(bits, group_size, symmetric)) {
    return GraphStatus::ErrorFatal;
  }

  auto x_dims = x.dims();
  const int32_t gemm_k = (int32_t)x_dims[x_dims.size() - 1];
  if (gemm_k % 4 != 0) {
    return GraphStatus::ErrorFatal;
  }

  const int32_t num_rows = tman_a8_num_rows(x);
  TmanA8Layout layout;
  if (!tman_a8_populate_layout(&layout, num_rows, 0, gemm_k, group_size)) {
    return GraphStatus::ErrorFatal;
  }
  auto l_dims = l.dims();
  // 注意这里对外 ABI 仍按 4D packed tensor 检查：
  // [1, row, group, qgroup_vector_blocks * (4 * 16)]
  // 逻辑上的 block 结构由 helper 内部继续解释。
  if (l_dims.size() != 4 ||
      (int32_t)l_dims[0] != 1 ||
      (int32_t)l_dims[1] != num_rows ||
      (int32_t)l_dims[2] != layout.num_groups ||
      (int32_t)l_dims[3] !=
          layout.qgroup_vector_blocks * TMAN_A8_LUT_VALUES_PER_VECTOR_BLOCK) {
    return GraphStatus::ErrorFatal;
  }

  const int8_t* x_ptr = (const int8_t*)x.raw_data_const();
  int16_t* l_ptr = (int16_t*)l.raw_data();
  tman_a8_precompute_lut_scalar(layout, x_ptr, l_ptr);
  return GraphStatus::Success;
}

__attribute__((unused)) static float tmanprecomputea8CostFunc(const Op *op)
{
  float cost = 0.0;
  return cost;
}

END_PKG_OP_DEFINITION(PKG_TMANPrecomputeA8);
