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

BEGIN_PKG_OP_DEFINITION(PKG_TMANFinalize);

static Qnn_Scalar_t sg_opDefaultGroup_SizeScalar = {.dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
                                                   .int32Value = 64};
static Qnn_Param_t sg_opDefaultGroup_Size = {.paramType = QNN_PARAMTYPE_SCALAR,
                                            .scalarParam = sg_opDefaultGroup_SizeScalar};
static Qnn_Scalar_t sg_opDefaultBitsScalar = {.dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
                                             .int32Value = 2};
static Qnn_Param_t sg_opDefaultBits = {.paramType = QNN_PARAMTYPE_SCALAR,
                                      .scalarParam = sg_opDefaultBitsScalar};
static Qnn_Scalar_t sg_opDefaultSymmetricScalar = {.dataType = Qnn_DataType_t::QNN_DATATYPE_INT_32,
                                                  .int32Value = 0};
static Qnn_Param_t sg_opDefaultSymmetric = {.paramType = QNN_PARAMTYPE_SCALAR,
                                           .scalarParam = sg_opDefaultSymmetricScalar};

template<typename TensorType>
GraphStatus tmanfinalizeImpl(TensorType& y,
                             const TensorType& c,
                             const Int32Tensor& t_group_size,
                             const Int32Tensor& t_bits,
                             const Int32Tensor& t_symmetric);

static float tmanfinalizeCostFunc(const Op *op);

DEF_PACKAGE_OP((tmanfinalizeImpl<Tensor>), "TMANFinalize")

// Tcm("y") results in [ERROR] [Qnn ExecuTorch]: graph_prepare.cc:217:ERROR:could not create op: q::Add.tcm
// Reason: embedding (Gather) outputs are in MainMemory
//         but TMANLinear outputs are in Tcm
//         add(embedding, TMANLinear) thus causes a conflict
// TODO:
// - implement custom TMANOpPackage::Add
DEF_TENSOR_PROPERTIES(Op("TMANFinalize", "c", "group_size", "bits", "symmetric"),
                      Flat("*", "c"),
                      MainMemory("*", "group_size", "bits", "symmetric"),
                      Tcm("c"))

DEF_PACKAGE_PARAM_ORDER("TMANFinalize",
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
GraphStatus tmanfinalizeImpl(TensorType& y,
                             const TensorType& c,
                             const Int32Tensor& t_group_size,
                             const Int32Tensor& t_bits,
                             const Int32Tensor& t_symmetric)
{
  using XType = __fp16;
  using CType = float;

  const int32_t gemm_m = y.dims()[3];
  const int32_t gemm_n = y.dims()[2];

  const int32_t bits = ((const int32_t*)t_bits.raw_data_const())[0];
  const DType y_dtype = y.get_dtype();
  const int32_t c_elements = c.total_storage_bytes() / sizeof(int32_t);

  if (tman_is_q8_dtype(y_dtype))
  {
    // W4A8 新路径：
    // - 输入 c 中保存的是 int32 中间结果
    // - 结果已经处于输出 q-domain
    // - Finalize 仅负责做存储域回写
    const int32_t storage_zero_point = tman_q8_storage_zero_point(y_dtype);
    if (tman_w4a8_finalize_to_q8(
            gemm_m,
            y_dtype,
            storage_zero_point,
            reinterpret_cast<const int32_t *>(c.raw_data_const()),
            c_elements,
            y.raw_data()) != 0)
    {
      return GraphStatus::ErrorDimensions;
    }
    return GraphStatus::Success;
  }

  // 旧路径只接受 Float16 输出，继续走 bit-serial + fp16 收尾。
  if (y_dtype != DType::Float16)
  {
    return GraphStatus::ErrorDimensions;
  }

  const CType* c_ptr = (const CType*)c.raw_data_const();
  XType* y_ptr       = (XType*)y.raw_data();

  if (bits == 2)
  {
    hvx_bit_serial<XType, CType, 2>(gemm_m, gemm_n, c_ptr, y_ptr);
  }
  else if (bits == 4)
  {
    hvx_bit_serial<XType, CType, 4>(gemm_m, gemm_n, c_ptr, y_ptr);
  }
  else
  {
    return GraphStatus::ErrorDimensions;
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float tmanfinalizeCostFunc(const Op *op)
{
  float cost = 0.0;  // add cost computation here
  return cost;
}

END_PKG_OP_DEFINITION(PKG_TMANFinalize);
