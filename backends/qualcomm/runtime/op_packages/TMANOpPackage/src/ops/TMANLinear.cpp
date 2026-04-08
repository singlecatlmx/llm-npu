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

#ifndef PREPARE_DISABLED
API_EXPORT QuickShape simpledim_chunk1_4d_split_start(Replacement &rpx, Split_Context const &splitinfo, OpRef const &orig, int dim)
{
  size_t dims[4] = { 0, 0, 0, 0 };
  dims[dim] = splitinfo.start / splitinfo.size;
  return QuickShape(dims[0], dims[1], dims[2], dims[3]);
}

API_EXPORT QuickShape simpledim_chunk1_4d_split_size(Replacement &rpx, Split_Context const &splitinfo, OpRef const &orig, int dim)
{
  size_t dims[4] = {
    orig.dim(rpx.graph(), 0),
    orig.dim(rpx.graph(), 1),
    orig.dim(rpx.graph(), 2),
    orig.dim(rpx.graph(), 3)
  };
  dims[dim] = 1;
  return QuickShape(dims[0], dims[1], dims[2], dims[3]);
}
#endif

BEGIN_PKG_OP_DEFINITION(PKG_TMANLinear);

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
GraphStatus tmanlinearImpl(TensorType& c,
                           const TensorType& l,
                           const TensorType& qweight,
                           const TensorType& scales,
                           const Int32Tensor& t_group_size,
                           const Int32Tensor& t_bits,
                           const Int32Tensor& t_symmetric);

static float tmanlinearCostFunc(const Op *op);

DEF_PACKAGE_OP((tmanlinearImpl<Tensor>), "TMANLinear")

DEF_TENSOR_PROPERTIES(
  Op("TMANLinear", "l", "qweight", "scales", "group_size", "bits", "symmetric"),
  Flat("*", "l", "qweight", "scales"),
  MainMemory("group_size", "bits", "symmetric"),
  Tcm("*", "l", "qweight", "scales"))

#define SIZE_OF(WEIGHT) MUL(ELEMENTSIZE_OF(WEIGHT), DIM_OF(WEIGHT, 0), DIM_OF(WEIGHT, 1), DIM_OF(WEIGHT, 2), DIM_OF(WEIGHT, 3))

// GPTQ
DEF_PACKAGE_OPTIMIZATION(
  EARLY,
  Op("TMANLinear", "l", "qweight", "scales", "group_size", "bits", "symmetric"),
  AND(GT(DIM_OF("qweight", 2), 1), GT(SIZE_OF("scales"), 128)),
  AUTOSPLIT(3, "I", DIV(DIM_OF("*", 3), DIM_OF("qweight", 2)),
    Op(
      "TMANLinear", "l",
      AUTOSPLIT_SLICE("qweight",
        AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_start, "I", "qweight", 2),
        AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_size, "I", "qweight", 2)),
      AUTOSPLIT_SLICE("scales",
        AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_start, "I", "scales", 2),
        AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_size, "I", "scales", 2)),
      "group_size", "bits", "symmetric")))

// BitNet: weight scale shouldn't be split
DEF_PACKAGE_OPTIMIZATION(
  EARLY + 1,
  Op("TMANLinear", "l", "qweight", "scales", "group_size", "bits", "symmetric"),
  AND(GT(DIM_OF("qweight", 2), 1), LE(SIZE_OF("scales"), 128)),
  AUTOSPLIT(3, "I", DIV(DIM_OF("*", 3), DIM_OF("qweight", 2)),
    Op(
      "TMANLinear", "l",
      AUTOSPLIT_SLICE("qweight",
        AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_start, "I", "qweight", 2),
        AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_size, "I", "qweight", 2)),
      "scales", "group_size", "bits", "symmetric")))

DEF_PACKAGE_PARAM_ORDER("TMANLinear",
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
GraphStatus tmanlinearImpl(TensorType& c,
                           const TensorType& l,
                           const TensorType& qweight,
                           const TensorType& scales,
                           const Int32Tensor& t_group_size,
                           const Int32Tensor& t_bits,
                           const Int32Tensor& t_symmetric)
{
  using LType = int16_t;
  using XType = __fp16;
  using CType = float;

  constexpr int32_t ACT_GROUP_SIZE = 256;
  constexpr int32_t LUT_G          = 4;
  constexpr int32_t LUT_SIZE       = 16;
  constexpr int32_t TILE_K         = 256;

  const int32_t group_size = ((const int32_t*)t_group_size.raw_data_const())[0];
  const int32_t bits       = ((const int32_t*)t_bits.raw_data_const())[0];
  const bool zero_point    = ((const int32_t*)t_symmetric.raw_data_const())[0] == 0;

  const int32_t gemm_n = c.dims()[2];
  const int32_t gemm_m = c.dims()[3] / sizeof(float) / bits;
  const int32_t gemm_k = qweight.dims()[2] * qweight.dims()[3] * 32 / bits / gemm_m;

  const int32_t l_size  = gemm_k / LUT_G * LUT_SIZE;
  const int32_t ls_size = (ACT_GROUP_SIZE == -1) ? 1 : (gemm_k / ACT_GROUP_SIZE);
  const int32_t ls_reserved = MAX(ls_size, 128 / sizeof(float));

  const LType* l_ptr  = (const LType*)l.raw_data_const();
  const float* ls_ptr = (const float*)(l_ptr + l_size);
  const float* lb_ptr = ls_ptr + ls_reserved;

  const uint8_t* w_ptr = (const uint8_t*)qweight.raw_data_const();
  const XType* s_ptr   = (const XType*)scales.raw_data_const();
  CType* c_ptr         = (CType*)c.raw_data();
  const int32_t c_elements = c.total_storage_bytes() / sizeof(int32_t);

  if (bits == TMAN_W4A8_BITS && lb_ptr[0] == TMAN_W4A8_SENTINEL)
  {
    // 看到 sentinel，说明上游 Precompute 已经走了 W4A8 新路径。
    // 此时不再进入旧的 hvx_tbl(float 累加) 实现，而是改走新的
    // int32 主路径。
    const int32_t p_tiles = qweight.dims()[2];
    if (p_tiles <= 0) return GraphStatus::ErrorDimensions;

    // qweight 在 builder 侧已经按 tile_p 预处理过。
    // 这里根据 qweight 第二维 tile 数反推出实际 tile_p。
    const int32_t tile_p = (gemm_m * bits) / p_tiles;
    if (tile_p <= 0 || (gemm_m * bits) % p_tiles != 0)
    {
      return GraphStatus::ErrorDimensions;
    }

    // 新路径直接把“输出 q-domain 的 int32”写入 c。
    if (tman_w4a8_linear_from_lut(
            gemm_m,
            gemm_k,
            group_size,
            zero_point,
            bits,
            tile_p,
            l_ptr,
            ls_ptr,
            lb_ptr,
            w_ptr,
            s_ptr,
            reinterpret_cast<int32_t *>(c.raw_data()),
            c_elements) != 0)
    {
      return GraphStatus::ErrorDimensions;
    }

    return GraphStatus::Success;
  }

  // 下面全部是旧的 W4A16 / BitNet 路径，保持不变。
  if (zero_point && bits == 2 && group_size == 64)  // w2g64, symmetric=False
  {
    hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 64, true, 2, TILE_K, LUT_G, true>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, c_ptr);
  }
  else if (!zero_point && bits == 4 && group_size == 128)  // w4g128, symmetric=True
  {
    hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 128, false, 4, TILE_K, LUT_G, true>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, c_ptr);
  }
  else if (zero_point && bits == 4 && group_size == 128)  // w4g128, symmetric=False
  {
    hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 128, true, 4, TILE_K, LUT_G, true>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, c_ptr);
  }
  else if (zero_point && bits == 4 && group_size == 64)  // w4g64, symmetric=False
  {
    hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 64, true, 4, TILE_K, LUT_G, true>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, c_ptr);
  }
  else if (!zero_point && bits == 2 && group_size == 0)  // bitnet
  {
    hvx_tbl<LType, XType, CType, -1, 0, false, 2, TILE_K, LUT_G, true>(gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, c_ptr);
  }
  else
  {
    return GraphStatus::ErrorDimensions;
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float tmanlinearCostFunc(const Op *op)
{
  float cost = 0.0;  // add cost computation here
  return cost;
}

END_PKG_OP_DEFINITION(PKG_TMANLinear);
