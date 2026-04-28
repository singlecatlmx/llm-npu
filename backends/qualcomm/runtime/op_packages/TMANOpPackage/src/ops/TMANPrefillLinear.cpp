//==============================================================================
// Auto Generated Code for TMANOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"
// #include "HTP/core/op.h" // scratch hook head file

#include "hvx_funcs.h"

#ifndef PREPARE_DISABLED
static API_EXPORT QuickShape simpledim_chunk1_4d_split_start(Replacement &rpx, Split_Context const &splitinfo, OpRef const &orig, int dim)
{
  size_t dims[4] = { 0, 0, 0, 0 };
  dims[dim] = splitinfo.start / splitinfo.size;
  return QuickShape(dims[0], dims[1], dims[2], dims[3]);
}

static API_EXPORT QuickShape simpledim_chunk1_4d_split_size(Replacement &rpx, Split_Context const &splitinfo, OpRef const &orig, int dim)
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


BEGIN_PKG_OP_DEFINITION(PKG_TMANPrefillLinear);


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
GraphStatus tmanprefilllinearImpl(TensorType& y,  //
                                  // inputs
                                  const TensorType& a,       
                                  const TensorType& lut1_,   
                                  const TensorType& eye_tile_,
                                  const TensorType& hf_lut,
                                  const TensorType& scales_off,
                                  const TensorType& qweight,    
                                  const TensorType& scales,   
                                  // params
                                  const Int32Tensor& t_group_size,
                                  const Int32Tensor& t_bits, 
                                  const Int32Tensor& t_symmetric
                                  // scratch outputs
                                //   PlainFloatTensor& act_temp_,
                                //   PlainFloatTensor& weight_temp_,
                                //   PlainFloatTensor& output_temp_ 
                                );   

static float tmanprefilllinearCostFunc(const Op *op);

DEF_PACKAGE_OP((tmanprefilllinearImpl<Tensor>), "TMANPrefillLinear")

DEF_TENSOR_PROPERTIES(Op("TMANPrefillLinear", "a", "lut1_", "eye_tile_", "hf_lut", "scales_off", "qweight", "scales", "group_size", "bits", "symmetric"),
                      Flat("*", "a", "lut1_", "eye_tile_", "qweight", "scales"),
                      MainMemory("*", "group_size", "bits", "symmetric"),
                      Tcm("a", "lut1_", "eye_tile_", "hf_lut", "scales_off", "qweight", "scales"))

#define SIZE_OF(WEIGHT) MUL(ELEMENTSIZE_OF(WEIGHT), DIM_OF(WEIGHT, 0), DIM_OF(WEIGHT, 1), DIM_OF(WEIGHT, 2), DIM_OF(WEIGHT, 3))


// GPTQ
DEF_PACKAGE_OPTIMIZATION(
    EARLY,
    Op("TMANPrefillLinear", "a", "lut1_", "eye_tile_", "hf_lut", "scales_off", "qweight", "scales", "group_size", "bits", "symmetric"),
    AND(GT(DIM_OF("qweight", 2), 1), GT(SIZE_OF("scales"), 128)),
    AUTOSPLIT(3, "I", DIV(DIM_OF("*", 3), DIM_OF("qweight", 2)),
        Op(
        "TMANPrefillLinear", 
        "a", 
        "lut1_", 
        "eye_tile_", 
        "hf_lut", 
        "scales_off",
        AUTOSPLIT_SLICE("qweight",
            AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_start, "I", "qweight", 2),
            AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_size, "I", "qweight", 2)),
        AUTOSPLIT_SLICE("scales",
            AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_start, "I", "scales", 2),
            AUTOSPLIT_SHAPEFN_APPLY(simpledim_chunk1_4d_split_size, "I", "scales", 2)),
        "group_size", 
        "bits", 
        "symmetric")))         


DEF_PACKAGE_PARAM_ORDER("TMANPrefillLinear",
                        "group_size",
                        false,
                        &sg_opDefaultGroup_Size,
                        "bits",
                        false,
                        &sg_opDefaultBits,
                        "symmetric",
                        false,
                        &sg_opDefaultSymmetric)


template<typename TensorType = Tensor>
GraphStatus tmanprefilllinearImpl(TensorType& y,  //
                                  // inputs
                                  const TensorType& a,       
                                  const TensorType& lut1_,   
                                  const TensorType& eye_tile_,
                                  const TensorType& hf_lut,
                                  const TensorType& scales_off,
                                  const TensorType& qweight,    
                                  const TensorType& scales,   
                                  // params
                                  const Int32Tensor& t_group_size,
                                  const Int32Tensor& t_bits, 
                                  const Int32Tensor& t_symmetric
                                  // scratch outputs
                                //   PlainFloatTensor& act_temp_,
                                //   PlainFloatTensor& weight_temp_,
                                //   PlainFloatTensor& output_temp_
                                )
{
    const int32_t group_size = ((const int32_t*)t_group_size.raw_data_const())[0];
    const int32_t Bits       = ((const int32_t*)t_bits.raw_data_const())[0];
    const bool zero_point = ((const int32_t*)t_symmetric.raw_data_const())[0] == 0;

    const int32_t gemm_n = y.dims()[2];
    const int32_t gemm_m = y.dims()[3];
    const int32_t gemm_k = a.dims()[3];

    int32_t TileK         = 256;
    int32_t TileM         = gemm_m; // need to be same with the tile_p / Bits
    int32_t TileN         = 32;
    
    const __fp16*   a_ptr = (const __fp16*)a.raw_data_const(); 
    const uint8_t*  w_ptr = (const uint8_t*)qweight.raw_data_const();
    const __fp16*   s_ptr = (const __fp16*)scales.raw_data_const();
    __fp16*         c_ptr = (__fp16*)y.raw_data();

    const uint16_t* lut1_ptr        = (const uint16_t*)lut1_.raw_data_const();
    const __fp16 *eye_tile_ptr      = (const __fp16*)eye_tile_.raw_data_const();
    const __fp16 *hf_lut_vtcm_ptr   = (const __fp16*)hf_lut.raw_data_const();
    const __fp16 *scales_off_ptr    = (const __fp16*)scales_off.raw_data_const();

    // __fp16*    act_temp    = (__fp16*)act_temp_.raw_data();
    // __fp16*    weight_temp = (__fp16*)weight_temp_.raw_data();
    // __fp16*    output_temp = (__fp16*)output_temp_.raw_data();

    for (int i = 0; i < gemm_m * gemm_n; i++) {
        c_ptr[i] = (__fp16)1.0;
    }
    
    // if (Bits == 4) {
    //     if (group_size == 128) {
    //         if (zero_point) {
    //             // Case: Bits=4, Group=128, ZeroPoint=true
    //             hmx_hvx_tbl_gemm<128, true, 4, 4>(
    //                 TileN,
    //                 TileM,
    //                 TileK,
    //                 gemm_m, gemm_k, gemm_n, a_ptr, w_ptr, s_ptr, c_ptr, 
    //                 act_temp, weight_temp, output_temp, lut1_ptr, 
    //                 eye_tile_ptr, hf_lut_vtcm_ptr, scales_off_ptr);
    //         } else {
    //             // Case: Bits=4, Group=128, ZeroPoint=false
    //             hmx_hvx_tbl_gemm<128, false, 4, 4>(
    //                 TileN,
    //                 TileM,
    //                 TileK,
    //                 gemm_m, gemm_k, gemm_n, a_ptr, w_ptr, s_ptr, c_ptr, 
    //                 act_temp, weight_temp, output_temp, lut1_ptr, 
    //                 eye_tile_ptr, hf_lut_vtcm_ptr, scales_off_ptr);
    //         }
    //     } else if (group_size == 64) {
    //         if (zero_point) {
    //             // Case: Bits=4, Group=64, ZeroPoint=true
    //             hmx_hvx_tbl_gemm<64, true, 4, 4>(
    //                 TileN,
    //                 TileM,
    //                 TileK,
    //                 gemm_m, gemm_k, gemm_n, a_ptr, w_ptr, s_ptr, c_ptr, 
    //                 act_temp, weight_temp, output_temp, lut1_ptr, 
    //                 eye_tile_ptr, hf_lut_vtcm_ptr, scales_off_ptr);
    //         } else {
    //             // Case: Bits=4, Group=64, ZeroPoint=false
    //             hmx_hvx_tbl_gemm<64, false, 4, 4>(
    //                 TileN,
    //                 TileM,
    //                 TileK,
    //                 gemm_m, gemm_k, gemm_n, a_ptr, w_ptr, s_ptr, c_ptr, 
    //                 act_temp, weight_temp, output_temp, lut1_ptr, 
    //                 eye_tile_ptr, hf_lut_vtcm_ptr, scales_off_ptr);
    //         }
    //     } else {
    //         return GraphStatus::ErrorDimensions;
    //     }
    // } else if (Bits == 2) {
    //     if (group_size == 128) {
    //         if (zero_point) {
    //             // Case: Bits=2, Group=128, ZeroPoint=true
    //             hmx_hvx_tbl_gemm<128, true, 2, 4>(
    //                 TileN,
    //                 TileM,
    //                 TileK,
    //                 gemm_m, gemm_k, gemm_n, a_ptr, w_ptr, s_ptr, c_ptr, 
    //                 act_temp, weight_temp, output_temp, lut1_ptr, 
    //                 eye_tile_ptr, hf_lut_vtcm_ptr, scales_off_ptr);
    //         } else {
    //             // Case: Bits=2, Group=128, ZeroPoint=false
    //             hmx_hvx_tbl_gemm<128, false, 2, 4>(
    //                 TileN,
    //                 TileM,
    //                 TileK,
    //                 gemm_m, gemm_k, gemm_n, a_ptr, w_ptr, s_ptr, c_ptr, 
    //                 act_temp, weight_temp, output_temp, lut1_ptr, 
    //                 eye_tile_ptr, hf_lut_vtcm_ptr, scales_off_ptr);
    //         }
    //     } else if (group_size == 64) {
    //         if (zero_point) {
    //             // Case: Bits=2, Group=64, ZeroPoint=true
    //             hmx_hvx_tbl_gemm<64, true, 2, 4>(
    //                 TileN,
    //                 TileM,
    //                 TileK,
    //                 gemm_m, gemm_k, gemm_n, a_ptr, w_ptr, s_ptr, c_ptr, 
    //                 act_temp, weight_temp, output_temp, lut1_ptr, 
    //                 eye_tile_ptr, hf_lut_vtcm_ptr, scales_off_ptr);
    //         } else {
    //             // Case: Bits=2, Group=64, ZeroPoint=false
    //             hmx_hvx_tbl_gemm<64, false, 2, 4>(
    //                 TileN,
    //                 TileM,
    //                 TileK,
    //                 gemm_m, gemm_k, gemm_n, a_ptr, w_ptr, s_ptr, c_ptr, 
    //                 act_temp, weight_temp, output_temp, lut1_ptr, 
    //                 eye_tile_ptr, hf_lut_vtcm_ptr, scales_off_ptr);
    //         }
    //     } else {
    //         return GraphStatus::ErrorDimensions;
    //     }
    // } else {
    //     // 不支持的 Bits 
    //     return GraphStatus::ErrorDimensions;
    // }
    
    
    return GraphStatus::Success;
}

static float tmanprefilllinearCostFunc(const Op *op)
{
  return 0.0;
}

// #ifndef PREPARE_DISABLED
// #include "HTP/core/op.h"
// #endif

// namespace {

// class TMANPrefillLinearHook : public hnnx::OpHookBase {
// public:
//     virtual GraphStatus pre_allocate(hnnx::OpIoPtrs const &iop, Op &op) const override
//     {
//         Tensor const *const out0 = op.get_output(0);
//         size_t m = out0->dim(3);

//         size_t TileK         = 256;
//         size_t TileM         = m; // need to be same with the tile_p / Bits
//         size_t TileN         = 32;

//         size_t act_dims[1] = { TileN * TileK * 2 }; // 32KB
//         GraphStatus res1 = hnnx::change_output_tensor_shape(op, 1, iop.graph(), 1, act_dims);

//         size_t weight_dims[1] = { TileM * TileK * 2 }; // 128KB
//         GraphStatus res2 = hnnx::change_output_tensor_shape(op, 2, iop.graph(), 1, weight_dims);

//         size_t output_dims[1] = { TileN * TileM * 2 }; 
//         GraphStatus res3 = hnnx::change_output_tensor_shape(op, 3, iop.graph(), 1, output_dims);

//         if (res1 != GraphStatus::Success || res2 != GraphStatus::Success || res3 != GraphStatus::Success) {
//             return GraphStatus::ErrorFatal;
//         }
//         return GraphStatus::Success;
//     }
// };

// } // namespace

// #ifndef PREPARE_DISABLED
// CTOR_OPHOOK((tmanprefilllinearImpl<Tensor>), TMANPrefillLinearHook)
// #endif

END_PKG_OP_DEFINITION(PKG_TMANPrefillLinear);
