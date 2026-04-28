#pragma once

// #ifdef __cplusplus
//     #define restrict __restrict
// #endif

#define _Float16 __fp16

// typedef _Float16  __fp16;

// using _Float16 = __fp16;




#include <hexagon_types.h>
#include <string.h>
#include <stdint.h>
#include <cstring>
#include <type_traits>
#include <stdbool.h>
#include <math.h>


#include "hmx_utils.h"

extern "C" {
    #include "AEEStdErr.h"
    #include <stddef.h>
    #include "hexkl_micro.h"
    #include "HAP_farf.h"
}


#define UNUSED(x) (void)(x)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

static inline int32_t _fp32_to_bits(float x)
{
  union {
    float f;
    int32_t i;
  } u;
  u.f = x;
  return u.i;
}

static inline int16_t _fp16_to_bits(const __fp16 *x)
{
  union {
    __fp16 f;
    int16_t i;
  } u;
  u.f = *x;
  return u.i;
}

#ifndef _HVX_INTERNAL_H
#define _HVX_INTERNAL_H

#define VLEN 128
#define vmem(A)     *((HVX_Vector *)(A))
#define HVX_INLINE_ALWAYS inline __attribute__((unused,always_inline))
static HVX_INLINE_ALWAYS void l2fetch(const void *p, uint32_t stride,
                                      uint32_t width, uint32_t height,
                                      uint32_t dir)
{
#ifdef __hexagon__
    uint64_t control = HEXAGON_V64_CREATE_H(dir, stride, width, height);
    __asm__ __volatile__ (" l2fetch(%0,%1) " : :"r"(p),"r"(control));
#endif
}

#endif /* _HVX_INTERNAL_H */


static inline HVX_VectorPair HVX_int8_cvt_hf(HVX_Vector w_int8, HVX_Vector hf_lut){
    HVX_VectorPair w_hf = Q6_Wh_vlut16_VbVhR_nomatch(w_int8, hf_lut, 0);
    HVX_VectorPair result = Q6_W_vshuff_VVR(Q6_V_hi_W(w_hf), Q6_V_lo_W(w_hf), -2);
    return result;
}



#define HVX_VP_APPLY2(op, a, b) \
    Q6_W_vcombine_VV(op(Q6_V_hi_W(a), Q6_V_hi_W(b)), \
                     op(Q6_V_lo_W(a), Q6_V_lo_W(b)))

/* --- ?????????? --- */

static inline HVX_VectorPair hvx_vp_mpy_fp16(HVX_VectorPair a, HVX_VectorPair b) {
    return HVX_VP_APPLY2(Q6_Vhf_vmpy_VhfVhf, a, b);
}

static inline HVX_VectorPair hvx_vp_add_fp16(HVX_VectorPair a, HVX_VectorPair b) {
    return HVX_VP_APPLY2(Q6_Vqf16_vadd_VhfVhf, a, b);
}

static inline HVX_VectorPair hvx_int8_w_fp16_s_z_mpy(HVX_Vector w_int8, 
                                                     HVX_VectorPair s_fp16, 
                                                     HVX_VectorPair z_fp16,
                                                     HVX_Vector hf_lut) {

    HVX_VectorPair w_fp16 = HVX_int8_cvt_hf(w_int8, hf_lut);
    HVX_VectorPair acc    = hvx_vp_add_fp16(w_fp16, z_fp16);
    return hvx_vp_mpy_fp16(acc, s_fp16);
}


static inline HVX_VectorPair hvx_int8_w_fp16_s_mpy(HVX_Vector w_int8, 
                                                   HVX_VectorPair s_fp16,
                                                   HVX_Vector hf_lut) {
    // u8bec_print(w_int8, "w_int8");
    HVX_VectorPair w_fp16 = HVX_int8_cvt_hf(w_int8, hf_lut);     
    // count_pair_one_num(w_fp16, "w_fp16", 1.0);
    // count_pair_one_num(s_fp16, "s_fp16", 1.0);                                   
    return hvx_vp_mpy_fp16(w_fp16, s_fp16);
}

                            
static inline void flat_to_crouton(const _Float16* flatA, const _Float16* flatB, _Float16* hmx_tile0, _Float16* hmx_tile1) {
    const HVX_Vector v0 = vmem(flatA);
    const HVX_Vector v1 = vmem(flatB);
    const HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

    *((HVX_Vector*)(hmx_tile0)) = Q6_V_lo_W(vp);
    *((HVX_Vector*)(hmx_tile1)) = Q6_V_hi_W(vp);
}


static inline void crouton_to_flat(const _Float16* flatA, const _Float16* flatB, _Float16* hmx_tile0, _Float16* hmx_tile1) {
    const HVX_Vector v0 = vmem(hmx_tile0);
    const HVX_Vector v1 = vmem(hmx_tile1);
    const HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, -2);

    *((HVX_Vector*)(flatA)) = Q6_V_lo_W(vp);
    *((HVX_Vector*)(flatB)) = Q6_V_hi_W(vp);
}

static inline void weight_deconquantization (const uint8_t *wt_tile_base, const _Float16 *s_tile_base, int32_t TileM, int32_t VecM, int32_t HMX_len, int32_t VecK, int TileK, int32_t GroupSize, int32_t Bits, 
                               HVX_Vector LUT1_0, HVX_Vector LUT1_1, bool ZeroPoint, _Float16 *weight_temp, HVX_Vector mask_vec, int8_t shift_len, HVX_Vector hf_lut);

static inline void rm_to_ah(const _Float16* act_tile_base, _Float16* act_temp, const int32_t N, const int32_t K, const int32_t n_dim_tile_num, const int32_t k_dim_tile_num); // convert activation from flat to crouton


static inline void hmx_compute_tiles(
    const _Float16* act_tile_ptr, const _Float16* wt_tile_ptr, _Float16* output_tile_ptr, const _Float16 *scales_off, const _Float16* eye_tile_ptr,
    int32_t k_tiles, int32_t n_tiles, int32_t m_tiles, bool load_output_accumulated_value
){
    int      res  = AEE_SUCCESS;
    int      res2 = AEE_SUCCESS;
    /* ---- Lock HMX (exclusive access to the matrix accelerator) ---- */
    res = hexkl_micro_hmx_lock();
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[MICRO][ERROR] hexkl_micro_hmx_lock failed (%d)", res);
        return ;
    }
    FARF(ALWAYS, "[MICRO] HMX locked -- we have exclusive NPU access");
    
    hmx_set_scales(scales_off);

    for (int32_t n_tile = 0; n_tile < n_tiles; n_tile++) {
        for (int32_t m_tile = 0; m_tile < m_tiles; m_tile++) {
            hmx_clear_acc();
            _Float16 *act_ptr = (_Float16*)(act_tile_ptr + n_tile * k_tiles * 1024);
            _Float16 *wt_ptr  = (_Float16*)(wt_tile_ptr + m_tile * k_tiles * 1024);
            _Float16 *output_ptr = (_Float16*)(output_tile_ptr + (n_tile * m_tiles + m_tile)*1024);

            if (load_output_accumulated_value) {
                hmx_load_tiles((const void*)output_ptr, (const void*)eye_tile_ptr, 1);
            }
            hmx_load_tiles((const void*)act_ptr, (const void*)wt_ptr, k_tiles);
            hmx_store_acc(output_ptr);
        }
    }

    res2 = hexkl_micro_hmx_unlock();
    if (res2 != AEE_SUCCESS) {
        FARF(ALWAYS, "[MICRO][ERROR] hexkl_micro_hmx_unlock failed (%d)", res2);
        res |= res2;
    } else {
        FARF(ALWAYS, "[MICRO] HMX unlocked");
    }
}


template <int GroupSize = 128,
          bool ZeroPoint = false,
          int Bits = 4,
          int g = 4>
int hmx_hvx_tbl_gemm(
          int TileN,
          int TileM,
          int TileK, int32_t GemmM, int32_t GemmK, int32_t GemmN, const _Float16* a, const uint8_t* w, const _Float16 *s, _Float16 *c, 
    _Float16* act_temp, _Float16* weight_temp, _Float16* output_temp, const uint16_t* LUT1, const _Float16* eye_tile, const _Float16* hf_lut_ptr,
    const _Float16* scales_off) {
    /*TODO: 
    act_temp, weight_temp, LUT1, output_temp, eye_tile
    */
    // every fp 16 hmx tile is 32 * 32 * 2B = 2KB

    // int GroupSize = 128;
    // bool ZeroPoint = false;
    // int Bits = 4;
    // int TileK = 256;
    // int g = 4;

    bool load_output_accumulated_value = false;

    const int32_t HMX_len = 32;
    const int32_t VecK = 64;
    const int32_t VecM = 128;
    const int32_t VecN = 32;
    const int8_t mask_4bit = 0b1111;
    const int8_t shift_len = 4;
    const HVX_Vector mask_vec = Q6_Vb_vsplat_R(mask_4bit); 
    
    const int32_t K = GemmK;
    const int32_t M = GemmM;
    const int32_t N = GemmN;


    const int32_t q_group_size = GroupSize;

    // const int32_t TileM = 512 / Bits; // TileM TileN is logical Tile, not memory tile
    // const int32_t TileN = 32;

    memset(c, 0, sizeof(_Float16) * GemmM * GemmN);
    
    HVX_Vector LUT1_0 = vmem(LUT1);
    HVX_Vector LUT1_1 = vmem(LUT1 + VLEN / sizeof(uint16_t));
    HVX_Vector hf_lut = vmem(hf_lut_ptr);
    
    for (int32_t tile_n = 0; tile_n < N; tile_n += TileN) {
        for (int32_t tile_m = 0; tile_m < M; tile_m += TileM) {
            for (int32_t tile_k = 0; tile_k < K; tile_k += TileK) {
                // process one activation tile, load activation and convert flat to crouton 
                const _Float16 *act_tile_base = a + tile_n * K + tile_k;
                rm_to_ah(act_tile_base, act_temp, N, K, TileN, TileK);
                const uint8_t *wt_tile_base = w + (K * Bits / 8) * tile_m + (TileM * Bits / 8) * tile_k;
                const _Float16 *s_tile_base  = s + (K / GroupSize) * tile_m * (1 + ZeroPoint)
                                           + (TileM * (tile_k / GroupSize)) * (1 + ZeroPoint);
                weight_deconquantization(wt_tile_base, s_tile_base, TileM, VecM, HMX_len, VecK, TileK, GroupSize, Bits, LUT1_0, LUT1_1, ZeroPoint, weight_temp, mask_vec, shift_len, hf_lut);
                // hmx compute
                if (tile_k == 0) load_output_accumulated_value = false;
                else load_output_accumulated_value = true;
                hmx_compute_tiles(act_temp, weight_temp, output_temp, scales_off, eye_tile, TileK / HMX_len, TileN / HMX_len, TileM / HMX_len, load_output_accumulated_value);
            }
            // process a output tile, convert crouton to flat and write it back
            _Float16 *c_tile_base = c + tile_n * M + tile_m;
            for (int32_t j = 0; j < TileM / HMX_len; j+=2) {
                for (int32_t row = 0; row < TileN; row+=2) {
                    const _Float16* flatA = c_tile_base +  row    * M + j * HMX_len;
                    const _Float16* flatB = c_tile_base + (row+1) * M + j * HMX_len;
                    _Float16* hmx_tile0 = output_temp + j * 1024     + row * HMX_len;
                    _Float16* hmx_tile1 = output_temp + (j+1) * 1024 + row * HMX_len;
                    crouton_to_flat(flatA, flatB, hmx_tile0, hmx_tile1);
                }
            }
        }
    }
    return 0;
}

static inline void rm_to_ah(const _Float16* act_tile_base, _Float16* act_temp, const int32_t N, const int32_t K, const int32_t n_tile_dim, const int32_t k_tile_dim) {
    const int32_t HMX_len = 32;

    for (int32_t inter_n = 0; inter_n < n_tile_dim; inter_n += 2) {
        for (int32_t inter_k = 0; inter_k < k_tile_dim; inter_k += HMX_len * 2) {
            const int act_tile_0_index = (inter_n / HMX_len) * (k_tile_dim / HMX_len) + (inter_k / HMX_len);
            const int act_tile_1_index = act_tile_0_index + 1;
            const _Float16 *row0 = act_tile_base + inter_n * K + inter_k;
            const _Float16 *row1 = row0 + K;

            int32_t offset = (inter_n % HMX_len) * HMX_len;
            _Float16* act_tile0 = act_temp + act_tile_0_index * (HMX_len * HMX_len) + offset;
            _Float16* act_tile1 = act_tile0 + (HMX_len * HMX_len);

            HVX_Vector vec_row0 = vmem(row0);
            HVX_Vector vec_row1 = vmem(row1);
            HVX_VectorPair vp = Q6_W_vshuff_VVR(vec_row1, vec_row0, -2);
            vmem(act_tile0) = Q6_V_lo_W(vp);
            vmem(act_tile1) = Q6_V_hi_W(vp);
        }
    }
}

static inline void weight_deconquantization (const uint8_t *wt_tile_base, const _Float16 *s_tile_base, int32_t TileM, int32_t VecM, int32_t HMX_len, int32_t VecK, int TileK, int32_t GroupSize, int32_t Bits, 
                               HVX_Vector LUT1_0, HVX_Vector LUT1_1, bool ZeroPoint, _Float16 *weight_temp, HVX_Vector mask_vec, int8_t shift_len, HVX_Vector hf_lut) {
    // a scale tile is (TileK / GroupSize * TileM * sizeof(_Float16)) Bytes
    // col tile is (K / GroupSize * TileM * sizeof(_Float16)) Bytes
    for(int32_t inter_m = 0; inter_m < TileM; inter_m += VecM) {
        for (int32_t group_k = 0; group_k < TileK; group_k += GroupSize){
            const _Float16 *s_ptr = s_tile_base + (TileK / GroupSize) * inter_m + (group_k / GroupSize) * VecM;
            HVX_Vector     s_vec_lo_fp16  = vmem(s_ptr);
            HVX_Vector     s_vec_hi_fp16  = vmem(s_ptr + VLEN / sizeof(_Float16));
            // is there need to be shuff or just need to compose together
            HVX_VectorPair s_pair_lo_fp16 = Q6_W_vshuff_VVR(s_vec_lo_fp16, s_vec_lo_fp16, -2);
            HVX_VectorPair s_pair_hi_fp16 = Q6_W_vshuff_VVR(s_vec_hi_fp16, s_vec_hi_fp16, -2);
            HVX_VectorPair z_pair_lo_fp16, z_pair_hi_fp16;
            if (ZeroPoint) {
                HVX_Vector     z_vec_lo_fp16  = vmem(s_ptr + VecM);
                HVX_Vector     z_vec_hi_fp16  = vmem(s_ptr + VecM + VLEN / sizeof(_Float16));
                // is there need to be shuff or just need to compose together
                z_pair_lo_fp16 = Q6_W_vshuff_VVR(z_vec_lo_fp16, z_vec_lo_fp16, -2);
                z_pair_hi_fp16 = Q6_W_vshuff_VVR(z_vec_hi_fp16, z_vec_hi_fp16, -2);
            }
            for(int32_t inter_k = 0; inter_k < GroupSize; inter_k += 8) { // process a HVX Vector, 128B
                int32_t index_k = group_k + inter_k;
                // FARF(ALWAYS, "[DEBUG] index_k is %d", index_k);
                HVX_VectorPair w_temp_0 = Q6_W_vzero();
                HVX_VectorPair w_temp_1 = Q6_W_vzero();
                HVX_VectorPair w_temp_2 = Q6_W_vzero();
                HVX_VectorPair w_temp_3 = Q6_W_vzero();

                for (int32_t bit = 0; bit < Bits; bit++) {
                    const uint8_t* w_bit_base = wt_tile_base + (TileM * TileK * bit) / 8
                                                + inter_m * (TileK / 8) + index_k / 8 * VecM; 
                    HVX_Vector w_vec = vmem(w_bit_base);

                    HVX_Vector w_vec_bo = Q6_V_vand_VV(w_vec, mask_vec);
                    HVX_Vector w_vec_to = Q6_Vh_vasr_VhR(w_vec, shift_len);
                    
                    HVX_VectorPair c_vec_bo   = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_bo, LUT1_0, bit);
                    HVX_VectorPair c_vec_bo_2 = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_bo, LUT1_1, bit);
                    
                    HVX_VectorPair c_vec_to   = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_to, LUT1_0, bit);
                    HVX_VectorPair c_vec_to_2 = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_to, LUT1_1, bit);
                    
                    HVX_Vector res_low  = Q6_V_vor_VV(Q6_V_lo_W(c_vec_bo), Q6_V_lo_W(w_temp_0));
                    HVX_Vector res_high = Q6_V_vor_VV(Q6_V_hi_W(c_vec_bo), Q6_V_hi_W(w_temp_0));
                    w_temp_0 = Q6_W_vcombine_VV(res_high, res_low);

                    res_low  = Q6_V_vor_VV(Q6_V_lo_W(c_vec_bo_2), Q6_V_lo_W(w_temp_1));
                    res_high = Q6_V_vor_VV(Q6_V_hi_W(c_vec_bo_2), Q6_V_hi_W(w_temp_1));
                    w_temp_1 = Q6_W_vcombine_VV(res_high, res_low);
                    res_low  = Q6_V_vor_VV(Q6_V_lo_W(c_vec_to), Q6_V_lo_W(w_temp_2));
                    res_high = Q6_V_vor_VV(Q6_V_hi_W(c_vec_to), Q6_V_hi_W(w_temp_2));
                    w_temp_2 = Q6_W_vcombine_VV(res_high, res_low);
                    res_low  = Q6_V_vor_VV(Q6_V_lo_W(c_vec_to_2), Q6_V_lo_W(w_temp_3));
                    res_high = Q6_V_vor_VV(Q6_V_hi_W(c_vec_to_2), Q6_V_hi_W(w_temp_3));
                    w_temp_3 = Q6_W_vcombine_VV(res_high, res_low);
                }
                // end_bit_loop:
                //     FARF(ALWAYS, "");
                HVX_VectorPair w_temps[4] = {w_temp_0, w_temp_1, w_temp_2, w_temp_3};
                const int32_t j = ((index_k) % 32) / 8;
                for (int32_t k = 0; k < 4; ++k) {
                    HVX_VectorPair row0, row1;
                    
                    if (ZeroPoint) {
                        row0 = hvx_int8_w_fp16_s_z_mpy(Q6_V_lo_W(w_temps[k]), s_pair_lo_fp16, z_pair_lo_fp16, hf_lut);
                        row1 = hvx_int8_w_fp16_s_z_mpy(Q6_V_hi_W(w_temps[k]), s_pair_hi_fp16, z_pair_hi_fp16, hf_lut);
                    } else {
                        row0 = hvx_int8_w_fp16_s_mpy(Q6_V_lo_W(w_temps[k]), s_pair_lo_fp16, hf_lut);
                        row1 = hvx_int8_w_fp16_s_mpy(Q6_V_hi_W(w_temps[k]), s_pair_hi_fp16, hf_lut);
                    }
                    // process a total weight tile
                    const int32_t index = (index_k) / HMX_len + (inter_m / HMX_len) * VecK / HMX_len;
                    HVX_Vector o[4];
                    o[0] = Q6_V_lo_W(row0);
                    o[1] = Q6_V_hi_W(row0);
                    o[2] = Q6_V_lo_W(row1);
                    o[3] = Q6_V_hi_W(row1);
                    for (int w = 0; w < 4; w++) {
                        _Float16* tile_temp = (weight_temp + (index + TileK / HMX_len * w) * 1024 + j * 256 + k * 64);
                         vmem(tile_temp) = o[w];
                    }
                }
            }
        }
    }
}

template <typename LType,
          typename XType,
          int ActGroupSize = -1, 
          int GroupSize = 0, 
          bool ZeroPoint = false,
          int g = 4>
inline typename std::enable_if_t<std::is_same<LType, int16_t>::value && std::is_same<XType, __fp16>::value, int>
hvx_lut_ctor(int32_t GemmK, int32_t GemmN, const XType *x, LType *l, float *ls, float *lb)
{
    
    UNUSED(GemmN); 
    
    const int32_t Q = GemmK / g; 

    
    const int32_t q_act_group_size = (ActGroupSize < 0) ? (Q / -ActGroupSize) : (ActGroupSize / g);
    
    const int32_t q_group_size     = (GroupSize == 0) ? Q : (GroupSize / g);

    constexpr int32_t lut_size = 16;
    constexpr float max_int16 = 32767.0f;

    
    
    constexpr int32_t VecQ = VLEN / sizeof(XType);

    const HVX_Vector zero_vec = Q6_V_vzero();
    const HVX_Vector ones_vec = Q6_Vh_vsplat_R(0x3C00);  
    const HVX_Vector abs_mask = Q6_Vh_vsplat_R(0x7FFF); 

    
    using lb_t = typename std::conditional<ZeroPoint, __fp16, float>::type;
    lb_t *lb_p = reinterpret_cast<lb_t *>(lb);

    
    XType __attribute__((aligned(VLEN))) tmp_buf[VLEN / sizeof(XType)];

    HVX_Vector lb_val_vec = zero_vec;
    for (int32_t group_q = 0; group_q < Q; group_q += q_act_group_size)
    {
        
        HVX_Vector ls_val_vec = zero_vec;
        for (int32_t q = 0; q < q_act_group_size; q += VecQ)
        {
        const XType *x_base = x + (group_q + q) * g;

        HVX_Vector x0 = vmem(x_base);
        HVX_Vector x1 = vmem(x_base + VecQ);
        HVX_Vector x2 = vmem(x_base + VecQ * 2);
        HVX_Vector x3 = vmem(x_base + VecQ * 3);

        
        
        HVX_VectorPair x01 = Q6_W_vdeal_VVR(x1, x0, -2);
        HVX_VectorPair x23 = Q6_W_vdeal_VVR(x3, x2, -2);
        
        HVX_VectorPair x02 = Q6_W_vdeal_VVR(Q6_V_lo_W(x23), Q6_V_lo_W(x01), -2);
        HVX_VectorPair x13 = Q6_W_vdeal_VVR(Q6_V_hi_W(x23), Q6_V_hi_W(x01), -2);

        
        
        HVX_Vector x0_abs = Q6_V_vand_VV(Q6_V_lo_W(x02), abs_mask);
        HVX_Vector x1_abs = Q6_V_vand_VV(Q6_V_lo_W(x13), abs_mask);
        HVX_Vector x2_abs = Q6_V_vand_VV(Q6_V_hi_W(x02), abs_mask);
        HVX_Vector x3_abs = Q6_V_vand_VV(Q6_V_hi_W(x13), abs_mask);

        
        HVX_Vector x01_abs = Q6_Vqf16_vadd_VhfVhf(x0_abs, x1_abs);
        HVX_Vector x23_abs = Q6_Vqf16_vadd_VhfVhf(x2_abs, x3_abs);
        HVX_Vector sum_abs = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(x01_abs, x23_abs));

        ls_val_vec = Q6_Vhf_vmax_VhfVhf(ls_val_vec, sum_abs);
        }

        
        for (int32_t i = VLEN / 2; i >= 2; i >>= 1)
        {
        ls_val_vec = Q6_Vhf_vmax_VhfVhf(ls_val_vec, Q6_V_vlalign_VVR(ls_val_vec, zero_vec, i));
        }
        vmem(tmp_buf) = ls_val_vec;
        float ls_val = (float)tmp_buf[VLEN / 2 - 1] / max_int16;
        ls[group_q / q_act_group_size] = ls_val;

        float tls_val = ls_val ? 1.0f / ls_val : 0.0f;
        HVX_Vector tls_val_qf32 = Q6_Vqf32_vadd_VsfVsf(Q6_V_vsplat_R(_fp32_to_bits(tls_val)), zero_vec);

        
        
        for (int32_t q = 0; q < q_act_group_size; q += VecQ)
        {
        const XType *x_base = x + (group_q + q) * g;

        HVX_Vector x0 = vmem(x_base);
        HVX_Vector x1 = vmem(x_base + VecQ);
        HVX_Vector x2 = vmem(x_base + VecQ * 2);
        HVX_Vector x3 = vmem(x_base + VecQ * 3);

        
        HVX_VectorPair x01 = Q6_W_vdeal_VVR(x1, x0, -2);
        HVX_VectorPair x23 = Q6_W_vdeal_VVR(x3, x2, -2);
        HVX_VectorPair x02 = Q6_W_vdeal_VVR(Q6_V_lo_W(x23), Q6_V_lo_W(x01), -2);
        HVX_VectorPair x13 = Q6_W_vdeal_VVR(Q6_V_hi_W(x23), Q6_V_hi_W(x01), -2);

        
        HVX_VectorPair x0_qf32 = Q6_Wqf32_vmpy_VhfVhf(Q6_V_lo_W(x02), ones_vec);
        HVX_VectorPair x1_qf32 = Q6_Wqf32_vmpy_VhfVhf(Q6_V_lo_W(x13), ones_vec);
        HVX_VectorPair x2_qf32 = Q6_Wqf32_vmpy_VhfVhf(Q6_V_hi_W(x02), ones_vec);
        HVX_VectorPair x3_qf32 = Q6_Wqf32_vmpy_VhfVhf(Q6_V_hi_W(x13), ones_vec);

        HVX_Vector l_tmp_lo_qf32[lut_size];
        HVX_Vector l_tmp_hi_qf32[lut_size];
    #pragma unroll
        for (int32_t i = 1; i < lut_size; i += 2)
        {
            if (i & 0b0010) {
            l_tmp_lo_qf32[i] = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(x0_qf32), Q6_V_lo_W(x1_qf32));
            l_tmp_hi_qf32[i] = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(x0_qf32), Q6_V_hi_W(x1_qf32));
            } else {
            l_tmp_lo_qf32[i] = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_lo_W(x0_qf32), Q6_V_lo_W(x1_qf32));
            l_tmp_hi_qf32[i] = Q6_Vqf32_vsub_Vqf32Vqf32(Q6_V_hi_W(x0_qf32), Q6_V_hi_W(x1_qf32));
            }
            if (i & 0b0100) {
            l_tmp_lo_qf32[i] = Q6_Vqf32_vadd_Vqf32Vqf32(l_tmp_lo_qf32[i], Q6_V_lo_W(x2_qf32));
            l_tmp_hi_qf32[i] = Q6_Vqf32_vadd_Vqf32Vqf32(l_tmp_hi_qf32[i], Q6_V_hi_W(x2_qf32));
            } else {
            l_tmp_lo_qf32[i] = Q6_Vqf32_vsub_Vqf32Vqf32(l_tmp_lo_qf32[i], Q6_V_lo_W(x2_qf32));
            l_tmp_hi_qf32[i] = Q6_Vqf32_vsub_Vqf32Vqf32(l_tmp_hi_qf32[i], Q6_V_hi_W(x2_qf32));
            }
            if (i & 0b1000) {
            l_tmp_lo_qf32[i] = Q6_Vqf32_vadd_Vqf32Vqf32(l_tmp_lo_qf32[i], Q6_V_lo_W(x3_qf32));
            l_tmp_hi_qf32[i] = Q6_Vqf32_vadd_Vqf32Vqf32(l_tmp_hi_qf32[i], Q6_V_hi_W(x3_qf32));
            } else {
            l_tmp_lo_qf32[i] = Q6_Vqf32_vsub_Vqf32Vqf32(l_tmp_lo_qf32[i], Q6_V_lo_W(x3_qf32));
            l_tmp_hi_qf32[i] = Q6_Vqf32_vsub_Vqf32Vqf32(l_tmp_hi_qf32[i], Q6_V_hi_W(x3_qf32));
            }
        }

        
    #pragma unroll
        for (int32_t i = 0; i < lut_size; i += 2)
        {
            
            l_tmp_lo_qf32[i] = Q6_Vqf32_vsub_Vqf32Vqf32(zero_vec, l_tmp_lo_qf32[lut_size - 1 - i]);
            l_tmp_hi_qf32[i] = Q6_Vqf32_vsub_Vqf32Vqf32(zero_vec, l_tmp_hi_qf32[lut_size - 1 - i]);
        }

        lb_val_vec = Q6_Vqf32_vadd_Vqf32Vqf32(lb_val_vec, l_tmp_lo_qf32[lut_size - 1]);
        lb_val_vec = Q6_Vqf32_vadd_Vqf32Vqf32(lb_val_vec, l_tmp_hi_qf32[lut_size - 1]);

        
        HVX_Vector l_tmp[lut_size];
    #pragma unroll
        for (int32_t i = 0; i < lut_size; i += 1)
        {
            HVX_Vector l_tmp_lo = Q6_Vqf32_vmpy_Vqf32Vqf32(l_tmp_lo_qf32[i], tls_val_qf32);
            HVX_Vector l_tmp_hi = Q6_Vqf32_vmpy_Vqf32Vqf32(l_tmp_hi_qf32[i], tls_val_qf32);
            l_tmp_lo = Q6_Vw_equals_Vsf(Q6_Vsf_equals_Vqf32(l_tmp_lo));
            l_tmp_hi = Q6_Vw_equals_Vsf(Q6_Vsf_equals_Vqf32(l_tmp_hi));
            l_tmp[i] = Q6_Vh_vsat_VwVw(l_tmp_hi, l_tmp_lo);
        }

        
        
        
        HVX_VectorPair l_pa[lut_size / 2];
        HVX_VectorPair l_pb[lut_size / 2];

        
    #pragma unroll
        for (int32_t i = 0; i < lut_size; i += 2)
        {
            l_pa[i / 2] = Q6_W_vshuff_VVR(l_tmp[i + 1], l_tmp[i], -4);
        }
        
    #pragma unroll
        for (int32_t i = 0; i < lut_size / 2; i += 2)
        {
            l_pb[i + 0] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[i + 1]), Q6_V_lo_W(l_pa[i + 0]), -8);
            l_pb[i + 1] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[i + 1]), Q6_V_hi_W(l_pa[i + 0]), -8);
        }
        
    #pragma unroll
        for (int32_t i = 0; i < lut_size / 2; i += 4)
        {
            l_pa[i + 0] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pb[i + 2]), Q6_V_lo_W(l_pb[i + 0]), -16);
            l_pa[i + 1] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pb[i + 2]), Q6_V_hi_W(l_pb[i + 0]), -16);
            l_pa[i + 2] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pb[i + 3]), Q6_V_lo_W(l_pb[i + 1]), -16);
            l_pa[i + 3] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pb[i + 3]), Q6_V_hi_W(l_pb[i + 1]), -16);
        }
        
    #pragma unroll
        for (int32_t i = 0; i < lut_size / 2; i += 8)
        {
            l_pb[i + 0] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[i + 4]), Q6_V_lo_W(l_pa[i + 0]), -32);
            l_pb[i + 1] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[i + 4]), Q6_V_hi_W(l_pa[i + 0]), -32);
            l_pb[i + 2] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[i + 5]), Q6_V_lo_W(l_pa[i + 1]), -32);
            l_pb[i + 3] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[i + 5]), Q6_V_hi_W(l_pa[i + 1]), -32);
            l_pb[i + 4] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[i + 6]), Q6_V_lo_W(l_pa[i + 2]), -32);
            l_pb[i + 5] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[i + 6]), Q6_V_hi_W(l_pa[i + 2]), -32);
            l_pb[i + 6] = Q6_W_vshuff_VVR(Q6_V_lo_W(l_pa[i + 7]), Q6_V_lo_W(l_pa[i + 3]), -32);
            l_pb[i + 7] = Q6_W_vshuff_VVR(Q6_V_hi_W(l_pa[i + 7]), Q6_V_hi_W(l_pa[i + 3]), -32);
        }
        
        LType *l_base = l + (group_q + q) * lut_size;
    #pragma unroll
        for (int32_t i = 0; i < lut_size / 2; i += 1)
        {
            vmem(l_base + (i * 2)     * VLEN / sizeof(LType)) = Q6_V_lo_W(l_pb[i]);
            vmem(l_base + (i * 2 + 1) * VLEN / sizeof(LType)) = Q6_V_hi_W(l_pb[i]);
        }
        if ((q_group_size >= VecQ) && ((group_q + q) % q_group_size == (q_group_size - VecQ)))
        {
            
            for (int32_t i = VLEN / 2; i >= 4; i >>= 1)
            {
            lb_val_vec = Q6_Vqf32_vadd_Vqf32Vqf32(lb_val_vec, Q6_V_vlalign_VVR(lb_val_vec, zero_vec, i));
            }
            vmem(tmp_buf) = Q6_Vsf_equals_Vqf32(lb_val_vec);
            lb_p[(group_q + q) / q_group_size] = -((const float *)tmp_buf)[VLEN / 4 - 1] * 0.5f;
            lb_val_vec = zero_vec;
        }
        if (q_group_size < VecQ)
        {
            
            const int32_t sum_len = VLEN / (VecQ / q_group_size);
            for (int32_t i = sum_len / 2; i >= 4; i >>= 1)
            {
            lb_val_vec = Q6_Vqf32_vadd_Vqf32Vqf32(lb_val_vec, Q6_V_vlalign_VVR(lb_val_vec, zero_vec, i));
            }
            vmem(tmp_buf) = Q6_Vsf_equals_Vqf32(lb_val_vec);
            for (int32_t i = VLEN / 4 - 1; i >= 0; i -= sum_len / 4)
            {
            lb_p[(group_q + q) / q_group_size + i / (sum_len / 4)] = -((const float *)tmp_buf)[i] * 0.5f;
            }
            lb_val_vec = zero_vec;
        }
        }  
    }

    return 0;
}


template <typename LType = int16_t,
          typename XType = __fp16,
          typename CType = float,  
          int ActGroupSize = 256,  
          int GroupSize = 128,
          bool ZeroPoint = false,
          int Bits = 2,
          int TileK = 256,
          int g = 4,
          bool WeightsInVTCM = false>
inline typename std::enable_if_t<std::is_same<LType, int16_t>::value && std::is_same<XType, __fp16>::value && std::is_same<CType, float>::value && (GroupSize > 0), int>
hvx_tbl(int32_t GemmM, int32_t GemmK, int32_t GemmN, const LType *l, const float *ls, const float *lb, const uint8_t *w, const XType *s, CType *c)
{
  UNUSED(GemmN);

  
  constexpr int8_t mask_4bit = 0b1111; 
  
  constexpr int8_t shift_len = 4;
  

   
  const HVX_Vector mask_vec = Q6_Vb_vsplat_R(mask_4bit);
 
 
  const HVX_Vector ones_vec = Q6_Vh_vsplat_R(0x3C00);  
 

  constexpr int32_t lut_size = 16;
  
  constexpr int32_t lut_bytes = lut_size * sizeof(LType); 
  
  
  
  
  
  const int32_t Q = GemmK / g;
  const int32_t P = GemmM * Bits; 

  constexpr int32_t q_group_size     = GroupSize / g;

  constexpr int32_t q_act_group_size = ActGroupSize / g;
  
  constexpr int32_t cmp_blk_size = MIN(GroupSize / g, ActGroupSize / g);
  

  
  
  

  constexpr int32_t VecQ = VLEN / lut_bytes; 
  constexpr int32_t VecP = VLEN / sizeof(uint8_t); 

  constexpr int32_t TileQ = TileK / g; 
  
  const int32_t TileP = P;

  
  static_assert((ActGroupSize % GroupSize == 0) || (GroupSize % ActGroupSize) == 0, "ActGroupSize or GroupSize must be divisible by the other");
  
  static_assert((cmp_blk_size % VecQ == 0), "cmp_blk_size must be divisible by VecQ");
  static_assert((TileQ % cmp_blk_size == 0), "TileQ must be divisible by cmp_blk_size");  
  static_assert((Bits <= 4 && Bits >= 2), "2 <= Bits <= 4 is required");  

  
  HVX_Vector lvec_arr[TileQ / VecQ]; 

  memset(c, 0, sizeof(CType) * TileP); 

  for (int32_t tile_q = 0; tile_q < Q; tile_q += TileQ)
  {
    #pragma unroll
    for (int32_t vec_q = 0; vec_q < TileQ; vec_q += VecQ)
    {
      lvec_arr[vec_q / VecQ] = vmem(l + (tile_q + vec_q) * lut_size); 
    }

    
    
    
    
    constexpr int32_t s_l2fetch_p    = VecP * Bits;
    constexpr int32_t s_l2fetch_size = (s_l2fetch_p / Bits) * (TileQ / q_group_size) * (1 + ZeroPoint);
    constexpr int32_t s_l2fetch_one  = (s_l2fetch_p / Bits) * (1 + ZeroPoint);

    const uint8_t *w_tile_base = w + tile_q * TileP * g / 8; 
    const XType   *s_tile_base = s + (tile_q / q_group_size) * (TileP / Bits) * (1 + ZeroPoint); 

    if (!WeightsInVTCM) {
      l2fetch(s_tile_base + s_l2fetch_one, VLEN, VLEN, (s_l2fetch_size - s_l2fetch_one) * sizeof(XType) / VLEN, 0);
    }

    if (tile_q + TileQ < Q) 
    {
      constexpr int32_t l1cache_line = 64;
      for (int i = (tile_q + TileQ) / q_act_group_size; i < (tile_q + TileQ * 2) / q_act_group_size; i += l1cache_line / sizeof(float))
      {
        Q6_dcfetch_A((void *)(ls + i));
      }
      for (int i = (tile_q + TileQ) / q_group_size; i < (tile_q + TileQ * 2) / q_group_size; i += l1cache_line / sizeof(float))
      {
        Q6_dcfetch_A((void *)(lb + i));
      }
    }

    #pragma unroll(Bits) 
    for (int32_t vec_p = 0; vec_p < TileP; vec_p += VecP) 
    {
      
      
      
      HVX_Vector c_vec_0 = vmem(c + (vec_p +  0));  
      HVX_Vector c_vec_1 = vmem(c + (vec_p + 32));
      HVX_Vector c_vec_2 = vmem(c + (vec_p + 64));
      HVX_Vector c_vec_3 = vmem(c + (vec_p + 96));

      
      HVX_VectorPair c_vec_lo; 
      HVX_VectorPair c_vec_hi;

      const uint8_t *w_base = w_tile_base + vec_p * TileQ * g / 8;
      const XType   *s_base = s_tile_base + vec_p / (VecP * Bits) * (TileQ / q_group_size) * VecP * (1 + ZeroPoint);
      if (!WeightsInVTCM)
      {
        if (vec_p + VecP < TileP)
        {
          l2fetch(w_base + VecP * TileQ * g / 8, VecP, VecP, TileQ * g / 8, 0);
          if (vec_p % s_l2fetch_p == 0)
          {
            l2fetch(s_base + s_l2fetch_size, VLEN, VLEN, s_l2fetch_size * sizeof(XType) / VLEN, 0);
          }
        }
      }

      #pragma unroll
      for (int32_t vec_q = 0; vec_q < TileQ; vec_q += VecQ)
      {
        HVX_Vector w_vec_lo = vmem(w_base + vec_q * VecP * g / 8 + 0);
        HVX_Vector w_vec_hi = vmem(w_base + vec_q * VecP * g / 8 + VLEN);

        HVX_Vector w_vec_lo_bo = Q6_V_vand_VV(w_vec_lo, mask_vec);    
        HVX_Vector w_vec_hi_bo = Q6_V_vand_VV(w_vec_hi, mask_vec);     
        HVX_Vector w_vec_lo_to = Q6_Vh_vasr_VhR(w_vec_lo, shift_len);  
        HVX_Vector w_vec_hi_to = Q6_Vh_vasr_VhR(w_vec_hi, shift_len);  

        
        
        HVX_VectorPair c_vec_lo_bo = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_lo_bo, lvec_arr[vec_q / VecQ], 0);  
        HVX_VectorPair c_vec_hi_bo = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_hi_bo, lvec_arr[vec_q / VecQ], 1);  
        HVX_VectorPair c_vec_lo_to = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_lo_to, lvec_arr[vec_q / VecQ], 2);  
        HVX_VectorPair c_vec_hi_to = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_hi_to, lvec_arr[vec_q / VecQ], 3);  

        
        const bool cmp_blk_head  = (vec_q % cmp_blk_size == 0);
        const bool cmp_blk_tail  = (vec_q % cmp_blk_size == (cmp_blk_size - VecQ));
        const bool q_group_tail  = (vec_q % q_group_size == (q_group_size - VecQ));

        
        
        
        
        if (cmp_blk_head)
        {
          
          c_vec_lo = Q6_Ww_vadd_VhVh(Q6_V_lo_W(c_vec_lo_bo), Q6_V_lo_W(c_vec_hi_bo)); 
          c_vec_hi = Q6_Ww_vadd_VhVh(Q6_V_hi_W(c_vec_lo_bo), Q6_V_hi_W(c_vec_hi_bo));
        }
        else
        {
          c_vec_lo = Q6_Ww_vaddacc_WwVhVh(c_vec_lo, Q6_V_lo_W(c_vec_lo_bo), Q6_V_lo_W(c_vec_hi_bo)); 
          c_vec_hi = Q6_Ww_vaddacc_WwVhVh(c_vec_hi, Q6_V_hi_W(c_vec_lo_bo), Q6_V_hi_W(c_vec_hi_bo));
        }
        c_vec_lo = Q6_Ww_vaddacc_WwVhVh(c_vec_lo, Q6_V_lo_W(c_vec_lo_to), Q6_V_lo_W(c_vec_hi_to));
        c_vec_hi = Q6_Ww_vaddacc_WwVhVh(c_vec_hi, Q6_V_hi_W(c_vec_lo_to), Q6_V_hi_W(c_vec_hi_to)); 

        
        if (cmp_blk_tail)
        {
          const XType *s_ptr = s_base + (vec_q / q_group_size) * VecP * (1 + ZeroPoint);
          
          HVX_Vector s_vec_lo_fp16 = vmem(s_ptr);
          HVX_Vector s_vec_hi_fp16 = vmem(s_ptr + VLEN / sizeof(XType));

          
          HVX_VectorPair s_vec_lo = Q6_Wqf32_vmpy_VhfVhf(s_vec_lo_fp16, ones_vec); 
          
          HVX_VectorPair s_vec_hi = Q6_Wqf32_vmpy_VhfVhf(s_vec_hi_fp16, ones_vec); 

          HVX_Vector ls_vec = Q6_V_vsplat_R(_fp32_to_bits(ls[(tile_q + vec_q) / q_act_group_size]));
          HVX_Vector lb_vec;
          if (ZeroPoint) {
            lb_vec = Q6_Vh_vsplat_R(_fp16_to_bits(reinterpret_cast<const __fp16 *>(lb) + (tile_q + vec_q) / q_group_size));
          } else {
            lb_vec = Q6_V_vsplat_R(_fp32_to_bits(lb[(tile_q + vec_q) / q_group_size]));
          }

          
          
          HVX_Vector c_vec_0_sf = Q6_Vsf_equals_Vw(Q6_V_lo_W(c_vec_lo)); 
          HVX_Vector c_vec_1_sf = Q6_Vsf_equals_Vw(Q6_V_lo_W(c_vec_hi));
          HVX_Vector c_vec_2_sf = Q6_Vsf_equals_Vw(Q6_V_hi_W(c_vec_lo));
          HVX_Vector c_vec_3_sf = Q6_Vsf_equals_Vw(Q6_V_hi_W(c_vec_hi)); 

          
          HVX_Vector c_vec_0_qf32 = Q6_Vqf32_vmpy_VsfVsf(c_vec_0_sf, ls_vec);
          HVX_Vector c_vec_1_qf32 = Q6_Vqf32_vmpy_VsfVsf(c_vec_1_sf, ls_vec);
          HVX_Vector c_vec_2_qf32 = Q6_Vqf32_vmpy_VsfVsf(c_vec_2_sf, ls_vec);
          HVX_Vector c_vec_3_qf32 = Q6_Vqf32_vmpy_VsfVsf(c_vec_3_sf, ls_vec); 

          
          
          if (q_group_tail && (vec_p % (VecP * Bits) == VecP))
          {
            
            
            
            if (ZeroPoint)
            {
              HVX_Vector z_vec_lo_fp16 = vmem(s_ptr + VecP);
              HVX_Vector z_vec_hi_fp16 = vmem(s_ptr + VecP + VLEN / sizeof(XType));

              HVX_VectorPair zlb_vec_lo = Q6_Wqf32_vmpy_VhfVhf(z_vec_lo_fp16, lb_vec);
              HVX_VectorPair zlb_vec_hi = Q6_Wqf32_vmpy_VhfVhf(z_vec_hi_fp16, lb_vec);

              c_vec_0_qf32 = Q6_Vqf32_vadd_Vqf32Vqf32(c_vec_0_qf32, Q6_V_lo_W(zlb_vec_lo));
              c_vec_1_qf32 = Q6_Vqf32_vadd_Vqf32Vqf32(c_vec_1_qf32, Q6_V_lo_W(zlb_vec_hi));
              c_vec_2_qf32 = Q6_Vqf32_vadd_Vqf32Vqf32(c_vec_2_qf32, Q6_V_hi_W(zlb_vec_lo));
              c_vec_3_qf32 = Q6_Vqf32_vadd_Vqf32Vqf32(c_vec_3_qf32, Q6_V_hi_W(zlb_vec_hi));
            }
            else
            {
              c_vec_0_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(c_vec_0_qf32, lb_vec);
              c_vec_1_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(c_vec_1_qf32, lb_vec);
              c_vec_2_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(c_vec_2_qf32, lb_vec);
              c_vec_3_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(c_vec_3_qf32, lb_vec);
            }
          }

          
          c_vec_0_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(c_vec_0_qf32, Q6_V_lo_W(s_vec_lo));
          c_vec_1_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(c_vec_1_qf32, Q6_V_lo_W(s_vec_hi));
          c_vec_2_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(c_vec_2_qf32, Q6_V_hi_W(s_vec_lo));
          c_vec_3_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(c_vec_3_qf32, Q6_V_hi_W(s_vec_hi));

          c_vec_0 = Q6_Vqf32_vadd_Vqf32Vqf32(c_vec_0, c_vec_0_qf32);
          c_vec_1 = Q6_Vqf32_vadd_Vqf32Vqf32(c_vec_1, c_vec_1_qf32);
          c_vec_2 = Q6_Vqf32_vadd_Vqf32Vqf32(c_vec_2, c_vec_2_qf32);
          c_vec_3 = Q6_Vqf32_vadd_Vqf32Vqf32(c_vec_3, c_vec_3_qf32);
        }
      }

      vmem(c + (vec_p +  0)) = c_vec_0;
      vmem(c + (vec_p + 32)) = c_vec_1;
      vmem(c + (vec_p + 64)) = c_vec_2;
      vmem(c + (vec_p + 96)) = c_vec_3; 
    }
  }

  return 0;
}


template <typename LType = int16_t,
          typename XType = __fp16,
          typename CType = float,  
          int ActGroupSize = 256,  
          int GroupSize = 128,
          bool ZeroPoint = false,
          int Bits = 2,
          int TileK = 256,
          int g = 4,
          bool WeightsInVTCM = false>
inline typename std::enable_if_t<std::is_same<LType, int16_t>::value && std::is_same<XType, __fp16>::value && std::is_same<CType, float>::value && GroupSize == 0, int>
hvx_tbl(int32_t GemmM, int32_t GemmK, int32_t GemmN, const LType *l, const float *ls, const float *lb, const uint8_t *w, const XType *s, CType *c)
{
  UNUSED(GemmN);

  
  constexpr int8_t mask_4bit = 0b1111;
  constexpr int8_t shift_len = 4;

  const HVX_Vector mask_vec = Q6_Vb_vsplat_R(mask_4bit);
  const HVX_Vector ones_vec = Q6_Vh_vsplat_R(0x3C00);  

  constexpr int32_t lut_size = 16;
  constexpr int32_t lut_bytes = lut_size * sizeof(LType);
  
  
  
  
  
  const int32_t Q = GemmK / g;
  const int32_t P = GemmM * Bits;

  constexpr int32_t VecQ = VLEN / lut_bytes;
  constexpr int32_t VecP = VLEN / sizeof(uint8_t);

  constexpr int32_t TileQ = TileK / g;
  
  const int32_t TileP = P;

  
  static_assert((ActGroupSize == -1), "For BitNet model, only per-tensor quantization is supported");
  static_assert(!ZeroPoint, "For BitNet model, the quantization should be symmetric");
  
  static_assert((Bits <= 4 && Bits >= 2), "2 <= Bits <= 4 is required");  

  
  HVX_Vector lvec_arr[TileQ / VecQ];

  memset(c, 0, sizeof(CType) * TileP);

  for (int32_t tile_q = 0; tile_q < Q; tile_q += TileQ)
  {
#pragma unroll
    for (int32_t vec_q = 0; vec_q < TileQ; vec_q += VecQ)
    {
      lvec_arr[vec_q / VecQ] = vmem(l + (tile_q + vec_q) * lut_size);
    }

    const uint8_t *w_tile_base = w + tile_q * TileP * g / 8;

#pragma unroll(Bits)
    for (int32_t vec_p = 0; vec_p < TileP; vec_p += VecP)
    {
      
      
      
      HVX_Vector c_vec_0 = vmem(c + (vec_p +  0));
      HVX_Vector c_vec_1 = vmem(c + (vec_p + 32));
      HVX_Vector c_vec_2 = vmem(c + (vec_p + 64));
      HVX_Vector c_vec_3 = vmem(c + (vec_p + 96));

      
      HVX_VectorPair c_vec_lo = Q6_W_vcombine_VV(c_vec_2, c_vec_0);
      HVX_VectorPair c_vec_hi = Q6_W_vcombine_VV(c_vec_3, c_vec_1);

      const uint8_t *w_base = w_tile_base + vec_p * TileQ * g / 8;
      if (!WeightsInVTCM)
      {
        if (vec_p + VecP < TileP)
        {
          l2fetch(w_base + VecP * TileQ * g / 8, VecP, VecP, TileQ * g / 8, 0);
        }
      }

#pragma unroll
      for (int32_t vec_q = 0; vec_q < TileQ; vec_q += VecQ)
      {
        HVX_Vector w_vec_lo = vmem(w_base + vec_q * VecP * g / 8 + 0);
        HVX_Vector w_vec_hi = vmem(w_base + vec_q * VecP * g / 8 + VLEN);

        HVX_Vector w_vec_lo_bo = Q6_V_vand_VV(w_vec_lo, mask_vec);     
        HVX_Vector w_vec_hi_bo = Q6_V_vand_VV(w_vec_hi, mask_vec);     
        HVX_Vector w_vec_lo_to = Q6_Vh_vasr_VhR(w_vec_lo, shift_len);  
        HVX_Vector w_vec_hi_to = Q6_Vh_vasr_VhR(w_vec_hi, shift_len);  

        
        
        HVX_VectorPair c_vec_lo_bo = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_lo_bo, lvec_arr[vec_q / VecQ], 0);  
        HVX_VectorPair c_vec_hi_bo = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_hi_bo, lvec_arr[vec_q / VecQ], 1);  
        HVX_VectorPair c_vec_lo_to = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_lo_to, lvec_arr[vec_q / VecQ], 2);  
        HVX_VectorPair c_vec_hi_to = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_hi_to, lvec_arr[vec_q / VecQ], 3);  

        
        
        
        
        c_vec_lo = Q6_Ww_vaddacc_WwVhVh(c_vec_lo, Q6_V_lo_W(c_vec_lo_bo), Q6_V_lo_W(c_vec_hi_bo));
        c_vec_hi = Q6_Ww_vaddacc_WwVhVh(c_vec_hi, Q6_V_hi_W(c_vec_lo_bo), Q6_V_hi_W(c_vec_hi_bo));
        c_vec_lo = Q6_Ww_vaddacc_WwVhVh(c_vec_lo, Q6_V_lo_W(c_vec_lo_to), Q6_V_lo_W(c_vec_hi_to));
        c_vec_hi = Q6_Ww_vaddacc_WwVhVh(c_vec_hi, Q6_V_hi_W(c_vec_lo_to), Q6_V_hi_W(c_vec_hi_to));
      }

      vmem(c + (vec_p +  0)) = Q6_V_lo_W(c_vec_lo);
      vmem(c + (vec_p + 32)) = Q6_V_lo_W(c_vec_hi);
      vmem(c + (vec_p + 64)) = Q6_V_hi_W(c_vec_lo);
      vmem(c + (vec_p + 96)) = Q6_V_hi_W(c_vec_hi);
    }
  }

  HVX_Vector ls_vec = Q6_V_vsplat_R(_fp32_to_bits(ls[0]));
  HVX_Vector lb_vec = Q6_V_vsplat_R(_fp32_to_bits(lb[0]));
  HVX_Vector s_vec = Q6_V_lo_W(Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vsplat_R(_fp16_to_bits(s)), ones_vec));
#pragma unroll(Bits)
  for (int32_t vec_p = 0; vec_p < TileP; vec_p += VecP)
  {
#pragma unroll
    for (int32_t vec_c = vec_p; vec_c < vec_p + VecP; vec_c += VecP / sizeof(CType))
    {
      HVX_Vector c_vec = vmem(c + vec_c);
      c_vec = Q6_Vsf_equals_Vw(c_vec);
      c_vec = Q6_Vqf32_vmpy_VsfVsf(c_vec, ls_vec);     
      if (vec_p % (VecP * Bits) == VecP)
      {
        c_vec = Q6_Vqf32_vadd_Vqf32Vsf(c_vec, lb_vec); 
      }
      c_vec = Q6_Vqf32_vmpy_Vqf32Vqf32(c_vec, s_vec);  
      vmem(c + vec_c) = c_vec;
    }
  }

  return 0;
}

template <typename XType = __fp16,
          typename CType = float,  
          int Bits = 2>
inline typename std::enable_if_t<std::is_same<XType, __fp16>::value && std::is_same<CType, float>::value, int>
hvx_bit_serial(int32_t GemmM, int32_t GemmN, const CType *c, XType *y)
{
  UNUSED(GemmN);

  const int32_t P = GemmM * Bits;

  constexpr int32_t VecP = VLEN / sizeof(uint8_t);
  
  const int32_t TileP = P;

  static_assert((Bits <= 4 && Bits >= 2), "2 <= Bits <= 4 is required");  

  
  const HVX_Vector f0_5_vec = Q6_V_vsplat_R(0x4000007e);  
  const HVX_Vector f2_0_vec = Q6_V_vsplat_R(0x40000080);  
  const HVX_Vector f4_0_vec = Q6_V_vsplat_R(0x40000081);  

  for (int32_t vec_p = 0; vec_p < TileP; vec_p += VecP * Bits)
  {
    
    HVX_Vector c_bits[Bits * 4];
#pragma unroll
    for (int32_t b = 0; b < Bits * 4; b++)
    {
      c_bits[b] = vmem(c + (vec_p + b * 32));
    }

#pragma unroll
    for (int32_t i = 0; i < 4; i++)
    {
      c_bits[i] = Q6_Vqf32_vmpy_Vqf32Vqf32(c_bits[i], f0_5_vec);
    }
    if (Bits >= 2)
    {
#pragma unroll
      for (int32_t i = 0; i < 4; i++)
      {
        c_bits[i] = Q6_Vqf32_vadd_Vqf32Vqf32(c_bits[i], c_bits[i + 4]);
      }
    }
    if (Bits >= 3)
    {
#pragma unroll
      for (int32_t i = 0; i < 4; i++)
      {
        c_bits[i + 8] = Q6_Vqf32_vmpy_Vqf32Vqf32(c_bits[i + 8], f2_0_vec);
        c_bits[i] = Q6_Vqf32_vadd_Vqf32Vqf32(c_bits[i], c_bits[i + 8]);
      }
    }
    if (Bits == 4)
    {
#pragma unroll
      for (int32_t i = 0; i < 4; i++)
      {
        c_bits[i + 12] = Q6_Vqf32_vmpy_Vqf32Vqf32(c_bits[i + 12], f4_0_vec);
        c_bits[i] = Q6_Vqf32_vadd_Vqf32Vqf32(c_bits[i], c_bits[i + 12]);
      }
    }
    
    HVX_Vector c_bitsum_lo = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(c_bits[2], c_bits[0])); 
    HVX_Vector c_bitsum_hi = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(c_bits[3], c_bits[1])); 
    vmem(y + vec_p / Bits +  0) = c_bitsum_lo;
    vmem(y + vec_p / Bits + 64) = c_bitsum_hi;
  }

  return 0;
}


#undef _Float16