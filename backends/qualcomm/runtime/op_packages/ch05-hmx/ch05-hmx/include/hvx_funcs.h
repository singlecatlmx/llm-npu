#pragma once

#include <hexagon_types.h>
#include <string.h>
#include <stdint.h>
#include "hmx_utils.h"
#include <stdbool.h>
#include <math.h>

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  ((byte) & 0x80 ? '1' : '0'), \
  ((byte) & 0x40 ? '1' : '0'), \
  ((byte) & 0x20 ? '1' : '0'), \
  ((byte) & 0x10 ? '1' : '0'), \
  ((byte) & 0x08 ? '1' : '0'), \
  ((byte) & 0x04 ? '1' : '0'), \
  ((byte) & 0x02 ? '1' : '0'), \
  ((byte) & 0x01 ? '1' : '0')

#define UNUSED(x) (void)(x)

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

void hfvec_print(HVX_Vector a, char* name) {
    _Float16 *out = (_Float16*)debug_ptr;
    *(HVX_Vector*)out = a;
    FARF(ALWAYS, "============ [DEBUG] ============ ");
    for (int i = 0; i < 64; i++)
    FARF(ALWAYS, "[DEBUG] %s[%d] = %.1f", name, i, (float)out[i]);
}

void hvec_print(HVX_Vector a, char* name) {
    uint16_t* out = (uint16_t*) debug_ptr;
    *(HVX_Vector*)out = a;
    FARF(ALWAYS, "============ [DEBUG] ============ ");
    for (int i = 0; i < 64; i++)
    FARF(ALWAYS, "[DEBUG] %s[%d] = %d", name, i, (uint16_t)out[i]);
}

void bit_hvec_print(HVX_Vector a, char* name){
    uint16_t *out = (uint16_t*)debug_ptr;
    *(HVX_Vector*)out = a;
    FARF(ALWAYS, "============ [DEBUG BINARY] ============ ");
    for (int i = 0; i < 64; i++) {
        uint16_t val = out[i];
        FARF(ALWAYS, "[DEBUG] %s[%d] = " BYTE_TO_BINARY_PATTERN BYTE_TO_BINARY_PATTERN, 
            name, i, BYTE_TO_BINARY(val >> 8), BYTE_TO_BINARY(val));
    }
}


void u8bec_print(HVX_Vector a, char *name) {
    uint8_t *out = (uint8_t*) debug_ptr;
    *(HVX_Vector*)out = a;
    FARF(ALWAYS, "============ [DEBUG] ============ ");
    for (int i = 0; i < 128; i++)
    FARF(ALWAYS, "[DEBUG] %s[%d] = %d", name, i, (uint8_t)out[i]);
}

void count_one_num(HVX_Vector a, char *name, float num) {
    int32_t count = 0;
    _Float16* out = (_Float16*) debug_ptr;
    *(HVX_Vector*)out = a;
    for (int i = 0; i < 64; i++) {
        if (((float)out[i] - num <= 0.01 && (float)out[i] - num >= -0.01)) count++;
    }
    FARF(ALWAYS, "[DEBUG] %s count one is %d", name, count);
}

void count_pair_one_num(HVX_VectorPair a, char *name, float num) {
    int32_t count = 0;
    _Float16* out = (_Float16*) debug_ptr;
    *(HVX_Vector*)out = Q6_V_hi_W(a);
    for (int i = 0; i < 64; i++) {
        if (((float)out[i] - num <= 0.01 && (float)out[i] - num >= -0.01)) count++;
    }
    *(HVX_Vector*)out = Q6_V_lo_W(a);
    for (int i = 0; i < 64; i++) {
        if (((float)out[i] - num <= 0.01 && (float)out[i] - num >= -0.01)) count++;
    }
    FARF(ALWAYS, "[DEBUG] %s count one is %d", name, count);
}

void count_pair_one_num_fixed(HVX_VectorPair a, char *name, uint8_t target) {
    int32_t count = 0;
    uint8_t *out = (uint8_t*)debug_ptr;
    *(HVX_Vector*)out = Q6_V_hi_W(a);
    for (int i = 0; i < 128; i++) {
        if ((uint8_t)out[i] == (uint8_t)target) count++;
    }
    *(HVX_Vector*)out = Q6_V_lo_W(a);
    for (int i = 0; i < 128; i++) {
        if ((uint8_t)out[i] == (uint8_t)target) count++;
    }
    FARF(ALWAYS, "[DEBUG] %s count one is %d", name, count);
}

void count_vec_one_num(HVX_Vector a, char *name, float num) {
    int32_t count = 0;
    _Float16 *out = (_Float16*) debug_ptr;
    *(HVX_Vector*)out = a;
    for (int i = 0; i < 64; i++) {
        if (((float)out[i] - num <= 0.0001 && (float)out[i] - num >= -0.0001)) count++;
    }
    FARF(ALWAYS, "[DEBUG] %s count one is %d", name, count);
}

static inline HVX_VectorPair HVX_int8_cvt_hf(HVX_Vector w_int8, HVX_Vector hf_lut){
    HVX_VectorPair w_hf = Q6_Wh_vlut16_VbVhR_nomatch(w_int8, hf_lut, 0);
    HVX_VectorPair result = Q6_W_vshuff_VVR(Q6_V_hi_W(w_hf), Q6_V_lo_W(w_hf), -2);
    return result;
}


#define HVX_VP_APPLY2(op, a, b) \
    Q6_W_vcombine_VV(op(Q6_V_hi_W(a), Q6_V_hi_W(b)), \
                     op(Q6_V_lo_W(a), Q6_V_lo_W(b)))


static inline HVX_VectorPair hvx_vp_mpy_fp16(HVX_VectorPair a, HVX_VectorPair b) {
    return HVX_VP_APPLY2(Q6_Vhf_vmpy_VhfVhf, a, b);
}

static inline HVX_VectorPair hvx_vp_add_fp16(HVX_VectorPair a, HVX_VectorPair b) {
    return HVX_VP_APPLY2(Q6_Vhf_vadd_VhfVhf, a, b);
}

/**
 * ????: (cvt(w_int8) + z_fp16) * s_fp16
 */
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


static inline void flat_to_crouton(_Float16* flatA, _Float16* flatB, _Float16* hmx_tile0, _Float16* hmx_tile1) {
    const HVX_Vector v0 = vmem(flatA);
    const HVX_Vector v1 = vmem(flatB);
    const HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

    *((HVX_Vector*)(hmx_tile0)) = Q6_V_lo_W(vp);
    *((HVX_Vector*)(hmx_tile1)) = Q6_V_hi_W(vp);
}


static inline void crouton_to_flat(_Float16* flatA, _Float16* flatB, _Float16* hmx_tile0, _Float16* hmx_tile1) {
    const HVX_Vector v0 = vmem(hmx_tile0);
    const HVX_Vector v1 = vmem(hmx_tile1);
    const HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, -2);

    *((HVX_Vector*)(flatA)) = Q6_V_lo_W(vp);
    *((HVX_Vector*)(flatB)) = Q6_V_hi_W(vp);
}


void weight_deconquantization (uint8_t *wt_tile_base, _Float16 *s_tile_base, int32_t TileM, int32_t VecM, int32_t HMX_len, int32_t VecK, int TileK, int32_t GroupSize, int32_t Bits, 
                               HVX_Vector LUT1_0, HVX_Vector LUT1_1, bool ZeroPoint, _Float16 *weight_temp, HVX_Vector mask_vec, int8_t shift_len, HVX_Vector hf_lut);

void rm_to_ah(_Float16* act_tile_base, _Float16* act_temp, const int32_t N, const int32_t K, const int32_t n_dim_tile_num, const int32_t k_dim_tile_num); // convert activation from flat to crouton
void ah_to_rm(); // convert output activation from crouton to flat

static inline void hmx_compute_tiles(
    _Float16* act_tile_ptr, _Float16* wt_tile_ptr, _Float16* output_tile_ptr, _Float16 *scales_off, _Float16* eye_tile_ptr,
    int32_t k_tiles, int32_t n_tiles, int32_t m_tiles, bool load_output_accumulated_value
){
    // FARF(ALWAYS, "[DEBUG] act: %p, wt: %p, out: %p", act_tile_ptr, wt_tile_ptr, output_tile_ptr);
    // int weight_tile_count_one = 0;
    // int act_tile_count_one = 0;
    // for (int i = 0; i < 256 * 128; i++) {
    //     if (fabsf((float)wt_tile_ptr[i] - 1.0f) < 1e-3f) weight_tile_count_one++;
    //     else {
    //         FARF(ALWAYS, "[DEBUG] wt_tile_ptr[%i] = %.8f", i, (float)wt_tile_ptr[i]);
    //     }
    // }
    // FARF(ALWAYS, "[DEBUG] wt_tile_ptr[0]")
    // for (int i = 0; i < 32 * 256; i++) {
    //     if (fabsf((float)act_tile_ptr[i] - 1.0f) < 1e-3f) act_tile_count_one++;
    // }
    // FARF(ALWAYS, "[DEBUG] weight_tile_count_one = %d  act_tile_count_one = %d", weight_tile_count_one, act_tile_count_one);


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
    // for (int i = 0; i < 128; i+=16) {
    //     FARF(ALWAYS, "[DEBUG] scales_off[%d] = %.8f", i, (float)scales_off[i]);
    // }

    for (int32_t n_tile = 0; n_tile < n_tiles; n_tile++) {
        for (int32_t m_tile = 0; m_tile < m_tiles; m_tile++) {
            hmx_clear_acc();
            _Float16 *act_ptr = (_Float16*)(act_tile_ptr + n_tile * k_tiles * 1024);
            _Float16 *wt_ptr  = (_Float16*)(wt_tile_ptr + m_tile * k_tiles * 1024);
            _Float16 *output_ptr = (_Float16*)(output_tile_ptr + (n_tile * m_tiles + m_tile)*1024);

            int weight_count_one = 0;
            int weight_count_none_one = 0;
            int act_count_one = 0;
            for (int i = 0; i < k_tiles * 1024; i++) {
                if (fabsf((float)wt_ptr[i] - 1.0f) < 1e-3f) weight_count_one++;
                else weight_count_none_one++;
                if (fabsf((float)act_ptr[i] - 1.0f) < 1e-3f) act_count_one++;
            }
            // FARF(ALWAYS, "[DEBUG] weight_count_one = %d  act_count_one = %d weight_count_none_one = %d" , weight_count_one, act_count_one, weight_count_none_one);
            // goto end_loop;

            if (load_output_accumulated_value) {
                hmx_load_tiles((const void*)output_ptr, (const void*)eye_tile_ptr, 1);
            }
            hmx_load_tiles((const void*)act_ptr, (const void*)wt_ptr, k_tiles);
            hmx_store_acc(output_ptr);
        }
    }
    end_loop:
        FARF(ALWAYS, "");

    // Q6_Vh_vsplat_R(0x3C00) Q6_V_vzero()

    res2 = hexkl_micro_hmx_unlock();
    if (res2 != AEE_SUCCESS) {
        FARF(ALWAYS, "[MICRO][ERROR] hexkl_micro_hmx_unlock failed (%d)", res2);
        res |= res2;
    } else {
        FARF(ALWAYS, "[MICRO] HMX unlocked");
    }

    // for (int i = 0; i < 1024; i+=1) {
    //     FARF(ALWAYS, "[DEBUG] output_tile_ptr[%d] = %.8f", i, (float)output_tile_ptr[i]);
    // }
}


int hmx_hvx_tbl_gemm(int32_t GemmM, int32_t GemmK, int32_t GemmN, const _Float16* a, const uint8_t* w, const _Float16 *s, _Float16 *c, 
    _Float16* act_temp, _Float16* weight_temp, _Float16* output_temp, const uint16_t* LUT1, const _Float16* eye_tile, const _Float16* hf_lut_ptr,
    _Float16* scales_off) {
    /*TODO: 
    act_temp, weight_temp, LUT1, output_temp, eye_tile
    */
    // every fp 16 hmx tile is 32 * 32 * 2B = 2KB


    int GroupSize = 128;
    bool ZeroPoint = false;
    int Bits = 4;
    int TileK = 256;
    int g = 4;

    bool load_output_accumulated_value = false;

    const int32_t num_nTileKs = 2;
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

    const int32_t TileM = 512 / Bits; // TileM TileN is logical Tile, not memory tile
    const int32_t TileN = 32;

    const int32_t nTileK = num_nTileKs * TileK; 
    
    memset(c, 0, sizeof(_Float16) * GemmM * GemmN);
    
    HVX_Vector LUT1_0 = vmem(LUT1);
    HVX_Vector LUT1_1 = vmem(LUT1 + VLEN / sizeof(uint16_t));

    HVX_Vector hf_lut = vmem(hf_lut_ptr);

    
    for (int32_t tile_n = 0; tile_n < N; tile_n += TileN) {
        for (int32_t tile_m = 0; tile_m < M; tile_m += TileM) {
            for (int32_t tile_k = 0; tile_k < K; tile_k += TileK) {
                // process one activation tile, load activation and convert flat to crouton 
                _Float16 *act_tile_base = a + tile_n * K + tile_k;
                rm_to_ah(act_tile_base, act_temp, N, K, TileN, TileK);
                // FARF(ALWAYS, "[DEBUG] activation[0] = %.1f", (float)act_temp[3]);
                // process one weight tile, load weight, then (1) dequantization(already finished flat to crouton convertion) (2) hmx tile matrix multiplication
                // a weight tile is (TileK * TileM * Bits / 8)  Bytes
                // every col tile is (TileM * Bits * K / 8) Bytes
                // this is (tile_m / TileM, tile_k / TileK) tile
                uint8_t *wt_tile_base = w + (K * Bits / 8) * tile_m + (TileM * Bits / 8) * tile_k;
                // a scale tile is (TileK / GroupSize * TileM * sizeof(_Float16)) Bytes
                // col tile is (K / GroupSize * TileM * sizeof(_Float16)) Bytes
                _Float16 *s_tile_base  = s + (K / GroupSize) * tile_m * (1 + ZeroPoint)
                                           + (TileM * (tile_k / GroupSize)) * (1 + ZeroPoint);
                weight_deconquantization(wt_tile_base, s_tile_base, TileM, VecM, HMX_len, VecK, TileK, GroupSize, Bits, LUT1_0, LUT1_1, ZeroPoint, weight_temp, mask_vec, shift_len, hf_lut);
                // FARF(ALWAYS, "[DEBUG] weight[0] = %.1f", (float)weight_temp[1024]);
                // hmx compute
                if (tile_k == 0) load_output_accumulated_value = false;
                else load_output_accumulated_value = true;
                hmx_compute_tiles(act_temp, weight_temp, output_temp, scales_off, eye_tile, TileK / HMX_len, TileN / HMX_len, TileM / HMX_len, load_output_accumulated_value);
                // goto end_loop;

                // for (int i = 0; i < 1024; i++) {
                //     FARF(ALWAYS, "[DEBUG] output_temp[%d] = %.1f", i, (float)output_temp[i]);
                // }
                // goto end_loop;
            }
            // for (int i = 0; i < 1024; i++) {
            //     FARF(ALWAYS, "[DEBUG] output_temp[%d] = %.1f", i, (float)output_temp[i]);
            // }
            // process a output tile, convert crouton to flat and write it back
            
            _Float16 *c_tile_base = c + tile_n * M + tile_m;
            for (int32_t j = 0; j < TileM / HMX_len; j+=2) {
                for (int32_t row = 0; row < TileN; row+=2) {
                    _Float16* flatA = c_tile_base +  row    * M + j * HMX_len;
                    _Float16* flatB = c_tile_base + (row+1) * M + j * HMX_len;
                    _Float16* hmx_tile0 = output_temp + j * 1024     + row * HMX_len;
                    _Float16* hmx_tile1 = output_temp + (j+1) * 1024 + row * HMX_len;
                    crouton_to_flat(flatA, flatB, hmx_tile0, hmx_tile1);
                }
            }
        }
    }
    end_loop:
        FARF(ALWAYS,"");
    return 0;
}

void rm_to_ah(_Float16* act_tile_base, _Float16* act_temp, const int32_t N, const int32_t K, const int32_t n_tile_dim, const int32_t k_tile_dim) {
    const int32_t HMX_len = 32;

    for (int32_t inter_n = 0; inter_n < n_tile_dim; inter_n += 2) {
        for (int32_t inter_k = 0; inter_k < k_tile_dim; inter_k += HMX_len * 2) {
            const int act_tile_0_index = (inter_n / HMX_len) * (k_tile_dim / HMX_len) + (inter_k / HMX_len);
            const int act_tile_1_index = act_tile_0_index + 1;
            _Float16 *row0 = act_tile_base + inter_n * K + inter_k;
            _Float16 *row1 = row0 + K;

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

void ah_to_rm(){

}

void weight_deconquantization (uint8_t *wt_tile_base, _Float16 *s_tile_base, int32_t TileM, int32_t VecM, int32_t HMX_len, int32_t VecK, int TileK, int32_t GroupSize, int32_t Bits, 
                               HVX_Vector LUT1_0, HVX_Vector LUT1_1, bool ZeroPoint, _Float16 *weight_temp, HVX_Vector mask_vec, int8_t shift_len, HVX_Vector hf_lut) {
    // a scale tile is (TileK / GroupSize * TileM * sizeof(_Float16)) Bytes
    // col tile is (K / GroupSize * TileM * sizeof(_Float16)) Bytes
    for(int32_t inter_m = 0; inter_m < TileM; inter_m += VecM) {
        for (int32_t group_k = 0; group_k < TileK; group_k += GroupSize){
            _Float16 *s_ptr = s_tile_base + (TileK / GroupSize) * inter_m + (group_k / GroupSize) * VecM;
            HVX_Vector     s_vec_lo_fp16  = vmem(s_ptr);
            HVX_Vector     s_vec_hi_fp16  = vmem(s_ptr + VLEN / sizeof(_Float16));

            // // scales and zeros is right
            // hfvec_print(s_vec_lo_fp16, "s_vec_lo_fp16");
            // goto end_loop;
            // break;
            // FARF(ALWAYS, "[DEBUG] zeropoint is %d", ZeroPoint);
            // break;
            // count_one_num(s_vec_lo_fp16, "s_vec_lo_fp16", 1.0);
            // count_one_num(s_vec_hi_fp16, "s_vec_hi_fp16", 1.0);

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

            // count_pair_one_num(s_pair_lo_fp16, "s_pair_lo_fp16", 1.0);
            // count_pair_one_num(s_pair_hi_fp16, "s_pair_hi_fp16", 1.0);

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
                    // FARF(ALWAYS, "[DEBUG] bit base addr %d", (TileM * TileK * bit) / 8 + inter_m * (TileK / 8) + index_k / 8 * VecM);
                    // HVX_Vector w_vec_shuff = vmem(w_bit_base);
                    // HVX_Vector w_vec = Q6_Vb_vdeal_Vb(w_vec_shuff);
                    // do not need shuff, because  Q6_Wh_vlut16_VbVhR_nomatch needs shuffled input
                    HVX_Vector w_vec = vmem(w_bit_base);

                    // u8bec_print(w_vec, "w_vec");

                    HVX_Vector w_vec_bo = Q6_V_vand_VV(w_vec, mask_vec);
                    HVX_Vector w_vec_to = Q6_Vh_vasr_VhR(w_vec, shift_len);
                    
                    // if (bit == 1) {
                    //     u8bec_print(w_vec_bo, "w_vec_bo");
                    //     // hvec_print(LUT1_0, "LUT1_0");
                    //     u8bec_print(w_vec_to, "w_vec_to");
                    // }
                    // goto end_bit_loop;
                    // goto end_loop;
                    
                    HVX_VectorPair c_vec_bo   = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_bo, LUT1_0, bit);
                    // if (bit == 3) {
                    //     bit_hvec_print(Q6_V_lo_W(c_vec_bo), "c_vec_bo");
                    // }
                    // goto end_bit_loop;
                    HVX_VectorPair c_vec_bo_2 = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_bo, LUT1_1, bit);
                    // bit_hvec_print(Q6_V_lo_W(c_vec_bo_2), "c_vec_bo");
                    // goto end_bit_loop;
                    // output the normal seq
                    
                    HVX_VectorPair c_vec_to   = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_to, LUT1_0, bit);
                    HVX_VectorPair c_vec_to_2 = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_to, LUT1_1, bit);

                    // if (bit == 1) {
                    //     bit_hvec_print(Q6_V_lo_W(c_vec_to_2), "c_vec_to_2");
                    // }
                    
                    HVX_Vector res_low  = Q6_V_vor_VV(Q6_V_lo_W(c_vec_bo), Q6_V_lo_W(w_temp_0));
                    HVX_Vector res_high = Q6_V_vor_VV(Q6_V_hi_W(c_vec_bo), Q6_V_hi_W(w_temp_0));
                    w_temp_0 = Q6_W_vcombine_VV(res_high, res_low);
                    // if (bit == 3) {
                    //     bit_hvec_print(Q6_V_lo_W(w_temp_0), "w_temp_0_lo");
                    // }
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
                end_bit_loop:
                    FARF(ALWAYS, "");

                // u8bec_print(Q6_V_lo_W(w_temp_0), "w_temp_0"); // should be all one 
                // u8bec_print(Q6_V_lo_W(w_temp_1), "w_temp_1"); // should be all one 
                // u8bec_print(Q6_V_lo_W(w_temp_2), "w_temp_2"); // should be all one 
                // u8bec_print(Q6_V_lo_W(w_temp_3), "w_temp_3"); // should be all one 
                // goto end_loop;

                
                HVX_VectorPair w_temps[4] = {w_temp_0, w_temp_1, w_temp_2, w_temp_3};
                // count_pair_one_num_fixed(w_temps[0], "w_temp0", 1);
                // count_pair_one_num_fixed(w_temps[1], "w_temp1", 1);
                // count_pair_one_num_fixed(w_temps[2], "w_temp2", 1);
                // count_pair_one_num_fixed(w_temps[3], "w_temp3", 1);
                const int32_t j = ((index_k) % 32) / 8;
                // FARF(ALWAYS, "[DEBUG] row is %d", index_k);
                for (int32_t k = 0; k < 4; ++k) {
                    HVX_VectorPair row0, row1;
                    
                    if (ZeroPoint) {
                        row0 = hvx_int8_w_fp16_s_z_mpy(Q6_V_lo_W(w_temps[k]), s_pair_lo_fp16, z_pair_lo_fp16, hf_lut);
                        row1 = hvx_int8_w_fp16_s_z_mpy(Q6_V_hi_W(w_temps[k]), s_pair_hi_fp16, z_pair_hi_fp16, hf_lut);
                    } else {
                        row0 = hvx_int8_w_fp16_s_mpy(Q6_V_lo_W(w_temps[k]), s_pair_lo_fp16, hf_lut);
                        row1 = hvx_int8_w_fp16_s_mpy(Q6_V_hi_W(w_temps[k]), s_pair_hi_fp16, hf_lut);
                        // count_pair_one_num_fixed(w_temps[k], "w_temps", 1);
                        // count_pair_one_num(s_pair_lo_fp16, "s_pair_lo_fp16", 1.0);
                        // count_pair_one_num(row0, "row0", 1.0);
                        // count_pair_one_num(row1, "row1", 1.0);
                    }
                    // FARF(ALWAYS, "[DEBUG] row is %d", index_k + k * 2);
                    // count_pair_one_num(row0, "row0", 1.0);
                    // count_pair_one_num(row1, "row1", 1.0);
                    // goto end_loop;
                    // hfvec_print(Q6_V_hi_W(row0), "row0");
                    // hfvec_print(Q6_V_hi_W(row1), "row1");
                    // goto end_loop;
                    // process a total weight tile
                    const int32_t index = (index_k) / HMX_len + (inter_m / HMX_len) * VecK / HMX_len;
                    // FARF(ALWAYS, "[DEBUG] index is %d", index);
                    // goto end_loop;


                    HVX_Vector o[4];
                    o[0] = Q6_V_lo_W(row0);
                    o[1] = Q6_V_hi_W(row0);
                    o[2] = Q6_V_lo_W(row1);
                    o[3] = Q6_V_hi_W(row1);
                    // hfvec_print(o[0], "o1");
                    // hfvec_print(o[1], "o1");
                    // hfvec_print(o[2], "o2");
                    // hfvec_print(o[3], "o3");
                    // count_vec_one_num(o[0], "o[0]", 1.0);
                    // count_vec_one_num(o[1], "o[1]", 1.0);
                    // count_vec_one_num(o[2], "o[2]", 1.0);
                    // count_vec_one_num(o[3], "o[3]", 1.0);
                    //HVX_Vector test_val = Q6_Vh_vsplat_R(0b0011110000000000);
                    for (int w = 0; w < 4; w++) {
                        _Float16* tile_temp = (weight_temp + (index + TileK / HMX_len * w) * 1024 + j * 256 + k * 64);
                         vmem(tile_temp) = o[w];
                    }
                    // vmem(tile_temp) = test_val;

                    // FARF(ALWAYS, "[DEBUG] tile_temp0 addr is %d", (index + TileK / HMX_len * 0) * 1024 + j * 256 + k * 64);
                    // FARF(ALWAYS, "[DEBUG] tile_temp0 addr is %d", (index + TileK / HMX_len * 1) * 1024 + j * 256 + k * 64);
                    // FARF(ALWAYS, "[DEBUG] tile_temp0 addr is %d", (index + TileK / HMX_len * 2) * 1024 + j * 256 + k * 64);
                    // FARF(ALWAYS, "[DEBUG] tile_temp0 addr is %d", (index + TileK / HMX_len * 3) * 1024 + j * 256 + k * 64);
                    // goto end_loop;

                    // vmem(tile_temp0) = Q6_V_lo_W(row0);
                    // vmem(tile_temp1) = Q6_V_hi_W(row0);
                    // vmem(tile_temp2) = Q6_V_lo_W(row1);
                    // vmem(tile_temp3) = Q6_V_hi_W(row1);
                    
                    // hfvec_print(*(HVX_Vector*)tile_temp0, "tile_temp0");
                    // hfvec_print(*(HVX_Vector*)tile_temp1, "tile_temp1");
                    // hfvec_print(*(HVX_Vector*)tile_temp2, "tile_temp2");
                    // hfvec_print(*(HVX_Vector*)tile_temp3, "tile_temp3");
                    // goto end_loop;
                }
            }
            // goto end_loop;
            end_k_loop:
                FARF(ALWAYS, "");
        }
    }

    end_loop:
        FARF(ALWAYS, "");
}



// for(int32_t inter_m = 0; inter_m < TileM; inter_m += VecM) {
//                     for (int32_t group_k = 0; group_k < TileK; group_k += GroupSize){
//                         _Float16 *s_ptr = s_tile_base + (TileK / GroupSize) * inter_m + (group_k / GroupSize) * VecM;
//                         HVX_Vector     s_vec_lo_fp16  = vmem(s_ptr);
//                         HVX_Vector     s_vec_hi_fp16  = vmem(s_ptr + VLEN / sizeof(_Float16));
//                         // is there need to be shuff or just need to compose together
//                         HVX_VectorPair s_pair_lo_fp16 = Q6_W_vshuff_VVR(s_vec_lo_fp16, s_vec_lo_fp16, -2);
//                         HVX_VectorPair s_pair_hi_fp16 = Q6_W_vshuff_VVR(s_vec_hi_fp16, s_vec_hi_fp16, -2);
//                         HVX_VectorPair z_pair_lo_fp16, z_pair_hi_fp16;
//                         if (ZeroPoint) {
//                             HVX_Vector     z_vec_lo_fp16  = vmem(s_ptr + VecM);
//                             HVX_Vector     z_vec_hi_fp16  = vmem(s_ptr + VecM + VLEN / sizeof(_Float16));
//                             // is there need to be shuff or just need to compose together
//                             z_pair_lo_fp16 = Q6_W_vshuff_VVR(z_vec_lo_fp16, z_vec_lo_fp16, -2);
//                             z_pair_hi_fp16 = Q6_W_vshuff_VVR(z_vec_hi_fp16, z_vec_hi_fp16, -2);
//                         }
//                         for(int32_t inter_k = 0; inter_k < GroupSize; inter_k += 8) { // process a HVX Vector, 128B
//                             int32_t index_k = group_k + inter_k;
//                             HVX_VectorPair w_temp_0 = Q6_W_vzero();
//                             HVX_VectorPair w_temp_1 = Q6_W_vzero();
//                             HVX_VectorPair w_temp_2 = Q6_W_vzero();
//                             HVX_VectorPair w_temp_3 = Q6_W_vzero();

//                             for (int32_t bit = 0; bit < Bits; bit++) {
//                                 const uint8_t* w_bit_base = wt_tile_base + (TileM * TileK * bit) / 8 
//                                                           + inter_m * (TileK / 8) + index_k / 8 * VecM; 
//                                 HVX_Vector w_vec_shuff = vmem(w_bit_base);
//                                 HVX_Vector w_vec = Q6_Vb_vdeal_Vb(w_vec_shuff);

//                                 HVX_Vector w_vec_bo = Q6_V_vand_VV(w_vec, mask_vec);   
//                                 HVX_Vector w_vec_to = Q6_Vh_vasr_VhR(w_vec, shift_len); 
                                
//                                 HVX_VectorPair c_vec_bo   = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_bo, LUT1_0, bit); 
//                                 HVX_VectorPair c_vec_bo_2 = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_bo, LUT1_1, bit);
                                
//                                 HVX_VectorPair c_vec_to   = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_to, LUT1_0, bit);
//                                 HVX_VectorPair c_vec_to_2 = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_to, LUT1_1, bit);
                                
//                                 HVX_Vector res_low  = Q6_V_vor_VV(Q6_V_lo_W(c_vec_bo), Q6_V_lo_W(w_temp_0));
//                                 HVX_Vector res_high = Q6_V_vor_VV(Q6_V_hi_W(c_vec_bo), Q6_V_hi_W(w_temp_0));
//                                 w_temp_0 = Q6_W_vcombine_VV(res_high, res_low);
//                                 res_low  = Q6_V_vor_VV(Q6_V_lo_W(c_vec_bo_2), Q6_V_lo_W(w_temp_1));
//                                 res_high = Q6_V_vor_VV(Q6_V_hi_W(c_vec_bo_2), Q6_V_hi_W(w_temp_1));
//                                 w_temp_1 = Q6_W_vcombine_VV(res_high, res_low);
//                                 res_low  = Q6_V_vor_VV(Q6_V_lo_W(c_vec_to), Q6_V_lo_W(w_temp_2));
//                                 res_high = Q6_V_vor_VV(Q6_V_hi_W(c_vec_to), Q6_V_hi_W(w_temp_2));
//                                 w_temp_2 = Q6_W_vcombine_VV(res_high, res_low);
//                                 res_low  = Q6_V_vor_VV(Q6_V_lo_W(c_vec_to_2), Q6_V_lo_W(w_temp_3));
//                                 res_high = Q6_V_vor_VV(Q6_V_hi_W(c_vec_to_2), Q6_V_hi_W(w_temp_3));
//                                 w_temp_3 = Q6_W_vcombine_VV(res_high, res_low);
//                             }
//                             HVX_VectorPair w_temps[4] = {w_temp_0, w_temp_1, w_temp_2, w_temp_3};
//                             const int32_t j = (index_k % 32) / 8;
//                             for (int32_t k = 0; k < 4; ++k) {
//                                 HVX_VectorPair row0, row1;
                                
//                                 if (ZeroPoint) {
//                                     row0 = hvx_int8_w_fp16_s_z_mpy(Q6_V_lo_W(w_temps[k]), s_pair_lo_fp16, z_pair_lo_fp16);
//                                     row1 = hvx_int8_w_fp16_s_z_mpy(Q6_V_hi_W(w_temps[k]), s_pair_hi_fp16, z_pair_hi_fp16);
//                                 } else {
//                                     row0 = hvx_int8_w_fp16_s_mpy(Q6_V_lo_W(w_temps[k]), s_pair_lo_fp16);
//                                     row1 = hvx_int8_w_fp16_s_mpy(Q6_V_hi_W(w_temps[k]), s_pair_hi_fp16);
//                                 }
//                                 // process a total weight tile
//                                 const int32_t index = index_k / HMX_len + (inter_m / HMX_len) * VecK / HMX_len;
//                                 HVX_Vector* tile_temp0 = (HVX_Vector*)(weight_temp + (index + VecM / HMX_len * 0) * 1024 + j * 256 + k * 64);
//                                 HVX_Vector* tile_temp1 = (HVX_Vector*)(weight_temp + (index + VecM / HMX_len * 1) * 1024 + j * 256 + k * 64);
//                                 HVX_Vector* tile_temp2 = (HVX_Vector*)(weight_temp + (index + VecM / HMX_len * 2) * 1024 + j * 256 + k * 64);
//                                 HVX_Vector* tile_temp3 = (HVX_Vector*)(weight_temp + (index + VecM / HMX_len * 3) * 1024 + j * 256 + k * 64);
//                                 *tile_temp0 = Q6_V_lo_W(row0);
//                                 *tile_temp1 = Q6_V_hi_W(row0);
//                                 *tile_temp2 = Q6_V_lo_W(row1);
//                                 *tile_temp3 = Q6_V_hi_W(row1);
//                             }
//                         }
//                     }
//                 }


// int hmx_hvx_tbl_gemm(int32_t GemmM, int32_t GemmK, int32_t GemmN, const _Float16* a, const uint8_t* w, const _Float16 *s, _Float16 *c, 
//     _Float16* act_temp, _Float16* weight_temp, _Float16* output_temp, const uint16_t* LUT1, const _Float16* eye_tile, const _Float16* hf_lut_ptr) {
//     /*TODO: 
//     act_temp, weight_temp, LUT1, output_temp, eye_tile
//     */
//     // every fp 16 hmx tile is 32 * 32 * 2B = 2KB


//     int GroupSize = 128;
//     bool ZeroPoint = false;
//     int Bits = 4;
//     int TileK = 256;
//     int g = 4;

//     const int32_t num_nTileKs = 2;
//     const int32_t HMX_len = 32;
//     const int32_t VecK = 64;
//     const int32_t VecM = 128;
//     const int32_t VecN = 32;
//     const int8_t mask_4bit = 0b1111;
//     const int8_t shift_len = 4;
//     const HVX_Vector mask_vec = Q6_Vb_vsplat_R(mask_4bit); 
    
//     const int32_t K = GemmK;
//     const int32_t M = GemmM;
//     const int32_t N = GemmN;

//     const int32_t q_group_size = GroupSize;

//     const int32_t TileM = 512 / Bits; // TileM TileN is logical Tile, not memory tile
//     const int32_t TileN = 32;

//     const int32_t nTileK = num_nTileKs * TileK; 
    
//     memset(c, 0, sizeof(_Float16) * GemmM * GemmN);
    
//     HVX_Vector LUT1_0 = vmem(LUT1);
//     HVX_Vector LUT1_1 = vmem(LUT1 + VLEN / sizeof(uint16_t));

//     HVX_Vector hf_lut = vmem(hf_lut_ptr);

    
//     for (int32_t tile_n = 0; tile_n < N; tile_n += TileN) {
//         for (int32_t tile_m = 0; tile_m < M; tile_m += TileM) {
//             for (int32_t tile_k = 0; tile_k < K; tile_k += TileK) {
//                 // process one activation tile
//                 _Float16 *act_tile_base = a + tile_n * K + tile_k;
//                 for (int32_t inter_n = 0; inter_n < TileN; inter_n += VecN) { // this layer of loop can be ignored
//                     for (int32_t inter_k = 0; inter_k < TileK; inter_k += VecK) { 
//                         const int32_t index = inter_k / HMX_len;
//                         for (int32_t i = 0; i < VecN; i += 2) {
//                             const _Float16* flatA = act_tile_base +  i    * K + inter_k;
//                             const _Float16* flatB = act_tile_base + (i+1) * K + inter_k;
//                             uint32_t offset = i * HMX_len;
//                             _Float16* act_ptr_0 = act_temp + index * 1024 + offset;
//                             _Float16* act_ptr_1 = act_temp + (index + 1) * 1024 + offset;
//                             flat_to_crouton((_Float16*)flatA, (_Float16*)flatB, act_ptr_0, act_ptr_1);
//                         }
//                     }
//                 }

//                 // process one weight tile
//                 // a weight tile is (TileK * TileM * Bits / 8)  Bytes
//                 // every col tile is (TileM * Bits * K / 8) Bytes
//                 // this is (tile_m / TileM, tile_k / TileK) tile 
//                 uint8_t *wt_tile_base = w + (K * Bits / 8) * tile_m + (TileM * Bits / 8) * tile_k;
//                 // a scale tile is (TileK / GroupSize * TileM * sizeof(_Float16)) Bytes
//                 // col tile is (K / GroupSize * TileM * sizeof(_Float16)) Bytes
//                 _Float16 *s_tile_base  = s + (K / GroupSize) * tile_m * (1 + ZeroPoint)
//                                            + (TileM * (tile_k / GroupSize)) * (1 + ZeroPoint);
//                 weight_deconquantization(wt_tile_base, s_tile_base, TileM, VecM, HMX_len, VecK, TileK, GroupSize, Bits, LUT1_0, LUT1_1, ZeroPoint, weight_temp, mask_vec, shift_len, hf_lut);
//                 // hmx compute
//                 const int32_t num_hmx_tiles = TileK / HMX_len;
                

//                 for (int32_t t = 0; t < TileM / HMX_len; t++) {
//                     #ifdef __hexagon__
//                     asm volatile("mxclracc.hf"); 
//                     #endif
//                     if (tile_k != 0) {
//                         hmx_load_tiles_fp16(output_temp + t * 1024, eye_tile, 1);
//                     }
//                     hmx_load_tiles_fp16(act_temp + 0, weight_temp + (t * num_hmx_tiles) * 1024, num_hmx_tiles);
//                     hmx_consume_accumulator_fp16(output_temp + t * 1024); 
//                 }
//             }
            
//             _Float16 *c_tile_base = c + tile_n * M + tile_m;
//             for (int32_t j = 0; j < TileM / HMX_len; j+=2) {
//                 for (int32_t row = 0; row < TileN; row+=2) {
//                     _Float16* flatA = c_tile_base +  row    * M + j * HMX_len;
//                     _Float16* flatB = c_tile_base + (row+1) * M + j * HMX_len;
//                     _Float16* hmx_tile0 = output_temp + j * 1024     + row * HMX_len;
//                     _Float16* hmx_tile1 = output_temp + (j+1) * 1024 + row * HMX_len;
//                     crouton_to_flat(flatA, flatB, hmx_tile0, hmx_tile1);
//                 }
//             }
//         }
//     }
//     return 0;
// }

// void weight_deconquantization (uint8_t *wt_tile_base, _Float16 *s_tile_base, int32_t TileM, int32_t VecM, int32_t HMX_len, int32_t VecK, int TileK, int32_t GroupSize, int32_t Bits, 
//                                HVX_Vector LUT1_0, HVX_Vector LUT1_1, bool ZeroPoint, _Float16 *weight_temp, HVX_Vector mask_vec, int8_t shift_len, HVX_Vector hf_lut) {
//     // a scale tile is (TileK / GroupSize * TileM * sizeof(_Float16)) Bytes
//     // col tile is (K / GroupSize * TileM * sizeof(_Float16)) Bytes
//     for(int32_t inter_m = 0; inter_m < TileM; inter_m += VecM) {
//         for (int32_t group_k = 0; group_k < TileK; group_k += GroupSize){
//             _Float16 *s_ptr = s_tile_base + (TileK / GroupSize) * inter_m + (group_k / GroupSize) * VecM;
//             HVX_Vector     s_vec_lo_fp16  = vmem(s_ptr);
//             HVX_Vector     s_vec_hi_fp16  = vmem(s_ptr + VLEN / sizeof(_Float16));

//             // // scales and zeros is right
//             // hfvec_print(s_vec_lo_fp16, "s_vec_lo_fp16");
//             // goto end_loop;
//             // break;
//             // FARF(ALWAYS, "[DEBUG] zeropoint is %d", ZeroPoint);
//             // break;
//             // count_one_num(s_vec_lo_fp16, "s_vec_lo_fp16", 1.0);
//             // count_one_num(s_vec_hi_fp16, "s_vec_hi_fp16", 1.0);

//             // is there need to be shuff or just need to compose together
//             HVX_VectorPair s_pair_lo_fp16 = Q6_W_vshuff_VVR(s_vec_lo_fp16, s_vec_lo_fp16, -2);
//             HVX_VectorPair s_pair_hi_fp16 = Q6_W_vshuff_VVR(s_vec_hi_fp16, s_vec_hi_fp16, -2);
//             HVX_VectorPair z_pair_lo_fp16, z_pair_hi_fp16;
//             if (ZeroPoint) {
//                 HVX_Vector     z_vec_lo_fp16  = vmem(s_ptr + VecM);
//                 HVX_Vector     z_vec_hi_fp16  = vmem(s_ptr + VecM + VLEN / sizeof(_Float16));
//                 // is there need to be shuff or just need to compose together
//                 z_pair_lo_fp16 = Q6_W_vshuff_VVR(z_vec_lo_fp16, z_vec_lo_fp16, -2);
//                 z_pair_hi_fp16 = Q6_W_vshuff_VVR(z_vec_hi_fp16, z_vec_hi_fp16, -2);
//             }

//             // count_pair_one_num(s_pair_lo_fp16, "s_pair_lo_fp16", 1.0);
//             // count_pair_one_num(s_pair_hi_fp16, "s_pair_hi_fp16", 1.0);

//             for(int32_t inter_k = 0; inter_k < GroupSize; inter_k += 8) { // process a HVX Vector, 128B
//                 int32_t index_k = group_k + inter_k;
//                 // FARF(ALWAYS, "[DEBUG] index_k is %d", index_k);
//                 HVX_VectorPair w_temp_0 = Q6_W_vzero();
//                 HVX_VectorPair w_temp_1 = Q6_W_vzero();
//                 HVX_VectorPair w_temp_2 = Q6_W_vzero();
//                 HVX_VectorPair w_temp_3 = Q6_W_vzero();

//                 for (int32_t bit = 0; bit < Bits; bit++) {
//                     const uint8_t* w_bit_base = wt_tile_base + (TileM * TileK * bit) / 8
//                                                 + inter_m * (TileK / 8) + index_k / 8 * VecM; 
//                     // FARF(ALWAYS, "[DEBUG] bit base addr %d", (TileM * TileK * bit) / 8 + inter_m * (TileK / 8) + index_k / 8 * VecM);
//                     // HVX_Vector w_vec_shuff = vmem(w_bit_base);
//                     // HVX_Vector w_vec = Q6_Vb_vdeal_Vb(w_vec_shuff);
//                     // do not need shuff, because  Q6_Wh_vlut16_VbVhR_nomatch needs shuffled input
//                     HVX_Vector w_vec = vmem(w_bit_base);

//                     // u8bec_print(w_vec, "w_vec");

//                     HVX_Vector w_vec_bo = Q6_V_vand_VV(w_vec, mask_vec);
//                     HVX_Vector w_vec_to = Q6_Vh_vasr_VhR(w_vec, shift_len);
                    
//                     // if (bit == 1) {
//                     //     u8bec_print(w_vec_bo, "w_vec_bo");
//                     //     // hvec_print(LUT1_0, "LUT1_0");
//                     //     u8bec_print(w_vec_to, "w_vec_to");
//                     // }
//                     // goto end_bit_loop;
//                     // goto end_loop;
                    
//                     HVX_VectorPair c_vec_bo   = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_bo, LUT1_0, bit);
//                     // if (bit == 3) {
//                     //     bit_hvec_print(Q6_V_lo_W(c_vec_bo), "c_vec_bo");
//                     // }
//                     // goto end_bit_loop;
//                     HVX_VectorPair c_vec_bo_2 = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_bo, LUT1_1, bit);
//                     // bit_hvec_print(Q6_V_lo_W(c_vec_bo_2), "c_vec_bo");
//                     // goto end_bit_loop;
//                     // output the normal seq
                    
//                     HVX_VectorPair c_vec_to   = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_to, LUT1_0, bit);
//                     HVX_VectorPair c_vec_to_2 = Q6_Wh_vlut16_VbVhR_nomatch(w_vec_to, LUT1_1, bit);

//                     // if (bit == 1) {
//                     //     bit_hvec_print(Q6_V_lo_W(c_vec_to_2), "c_vec_to_2");
//                     // }
                    
//                     HVX_Vector res_low  = Q6_V_vor_VV(Q6_V_lo_W(c_vec_bo), Q6_V_lo_W(w_temp_0));
//                     HVX_Vector res_high = Q6_V_vor_VV(Q6_V_hi_W(c_vec_bo), Q6_V_hi_W(w_temp_0));
//                     w_temp_0 = Q6_W_vcombine_VV(res_high, res_low);
//                     // if (bit == 3) {
//                     //     bit_hvec_print(Q6_V_lo_W(w_temp_0), "w_temp_0_lo");
//                     // }
//                     res_low  = Q6_V_vor_VV(Q6_V_lo_W(c_vec_bo_2), Q6_V_lo_W(w_temp_1));
//                     res_high = Q6_V_vor_VV(Q6_V_hi_W(c_vec_bo_2), Q6_V_hi_W(w_temp_1));
//                     w_temp_1 = Q6_W_vcombine_VV(res_high, res_low);
//                     res_low  = Q6_V_vor_VV(Q6_V_lo_W(c_vec_to), Q6_V_lo_W(w_temp_2));
//                     res_high = Q6_V_vor_VV(Q6_V_hi_W(c_vec_to), Q6_V_hi_W(w_temp_2));
//                     w_temp_2 = Q6_W_vcombine_VV(res_high, res_low);
//                     res_low  = Q6_V_vor_VV(Q6_V_lo_W(c_vec_to_2), Q6_V_lo_W(w_temp_3));
//                     res_high = Q6_V_vor_VV(Q6_V_hi_W(c_vec_to_2), Q6_V_hi_W(w_temp_3));
//                     w_temp_3 = Q6_W_vcombine_VV(res_high, res_low);
//                 }
//                 end_bit_loop:
//                     FARF(ALWAYS, "");

//                 // u8bec_print(Q6_V_lo_W(w_temp_0), "w_temp_0"); // should be all one 
//                 // u8bec_print(Q6_V_lo_W(w_temp_1), "w_temp_1"); // should be all one 
//                 // u8bec_print(Q6_V_lo_W(w_temp_2), "w_temp_2"); // should be all one 
//                 // u8bec_print(Q6_V_lo_W(w_temp_3), "w_temp_3"); // should be all one 
//                 // goto end_loop;

                
//                 HVX_VectorPair w_temps[4] = {w_temp_0, w_temp_1, w_temp_2, w_temp_3};
//                 // count_pair_one_num_fixed(w_temps[0], "w_temp0", 1);
//                 // count_pair_one_num_fixed(w_temps[1], "w_temp1", 1);
//                 // count_pair_one_num_fixed(w_temps[2], "w_temp2", 1);
//                 // count_pair_one_num_fixed(w_temps[3], "w_temp3", 1);
//                 const int32_t j = ((index_k) % 32) / 8;
//                 // FARF(ALWAYS, "[DEBUG] row is %d", index_k);
//                 for (int32_t k = 0; k < 4; ++k) {
//                     HVX_VectorPair row0, row1;
                    
//                     if (ZeroPoint) {
//                         row0 = hvx_int8_w_fp16_s_z_mpy(Q6_V_lo_W(w_temps[k]), s_pair_lo_fp16, z_pair_lo_fp16, hf_lut);
//                         row1 = hvx_int8_w_fp16_s_z_mpy(Q6_V_hi_W(w_temps[k]), s_pair_hi_fp16, z_pair_hi_fp16, hf_lut);
//                     } else {
//                         row0 = hvx_int8_w_fp16_s_mpy(Q6_V_lo_W(w_temps[k]), s_pair_lo_fp16, hf_lut);
//                         row1 = hvx_int8_w_fp16_s_mpy(Q6_V_hi_W(w_temps[k]), s_pair_hi_fp16, hf_lut);
//                         // count_pair_one_num_fixed(w_temps[k], "w_temps", 1);
//                         // count_pair_one_num(s_pair_lo_fp16, "s_pair_lo_fp16", 1.0);
//                         // count_pair_one_num(row0, "row0", 1.0);
//                         // count_pair_one_num(row1, "row1", 1.0);
//                     }
//                     // FARF(ALWAYS, "[DEBUG] row is %d", index_k + k * 2);
//                     // count_pair_one_num(row0, "row0", 1.0);
//                     // count_pair_one_num(row1, "row1", 1.0);
//                     // goto end_loop;
//                     // hfvec_print(Q6_V_hi_W(row0), "row0");
//                     // hfvec_print(Q6_V_hi_W(row1), "row1");
//                     // goto end_loop;
//                     // process a total weight tile
//                     const int32_t index = (index_k) / HMX_len + (inter_m / HMX_len) * VecK / HMX_len;
//                     // FARF(ALWAYS, "[DEBUG] index is %d", index);
//                     // goto end_loop;


//                     HVX_Vector o[4];
//                     o[0] = Q6_V_lo_W(row0);
//                     o[1] = Q6_V_hi_W(row0);
//                     o[2] = Q6_V_lo_W(row1);
//                     o[3] = Q6_V_hi_W(row1);
//                     // hfvec_print(o[0], "o1");
//                     // hfvec_print(o[1], "o1");
//                     // hfvec_print(o[2], "o2");
//                     // hfvec_print(o[3], "o3");
//                     // count_vec_one_num(o[0], "o[0]", 1.0);
//                     // count_vec_one_num(o[1], "o[1]", 1.0);
//                     // count_vec_one_num(o[2], "o[2]", 1.0);
//                     // count_vec_one_num(o[3], "o[3]", 1.0);
//                     //HVX_Vector test_val = Q6_Vh_vsplat_R(0b0011110000000000);
//                     for (int w = 0; w < 4; w++) {
//                         _Float16* tile_temp = (weight_temp + (index + TileK / HMX_len * w) * 1024 + j * 256 + k * 64);
//                          vmem(tile_temp) = o[w];
//                     }
//                     // vmem(tile_temp) = test_val;

//                     // FARF(ALWAYS, "[DEBUG] tile_temp0 addr is %d", (index + TileK / HMX_len * 0) * 1024 + j * 256 + k * 64);
//                     // FARF(ALWAYS, "[DEBUG] tile_temp0 addr is %d", (index + TileK / HMX_len * 1) * 1024 + j * 256 + k * 64);
//                     // FARF(ALWAYS, "[DEBUG] tile_temp0 addr is %d", (index + TileK / HMX_len * 2) * 1024 + j * 256 + k * 64);
//                     // FARF(ALWAYS, "[DEBUG] tile_temp0 addr is %d", (index + TileK / HMX_len * 3) * 1024 + j * 256 + k * 64);
//                     // goto end_loop;

//                     // vmem(tile_temp0) = Q6_V_lo_W(row0);
//                     // vmem(tile_temp1) = Q6_V_hi_W(row0);
//                     // vmem(tile_temp2) = Q6_V_lo_W(row1);
//                     // vmem(tile_temp3) = Q6_V_hi_W(row1);
                    
//                     // hfvec_print(*(HVX_Vector*)tile_temp0, "tile_temp0");
//                     // hfvec_print(*(HVX_Vector*)tile_temp1, "tile_temp1");
//                     // hfvec_print(*(HVX_Vector*)tile_temp2, "tile_temp2");
//                     // hfvec_print(*(HVX_Vector*)tile_temp3, "tile_temp3");
//                     // goto end_loop;
//                 }
//             }
//             // goto end_loop;
//             end_k_loop:
//                 FARF(ALWAYS, "");
//         }
//     }

//     end_loop:
//         FARF(ALWAYS, "");
// }
