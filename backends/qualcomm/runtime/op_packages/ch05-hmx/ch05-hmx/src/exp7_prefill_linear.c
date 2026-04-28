#include "AEEStdErr.h"
#include <stddef.h>        /* size_t — needed before hexkl headers */
#include "hexkl_micro.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HAP_farf.h"

// #include <hexagon_types.h>
// #include <string.h>
// #include <stdint.h>
// #include "hmx_utils.h"
// #include <stdbool.h>


#define N_define 64
#define M_define 1024
#define K_define 2048

static inline uint8_t *vtcm_seq_alloc(uint8_t **ptr, size_t size) {
    uint8_t *p = *ptr;
    size = (size + 127) & ~(size_t)127;   /* 128 byte alignment */
    *ptr += size;
    return p;
}

static inline uint8_t *vtcm_seq_alloc_align_256(uint8_t **ptr, size_t size) {
    const size_t align = 256; // HMX Tile 要求的 2KB 对齐
    
    // 1. 首先将当前指针地址向上对齐到 2048 字节
    uintptr_t curr_addr = (uintptr_t)*ptr;
    uintptr_t aligned_addr = (curr_addr + (align - 1)) & ~(uintptr_t)(align - 1);
    
    // 2. 将对齐后的地址作为起始分配地址
    uint8_t *p = (uint8_t *)aligned_addr;
    
    // 3. 同样将申请的大小向上对齐到 2048，确保下一个分配的起始点也是对齐的
    size = (size + (align - 1)) & ~(size_t)(align - 1);
    
    // 4. 更新全局指针
    *ptr = p + size;
    
    return p;
}

static inline uint8_t *vtcm_seq_hmx_temp_alloc(uint8_t **ptr, size_t size) {
    const size_t align = 2048; // HMX Tile 要求的 2KB 对齐
    
    // 1. 首先将当前指针地址向上对齐到 2048 字节
    uintptr_t curr_addr = (uintptr_t)*ptr;
    uintptr_t aligned_addr = (curr_addr + (align - 1)) & ~(uintptr_t)(align - 1);
    
    // 2. 将对齐后的地址作为起始分配地址
    uint8_t *p = (uint8_t *)aligned_addr;
    
    // 3. 同样将申请的大小向上对齐到 2048，确保下一个分配的起始点也是对齐的
    size = (size + (align - 1)) & ~(size_t)(align - 1);
    
    // 4. 更新全局指针
    *ptr = p + size;
    
    return p;
}


uint8_t *ptr = NULL;
uint8_t *debug_ptr = NULL;

#include "hvx_funcs.h"

void hfvec_print(HVX_Vector a, char *name);
void u8bec_print(HVX_Vector a, char *name);
void hvec_print(HVX_Vector a, char* name);

void* load_bin(const char* path, size_t* out_size) {
    FARF(ALWAYS, "[load_bin] Attempting to open: %s", path);
    FILE* file = fopen(path, "rb");
    if (!file) {
        // 关键：在 DSP 日志里打印错误，而不是 perror
        FARF(ALWAYS, "[load_bin] ERROR: Cannot open %s", path);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file); 

    if (size <= 0) {
        FARF(ALWAYS, "[load_bin] ERROR: File %s size is %ld", path, size);
        fclose(file);
        return NULL;
    }

    // 使用 128 字节对齐分配内存，Hexagon 硬件更友好
    void* buffer = memalign(128, size);
    if (!buffer) {
        FARF(ALWAYS, "[load_bin] ERROR: Malloc failed for %s", path);
        fclose(file);
        return NULL;
    }

    size_t read_bytes = fread(buffer, 1, size, file);
    fclose(file);

    if (read_bytes != (size_t)size) {
        FARF(ALWAYS, "[load_bin] ERROR: Read %zu, expected %ld for %s", read_bytes, size, path);
        free(buffer);
        return NULL;
    }

    if (out_size != NULL) *out_size = (size_t)size;
    FARF(ALWAYS, "[load_bin] Successfully loaded %s (%ld bytes)", path, size);
    return buffer;
}

// for every hvx fucntion, write corresponding function test
// TODO:
void Q6_Wh_vlut16_VbVhR_nomatch_test() {
    uint8_t *weight = (uint8_t*) vtcm_seq_alloc(&ptr, 128);
    int even = 0;
    int odd = 1;
    for (int i = 0; i < 128; i++) {
        if(i < 64) {
            weight[even] = i;
            even += 2;
        }
        else {
            weight[odd] = i;
            odd += 2;
        }
        // weight[i] = i;
    }
    uint16_t *original_lut = (uint16_t*)vtcm_seq_alloc(&ptr, 128);
    for (int i = 0; i < 64; i++) original_lut[i] = i;
    uint16_t *formatted_lut = (uint16_t*)vtcm_seq_alloc(&ptr, 128);
    // lut0 in even upper 32 eles, lut1 in even lower 32 eles, lut2 in odd upper 32 eles, lut3 in odd lower 32 eles
    int index_even = 0;
    int index_odd = 1;
    for (int i = 0; i < 64; i++) {
        // uint8_t new_index = i / 32 + (((i%32) / 16) * 16 + (i % 16)) * 2;
        if (i < 32) {
            formatted_lut[index_even] = original_lut[i];
            index_even += 2;
        } else {
            formatted_lut[index_odd] = original_lut[i];
            index_odd += 2;
        }
        // formatted_lut[new_index] = original_lut[i];
    }

    HVX_Vector W = *(HVX_Vector*)weight;
    HVX_Vector LUT = *(HVX_Vector*)formatted_lut;

    // hvec_print(LUT, "LUT");

    HVX_VectorPair out0 = Q6_Wh_vlut16_VbVhR_nomatch(W, LUT, 0);
    hvec_print(Q6_V_lo_W(out0), "out0");
    hvec_print(Q6_V_hi_W(out0), "out0");
    // HVX_VectorPair out1 = Q6_W_vshuff_VVR(Q6_V_hi_W(out0), Q6_V_lo_W(out0), -2);
    // hvec_print(Q6_V_lo_W(out1), "out1");
    // hvec_print(Q6_V_hi_W(out1), "out1");
}

void test_write_weight_temp (uint8_t *wt_tile_base, _Float16 *s_tile_base, int32_t TileM, int32_t VecM, int32_t HMX_len, int32_t VecK, int TileK, int32_t GroupSize, int32_t Bits, 
                               HVX_Vector LUT1_0, HVX_Vector LUT1_1, bool ZeroPoint, _Float16 *weight_temp, HVX_Vector mask_vec, int8_t shift_len) {
    // a scale tile is (TileK / GroupSize * TileM * sizeof(_Float16)) Bytes
    // col tile is (K / GroupSize * TileM * sizeof(_Float16)) Bytes
    for(int32_t inter_m = 0; inter_m < TileM; inter_m += VecM) {
        for (int32_t group_k = 0; group_k < TileK; group_k += GroupSize){
            for(int32_t inter_k = 0; inter_k < GroupSize; inter_k += 8) { // process a HVX Vector, 128B
                int32_t index_k = group_k + inter_k;
                // FARF(ALWAYS, "[DEBUG] index_k is %d", index_k);

                const int32_t j = ((index_k) % 32) / 8;
                // FARF(ALWAYS, "[DEBUG] row is %d", index_k);
                for (int32_t k = 0; k < 4; ++k) {
                    
                    // FARF(ALWAYS, "[DEBUG] row is %d", index_k + k * 2);
                    // count_none_one_num(row0, "row0");
                    // count_none_one_num(row1, "row1");
                    // hfvec_print(Q6_V_hi_W(row0), "row0");
                    // hfvec_print(Q6_V_hi_W(row1), "row1");
                    // goto end_loop;
                    // process a total weight tile
                    const int32_t index = (index_k) / HMX_len + (inter_m / HMX_len) * VecK / HMX_len;
                    // FARF(ALWAYS, "[DEBUG] index is %d", index);
                    // goto end_loop;

                    HVX_Vector o[4];
                    o[0] = Q6_Vh_vsplat_R(0b0100010000000000);
                    o[1] = Q6_Vh_vsplat_R(0b0100010000000000);
                    o[2] = Q6_Vh_vsplat_R(0b0100010000000000);
                    o[3] = Q6_Vh_vsplat_R(0b0100010000000000);
                    // hfvec_print(o[0], "o1");
                    // hfvec_print(o[1], "o1");
                    // hfvec_print(o[2], "o2");
                    // hfvec_print(o[3], "o3");
                    // goto end_loop;
                    for (int w = 0; w < 4; w++) {
                        _Float16* tile_temp = (weight_temp + (index + TileK / HMX_len * w) * 1024 + j * 256 + k * 64);
                        vmem(tile_temp) = o[w];
                    }
                }
                // goto end_loop;
            }
        }
    }

    end_loop:
        FARF(ALWAYS, "");
}

   

void test_int_to_fp16() {
    HVX_Vector w_int8 = Q6_Vb_vsplat_R(0b0001);
    _Float16* hf_lut_ptr = load_bin("hf_lut.bin", NULL);
    _Float16* hf_lut_vtcm_ptr = (_Float16*) vtcm_seq_alloc(&ptr, VLEN);
    memcpy(hf_lut_vtcm_ptr, hf_lut_ptr, VLEN);
    HVX_Vector hf_lut = vmem(hf_lut_vtcm_ptr);
    u8bec_print(w_int8, "w_int8");
    HVX_VectorPair w_fp16 = HVX_int8_cvt_hf(w_int8, hf_lut);   
    count_pair_one_num(w_fp16, "w_fp16", 1.0);
}

void test_rm_to_ah(_Float16* act_tile_base, _Float16* act_temp) {
    rm_to_ah(act_tile_base, act_temp, 64, 2048, 32, 256);
    for (int i = 0; i < 1024; i++) {
        FARF(ALWAYS, "[DEBUG] act_temp[%d] = %.1f", i, (float)act_temp[i]);
    }
}

void test_hmx_multiplication() {
    
}


int main(void) {
    int      res  = AEE_SUCCESS;
    int      res2 = AEE_SUCCESS;
    uint8_t *vtcm_base = NULL;
    uint32_t vtcm_size = 0;

    FARF(ALWAYS, "=== Prefill Linear Test === ");

    /* ---- Hardware init: map VTCM ---- */
    res = hexkl_micro_hw_init(&vtcm_base, &vtcm_size);
    if (res != AEE_SUCCESS) {
        FARF(ALWAYS, "[MICRO][ERROR] hexkl_micro_hw_init failed (%d)", res);
        return res;
    }
    FARF(ALWAYS, "[MICRO] VTCM base = %p, size = %u bytes (%u KB)",
           (void *)vtcm_base, (unsigned)vtcm_size, (unsigned)vtcm_size / 1024);
    
    /* ---- Print library version ---- */
    {
        int  major = 0, minor = 0, patch = 0, hex_ver = 0;
        char prerel[HEXKL_PREREL_STR_LEN];
        hexkl_micro_get_version(&major, &minor, &patch, prerel, &hex_ver);
        FARF(ALWAYS, "[MICRO] HexKL version: %d.%d.%d-%s  (Hexagon V%d)",
               major, minor, patch, prerel, hex_ver);
    }

    /* ---- Lock HMX (exclusive access to the matrix accelerator) ---- */
    // res = hexkl_micro_hmx_lock();
    // if (res != AEE_SUCCESS) {
    //     FARF(ALWAYS, "[MICRO][ERROR] hexkl_micro_hmx_lock failed (%d)", res);
    //     return res;
    // }
    // FARF(ALWAYS, "[MICRO] HMX locked -- we have exclusive NPU access");

    size_t act_bytes = N_define * K_define * sizeof(_Float16);
    size_t wt_bytes  = (M_define * K_define * sizeof(uint8_t)) / 2;
    size_t out_bytes = N_define * M_define * sizeof(_Float16);
    size_t scales_bytes = (N_define * K_define * sizeof(_Float16)) / 128;
    size_t eye_tile_bytes = 0;
    size_t lut1_bytes = 0;

    _Float16 *act      = (_Float16 *)load_bin("activation.bin", &act_bytes);
    uint8_t  *wt       = (uint8_t *)load_bin("weights.bin", &wt_bytes);
    _Float16 *out_ddr  = (_Float16 *)malloc(out_bytes);
    _Float16 *eye_tile = (_Float16 *)load_bin("eye_tile.bin", &eye_tile_bytes);
    _Float16 *scales   = (_Float16 *)load_bin("scales.bin", &scales_bytes);
    uint16_t *lut1     = (uint16_t *)load_bin("lut1.bin", &lut1_bytes);
    _Float16* hf_lut_ptr = load_bin("hf_lut.bin", NULL);

    const int32_t gemm_k = K_define;
    const int32_t gemm_n = N_define;
    const int32_t gemm_m = M_define;

    ptr = vtcm_base;

    debug_ptr = (uint8_t*) vtcm_seq_alloc(&ptr, VLEN);

    _Float16 *a_ptr        = (_Float16 *)vtcm_seq_alloc(&ptr, act_bytes);
    uint8_t *w_ptr         = (uint8_t *)vtcm_seq_alloc(&ptr, wt_bytes);
    _Float16 *s_ptr        = (_Float16 *)vtcm_seq_alloc(&ptr, scales_bytes);
    _Float16 *c_ptr        = (_Float16 *)vtcm_seq_alloc(&ptr, out_bytes);
    uint16_t *lut1_ptr     = (uint16_t *)vtcm_seq_alloc(&ptr, lut1_bytes);
    _Float16 *eye_tile_ptr = (_Float16 *)vtcm_seq_alloc(&ptr, eye_tile_bytes);
    _Float16 *act_temp     = (_Float16 *)vtcm_seq_hmx_temp_alloc(&ptr, 32 * 256 * 2);
    _Float16 *weight_temp  = (_Float16 *)vtcm_seq_hmx_temp_alloc(&ptr, 256 * 128 * 2);
    _Float16 *output_temp  = (_Float16 *)vtcm_seq_hmx_temp_alloc(&ptr, 32 * 128 * 2);
    _Float16* hf_lut_vtcm_ptr = (_Float16*) vtcm_seq_alloc(&ptr, VLEN);
    _Float16* scales_off = (_Float16*) vtcm_seq_alloc_align_256(&ptr, VLEN * 2);

    vmem(scales_off) = Q6_Vh_vsplat_R(0x3C00);
    vmem(scales_off + VLEN / 2) = Q6_V_vzero();
    //vmem(scales_off + VLEN / 2) = Q6_Vh_vsplat_R(0x3C00);
    // for (int i = 0; i < 128; i++) {
    //     FARF(ALWAYS, "[DEBUG] scales_off[%d] = %.1f", i, (float)scales_off[i]);
    // }
    

    // memset(out_hmx, 0, out_bytes);
    // for (int i = 0; i < 256 * 128; i++) {
    //     weight_temp[i] = 3.0f; // 确确实实的 2.0
    // }
    // for(int i = 0; i < 64; i++)
    // FARF(ALWAYS, "[DEBUG] weight temp [%d] is %.1f", i, (float)weight_temp[i]);

    memcpy(a_ptr, act, act_bytes);
    memcpy(w_ptr, wt, wt_bytes);
    memcpy(s_ptr, scales, scales_bytes);
    memcpy(lut1_ptr, lut1, lut1_bytes);
    memcpy(eye_tile_ptr, eye_tile, eye_tile_bytes);
    memcpy(hf_lut_vtcm_ptr, hf_lut_ptr, VLEN);


    // FARF(ALWAYS, "[DEBUG] w_ptr[0] %d", (uint8_t)w_ptr[0]);
    // FARF(ALWAYS, "[DEBUG] a_ptr[0] %.1f", (float)a_ptr[0]);
    // FARF(ALWAYS, "[DEBUG] s_ptr[0] %.1f", (float)s_ptr[0]);
    
    hmx_hvx_tbl_gemm(
        gemm_m,
        gemm_k,
        gemm_n,
        a_ptr,
        w_ptr,
        s_ptr,
        c_ptr,
        act_temp,
        weight_temp,
        output_temp,
        lut1_ptr,
        eye_tile_ptr,
        hf_lut_vtcm_ptr,
        scales_off
    );
    // void weight_deconquantization (uint8_t *wt_tile_base, _Float16 *s_tile_base, int32_t TileM, int32_t VecM, int32_t HMX_len, int32_t VecK, int TileK, int32_t GroupSize, int32_t Bits, 
    //                           HVX_Vector LUT1_0, HVX_Vector LUT1_1, bool ZeroPoint, _Float16 *weight_temp, HVX_Vector mask_vec, int8_t shift_len) 

    // test_rm_to_ah(a_ptr, act_temp);
    // Q6_Wh_vlut16_VbVhR_nomatch_test();
    // FARF(ALWAYS, "[DEBUG] %d", eye_tile_bytes);
    // HVX_Vector LUT1_0 = vmem(lut1_ptr);
    // HVX_Vector LUT1_1 = vmem(lut1_ptr + VLEN / sizeof(uint16_t));
    // const int8_t mask_4bit = 0b1111;
    // const int8_t shift_len = 4;
    // const HVX_Vector mask_vec = Q6_Vb_vsplat_R(mask_4bit); 
    // const HVX_Vector hf_lut = vmem(hf_lut_vtcm_ptr);
    // weight_deconquantization(w_ptr, s_ptr, 128, 128, 32, 64, 256, 128, 4, LUT1_0, LUT1_1, false, weight_temp, mask_vec, shift_len, hf_lut);
    // test_write_weight_temp(w_ptr, s_ptr, 128, 128, 32, 64, 256, 128, 4, LUT1_0, LUT1_1, false, weight_temp, mask_vec, shift_len);
    
    // test_int_to_fp16();


    // test lut1
    // hvec_print(LUT1_0, "LUT1_0");
    // _Float16 * w222 = weight_temp;
    // for(int i = 0; i < 256 * 128; i+=1)
    // FARF(ALWAYS, "[DEBUG] weight temp [%d] is %.1f", i, (float)w222[i]);
    // memcpy(out_ddr, c_ptr, out_bytes);
    // FARF(ALWAYS, "out first ele is %f", (float)out_ddr[0]);
    for (int i = 0; i < 1024; i++) {
        FARF(ALWAYS, "[DEBUG] c_ptr[%d] = %.1f", i, (float)c_ptr[i]);
    }
    
    // res2 = hexkl_micro_hmx_unlock();
    // if (res2 != AEE_SUCCESS) {
    //     FARF(ALWAYS, "[MICRO][ERROR] hexkl_micro_hmx_unlock failed (%d)", res2);
    //     res |= res2;
    // } else {
    //     FARF(ALWAYS, "[MICRO] HMX unlocked");
    // }

    free(act);
    free(wt);
    free(scales);
    free(lut1);
    free(eye_tile);
    free(out_ddr);
    

    return 0;
}