#include "AEEStdErr.h"
#include <stddef.h>
#include "hexkl_micro.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HAP_farf.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"

// ---------------------------------------------------------------- //
// --- 宏定义与 VTCM 分配器 ---
// ---------------------------------------------------------------- //
#define VLEN 128

static inline uint8_t *vtcm_seq_alloc(uint8_t **ptr, size_t size) {
    uint8_t *p = *ptr;
    size = (size + 127) & ~(size_t)127;   /* 128 字节对齐 */
    *ptr += size;
    return p;
}

#define HVX_VP_APPLY2(op, a, b) \
    Q6_W_vcombine_VV(op(Q6_V_hi_W(a), Q6_V_hi_W(b)), \
                     op(Q6_V_lo_W(a), Q6_V_lo_W(b)))

// ---------------------------------------------------------------- //
// --- 待测向量函数 ---
// ---------------------------------------------------------------- //
static inline HVX_VectorPair hvx_vp_mpy_fp16(HVX_VectorPair a, HVX_VectorPair b) {
    // return HVX_VP_APPLY2(Q6_Vqf16_vmpy_VhfVhf, a, b);
    return HVX_VP_APPLY2(Q6_Vhf_vmpy_VhfVhf, a, b);
}


static inline HVX_VectorPair hvx_int8_w_fp16_s_mpy(HVX_Vector w_int8, HVX_VectorPair s_fp16) {
    // HVX_VectorPair w_qf16 = Q6_Whf_vcvt_Vub(w_int8);
    // HVX_Vector w_qf16_lo = Q6_Vhf_equals_Vqf16(Q6_V_lo_W(w_qf16)); 
    // HVX_Vector w_qf16_hi = Q6_Vhf_equals_Vqf16(Q6_V_hi_W(w_qf16));
    // HVX_VectorPair w_hf = Q6_W_vcombine_VV(w_qf16_hi, w_qf16_lo);
    // return hvx_vp_mpy_fp16(w_hf, s_fp16);

    HVX_VectorPair w_hf = Q6_Whf_vcvt_Vub(w_int8);
    
    // 2. 直接进行向量对乘法。
    // w_hf 是 VectorPair, s_fp16 也是 VectorPair。
    return hvx_vp_mpy_fp16(w_hf, s_fp16);
}

static inline void flat_to_crouton(_Float16* flatA, _Float16* flatB, _Float16* hmx_tile0, _Float16* hmx_tile1) {
    const HVX_Vector v0 = *(HVX_Vector*)flatA;
    const HVX_Vector v1 = *(HVX_Vector*)flatB;
    const HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

    *((HVX_Vector*)(hmx_tile0)) = Q6_V_lo_W(vp);
    *((HVX_Vector*)(hmx_tile1)) = Q6_V_hi_W(vp);
}

static inline void crouton_to_flat(_Float16* flatA, _Float16* flatB, _Float16* hmx_tile0, _Float16* hmx_tile1) {
    const HVX_Vector v0 = *(HVX_Vector*)hmx_tile0;
    const HVX_Vector v1 = *(HVX_Vector*)hmx_tile1;
    const HVX_VectorPair vp = Q6_W_vdeal_VVR(v1, v0, -2);

    *((HVX_Vector*)(flatA)) = Q6_V_lo_W(vp);
    *((HVX_Vector*)(flatB)) = Q6_V_hi_W(vp);
}

// ---------------------------------------------------------------- //
// --- 主测试程序 ---
// ---------------------------------------------------------------- //
int main(void) {
    int res = AEE_SUCCESS;
    uint8_t *vtcm_base = NULL;
    uint32_t vtcm_size = 0;

    FARF(ALWAYS, "=== HVX VTCM Unit Test Start ===");

    res = hexkl_micro_hw_init(&vtcm_base, &vtcm_size);
    if (res != AEE_SUCCESS) return res;

    // res = hexkl_micro_hmx_lock();
    // if (res != AEE_SUCCESS) return res;

    // 模拟 VTCM 内存布局
    uint8_t *ptr = vtcm_base;
    _Float16 *a_ptr = (_Float16 *)vtcm_seq_alloc(&ptr, VLEN);
    _Float16 *b_ptr = (_Float16 *)vtcm_seq_alloc(&ptr, VLEN);
    _Float16 *tmp0  = (_Float16 *)vtcm_seq_alloc(&ptr, VLEN);
    _Float16 *tmp1  = (_Float16 *)vtcm_seq_alloc(&ptr, VLEN);
    _Float16 *c_ptr = (_Float16 *)vtcm_seq_alloc(&ptr, VLEN * 2);

    // // 1. 测试布局转换
    // for(int i=0; i<64; i++) { 
    //     a_ptr[i] = (_Float16)i; 
    //     b_ptr[i] = (_Float16)(i+100); 
    // }

    // flat_to_crouton(a_ptr, b_ptr, tmp0, tmp1);
    // // FARF(ALWAYS, "[PASS] temp0 first ele is %.1f\n", (float)tmp0[3]);
    // // 清除 a, b 内容验证恢复效果
    // memset(a_ptr, 0, VLEN);
    // memset(b_ptr, 0, VLEN);
    // crouton_to_flat(a_ptr, b_ptr, tmp0, tmp1);

    // if (a_ptr[1] == (_Float16)1 && b_ptr[1] == (_Float16)101) {
    //     FARF(ALWAYS, "[PASS] Flat <-> Crouton successful");
    // } else {
    //     FARF(ALWAYS, "[FAIL] Layout mismatch! a[0]=%.1f, b[0]=%.1f", (float)a_ptr[1], (float)b_ptr[1]);
    // }

    // 2. 测试权重乘法
    uint8_t *w_ptr = (uint8_t *)vtcm_seq_alloc(&ptr, VLEN);
    for (int i = 0; i < 128; i++) w_ptr[i] = 2;
    // FARF(ALWAYS, "[PASS] w_int[0] is %.1f", (float)w_ptr[1]);
    _Float16 *s_ptr = (_Float16 *)vtcm_seq_alloc(&ptr, VLEN * 2);
    for(int i=0; i<128; i++) s_ptr[i] = (_Float16)1.5f; // 所有 scale = 1.5

    HVX_Vector v_w = *(HVX_Vector*)w_ptr;
    
    HVX_Vector v_s_0 = *(HVX_Vector*)s_ptr;
    HVX_Vector v_s_1 = *(HVX_Vector*)(s_ptr+64);
    HVX_VectorPair v_s = Q6_W_vcombine_VV(v_s_1, v_s_0);
    
    HVX_VectorPair v_res = hvx_int8_w_fp16_s_mpy(v_w, v_s);
    _Float16 *out_ptr = (_Float16 *)vtcm_seq_alloc(&ptr, VLEN * 2);
    // *(HVX_VectorPair*)out_ptr = v_res;
    *(HVX_Vector*)out_ptr = Q6_V_lo_W(v_res);
    _Float16 *next_out_ptr= out_ptr + 64;
    *(HVX_Vector*)next_out_ptr = Q6_V_lo_W(v_res);

    // 验证：2 * 1.5 = 3.0
    if ((float)out_ptr[0] == 3.0f) {
        FARF(ALWAYS, "[PASS] Int8 * FP16 Scale = 3.0 successful");
    } else {
        FARF(ALWAYS, "[FAIL] Math error! Got %.2f", (float)out_ptr[0]);
    }


    // HVX_VectorPair w_fp16 = Q6_Whf_vcvt_Vub(v_w);
    // _Float16 *w_fp16_ptr = (_Float16*)vtcm_seq_alloc(&ptr, VLEN*2);
    // // *(HVX_VectorPair*)w_fp16_ptr = w_fp16;
    // *(HVX_Vector*)w_fp16_ptr = Q6_V_lo_W(w_fp16);
    // _Float16 *next_w_fp16_ptr = w_fp16_ptr + 64;
    // *(HVX_Vector*)next_w_fp16_ptr = Q6_V_lo_W(w_fp16);

    // // FARF(ALWAYS, "[PASS] w_fp16_ptr[0] is %.1f", (float)w_fp16_ptr[0]);
    
    // for (int i=0; i < 128; i++)
    // FARF(ALWAYS, "[PASS] w_fp16_ptr[%d] is %.1f", i, (float)w_fp16_ptr[i]);


    // hexkl_micro_hmx_unlock();
    FARF(ALWAYS, "=== HVX VTCM Unit Test Complete ===");
    return 0;
}