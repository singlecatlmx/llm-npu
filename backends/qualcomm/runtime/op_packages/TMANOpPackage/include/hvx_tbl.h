#include <vector>
#include <cmath>
#include <algorithm>

// 辅助函数：模拟 4-bit 提取
inline uint8_t get_4bit(const uint8_t* w_ptr, int index) {
    // 假设 w 存储方式是每字节两个 4-bit 索引
    if (index % 2 == 0) return w_ptr[index / 2] & 0x0F;
    else return (w_ptr[index / 2] >> 4) & 0x0F;
}

template <typename LType, typename XType, typename CType>
int simple_tbl_plain_cpp(
    int32_t GemmM, int32_t GemmK, int32_t GemmN,
    const LType *l,     // LUT 表 (由 ctor 生成)
    const float *ls,    // 激活值 Scale
    const float *lb,    // 激活值 Bias
    const uint8_t *w,   // 权重的索引 (4-bit packed)
    const XType *s,     // 权重的 Scale/ZeroPoint
    CType *c,           // 输出结果
    int GroupSize,      // 权重分组大小
    int ActGroupSize,   // 激活值分组大小
    bool ZeroPoint,     // 是否有零点
    int Bits            // 权重位数 (2-4 bit)
) {
    const int g = 4; // 每 4 个元素一组 LUT
    const int32_t Q = GemmK / g;
    const int32_t lut_size = 16;
    
    const int32_t q_group_size = GroupSize / g;
    const int32_t q_act_group_size = ActGroupSize / g;
    
    // 初始化输出
    for(int i = 0; i < GemmM; ++i) c[i] = 0;

    // --- 核心计算逻辑 ---
    // 遍历 M 维度 (输出通道/行)
    for (int32_t m = 0; m < GemmM; ++m) {
        
        float m_accumulator = 0.0f; // 这一行的最终累加值

        // 遍历 Q 维度 (K 维度被压缩后的维度)
        for (int32_t q = 0; q < Q; ++q) {
            
            // 1. 获取权重索引 (从压缩的 uint8 中提取 4-bit)
            // 这里的索引计算模拟了原代码中 vec_p/vec_q 的寻址逻辑
            // w 的形状大致为 [Q, M] (经过重排)
            uint8_t weight_idx = get_4bit(w + (m * Q / 2), q); 

            // 2. 查表 (Table Lookup)
            // 从预计算的 LUT 中直接拿到激活值的线性组合
            int16_t lut_val = l[q * lut_size + weight_idx];

            // 3. 计算这一小组 (g=4) 的部分和
            // 公式: partial = lut_val * act_scale
            float act_scale = ls[q / q_act_group_size];
            float partial_sum = (float)lut_val * act_scale;

            // 4. 应用激活值 Bias (lb)
            // 只有当到达权重组的特定边界时，才加入 bias 修正
            // 原代码逻辑：if (q_group_tail) partial_sum += lb
            if ((q + 1) % q_group_size == 0) {
                partial_sum += lb[q / q_group_size];
            }

            // 5. 应用权重 Scale (s)
            // 权重 Scale 通常是按组（GroupSize）给出的
            // s 包含 [weight_scale, weight_zp] (如果 ZeroPoint 为 true)
            float weight_scale = (float)s[(m * (Q / q_group_size)) + (q / q_group_size)];
            
            // 如果有非对称量化 ZeroPoint，会有额外的补偿计算
            if (ZeroPoint) {
                float weight_zp = (float)s[/* 偏移地址获取 zp */];
                // 原代码：c = (c * ls + lb + zp * lb * 2) * s
                // 这里简化表达核心意图
                partial_sum += (weight_zp * 2.0f) * lb[q / q_group_size];
            }

            m_accumulator += partial_sum * weight_scale;
        }

        c[m] = m_accumulator;
    }

    return 0;
}