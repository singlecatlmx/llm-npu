//==============================================================================
// Auto Generated Code for TMANOpPackage
//==============================================================================

#include <cstring> // memcpy (scalar tail fp16 conversion)
#include "../../include/hvx_funcs.h" // VLEN, vmem, HVX_Vector, l2fetch
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

BEGIN_PKG_OP_DEFINITION(PKG_TMANAdd);

// op execute function declarations
template <typename TensorType>
GraphStatus
tmanaddImpl(TensorType& out_0, const TensorType& a, const TensorType& b);

// forward declaration of sample cost function
static float tmanaddCostFunc(const Op* op);

DEF_PACKAGE_OP((tmanaddImpl<Tensor>), "TMANAdd")

//   - 输出 "*" 放在 Tcm
//   - 输入 "a" 放在 Tcm（来自 TMANFinalize 的输出）
//   - 输入 "b" 放在 MainMemory（来自 Embedding/Gather 的输出）
//   - Flat 约束确保所有张量是连续布局
DEF_TENSOR_PROPERTIES(
    Op("TMANAdd", "a", "b"),
    Flat("*", "a", "b"),
    Tcm("*", "a"),
    MainMemory("b"))

/* execute functions for ops */

//==============================================================================
// TMANAdd 优化实现
//
// 内存布局:
//   a   → TCM  (来自 TMANFinalize，已在片上高速缓存)
//   b   → MainMemory / DDR (来自 Embedding/Gather)
//   out → TCM  (输出留在片上，给下游算子直接使用)
//
// 性能瓶颈: 读取 b 的 DDR 延迟 (~100-300 cycles vs TCM ~1 cycle)
//
// 优化策略:
//   1. 加深 L2 预取窗口: 4 → 8 chunks (2KB → 4KB)
//   2. 加大初始预取: min(8KB, total) 一次性拉入 L2
//      确保主循环第一次迭代不会 stall 在 DDR 上
//   3. 交错 load/compute 保持不变 (2 live regs, 良好 VLIW 调度)
//
//==============================================================================

template <typename TensorType>
GraphStatus
tmanaddImpl(TensorType& out_0, const TensorType& a, const TensorType& b) {
  using XType = __fp16;

  const int32_t total_elements =
      out_0.dims()[0] * out_0.dims()[1] * out_0.dims()[2] * out_0.dims()[3];

  const XType* a_ptr = (const XType*)a.raw_data_const();
  const XType* b_ptr = (const XType*)b.raw_data_const();
  XType* out_ptr = (XType*)out_0.raw_data();

  // 每个 HVX 向量 = 64 个 fp16 = 128 bytes
  constexpr int32_t VecSize = VLEN / (int32_t)sizeof(XType); // 64
  constexpr int32_t UNROLL = 4;
  constexpr int32_t ChunkElems = VecSize * UNROLL; // 256 elements
  constexpr int32_t ChunkBytes = VLEN * UNROLL; // 512 bytes
  constexpr int32_t PF_AHEAD = 8; // 预取深度: 8 chunks = 4KB

  // ---- L2 预取: 将 b 的前 8 段 (4KB) 从 DDR 异步拉到 L2 缓存 ----
  // a/out 在 TCM 无需预取; b 在 DDR, 是唯一的延迟瓶颈
  // 初始预取覆盖 4KB, 确保主循环前几次迭代都有 L2 命中
  {
    int32_t total_bytes = total_elements * (int32_t)sizeof(XType);
    int32_t pf_bytes = MIN(ChunkBytes * PF_AHEAD, total_bytes);
    if (pf_bytes > 0) {
      l2fetch(b_ptr, (uint32_t)pf_bytes, (uint32_t)pf_bytes, 1, 0);
    }
  }

  int32_t i = 0;

  // ---- 主循环: 4x 展开, 交错 load/compute 隐藏 DDR 延迟 ----
  // 每对 {a[k], b[k]} 加载后立即计算+存储:
  //   - 寄存器压力低 (同时只有 2 个 live vector)
  //   - VLIW 编译器可将 TCM store 与下一轮 DDR load 并行调度
  for (; i + ChunkElems <= total_elements; i += ChunkElems) {
    // 滚动预取 b 的后续数据 (提前 8 个 chunk ≈ 4KB)
    // DDR 延迟 ~100-200ns, 每 chunk ~15ns → 8 chunks 覆盖 ~120ns
    {
      const XType* pf = b_ptr + i + ChunkElems * PF_AHEAD;
      if (pf < b_ptr + total_elements) {
        l2fetch(pf, ChunkBytes, ChunkBytes, 1, 0);
      }
    }

    // 交错模式: load a[k] + b[k] → compute → store, 立刻处理下一对
    // fp16 → qf16 加法 → fp16 (V75 验证可靠路径)
    HVX_Vector va0 = vmem(a_ptr + i);
    HVX_Vector vb0 = vmem(b_ptr + i);
    vmem(out_ptr + i) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(va0, vb0));

    HVX_Vector va1 = vmem(a_ptr + i + VecSize);
    HVX_Vector vb1 = vmem(b_ptr + i + VecSize);
    vmem(out_ptr + i + VecSize) =
        Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(va1, vb1));

    HVX_Vector va2 = vmem(a_ptr + i + VecSize * 2);
    HVX_Vector vb2 = vmem(b_ptr + i + VecSize * 2);
    vmem(out_ptr + i + VecSize * 2) =
        Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(va2, vb2));

    HVX_Vector va3 = vmem(a_ptr + i + VecSize * 3);
    HVX_Vector vb3 = vmem(b_ptr + i + VecSize * 3);
    vmem(out_ptr + i + VecSize * 3) =
        Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(va3, vb3));
  }

  // ---- 残余完整向量 (1-3 个 vector) ----
  for (; i + VecSize <= total_elements; i += VecSize) {
    HVX_Vector va = vmem(a_ptr + i);
    HVX_Vector vb = vmem(b_ptr + i);
    vmem(out_ptr + i) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(va, vb));
  }

  // ---- 尾部处理: HVX 向量运算 + partial store ----
  // 仍用 HVX 做加法 (安全过读, QNN 张量分配有 vector 对齐 padding),
  // 通过对齐临时缓冲区 + memcpy 只写入有效元素, 避免越界写
  {
    int32_t tail = total_elements - i;
    if (tail > 0) {
      HVX_Vector va = vmem(a_ptr + i);
      HVX_Vector vb = vmem(b_ptr + i);
      HVX_Vector vsum = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_VhfVhf(va, vb));

      XType __attribute__((aligned(VLEN))) tmp[VecSize];
      vmem(tmp) = vsum;
      memcpy(out_ptr + i, tmp, tail * sizeof(XType));
    }
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float tmanaddCostFunc(const Op* op) {
  /*
   * add code here
   * */

  float cost = 0.0; // add cost computation here
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_TMANAdd);