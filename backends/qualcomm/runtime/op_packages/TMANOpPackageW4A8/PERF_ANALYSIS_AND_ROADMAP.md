# TMANOpPackageW4A8 性能实现分析与开发迭代路线

本文回答一个核心问题：

`TMANOpPackageW4A8` 当前是不是还停留在标量计算，没有真正利用 HVX 做高度优化？

结论先说：

- 如果看“注册并实际接入图执行的正式算子路径”，答案是：`不是纯标量`。当前 `TMANPrecomputeA8 / TMANLinearA8 / TMANFinalizeA8` 的核心热点仍然复用了原始 `TMANOpPackage` 中的 `hvx_lut_ctor / hvx_tbl / hvx_bit_serial`，这些都是 HVX kernel。
- 但如果问“它是不是已经做成了专门面向 W4A8 的高度优化实现”，答案也是：`还没有`。当前 W4A8 更像是在原始 FP16/W4 TMAN 主干外面，增加了输入侧 A8 量化语义和输出侧 output-scale 量化反量化语义，整体仍是“HVX 主干复用 + 标量前后处理 + 若干未接线的 A8 原型 helper”的状态。

因此，最准确的描述是：

`当前 W4A8 不是纯标量实现，但它还不是一个真正 A8-native 的深度 HVX 优化版本。`

## 1. 直接证据：正式算子的实际调用链

## 1.1 原始 TMANOpPackage 的 FP16 路径

原始包 `backends/qualcomm/runtime/op_packages/TMANOpPackage` 的 3 个算子入口都非常直接：

- `TMANPrecompute` 直接分派到 `hvx_lut_ctor`
- `TMANLinear` 直接分派到 `hvx_tbl`
- `TMANFinalize` 直接分派到 `hvx_bit_serial`

对应代码：

- `src/ops/TMANPrecompute.cpp`
- `src/ops/TMANLinear.cpp`
- `src/ops/TMANFinalize.cpp`

这条路径的特点是：

- 外层 wrapper 很薄
- 热点计算基本都在 `include/hvx_funcs.h` 的 HVX 实现里
- 没有额外的输入/输出量化补丁逻辑

## 1.2 当前 W4A8 正式路径

`TMANOpPackageW4A8` 的注册入口在 `src/TMANOpPackageInterface.cpp`，真正注册到 HTP Core 的只有 3 个算子：

- `TMANPrecomputeA8`
- `TMANLinearA8`
- `TMANFinalizeA8`

对应正式调用链如下：

1. `TMANPrecomputeA8`
   先按外部 `input_scale` 做一遍逐元素的 activation 量化/反量化，然后调用 `hvx_lut_ctor`

2. `TMANLinearA8`
   直接调用 `hvx_tbl`

3. `TMANFinalizeA8`
   先调用 `hvx_bit_serial`，然后再做一遍逐元素的 output quant/dequant

也就是说，正式路径里：

- `LinearA8` 的核心 GEMM/LUT 查表计算仍是 HVX
- `PrecomputeA8` 和 `FinalizeA8` 则是在 HVX 主干外，又包了一层标量逻辑

## 1.3 一张表看清现状

| 阶段 | 原始 FP16 TMAN | 当前 W4A8 正式路径 | 判断 |
|---|---|---|---|
| Precompute | 直接 `hvx_lut_ctor` | 标量量化/反量化 + `hvx_lut_ctor` | 混合路径 |
| Linear | 直接 `hvx_tbl` | 直接 `hvx_tbl` | HVX 热点 |
| Finalize | 直接 `hvx_bit_serial` | `hvx_bit_serial` + 标量 output quant/dequant | 混合路径 |

所以用户如果问“现在是不是还完全停留在标量计算”，严格来说是不成立的。

但如果问“是不是已经做成完整的 A8 专用高性能 kernel”，这个判断又是成立的：也还没有。

## 2. 为什么会让人感觉它还是标量实现

这个判断之所以很容易产生，主要有两个原因。

## 2.1 `hvx_funcs.h` 里新增了一整套 A8 helper，而且大多是标量或半标量

`TMANOpPackageW4A8/include/hvx_funcs.h` 前半部分新增了大量 `tman_a8_*` 和 `tman_w4a8_*` helper，例如：

- `tman_a8_precompute_lut_scalar`
- `tman_a8_linear_scalar`
- `tman_a8_linear_from_packed_int4_w4g128_fp16_fast`
- `tman_a8_finalize_scalar`
- `tman_a8_finalize_to_qnn_int8_hvx`
- `tman_w4a8_precompute_fp16_to_packed_sym_g128`

这些函数里确实有不少是：

- 完全标量
- 或者只有局部步骤用了 HVX
- 或者只是“HVX store / HVX scale apply”，主循环本身仍是标量

例如：

- `tman_w4a8_precompute_fp16_to_packed_sym_g128`
  是一套完全标量的 W4A8 precompute 参考实现

- `tman_a8_linear_from_packed_int4_w4g128_fp16_fast`
  名字里有 `fast`，但主循环仍然是：
  - 标量展开 `qgb/lane/rt/mt`
  - 用 `tman_a8_group_dot_from_packed_int4_fast` 做标量 LUT 索引累加
  - 只有 `tman_a8_apply_scales_w4g128_fp16_hvx_4x32` 这一步才用 HVX 做一部分 scale apply

- `tman_a8_finalize_to_qnn_int8_hvx`
  也只是：
  - 先标量打包 output block
  - 再在满足条件时用 HVX 做 128B 写回

从“这些 helper 自身”的优化程度来看，确实还谈不上高度 HVX 化。

## 2.2 但这些 helper 目前大多没有接到正式算子上

这是最关键的一点。

从当前代码引用关系看：

- `TMANPrecomputeA8` 没有调用 `tman_w4a8_precompute_fp16_to_packed_sym_g128`
- `TMANLinearA8` 没有调用 `tman_a8_linear_from_packed_int4_w4g128_fp16_fast`
- `TMANFinalizeA8` 也没有调用 `tman_a8_finalize_to_qnn_int8_hvx`

真正接到正式算子上的，是下面这套老主干：

- `hvx_lut_ctor`
- `hvx_tbl`
- `hvx_bit_serial`

因此：

- `A8 helper 家族` 当前更像实验/候选/验证代码
- `正式图执行路径` 仍以原始 TMAN 的 HVX kernel 为主

## 3. 与原始 FP16 TMAN 实现的逐段对比

## 3.1 Precompute：W4A8 增加了一个额外标量前处理

### 原始 FP16 TMAN

`TMANPrecompute` 的行为很直接：

- 解析参数
- 根据 `group_size/symmetric` 选择模板参数
- 直接调用 `hvx_lut_ctor`

这意味着：

- activation 读一次
- LUT/ls/lb 在 HVX kernel 内生成
- 没有额外 temporary activation buffer

### 当前 W4A8

`TMANPrecomputeA8` 先做了两件原始 FP16 路径没有的事：

1. 读取外部 `input_scale`
2. 对每个 `x[k]` 执行：
   - `q_a = round_clip(x[k] / input_scale)`
   - `x_quantized[k] = q_a * input_scale`

然后再把这份 `x_quantized_aligned` 交给 `hvx_lut_ctor`。

这意味着 W4A8 precompute 比原始 FP16 路径多出：

- 一次额外的逐元素量化 pass
- 一次额外的临时 buffer 分配
- 一次额外的 128B 对齐处理

它的优点是：

- 语义清楚
- 与导出侧的 `input_scale` 一致
- 风险低，复用旧 kernel 快

它的代价是：

- 还不是 A8-native precompute
- activation 被“先量化再反量化”为 fp16，才进入旧 LUT 构造器
- 数据路径上存在额外内存流量和标量开销

结论：

`PrecomputeA8` 当前属于“数学语义先落地，性能尚未收敛”的实现。

## 3.2 Linear：W4A8 热点计算本质上仍是原始 HVX kernel

### 原始 FP16 TMAN

`TMANLinear` 在 wrapper 里解析完参数后，会根据：

- `group_size`
- `bits`
- `symmetric/zero_point`

把计算分派到对应模板实例化的 `hvx_tbl`。

### 当前 W4A8

`TMANLinearA8` 虽然只支持一个极窄配置：

- `group_size = 128`
- `bits = 4`
- `symmetric = true`
- `gemm_n = 1`

但真正计算时依然是：

```cpp
hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 128, false, 4, TILE_K, LUT_G, true>(...)
```

这与原始 FP16 路径在 `w4g128 symmetric=True` 情况下调用的主 kernel 是同一体系。

因此有两个重要结论：

1. 当前 `LinearA8` 的真正热点不是标量
2. 它也还没有利用“配置已经固定为 W4G128 symmetric + N=1”这个事实去做更激进的专用 kernel specialization

换句话说，它现在是：

- `性能层面`：借用了老 kernel 的成熟 HVX 路径
- `结构层面`：还没长成真正的 W4A8 专用实现

## 3.3 Finalize：主体仍是 HVX，但 output_scale 后处理是标量

### 原始 FP16 TMAN

`TMANFinalize` 非常简单：

- 根据 `bits`
- 直接调用 `hvx_bit_serial<2/4>`

没有额外后处理。

### 当前 W4A8

`TMANFinalizeA8` 做的是：

1. `hvx_bit_serial<4>(...)`
2. 再逐元素执行：
   - `y_q = round_clip(y_fp / output_scale)`
   - `y_fp = y_q * output_scale`

因此：

- bit-serial 合成仍是 HVX
- output activation quant/dequant 这一步目前还是标量 loop

这一步相对 `LinearA8` 的总体耗时通常不会最大，但它确实是纯新增开销，而且是可以进一步 fuse/vectorize 的。

## 4. 当前 W4A8 的真实状态归纳

把原始 FP16 TMAN 和当前 W4A8 放在一起看，可以把现状归纳成 4 句话：

1. `TMANOpPackageW4A8` 不是纯标量实现
2. 它的核心 GEMM/LUT/bit-serial 热点仍在原始 TMAN 的 HVX kernel 上
3. 它现在还没有形成真正的 A8-native 专用高性能内核
4. 它目前更像“以最小风险接入 A8 语义”的过渡版本，而不是最终性能版本

## 5. 当前实现里最值得优先优化的地方

如果从性能收益和实现成本一起看，当前优先级大致如下。

## 5.1 第一优先级：`LinearA8` 专用 kernel 化

原因：

- 端到端耗时中，`LinearA8` 仍是绝对热点
- 当前虽然用了 HVX，但它复用的是原始更泛化的 `hvx_tbl`
- 而 W4A8 现在实际只支持 `W4G128 symmetric + N=1`

这说明还有明显的 specialization 空间：

- 可以删掉不需要的 `ZeroPoint` 分支
- 可以删掉不需要的 `bits/group_size` 泛化
- 可以直接针对 `N=1`、`W4G128`、固定 `TileK` 设计 kernel
- 可以为当前权重打包布局设计更贴身的读法

这是最可能带来主要速度提升的部分。

## 5.2 第二优先级：PrecomputeA8 的 A8-native 化

当前 `PrecomputeA8` 最大的问题不是“没有 HVX”，而是：

- 它先把激活量化到 int8
- 然后又恢复回 fp16
- 再喂给旧 `hvx_lut_ctor`

这条链路在数学语义上没问题，但从数据流角度看并不干净。

更理想的路径应该是：

- 直接接收 `x + input_scale`
- 在 kernel 内完成量化
- 直接构造 A8 对应的 packed LUT/ls/lb

这样可以减少：

- 一次额外临时 buffer
- 一次标量循环
- 一次不必要的 fp16 中间态

## 5.3 第三优先级：FinalizeA8 的融合与向量化

`FinalizeA8` 当前主要问题是：

- 先 `hvx_bit_serial`
- 再做一遍标量 output quant/dequant

优化方向很直接：

- 把 `output_scale` 融到 `hvx_bit_serial` 后半段
- 或者写一个 `hvx_bit_serial_with_output_scale`
- 或者进一步支持输出 int8，再由图优化决定是否反量化

这一块相较于 `LinearA8` 的收益通常更小，但实现相对直接，也很适合当作中期优化项。

## 5.4 第四优先级：代码组织与误导性实现清理

当前 `TMANOpPackageW4A8` 里存在几个会明显增加维护成本的点：

- `include/hvx_funcs.h` 里有大量 A8 helper，但大多未接线
- `src/ops/TMANFinalize.cpp` 仍然存在，但并没有在 `TMANOpPackageInterface.cpp` 里注册
- `Makefile` 又是通过 `src/ops/*.cpp` 的 wildcard 编译所有 op 源文件

这会导致：

- 读代码时很容易误判真正热点路径
- 新人容易把实验 helper 当成正式实现
- 构建产物里混入未注册算子的对象文件

这类问题未必直接影响运行速度，但会明显影响后续性能开发效率。

## 6. 对应的优化思路

下面给出更落地的优化建议。

## 6.1 短期优化：在不推翻当前架构的前提下提速

### 目标

保留“复用原始 HVX 主干”的总体架构，先消掉显眼的额外开销。

### 建议

1. 为 `PrecomputeA8 / LinearA8 / FinalizeA8` 分别建立 device 侧 cycle 基线
2. 给 `PrecomputeA8` 的 activation quantization loop 做 HVX 化
3. 去掉 `std::vector + 手动对齐` 带来的额外分配开销，改成可复用 scratch
4. 给 `FinalizeA8` 写 `hvx_bit_serial_with_output_scale`
5. 明确标注/隔离未接线 helper，避免未来误接入较慢实现

### 预期收益

- 风险小
- 不影响现有数值语义
- 能比较快地把“前后处理纯标量开销”压下去

## 6.2 中期优化：实现真正的 W4A8 专用内核

### 目标

不再把 W4A8 当成“旧 FP16/W4 主干外包一层”，而是直接做专用 kernel。

### 建议

1. 新写 `hvx_lut_ctor_a8_sym_g128`
   直接消化 `input_scale`

2. 新写 `hvx_tbl_w4a8_sym_g128_n1`
   专门针对：
   - `group_size = 128`
   - `bits = 4`
   - `symmetric = true`
   - `gemm_n = 1`

3. 为权重和 scale 布局重新确认最优打包方式
   目标是减少 generic `hvx_tbl` 中不再需要的泛化代价

4. 重新评估 autosplit 规则
   因为专用 kernel 的 tile 优选范围可能与原始 `TMANLinear` 不同

### 预期收益

- 这是最可能带来主要速度提升的一步
- 也是从“过渡版本”走向“正式性能版本”的关键一步

## 6.3 长期优化：图级融合与接口演化

### 目标

减少中间 buffer，进一步靠近最终部署形态。

### 建议

1. 支持 int8 output 变体
   让后续节点或图优化决定是否反量化

2. 考虑 `LinearA8 + FinalizeA8` 融合
   减少 `c` 这个 `M * bits` float scratch 的带宽成本

3. 视业务需要支持 prompt 阶段 `N > 1`
   当前实现完全收敛到 decoder `N = 1`

4. 增加 cost model 和 profile-guided dispatch

### 预期收益

- 更好的系统级性能
- 更少的中间内存流量
- 更接近生产级端到端优化

## 7. 建议的开发迭代路线

下面给出一份适合工程推进的迭代计划。

## Phase 0：基线与清理

### 目标

先明确“现在慢在哪里”，并把代码组织上的歧义清掉。

### 工作项

1. 为 `PrecomputeA8 / LinearA8 / FinalizeA8` 建立 profile 基线
2. 在 V75 真机和 x86 HTP simulator 上分别记录代表 shape 的耗时
3. 把未接线 helper 标记为 `experimental/reference`
4. 清理未注册但仍参与编译的历史 op 源文件，或至少在文档中明确其状态

### 验收标准

- 拿到分阶段耗时占比
- 团队对“当前真正热点路径”有一致认知

## Phase 1：去掉前后处理中的显式标量瓶颈

### 目标

不改 `LinearA8` 主内核，先压缩 Precompute/Finalize 的纯额外开销。

### 工作项

1. `PrecomputeA8` 量化 loop HVX 化
2. scratch buffer 复用
3. 新增 `hvx_bit_serial_with_output_scale`
4. 保证现有测试全部通过

### 验收标准

- 行为与当前实现保持一致
- `PrecomputeA8` 和 `FinalizeA8` 单独耗时明显下降

## Phase 2：专用 `LinearA8` kernel

### 目标

针对 `W4G128 symmetric + N=1` 做真正的专用 `hvx_tbl` 替代。

### 工作项

1. 设计 `hvx_tbl_w4a8_sym_g128_n1`
2. 明确权重/scale 的最佳打包格式是否需要调整
3. 引入新旧 kernel 对照测试
4. 在大 shape 上验证收益

### 验收标准

- 端到端性能出现主要提升
- 关键大 shape 上稳定优于当前复用版 `hvx_tbl`

## Phase 3：A8-native Precompute

### 目标

让输入侧真正变成 A8-native，而不是“量化再反量化后喂旧 kernel”。

### 工作项

1. 实现 `hvx_lut_ctor_a8_sym_g128`
2. 去掉多余 temporary fp16 activation buffer
3. 重新验证 `ls/lb` 数学语义

### 验收标准

- PrecomputeA8 数据路径更干净
- 无额外数值回退

## Phase 4：图级融合与接口扩展

### 目标

把中间 scratch 和后处理进一步压缩。

### 工作项

1. 评估 `LinearA8 + FinalizeA8` 融合
2. 评估 int8 output 变体
3. 视需求扩展到 `N > 1`

### 验收标准

- 中间带宽下降
- 系统级吞吐继续提升

## 8. 推荐的实际优先顺序

如果只按“投入产出比”排序，我建议这样做：

1. 先做 Phase 0
2. 然后优先做 `LinearA8` 专用 kernel 方案设计与 profiling 验证
3. 再做 `PrecomputeA8` A8-native 化
4. 再做 `FinalizeA8` 融合
5. 最后做接口扩展和图级融合

原因很简单：

- `LinearA8` 仍然是主要热点
- `Precompute/Finalize` 的标量补丁虽然不优雅，但不是当前最大头
- 先把主算子做成真正的专用 kernel，整体收益最大

## 9. 最终判断

最后再用一句话总结当前实现状态：

`TMANOpPackageW4A8` 当前已经用上了原始 TMAN 的 HVX 主干，因此不是“还停留在标量实现”；但它仍然不是一个完成度很高的 A8 专用高性能版本，而是一个“用标量前后处理把 A8 语义接到旧 HVX 主干上”的过渡实现。真正需要投入优化的重点，是把这条路径从“复用型适配”推进到“W4A8-native 专用 kernel”。`
