# TMAN W4A8 实现与优化记录

## 1. 文档目的

这份文档总结当前 `TMANOpPackage-dev` 中 W4A8 路径的实现状态、接口约束、优化演进过程，以及当前已经完成的验证结论。

当前覆盖的三个算子是：

- `TMANPrecomputeA8`
- `TMANLinearA8`
- `TMANFinalizeA8`

目标配置是：

- 权重：`int4`
- 激活：`int8`
- 量化模式：`W4G128`
- `symmetric = true`

整体目标分三层：

1. 先把接口和数学语义做对
2. 在不破坏正确性的前提下逐步压缩传输和减少中间开销
3. 再逐步向原 FP16 TMAN 链路的调度方式和高性能实现靠拢

---

## 2. 当前外部 ABI

当前 ABI 由以下文件定义：

- [TMANOpPackageHtp.xml](/home/tang/executorch/op_gen/TMANOpPackage-dev/config/TMANOpPackageHtp.xml)
- [TMANPrecomputeA8.cpp](/home/tang/executorch/op_gen/TMANOpPackage-dev/src/ops/TMANPrecomputeA8.cpp)
- [TMANLinearA8.cpp](/home/tang/executorch/op_gen/TMANOpPackage-dev/src/ops/TMANLinearA8.cpp)
- [TMANFinalizeA8.cpp](/home/tang/executorch/op_gen/TMANOpPackage-dev/src/ops/TMANFinalizeA8.cpp)

当前张量 ABI 如下：

- `x`：`INT8`，shape 为 `[1, 1, N, K]`
- `l`：`INT16`，shape 为 `[1, 1, N, G * (group_size / 16) * 64]`
- `qweight`：`UINT8`，shape 为 `[1, M, G, group_size / 2]`
  每个 byte 打包两个逻辑 int4 权重
- `scales`：`FLOAT16`，shape 为 `[1, 1, M, G]`
- `c`：`FLOAT32`，shape 为 `[1, 1, N, M]`
- `y`：`INT8`，shape 为 `[1, 1, N, M]`

当前运行时约束：

- 只支持 `bits = 4`
- 只支持 `group_size = 128`
- 只支持 `symmetric = 1`

也就是说，这条链路现在是面向一个非常明确、非常窄的目标配置，而不是一个通用量化线性层实现。

---

## 3. 当前实现结构

核心 helper 代码集中在：

- [hvx_funcs.h](/home/tang/executorch/op_gen/TMANOpPackage-dev/include/hvx_funcs.h)

当前 helper 层大致可以分成几类：

1. 布局和 offset 计算
2. LUT 构造
3. packed int4 解码与 group dot
4. `LinearA8` 的 tiled scalar / fast path
5. `FinalizeA8` 的 round-clamp 与输出格式处理

当前三个 op 的职责都刻意保持较薄：

- `TMANPrecomputeA8`
  主要负责检查输入 shape，然后调用 LUT 构造 helper
- `TMANLinearA8`
  主要负责检查 packed layout，然后调用线性层 helper
- `TMANFinalizeA8`
  主要负责检查输出 shape，然后把 float accumulator 收敛到 int8

这样做的目的是：

- 把数学逻辑集中在一个地方维护
- 后续切换更快的 kernel 时，不需要反复改 op 接口层

---

## 4. 优化演进过程

### 阶段 1：最小可用 W4A8 路径

第一版目标非常直接：

- 先做出一条最小可用、最容易验证的 W4A8 三段式
- 优先验证数学语义，不追求性能

这一版的特点是：

- 逻辑 int4 权重最初用 `int8` 容器传输
- `scales` 最初用 `float32`
- 整体实现是简单 scalar 路径

它的价值在于：

- 先建立了第一条能跑通的 A8 正确性链路
- 让后面的优化都有一个明确的 reference 基线

### 阶段 2：修正 QNN `INT8` transport 语义

在早期调试中发现，当前这条路径下 QNN 对 `INT8` 的 transport 存在 sign-bit flip 行为。

修正方式：

- `PrecomputeA8` 在构造 LUT 前先恢复成真正的有符号 `int8`
- `FinalizeA8` 在离开 custom op 前先做一次预翻转

这一步解决了早期这些典型问题：

- `zero-input` 结果不为 0
- `zero-weight` 结果不为 0
- `FinalizeA8` 输出读回后直接出现 `-128`

### 阶段 3：移除大块临时 `vector` 拷贝

最早能工作的版本里，A8 三个 op 内部都有整块 `std::vector` 临时缓冲：

- `TMANPrecomputeA8`
- `TMANLinearA8`
- `TMANFinalizeA8`

后面把 transport 处理下沉到 helper，去掉了这些整块临时拷贝。

关键 helper 包括：

- `tman_a8_precompute_lut_from_qnn_int8_scalar`
- `tman_a8_finalize_to_qnn_int8_scalar`

收益：

- op 内部中间开销更低
- 热路径更干净
- 更接近后续继续做 fast path 的形态

### 阶段 4：把 `LinearA8` 改成 tile 化内核

最初 `LinearA8` 是逐行逐输出的普通 scalar 循环。

后面改成了 tile 化结构：

- row tile：`4`
- output tile：`8`

这一步还不是 HVX kernel，但结构上已经更接近原 FP16 TMAN 链路。

相关 helper：

- `tman_a8_linear_from_qnn_int4_tiled_scalar`
- 后续又演进为 packed 权重版本

### 阶段 5：补 conservative autosplit

为了让 A8 在大图里更接近原 FP16 链路的调度方式，给 `TMANLinearA8` 补了一条保守 autosplit：

- 只沿输出通道 `M` 切
- `l` 不切
- `qweight` 与 `scales` 跟着 `M` 一起切

对应文件：

- [TMANLinearA8.cpp](/home/tang/executorch/op_gen/TMANOpPackage-dev/src/ops/TMANLinearA8.cpp)

这里特别要注意：

- 这一步保留了 split 逻辑
- 但没有把 A8 直接推成和 FP16 一样的 TCM placement

原因是之前做过实验：

- 一旦把 `LinearA8` 的相关张量往 FP16 那种 TCM 方式推
- x86 prepare 会触发 `q::flat_from_vtcm` 失败

所以目前做法是：

- 保留安全的 autosplit
- 暂时不强推 A8 的完整 TCM 链

### 阶段 6：把 `qweight` 压成真正的 packed int4

最早工作的 A8 版本里，逻辑 int4 权重仍然是用 `int8` 容器传的。

这一步把它改成了真正的 nibble-packed 传输：

- `qweight` 改成 `UINT8`
- shape 从 `[1, M, G, group_size]` 改成 `[1, M, G, group_size / 2]`
- 测试也同步改成每个 byte 打包两个权重

对应 helper：

- `tman_a8_weight_packed_offset`
- `tman_a8_group_dot_from_packed_int4`
- `tman_a8_linear_from_packed_int4_tiled_scalar`

这一步的直接收益是：

- 权重带宽减半
- ABI 也更接近真正想要的 W4 传输形式

### 阶段 7：把 `scales` 压成 `FLOAT16`

之后继续压缩传输，把 `scales` 从 `FLOAT32` 压成了 `FLOAT16`：

- XML 改成 `QNN_DATATYPE_FLOAT_16`
- `LinearA8` 按 `__fp16` 读取 scale
- 测试在 graph create 前先把 float scale 压成 fp16

这一步的特点是：

- 只压 transport
- 不改 golden 的 float 计算逻辑
- 所以可以明确知道它没有改变数学语义

### 阶段 8：packed int4 解码 fast path

压完 packed 权重之后，`LinearA8` 最热的部分就变成了 packed nibble 解码。

这一步加了：

- 单 byte 的 table-driven decode
- `tman_a8_group_dot_from_packed_int4_fast`

这样就不再需要每次都做逐 lane、逐 bit 的重复解码。

### 阶段 9：`W4G128 symmetric` 专用 fast path

在当前配置已经非常收敛后，又加了一条专门针对：

- `W4G128`
- `symmetric = true`
- packed int4
- fp16 scales

的专用快路径：

- `tman_a8_group_acc_w4g128_fast`
- `tman_a8_linear_from_packed_int4_w4g128_fp16_fast`

这条路径的特点是：

- 假设 `qgroups_per_group = 32`
- 假设 `qgroup_vector_blocks = 8`
- 这样 group 内部的循环可以按固定形状写
- 动态判断和动态布局恢复开销更小

在 `TMANLinearA8` 里现在的 dispatch 是：

- 如果 layout 正好是目标 `W4G128`，走专用 fast path
- 否则走当前 tiled scalar fallback

这一步的意义在于：

- 先在当前唯一目标配置上做专用优化
- 不影响其他 fallback 路径
- 为后面继续往 HVX fast path 推进打基础

### 阶段 10：`FinalizeA8` HVX fast path

在 `LinearA8` 已经有专用快路径之后，最适合继续做 HVX 化的是 `FinalizeA8`。

原因很直接：

- `FinalizeA8` 的输入输出语义简单
- 只需要做 `float32 -> round -> clamp -> int8`
- 不牵扯 packed weight、group 结构或 LUT layout

当前实现已经改成：

- 优先走 HVX fast path
- 尾部不足一个完整向量块时走现有 scalar tail

当前 HVX 处理链是：

1. `float32 -> int32`，使用现成 rounding helper
2. clamp 到 `[-127, 127]`
3. `int32 -> int16` 饱和压缩
4. `int16 -> int8` 饱和打包
5. 最后做 QNN `INT8` transport 所需的 sign-flip

这一步的意义在于：

- `FinalizeA8` 已经不再完全依赖纯标量循环
- 不会影响前面已经稳定的 `PrecomputeA8 / LinearA8` 路径
- 后续如果继续推进整条 A8 链的 HVX 化，这一段已经先打好了基础

### 阶段 11：完整 TCM 链闭环打通

前面的几个阶段里，A8 一直没有完全复刻原 FP16 的 TCM handoff。

这一轮重新对齐了三段式 placement：

- `TMANPrecomputeA8`
  允许 `x` 和输出 `l` 进入 TCM
- `TMANLinearA8`
  让 `l / qweight / scales / c` 按原 TMAN 思路进入 TCM
- `TMANFinalizeA8`
  在 TCM 中消费 `c`

也就是说，现在这条链已经重新收敛成：

- `PrecomputeA8 -> LinearA8 -> FinalizeA8`
- 中间 `l` 和 `c` 都可以在 TCM 中闭环流动

这个状态不是停留在理论设计，而是已经经过现有回归验证。

从现象上也能看到变化：

- 同样一套 x86 测试
- 打通完整 TCM 闭环之后，DDR `read / write` 相比之前进一步下降

这说明当前链路已经明显比早期版本更接近原 FP16 TMAN 的真实数据流。

### 阶段 12：`LinearA8` full-tile 专用快路径

在 `W4G128 symmetric` 专用快路径已经可用之后，又进一步加了一层针对常见 tile 的专用实现。

当前最常见的 tile 形态是：

- row tile = `4`
- output tile = `8`

因此这一步新增的是：

- full-tile `4x8` 专用快路径
- 把 `lut / weight / scale` 的地址恢复尽量提前到组循环外层
- 减少热循环内部重复的 offset 计算

这一步仍然没有改变外部 ABI，也没有改变数学语义，目的只是：

- 降低当前 `LinearA8` 专用 fast path 的标量控制开销
- 让编译器更容易对固定 tile 进行展开
- 继续为后续真正的 HVX `LinearA8` kernel 铺路

### 阶段 13：A8 自定义 autosplit 对齐原 FP16 链路

原 FP16 链路里，`TMANLinear` 使用了自定义 autosplit shape helper，而不是简单的通用切片。

原因是：

- FP16 旧链路切的是 packed chunk 维
- 输出维和 `qweight/scales` 的物理维并不是直接一一对应

而当前 A8 ABI 与 FP16 不同：

- 输出通道 `M` 直接暴露在
  - `qweight` 的 dim=1
  - `scales` 的 dim=2

所以这一步做的不是“照搬 FP16 helper”，而是按 A8 ABI 新实现了一套自定义 autosplit：

- 仍然沿输出通道 `M` 做 split
- `l` 不切
- `qweight` 在 dim=1 跟着切
- `scales` 在 dim=2 跟着切
- 使用自定义 shape helper 保证 start/size 语义和 A8 当前布局一致

这一步的意义在于：

- 让 A8 的 autosplit 真正与自己的 ABI 对齐
- 不再依赖通用 slice 规则去“碰运气”
- 更接近原 FP16 链路“按实际物理布局做定制切分”的思路

### 阶段 14：`LinearA8` full-tile group 级共享累加优化

在 `W4G128 symmetric` 的 full-tile 快路径已经存在之后，又继续做了一层更细的内核优化：

- 对 `4x8` full tile 引入 group 级共享累加 helper
- 把同一个 group 内的：
  - LUT 基址
  - weight group 基址
  - scale tile
  尽量提前缓存
- 让一个 `row_tile x m_tile` 的 group 累加共享同一层循环结构

对应思路可以概括成：

- 以前是“每个 row、每个 output 分别去做 group 累加”
- 现在是“一个完整 `4x8` tile 一起做 group 累加”

这一步的收益点主要是：

- 减少热路径里重复的 offset 恢复
- 减少函数调用和内层循环控制开销
- 继续把当前 fast path 往真正的 micro-kernel 形态推进

这一步仍然没有改变外部 ABI，也没有改变测试语义。

### 阶段 15：`LinearA8` full-tile `4x32` 扩展

在 full-tile `4x8` 专用快路径已经稳定后，又进一步把目标布局下的大输出通道场景扩展成：

- 优先使用 `4x32` output tile

这个版本的思路是：

1. 对目标 `W4G128 symmetric` 布局，优先使用更大的 output tile
2. 在 `4x32` full tile 上做 group 级共享累加
3. 对 scale 融合使用更适合当前 tile 规模的路径
4. 仍然保留：
   - `4x8` 专用路径
   - generic fallback

这样做的目的，是让：

- 大输出通道场景
- 尤其是 autosplit 之后的每个 chunk

能用上更接近真正 kernel 的 tile 规模，而不是始终停留在较小的 output tile。

这一步的收益主要体现在：

- 减少 tile 外层循环次数
- 进一步降低热路径的控制开销
- 让 `LinearA8` 更接近后续真正 HVX kernel 的组织形态

---

## 5. 当前测试覆盖

主测试文件：

- [test_tman_linear_a8.cpp](/home/tang/executorch/op_gen/test/test_tman_linear_a8.cpp)

Makefile：

- [Makefile.linear_a8](/home/tang/executorch/op_gen/test/Makefile.linear_a8)

当前 x86 覆盖包括：

1. `PrecomputeA8 -> LinearA8` accumulator 对比
2. 完整 `PrecomputeA8 -> LinearA8 -> FinalizeA8`
3. baseline case
4. row / output / group tail case
5. zero-input case
6. zero-weight case
7. saturating case
8. autosplit 大输出通道 case

最新验证命令：

```bash
make -f Makefile.linear_a8 run_x86
```

最新结果：

- `Passed: 7`
- `Failed: 0`

后续加入 autosplit 大输出通道验证后，当前最新 `run_x86` 结果为：

- `Passed: 9`
- `Failed: 0`

离线编译路径也已验证：

```bash
make -f Makefile.linear_a8 compile_context
```

当前结果：

- context binary 生成成功
- metadata 生成成功
- 当前离线编译会生成 `5` 个 graph
  原因是测试里新增了 autosplit 大输出通道验证图

另外，用户此前已经在端侧验证过当前版本可以正常执行。

---

## 6. Context Binary 体积变化

从最近几轮优化观察到：

- 早期 A8 版本 context 大小约为 `167936` bytes
- `qweight` 压成 packed int4 后，context 降到约 `159744` bytes
- `scales` 再压成 fp16 后，在当前这套小图测试下仍然是 `159744` bytes`
- 加入 `W4G128` 专用 fast path 后，context 大小仍然保持 `159744` bytes`
- 加入 `FinalizeA8` HVX fast path 后，context 大小仍然保持 `159744` bytes`
- 打通完整 TCM 链之后，context 大小仍然保持 `159744` bytes`
- 加入 `LinearA8` full-tile 专用快路径后，context 大小仍然保持 `159744` bytes`
- 加入 autosplit 大输出通道验证图后，当前 `compile_context` 输出为 `212992` bytes`

这里需要注意：

- 这个 `212992` bytes 不能直接和前面的 `159744` 做性能意义上的一一对比
- 因为现在离线编译的 graph 数量已经从 `4` 个增加到了 `5` 个
- 也就是说，这个变化主要反映的是“测试图数量增加”，而不单纯是 kernel 体积变化

这个现象的解释是：

- `qweight` 压缩带来了直接可见的图二进制收益
- `scales` 压缩更多体现为运行时带宽收益
- 当前测试图比较小，QNN 自身的对齐和元数据开销掩盖了继续缩小 binary 的效果

所以：

- context size 不再下降，不代表 `fp16 scales` 或 fast path 没价值
- 只是当前测试图太小，看不到更细粒度的 binary 收益

---

## 7. TCM 链状态说明

原 FP16 当前是这样工作的：

- `TMANLinear` 里 `l/qweight/scales` 放进 TCM
- `TMANFinalize` 里 `c` 也在 TCM 中消费

对应文件：

- [TMANLinear.cpp](/home/tang/executorch/op_gen/TMANOpPackage-dev/src/ops/TMANLinear.cpp)
- [TMANFinalize.cpp](/home/tang/executorch/op_gen/TMANOpPackage-dev/src/ops/TMANFinalize.cpp)

早期 A8 版本没有直接照搬 FP16 的完整 TCM 链，主要是因为：

1. 当时 placement 与 producer / consumer 的定义还不匹配
2. 会在 prepare 阶段引出 `flat_from_vtcm` 之类的中间转换问题

随着 transport 收紧、`LinearA8` fast path 稳定、`FinalizeA8` HVX 化完成之后，
当前版本已经能够把这条链重新打通。

所以现在更准确的结论是：

- A8 的完整 TCM 链闭环已经打通
- 当前剩下的重点不再是“能不能闭环”
- 而是“这条闭环链还能不能继续做快”

---

## 8. 当前状态小结

已经完成并稳定的内容：

- x86 正确性稳定
- 用户反馈端侧执行通过
- `qweight` 已经是真正 packed int4
- `scales` 已经压成 `FLOAT16`
- A8 三个 op 不再依赖大块临时 `vector` transport 拷贝
- `LinearA8` 已经有 tiled scalar kernel
- `LinearA8` 已经有 conservative autosplit
- `LinearA8` 已经接入 `W4G128 symmetric` 专用 fast path
- `LinearA8` 已经接入 full-tile `4x8` 专用快路径
- `LinearA8` 已经接入 full-tile group 级共享累加优化
- `LinearA8` 已经接入 full-tile `4x32` 扩展路径
- `LinearA8` 已经接入 A8 ABI 对齐的自定义 autosplit
- `FinalizeA8` 已经接入 HVX fast path
- `PrecomputeA8 -> LinearA8 -> FinalizeA8` 的完整 TCM 链闭环已经打通
- autosplit 大输出通道 case 已经在半链路和全链路上稳定通过

还没有完成的内容：

- `LinearA8` 的真正 HVX intrinsic / HVX vectorized fast path
- 更大模型下的性能 profiling 与针对性调优

---

## 9. 下一步建议

建议的下一阶段顺序：

1. 继续做 `LinearA8` 的真正 HVX fast path
2. 保留当前专用 fast path 和 tiled scalar 作为 fallback
3. 在 `LinearA8` HVX 路径稳定后，再做更细的 graph-level placement 优化
4. 最后结合更大模型做 profiling，再决定是否继续细化 `PrecomputeA8`

这样做的原因是：

- 当前最高价值的优化点仍然是 `LinearA8`
- transport 这层已经基本压到比较合理的状态
- 继续硬推 TCM 很容易重新踩到 prepare/transport 问题
- 先把算子本体变快，收益最直接，也最稳
