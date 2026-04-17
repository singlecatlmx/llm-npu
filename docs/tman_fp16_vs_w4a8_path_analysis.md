# TMAN FP16 与 W4A8 路径差异分析

## 1. 文档目的

本文档用于回答下面这个实际联调问题：

- 为什么原有 `TMAN fp16` 路径能跑通
- 为什么新的 `TMAN W4A8` 路径会在 host 侧导出、partition、lowering 阶段报错
- 两条路径具体在哪些阶段开始分叉
- 当前排查 `W4A8` 问题时应该优先看哪些文件

本文档重点不是性能，而是帮助合作者快速理解：

- `fp16` 和 `W4A8` 并不是同一条 lowering 路径
- `W4A8` 是一条更新、状态更多、额外依赖更多的正式路径
- 当前遇到的大多数问题，发生在 host 侧而不是设备侧

## 2. 一句话结论

`TMAN fp16` 能通，不代表 `TMAN W4A8` 也应该天然能通。

原因很简单：

- `fp16` 走的是老的 `tman.linear.default` 路径
- `W4A8` 走的是新的 `tman.linear_a8.default` 路径
- `W4A8` 比 `fp16` 多了 `input_scale / output_scale`、新的 custom op、新的 builder visitor、新的 runtime op package，以及 fake program 和 real program 之间的额外兼容问题

所以从工程角度看，`W4A8` 不是“在旧 TMAN 上加两个 scale”，而是“一条新的正式链路”。

## 3. 两条路径的入口差异

Qwen3 侧入口在 [examples/qualcomm/oss_scripts/qwen3/qwen3.py](/home/tang/llm-npu/examples/qualcomm/oss_scripts/qwen3/qwen3.py)。

核心分叉点在：

- `--use_tman`
- `--use_tman_a8`

对应逻辑见 [qwen3.py#L585](/home/tang/llm-npu/examples/qualcomm/oss_scripts/qwen3/qwen3.py#L585) 和 [qwen3.py#L660](/home/tang/llm-npu/examples/qualcomm/oss_scripts/qwen3/qwen3.py#L660)。

当开启 `--use_tman_a8` 时，代码会额外做两件事：

1. 加载外部 `activation_scales_a8_per-tensor_sym.pt`
2. 把 GPTQ 线性层替换成 `TMANLinearA8`，然后把 `input_scale / output_scale` 写进模块 buffer

这一步在 `fp16` 路径里是不存在的。

## 4. 模块替换后的结构差异

共用基础模块在 [backends/qualcomm/utils/utils.py](/home/tang/llm-npu/backends/qualcomm/utils/utils.py)。

旧 `fp16` 路径替换成：

- `TMANLinear`

新 `W4A8` 路径替换成：

- `TMANLinearA8`

相关定义见：

- [utils.py#L228](/home/tang/llm-npu/backends/qualcomm/utils/utils.py#L228)
- [utils.py#L284](/home/tang/llm-npu/backends/qualcomm/utils/utils.py#L284)

二者的关键差异：

`TMANLinear`

- 只带 GPTQ 权重相关参数
- forward 走 `tman_linear(...)`
- 对外仍然是普通 `fp16 -> fp16`

`TMANLinearA8`

- 继承自 `TMANLinear`
- 额外挂两个 buffer：
  `input_scale`
  `output_scale`
- forward 走 `tman_linear_a8(...)`

这两个 scale buffer 是 `W4A8` 正式路径最核心的新增状态。

## 5. Eager / Custom Op 层面的分叉

旧 `fp16` custom op 在：

- [backends/qualcomm/builders/custom_ops.py](/home/tang/llm-npu/backends/qualcomm/builders/custom_ops.py) 中的 `tman::linear`

新 `W4A8` custom op 在同一个文件中的：

- `tman::linear_a8`

两者语义不同：

`tman::linear`

- 直接做 GPTQ 权重反量化
- 执行 matmul
- 返回浮点输出

`tman::linear_a8`

- 先用外部 `input_scale` 做 A8 量化再反量化
- 再走 GPTQ 权重反量化和 matmul
- 最后再按 `output_scale` 做一次量化再反量化

也就是说，`W4A8` 在 eager 侧就比 `fp16` 多了一整套激活量化误差模型。

这也是为什么此前会出现 `output_scale` 被删掉又使用的问题：

- `fp16` 根本没有这条逻辑
- `W4A8` 有

## 6. Builder / Visitor 层面的分叉

这一步是当前最容易混淆、也是最容易出 bug 的地方。

旧 `fp16` builder visitor：

- [backends/qualcomm/builders/op_tman_linear.py](/home/tang/llm-npu/backends/qualcomm/builders/op_tman_linear.py)

新 `W4A8` builder visitor：

- [backends/qualcomm/builders/op_tman_linear_a8.py](/home/tang/llm-npu/backends/qualcomm/builders/op_tman_linear_a8.py)

所以严格来说：

- `W4A8` 不走 `op_tman_linear.py`
- `W4A8` 走的是 `op_tman_linear_a8.py`

这是一个非常重要的边界。

如果只是定位 A8 专属 bug，优先看：

- `op_tman_linear_a8.py`
- `custom_ops.py`
- `qwen3.py`
- `runtime/op_packages/TMANOpPackageW4A8`

## 7. 为什么 partition 阶段会先炸

QNN partitioner 在做 support check 时，不只是看 op 名字。

它会直接调用 visitor 的 `define_node(...)` 来构造 `QnnOpWrapper`，相关逻辑在：

- [backends/qualcomm/partition/qnn_partitioner.py#L83](/home/tang/llm-npu/backends/qualcomm/partition/qnn_partitioner.py#L83)

更关键的是，partitioner 用的不是 real program，而是 fake program：

- [exir/backend/backend_api.py#L388](/home/tang/llm-npu/exir/backend/backend_api.py#L388)

这意味着：

- graph 结构是一样的
- 但 state_dict 里的常量会被换成 `FakeTensor`

所以在 partition support 阶段，如果 builder 还试图立即读取真实 `qweight/scales/qzeros` 数据，就会踩到 fake tensor 兼容问题。

这也是 `W4A8` 比 `fp16` 更容易先在 host 侧报错的一个直接原因：

- `W4A8` 新 visitor 的逻辑更复杂
- 它更依赖新增常量和新增参数
- 它更容易在 fake program 阶段就尝试做“真实数据预处理”

## 8. 为什么 fp16 能通而 A8 会挂

### 8.1 fp16 路径更老、更短

旧 `fp16` 路径已经存在更久，逻辑更短：

- 不需要外部激活 scale 文件
- 不需要 `input_scale / output_scale` buffer
- 不需要 output finalize A8 误差模型
- 不需要 `TMANOpPackageW4A8`

### 8.2 A8 路径额外状态更多

`W4A8` 新增了这些状态：

- `input_scale`
- `output_scale`
- `activation_scales_a8_per-tensor_sym.pt`
- `tman.linear_a8.default`
- `TMANFinalizeA8`
- `TMANOpPackageW4A8`

状态越多，export / partition / lowering 阶段的脆弱点就越多。

### 8.3 A8 路径会更早暴露 fake / real 兼容问题

当前 host 侧 lowering 的典型链路是：

1. compile
2. partition support check
3. fake program visitor define_node
4. real program preprocess
5. runtime compile

如果 visitor 在第 3 步就强依赖真实常量，A8 会先挂在这里。

而 `fp16` 路径如果历史上没踩过这类组合情况，就可能一直表现为“能通”。

## 9. W4A8 路径的完整调用链

可以粗略理解为下面这条链路：

```text
qwen3.py
  -> convert_qlinear_to_tman_linear_a8(...)
  -> apply_tman_a8_activation_scales(...)
  -> custom op: tman.linear_a8.default
  -> partitioner support check
  -> builder: op_tman_linear_a8.py
  -> QNN preprocess
  -> runtime package: TMANOpPackageW4A8
```

而旧 `fp16` 路径更像：

```text
qwen3.py
  -> convert_qlinear_to_tman_linear(...)
  -> custom op: tman.linear.default
  -> partitioner support check
  -> builder: op_tman_linear.py
  -> QNN preprocess
  -> runtime package: TMANOpPackage
```

这两条链路共享的是 Qualcomm backend 的公共层，但 A8 明显多了：

- scale 文件接入
- scale buffer 注入
- A8 custom op
- A8 runtime package

## 10. 当前已经暴露过的 A8 专属问题

### 10.1 eager custom op 语义错误

问题：

- `output_scale` 被删除后又继续使用

现象：

- 校准阶段直接抛 `UnboundLocalError`

影响范围：

- `W4A8` 专属

### 10.2 OpPackage 环境变量指错

问题：

- `--use_tman_a8` 开启时，`QNN_OP_PACKAGE_PATHS` 仍然指向旧 `TMANOpPackage`

现象：

- host 侧会先告警
- 即使前面编过了，后面 runtime 也可能不匹配

影响范围：

- `W4A8` 专属

### 10.3 partition fake program 阶段读取真权重

问题：

- A8 visitor 在 `FakeTensor` 状态下过早尝试 `numpy()` 或真实权重解包

现象：

- `.numpy() is not supported for tensor subclasses`
- `FakeTensor` / `as_subclass(torch.Tensor)` 相关错误

影响范围：

- 主要发生在 `W4A8`

### 10.4 quant config 提示

日志里经常会看到：

```text
No quant config is implemented for op, tman.linear_a8.default
```

它说明：

- `tman.linear_a8.default` 没走普通 quant config 注册表

但它通常不是当前流程退出的直接根因。

## 11. 当前看 A8 问题时，哪些文件是关键

### 11.1 A8 专属关键文件

- [examples/qualcomm/oss_scripts/qwen3/qwen3.py](/home/tang/llm-npu/examples/qualcomm/oss_scripts/qwen3/qwen3.py)
- [backends/qualcomm/utils/utils.py](/home/tang/llm-npu/backends/qualcomm/utils/utils.py)
- [backends/qualcomm/builders/custom_ops.py](/home/tang/llm-npu/backends/qualcomm/builders/custom_ops.py)
- [backends/qualcomm/builders/op_tman_linear_a8.py](/home/tang/llm-npu/backends/qualcomm/builders/op_tman_linear_a8.py)
- [backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8](/home/tang/llm-npu/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8)

### 11.2 Qualcomm backend 公共层

- [backends/qualcomm/builders/node_visitor.py](/home/tang/llm-npu/backends/qualcomm/builders/node_visitor.py)
- [backends/qualcomm/builders/utils.py](/home/tang/llm-npu/backends/qualcomm/builders/utils.py)
- [backends/qualcomm/partition/qnn_partitioner.py](/home/tang/llm-npu/backends/qualcomm/partition/qnn_partitioner.py)
- [backends/qualcomm/qnn_preprocess.py](/home/tang/llm-npu/backends/qualcomm/qnn_preprocess.py)

### 11.3 不是 A8 主路径，但容易让人误会的文件

- [backends/qualcomm/builders/op_tman_linear.py](/home/tang/llm-npu/backends/qualcomm/builders/op_tman_linear.py)

这个文件处理的是：

- `tman.linear.default`
- `tman.bitnet_linear.default`

它不是 `W4A8` 的主 visitor。

如果 `fp16` 原本能通，那么排查 A8 时不应该先怀疑这个文件。

## 12. 推荐排查顺序

如果后面继续联调 `W4A8`，建议按这个顺序看：

1. `qwen3.py` 是否真的走到了 `--use_tman_a8`
2. `activation scale` 文件是否存在且 key 能匹配模块名
3. `QNN_OP_PACKAGE_PATHS` 是否指向 `TMANOpPackageW4A8`
4. `custom_ops.py` 的 eager 语义是否能通过 calibration
5. `op_tman_linear_a8.py` 是否在 fake partition 阶段错误读取真实权重
6. real preprocess 阶段是否能生成正确 QNN op wrapper
7. runtime package 是否能完成 context compile

## 13. 当前结论

到目前为止，`W4A8` 暴露出来的问题大多是：

- host 侧导出问题
- fake / real program 兼容问题
- 新 builder visitor 的边界问题
- A8 专属 scale 和 op package 接线问题

它们并不能直接说明：

- `Hexagon v75` 上的 `W4A8` runtime 内核本身一定有问题

更准确的说法应该是：

- `fp16` 路径已经是较成熟的旧路径
- `W4A8` 路径是新的正式链路
- 当前主要卡在 host 侧编译链路，还没有稳定进入“设备侧数值和性能问题”阶段

这也是为什么现阶段看到 `fp16` 通、`W4A8` 挂，是合理且符合工程现状的。

## 14. Runtime 契约分析

下面这部分记录当前对 runtime 包里 `PrecomputeA8 -> LinearA8 -> FinalizeA8`
这条链路的更细粒度判断。

### 14.1 当前更可疑的不是单点低级错误，而是契约不完全一致

在正式包里：

- [TMANPrecomputeA8.cpp](/home/tang/llm-npu/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/src/ops/TMANPrecomputeA8.cpp)
- [TMANLinearA8.cpp](/home/tang/llm-npu/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/src/ops/TMANLinearA8.cpp)
- [hvx_funcs.h](/home/tang/llm-npu/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/include/hvx_funcs.h)

三者之间目前最大的风险点，不是“某一行明显写反了”，而是：

- `PrecomputeA8` 写出的 `[lut][ls][lb]`
- `LinearA8` 读取 `[lut][ls][lb]` 的方式
- 以及 `hvx_tbl(...)` 内部对 `ls/lb` 的使用时机

这三者未必完全一致。

### 14.2 PrecomputeA8 当前语义

[TMANPrecomputeA8.cpp](/home/tang/llm-npu/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/src/ops/TMANPrecomputeA8.cpp)
当前采用的是“保守标量参考实现”：

1. 使用外部 `input_scale`
2. 对输入做 `q_a = round_clip(x / s_a)`
3. 每四个激活生成一个 16-entry signed LUT
4. `ls[i] = s_a`
5. `lb[group] = -0.5 * sum(q_a) * s_a`

这条实现本身是自洽的，而且适合作为正确性优先路径。

### 14.3 LinearA8 当前读取方式

[TMANLinearA8.cpp](/home/tang/llm-npu/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/src/ops/TMANLinearA8.cpp)
里最终调用的是：

```cpp
hvx_tbl<LType, XType, CType, ACT_GROUP_SIZE, 128, false, 4, TILE_K, LUT_G, true>(
    gemm_m, gemm_k, gemm_n, l_ptr, ls_ptr, lb_ptr, w_ptr, s_ptr, c_ptr);
```

也就是说，真正决定数值行为的是
[hvx_funcs.h](/home/tang/llm-npu/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/include/hvx_funcs.h)
里的 `hvx_tbl(...)`。

### 14.4 当前最值得怀疑的 3 个契约点

#### 14.4.1 `lb` 的使用时机

在 `PrecomputeA8` 里，`lb` 是按 group 写入：

```cpp
lb_ptr[group_idx++] = -0.5f * static_cast<float>(group_sum) * s_a;
```

而在 `hvx_tbl(...)` 里，`lb` 不是每一步都加，而是在

- `cmp_blk_tail`
- `q_group_tail`
- `vec_p % (VecP * Bits) == VecP`

这些条件成立时才被加进去，见：
[hvx_funcs.h#L1413](/home/tang/llm-npu/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/include/hvx_funcs.h#L1413)
到
[hvx_funcs.h#L1471](/home/tang/llm-npu/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/include/hvx_funcs.h#L1471)

这意味着：

- `lb` 的数学定义即使正确
- 如果它的“写入粒度”和 `hvx_tbl` 的“读取时机”不完全匹配
- 也会出现系统性的符号或偏移错误

#### 14.4.2 `ls` 的粒度

`PrecomputeA8` 当前使用：

- `ls_size = gemm_k / 256`
- 每个 `ls` 槽位都写 `s_a`

而 `hvx_tbl(...)` 读取时用的是：

```cpp
ls[(tile_q + vec_q) / q_act_group_size]
```

见 [hvx_funcs.h#L1423](/home/tang/llm-npu/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/include/hvx_funcs.h#L1423)

因此要求：

- `ActGroupSize`
- `q_act_group_size`
- `tile_q`
- `vec_q`

和 `PrecomputeA8` 的 `ls` 布局必须完全一致。

如果这里只是“看起来相同”但实际切块边界不一致，就会导致整组输出偏移。

#### 14.4.3 LUT 顺序

`PrecomputeA8` 构 LUT 时用的是：

```cpp
acc += (idx & 0x1) ? x0 : -x0;
acc += (idx & 0x2) ? x1 : -x1;
acc += (idx & 0x4) ? x2 : -x2;
acc += (idx & 0x8) ? x3 : -x3;
```

而 `hvx_tbl(...)` 通过 `Q6_Wh_vlut16_VbVhR_nomatch(...)`
从 LUT 中取值。

如果：

- `PrecomputeA8` 生成 LUT 的符号/索引顺序
- 和 `hvx_tbl(...)` 假定的 LUT 顺序

并不是同一套约定，那么即便 `ls/lb` 没问题，也会产生明显的数值偏差。

### 14.5 当前最合理的怀疑顺序

在 runtime 包里，如果继续沿实现排查，建议优先怀疑：

1. `lb` 的定义与 `hvx_tbl(...)` 使用时机是否完全一致
2. `ls` 的 group 粒度与 `q_act_group_size` 是否完全一致
3. LUT 的生成顺序与 `Q6_Wh_vlut16_VbVhR_nomatch(...)` 的取值顺序是否一致

相对来说，当前 `TMANPrecomputeA8.cpp` 本身没有明显的低级错误；
更像是它与 `TMANLinearA8` / `hvx_tbl(...)` 的契约还需要进一步核对。

### 14.6 一个新的重要判断：当前标量 PrecomputeA8 可能已经打破了 hvx_tbl 契约

在正式包里，`TMANPrecomputeA8.cpp` 当前已经被切到一条“保守标量参考实现”：

- 不再走 `hvx_lut_ctor(...)`
- 而是直接写：
  - signed LUT
  - `ls = s_a`
  - `lb = -0.5 * sum(q_a) * s_a`

这条实现从局部数学上看是自洽的，但它未必仍然满足 `hvx_tbl(...)` 的原始契约。

原因在于：

- `hvx_tbl(...)` 不是一个“任意 `[lut][ls][lb]` 都能正确消费”的通用内核
- 它实际上默认了：
  1. LUT 的排列方式来自 `hvx_lut_ctor(...)`
  2. `ls` 的粒度与 `q_act_group_size` 完全一致
  3. `lb` 的数值定义与其内部 `q_group_tail / cmp_blk_tail` 的加法时机相匹配

因此，一旦 `PrecomputeA8` 改成了另一种 LUT 生成策略，就会出现一个很关键的现象：

- `x86` 和 `v75` 设备可能彼此对齐
- 但整个 pipeline 的数学结果仍然不对

这说明：

- “设备与 x86 一致”
- 不等于
- “实现与目标数学语义一致”

换句话说：

- 之前把 `PrecomputeA8` 改成标量参考实现，更适合做“设备分支一致性验证”
- 但它不一定还能作为“正式 runtime 全链路正确实现”

这也是为什么：

- 端侧 stage test 可能显示某些阶段都 pass
- 但完整 runtime pipeline 在 x86 上仍然会出现明显的数值偏差

当前更合理的推断是：

- 如果要保持 `LinearA8` 不变，`PrecomputeA8` 最终仍然需要产出与 `hvx_lut_ctor(...)` 兼容的 packed 语义
- 如果要保留当前这条更直观的标量 LUT 语义，那么 `LinearA8` / `hvx_tbl(...)` 也必须同步调整

因此，当前 runtime 包的关键选择实际上是二选一：

1. 回到 `hvx_lut_ctor(...)` 契约，确保 `PrecomputeA8` 与 `hvx_tbl(...)` 匹配
2. 保留当前标量 LUT 语义，并同步重写 `LinearA8` 的消费逻辑

从改动成本和风险看，第一条更接近现有正式包的设计思路。
