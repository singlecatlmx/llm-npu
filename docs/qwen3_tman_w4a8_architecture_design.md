# Qwen3 TMAN W4A8 架构设计

## 1. 文档目的

本文档在 [description.md](/home/lmx/Projects/tman/test/executorch/docs/description.md) 和 [qwen3_tman_w4a8_requirements_analysis.md](/home/lmx/Projects/tman/test/executorch/docs/qwen3_tman_w4a8_requirements_analysis.md) 的基础上，进一步给出可直接指导编码实现的架构设计方案。

本文档重点回答四个问题：

- 整体架构怎么分层；
- 端到端数据和控制流怎么走；
- 具体要改哪些模块；
- 每个模块承担什么功能、采用什么实现逻辑。

## 2. 设计目标

本次架构设计围绕以下目标展开：

- 在现有 `W4A16` TMAN 路径旁边，新建一条专用 `W4A8` 路径；
- 保持 Qwen3 decoder 内部 GPTQ `W4` 权重不变；
- linear 激活采用外部 `.pt` 文件提供的对称 per-tensor int8 量化参数；
- 非 linear 算子继续保持现有 FP16 路径；
- `W4A8` 改造覆盖 Python 模块层、export/custom op 层、QNN lowering 层、OpPackage 层和 HVX kernel 层；
- 后续编码时能按模块并行推进，且不破坏旧的 `W4A16` 能力。

## 3. 设计原则

## 3.1 双轨并存，禁止直接覆盖旧路径

本次设计明确采用：

- `W4A16` 保持不动
- `W4A8` 新增并行支线

不推荐直接把现有 `tman::linear`、`TMANLinear`、`TMANPrecompute/TMANLinear/TMANFinalize` 原地改成同时兼容所有模式的超大分支。这样风险过高，回归面过大，也不利于后续排障。

## 3.2 外部 scale 是唯一事实来源

`activation_scales_a8_per-tensor_sym.pt` 是 linear 激活量化参数的唯一事实来源。

因此：

- `W4A8` linear 不再依赖 observer 校准得到输入输出 scale；
- `W4A8` custom path 不依赖全图 PTQ 自动注解来生成它自身的激活量化参数；
- 所有与 linear A8 相关的 scale、zero-point 规则必须由外部 scale 文件和固定语义共同确定。

## 3.3 其它算子保持 FP16，边界由 TMAN 自己负责

本次设计明确采用：

- 其它算子继续走现有 FP16 图；
- linear A8 的量化与反量化边界在 TMAN custom path 内部解决。

也就是说，周围计算图看到的仍然是 FP16 数据流，而不是把整个 Qwen3 图扩展成全图 8bit 量化图。

## 3.4 第一阶段只服务 Qwen3-1.7B GPTQ W4G128 Sym

本阶段设计只面向已经确认的目标：

- Qwen3-1.7B
- GPTQ int4
- `group_size = 128`
- `sym = true`
- decoder 内部 7 类 linear

暂不为了“潜在泛化”把当前设计复杂化。

## 4. 总体架构方案

## 4.1 架构结论

本次建议采用“5 层 + 1 条端到端支线”的整体架构：

1. 编译配置与模式选择层
2. 外部激活 scale 管理层
3. Python 模块替换与导出表达层
4. QNN lowering 与图边界转换层
5. Runtime OpPackage 与 HVX kernel 层

对应的端到端执行链路为：

```text
Qwen3 GPTQ 模型 + 外部 activation_scales.pt
    ->
编译入口解析 W4A8 配置
    ->
构建 ActivationScaleRegistry
    ->
把目标 linear 替换成 TMANLinearA8
    ->
导出成 tman::linear_a8
    ->
lower 成:
FP16输入
 -> Convert(量化到INT8)
 -> TMANPrecomputeA8
 -> TMANLinearA8
 -> TMANFinalizeA8
 -> Convert(反量化回FP16)
 -> FP16输出
    ->
QNN + TMANOpPackage + HVX kernel 执行
```

## 4.2 为什么选“新增 A8 支线”而不是“改旧 op”

我们评估过两种方案：

方案 A：直接扩展现有 `tman::linear` 和现有 `TMANPrecompute/TMANLinear/TMANFinalize`

方案 B：新增 `TMANLinearA8`、`tman::linear_a8`、`TMANPrecomputeA8/TMANLinearA8/TMANFinalizeA8`

本设计选方案 B，原因是：

- 旧 `W4A16` 路径已可工作，保留原样最稳；
- 新路径与旧路径的数学语义差异很大，不适合揉成一个大分支；
- 运行时 OpPackage 注册、host 编译和设备侧验证都更容易定位问题；
- 后续如果 `W4A8` 出现数值或性能问题，可以快速回退到 `W4A16`；
- 文档、日志、测试、benchmark 都能清楚区分 A16 与 A8。

## 5. 关键设计决策

## 5.1 新增独立模式入口，不复用现有 `--ptq`

架构上明确不把 `W4A8` 定义成现有 `--ptq 8a8w` 的别名。

推荐新增参数：

- `--tman_act_mode {a16,a8}`
- `--activation_scales_pt <path>`

推荐行为：

- `--use_tman --tman_act_mode a16`
  - 走现有 `W4A16`
- `--use_tman --tman_act_mode a8 --activation_scales_pt ...`
  - 走新 `W4A8`

推荐约束：

- 当 `--tman_act_mode a8` 时，禁止同时打开现有全图 `--ptq`
- 这样可以保证其它算子继续保持 FP16，不引入第二套量化源

## 5.2 新增 `TMANLinearA8` 与 `tman::linear_a8`

在 Python / export 层采用独立对象表达：

- 新增模块类：`TMANLinearA8`
- 新增导出 op：`tman::linear_a8`

这样后续判断逻辑会非常清晰：

- `TMANLinear` / `tman::linear` 代表旧 `W4A16`
- `TMANLinearA8` / `tman::linear_a8` 代表新 `W4A8`

## 5.3 QNN 侧新增 A8 三段式 op

保持现有 TMAN 的 3 段式结构不变，但新增 A8 变体：

- `TMANPrecomputeA8`
- `TMANLinearA8`
- `TMANFinalizeA8`

这样做的好处是：

- 复用现有 TMAN 三段式组织方式；
- 旧 A16 和新 A8 不互相污染；
- host / device / log / profiling 都更容易区分。

## 5.4 量化边界显式化

为了满足“linear A8，周围仍 FP16”的要求，`W4A8` 支线采用显式边界转换：

```text
FP16 x
 -> 输入 Convert 量化成 INT8
 -> TMAN A8 kernel 路径
 -> 输出 Convert 反量化回 FP16
 -> 下游非 linear
```

这意味着：

- 输入量化边界由 builder 显式插入；
- 输出反量化边界也由 builder 显式插入；
- 周围图仍保持原有 FP16 语义；
- 线性层内部真正变成 `W4A8`。

## 5.5 激活 scale 在 host 侧折叠进 effective scales

当前外部文件同时提供：

- `input_scale`
- `output_scale`

由于它们都是 per-tensor 标量，推荐在 host 侧把它们折叠进 weight group scales，得到：

```text
effective_scale = weight_scale * input_scale / output_scale
```

这样做的好处是：

- kernel 内部不需要同时管理三种 scale；
- `TMANLinearA8` 只需要消费重排后的 `effective_scales`；
- 输出边界的 `Convert` 仍然可以继续使用 `output_scale` 做 INT8 -> FP16 反量化；
- 输入边界的 `Convert` 使用 `input_scale` 做 FP16 -> INT8 量化。

因此，外部 scale 在架构上会有两种消费方式：

- 图边界的 quant/dequant encoding
- host 侧折叠得到的 `effective_scales`

## 6. 端到端执行流程设计

## 6.1 编译期流程

编译期流程建议如下：

1. `qwen3.py` 解析 `--use_tman`、`--tman_act_mode`、`--activation_scales_pt`
2. 若模式为 `a8`：
   - 加载外部 scale 文件
   - 构造 `ActivationScaleRegistry`
   - 校验所有目标 linear 都有 `input_scale` / `output_scale`
3. 将目标 GPTQ qlinear 替换为 `TMANLinearA8`
4. export 时生成 `tman::linear_a8`
5. lowering 时：
   - 插入输入 quantize Convert
   - 生成 `TMANPrecomputeA8 -> TMANLinearA8 -> TMANFinalizeA8`
   - 插入输出 dequantize Convert
6. QNN host 编译并输出 `.pte`

## 6.2 运行期流程

运行期每个 `W4A8` linear 建议按如下语义执行：

1. 输入 FP16 激活 `x_fp`
2. 用 `input_scale` 量化：

```text
x_q = clamp(round(x_fp / input_scale), -127, 127)
```

3. `TMANPrecomputeA8` 基于 `x_q` 构造整数 LUT
4. `TMANLinearA8` 读取 W4 权重并查 LUT、累加
5. `TMANFinalizeA8` 依据 folded `effective_scales` 做 requant，得到 `y_q`
6. 用 `output_scale` 反量化回 FP16：

```text
y_fp = y_q * output_scale
```

7. 交给下游非 linear 算子继续按 FP16 跑

## 6.3 A8 路径的数学目标

设计目标不是“纯粹在图上插一个 int8 fake 过程”，而是尽量实现与以下 reference 语义一致：

```text
x_q = Q(x_fp, input_scale)
w_fp = DequantGPTQ(w4, weight_scale)
y_fp_ref = (x_q * input_scale) @ w_fp
y_q = Q(y_fp_ref, output_scale)
y_fp_out = y_q * output_scale
```

周围计算图真正看到的是：

- `y_fp_out`

但 linear 内部真正执行的是：

- `W4A8`

## 7. 模块分层设计

## 7.1 配置与编译入口层

### 目标职责

- 解析 `W4A8` 模式；
- 控制与旧 `W4A16` / 全图 `PTQ` 的互斥关系；
- 在编译期把 scale 文件加载和模块替换串起来。

### 主要改造文件

- `examples/qualcomm/oss_scripts/qwen3/qwen3.py`

### 推荐新增逻辑

- 新增参数：
  - `--tman_act_mode`
  - `--activation_scales_pt`
- 新增约束检查：
  - `tman_act_mode=a8` 时必须 `--use_tman`
  - `tman_act_mode=a8` 时必须提供 `activation_scales_pt`
  - `tman_act_mode=a8` 时禁止和全图 `--ptq` 同时使用
- 新增编译期分支：
  - `convert_qlinear_to_tman_linear_a8(...)`

### 实现逻辑

`qwen3.py` 不负责 scale 数学处理，只负责：

- 组织模式；
- 构造 registry；
- 调用替换函数；
- 选择后续 lowering 分支。

## 7.2 外部激活 scale 管理层

### 目标职责

- 加载外部 `.pt` 文件；
- 标准化 key；
- 建立模块路径到 scale 的查询接口；
- 在编译开始前做完整覆盖校验。

### 推荐新增文件

- `backends/qualcomm/utils/tman_w4a8.py`

### 推荐核心对象

- `ActivationScaleRegistry`

### 推荐核心接口

- `load_activation_scales_pt(path)`
- `normalize_qwen3_linear_key(key)`
- `get_linear_scales(module_path)`
- `validate_expected_qwen3_linears(model)`

### 建议数据结构

```text
{
  "model.layers.0.self_attn.q_proj": {
    "input_scale": torch.Tensor([..]),
    "output_scale": torch.Tensor([..]),
  },
  ...
}
```

### 实现逻辑

- 加载后先把 `model.model.layers.X...` 规范化成当前静态模型所使用的路径；
- 只保留本阶段目标 linear；
- 提供按模块路径的 O(1) 查询；
- 在进入模块替换前一次性检查 28 层 x 7 linear 是否覆盖完整。

## 7.3 Python 模块替换层

### 目标职责

- 让导出图中明确出现 `W4A8` linear；
- 承载权重、scale、模式等静态信息；
- 与现有 `TMANLinear` 解耦。

### 主要改造文件

- `backends/qualcomm/utils/utils.py`

### 推荐新增对象

- `TMANLinearA8`
- `convert_qlinear_to_tman_linear_a8(...)`

### `TMANLinearA8` 推荐字段

- GPTQ 权重相关：
  - `qweight`
  - `scales`
  - `qzeros`
  - `g_idx`
  - `wf_unsqueeze_zero`
  - `wf_unsqueeze_neg_one`
- 元信息：
  - `bits`
  - `group_size`
  - `symmetric`
  - `gptq_v2`
- 激活量化相关：
  - `input_scale`
  - `output_scale`
  - `act_quant_min = -127`
  - `act_quant_max = 127`

### `forward(...)` 推荐语义

```text
return tman_linear_a8(
    x,
    qweight,
    scales,
    qzeros,
    g_idx,
    wf_unsqueeze_zero,
    wf_unsqueeze_neg_one,
    input_scale,
    output_scale,
    group_size,
    bits,
    symmetric,
    gptq_v2,
)
```

### 实现逻辑

- 不复用旧 `TMANLinear`，避免字段和语义混用；
- 替换函数通过模块全路径从 `ActivationScaleRegistry` 查询 scale；
- 若目标 linear 没有 scale，直接报错，不做隐式 fallback。

## 7.4 Export Custom Op 层

### 目标职责

- 在 FX / export 图里显式表达 `W4A8` linear 语义；
- 提供 fake / reference 实现，便于图导出与单元测试。

### 主要改造文件

- `backends/qualcomm/builders/custom_ops.py`

### 推荐新增对象

- `@torch.library.custom_op("tman::linear_a8")`
- `tman_linear_a8_fake(...)`

### 推荐 schema

输入建议包含：

- `x`
- `qweight`
- `scales`
- `qzeros`
- `g_idx`
- `wf_unsqueeze_zero`
- `wf_unsqueeze_neg_one`
- `input_scale`
- `output_scale`
- `group_size`
- `bits`
- `symmetric`
- `gptq_v2`

### fake / reference 实现逻辑

reference 实现按以下顺序执行：

1. `x_fp -> x_q`
2. GPTQ 权重反量化成 `w_fp`
3. `y_fp_ref = (x_q * input_scale) @ w_fp`
4. `y_q = round(y_fp_ref / output_scale)`
5. `y_fp_out = y_q * output_scale`

输出仍返回 FP16 / 输入同 dtype 的 tensor，便于周围图保持兼容。

## 7.5 QNN Lowering 与边界转换层

### 目标职责

- 把 `tman::linear_a8` lower 成 QNN 图中的显式量化边界 + TMAN A8 三段式 op；
- 在 host 侧完成 `effective_scales` 折叠和权重重排；
- 保证周围图仍是 FP16。

### 主要改造文件

- 新增：`backends/qualcomm/builders/op_tman_linear_a8.py`
- 修改：`backends/qualcomm/builders/qnn_constants.py`
- 修改：`backends/qualcomm/partition/common_defs.py`
- 修改：`backends/qualcomm/_passes/layout_transform.py`
- 修改：`backends/qualcomm/quantizer/annotators.py`
- 修改：`backends/qualcomm/quantizer/custom_annotation.py`

### 为什么推荐新建 `op_tman_linear_a8.py`

原因很简单：

- 旧 `op_tman_linear.py` 的逻辑和 FP16 kernel 假设绑定很深；
- A8 需要显式中间 int8 tensor 和新的 static tensor；
- 拆新文件更便于快速推进和回归隔离。

### Builder 侧推荐 lowering 结构

建议 lowering 成如下序列：

```text
input_fp16
 -> input_convert_a8
 -> TMANPrecomputeA8
 -> TMANLinearA8
 -> TMANFinalizeA8
 -> output_convert_fp16
 -> output_fp16
```

### Builder 侧关键实现逻辑

#### 1. 输入量化边界

- 创建一个中间 int8 tensor wrapper `x_q`
- 其 quant encoding 使用：
  - `scale = input_scale`
  - `zero_point = 0`

然后用 QNN `Convert` 从 FP16 输入转换到 `x_q`

#### 2. Weight static tensor

继续复用现有：

- `unpack_gptqv2(...)`
- `hvx_preprocess_weights(...)`

但在 A8 路径上额外做一步：

```text
effective_scales = weight_scales * input_scale / output_scale
```

再把 `effective_scales` 重排成 kernel 需要的 static tensor。

#### 3. Scratch tensor 设计

推荐 A8 路径下重新定义 scratch：

- `l_a8`
  - 存整数 LUT
  - 不再复用旧 A16 路径中 `ls/lb` 尾部布局
- `c_a8`
  - 存 int32 或等价的中间累加结果

这样可以把旧 A16 的动态激活组 scale / bias 补偿逻辑和新 A8 路径剥离开。

#### 4. 输出量化边界

- `TMANFinalizeA8` 产出中间 int8 tensor `y_q`
- `y_q` 的 quant encoding 使用：
  - `scale = output_scale`
  - `zero_point = 0`

随后再插入一个 `Convert` 将 `y_q` 反量化回 FP16，交给下游。

### 相关 allowlist / annotator 改造

由于新增了 `tman::linear_a8`，下面这些位置都要能识别它：

- `partition/common_defs.py`
- `_passes/layout_transform.py`
- `quantizer/annotators.py`
- `quantizer/custom_annotation.py`

其中 `custom_annotation.py` 里当前对 `tman::linear` 的“8a 不支持”限制，应只保留给旧路径，不能误伤新 `tman::linear_a8`。

## 7.6 Runtime OpPackage 与 HVX Kernel 层

### 目标职责

- 实现 A8 三段式 op 的 host / device 注册与执行；
- 保留旧 A16 op 不变；
- 尽量在 kernel 层走整数激活 + 低比特查表 + 输出 requant。

### 主要改造文件

- 新增：
  - `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANPrecomputeA8.cpp`
  - `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANLinearA8.cpp`
  - `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANFinalizeA8.cpp`
- 修改：
  - `backends/qualcomm/runtime/op_packages/TMANOpPackage/include/hvx_funcs.h`
  - `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/TMANOpPackageInterface.cpp`
  - `backends/qualcomm/runtime/op_packages/TMANOpPackage/config/TMANOpPackageHtp.xml`
  - `backends/qualcomm/runtime/op_packages/TMANOpPackage/Makefile`

### A8 Kernel 设计建议

#### 1. `TMANPrecomputeA8`

输入：

- `x_q`，对称 int8

职责：

- 每 `4` 个激活值构造一个 `16` 路 LUT
- LUT 项保存整数和，供后续 W4 查表

关键性质：

- 因为 `x_q` 是 int8，对 4 个数求任意符号组合和，其范围大致在 `[-508, 508]`
- 使用 `int16` 存 LUT 项是安全的

因此 `TMANPrecomputeA8` 可以完全摆脱旧 A16 路径里的：

- 动态 `ls`
- 激活侧 `lb`

#### 2. `TMANLinearA8`

输入：

- `l_a8`
- 重排后的 `qweight`
- 重排后的 `effective_scales`

职责：

- 沿用当前 W4 packed weight 的查表访问模式；
- 查整数 LUT；
- 做 group-wise scale 应用；
- 产生中间累加结果。

建议实现策略：

- 第一阶段优先保证正确性，可接受 `int32 + fp16/fp32 effective_scale` 路径；
- 第二阶段再考虑把 requant multiplier 固定点化。

#### 3. `TMANFinalizeA8`

输入：

- `c_a8`

职责：

- 做最终 round / clamp / cast；
- 产出 int8 输出 `y_q`

输出范围建议固定为：

- `[-127, 127]`

随后交由图外部 `Convert` 完成 `INT8 -> FP16`。

### OpPackage 注册改造

`TMANOpPackageInterface.cpp` 需要新增：

- 新 op 声明
- op name 列表
- validate 分支
- interface provider 输出

目标是让 A16 和 A8 两套 op 能同时注册。

## 7.7 测试与验证层

### 目标职责

- 验证 scale 加载、模块替换、export、lowering、kernel 数学正确性；
- 提供后续 benchmark 基础。

### 推荐改造文件

- 修改：`backends/qualcomm/tests/test_tman_linear.py`
- 推荐新增：
  - `backends/qualcomm/tests/test_tman_linear_a8.py`
  - `backends/qualcomm/tests/test_qwen3_tman_w4a8_export.py`

### 推荐测试分层

#### 1. Scale 管理单测

- key 规范化是否正确
- 缺失 scale 是否能正确报错
- 28 层 x 7 linear 覆盖校验是否正确

#### 2. Python reference 单测

- `tman::linear_a8` 的 reference 结果是否与手工公式一致

#### 3. Lowering 单测

- 导出图是否出现 `tman.linear_a8.default`
- QNN lowering 是否生成 `TMANPrecomputeA8/TMANLinearA8/TMANFinalizeA8`

#### 4. Kernel 单测

- 小尺寸 synthetic case 的 host / device 输出一致性

#### 5. 端到端测试

- Qwen3 compile-only 成功
- 设备侧推理成功

#### 6. 性能测试

- 对比 `W4A16` 与 `W4A8`
- 分 prefill / decode 两段统计

## 8. 文件级改造清单

| 文件 | 改造类型 | 功能 | 实现逻辑 |
| --- | --- | --- | --- |
| `examples/qualcomm/oss_scripts/qwen3/qwen3.py` | 修改 | 新增 W4A8 模式入口 | 解析 `--tman_act_mode` / `--activation_scales_pt`，构造 registry，调用 A8 替换逻辑 |
| `backends/qualcomm/utils/tman_w4a8.py` | 新增 | scale 加载与查询 | 负责 `.pt` 读取、key 规范化、校验、查询 |
| `backends/qualcomm/utils/utils.py` | 修改 | A8 模块替换层 | 新增 `TMANLinearA8` 与 `convert_qlinear_to_tman_linear_a8` |
| `backends/qualcomm/builders/custom_ops.py` | 修改 | 导出 A8 custom op | 新增 `tman::linear_a8` 与 fake/reference 实现 |
| `backends/qualcomm/builders/op_tman_linear_a8.py` | 新增 | A8 lowering | 生成显式 int8 边界和 A8 三段式 TMAN op |
| `backends/qualcomm/builders/qnn_constants.py` | 修改 | A8 QNN op 常量 | 增加 `OpTMANPrecomputeA8` / `OpTMANLinearA8` / `OpTMANFinalizeA8` |
| `backends/qualcomm/partition/common_defs.py` | 修改 | 分区识别新 op | allowlist 增加 `tman.linear_a8.default` |
| `backends/qualcomm/_passes/layout_transform.py` | 修改 | layout pass 识别新 op | 把 `tman.linear_a8.default` 纳入支持集合 |
| `backends/qualcomm/quantizer/annotators.py` | 修改 | annotator 兼容新 op | 为 `tman::linear_a8` 注册独立注解逻辑 |
| `backends/qualcomm/quantizer/custom_annotation.py` | 修改 | 去除旧 8a 阻塞 | 不再让旧 `tman::linear` 的 8a 限制误伤新 A8 op |
| `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANPrecomputeA8.cpp` | 新增 | A8 LUT 预处理 | 输入 int8，产出整数 LUT scratch |
| `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANLinearA8.cpp` | 新增 | A8 查表与累加 | 使用 `effective_scales` 做 group-wise 累加 |
| `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANFinalizeA8.cpp` | 新增 | A8 输出量化 | 输出 int8 tensor |
| `backends/qualcomm/runtime/op_packages/TMANOpPackage/include/hvx_funcs.h` | 修改 | A8 HVX kernel | 增加 A8 LUT 构造、查表、输出 finalize kernel |
| `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/TMANOpPackageInterface.cpp` | 修改 | Op 注册 | 注册 A8 三个新 op |
| `backends/qualcomm/runtime/op_packages/TMANOpPackage/config/TMANOpPackageHtp.xml` | 修改 | Op 定义声明 | 补充 A8 op 定义 |
| `backends/qualcomm/runtime/op_packages/TMANOpPackage/Makefile` | 修改 | 构建入口 | 确保新 A8 源文件参与 x86/v75/android 构建 |
| `backends/qualcomm/tests/*` | 修改 / 新增 | 测试 | 增加 scale、export、lowering、kernel、e2e 测试 |

## 9. 推荐数据结构设计

## 9.1 `ActivationScaleRegistry`

```text
class ActivationScaleRegistry:
    scales: Dict[str, Dict[str, torch.Tensor]]
    methods:
        load()
        normalize_key()
        validate()
        get(module_path)
```

职责：

- 只管理 scale 数据；
- 不依赖 FX、QNN、TMAN 实现细节；
- 提供稳定查询接口给模块替换层。

## 9.2 `TMANLinearA8`

```text
class TMANLinearA8(nn.Module):
    qweight
    scales
    qzeros
    g_idx
    wf_unsqueeze_zero
    wf_unsqueeze_neg_one
    input_scale
    output_scale
    bits
    group_size
    symmetric
    gptq_v2
```

职责：

- 承载 A8 所需全部静态信息；
- 让导出图自然生成 `tman::linear_a8`。

## 9.3 `tman::linear_a8` schema

建议让 custom op 显式携带：

- 旧 GPTQ 全部静态参数
- `input_scale`
- `output_scale`

这样 builder 可以不依赖外部上下文，直接从 op 参数拿到 lowering 所需信息。

## 10. 推荐实现顺序

建议按以下顺序推进编码：

1. 配置入口和 scale registry
2. `TMANLinearA8` 与 `tman::linear_a8`
3. builder / lowering 侧的显式边界转换
4. A8 OpPackage 注册和空壳实现先打通编译链
5. A8 HVX kernel 数学实现
6. 单层 reference 与 kernel 对齐
7. Qwen3 compile-only
8. 设备侧推理
9. benchmark

这样的顺序有两个好处：

- 能尽快把“导出链路通不通”和“kernel 数学对不对”拆开验证；
- 避免一开始就把所有问题都堆到设备端排查。

## 11. 风险控制与回退策略

## 11.1 风险控制

本架构通过以下方式控制风险：

- 新增独立 A8 模块、独立 export op、独立 QNN op；
- 旧 A16 路径不动；
- scale 文件校验前置；
- 不与现有全图 PTQ 复用同一条参数生成链路；
- runtime 层显式区分 A16 和 A8 op 名称。

## 11.2 回退策略

若 `W4A8` 任一阶段出现问题，应能快速回到：

- `--use_tman --tman_act_mode a16`

也就是说，回退策略不是“手动改代码”，而是“架构上天然保留旧路径”。

## 12. 最终架构结论

本次 `Qwen3 W4A8` 改造的推荐架构是：

- 保留现有 `W4A16` TMAN 路径完全不变；
- 新增一条独立的 `W4A8` 支线；
- 在 Python 层新增 `TMANLinearA8`；
- 在 export 层新增 `tman::linear_a8`；
- 在 QNN lowering 层显式插入 FP16<->INT8 边界 `Convert`；
- 在 runtime 层新增 `TMANPrecomputeA8/TMANLinearA8/TMANFinalizeA8`；
- 在 HVX kernel 层新增整数 LUT 和 requant 路径；
- 其它非 linear 算子继续保持 FP16 图执行。

这套架构的核心优点是：

- 语义清晰
- 风险隔离好
- 便于分模块并行编码
- 便于回归验证
- 便于后续继续扩展到更多模型或更激进优化

后续编码工作应以本文档为实现蓝图，优先把：

- scale 管理
- A8 custom op 表达
- A8 lowering
- A8 OpPackage

四条主线打通，再进入 kernel 调优和 benchmark 阶段。

## 13. 当前实际落地架构

本节用于回写“设计方案已经如何落地”，以及“当前代码与原始设计之间已收敛到什么程度”。

## 13.1 当前已落地的模块边界

截至当前，下面这些设计项已经实现：

- 编译入口层：
  - `examples/qualcomm/oss_scripts/qwen3/qwen3.py`
  - 已支持：
    - `--tman_act_mode {a16,a8}`
    - `--activation_scales_pt`
- 外部 scale 管理层：
  - `backends/qualcomm/utils/tman_w4a8.py`
- Python 模块层：
  - `TMANLinearA8`
  - `convert_qlinear_to_tman_linear_a8(...)`
- Export Custom Op 层：
  - `tman::linear_a8`
- QNN Lowering 层：
  - `backends/qualcomm/builders/op_tman_linear_a8.py`
- Runtime OpPackage 层：
  - `TMANPrecomputeA8`
  - `TMANLinearA8`
  - `TMANFinalizeA8`

因此，当前代码已经不再是“只存在设计文档”，而是已经具备一条可运行、可验证的 `W4A8` baseline 支线。

## 13.2 当前实际 static tensor 与 scratch 设计

与原始设计相比，当前已落地的数据表示更具体了：

### 1. 权重 static tensor

当前 runtime 使用的 A8 static qweight 已收敛为：

- `[M, G, Qg, bits]`

含义：

- `M`：输出通道数
- `G`：weight quant group 数，`G = K / group_size`
- `Qg`：每个 group 中包含的 `4-activation qgroup` 数，`Qg = group_size / 4`
- `bits`：每个 qgroup 在各 bit-plane 上对应的 `0~15` nibble index

这比早期的：

- `[M, K]`

或中间态的：

- `[M, Q, bits]`

都更接近最终 Hexagon/HVX 的 group/block 处理方式。

### 2. LUT scratch

当前 A8 `Precompute` 输出的 LUT scratch 在语义上按：

- `[row, group, qgroup_in_group, 16]`

组织。

每个 `qgroup` 的 LUT 为：

- `16 * int16 = 32B`

每 `4` 个 qgroup 组成一个 block：

- `4 * 32B = 128B`

正好对应一条 HVX vector line。

当前实际代码里又增加了一层“逻辑布局”和“对外 ABI”的分离：

- 逻辑语义上，A8 qweight / LUT scratch 仍然按 block 方式组织；
- 对外 builder/runtime ABI 为了更稳妥地适配当前 QNN OpPackage / XML 的 4D 约束，已重新压回 4D packed 形态。

当前对外 ABI 形态为：

- qweight:
  - `[1, M, G, qgroup_vector_blocks * (4 * bits)]`
- LUT scratch:
  - `[1, row, group, qgroup_vector_blocks * (4 * 16)]`

这意味着：

- helper kernel 内部继续按 block / vector-line 逻辑处理；
- QNN schema / graph compile 阶段继续看到 4D packed tensor。

## 13.3 当前 helper kernel 组织方式

当前 `W4A8` kernel 关键逻辑已经统一收敛到：

- `backends/qualcomm/runtime/op_packages/TMANOpPackage/include/hvx_funcs.h`

并已引入下面这些与最终 HVX 化高度相关的结构：

- `TmanA8Layout`
- `TMAN_A8_QGROUPS_PER_VECTOR`
- `tman_a8_lut_offset`
- `tman_a8_lut_block_offset`
- `tman_a8_weight_idx_offset`
- `tman_a8_scale_offset`
- `tman_a8_build_lut_block_scalar`
- `tman_a8_group_dot_block_scalar`
- `tman_a8_precompute_lut_scalar`
- `tman_a8_linear_scalar`
- `tman_a8_finalize_scalar`

这意味着：

- op 层已经足够薄；
- layout / offset / tile 结构已经稳定；
- 后续真正替换成 HVX intrinsic 时，改动将主要集中在 helper kernel 层。

在此基础上，当前 helper kernel 又完成了两轮新的收敛：

### 1. tile / pack-store 组织收敛

当前已引入：

- `TmanA8IntTile`
- `TmanA8FloatTile`
- `TmanA8OutputBlock`
- `tman_a8_accumulate_group_tile_scalar`
- `tman_a8_apply_scales_to_group_tile_scalar`
- `tman_a8_store_accum_tile_scalar`
- `tman_a8_pack_output_block_scalar`
- `tman_a8_store_output_block`

这说明 `LinearA8` / `FinalizeA8` 已不再是纯逐元素标量写回，而是开始按：

- `row_tile x m_tile`
- `128B output block`

表达更接近最终 HVX 的处理边界。

### 2. 部分 HVX intrinsic 化

当前又新增了两类更接近最终设备路径的实现：

- `LinearA8`
  - 目标专用 `W4G128 sym` helper 中已引入 LUT block 的向量寄存器 staging
- `FinalizeA8`
  - 已引入一段更实质的 HVX 路径：
    - `fp32 vector -> s32 round -> clamp -> s16 sat`
    - 再做 block 级 pack/store

因此，当前 A8 helper 已经不是“纯 scalar baseline”，而是“部分 intrinsic 化的 target-specific helper”。

## 13.4 当前正确性分层结论

当前项目已经形成三套语义层次清晰的 reference：

1. `high-level reference`
   - `dequantize_gptq_weight + matmul`
2. `grouped reference`
   - 代表当前项目想要收敛到的 group-wise 数学语义
3. `runtime baseline`
   - 当前 LUT/index/block 风格实现

当前验证结论是：

- `grouped reference == runtime baseline`
- 与 `high-level reference` 相比，极少数模块会出现 `1 bin` 级差异

因此，当前更贴近实现目标的正确性口径应该优先参考：

- `grouped reference`
- `runtime baseline`

而不是单纯要求所有模块都与 `dequantize+matmul` 逐点完全一致。

## 13.5 当前可运行验证入口

当前机器上没有完整 ExecuTorch 运行环境，因此已经配套建立了轻量验证工具链：

- `backends/qualcomm/tests/test_tman_w4a8.py`
- `tools/tman_w4a8_reference_check.py`
- `tools/tman_w4a8_qwen3_scale_audit.py`
- `tools/tman_w4a8_real_layer_probe.py`
- `tools/tman_w4a8_model_slice_audit.py`

目前推荐使用的真实权重文件为：

- `/home/lmx/Projects/tman/models/Qwen3-1.7B-GPTQ-Int4/model.safetensors`

该文件当前已确认：

- `gptq_v2 = False`

代码已兼容此格式。

当前轻量验证结论也应同步更新为：

- `backends/qualcomm/tests/test_tman_w4a8.py`
  - 当前已通过 `15` 个轻量测试
- `tools/tman_w4a8_reference_check.py`
  - 当前通过

## 13.6 当前阶段的架构判断

当前项目已经从：

- “A8 扩展设计阶段”

进入到：

- “A8 helper 持续向 HVX/Hexagon 风格收敛阶段”
- “可以阶段性交付给其它合作者做 QNN / OpPackage 真编译与 device smoke test 的阶段”

也就是说，后续架构工作不再以扩充新模块为主，而应以：

- helper kernel 的 HVX 风格化
- block/tile 数据布局稳定
- runtime 与 grouped reference 的一致性保持
- compile-only / package build / device smoke test 打通

为主线推进。

## 13.7 当前推荐的后续演进顺序

结合当前已落地状态，后续最合理的实现顺序为：

1. 在完整 ExecuTorch / QNN 环境中先做 compile-only 验证
2. 用真实 `TMANOpPackage` 构建链做 x86 / v75 / android 编译
3. 用真实设备做 graph prepare / runtime load / inference smoke test
4. 在 smoke test 稳定后，继续推进 `LinearA8` / `FinalizeA8` 的最后一段 intrinsic 化
5. 最后做 benchmark 与性能归因
