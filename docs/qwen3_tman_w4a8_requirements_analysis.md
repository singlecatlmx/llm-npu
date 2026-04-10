# Qwen3 TMAN W4A8 需求分析

## 1. 文档目的

本文档用于把当前项目现状、`W4A16` TMAN linear 的真实执行链路、以及下一步 `W4A8` 改造的目标、边界、风险和验收标准一次性说明白，作为后续编码工作的统一参考。

这份文档重点避免后续把下面几件事混在一起：

- ExecuTorch Qualcomm backend 的通用能力
- 当前 TMAN 自定义 linear 的现有能力
- `qwen3.py` 里的全图 PTQ 模式
- 本次真正要做的事情：基于外部激活 scale 的 `W4A8` TMAN linear

## 2. 结论摘要

### 2.1 当前项目的真实状态

当前仓库是一个带 Qualcomm QNN backend 和自定义 `TMANOpPackage` 的 ExecuTorch 工程。它的核心优化点是把 GPTQ `W4` 的线性层改造成一条 TMAN LUT 路径：

- Python 侧读取 GPTQ 打包权重，并重排成 HVX 友好的布局；
- 导出图中使用自定义 op `tman::linear`；
- QNN lowering 时把 `tman::linear` 展开成：
  `Convert -> TMANPrecompute -> TMANLinear -> TMANFinalize -> Convert`；
- Hexagon HVX 内核完成 LUT 构造、查表累加、位串行恢复和最终输出。

当前 TMAN 路径本质上是一条 `W4A16` 路径：

- 权重：GPTQ int4
- 激活：TMAN kernel 内部按 FP16/A16 路径处理
- 输出：回到 FP16

### 2.2 本次需求的真实目标

本次需求不是“直接打开现有 `8a8w` PTQ 选项”，而是：

- 保持 GPTQ `W4` 权重不变；
- 把 TMAN linear 的激活路径从 `A16` 升级为 `A8`；
- linear 的 `input_scale` / `output_scale` 从外部 `.pt` 文件导入；
- 激活量化方式是对称 per-tensor int8；
- 非 linear 算子仍走原有路径；
- 需要把 linear 的 int8 激活量化与周围非 linear 的 FP16/A16 计算正确衔接起来。

### 2.3 最重要的工程判断

当前代码并没有一条现成的 `W4A8` 通路，原因很明确：

- `TMANLinear` 模块本身不携带激活 scale；
- `op_tman_linear.py` 里始终会在 TMAN 前后插 `Convert`，内部仍按 FP16 处理；
- `TMANPrecompute.cpp` / `TMANLinear.cpp` / `TMANFinalize.cpp` 都是围绕现有 FP16 激活流写的；
- `backends/qualcomm/quantizer/custom_annotation.py` 中明确写了 `tman::linear currently does not support 8a`；
- 外部 `activation_scales_a8_per-tensor_sym.pt` 还没有接进模块构建、导出、lowering、运行时。

所以，`W4A8` 必须被视为一条新的 TMAN 专用路径，而不是改几个参数就能打通的小修小补。

## 3. 已确认的项目事实

## 3.1 目标模型的真实配置

从 `/home/lmx/Test/GPTQ_Opus/Qwen3-1.7B_w4a8_gptq/quantize_config.json` 可确认：

- `bits = 4`
- `group_size = 128`
- `sym = true`
- `desc_act = false`
- `quant_method = gptq`

从 `/home/lmx/Test/GPTQ_Opus/Qwen3-1.7B_w4a8_gptq/config.json` 可确认：

- 模型类型：`Qwen3ForCausalLM`
- 层数：`28`
- `hidden_size = 2048`
- `intermediate_size = 6144`
- `num_attention_heads = 16`
- `num_key_value_heads = 8`
- `head_dim = 128`

这说明本次近端目标可以明确聚焦在：

- `Qwen3-1.7B`
- GPTQ `W4`
- `group_size = 128`
- `sym = true`

## 3.2 外部激活 scale 文件的真实结构

从 `/home/lmx/Test/GPTQ_Opus/Qwen3-1.7B_w4a8_gptq/activation_scales_a8_per-tensor_sym.pt` 可确认：

- 文件对象类型：`dict`
- key 总数：`392`
- 每个 value：`torch.float32`，shape 为 `[1]`
- key 示例：
  - `model.model.layers.0.self_attn.q_proj.input_scale`
  - `model.model.layers.0.self_attn.q_proj.output_scale`

覆盖的 linear 类型一共 7 类：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

覆盖范围是：

- `28` 层
- 每层 `7` 个 linear
- 每个 linear 各有 `input_scale` 和 `output_scale`

即：

- `28 x 7 x 2 = 392`

同时也确认了当前文件**没有覆盖**：

- `lm_head`
- embedding

因此，本次任务的精确范围就是：Qwen3 解码层内部的这 7 类 linear，而不是模型内所有矩阵乘法。

## 4. 当前架构与真实执行链路

## 4.1 主入口脚本

主示例入口是 `examples/qualcomm/oss_scripts/qwen3/qwen3.py`。

它的核心流程是：

1. 从 `model/static_qwen3.py` 构造静态 Qwen3 模型骨架；
2. 通过 `GPTQModel.from_quantized(...)` 加载 GPTQ 量化模型；
3. 先把 `nn.Linear` 替换成 GPTQ 的 `TorchQuantLinear`；
4. 再加载 quantized state dict；
5. 如果打开 `--use_tman`：
   - 把 GPTQ qlinear 替换成 `TMANLinear`
   - attention 路径切到 `prepare_tman()`
6. 导出 / 可选 PTQ / 调 Qualcomm backend 做 lowering；
7. 把 `tman::linear` 降成 QNN 自定义 op 序列；
8. 最终经由 QNN + `TMANOpPackage` 在 HTP / Hexagon 上执行。

## 4.2 静态模型结构

`examples/qualcomm/oss_scripts/qwen3/model/static_qwen3.py` 中：

- `Qwen3Attention`
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
- `Qwen3MLP`
  - `gate_proj`
  - `up_proj`
  - `down_proj`
- `Qwen3DecoderLayer`
  - `self_attn`
  - `mlp`
- `Qwen3Model`
  - `layers = ModuleList([...])`
- `Qwen3ForCausalLM`
  - `model`
  - `lm_head`

在 TMAN 模式下：

- attention 侧通过 `prepare_tman()` 切到 `forward_tman()`
- MLP 侧的 linear 通过 `convert_qlinear_to_tman_linear(...)` 原地替换

## 4.3 Python 侧 linear 转换链路

相关逻辑主要在 `backends/qualcomm/utils/utils.py`：

- `convert_linear_to_qlinear(...)`
  - 把 `nn.Linear` 替换成 GPTQ `TorchQuantLinear`
- `convert_qlinear_to_tman_linear(...)`
  - 把 GPTQ qlinear 替换成 `TMANLinear`
- `TMANLinear`
  - 当前保存的 GPTQ 相关张量：
    - `qweight`
    - `scales`
    - `qzeros`
    - `g_idx`
    - `wf_unsqueeze_zero`
    - `wf_unsqueeze_neg_one`
  - 当前保存的元信息：
    - `group_size`
    - `bits`
    - `symmetric`
    - `gptq_v2`
  - `forward(...)` 内部调用自定义 op `tman_linear(...)`

当前 `TMANLinear` 明显**还没有**保存：

- `input_scale`
- `output_scale`
- 激活量化模式

这是后续改造必须补上的第一层数据承载点。

## 4.4 FX 自定义 op 层

`backends/qualcomm/builders/custom_ops.py` 里定义了导出时使用的：

- `@torch.library.custom_op("tman::linear")`

当前它的 CPU / 参考实现是：

- 先把 GPTQ 权重反量化；
- 再直接做 `torch.matmul(x, weights)`；
- 输出 dtype 跟随输入 dtype。

这只是导出 / 假执行 / 参考比对用的语义实现，不是最终推理实现。它目前也**没有表达**未来 `W4A8` 所需的激活量化语义。

## 4.5 PTQ 与 Pass 链路

`qwen3.py` 通过 `SingleLlama.quantize(...)` 加上 Qualcomm pass manager 走完整条量化链路：

- `torch.export.export(...)`
- `prepare_pt2e(...)`
- 用 prompt 做校准
- `convert_pt2e(...)`
- 再经过 `backends/qualcomm/_passes/qnn_pass_manager.py` 中定义的一系列 pass

当前这条链路的关键事实是：

- quant attrs 来自 observer / QDQ 节点；
- `AnnotateQuantAttrs` 是从图里的 quant / dequant 节点中回收 scale、zero_point；
- 它本质上是“观测驱动”的量化参数生成方式。

而本次需求要求 linear 的激活 scale 必须来自外部 `.pt` 文件，所以这与当前 observer 驱动链路存在天然错位，后续必须显式补一条“固定外部量化参数注入”的通路。

## 4.6 QNN lowering 路径

最关键的 lowering 文件是 `backends/qualcomm/builders/op_tman_linear.py`。

当前 `tman::linear` lowering 的核心步骤是：

1. 用 `unpack_gptqv2(...)` 解 GPTQ 权重；
2. 用 `hvx_preprocess_weights(...)` 把权重重排成 HVX 友好的布局；
3. 构造中间 scratch tensor：
   - `l`：LUT 相关 scratch
   - `c`：累加 scratch
4. 如果 `self.add_convert` 为 `True`：
   - 在 TMAN 前插 input `Convert`
   - 在 TMAN 后插 output `Convert`
5. 最终发出 3 个 TMAN 自定义 op：
   - `TMANPrecompute`
   - `TMANLinear`
   - `TMANFinalize`

当前一个非常关键的事实是：

- `self.add_convert = True`

也就是说，当前 lowering 固定假设 TMAN 内部走 FP16 激活路径，外层用 `Convert` 做包裹，这正是 `W4A16` 的代码形态。

## 4.7 Runtime OpPackage 路径

核心文件是：

- `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANPrecompute.cpp`
- `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANLinear.cpp`
- `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANFinalize.cpp`
- `backends/qualcomm/runtime/op_packages/TMANOpPackage/include/hvx_funcs.h`

当前 3 个 op 的职责分工是：

- `TMANPrecompute`
  - 输入：激活张量 `x`
  - 输出：LUT scratch `l`
  - 内部生成：
    - LUT 项
    - `ls`：激活组 scale
    - `lb`：LUT bias / 补偿项

- `TMANLinear`
  - 输入：
    - LUT scratch `l`
    - 重排后的 `qweight`
    - 重排后的 `scales`
  - 输出：
    - 累加 scratch `c`
  - 内部完成 HVX 查表、部分累加、weight scale 应用

- `TMANFinalize`
  - 输入：scratch `c`
  - 输出：FP16 结果
  - 内部完成位串行恢复

### 4.7.1 当前 kernel 的基础假设

当前实现和 FP16 激活路径是强绑定的：

- `TMANPrecompute` 使用 `XType = __fp16`
- `TMANLinear` 使用 `XType = __fp16`
- `TMANFinalize` 输出 `XType = __fp16`

### 4.7.2 当前支持的 weight 组合并不泛化

`TMANPrecompute.cpp` / `TMANLinear.cpp` 当前是按固定分支支持若干组合：

- `w2g64` 非对称
- `w4g128` 对称
- `w4g128` 非对称
- `w4g64` 非对称
- bitnet 特例

对当前目标模型来说，真正命中的分支是：

- `w4g128`
- `sym = true`

因此，后续 `W4A8` 第一阶段完全可以先聚焦：

- `bits = 4`
- `group_size = 128`
- `sym = true`

而不是一开始就宣称支持所有组合。

## 4.8 Host / Device 打包链路

`TMANOpPackage` 的构建由
`backends/qualcomm/runtime/op_packages/TMANOpPackage/Makefile`
负责。

当前可见的关键构建目标包括：

- `x86_64-linux-clang`
- `hexagon-v75`
- `aarch64-android`

这意味着 `W4A8` 改造不只是设备内核改造，还包括：

- host 侧注册 / 编译 / 校验仍然要通；
- x86 版本仍要能编；
- Hexagon v75 版本仍要能编；
- Android 侧运行时链路也不能断。

## 5. 当前能力边界

## 5.1 当前已经具备的能力

- Qwen3 静态导出与 Qualcomm backend lowering
- GPTQ int4 权重加载
- 部分 linear 的 TMAN 自定义路径
- HVX 友好的权重重排
- Hexagon 侧 `W4A16` 查表推理路径
- SM8650 / v75 平台上的 Qualcomm backend 部署能力

## 5.2 当前明确还不具备的能力

- 外部激活 scale 导入到 TMAN linear
- 每层 linear 固定 `input_scale` / `output_scale` 透传
- TMAN 内部 int8 激活计算路径
- `W4A8` 的 custom op ABI
- `W4A8` 的 host / device OpPackage 实现
- 一个语义清晰、和现有全图 PTQ 模式解耦的 `W4A8` 编译入口

## 5.3 一个必须正视的语义歧义

当前脚本里常见用法是：

- `--ptq 16a4w --use_tman`

但任务描述强调的是：

- 只改 linear 激活位宽到 int8；
- 其它算子仍走原有路径；
- linear 的 int8 量化结果要和周围 FP16 计算衔接。

这说明后续实现**不能**简单把：

- “W4A8”

等价理解成：

- “现有全图 `8a8w` PTQ 模式”

从需求文本看，本次更像是一条 TMAN 专用的局部激活量化升级路径，而不是一条整图量化模式切换路径。

## 6. 目标需求定义

## 6.1 核心目标

在 Qwen3 Qualcomm 路径上实现 `W4A8` 的 TMAN linear 推理：

- `W4`：保持 GPTQ int4 权重不变；
- `A8`：linear 激活采用对称 per-tensor int8 量化；
- `input_scale` / `output_scale` 从外部 `.pt` 文件导入；
- 非 linear 算子保持原有执行路径；
- linear 输出必须能正确衔接回周围计算图。

## 6.2 本阶段纳入范围的算子

对当前 Qwen3 目标模型，本阶段只覆盖以下 7 类 linear：

- attention：
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
- MLP：
  - `gate_proj`
  - `up_proj`
  - `down_proj`

覆盖全部 `28` 个 decoder layer。

## 6.3 本阶段明确不纳入范围

除非后续明确扩 scope，第一阶段不要求覆盖：

- `lm_head`
- embedding
- normalization
- softmax / matmul / 普通 elementwise
- 所有模型 / 所有算子的通用 int8 化
- 非 Qwen3 模型
- `desc_act=True` 的 GPTQ 变体

## 7. 详细功能需求

## 7.1 外部 scale 加载需求

系统必须：

- 在编译 / 导出阶段加载外部激活 scale 文件；
- 校验文件存在、key 数量、dtype 和层覆盖是否完整；
- 建立从文件 key 到当前模型模块名的确定性映射；
- 一旦发现任何 linear 缺失 `input_scale` 或 `output_scale`，必须明确报错并终止。

这里有一个已经确认的真实问题：

- 外部文件 key 前缀是 `model.model.layers...`
- 当前静态模型结构是 `Qwen3ForCausalLM.model.layers...`

所以“key 规范化”是正式需求，不是一个小细节。

## 7.2 Python 模块承载需求

`TMANLinear` 或其后继实现必须能够携带：

- GPTQ 权重与元信息；
- 激活量化模式；
- `input_scale`；
- `output_scale`。

并且这些信息要能贯穿：

- 模块替换
- 导出
- lowering
- 序列化到 QNN 常量 / 参数

## 7.3 自定义 op 语义需求

导出的自定义 op 必须能够明确表达 `W4A8` 语义。

可接受的方向只有两种：

1. 扩展现有 `tman::linear`
2. 新增一个独立 op，例如 `tman::linear_a8`

但无论选哪种，都必须能表达：

- 输入激活量化
- 内部整数 / LUT 计算路径
- 输出 requant / dequant 逻辑

同时，CPU / 参考实现也必须同步更新，保证：

- export 能继续工作；
- 单测能对齐 PyTorch reference；
- 数值语义与最终 `W4A8` 设计一致。

## 7.4 激活量化语义需求

linear 激活量化必须满足：

- 粒度：per-tensor
- 对称：symmetric
- zero point：固定为 `0`
- scale 来源：外部 `.pt`

这个要求同时适用于：

- linear 输入量化
- linear 输出量化定义

并且这些 linears 的量化参数**不能再依赖** runtime prompt 校准或 observer 自动产出的 qparams。

## 7.5 边界衔接需求

`W4A8` linear 必须和周围算子正确衔接。

至少需要明确：

- FP16/A16 张量在什么位置被量化成 int8；
- linear 结果在什么位置被转换回周围图所需表示；
- linear 输出是：
  - 只在内部短暂保持 int8，随后立刻还原；
  - 还是图级别显式保留成 int8 再交给下游。

结合任务描述，第一阶段更推荐的基线是：

- TMAN 内部变成 A8；
- 外部周围计算图仍看到原有执行路径；
- 也就是“TMAN 内部 A8，边界外仍按原逻辑衔接”。

## 7.6 Lowering 需求

`backends/qualcomm/builders/op_tman_linear.py` 必须能表达 `W4A8` 模式。

这通常意味着要完成下面几类事情：

- 不能再把 TMAN 视为纯 FP16 内核；
- 需要更新或条件化控制当前 `add_convert = True` 的行为；
- 需要给 TMAN 输入 / 输出定义正确的 QNN tensor dtype / quant encoding；
- 需要把激活 scale 常量传进自定义 op 序列；
- 同时保留当前 `W4A16` 路径作为 fallback。

## 7.7 OpPackage 需求

`TMANPrecompute`、`TMANLinear`、`TMANFinalize` 都必须支持新的 A8 路径。

这很可能意味着：

- 新的输入 / 输出 dtype；
- 新的静态参数或 tensor 输入；
- 新的 int8 激活算术路径；
- 更新补偿和缩放流程。

一个非常重要的底层要求是：

- 当前 `ls / lb` 逻辑是围绕 FP16 激活路径设计的；
- `W4A8` 不能被当成“改个 typedef”。

后续必须把 kernel 数学语义和 scratch 语义重新审视，而不是表面改 dtype。

## 7.8 Runtime 与打包需求

该特性必须同时在下面两侧成立：

- host 侧 compile / validation
- device 侧 execution

因此，相关构建目标必须继续可用：

- `x86_64-linux-clang`
- `hexagon-v75`
- `aarch64-android`

同时，新 ABI 也必须与 `QNN_OP_PACKAGE_PATHS` 的注册方式兼容。

## 7.9 CLI 与配置入口需求

编译入口必须把 `W4A8` 语义显式化。

推荐要求是：

- 新增专用模式开关，而不是继续复用当前宽泛的 `--ptq`；
- 新增明确的外部激活 scale 文件路径参数。

例如可以接受的方向：

- `--tman_act_mode a8`
- `--activation_scales_pt <path>`

这样做的目的，是明确区分：

- 现有全图 PTQ 模式：
  - `8a8w`
  - `16a4w`
  - `16a4w_block`
- 与本次要做的 TMAN 专用 `W4A8`

## 8. 非功能需求

## 8.1 正确性

`W4A8` 路径必须能够和 PyTorch reference 对齐验证。

至少要覆盖：

- 单个 linear 的正确性
- decoder block 的正确性
- Qwen3 端到端生成的可用性

## 8.2 性能

需求目标是“加速 linear-heavy 场景”，尤其是：

- decode latency
- prefill throughput
- 内存带宽压力

因此，性能测试是必选项，不是可选项。

## 8.3 内存

实现必须控制 scratch buffer 增长。

当前 TMAN 路径已经有：

- `l`：LUT scratch
- `c`：accumulation scratch

`W4A8` 设计需要明确：

- 这些 buffer 是否复用；
- 尺寸是否变化；
- 是否需要新增 scratch tensor。

## 8.4 兼容性

第一阶段的兼容目标建议控制在：

- 模型：Qwen3-1.7B GPTQ int4 group128 symmetric
- SoC：SM8650
- HTP arch：v75

其它模型族、其它权重量化配置可以后续再扩。

## 8.5 可维护性

实现过程中必须保留：

- 当前 `W4A16` 路径；
- 旧命令 / 旧模型继续编译运行的能力；
- `W4A16` 和 `W4A8` 的代码边界清晰可读。

## 9. 受影响模块与职责分工

| 模块 / 文件 | 当前职责 | `W4A8` 需要承担的职责 |
| --- | --- | --- |
| `examples/qualcomm/oss_scripts/qwen3/qwen3.py` | 主编译 / 导出入口 | 加载外部 scale 文件，增加模式配置入口 |
| `examples/qualcomm/oss_scripts/qwen3/model/static_qwen3.py` | 静态 Qwen3 图 | 基本结构应保持稳定，但模块命名必须和 scale 映射对齐 |
| `backends/qualcomm/utils/utils.py` | qlinear -> TMANLinear 替换 | 给 `TMANLinear` 注入 `input_scale` / `output_scale` 与模式信息 |
| `backends/qualcomm/builders/custom_ops.py` | 导出时 custom op 和 fake impl | 扩展 op schema，补齐 `W4A8` 参考语义 |
| `backends/qualcomm/builders/op_tman_linear.py` | 把 `tman::linear` lower 成 QNN 自定义序列 | 支持 A8 输入输出与激活 scale 透传 |
| `backends/qualcomm/builders/utils.py` | GPTQ 解包和 HVX 权重重排 | 大概率复用，但需要确认是否存在 A16 假设 |
| `backends/qualcomm/quantizer/annotators.py` | `tman::linear` 的量化注解 | 保证新路径不再被 8a 能力缺口卡住 |
| `backends/qualcomm/quantizer/custom_annotation.py` | 特殊量化规则 | 去掉当前 `tman linear does not support 8a` 的阻塞点 |
| `backends/qualcomm/_passes/*` | quant attrs 传播与 lowering pass | 可能需要新增“固定外部 quant attrs 注入”路径 |
| `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANPrecompute.cpp` | 激活侧 LUT 预处理 | 增加 `W4A8` 激活侧逻辑 |
| `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANLinear.cpp` | LUT 查表与累加 | 增加 A8 路径的查表 / 累加 / scale 处理 |
| `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANFinalize.cpp` | 位串行恢复到 fp16 | 增加输出 requant / dequant 逻辑 |
| `backends/qualcomm/runtime/op_packages/TMANOpPackage/include/hvx_funcs.h` | 核心 HVX kernel | 新增或特化 `W4A8` kernel |
| `backends/qualcomm/runtime/op_packages/TMANOpPackage/Makefile` | host / device 构建 | 确保新 kernel 在 x86 / v75 / android 都能编 |
| `backends/qualcomm/tests/` | 校验体系 | 增加 `W4A8` 的单测和集成测试 |

## 10. 关键风险与待确认问题

## 10.1 外部 key 命名不一致

外部 scale 文件是 `model.model.layers...`，而当前静态模型路径预期是 `model.layers...`。

这个问题必须在第一步就定义清楚映射规则，否则后面所有 scale 注入都不稳定。

## 10.2 `output_scale` 的语义需要最终钉死

任务文本给了 `input_scale` 和 `output_scale`，但后续实现前还需要最后确认一件事：

- linear 输出是否先 requant 成 int8，再立刻 dequant 回 FP16；
- 还是希望把 int8 输出在图里显式保留下来。

从当前任务描述看，第一种更符合“其它算子保持原路径”的目标。

## 10.3 与现有 `--ptq` 的关系

当前脚本很多时候会带 `--ptq 16a4w`，但本次需求语义并不等价于：

- `8a8w`
- `16a4w`

这是一个产品语义边界，必须尽早冻结，避免后面实现一半再推翻。

## 10.4 Kernel 数学语义重构风险

当前 FP16 TMAN 路径依赖：

- 动态激活组 scale `ls`
- LUT bias / 补偿项 `lb`
- kernel 内部的 FP16 scale 应用

`W4A8` 可能会让一部分逻辑简化，但它依然是一项真正的 kernel 语义重构工作，不是机械替换 dtype。

## 10.5 Host 编译依赖风险

即便最终运行发生在 Hexagon，host 侧的 compile / register / validate 仍然要工作。

因此，“设备 kernel 写好了”并不等于特性完成。

## 10.6 Scope 失控风险

如果一开始就同时试图解决：

- Qwen3
- 所有 GPTQ 变体
- 所有 SoC
- 所有 PTQ 模式
- 所有 linear-like 算子

风险会很高。

建议第一阶段先把 Qwen3 这条链路闭环，再考虑泛化。

## 11. 推荐的第一阶段实现边界

第一阶段建议明确聚焦：

- Qwen3-1.7B
- GPTQ int4
- `group_size = 128`
- `sym = true`
- 28 层 decoder
- 每层 7 个 linear
- 外部标量 `input_scale` / `output_scale`
- TMAN 专用 `W4A8` 路径
- SM8650 / v75

第一阶段建议明确不做：

- `lm_head`
- embedding
- 整图 8bit 化重构
- 非 GPTQ 或 `desc_act=True`
- 所有模型族泛化

## 12. 验收标准

只有下面这些都满足，才应判定该特性完成。

## 12.1 数据与接口验收

- 外部 scale 文件可以成功加载；
- 所有目标 linear 都能匹配到；
- 一旦有缺失 / 命名不一致，能快速报错并指出具体层。

## 12.2 导出验收

- 导出图中明确携带 `W4A8` TMAN 所需信息；
- 不能请求 `W4A8` 却悄悄退回旧的 `W4A16` 路径。

## 12.3 Lowering 验收

- host 侧 QNN lowering 能成功完成；
- 生成的图中包含预期的 `W4A8` TMAN 自定义 op 序列。

## 12.4 构建验收

- `TMANOpPackage` 在以下目标上均能成功构建：
  - x86
  - v75
  - android 侧所需目标

## 12.5 Runtime 验收

- 能成功生成 `.pte`
- 能在目标设备上正常推理

## 12.6 数值验收

- 单层 reference 对比通过；
- 端到端模型输出在可接受范围内稳定。

## 12.7 性能验收

- 相比当前 `W4A16` 路径，至少要完成 prefill / decode 的 benchmark 记录；
- 即使第一阶段优先正确性，也必须把性能数据沉淀下来。

## 13. 推荐验证计划

建议把验证分成 5 层：

1. Python 单测
   - scale 文件加载
   - key 规范化
   - `TMANLinear` 的 `W4A8` 参考数学语义

2. Lowering 单测
   - 导出图里能看到预期 custom op / 参数
   - QNN lowering 在 host 侧能通过

3. OpPackage 单测
   - 用小尺寸 synthetic linear 验证 host / device 输出

4. 端到端测试
   - Qwen3 compile-only
   - Qwen3 设备侧推理

5. Benchmark
   - `W4A16` vs `W4A8`
   - 分开统计 prefill 和 decode

## 14. 最终需求表述

下一步编码工作的本质，是在当前 ExecuTorch + QNN + TMAN 栈之上，为 Qwen3 构建一条专用的 `W4A8` TMAN linear 路径，并把外部 per-linear 激活 scale 文件作为 linear 激活量化参数的唯一事实来源。

这项工作必须完整闭环下面整条链路：

- scale 加载
- 模块承载
- export
- lowering
- OpPackage ABI
- HVX kernel
- host 构建
- device runtime
- 正确性验证
- 性能验证

同时必须保留当前 `W4A16` 路径，并把第一阶段范围严格控制在 Qwen3 decoder 内部这 7 类 linear 上。这样后续编码工作才有清晰边界，也更容易稳定落地。

## 15. 当前实现回写

本节用于把已经完成的编码工作回写到需求文档中，避免后续窗口继续开发时仍把项目视为“纯设计阶段”。

## 15.1 已落地的 W4A8 扩展项

截至当前，下面这些能力已经完成了第一阶段落地：

- `qwen3.py` 已新增专用模式入口：
  - `--tman_act_mode {a16,a8}`
  - `--activation_scales_pt`
- 已新增外部激活 scale 管理层：
  - `backends/qualcomm/utils/tman_w4a8.py`
  - 支持 key 规范化、覆盖校验、模块路径查询
- 已新增 Python 模块扩展：
  - `TMANLinearA8`
  - `convert_qlinear_to_tman_linear_a8(...)`
- 已新增导出层 custom op：
  - `tman::linear_a8`
- 已新增 A8 builder/lowering：
  - `backends/qualcomm/builders/op_tman_linear_a8.py`
- 已新增 A8 runtime 三段式 op：
  - `TMANPrecomputeA8`
  - `TMANLinearA8`
  - `TMANFinalizeA8`
- A8 runtime 的核心 helper 已统一下沉到：
  - `backends/qualcomm/runtime/op_packages/TMANOpPackage/include/hvx_funcs.h`

## 15.2 当前 runtime 组织状态

当前 `W4A8` runtime 已不是最初的“直接乘法 baseline”，而是已经收敛到更接近 TMAN 真正数据流的实现组织：

- 激活侧按 `4` 个激活构造 `16` 项 LUT
- 权重侧静态 tensor 已从 unpack 后的 `[M, K]` W4 权重，收敛为 grouped LUT 索引张量
- 当前 runtime static qweight 逻辑形状为：
  - `[M, G, Qg, bits]`
  - 其中：
    - `G = K / group_size`
    - `Qg = group_size / 4`
- `PrecomputeA8` 侧的 LUT scratch 已显式按：
  - `[row, group, qgroup_in_group, 16]`
  组织
- `hvx_funcs.h` 中已经存在 block/vector-line 风格 helper：
  - `TMAN_A8_QGROUPS_PER_VECTOR = 4`
  - 即 `4 qgroup * 16 int16 = 128B = 1 条 HVX vector line`

这说明后续真正替换成 HVX intrinsic 时，重点将集中在 helper 层，而不是继续变更 op 接口。

同时，当前实现又补上了两点新的工程收敛：

### 15.2.1 当前 A8 路径已经收敛到目标专用配置

当前 builder / Python 模块层 / runtime op 已统一把 A8 路径收敛到项目真实目标：

- `bits = 4`
- `group_size = 128`
- `symmetric = true`

也就是说，当前 A8 不再保留“看起来像可泛化到多配置”的外观，而是明确服务：

- `Qwen3-1.7B`
- GPTQ `W4G128`
- `sym = true`

### 15.2.2 当前对外 ABI 已压回 4D packed 形态

逻辑语义上，当前 A8 helper 仍然按 block / vector-line 方式理解：

- qweight logical:
  - `[M, G, Qg, bits]`
- LUT scratch logical:
  - `[row, group, qgroup_in_group, 16]`

但为了降低 QNN / OpPackage 真编译阶段的 rank/schema 风险，当前 builder/runtime 对外 ABI 已重新压回 4D packed tensor：

- qweight ABI:
  - `[1, M, G, qgroup_vector_blocks * (4 * bits)]`
- LUT scratch ABI:
  - `[1, row, group, qgroup_vector_blocks * (4 * 16)]`

也就是说：

- helper kernel 内部继续按 block/vector-line 逻辑消费数据；
- QNN 接口层继续维持更稳妥的 4D physical tensor 形态。

## 15.3 当前真实验证结论

当前机器没有完整的 ExecuTorch 运行环境，因此已建立了一组轻量工具脚本，在 `gptq` conda 环境中对真实权重和真实外部 scale 做局部与批量验证。

当前主要验证使用的权重文件是：

- `/home/lmx/Projects/tman/models/Qwen3-1.7B-GPTQ-Int4/model.safetensors`

对应外部激活 scale 文件是：

- `/home/lmx/Test/GPTQ_Opus/Qwen3-1.7B_w4a8_gptq/activation_scales_a8_per-tensor_sym.pt`

已确认：

- `model.safetensors` 对应 `gptq_v2 = False`
- 当前代码已兼容 GPTQ v1 / v2 两种 qzeros 表示

当前可重复运行的验证入口包括：

- `python backends/qualcomm/tests/test_tman_w4a8.py`
- `python tools/tman_w4a8_reference_check.py`
- `python tools/tman_w4a8_qwen3_scale_audit.py ...`
- `python tools/tman_w4a8_real_layer_probe.py ...`
- `python tools/tman_w4a8_model_slice_audit.py ...`

验证结论如下：

- `test_tman_w4a8.py` 当前已通过 `15` 个轻量测试
- scale 文件对目标 `196` 个 decoder 内部 linear 覆盖完整
- `runtime baseline` 与 `grouped reference` 完全一致
- 与 `high-level dequant+matmul reference` 相比，极少数模块会出现 `1 个 INT8 bin` 的差异

另外，当前 helper kernel 已经不再停留在纯标量 baseline：

- `LinearA8` 已引入目标专用的 block staging helper，开始显式按 LUT block 做向量寄存器式载入
- `FinalizeA8` 已引入一段更实质的 HVX 路径：
  - `fp32 vector -> s32 round -> clamp -> s16 sat`
  - 最后再做 block 级 pack/store

这说明当前项目已进入：

- “部分 HVX intrinsic 化 + compile/device handoff 准备就绪”

的阶段，而不再只是“轻量 reference baseline”。

## 15.4 当前已知边界

在使用：

- 权重：`/home/lmx/Projects/tman/models/Qwen3-1.7B-GPTQ-Int4/model.safetensors`
- scale：`/home/lmx/Test/GPTQ_Opus/Qwen3-1.7B_w4a8_gptq/activation_scales_a8_per-tensor_sym.pt`
- `out_features_slice = 8`

进行全量 `196` 个目标 linear 切片审计时：

- 若采用严格浮点差异口径，少数模块仍会出现 `1 个 INT8 bin` 等价的差异
- 这类差异目前已经被归类为：
  - `high-level reference`
  - 与 `grouped/runtime` 的参考定义差异
- 不再被视为当前 runtime baseline 的逻辑错误

当前两个典型模块是：

- `model.layers.17.self_attn.k_proj`
- `model.layers.6.mlp.down_proj`

在这两个模块上：

- `grouped reference == runtime baseline`
- `high-level reference` 与两者存在 `1 bin` 级偏差

因此，对当前阶段更贴近项目目标的判定口径应为：

- `bin_only + qdiff_tol = 1`

在此口径下，`196 / 196` 个目标 linear 已全部通过。

## 15.5 当前优先级更新

当前项目已从“设计阶段”进入：

- “A8 helper kernel 持续向 HVX/Hexagon 收敛”
- “可以阶段性交付给其它合作者做真编译 / device smoke test”

并行推进的阶段。

后续优先级建议调整为：

1. 在完整 ExecuTorch / QNN 环境中做 compile-only 验证
2. 用真实 `TMANOpPackage` 构建链做 x86 / v75 / android 编译
3. 用真实设备做 graph prepare / runtime load / inference smoke test
4. 在 smoke test 稳定后，继续把 `LinearA8` / `FinalizeA8` 的最后一段 pack 与查表路径进一步 intrinsic 化
5. 最后做 benchmark 与性能归因
