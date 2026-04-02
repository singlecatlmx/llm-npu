# Qwen3 TMAN Linear 代码流程与 W4A8 扩展实验方案

## 1. 目标

本文基于当前仓库里的 Qwen3 Qualcomm 示例和 TMAN 自定义算子实现，回答三件事：

1. `examples/qualcomm/oss_scripts/qwen3/qwen3.py` 走 TMAN 路径时，代码实际是怎样执行的。
2. 当前 `TMANPrecompute.cpp` / `TMANLinear.cpp` / `TMANFinalize.cpp` 这套查找表加速链路，真实支持到什么程度。
3. 如果要把当前 W4A16 路径扩展为 `W4A8`，并让激活采用 `per-channel int8` 量化、量化系数从外部 `pt` 文件导入，整体上是否可行，推荐怎样做实验分阶段推进。

本文只做代码分析和方案设计，不直接改算子实现。

## 2. 当前 Qwen3 + TMAN 的执行主链

### 2.1 模型与权重加载

入口在 `examples/qualcomm/oss_scripts/qwen3/qwen3.py` 的 `compile()`。

当前 TMAN 路径的大致流程是：

1. 从 `Qwen3Config.from_pretrained(args.model_dir)` 读取配置。
2. 用 `meta device` 构造一个或两个 `Qwen3ForCausalLM` 实例。
   - `kv` 模式只构造一个图。
   - `hybrid` 模式会构造 `kv` 和 `prefill` 两个图。
3. 调用 `GPTQModel.from_quantized(args.model_dir)` 加载 GPTQ 模型。
4. 从 `qmodel.model.state_dict()` 取出量化权重状态。
5. 通过 `convert_linear_to_qlinear()`，把 `Qwen3Attention` 和 `Qwen3MLP` 里的 `nn.Linear` 全部替换成 GPTQ 的 `TorchQuantLinear`。
6. 再调用 `load_state_dict(assign=True)`，把 GPTQ 模型里的 `qweight/scales/qzeros/g_idx/...` 装载到这些 `TorchQuantLinear` 模块里。

这里的关键点是：

- 线性层的“权重导入”目前完全沿用了 GPTQModel 的状态字典机制。
- 这意味着“从外部 `pt` 文件给 TMAN 线性层导入额外参数”在工程模式上是可复用的，不需要重新发明一套加载框架。

### 2.2 Qwen3 里哪些线性层会走 TMAN

模型定义在 `examples/qualcomm/oss_scripts/qwen3/model/static_qwen3.py`。

Qwen3 的线性层主要来自：

- Attention:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
- MLP:
  - `gate_proj`
  - `up_proj`
  - `down_proj`
- 最后输出头：
  - `lm_head`

当前 `qwen3.py` 只对 `layer.self_attn` 和 `layer.mlp` 做 `convert_linear_to_qlinear()`，因此 TMAN 的主要覆盖面是注意力和 MLP 内部线性层。

### 2.3 从 GPTQ QLinear 切到 TMANLinear

当 `args.use_tman=True` 时，`qwen3.py` 会执行：

- `convert_qlinear_to_tman_linear(layer.self_attn)`
- `layer.self_attn.prepare_tman()`
- `convert_qlinear_to_tman_linear(layer.mlp)`

这一步发生在 `backends/qualcomm/utils/utils.py`：

- `convert_qlinear_to_tman_linear()` 会把每个 `TorchQuantLinear` 替换成 `TMANLinear` 模块。
- `TMANLinear` 本质上保存了 GPTQ 的：
  - `qweight`
  - `scales`
  - `qzeros`
  - `g_idx`
  - `wf_unsqueeze_zero`
  - `wf_unsqueeze_neg_one`
  - `bits`
  - `group_size`
  - `symmetric`
  - `gptq_v2`

也就是说，当前 TMAN 路径并没有重新定义权重格式，而是复用了 GPTQ 权重表示，然后在 lowering 阶段重新打包成 HVX/HTP 友好的布局。

### 2.4 `prepare_tman()` 做了什么

`Qwen3Attention.prepare_tman()` 只做了一件事：

- 把 `forward` 切换成 `forward_tman`

`forward_tman()` 和普通 attention 的差别是：

- 先通过 `q_proj/k_proj/v_proj` 做投影。
- 再 reshape 成 head 维度。
- 再拆分成单头 list。
- 后续每个 head 独立做 matmul/softmax/value 聚合。

这样做的意义是：

- 让 attention 的图结构更适合 Qualcomm backend 现有的量化标注和 lowering 规则。
- 也让 `tman::linear` 更容易只承担“线性投影”这一块，而不是和后处理逻辑纠缠。

### 2.5 `convert_linear_to_conv2d()` 和 `SingleLlama` 包装

在 TMAN 或 SHA 路径切换完成后，`qwen3.py` 还会做两步：

1. `convert_linear_to_conv2d()`
2. 用 `SingleLlama(...)` 包装模型

这里需要注意：

- `convert_linear_to_conv2d()` 只会替换还保留为 `nn.Linear` 的模块
- 已经替换成 `TMANLinear` 的层不会被改成 `Conv2d`

因此实际结果是：

- TMAN 覆盖的 attention / MLP 线性层继续保留为自定义 op 路径
- 其他仍是普通 `nn.Linear` 的层，例如 `lm_head`，会转成 `Conv2d` 走 QNN 原生路径
- `SingleLlama` 负责保存 example inputs、量化配置、pass 配置以及后续 lowering 入口

### 2.6 PTQ 与量化图生成

在 `args.ptq` 非空时，`SingleLlama.quantize()` 会执行：

1. `torch.export.export()` 导出当前图。
2. `make_quantizer()` 构造 QNN quantizer。
3. `prepare_pt2e()` 插入 observer。
4. 用 `calibrate()` 跑 prompt，收集激活统计量。
5. `convert_pt2e()` 生成量化图。

和 TMAN 强相关的两点：

1. `backends/qualcomm/quantizer/annotators.py` 里已经给 `torch.ops.tman.linear.default` 注册了 annotator。
   - 但它只把 `tman::linear` 当成“单输入单输出”的量化节点。
   - `qweight/scales` 这些静态参数不会被当作普通量化输入处理。
2. `backends/qualcomm/quantizer/custom_annotation.py` 里有明确注释：
   - `tman::linear currently does not support 8a`

这说明现在的图侧设计默认 TMAN linear 不是 W4A8 完成态。

另外还有一个和本文目标直接相关的细节：

- 当前 Qualcomm 量化栈的 `8a8w` 默认 activation dtype 是 `torch.uint8`
- 但本文目标是线性层输入输出都为 `int8`

这意味着后续做 W4A8 时，必须显式决定：

- 是继续复用现有 QNN 的 `uint8` 生态
- 还是单独为 TMAN W4A8 开一条 `signed int8` 路径

本文后续默认推荐第二种，也就是为 TMAN W4A8 单独走 `int8` 设计。

### 2.7 Lowering 到 QNN / HTP

`TMANLinear` 在图里出现的是 Python custom op `tman::linear`，但 lowering 到 QNN 时并不是一个单节点。

真正的 lowering 发生在 `backends/qualcomm/builders/op_tman_linear.py`，它会把一个 `tman::linear` 节点拆成：

1. `qti.aisw::Convert`，把输入量化张量转成 `fp16`
2. `TMANOpPackage::TMANPrecompute`
3. `TMANOpPackage::TMANLinear`
4. `TMANOpPackage::TMANFinalize`
5. `qti.aisw::Convert`，把输出从 `fp16` 转回图里的量化张量

这里最关键的实现细节有三个：

1. `self.add_convert = True`
   - 当前 builder 默认总是给 TMAN 前后插 `Convert`
   - 这说明当前自定义算子实际工作在 `fp16` 激活域，不是 `int8` 激活域
2. `unpack_gptqv2()` + `hvx_preprocess_weights()`
   - builder 会在编译时把 GPTQ 权重重新排布成 HVX 查找表友好的 packed layout
3. 中间 scratch buffer `l/c`
   - `l` 是 Precompute 的输出缓存
   - `c` 是 Linear 的输出缓存

### 2.8 OpPackage 运行时执行

运行时 C++ 实现位于：

- `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANPrecompute.cpp`
- `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANLinear.cpp`
- `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANFinalize.cpp`
- `backends/qualcomm/runtime/op_packages/TMANOpPackage/include/hvx_funcs.h`

执行链含义如下：

#### TMANPrecompute

- 输入：`fp16 x`
- 输出：scratch `l`
- 核心函数：`hvx_lut_ctor<LType=int16_t, XType=__fp16>()`

它会：

- 以 `g=4` 为一组，把输入激活构造成 16 项 LUT
- 生成：
  - `l`: `int16 LUT`
  - `ls`: LUT scale，当前是 `float`
  - `lb`: LUT bias，当前是 `float` 或 `fp16`

#### TMANLinear

- 输入：
  - `l`
  - 预处理后的 `qweight`
  - 预处理后的 `scales`
- 输出：scratch `c`
- 核心函数：`hvx_tbl<..., CType=float>()`

它会：

- 从 `l` 中查表得到 bit-plane partial sums
- 把 partial sum 乘以 `ls/lb` 以及权重量化尺度
- 输出到 `c`

注意：当前 `c` 逻辑上是 `float` 聚合缓存，不是整数累加缓存。

#### TMANFinalize

- 输入：`c`
- 输出：`fp16 y`
- 核心函数：`hvx_bit_serial<XType=__fp16, CType=float>()`

它会：

- 把 bit-plane partial sums 做 bit-serial 汇总
- 最后输出 `fp16`

## 3. 当前实现的真实能力边界

### 3.1 当前 TMAN 更接近 W4A16，不是 W4A8

从 builder 和 C++ 内核一起看，当前 TMAN 流程实际上是：

`quantized graph tensor -> Convert -> fp16 x -> int16 LUT -> float scratch -> fp16 y -> Convert -> quantized graph tensor`

因此它更准确地说是：

- 权重侧：4bit GPTQ 权重
- 激活侧：OpPackage 内部使用 `fp16`
- 图侧：可以挂在量化图里，但算子本体并不是 int8 原生执行
- 同时，现有通用 `8a8w` 量化配置默认还是 `uint8 activation`，并非本文目标中的 `signed int8`

### 3.2 当前没有真正启用“自定义 op 的 per-channel 激活量化”

虽然 Qualcomm builder 层本身支持 `AXIS_SCALE_OFFSET` 编码，但当前 TMAN 路径仍有三个缺口：

1. `QnnQuantizer` 里的 `get_ptq_per_channel_quant_config()` 只给 weight 用 `per_channel_symmetric`
   - activation 仍然是 `per_tensor`
2. `TMANOpPackageInterface.cpp` 里没有启用：
   - `LIST_PACKAGE_PER_CHANNEL_QUANTIZED_OPS(...)`
3. 当前 OpPackage C++ 实现签名也没有显式接收 per-channel activation scales

这意味着：

- “QNN 框架层支持 per-channel 编码”不等于“当前 TMAN 自定义算子已经接好了 per-channel activation”。

### 3.3 当前 builder 的 `add_convert=False` 并未形成完整路径

`op_tman_linear.py` 里 `_get_l_size()` 已经有 `need_dequant` 参数，看起来像预留过“不要 Convert、直接吃量化输入”的思路。

但当前事实是：

- `self.add_convert` 固定为 `True`
- `TMANPrecompute.cpp` 的输入类型写死为 `__fp16`
- `hvx_lut_ctor` 也是 `XType=__fp16`

因此当前仓库里并没有一条真正跑通的“TMAN 直接接 int8 输入”的现成链路。

### 3.4 当前 HTP XML 与现实现状不完全一致

`backends/qualcomm/runtime/op_packages/TMANOpPackage/config/TMANOpPackageHtp.xml` 目前只描述了 `TMANLinear`，而且数据类型描述偏向 `UFIXED_POINT_16`。

但现在的实际实现是：

- Python builder 拆成 `Precompute -> Linear -> Finalize`
- OpPackage interface 里也注册了这三个 op

这说明：

- XML 描述层和实际实现层存在偏差
- 如果后续认真做 W4A8，最好顺手把 OpDef 描述也一起收敛

## 4. 对 “W4A8 + per-channel activation + 外部 pt 系数导入” 的可行性判断

## 4.1 可行，但建议拆成两层目标

这个需求总体可行，但最好拆成两个层次：

### 第一层：工程可落地版本

目标：

- 整个线性层图接口输入输出都是 `int8`
- 权重保持 `W4 GPTQ`
- 激活量化系数从外部 `pt` 文件导入
- 中间 LUT 可继续用 `int16`
- 中间累加建议先用 `int32`

这个版本优先解决“接口与系统打通”。

这里要单独强调一次：

- 这一步不是简单复用现有 `8a8w`
- 因为现有 `8a8w` 默认是 `uint8 activation`
- 而本方案目标是 `signed int8 activation`

### 第二层：高性能整数化版本

目标：

- 尽量去掉 OpPackage 内部的 `fp16/float` 路径
- 把 `TMANLinear` 主干改成更纯粹的 `int16/int32` 查表与累加
- `Finalize` 中完成 requant 到 `int8`

这个版本才是严格意义上的 “native W4A8”。

## 4.2 权重侧是可复用的

权重侧基本不构成主障碍，因为现有 TMAN 已经完成了：

- 从 GPTQ state_dict 读取 `qweight/scales/qzeros`
- 在 builder 编译阶段把权重打包成 HVX 友好格式

对 W4A8 来说，权重仍然是 `W4`，所以：

- `unpack_gptqv2()`
- `hvx_preprocess_weights()`

这部分总体可延用，主要变化会落在：

- 激活量化参数导入
- `Precompute`
- `Finalize`
- 量化图标注逻辑

## 4.3 外部 `pt` 导入激活量化系数是可行的

从工程模式看，这件事是很可行的，因为当前权重导入本身已经走的是：

- Python 模块替换
- `load_state_dict()`
- buffer/parameter 注册

因此新增一份外部激活量化文件可以直接复用这一路径。

推荐外部文件格式如下：

```python
{
    "meta": {
        "dtype": "int8",
        "scheme": "per_channel_symmetric",
        "axis": "last",
    },
    "model.layers.0.self_attn.q_proj.input_scale":  torch.Tensor[hidden_size],
    "model.layers.0.self_attn.q_proj.output_scale": torch.Tensor[out_features],
    "model.layers.0.self_attn.k_proj.input_scale":  torch.Tensor[hidden_size],
    "model.layers.0.self_attn.k_proj.output_scale": torch.Tensor[out_features],
    ...
}
```

如果坚持做非对称量化，也可以额外带：

```python
"...input_zero_point":  torch.Tensor[hidden_size]
"...output_zero_point": torch.Tensor[out_features]
```

但从当前代码和 HTP 限制看，首版强烈建议只做：

- `per-channel symmetric int8`
- `zero_point = 0`

理由是：

1. 当前 Qualcomm per-channel 路径天然更偏向 scale-only。
2. 对 TMAN LUT 来说，非对称 activation 会显著增加偏置处理复杂度。
3. 对 int8 I/O 来说，symmetric 更容易把 `Precompute` 和 `Finalize` 保持成纯整数主路径。

## 4.4 纯靠 QNN 的 per-channel 量化编码去做，有风险

理论上可以尝试让 TMAN custom op 直接接 `QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET`。

但这条路径当前有几个风险点：

1. `TMANOpPackageInterface.cpp` 还没有启用 `LIST_PACKAGE_PER_CHANNEL_QUANTIZED_OPS(...)`
2. HTP 对 per-channel 支持有约束，注释里明确写了“沿最后一维，offset 总为 0”
3. 当前 OpPackage C++ 实现并没有显式处理额外的 per-channel scale 输入
4. 仓库里没有现成的自定义 op per-channel activation 例子可以直接照搬

因此更稳妥的首版方案不是“完全依赖 QNN 的 per-channel 编码传参”，而是：

- 先把 activation scales 当作显式静态张量输入到自定义 op

这样做的好处是：

- 参数来源清晰
- Python/FX/builder/OpPackage 的控制面都在自己手里
- 不必第一阶段就吃透 HTP custom op 的 per-channel ABI 细节

## 4.5 “输入输出都是 int8，中间 int16/int32” 是可行的，但要重写一部分算子数学

这件事从算法上是可行的。

目标数学形式是：

```text
y_q = requant(
    sum_j (x_q[j] * s_x[j]) * ((w_q[i, j] - z_w[i, j]) * s_w[i, j])
)
```

如果采用对称 int8 激活，则可以简化为：

```text
y_q = requant(
    sum_j x_q[j] * w_eff[i, j]
)
```

其中：

- `x_q` 是 int8
- LUT 项可以放到 int16
- 主累加可以放到 int32
- 最终 requant 到 int8

但从当前代码现状看，要做到这一点，必须至少改动：

1. `TMANPrecompute`
   - 输入不能再是假定 `__fp16`
   - 要能处理 `int8 + per-channel input_scale`
2. `TMANLinear`
   - 当前 `hvx_tbl` 主路径用的是 `float` 聚合
   - 如果要做整数主路径，需要改成 `int32` 或 fixed-point 累加
3. `TMANFinalize`
   - 当前是 `float -> fp16`
   - 新路径需要做 `int32 -> int8 requant`
4. `op_tman_linear.py`
   - 不能再默认插入前后 `Convert`

结论是：

- “系统上可行”
- “不是小改”
- “建议先做功能打通版，再做高性能整数版”

## 5. 推荐实现方向

## 5.1 推荐先做显式系数输入版，不先碰 QNN per-channel ABI

推荐新增一条专门的 TMAN W4A8 路径，例如：

- Python custom op:
  - `tman::linear_w4a8`
- Python module:
  - `TMANLinearW4A8`
- Builder visitor:
  - 在 `op_tman_linear.py` 里新增 W4A8 分支

让自定义 op 显式接收：

- `x_q`，int8
- `qweight`
- `w_scales`
- `qzeros`
- `input_scale`
- `output_scale`
- `input_zero_point`，可选
- `output_zero_point`，可选
- `group_size`
- `bits`
- `symmetric`

这样做后，QNN 只负责搬运显式静态参数，不必第一阶段依赖 custom op 的 per-channel quant ABI。

## 5.2 推荐两阶段内核方案

### 方案 A：功能优先版

思路：

- 图接口使用 `int8`
- `Precompute` 里读取 `int8 x` 和 `input_scale`
- 在 Precompute 内部先恢复到适合 LUT 构造的中间值
- LUT 仍然生成 `int16`
- `Linear` 的中间聚合先保留一部分浮点或 mixed path
- `Finalize` 负责输出 `int8`

优点：

- 改动小
- 容易复用当前 `hvx_lut_ctor/hvx_tbl` 代码结构

缺点：

- 不能算完全整数化
- 性能未必是最终最优

### 方案 B：整数主路径版

思路：

- `Precompute` 直接从 `int8` 激活构造整数 LUT
- `Linear` 保持 `int16/int32` 查表与聚合
- `Finalize` 做 bit-serial 汇总 + requant 到 `int8`

优点：

- 更接近真正 W4A8
- 更符合“中间只用 int16/int32”的目标

缺点：

- 需要较大幅度重写 `hvx_tbl` 与 `hvx_bit_serial`
- 数值范围、scale 融合、舍入策略都要重新验证

建议顺序：

1. 先做方案 A，打通系统
2. 再做方案 B，追性能

## 5.3 推荐激活量化策略

首版推荐：

- 输入激活：`per-channel symmetric int8`
- 输出激活：`per-channel symmetric int8`
- channel axis：最后一维

理由：

1. 更符合线性层 `[..., K] -> [..., M]` 的形状习惯
2. 更利于在 `Precompute` 按输入通道读取 `input_scale[K]`
3. 更利于在 `Finalize` 按输出通道读取 `output_scale[M]`
4. 与 HTP 对 per-channel scale-only 的约束更一致

不推荐首版直接做：

- activation per-channel affine
- 非零 zero-point

## 6. 建议修改点清单

首版实验建议关注以下文件。

### Python 图与模块层

- `examples/qualcomm/oss_scripts/qwen3/qwen3.py`
  - 增加外部 activation quant `pt` 文件参数
  - 在 `use_tman` 分支注入 W4A8 模块构造与参数加载
- `backends/qualcomm/utils/utils.py`
  - 新增 `TMANLinearW4A8`
  - 新增 `convert_qlinear_to_tman_linear_w4a8()`
- `backends/qualcomm/builders/custom_ops.py`
  - 新增 Python 参考实现 `tman::linear_w4a8`
  - 用于导出、fake tensor、CPU 对照验证

### 量化与标注层

- `backends/qualcomm/quantizer/annotators.py`
  - 给 `tman::linear_w4a8` 注册 annotator
- `backends/qualcomm/quantizer/custom_annotation.py`
  - 去掉当前“碰到 tman linear 就停止 8a 传播”的限制，或者新增 W4A8 专用分支
- `backends/qualcomm/quantizer/qconfig.py`
  - 如果要直接走 PT2E 的 per-channel activation，需要新增 activation per-channel quant config

### Builder / Lowering 层

- `backends/qualcomm/builders/op_tman_linear.py`
  - 新增 W4A8 lowering 分支
  - 禁止在 W4A8 路径里插前后 `Convert`
  - 把 activation scale/output scale 当作静态 tensor 输入下发

### HTP OpPackage 层

- `backends/qualcomm/runtime/op_packages/TMANOpPackage/include/hvx_funcs.h`
  - 新增 int8 输入、int32 聚合、int8 输出路径
- `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANPrecompute.cpp`
  - 新增或扩展为 int8 激活入口
- `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANLinear.cpp`
  - 新增 int32 聚合路径
- `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANFinalize.cpp`
  - 新增 int8 requant 输出
- `backends/qualcomm/runtime/op_packages/TMANOpPackage/src/TMANOpPackageInterface.cpp`
  - 如果后续真要用 QNN per-channel ABI，再启用 per-channel op registration
- `backends/qualcomm/runtime/op_packages/TMANOpPackage/config/TMANOpPackageHtp.xml`
  - 收敛 OpDef 描述

## 7. 分阶段实验方案

## Phase 0：建立基线

目标：

- 明确当前 W4A16 TMAN 的正确性和性能基线

建议动作：

1. 固定模型为 Qwen3 GPTQ Int4。
2. 固定模式：
   - `kv`
   - `prefill_ar_len`
   - `max_seq_len`
3. 记录：
   - 编译成功率
   - 首 token 延迟
   - decode tokens/s
   - 峰值 scratch size
   - 输出质量基线

产出：

- 一份 baseline 表格，后续 W4A8 都和它比

## Phase 1：先做纯 PyTorch 参考实现

目标：

- 不碰 QNN/HTP，先验证 W4A8 数学路径是通的

建议动作：

1. 在 `backends/qualcomm/builders/custom_ops.py` 增加 `tman::linear_w4a8` 的 Python 参考实现。
2. 输入输出都使用 `int8` tensor。
3. 从外部 `pt` 文件读取：
   - `input_scale[K]`
   - `output_scale[M]`
4. 先用最直观的参考计算：

```text
int8 x
-> dequant with input_scale
-> matmul with dequantized W4 weight
-> requant with output_scale
-> int8 y
```

验证指标：

- 与浮点参考 `torch.matmul(dequant(x), dequant(w))` 的误差
- 与当前 `TMANLinear` 的 `fp16` 输出相比的误差

通过标准：

- 模块级误差可控
- 不依赖 QNN 即可证明算法成立

## Phase 2：接通外部 activation `pt` 文件导入

目标：

- 把“激活量化参数来自外部文件”这件事独立打通

建议动作：

1. 给 `qwen3.py` 增加例如：
   - `--tman_act_quant_pt`
2. 新增一份 loader：
   - 按模块全名查找对应 scale
3. 在 `TMANLinearW4A8` 中注册 buffer：
   - `input_scale`
   - `output_scale`
   - `input_zero_point`，可选
   - `output_zero_point`，可选

验证指标：

- 每个被替换的线性层都能对齐加载到对应 activation scale
- 缺参数时能清晰报错

建议首版映射键名直接复用模型模块路径，例如：

- `model.layers.0.self_attn.q_proj.input_scale`
- `model.layers.0.self_attn.q_proj.output_scale`

## Phase 3：把图接口改成真正的 int8 I/O

目标：

- 让 `tman::linear_w4a8` 在图中输入输出都是 int8

建议动作：

1. 在 quantizer/annotator 层给 W4A8 路径单独加标注。
2. 暂时不要强依赖 PT2E 自动生成 per-channel activation observer。
3. 把 activation quant 参数视为“外部已知静态参数”，图中只传递 int8 tensor。

这里的核心思路是：

- 外部 `pt` 文件决定 quant 参数
- 图里不一定非要重新学习/观察这些参数

这样可以降低首版系统复杂度。

## Phase 4：Builder 与 OpPackage 首版打通

目标：

- 在 QNN/HTP 上真正运行 `int8 -> TMAN -> int8`

建议动作：

1. 在 `op_tman_linear.py` 给 W4A8 分支关闭 `add_convert`。
2. 新增静态输入：
   - `input_scale`
   - `output_scale`
3. 保持 scratch 设计为：
   - `l`: `int16 LUT + 辅助 scale/bias`
   - `c`: `int32` 或 `float32`
4. C++ 先做“功能优先版”。

建议首版实现顺序：

1. `Precompute(int8 x, input_scale) -> l`
2. `Linear(l, packed_w, w_scale) -> c`
3. `Finalize(c, output_scale) -> int8 y`

通过标准：

- x86 builder 能过
- HTP context 能生成
- 小图能运行

## Phase 5：整数主路径优化

目标：

- 把当前 `float scratch` 逐步替换为 `int32 scratch`

建议动作：

1. 重新审视 `hvx_tbl` 当前浮点乘法位置：
   - `* ls`
   - `+ lb`
   - `* s`
2. 尝试把这些 scale 合并成 fixed-point multiplier + shift
3. 在 `Finalize` 中统一做 requant

更具体地说，需要尝试把：

```text
c_float = ((partial_sum * ls) + lb) * w_scale
```

重写成：

```text
c_int32 = partial_sum * alpha_int
y_int8  = requant(c_int32, output_scale)
```

这一步是性能关键，也是风险最大的阶段。

## Phase 6：端到端 Qwen3 回归

目标：

- 证明方案不仅能跑单层，也能跑完整 Qwen3

验证建议：

1. 层级验证
   - 只替换 `q_proj`
   - 再扩到 `q/k/v/o`
   - 再扩到 MLP
2. 图级验证
   - `kv` 图
   - `hybrid` 图
3. 模型级验证
   - logits cosine
   - top-k 一致率
   - 生成文本差异
   - perplexity
4. 性能验证
   - prefill latency
   - decode latency
   - tokens/s
   - VTCM / scratch buffer 使用量

## 8. 关键技术风险

## 风险 1：activation per-channel 现在没有现成 PT2E 配置

现有 `get_ptq_per_channel_quant_config()` 只给 weight 做 per-channel。

影响：

- 如果强行走“完全自动的 per-channel activation PTQ”，前期改动面会扩大

建议：

- 首版把 activation quant 参数视为外部静态输入，不先依赖 PT2E 自动生成

## 风险 2：custom op 的 QNN per-channel ABI 不确定

影响：

- 直接用 `AXIS_SCALE_OFFSET` 可能要改 OpPackage 注册和实现签名

建议：

- 首版显式增加 scale tensor 输入

## 风险 3：当前 `hvx_tbl` 深度依赖浮点路径

影响：

- 真正整数化不是小修

建议：

- 先功能版，再整数版

## 风险 4：attention 里 input/output channel 轴要严格对齐

尤其是：

- `q_proj/k_proj/v_proj/o_proj`
- `reshape`
- `split_with_sizes`
- `permute`

如果 activation scale 的 axis 和实际 lowering 后的最后一维不一致，就会出现隐性数值错误。

建议：

- 外部 `pt` 文件统一按“线性层逻辑输入维/输出维”存储
- builder 中在 axis reorder 后再做最终映射

## 风险 5：现有注释里明确写了 `tman::linear currently does not support 8a`

这不是一句注释而已，而是当前图侧作者已经确认过的一条能力边界。

建议：

- 把它当成真正的设计约束，而不是“小问题”

## 9. 建议的最小可行里程碑

如果目标是尽快看到结果，建议按下面的最小里程碑推进：

### Milestone 1

- 新增外部 `pt` 激活参数导入
- 新增 Python 参考版 `tman::linear_w4a8`
- 单层数值对齐

### Milestone 2

- QNN builder 接通 W4A8 分支
- 关闭 TMAN 前后 `Convert`
- HTP 上跑通单层或小图

### Milestone 3

- Qwen3 attention 全部投影层接入 W4A8
- 输出保持稳定

### Milestone 4

- MLP 也接入 W4A8
- 端到端评估性能与精度

### Milestone 5

- 把中间主路径进一步整数化
- 追求最终性能版本

## 10. 结论

结论可以概括为四句话：

1. 当前 TMAN 路径本质上是一个 `W4 + fp16 activation internal path`，不是完整的 `W4A8 int8-native`。
2. 通过外部 `pt` 文件导入激活量化系数是可行的，工程模式上可以直接复用 GPTQ 权重导入思路。
3. 如果要做 `int8` 输入输出、`int16/int32` 中间查表计算，核心工作量主要在 `Precompute/Linear/Finalize` 三段内核和 builder 去 `Convert`。
4. 推荐先做“显式 scale tensor 输入”的功能打通版，再决定是否进一步切到 QNN 原生 per-channel ABI 和更彻底的整数主路径。
