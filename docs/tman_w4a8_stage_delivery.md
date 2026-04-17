# W4A8 TMAN 阶段性交付文档

## 1. 文档目的

本文档用于说明本阶段在 ExecuTorch Qualcomm 路径上新增的正式 `W4A8` TMAN 实现，包括：

- 本阶段的目标与范围
- 最终落地的正式代码路径
- 关键设计选择
- 外部 `activation scale` 文件接入方式
- 已完成的验证
- 当前已知边界与后续建议

本文档对应的工作基础来源于：

- [op_gen/README.md](/home/lmx/Projects/tman/test/executorch/op_gen/README.md)
- `op_gen` 下完成过的算子级原型验证

## 2. 本阶段目标

本阶段目标不是直接做完整模型端到端性能优化，而是先把一条“**正式 runtime + builder 可接入的 W4A8 路径**”搭起来，并保证它具备可交付给他人继续联调测试的功能正确性。

本阶段落实的目标是：

- 在正式 Qualcomm runtime 路径中新增一套独立的 `W4A8` op package
- 不改动原有 `W4A16` 路径文件
- 在正式 builder 中新增一条独立的 `tman::linear_a8` lowering 路径
- 把外部 `activation_scales_a8_per-tensor_sym.pt` 的 `input_scale / output_scale` 接入正式路径
- 保持层间接口为 `FP16 in / FP16 out`
- 在本地 `x86` HTP 模拟器环境下完成算子级验证

## 3. 最终落地路径

### 3.1 正式 runtime op package

新增目录：

- [backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8)

该目录下的关键文件：

- [backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/src/ops/TMANPrecomputeA8.cpp](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/src/ops/TMANPrecomputeA8.cpp)
- [backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/src/ops/TMANLinearA8.cpp](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/src/ops/TMANLinearA8.cpp)
- [backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/src/ops/TMANFinalizeA8.cpp](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/src/ops/TMANFinalizeA8.cpp)
- [backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/include/hvx_funcs.h](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/include/hvx_funcs.h)
- [backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/src/TMANOpPackageInterface.cpp](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/src/TMANOpPackageInterface.cpp)
- [backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/config/TMANOpPackageHtp.xml](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/config/TMANOpPackageHtp.xml)
- [backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/Makefile](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/Makefile)

### 3.2 正式 builder / custom op 路径

关键文件：

- [backends/qualcomm/builders/custom_ops.py](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/builders/custom_ops.py)
- [backends/qualcomm/builders/op_tman_linear_a8.py](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/builders/op_tman_linear_a8.py)
- [backends/qualcomm/builders/qnn_constants.py](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/builders/qnn_constants.py)
- [backends/qualcomm/builders/__init__.py](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/builders/__init__.py)
- [backends/qualcomm/utils/utils.py](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/utils/utils.py)
- [backends/qualcomm/quantizer/custom_annotation.py](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/quantizer/custom_annotation.py)

### 3.3 Qwen3 接入入口

关键文件：

- [examples/qualcomm/oss_scripts/qwen3/qwen3.py](/home/lmx/Projects/tman/test/executorch/examples/qualcomm/oss_scripts/qwen3/qwen3.py)

新增命令行入口：

- `--use_tman_a8`
- `--tman_a8_scale_path`

## 4. 当前正式 W4A8 数值链路

当前正式链路为：

```text
FP16 input
  -> TMANPrecomputeA8(x, input_scale)
  -> TMANLinearA8(l, qweight, scales)
  -> TMANFinalizeA8(c, output_scale)
  -> FP16 output
```

### 4.1 TMANPrecomputeA8

职责：

- 接收 `fp16` 输入张量
- 接收外部 `input_scale`
- 用外部 `input_scale` 进行对称 `A8` 量化
- 再反量化回 `fp16`
- 复用旧 TMAN 的 `hvx_lut_ctor` 生成 packed buffer

当前 packed buffer 仍然保持旧 TMAN 兼容布局：

```text
[lut:int16][ls:float][lb:float]
```

这样做的原因：

- 保证与旧 `hvx_tbl` 兼容
- 尽量少改动底层查表与 bit-serial 主干
- 在功能正确性阶段优先降低风险

### 4.2 TMANLinearA8

职责：

- 接收 `PrecomputeA8` 产生的 packed buffer
- 接收旧 TMAN 兼容的 packed `qweight`
- 接收旧 TMAN 兼容的 packed `scales`
- 复用 `hvx_tbl` 产生 bit-plane `float scratch`

这一层**不处理 `output_scale`**。

原因：

- 保持旧 `TMANLinear` 的职责边界不变
- 避免破坏 `hvx_tbl` 的核心语义
- 让输出激活量化逻辑集中到最终输出阶段

### 4.3 TMANFinalizeA8

职责：

- 接收 `LinearA8` 产生的 `float scratch`
- 先复用旧 `bit-serial` 合成规则得到浮点输出
- 再按外部 `output_scale` 做一轮量化再反量化
- 最终输出 `fp16`

这是本阶段新增的“**轻量 output-dequant/finalize 变体**”。

它的意义是：

- `output_scale` 真正参与正式数值路径
- 层外仍然保持 `fp16` 接口
- 方便继续衔接后续 `fp16` 算子

## 5. 外部 activation scale 文件接入方式

当前使用的外部 scale 文件示例路径：

```text
/home/lmx/Projects/tman/models/Qwen3-1.7B-GPTQ-Int4/activation_scales_a8_per-tensor_sym.pt
```

本地已确认其结构是一个扁平 `dict[str, Tensor]`，典型 key 例如：

```text
model.model.layers.0.self_attn.q_proj.input_scale
model.model.layers.0.self_attn.q_proj.output_scale
model.model.layers.0.mlp.gate_proj.input_scale
...
```

模块注入流程：

1. `qwen3.py` 中开启 `--use_tman_a8`
2. 读取 `activation scale` 文件
3. 将 GPTQ 线性层替换成 `TMANLinearA8`
4. 调用 `apply_tman_a8_activation_scales(...)`
5. 按模块名匹配并把 `input_scale / output_scale` 写入模块 buffer
6. 进入 export / builder / runtime

当前策略是“**严格失败**”：

- 如果某个 `TMANLinearA8` 没有匹配到 `input_scale / output_scale`
- 或 scale 不是标量
- 会在导出阶段直接报错

这样做的目的是避免出现“代码能跑，但量化语义错误”的隐蔽问题。

## 6. 与 README 目标相比的当前状态

### 6.1 已经完成的部分

- 正式 runtime 路径存在独立 `W4A8` op package
- 正式 builder / custom op 路径已打通
- `input_scale / output_scale` 已接入正式数值路径
- 最终对外仍然是 `fp16 in / fp16 out`
- `x86` 模拟器和离线 `context` 编译均已通过

### 6.2 当前仍然偏保守的部分

当前 `PrecomputeA8` 不是 README 中更激进的“纯整数域 LUT 构造”版本，而是：

```text
fp16 -> 按 input_scale 量化到 a8 -> 反量化到 fp16 -> 复用旧 hvx_lut_ctor
```

这样做是本阶段有意选择的“正确性优先”折中：

- 已具备 W4A8 输入量化误差模型
- 复用旧 LUT/Linear/bit-serial 主干
- 风险更低，适合先交付联调

## 7. 已完成验证

### 7.1 Python 静态检查

已通过 `py_compile`：

- builder
- utils
- quantizer custom annotation
- `qwen3.py`

### 7.2 x86 HTP 模拟器算子级验证

使用：

- [op_gen/test/test_tman_linear_a8.cpp](/home/lmx/Projects/tman/test/executorch/op_gen/test/test_tman_linear_a8.cpp)
- [op_gen/test/Makefile.linear_a8](/home/lmx/Projects/tman/test/executorch/op_gen/test/Makefile.linear_a8)

默认连接的 runtime 包已经切到：

- [TMANOpPackageW4A8](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8)

通过的 case：

1. 基线 `K256 / M128`
2. `zero-input`
3. `zero-weight`
4. 多 group `K512 / M128 + 外部 input_scale`
5. 外部 `output_scale` 量化反量化
6. autosplit 大输出通道 `M256`

结果：

```text
Passed: 6
Failed: 0
```

### 7.3 离线 context 编译

已完成：

```text
make -f Makefile.linear_a8 compile_context
```

成功生成：

- `tman_linear_a8_ctx.bin`
- `tman_linear_a8_ctx.meta`

说明：

- runtime op package 的注册正常
- `QNN` graph prepare / finalize 正常
- 新增的 `TMANFinalizeA8` ABI 可以通过 `context` 编译链路

## 8. 当前支持范围

当前正式支持范围是：

- `W4`
- `group_size = 128`
- `symmetric = 1`
- `N = 1`
- `K % 256 == 0`

这是本阶段刻意收敛后的范围，目的是先确保路径稳定可测。

## 9. 交付给合作者时的重点说明

需要特别强调：

1. 本阶段已经具备“正式 runtime + builder + 外部 scale 接入 + x86 算子级验证”的完整链路
2. 还没有在完整 ExecuTorch 开发环境下跑整模型 `qwen3.py --compile_only`
3. 当前最适合的下一步工作是：
   - 让合作者在完整环境下验证 `compile_only`
   - 再做设备侧验证
   - 最后再考虑更激进的整数 LUT 优化

## 10. 后续建议

### 10.1 优先级最高

- 在完整 ExecuTorch / Qualcomm 环境下跑 `qwen3.py --compile_only --use_tman_a8`
- 验证 `QNN_OP_PACKAGE_PATHS` 是否正确指向 `TMANOpPackageW4A8`
- 验证 `activation_scales_a8_per-tensor_sym.pt` 与模型模块名匹配是否完整

### 10.2 之后再做

- 设备端 HTP 路径验证
- 将 `PrecomputeA8` 升级成 README 中更激进的“纯整数 LUT 构造”版本
- 若性能需要，再考虑把 `TMANFinalizeA8` 做成更高性能实现

