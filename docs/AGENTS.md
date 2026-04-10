# AGENTS

本文档用于给其它对话窗口或后续接手开发的代理，提供当前项目最关键的上下文、架构、实现状态、验证口径与后续任务。

## 1. 项目目标

当前项目基于 ExecuTorch + Qualcomm QNN backend + 自定义 `TMANOpPackage`，原本已经具备：

- GPTQ `W4` 权重
- TMAN `W4A16` linear 路径
- Hexagon / HVX 侧 LUT 查表推理能力

当前正在推进的目标是：

- 保留原有 `W4A16` 路径不变
- 新增一条 `W4A8` 扩展方案
- 只对 Qwen3 decoder 内部的 linear 做 A8 激活量化
- 非 linear 算子仍保持原有 FP16 路径
- 逐步把当前 A8 baseline 向更接近 HVX/Hexagon 风格的实现推进

## 2. 当前任务范围

当前聚焦范围严格限定在：

- 模型：`Qwen3-1.7B`
- 权重：GPTQ int4
- `group_size = 128`
- `sym = true`
- decoder 内部 7 类 linear：
  - `self_attn.q_proj`
  - `self_attn.k_proj`
  - `self_attn.v_proj`
  - `self_attn.o_proj`
  - `mlp.gate_proj`
  - `mlp.up_proj`
  - `mlp.down_proj`

不在当前范围内：

- `lm_head`
- embedding
- 全图 int8 化
- 非 Qwen3 模型泛化

## 3. 关键文件与路径

## 3.1 任务描述与文档

- 任务描述：
  - [description.md](/home/lmx/Projects/tman/test/executorch/docs/description.md)
- 需求分析：
  - [qwen3_tman_w4a8_requirements_analysis.md](/home/lmx/Projects/tman/test/executorch/docs/qwen3_tman_w4a8_requirements_analysis.md)
- 架构设计：
  - [qwen3_tman_w4a8_architecture_design.md](/home/lmx/Projects/tman/test/executorch/docs/qwen3_tman_w4a8_architecture_design.md)
- 交付说明：
  - [qwen3_tman_w4a8_delivery_handoff.md](/home/lmx/Projects/tman/test/executorch/docs/qwen3_tman_w4a8_delivery_handoff.md)

## 3.2 入口与核心实现

- 编译入口：
  - [qwen3.py](/home/lmx/Projects/tman/test/executorch/examples/qualcomm/oss_scripts/qwen3/qwen3.py)
- 外部 scale / reference / GPTQ helper：
  - [tman_w4a8.py](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/utils/tman_w4a8.py)
- Python 模块扩展：
  - [utils.py](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/utils/utils.py)
- custom op：
  - [custom_ops.py](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/builders/custom_ops.py)
- A8 builder：
  - [op_tman_linear_a8.py](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/builders/op_tman_linear_a8.py)

## 3.3 Runtime / OpPackage

- helper kernel：
  - [hvx_funcs.h](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackage/include/hvx_funcs.h)
- A8 三段式 op：
  - [TMANPrecomputeA8.cpp](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANPrecomputeA8.cpp)
  - [TMANLinearA8.cpp](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANLinearA8.cpp)
  - [TMANFinalizeA8.cpp](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANFinalizeA8.cpp)
- op 注册：
  - [TMANOpPackageInterface.cpp](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackage/src/TMANOpPackageInterface.cpp)
- op schema：
  - [TMANOpPackageHtp.xml](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackage/config/TMANOpPackageHtp.xml)

## 3.4 验证脚本

- unittest：
  - [test_tman_w4a8.py](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/tests/test_tman_w4a8.py)
- 轻量 reference 检查：
  - [tman_w4a8_reference_check.py](/home/lmx/Projects/tman/test/executorch/tools/tman_w4a8_reference_check.py)
- scale 覆盖审计：
  - [tman_w4a8_qwen3_scale_audit.py](/home/lmx/Projects/tman/test/executorch/tools/tman_w4a8_qwen3_scale_audit.py)
- 单层真实 probe：
  - [tman_w4a8_real_layer_probe.py](/home/lmx/Projects/tman/test/executorch/tools/tman_w4a8_real_layer_probe.py)
- 批量切片审计：
  - [tman_w4a8_model_slice_audit.py](/home/lmx/Projects/tman/test/executorch/tools/tman_w4a8_model_slice_audit.py)

## 4. 当前技术栈

- Python
- PyTorch
- ExecuTorch
- Qualcomm QNN backend
- 自定义 QNN OpPackage
- Hexagon / HVX
- GPTQ 量化权重
- `safetensors`

当前机器上**没有完整 ExecuTorch 运行环境**，但可以用 `gptq` conda 环境做轻量验证。

推荐可用环境：

- `conda activate gptq`

## 5. 当前权重与数据源

## 5.1 当前主要验证使用的权重文件

- `/home/lmx/Projects/tman/models/Qwen3-1.7B-GPTQ-Int4/model.safetensors`

当前已确认：

- `gptq_v2 = False`
- `bits = 4`
- `group_size = 128`
- `sym = true`

代码已兼容 GPTQ `v1/v2` 两种 qzeros 表示。

## 5.2 当前主要使用的 activation scale 文件

- `/home/lmx/Test/GPTQ_Opus/Qwen3-1.7B_w4a8_gptq/activation_scales_a8_per-tensor_sym.pt`

当前已确认：

- 覆盖 `28 * 7 = 196` 个目标 linear
- 每个 linear 同时包含：
  - `input_scale`
  - `output_scale`

## 6. 当前架构状态

## 6.1 已落地的 W4A8 架构

当前已经不是纯文档设计阶段，而是有一条可运行、可验证的 `W4A8` 支线：

- `qwen3.py` 已支持：
  - `--tman_act_mode {a16,a8}`
  - `--activation_scales_pt`
- `TMANLinearA8` 已实现
- `tman::linear_a8` 已实现
- A8 builder 已实现
- A8 runtime 三段式 op 已实现

## 6.2 当前 reference 分层

当前维护三套语义层次：

1. `high-level reference`
   - `dequantize_gptq_weight + matmul`
2. `grouped reference`
   - 当前项目目标实现的 group-wise 数学语义
3. `runtime baseline`
   - 当前 LUT/index/block 风格实现

关键结论：

- `grouped reference == runtime baseline`
- 少数模块与 `high-level reference` 有 `1 个 INT8 bin` 差异

因此当前更应把：

- `grouped reference`
- `runtime baseline`

视为贴近项目目标的正确性基线。

## 6.3 当前 runtime 数据组织

当前 `W4A8` runtime static qweight 已经收敛为：

- `[M, G, Qg, bits]`

其中：

- `M`：输出通道
- `G = K / group_size`
- `Qg = group_size / 4`
- `bits = 4`

当前 LUT scratch 语义为：

- `[row, group, qgroup_in_group, 16]`

每 `4` 个 qgroup 组成一个 block：

- `4 * 16 * int16 = 128B`

正好对应一条 HVX vector line。

当前需要特别注意的一点是：

- 逻辑语义上，A8 的 LUT scratch / qweight static 已经按 block 方式组织；
- 但为了降低 QNN 真编译阶段的 rank/schema 风险，当前对外 ABI 已重新压回 4D packed tensor。

当前对外 ABI 形态为：

- LUT scratch：
  - `[1, row, group, qgroup_vector_blocks * (4 * 16)]`
- qweight static：
  - `[1, M, G, qgroup_vector_blocks * (4 * bits)]`

也就是说：

- helper kernel 内部仍然按 block / vector-line 语义消费数据；
- builder / runtime 接口层则继续使用 4D packed 物理形态，便于交给 QNN/OpPackage 真编译阶段验证。

## 6.4 当前 helper kernel 收敛状态

`hvx_funcs.h` 中已经完成第一轮“HVX 风格化”收敛：

- `TmanA8Layout`
- `TMAN_A8_QGROUPS_PER_VECTOR = 4`
- `tman_a8_lut_block_offset`
- `tman_a8_build_lut_block_scalar`
- `tman_a8_group_dot_block_scalar`
- `tman_a8_precompute_lut_scalar`
- `tman_a8_linear_scalar`
- `tman_a8_finalize_scalar`

这说明后续真正替换成 HVX intrinsic 时，主要修改点已经收敛到 helper kernel 层。

在此基础上，当前又完成了第二轮“tile / pack-store 风格”收敛：

- `LinearA8` 已引入显式 tile helper：
  - `TmanA8IntTile`
  - `TmanA8FloatTile`
  - `tman_a8_accumulate_group_tile_scalar`
  - `tman_a8_apply_scales_to_group_tile_scalar`
  - `tman_a8_store_accum_tile_scalar`
- `FinalizeA8` 已引入显式输出 block helper：
  - `TmanA8OutputBlock`
  - `tman_a8_round_clamp_to_int8_scalar`
  - `tman_a8_pack_output_block_scalar`
  - `tman_a8_store_output_block`

这说明当前 A8 helper 已经不再是“纯逐元素标量写回”组织，而是开始按：

- `row_tile x m_tile` 累加 tile
- `128B` output block pack/store

去表达更接近 HVX 向量寄存器与 vector-line 的处理边界。

同时，当前 A8 路径的配置边界也已进一步收紧为项目真实目标：

- `bits = 4`
- `group_size = 128`
- `symmetric = true`

也就是说，当前 builder / Python 模块层 / runtime op 都已经开始明确把 A8 支线视为：

- `Qwen3-1.7B`
- GPTQ `W4G128`
- `sym = true`

的目标专用路径，而不是继续保留不必要的“泛配置”外观。

## 7. 当前验证结论

## 7.1 已通过的轻量验证

在 `gptq` 环境中，下面这些都能跑通：

- `python backends/qualcomm/tests/test_tman_w4a8.py`
- `python tools/tman_w4a8_reference_check.py`
- `python tools/tman_w4a8_qwen3_scale_audit.py ...`
- `python tools/tman_w4a8_real_layer_probe.py ...`
- `python tools/tman_w4a8_model_slice_audit.py ...`

## 7.2 全量 196 个目标 linear 切片审计结论

当前已经能对：

- 全部 `196` 个目标 linear
- 每个模块取前 `8` 个输出通道切片
- 使用真实权重 + 真实 activation scale

做批量审计。

当前结论：

- 若按严格浮点口径：
  - 少数模块会出现 `1 个 INT8 bin` 等价差异
- 若按：
  - `bin_only + qdiff_tol = 1`
  口径：
  - `196 / 196` 全部通过

这说明当前 baseline 已经足够稳定，可以继续向更底层实现推进。

## 7.3 当前已知差异模块

严格口径下当前典型模块为：

- `model.layers.17.self_attn.k_proj`
- `model.layers.6.mlp.down_proj`

当前结论：

- 这些差异是：
  - `high-level reference`
  - 与 `grouped/runtime`
  的参考定义差异
- 不是 runtime baseline 的逻辑 bug

## 8. 当前推荐的开发优先级

### 第一优先级

继续从 `PrecomputeA8` helper kernel 入手，推进更实质的 HVX 风格实现：

- 对齐 block
- block 级 build/store
- 更明确的 vector-line 组织

### 第二优先级

推进 `LinearA8`：

- block 级 LUT 消费
- 更像寄存器/向量处理的内层组织
- 在当前 tile helper 基础上继续替换成更实质的 HVX intrinsic / vector-register 实现
- 优先从当前 `W4G128 sym` 目标专用 helper 入手，不必先做泛化

### 第三优先级

处理 `FinalizeA8`：

- pack/clamp/store 风格优化
- 在当前 output-block helper 基础上继续推进真正的向量 pack/store

### 第四优先级

在具备完整 ExecuTorch / QNN 环境后：

- 做 compile-only 验证
- 做 device 侧验证
- 做性能 benchmark

## 9. 当前最常用命令

### 轻量测试

```bash
source /home/lmx/Softwares/anaconda3/etc/profile.d/conda.sh
conda activate gptq
python backends/qualcomm/tests/test_tman_w4a8.py
```

### reference 检查

```bash
source /home/lmx/Softwares/anaconda3/etc/profile.d/conda.sh
conda activate gptq
python tools/tman_w4a8_reference_check.py
```

### scale 覆盖审计

```bash
source /home/lmx/Softwares/anaconda3/etc/profile.d/conda.sh
conda activate gptq
python tools/tman_w4a8_qwen3_scale_audit.py \
  --scales_pt /home/lmx/Test/GPTQ_Opus/Qwen3-1.7B_w4a8_gptq/activation_scales_a8_per-tensor_sym.pt \
  --num_layers 28
```

### 全量 196 模块切片审计

```bash
source /home/lmx/Softwares/anaconda3/etc/profile.d/conda.sh
conda activate gptq
python tools/tman_w4a8_model_slice_audit.py \
  --weights_file /home/lmx/Projects/tman/models/Qwen3-1.7B-GPTQ-Int4/model.safetensors \
  --scales_pt /home/lmx/Test/GPTQ_Opus/Qwen3-1.7B_w4a8_gptq/activation_scales_a8_per-tensor_sym.pt \
  --num_layers 28 \
  --out_features_slice 8 \
  --max_modules 0 \
  --print_every 196 \
  --max_int8_bin_diff_tol 1 \
  --mismatch_mode bin_only \
  --seed 0
```

## 10. 后续接手时要注意的事

- 不要破坏 `W4A16` 原路径
- 当前更贴近目标的正确性口径是：
  - `grouped reference`
  - `runtime baseline`
- 若出现少量 `1 bin` 差异，先判断它是否属于：
  - `high-level reference`
  - 与 `grouped/runtime`
  的口径差异
- 当前最优先工作不是再加新功能，而是继续把 A8 helper kernel 往真正的 HVX/Hexagon 风格推进

## 11. 阶段性交付判断

当前代码已经基本具备“阶段性交给其它合作者做 QNN/OpPackage 真编译与 device 验证”的条件，原因是：

- `qwen3.py` 的 A8 编译入口已经接通
- A8 custom op / builder / runtime op / op 注册 / XML schema 都已接通
- A8 路径当前已明确收敛到项目真实目标：
  - `W4`
  - `group_size = 128`
  - `symmetric = true`
- 当前对外 ABI 已采用更稳妥的 4D packed 形态，降低了 QNN 编译阶段的 rank/schema 风险
- 轻量 Python/reference 验证已经稳定通过

但这仍然是“阶段性交付”，不是“最终可发布交付”，当前仍需外部环境完成下面几项：

- 用完整 ExecuTorch / QNN 环境做 compile-only 验证
- 用真实 `TMANOpPackage` 构建链做 x86 / v75 / android 目标编译
- 用真实设备做 graph prepare / runtime load / inference 验证
- 做性能 benchmark，确认当前 helper 收敛没有引入明显回退

因此，更准确的判断是：

- 现在适合交给其它合作者做：
  - QNN schema / graph compile / OpPackage build / device smoke test
- 现在还不适合宣称：
  - 已完成 device 侧可用性验证
  - 已完成性能目标验证
