# Qwen3 TMAN W4A8 交付说明

## 1. 文档目的

本文档用于把当前 `Qwen3 TMAN W4A8` 项目的实现状态、可执行命令、检查项、已知风险和通过标准集中整理成一份可直接交接给其它合作者的阶段性交付说明。

当前目标不是宣称“所有 device / benchmark 工作已经完成”，而是提供一份足够稳定、足够清晰的 handoff 包，支持后续合作者继续完成：

- QNN graph compile
- `TMANOpPackage` 真编译
- device smoke test
- 后续 benchmark

## 2. 当前交付范围

当前交付严格限定在：

- 模型：
  - `Qwen3-1.7B`
- 权重：
  - GPTQ `W4`
- 配置：
  - `group_size = 128`
  - `symmetric = true`
- 覆盖算子：
  - decoder 内部 7 类 linear
  - `q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj`

不在本次交付范围内：

- `lm_head`
- embedding
- 全图 int8 化
- 非 `W4G128 sym`
- 非 Qwen3 模型

## 3. 当前实现概况

当前仓库已经具备一条可运行、可验证、可继续向 device 侧推进的 `W4A8` 支线：

- 编译入口：
  - [qwen3.py](/home/lmx/Projects/tman/test/executorch/examples/qualcomm/oss_scripts/qwen3/qwen3.py)
  - 已支持：
    - `--use_tman`
    - `--tman_act_mode a8`
    - `--activation_scales_pt`
- scale 管理：
  - [tman_w4a8.py](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/utils/tman_w4a8.py)
- Python 模块：
  - [utils.py](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/utils/utils.py)
  - `TMANLinearA8`
- custom op：
  - [custom_ops.py](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/builders/custom_ops.py)
  - `tman::linear_a8`
- A8 builder：
  - [op_tman_linear_a8.py](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/builders/op_tman_linear_a8.py)
- runtime op：
  - [TMANPrecomputeA8.cpp](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANPrecomputeA8.cpp)
  - [TMANLinearA8.cpp](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANLinearA8.cpp)
  - [TMANFinalizeA8.cpp](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANFinalizeA8.cpp)
- helper kernel：
  - [hvx_funcs.h](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackage/include/hvx_funcs.h)
- op 注册 / schema：
  - [TMANOpPackageInterface.cpp](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackage/src/TMANOpPackageInterface.cpp)
  - [TMANOpPackageHtp.xml](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackage/config/TMANOpPackageHtp.xml)

### 3.1 当前 A8 路径的关键收敛点

当前实现已经明确收敛到项目真实目标配置：

- `bits = 4`
- `group_size = 128`
- `symmetric = true`

当前 helper kernel 已不再是最初的“纯标量 baseline”：

- `LinearA8`
  - 已引入 target-specific `W4G128 sym` helper
  - 已引入 LUT block 的向量寄存器式 staging
- `FinalizeA8`
  - 已引入一段更实质的 HVX 路径：
    - `fp32 vector -> s32 round -> clamp -> s16 sat`
    - 再做 block 级 pack/store

### 3.2 当前对外 ABI

逻辑语义上，A8 helper 内部仍按 block / vector-line 理解数据：

- qweight logical:
  - `[M, G, Qg, bits]`
- LUT scratch logical:
  - `[row, group, qgroup_in_group, 16]`

为了降低 QNN / OpPackage 真编译阶段的 rank/schema 风险，当前 builder/runtime 对外 ABI 已重新压回 4D packed：

- qweight ABI:
  - `[1, M, G, qgroup_vector_blocks * (4 * bits)]`
- LUT scratch ABI:
  - `[1, row, group, qgroup_vector_blocks * (4 * 16)]`

这意味着：

- helper kernel 内部继续按 block / vector-line 逻辑消费；
- QNN schema / graph compile 阶段继续看到 4D packed tensor。

## 4. W4A16 与当前 W4A8 链路对比

| 阶段 | 原 `W4A16` linear | 当前 `W4A8` linear |
| --- | --- | --- |
| 输入 | 图外和 TMAN 内部都以 FP16/A16 激活语义为主；builder 会在 TMAN 前后插 `Convert` 包裹旧路径 | 图外仍看到 FP16；进入 TMAN A8 支线前先显式 `Convert` 到对称 per-tensor INT8 |
| Precompute | 使用旧 `hvx_lut_ctor(...)`；输入是 FP16 激活；生成 LUT，并维护激活侧 `ls / lb` 等量 | 使用 `tman_a8_precompute_lut_scalar(...)`；输入是 INT8 激活；按 4 激活构造 16 项整数 LUT；不再走旧的 `ls / lb` 激活语义 |
| Linear | 使用旧 `hvx_tbl(...)`；沿用原始 HVX `vlut16 + bit-serial` 查表路径，消费旧格式 LUT / weight / scales | 使用 `tman_a8_linear_scalar(...)`；消费 A8 LUT、grouped LUT index 和 folded `effective_scale`；当前已引入 `W4G128 sym` 目标专用 helper 和 LUT block 向量 staging |
| Finalize | 使用旧 `hvx_bit_serial(...)`；把中间结果恢复到 FP16 输出 | 使用 `tman_a8_finalize_scalar(...)`；把 float accumulator 量化到 INT8；当前已引入 `fp32 vector -> s32 round -> clamp -> s16 sat` 的部分 HVX 路径，再做 block pack/store |
| 输出 | TMAN 路径直接产出 FP16，交回外部图 | TMAN A8 路径内部先产出 INT8，随后通过图外 `Convert` 反量化回 FP16，再交给其它非 linear |

## 5. 推荐执行命令

## 5.1 轻量本地验证

### unittest

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

## 5.2 OpPackage 真编译命令

下面这些命令需要完整 QNN / Hexagon / Android NDK 环境：

- `QNN_SDK_ROOT`
- `HEXAGON_SDK_ROOT`
- `ANDROID_NDK_ROOT`

### x86 host 版本

```bash
export QNN_SDK_ROOT=/path/to/qnn_sdk
export HEXAGON_SDK_ROOT=/path/to/hexagon-sdk
make -C backends/qualcomm/runtime/op_packages/TMANOpPackage htp_x86
```

### Hexagon v75 版本

```bash
export QNN_SDK_ROOT=/path/to/qnn_sdk
export HEXAGON_SDK_ROOT=/path/to/hexagon-sdk
make -C backends/qualcomm/runtime/op_packages/TMANOpPackage htp_v75
```

### Android aarch64 版本

```bash
export QNN_SDK_ROOT=/path/to/qnn_sdk
export HEXAGON_SDK_ROOT=/path/to/hexagon-sdk
export ANDROID_NDK_ROOT=/path/to/android-ndk
make -C backends/qualcomm/runtime/op_packages/TMANOpPackage htp_aarch64
```

## 5.3 Qwen3 compile-only 示例命令

下面命令是推荐 handoff 命令模板，依赖完整 ExecuTorch / QNN 环境。路径请按实际环境替换：

```bash
export QNN_SDK_ROOT=/path/to/qnn_sdk

python examples/qualcomm/oss_scripts/qwen3/qwen3.py \
  -m SM8650 \
  -b /path/to/build-android \
  -c \
  --llama_model qwen3 \
  --model_dir /path/to/qwen3_gptq_model_dir \
  --tokenizer_model /path/to/qwen3_tokenizer_dir_or_file \
  --use_tman \
  --tman_act_mode a8 \
  --activation_scales_pt /home/lmx/Test/GPTQ_Opus/Qwen3-1.7B_w4a8_gptq/activation_scales_a8_per-tensor_sym.pt \
  -a /tmp/qwen3_tman_w4a8_artifacts
```

说明：

- 当前 `W4A8` 路径不能和全图 `--ptq` 同时打开
- compile-only 阶段重点关注：
  - `.pte`
  - `*_quant_attrs.txt`
  是否成功生成

## 6. 交付检查项

建议其它合作者按下面顺序检查：

### 6.1 本地轻量检查

- `test_tman_w4a8.py` 通过
- `tman_w4a8_reference_check.py` 通过
- `tman_w4a8_qwen3_scale_audit.py` 通过
- `tman_w4a8_model_slice_audit.py`
  - 在 `bin_only + qdiff_tol = 1` 口径下通过

### 6.2 OpPackage 构建检查

- `htp_x86` 构建成功
- `htp_v75` 构建成功
- `htp_aarch64` 构建成功
- 新增 A8 op 没有 schema / validate 报错：
  - `TMANPrecomputeA8`
  - `TMANLinearA8`
  - `TMANFinalizeA8`

### 6.3 compile-only 检查

- `qwen3.py --use_tman --tman_act_mode a8` 成功结束
- 产物目录中成功生成：
  - `.pte`
  - `*_quant_attrs.txt`
- 没有出现明显的 rank / schema / tensor wrapper 校验错误

### 6.4 device smoke test 检查

- graph prepare 成功
- runtime load 成功
- device 侧至少能跑通一次 smoke inference
- 没有出现：
  - op validation failure
  - package load failure
  - graph prepare failure
  - obvious tensor rank mismatch

## 7. 已知风险

当前代码已经适合做阶段性交付，但仍存在下面这些已知风险：

### 7.1 真实编译与设备验证尚未在当前机器完成

当前机器没有完整 ExecuTorch / QNN 运行环境，因此：

- 还没有在本机完成 compile-only 真验证
- 还没有在本机完成 OpPackage 真编译
- 还没有在本机完成 device smoke test

### 7.2 `LinearA8` 仍未达到原 `W4A16 hvx_tbl(...)` 的成熟程度

当前 `LinearA8` 虽然已经有 target-specific helper 和 LUT block staging，但还没有完全替换成原 `W4A16` 那种成熟的 `vlut16 + register accumulation` 路径。

### 7.3 `FinalizeA8` 仍是“部分 HVX 化”

当前 `FinalizeA8` 已经做到了：

- `fp32 vector -> s32 round -> clamp -> s16 sat`

但最后一段从 `s16` 到最终 `s8` 的 pack 仍不是完全纯 HVX pack 实现。

### 7.4 当前路径是目标专用实现

当前 `W4A8` 明确只服务：

- `W4`
- `group_size = 128`
- `symmetric = true`
- Qwen3 decoder 内部 7 类 linear

任何超出当前范围的尝试，都不应被视为当前阶段的回归目标。

## 8. 通过标准

建议把通过标准分成“阶段性交付通过”和“最终交付通过”两档。

### 8.1 阶段性交付通过

满足下面条件即可认为 handoff 包可交给其它合作者继续推进：

- `python backends/qualcomm/tests/test_tman_w4a8.py` 通过
- `python tools/tman_w4a8_reference_check.py` 通过
- scale 审计和模型切片审计通过当前口径
- OpPackage schema / builder / runtime op / qwen3 A8 入口都已接通
- 对外 ABI 已采用 4D packed 形态，适合做 QNN 真编译验证

### 8.2 最终交付通过

只有满足下面条件，才应视为真正的完整交付：

- `TMANOpPackage` x86 / v75 / aarch64 构建全部通过
- `qwen3.py` compile-only 通过
- device smoke test 通过
- A8 路径确认被真实执行，而不是 silent fallback 到旧路径
- 至少完成一轮 prefill / decode benchmark
- 对数值偏差有明确记录，并确认没有超出当前项目接受范围

## 9. 阶段性交付结论

当前代码已经具备“阶段性交给其它合作者做 QNN / OpPackage 真编译与 device 验证”的条件，理由是：

- A8 入口、builder、runtime op、注册、XML schema 都已经接通
- A8 路径当前已明确收敛到真实目标配置
- 轻量 Python/reference 验证已经稳定通过
- 对外 ABI 已采用更稳妥的 4D packed 形态，降低了 QNN 编译阶段的 rank/schema 风险

但当前仍不应宣称：

- 已完成 device 侧可用性验证
- 已完成 benchmark 与性能目标验证

因此，当前更准确的交付判断是：

- 适合交给其它合作者做：
  - QNN schema / graph compile / OpPackage build / device smoke test
- 尚不适合宣称：
  - 最终可发布交付
