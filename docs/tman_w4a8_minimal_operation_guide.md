# W4A8 TMAN 最小操作说明

## 1. 文档用途

本文档只保留最小必要步骤，供合作者快速完成：

- 正式 W4A8 runtime op package 构建
- `x86` 模拟器算子级验证
- 离线 `context binary` 编译
- Qwen3 正式路径接入时的最小配置说明

## 2. 相关目录

### 2.1 正式 runtime 包

- [backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8](/home/lmx/Projects/tman/test/executorch/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8)

### 2.2 算子级测试

- [op_gen/test/test_tman_linear_a8.cpp](/home/lmx/Projects/tman/test/executorch/op_gen/test/test_tman_linear_a8.cpp)
- [op_gen/test/Makefile.linear_a8](/home/lmx/Projects/tman/test/executorch/op_gen/test/Makefile.linear_a8)

### 2.3 Qwen3 接入入口

- [examples/qualcomm/oss_scripts/qwen3/qwen3.py](/home/lmx/Projects/tman/test/executorch/examples/qualcomm/oss_scripts/qwen3/qwen3.py)

## 3. 前置环境

至少需要准备：

- `QNN_SDK_ROOT`
- `HEXAGON_SDK_ROOT`
- `ANDROID_NDK_ROOT`

本地已验证过的一组示例环境变量：

```bash
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.34.0.250424
export HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1
export ANDROID_NDK_ROOT=/opt/ndk/android-ndk-r28b
```

## 4. 构建正式 W4A8 op package

在仓库根目录执行：

```bash
make -C backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8 \
  htp_x86 \
  QNN_SDK_ROOT=$QNN_SDK_ROOT \
  HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT \
  HEXAGON_TOOLS_VERSION_X86=19.0.04
```

若需要设备侧包：

```bash
make -C backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8 \
  htp_v75 \
  QNN_SDK_ROOT=$QNN_SDK_ROOT \
  HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT
```

构建产物示例：

```text
backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/build/x86_64-linux-clang/libQnnTMANOpPackage.so
backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/build/hexagon-v75/libQnnTMANOpPackage.so
```

## 5. x86 模拟器算子级验证

在仓库根目录执行：

```bash
cd op_gen/test
make -f Makefile.linear_a8 run_x86
```

说明：

- 当前 `Makefile.linear_a8` 默认已经指向正式 runtime 包 `TMANOpPackageW4A8`
- 会自动：
  1. 构建正式 x86 op package
  2. 构建测试 harness
  3. 用 `QNN x86 HTP simulator` 执行测试

期望结果：

```text
Passed: 6
Failed: 0
Result: ALL TESTS PASSED
```

## 6. 离线 context 编译

在 `op_gen/test` 下执行：

```bash
make -f Makefile.linear_a8 compile_context
```

期望生成：

```text
tman_linear_a8_ctx.bin
tman_linear_a8_ctx.meta
```

这一步通过说明：

- runtime 包注册成功
- graph prepare / finalize 正常
- 新增的 `TMANFinalizeA8` 也能通过 `QNN context` 编译链路

## 7. Qwen3 正式 W4A8 路径最小使用方式

### 7.1 必要参数

新增的正式 W4A8 路径通过下面两个参数进入：

```bash
--use_tman_a8
--tman_a8_scale_path <可选>
```

如果不显式给 `--tman_a8_scale_path`，代码会默认使用：

```text
<model_dir>/activation_scales_a8_per-tensor_sym.pt
```

### 7.2 scale 文件要求

scale 文件需要能提供每个 TMAN 线性层的：

- `input_scale`
- `output_scale`

当前已确认真实文件是扁平 key 结构，例如：

```text
model.model.layers.0.self_attn.q_proj.input_scale
model.model.layers.0.self_attn.q_proj.output_scale
```

### 7.3 OpPackage 环境变量

开启 `--use_tman_a8` 时，`QNN_OP_PACKAGE_PATHS` 需要指向 **W4A8 正式包目录**，不是旧 `TMANOpPackage` 目录。

示例：

```bash
export QNN_OP_PACKAGE_PATHS="\
${EXECUTORCH_ROOT}/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/build/x86_64-linux-clang/libQnnTMANOpPackage.so:TMANOpPackageInterfaceProvider:CPU;\
${EXECUTORCH_ROOT}/backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8/build/hexagon-v75/libQnnTMANOpPackage.so:TMANOpPackageInterfaceProvider:HTP"
```

### 7.4 Qwen3 compile-only 示例

下面给出一个最小 compile-only 形式示例。实际环境参数可按合作者机器调整：

```bash
python examples/qualcomm/oss_scripts/qwen3/qwen3.py \
  -b build-android \
  -m SM8650 \
  --llama_model qwen3 \
  --model_dir /home/lmx/Projects/tman/models/Qwen3-1.7B-GPTQ-Int4 \
  --tokenizer_model /home/lmx/Projects/tman/models/Qwen3-1.7B-GPTQ-Int4/tokenizer.json \
  --prompt "Hello" \
  --model_mode kv \
  --max_seq_len 128 \
  --artifact /tmp/qwen3_w4a8 \
  --ptq 16a4w \
  --use_tman_a8 \
  --compile_only
```

如果 scale 文件不在默认位置，可额外加：

```bash
  --tman_a8_scale_path /path/to/activation_scales_a8_per-tensor_sym.pt
```

## 8. 当前支持范围

当前正式 W4A8 路径支持：

- `W4`
- `group_size = 128`
- `symmetric = 1`
- `N = 1`
- `K % 256 == 0`

不满足这些条件时，代码会尽量在导出或 op package 入口阶段直接报错。

## 9. 典型问题排查

### 9.1 scale 文件没找到

检查：

- `--use_tman_a8` 是否开启
- `--tman_a8_scale_path` 是否给对
- 默认路径 `<model_dir>/activation_scales_a8_per-tensor_sym.pt` 是否存在

### 9.2 scale 文件与模块名不匹配

当前逻辑是“严格失败”，如果某个 `TMANLinearA8` 没匹配到 `input_scale/output_scale`，会在导出阶段直接报错。

优先检查：

- 模型实际模块名
- scale 文件 key 前缀是否是 `model.` / `model.model.`

### 9.3 OpPackage 载入错误

优先检查：

- `QNN_OP_PACKAGE_PATHS` 是否指向 `TMANOpPackageW4A8`
- x86/HTP 两条路径是否都存在
- 是否误用了旧 `TMANOpPackage` 的库路径

### 9.4 算子级验证失败

先回到 `op_gen/test` 验证：

```bash
cd op_gen/test
make -f Makefile.linear_a8 run_x86
```

只有在这里通过后，再继续排查整模型 compile-only 问题。

## 10. 建议的最小联调顺序

1. 构建正式 W4A8 op package
2. 跑 `op_gen/test` 的 `x86` 模拟器回归
3. 跑 `compile_context`
4. 在完整 ExecuTorch 环境中跑 `qwen3.py --compile_only --use_tman_a8`
5. 再进入设备侧联调

