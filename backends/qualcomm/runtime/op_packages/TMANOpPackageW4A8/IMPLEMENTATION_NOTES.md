# TMANOpPackageW4A8 实现说明

本文基于当前工作区代码梳理 `backends/qualcomm/runtime/op_packages/TMANOpPackageW4A8` 的实际实现逻辑，并结合测试文件 `op_gen/test/test_tman_linear_a8.cpp` 说明整条算子链路现在到底做了什么、约束是什么、关键函数各自负责什么。

## 1. 当前算子包包含哪些算子

算子包入口在 `src/TMANOpPackageInterface.cpp`。

- 注册了 3 个自定义算子：
  - `TMANPrecomputeA8`
  - `TMANLinearA8`
  - `TMANFinalizeA8`
- `TMANOpPackageValidateOpConfig` 也明确约束了每个算子的输入/输出/参数个数：
  - `TMANPrecomputeA8`: 2 inputs, 1 output, 3 params
  - `TMANLinearA8`: 3 inputs, 1 output, 3 params
  - `TMANFinalizeA8`: 2 inputs, 1 output, 3 params

这三个算子组合起来，形成当前 W4A8 路径的完整执行链：

`TMANPrecomputeA8 -> TMANLinearA8 -> TMANFinalizeA8`

## 2. 这条 W4A8 路径当前支持的范围

从三个实现文件的参数检查逻辑可以看出，这一版是“收敛后的最小可用正式路径”，限制非常明确：

- 只支持 `group_size = 128`
- 只支持 `bits = 4`
- 只支持 `symmetric = true`
- 只支持 decoder 风格 `gemm_n = 1`
- `gemm_k` 必须同时满足：
  - `gemm_k % 128 == 0`
  - `gemm_k % 256 == 0`

也就是说，这不是一个泛化的 W4A8 实现，而是一条专门针对 `W4G128 symmetric + A8 activation + N=1` 的定制实现。

代码里多处注释都说明了设计目的：优先复用旧 TMAN 的 LUT / HVX bit-serial 主干，先把数学语义和端到端链路稳定下来，再考虑更广的配置和更激进的优化。

## 3. 整体数据流

这条链路的实际职责分工如下：

1. `TMANPrecomputeA8`
   把输入激活 `x(fp16)` 按外部传入的 `input_scale` 做对称 A8 量化，再把量化后的激活重建成 LUT 所需的 packed buffer `l`。

2. `TMANLinearA8`
   用 `l`、旧 TMAN 兼容的 `qweight/scales` 打包权重，调用复用下来的 `hvx_tbl` 主干，计算出中间 scratch `c`。

3. `TMANFinalizeA8`
   先把 `c` 这个 bit-plane scratch 合成为 fp16，再按外部 `output_scale` 做一次“量化再反量化”，最终输出 fp16 `y`。

注意两点：

- 层间接口仍然保持 `fp16 in / fp16 out`。
- A8 量化语义已经被真正纳入链路，但不是直接输出 int8，而是“约束到 A8 离散水平后，再恢复成 fp16”。

## 4. 张量语义与内存布局

### 4.1 输入输出张量

测试文件里构图时使用的张量语义如下：

- `x`: `FP16`, shape `[1, 1, 1, K]`
- `input_scale`: `FP32`, shape `[1, 1, 1, 1]`
- `l`: QNN 上声明成 `UINT8` raw buffer，shape `[1, 1, 1, packed_bytes]`
- `qweight`: `INT32`, shape `[1, 1, p_tiles, ints_per_tile]`
- `scales`: `INT32`, shape `[1, 1, p_tiles, ints_per_tile]`
- `c`: QNN 上声明成 `UINT8` raw buffer，shape `[1, 1, 1, M * bits * sizeof(float)]`
- `output_scale`: `FP32`, shape `[1, 1, 1, 1]`
- `y`: `FP16`, shape `[1, 1, 1, M]`

### 4.2 `l` 的真实语义

虽然 `l` 在 QNN 图里是一个按字节承载的 raw tensor，但它的真实内容是一个连续 packed buffer：

`[lut:int16][ls:float][lb:float]`

其中：

- `lut`：每 4 个激活值构造一个 16 项 LUT
- `ls`：activation scale，按 `gemm_k / 256` 个 float 存放
- `lb`：每个权重 group 的 bias 补偿，按 `gemm_k / group_size` 个 float 存放

对应 helper 在 `include/hvx_funcs.h`：

- `tman_w4a8_packed_lut_size`
- `tman_w4a8_packed_ls_size`
- `tman_w4a8_packed_lb_size`
- `tman_w4a8_packed_buffer_bytes`

### 4.3 `c` 的真实语义

`c` 也不是普通的 `M` 个 float 输出。

当前实现里，`c` 的字节数固定为：

`gemm_m * bits * sizeof(float)`

也就是说，`c` 实际承载的是 `M * 4` 个 float 的 bit-plane scratch。后续 `TMANFinalizeA8` 会调用 `hvx_bit_serial<..., Bits=4>` 把这 4 个 bit-plane 合成为最终 `M` 个 fp16 输出。

这一点非常关键，因为如果把 `c` 理解成普通 GEMM 输出，就会误判 `LinearA8` 和 `FinalizeA8` 的职责边界。

## 5. 三个算子的具体实现逻辑

## 5.1 `TMANPrecomputeA8`

文件：`src/ops/TMANPrecomputeA8.cpp`

签名：

```cpp
GraphStatus tmanprecomputea8Impl(
    TensorType& l,
    const TensorType& x,
    const TensorType& input_scale,
    const Int32Tensor& t_group_size,
    const Int32Tensor& t_bits,
    const Int32Tensor& t_symmetric)
```

### 实现流程

1. 读取并校验 `group_size/bits/symmetric`
2. 校验 `x/l/input_scale` 的 rank 和 shape
3. 要求：
   - `gemm_n == 1`
   - `gemm_k > 0`
   - `gemm_k % 128 == 0`
   - `gemm_k % 256 == 0`
4. 根据 `gemm_k` 计算 packed buffer 大小，校验 `l` 的最后一维是否等于预期字节数
5. 从 `l` 这块连续内存里切出：
   - `l_ptr`: LUT 起始地址
   - `ls_ptr`: activation scale 段
   - `lb_ptr`: bias 补偿段
6. 读取外部给定的 `input_scale`
7. 使用 `input_scale` 把输入 `x(fp16)` 量化到 `int8`，再立刻反量化回 `fp16`
8. 调用 `hvx_lut_ctor`，把这份“量化后再反量化”的激活写成旧 TMAN 所需的 LUT/scale/bias 布局

### 关键点

- 当前正式路径不在 runtime 内部动态估计 activation scale，而是严格使用外部输入的 `input_scale`
- 为了复用旧 `hvx_lut_ctor`，代码没有直接把 int8 值送进去，而是先得到 `q_a`，再恢复成 `q_a * s_a` 的 fp16 值
- 临时量化 buffer 用 `std::vector<XType>` 分配，并手动按 `VLEN=128B` 对齐，避免 HVX 读取非对齐地址

### 关键函数讲解

#### `tman_w4a8_round_clip_int8`

作用：把浮点值四舍五入并裁剪到 `[-127, 127]`。

用途：

- 输入激活量化
- 输出激活量化

这保证当前路径使用的是对称 int8，而不是带零点的非对称量化。

#### `hvx_lut_ctor`

这是旧 TMAN 路径复用下来的 LUT 构造主函数。

在当前实例化中：

```cpp
hvx_lut_ctor<int16_t, __fp16, 256, 128, false, 4>(...)
```

含义可以理解为：

- 激活侧按 256 元素统计 `ls`
- 权重量化 group 是 128
- 不使用 zero-point
- 每 4 个激活值构造一组 16 项 LUT

它会把输入激活写成三段 packed 数据：

- `lut`: 每 4 个值生成 16 项查找表
- `ls`: 每 256 个 K 对应一个激活 scale
- `lb`: 每个 128-group 对应一个 bias 补偿

其中 `lb` 的存在，是为了让后续按 bit-plane 查表累加时，能够恢复对称 4bit 权重 `[-8, 7]` 的数值语义。

## 5.2 `TMANLinearA8`

文件：`src/ops/TMANLinearA8.cpp`

签名：

```cpp
GraphStatus tmanlineara8Impl(
    TensorType& c,
    const TensorType& l,
    const TensorType& qweight,
    const TensorType& scales,
    const Int32Tensor& t_group_size,
    const Int32Tensor& t_bits,
    const Int32Tensor& t_symmetric)
```

### 实现流程

1. 检查 `group_size=128, bits=4, symmetric=true`
2. 检查 `l/qweight/scales/c` 都是 4D tensor
3. 要求 `gemm_n == 1`
4. 要求 `qweight/scales` 的前两维为 1
5. 通过 `c` 和 `qweight` 的 shape 反推出：
   - `gemm_m`
   - `gemm_k`
6. 要求：
   - `gemm_k > 0`
   - `gemm_k % group_size == 0`
   - `gemm_k % TILE_K(=256) == 0`
7. 校验 `l` 的 packed byte 数和 `c` 的 scratch byte 数
8. 从 `l` 中切出：
   - `l_ptr`
   - `ls_ptr`
   - `lb_ptr`
9. 调用 `hvx_tbl` 完成主计算，输出到 `c_ptr`

### 它真正做的计算

这一步不是直接产出最终 fp16，而是调用旧 TMAN 的 table-lookup + bit-plane accumulation 核心：

```cpp
hvx_tbl<int16_t, __fp16, float, 256, 128, false, 4, 256, 4, true>(...)
```

这次实例化说明：

- LUT 元素类型是 `int16_t`
- 权重 scale 类型是 `fp16`
- 聚合类型是 `float`
- activation group 为 256
- weight group 为 128
- 对称量化，不带 zero-point
- 权重 bit 数是 4
- `TileK = 256`
- 每组 LUT 基元大小 `g = 4`

### `hvx_tbl` 的核心思想

`hvx_tbl` 是整条链路里最核心的数值内核，逻辑可以概括为：

1. 从 `l` 中取出预计算好的 LUT
2. 从 `qweight` 中取出打包后的 4bit 权重索引
3. 用 HVX `vlut16` 类查表指令，把每个 4bit 模式映射成 LUT 值
4. 按 bit-plane 累加成 group 内整数和
5. 乘上激活 scale `ls`
6. 在 group 尾部加上 bias 补偿 `lb`
7. 再乘上权重 scale `s`
8. 把结果写进 `c`

需要特别注意：

- 这里写进 `c` 的不是最终输出，而是 `P = M * Bits` 路 float scratch
- 最终 bit-plane 合成不在 `hvx_tbl` 内部完成，而是交给 `TMANFinalizeA8`

### autosplit 优化

`TMANLinearA8.cpp` 里还定义了一个 autosplit 规则：

- 当 `qweight` 第 3 维大于 1
- 且 `scales` 的总字节数大于 128

就沿着第 3 维把 `qweight/scales` 切块，构造多个 `TMANLinearA8` 子节点执行。

对应辅助函数：

- `simpledim_chunk1_4d_split_start_a8`
- `simpledim_chunk1_4d_split_size_a8`

这套 helper 的名字专门带 `_a8` 后缀，是为了避免和旧 TMANLinear 路径的 autosplit helper 符号重名。

## 5.3 `TMANFinalizeA8`

文件：`src/ops/TMANFinalizeA8.cpp`

签名：

```cpp
GraphStatus tmanfinalizea8Impl(
    TensorType& y,
    const TensorType& c,
    const TensorType& output_scale,
    const Int32Tensor& t_group_size,
    const Int32Tensor& t_bits,
    const Int32Tensor& t_symmetric)
```

### 实现流程

1. 校验参数仍然是 `128/4/symmetric`
2. 校验：
   - `y` 是 `[1,1,1,M]`
   - `c` 是 `[1,1,1,M * bits * sizeof(float)]`
   - `output_scale` 是 `[1,1,1,1]`
3. 从 `c` 取出 float scratch，从 `output_scale` 取出单个标量
4. 调用 `tman_w4a8_finalize_fp16_with_output_scale`

### 关键函数讲解

#### `tman_w4a8_finalize_fp16_with_output_scale`

这是当前正式 W4A8 路径的补充函数，职责非常清晰：

1. 先调用 `hvx_bit_serial<__fp16, float, 4>`，把 `c` 中的 4 个 bit-plane 合成为 fp16
2. 再对每个输出元素执行：
   - `y_q = round_clip(y_fp / output_scale)`
   - `y = y_q * output_scale`

也就是：

- 先合成为层输出
- 再做一轮 A8 量化
- 最后再反量化回 fp16

因此 `TMANFinalizeA8` 不是“普通 cast/拷贝”，而是把 output activation quantization 真正纳入运行时链路的关键补丁。

#### `hvx_bit_serial`

这个函数负责把 `c` 中按 bit-plane 排列的中间结果恢复成最终输出。

它的输入逻辑是：

- `P = GemmM * Bits`
- 每 `Bits` 份 float 对应一个最终输出通道

以 `Bits=4` 为例，它会按类似下面的方式合成：

- 第 0 bit-plane 先乘 `0.5`
- 第 1 bit-plane 直接加
- 第 2 bit-plane 乘 `2`
- 第 3 bit-plane 乘 `4`

最后再缩成 fp16 写入 `y`

从代码形式看，这一步对应的是旧 TMAN bit-serial 表达的一种恢复方式，因此 `LinearA8` 和 `FinalizeA8` 必须配套理解，单看其中一边都容易误读。

## 6. `hvx_funcs.h` 里与当前路径最相关的关键 helper

虽然 `hvx_funcs.h` 很大，但当前 W4A8 正式路径真正依赖的核心 helper 主要是下面这些：

- `tman_w4a8_packed_lut_size`
  计算 LUT 段元素数，公式是 `(gemm_k / 4) * 16`

- `tman_w4a8_packed_ls_size`
  计算 `ls` 段元素数，公式是 `gemm_k / 256`

- `tman_w4a8_packed_lb_size`
  计算 `lb` 段元素数，公式是 `gemm_k / group_size`

- `tman_w4a8_packed_buffer_bytes`
  计算整个 `l` packed buffer 的字节数，并且保证 `ls/lb` 两段最少各留 `128B`

- `tman_w4a8_round_clip_int8`
  对称 int8 量化时的 rounding + clipping

- `hvx_lut_ctor`
  从激活构造 LUT、`ls`、`lb`

- `hvx_tbl`
  做 LUT table lookup、group accumulation、scale/bias 融合，输出 bit-plane float scratch

- `hvx_bit_serial`
  把 `c` 里的 bit-plane scratch 合成为最终 fp16

- `tman_w4a8_finalize_fp16_with_output_scale`
  在最终 fp16 输出上再做一次 output quant/dequant

## 7. 测试文件是如何验证这条链路的

测试文件：`op_gen/test/test_tman_linear_a8.cpp`

## 7.1 测试构图逻辑

测试通过 `build_pipeline_graph` 明确构造了完整链路：

- `TMANPrecomputeA8(x, input_scale) -> l`
- `TMANLinearA8(l, qweight, scales) -> c`
- `TMANFinalizeA8(c, output_scale) -> y`

注意：

- 文件头注释里写的是 `TMANFinalize`
- 但真正 `graphAddNode` 加入图里的算子已经是 `TMANFinalizeA8`

所以应该以构图代码为准，而不是以文件最上面的旧注释为准。

## 7.2 测试如何准备权重和 scale

测试不是直接喂简单矩阵，而是专门把数据重排成“旧 TMAN fp16 路径兼容格式”：

- `pack_old_fp16_qweight`
  把逻辑上的 4bit 权重编码成 `qweight` 所需的旧 packed 布局

- `pack_old_fp16_scales`
  把每个输出通道、每个 group 的 scale 打包成 `scales` 所需布局

这也说明当前 `TMANLinearA8` 并没有定义全新的权重存储格式，而是在输入激活 A8 化之后，仍然复用旧 linear 核心所要求的权重打包布局。

## 7.3 测试的 golden/reference 数值路径

`prepare_case` 里构造 golden 的逻辑与 runtime 刻意保持一致：

1. 根据输入 `x` 统计 `amax`
2. 生成动态 `s_a_dynamic = amax / 127`
3. 再乘 `input_scale_factor` 得到这次 case 使用的 `input_scale`
4. 用这个 `input_scale` 把 `x` 量化成 `q_a`
5. 用 `q_a * s_a` 与 GPTQ 风格权重、group scale 做浮点 reference
6. 显式先过一次 `fp16` 舍入
7. 再按 `output_scale` 做一轮输出量化反量化

最终 golden 为：

```text
acc
-> fp16 rounding
-> round_clip(acc_fp16 / output_scale)
-> dequant by output_scale
-> fp16
```

这和运行时的真实设计完全对应：

- `PrecomputeA8` 负责输入量化
- `LinearA8` 负责 LUT + bit-serial 主计算
- `FinalizeA8` 负责最终输出量化反量化

## 7.4 测试覆盖了哪些场景

`get_cases()` 里目前覆盖了这些重点 case：

- 基线 case：`K256 / M128`
- `zero_input`
- `zero_weight`
- 多 group：`K512 / M128`
- `output_scale` 生效场景
- autosplit 场景：`M256`
- 更大 shape：
  - `M512 K2048`
  - `M128 K6144`
  - `M1024 K2048`
  - `M2048 K2048`
  - `M2048 K6144`
  - `M6144 K2048`

因此测试既在验证数值语义，也在验证大 shape、分块执行和外部 scale 的接入是否正确。

## 8. 一句话总结当前实现

当前 `TMANOpPackageW4A8` 的本质，是一条“尽量复用旧 TMAN FP16/W4 主干”的正式 W4A8 过渡实现：

- 输入侧：引入外部 `input_scale`，把激活约束到 A8 语义
- 计算侧：继续复用旧的 `hvx_lut_ctor + hvx_tbl + hvx_bit_serial`
- 输出侧：新增 `TMANFinalizeA8`，把 `output_scale` 纳入链路，但层间接口仍保持 fp16

所以它不是一个完全重写的新 kernel，而是一个“激活 A8 化 + 输出量化反量化”包裹下的旧 TMAN 核心复用方案。
