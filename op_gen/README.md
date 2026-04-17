# W4A8 TMAN on ExecuTorch + Qualcomm HTP: Engineering README
你是深度学习和AI infra领域的资深专家，包括llm架构、模型推理优化等，精通executorch、llama.cpp等端侧推理框架，精通高通NPU等端侧平台。
该项目基于executorch实现了LLM高通NPU量化推理，目前linear算子改为了查找表实现进行计算W4A16加速。在python侧对权重采用位串行排布。查找表的做法是，预先计算出所有可能的低比特权重与激活值的乘积结果，并存入查找表。推理时直接通过查表获得结果，跳过了逐元素反量化和浮点运算。
在Hexagon DSP侧编写了高度优化的HVX内核代码，专门用于实现位串行LUT查找和高效的反量化算子。在backends/qualcomm/runtime/op_packages/TMANOpPackage/include/hvx_funcs.h和backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/下的TMANPrecompute.cpp、TMANLinear.cpp和TMANFinalize.cpp。以example中的qwen3.py为例，分析代码从python侧逐步执行到底层的流程。
目前，我们需要对该项目的linear算子进行INT8改造，实现W4A8推理，加速模型的推理计算。linear层的激活量化采用对称per-tensor量化，输入输出的scale由外部的pt文件导入，包含所有linear层的input_scale和output_scale，文件地址为/home/lmx/Projects/tman/models/Qwen3-1.7B-GPTQ-Int4下的activation_scales_a8_per-tensor_sym.pt和model.safetensors。需要注意的是，其它的算子仍采用原来的A16推理方式，衔接linar的INT8量化和其它算子的FP16计算。

## 1. 目标

本文档给出 **基于 T-MAN 当前源码结构** 的 `W4A8` 工程化落地方案。目标不是重新设计一套全新的算子栈，而是尽量复用现有 `TMANOpPackage` 的三段式执行路径，在 **ExecuTorch → QNN → Qualcomm HTP/NPU** 这条链路上，把当前 `W4A16`/`W2A16` 风格的 LUT-decoding 实现扩展为：

- **Weight**：`W4`，`per-group symmetric`
- **Activation**：`A8`，`per-tensor symmetric`
- **Layer interface**：`FP16 in / FP16 out`

这样做的原则是：

1. **不改变层间接口**，方便前后衔接 FP16 的 RMSNorm / SiLU / GELU / residual add。
2. **保留 T-MAN 的三算子拆分**：`TMANPrecompute -> TMANLinear -> TMANFinalize`。
3. **最大化复用现有 HVX/VLUT16 内核**，减少对 ExecuTorch lowering、QNN op package 和 HTP 调度的冲击。
4. **先做最小改动可运行版本**，再考虑进一步把 A8 激活表项压得更激进。

---

## 2. 现有源码的关键结论

### 2.1 当前实现就是一个标准的 QNN HTP OpPackage

源码中注册了三个自定义算子：

- `TMANPrecompute`
- `TMANLinear`
- `TMANFinalize`

它们在 `TMANOpPackageInterface.cpp` 中完成注册，并通过 `Makefile` 编译成 `libQnnTMANOpPackage.so`，目标包括：

- `hexagon-v68/v69/v73/v75/v79`
- `aarch64-android`
- `x86_64-linux-clang`

这说明当前方法的真实部署入口确实是：

`ExecuTorch graph -> QNN custom op -> HTP/HVX kernel`

---

### 2.2 当前 decoding 路径已经被拆成三段

从源码看，三段功能边界非常清晰：

#### `TMANPrecompute`

输入 `x(fp16)`，输出一个**打包缓冲区**，内部包含：

- `l`：量化后的 LUT 表项，`int16`
- `ls`：LUT scale，`float`
- `lb`：LUT bias，`float` 或 `fp16`

也就是说，`Precompute` 的输出不是单纯的 LUT，而是一个 **packed precompute buffer**。

#### `TMANLinear`

输入：

- `l`（上一步 packed buffer）
- `qweight`
- `scales`

输出：

- `c`：`float` scratch buffer

`Linear` 内部已经做了：

- `VLUT16` 查表
- `int32` 聚合
- 乘 `ls`
- 加 `lb`
- 乘权重 scale `s`

因此 `TMANLinear` 输出的并不是最终 `y`，而是**按 bit-plane 展开的浮点部分结果**。

#### `TMANFinalize`

输入 `c(float scratch)`，输出 `y(fp16)`。

内部只做一件事：

- 按 bit-serial 规则合并 bit-plane
- cast 到 `fp16`

对于 `Bits=4`，源码里的最终加权是：

- `bit0 * 0.5`
- `bit1 * 1`
- `bit2 * 2`
- `bit3 * 4`

然后输出 `fp16`。

---

### 2.3 当前 W4 symmetric 现成特化只支持 `group_size = 128`

在 `TMANPrecompute.cpp` / `TMANLinear.cpp` 的分发逻辑里，当前源码已经显式实例化的 `W4 symmetric` 路径是：

- `bits == 4`
- `group_size == 128`
- `symmetric == true`

没有现成的 `W4 symmetric + group_size == 64` 分支。

因此，**W4A8 第一版工程建议直接沿用 `group_size = 128`**，这样可以最大化复用现有模板与 dispatch 逻辑。

---

### 2.4 当前实现里 `symmetric` 参数和 `zero_point` 的判断是反向映射的

源码里使用的是：

```cpp
const bool zero_point = ((const int32_t*)t_symmetric.raw_data_const())[0] == 0;
```

这意味着：

- `symmetric = 0`  -> `zero_point = true`
- `symmetric = 1`  -> `zero_point = false`

所以对于我们新的 **symmetric W4A8** 版本，运行图中应该传：

```text
symmetric = 1
```

不能按名字想当然。

---

### 2.5 当前实现强依赖 `VLUT16 + int16 LUT entry`

`hvx_tbl` 中使用的是 `Q6_Wh_vlut16_VbVhR_nomatch`，也就是 **VLUT16**。

因此现有内核天然假设：

- LUT 索引是 4bit/8bit 形式的 byte lane
- LUT entry 是 **16-bit**

这对 `W4A8` 很关键：

> 我们即便把 activation 降到 A8，也不应该第一版就把 LUT entry 也降成 int8。

为了最大化复用现有 HVX kernel，**W4A8 v1 继续保留 `int16 LUT entry`**，只是表项的数值来源改成 A8 量化后的激活。

---

## 3. W4A8 v1 的总体设计

## 3.1 设计原则

`W4A8 v1` 的整体思路是：

- **层外保持 FP16**
- **层内改成 A8 激活参与 LUT 计算**
- **尽量不动 TMANLinear / TMANFinalize 的主框架**

因此整个数据流是：

```text
FP16 input
  -> TMANPrecomputeA8
  -> TMANLinearA8
  -> TMANFinalize
  -> FP16 output
```

其中：

- `TMANFinalize` 可以直接复用现有代码
- `TMANLinear` 的框架也尽量复用
- 主要改动集中在 `Precompute`，以及 `Linear` 中对 `ls/lb` 语义的重解释

---

## 3.2 推荐的第一版目标配置

为了让第一版更容易落地，推荐直接固定为：

- `bits = 4`
- `group_size = 128`
- `symmetric = 1`
- `activation = per-tensor symmetric A8`

也就是一个明确的：

```text
W4G128A8, symmetric weight, symmetric activation
```

这样可以减少模板实例化数量，也方便先把 graph lowering 和 runtime 跑通。

---

## 4. 三阶段的工程职责

# 4.1 Precompute 阶段

## 4.1.1 目标

把输入的 `FP16 activation` 转换成 `TMANLinear` 可直接消费的 packed buffer。

对 `W4A8` 而言，Precompute 是**主要改造点**。

---

## 4.1.2 现有源码行为

当前 `TMANPrecompute`：

1. 从 `x(fp16)` 中抽取 `4` 个 activation 为一组（`LUT_G = 4`）
2. 构造 16-entry LUT
3. 用 `int16` 存储 LUT 表项
4. 为每个 activation group 计算 `ls`
5. 为每个 weight group 计算 `lb`
6. 把 `l + ls + lb` 打包写到同一个输出 buffer

其中常量是：

- `ACT_GROUP_SIZE = 256`
- `LUT_G = 4`
- `LUT_SIZE = 16`

---

## 4.1.3 W4A8 的改造目标

对于 `W4A8`，Precompute 不再把 `FP16 x` 直接映射成“浮点来源的 int16 LUT”，而是改成：

1. **先做 per-tensor symmetric A8 量化**
2. 再用 `q_a(int8)` 构造 LUT
3. 继续输出 `int16` LUT entry
4. `ls` 不再表示“运行时表项归一化尺度”，而表示 **activation dequant scale `s_a`**
5. `lb` 保留，但改为基于 `q_a` 的整数和来构造

---

## 4.1.4 推荐实现方式

### Step A：计算 per-tensor 激活 scale

对当前 token 的整条输入向量 `x_fp16` 计算：

```text
amax = max(abs(x_fp16))
s_a  = amax / 127
```

若 `amax == 0`，则令 `s_a = 1`，并生成全零 LUT。

### Step B：量化成 int8

```text
q_a[k] = clip(round(x_fp16[k] / s_a), -127, 127)
```

这里建议使用对称范围 `[-127, 127]`，避免边界不一致带来的额外处理。

### Step C：按现有 `LUT_G = 4` 组织 LUT

继续沿用现有 4 个 activation 一组的 LUT 构造方式，保持对 `VLUT16` 的适配。

关键点是：

- **LUT 索引方式不变**
- **LUT entry 存储类型不变（int16）**
- **LUT 的数值来源改为 `q_a`**

因为 4 个 `int8` 相加的动态范围约为 `[-508, 508]`，完全能安全落在 `int16` 中，因此第一版没必要再做一次 LUT 内部 re-quant。

### Step D：`ls` 的新语义

现有实现里 `ls` 是“LUT quantization scale”。

对 `W4A8 v1`，建议把 `ls` 直接改成：

```text
ls = s_a
```

并且因为 activation 是 **per-tensor** 量化，`ls` 实际上对整个 token 只有一个值。为了兼容现有 `TMANLinear` 的索引方式，最简单做法是：

- 在 `ls` 存储区中把同一个 `s_a` **广播填充** 到所有会被访问的位置
- 不修改 `hvx_tbl` 对 `ls[(tile_q + vec_q) / q_act_group_size]` 的取值逻辑

这样可以做到：

- 不改 `ls_ptr` 的寻址方式
- 不改 `TMANLinear` 的向量化主体
- 只改 Precompute 的写值逻辑

### Step E：`lb` 的新语义

现有源码在 symmetric 路径下仍然保留 `lb`，并在 `TMANLinear` 中只对某个 bit-lane、每个 weight group 加一次。

这条路径不要删。

对于 `W4A8 v1`，建议继续沿用这个结构，只是把 `lb` 的来源改成基于 `q_a` 的整数 partial sum：

```text
lb = -0.5 * sum(q_a[group])
```

然后在 `Linear` 里仍按现有方式：

- 在 `q_group_tail`
- 且 `vec_p % (VecP * Bits) == VecP`

的时机加到对应 bit-plane。

**结论**：

Precompute 的输出布局仍保持：

```text
[lut:int16][ls:float][lb:float]
```

只是三者的来源变成了 A8 版本。

---

# 4.2 Linear 阶段

## 4.2.1 目标

保持 `TMANLinear` 的查表-聚合主框架不变，让它继续负责：

1. 从 `qweight` 中取 packed nibble 索引
2. 用 `VLUT16` 查 `lut:int16`
3. 做 `int32` 聚合
4. 应用 `ls`
5. 应用 `lb`
6. 乘 per-group 权重 scale `s_w`
7. 输出 `float` scratch `c`

---

## 4.2.2 现有源码中已经做了什么

`hvx_tbl` 的当前逻辑可以概括成：

1. **拆 nibble**：从 `qweight` 中拆出低 4bit / 高 4bit
2. **VLUT16 查表**：四路查表，对应不同 `Q` lane
3. **cmp block 内 int32 聚合**
4. 到 group/block 尾部时：
   - `* ls`
   - `+ lb`
   - `* s`
5. 把结果累计到 `c(float)`

这意味着：

> `TMANLinear` 不是一个“纯整数 dot-product”算子，而是一个“已经做完 scale/bias 处理的 per-bit partial-output producer”。

---

## 4.2.3 W4A8 下的最小改法

`Linear` 第一版尽量少改，只做下面几件事：

### 改动 1：继续使用现有 `VLUT16`

因为 Precompute 仍输出 `int16 LUT entry`，所以 `Q6_Wh_vlut16_VbVhR_nomatch` 可以原样复用。

### 改动 2：把 `ls` 理解为 activation scale `s_a`

现有逻辑：

```text
int32_acc * ls
```

在 `W4A8` 中，这一步就等价于：

```text
(sum over q_a) * s_a
```

因此 `TMANLinear` 主体不需要知道 activation 已经是 A8，它只需要知道 `ls` 现在是 A8 的反量化尺度。

### 改动 3：继续保留 `lb` 处理路径

现有 symmetric 路径会在每个 weight group 的特定时机加 `lb`。

这条逻辑不删，只把 `lb` 替换成 A8 版 Precompute 计算出的值。

### 改动 4：继续在 Linear 内部乘 `s_w`

这样 `TMANFinalize` 可以保持完全不变。

即：

```text
c = per-bit partial output in float domain
```

而不是输出原始 int32 accumulator。

---

## 4.2.4 建议保持不变的内核常量

为了降低风险，建议第一版保持下面这些常量不变：

- `LUT_G = 4`
- `LUT_SIZE = 16`
- `TILE_K = 256`
- `LType = int16`
- `CType = float`

这样做的原因是：

- `LUT_G = 4` 与 nibble/VLUT16 完全对齐
- `TILE_K = 256` 已经和当前 `hvx_tbl` 模板及 tiling 假设对齐
- `int16 LUT entry + float c` 是目前最稳的数值路径

---

# 4.3 Finalize 阶段

## 4.3.1 目标

把 `TMANLinear` 产出的 per-bit `float scratch` 合成为最终 `fp16 output`。

---

## 4.3.2 现有源码行为

`hvx_bit_serial<4>` 读取 `c` 中的 4 个 bit-plane，对它们做：

```text
bit0 * 0.5 + bit1 * 1 + bit2 * 2 + bit3 * 4
```

然后把结果转换为 `fp16` 输出。

这一步本质上完成了现有 TMAN 内部定义的 4-bit bit-serial 还原。

---

## 4.3.3 W4A8 的实现建议

**完全复用现有 `TMANFinalize.cpp` 与 `hvx_bit_serial<4>`。**

原因是：

- A8 的改动已经在 `Precompute` 和 `Linear` 中完成
- `Finalize` 不关心激活来自 FP16 还是 A8
- 只要 `Linear` 输出的 `c` 语义不变，`Finalize` 就不需要动

---

## 5. 推荐的工程改动清单

## 5.1 新增一个 W4A8 专用分支，而不是直接覆盖旧实现

建议不要一上来把现有 `TMANPrecompute/TMANLinear` 的所有路径混在一起改。

更稳的方式是：

### 方案 A：新增专用 op 名称

- `TMANPrecomputeA8`
- `TMANLinearA8`
- `TMANFinalize`（复用）

优点：

- graph 更清晰
- 不会影响原 W4A16/W2A16 路径
- 便于 A/B 测试

### 方案 B：保留 op 名称，内部加开关

增加一个新的 scalar param，例如：

- `act_bits = 8/16`

但这会同步修改：

- XML
- 参数校验
- ExecuTorch lowering
- runtime param 传递

对第一版来说更重。

**推荐先用方案 A。**

---

## 5.2 PrecomputeA8 需要做的实际代码改动

1. 复制 `TMANPrecompute.cpp` 为 `TMANPrecomputeA8.cpp`
2. 保留当前的 `ACT_GROUP_SIZE/LUT_G/LUT_SIZE`
3. 在 `hvx_lut_ctor` 之前增加：
   - `amax` 计算
   - `s_a` 计算
   - `fp16 -> int8` 量化
4. 新写一个 `hvx_lut_ctor_a8(...)`，输入改为 `int8 q_a`
5. 输出 packed buffer 仍然组织为：
   - `lut:int16`
   - `ls:float`（填 `s_a`）
   - `lb:float`
6. 只支持：
   - `bits == 4`
   - `group_size == 128`
   - `symmetric == 1`

---

## 5.3 LinearA8 需要做的实际代码改动

1. 复制 `TMANLinear.cpp` 为 `TMANLinearA8.cpp`
2. 保留 `hvx_tbl<..., Bits=4, GroupSize=128, ZeroPoint=false>` 主模板
3. 继续使用：
   - `VLUT16`
   - `int32` 中间累加
   - `float c`
4. 不改 `Finalize` 所需的 `c` 布局
5. 只调整注释和语义：
   - `ls` 不再是 LUT re-quant scale，而是 `s_a`
   - `lb` 来自 A8 量化后的 partial sum

如果希望代码上更清楚，建议把现有 `hvx_tbl` 再包一层：

```cpp
hvx_tbl_w4a8_sym_g128(...)
```

而不是直接在老模板里塞过多条件分支。

---

## 5.4 Finalize 保持不动

`TMANFinalize.cpp` 与 `hvx_bit_serial<4>` 第一版完全不改。

只要：

- `TMANLinearA8` 的输出 `c`
- bit-plane 的排列方式
- `Bits=4`

都和现有实现一致，`Finalize` 就能直接复用。

---

## 6. ExecuTorch 侧的图改写建议

## 6.1 图模式

对于一个线性层：

```text
x_fp16 -> linear(weight_w4, scale) -> y_fp16
```

在 lowering 后改写成：

```text
x_fp16
  -> TMANPrecomputeA8
  -> TMANLinearA8(qweight, scales)
  -> TMANFinalize
  -> y_fp16
```

---

## 6.2 QKV / gate-up 共享输入时的复用

如果同一个 `x_fp16` 同时喂给多个线性层，例如：

- `Q/K/V`
- `gate/up`

建议复用同一个 `TMANPrecomputeA8` 输出，再接多个 `TMANLinearA8`。

即：

```text
x_fp16
  -> TMANPrecomputeA8
      -> TMANLinearA8(Wq) -> TMANFinalize -> q_fp16
      -> TMANLinearA8(Wk) -> TMANFinalize -> k_fp16
      -> TMANLinearA8(Wv) -> TMANFinalize -> v_fp16
```

这样能最大程度继承 T-MAN 原本的“shared precompute”优势。

---

## 7. 内存与部署注意事项

## 7.1 Precompute 输出仍然是单一 packed buffer

这点不要改成多个 tensor 输出。

原因：

- 当前 `TMANLinear` 假设 `l/ls/lb` 连续排布
- 改成多输出会同步影响 XML、validate、graph lowering 与内核访存

因此工程上建议继续使用：

```text
packed_precompute = [lut][ls][lb]
```

---

## 7.2 Finalize 输出继续放 MainMemory

`TMANFinalize.cpp` 中已经写明：

- 若把最终输出强行放 `Tcm`
- 会与 embedding/Gather 输出的 `MainMemory` 产生 add 冲突

因此 W4A8 版本建议继续保持：

- `Linear` 内部 TCM 优先
- `Finalize` 输出回 `MainMemory`

不要在第一版里强行改这一点。

---

## 7.3 编译环境

从 `Makefile` 可知，编译该 package 需要至少准备：

- `QNN_INCLUDE`
- `QNN_TARGET_LIB`
- `HEXAGON_SDK_ROOT`
- `ANDROID_NDK_ROOT`

常用构建目标：

```bash
make htp_v79
make htp_aarch64
```

在 Snapdragon 8 Elite / 8 Gen 3 一类平台上，建议优先验证：

- `hexagon-v79` 或匹配芯片版本的 HTP 目标
- `aarch64-android` 端侧装载链路

---

## 8. 推荐的验证顺序

## Phase 1：功能跑通

先固定：

- `W4G128A8`
- `symmetric=1`
- `N=1 decoding`

验证：

1. 单层输出数值与 CPU reference 对齐
2. Precompute buffer 不越界
3. `TMANFinalize` 输出维度正确

---

## Phase 2：误差验证

对比基线：

- FP16 Linear
- 现有 W4A16 TMAN 路径
- 新 W4A8 路径

观察：

- 单层 MSE / cosine similarity
- 若接完整 block，观察 residual 后误差传播

---

## Phase 3：性能验证

重点观察三部分耗时：

1. `PrecomputeA8`
2. `LinearA8`
3. `Finalize`

并与现有 `W4A16` 进行对比：

- Precompute 是否因 A8 量化增加额外开销
- Linear 是否因 `ls` 变成全局 activation scale 而减少表项量化成本
- End-to-end decoding tokens/s 是否提升

---

## 9. 结论：第一版最稳的落地方式

基于当前源码，`W4A8` 的最稳工程路径不是推翻现有三段结构，而是：

1. **保留 `Precompute -> Linear -> Finalize` 三段拆分**
2. **把主要改动集中到 Precompute**
3. **Linear/Finalize 尽量复用原有 HVX/VLUT16 实现**
4. **第一版只做 `W4G128A8 + symmetric`**
5. **层外接口保持 FP16，不把 A8 扩散到图外**

一句话概括：

> 用 A8 激活替换现有 W4A16 路径中的激活来源，但不改变 TMAN 的三算子工程骨架、不改变 bit-serial Finalize 规则、不改变 ExecuTorch 到 QNN HTP 的接线方式。

这条路线代码改动集中、风险最低，也最容易先得到一个可验证、可部署、可继续迭代的 W4A8 版本。