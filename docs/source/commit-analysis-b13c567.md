# Commit b13c567 分析：实现 TMANAdd 算子

- **Commit:** b13c567a5953dda8f5bfa6b44460e381af4d5893
- **作者:** tang-jiapeng
- **日期:** 2026-03-11
- **提交信息:** Impl TMANAdd op

## 概述

本次提交为 Qualcomm HTP（Hexagon Tensor Processor）后端的 TMAN 算子包新增了一个自定义的 **TMANAdd** 算子，用于在 NPU 上执行 fp16 元素级加法。核心目的是解决 TMANFinalize 输出与 Embedding 输出之间的**内存层级冲突**，使 TMANFinalize 的输出能够留在高速片上缓存（TCM）而非被迫写回 DDR，从而提升推理性能。

## 变更文件（共 7 个文件，+599/-232 行）

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `.gitignore` | 修改 | 添加 `.vscode/` 到忽略列表 |
| `backends/qualcomm/builders/qnn_constants.py` | 修改 | 新增 `OpTMANAdd` 常量类 |
| `backends/qualcomm/builders/op_add.py` | 重构 | 加法节点分发逻辑，新增 TMANAdd 路径 |
| `backends/qualcomm/builders/op_tman_linear.py` | 修改 | 调整中间张量命名与 wrapper_idx，配合 TMANAdd 取用 |
| `TMANOpPackage/src/ops/TMANAdd.cpp` | 新增 | HVX 优化的 fp16 加法内核实现 |
| `TMANOpPackage/src/TMANOpPackageInterface.cpp` | 修改 | 注册 TMANAdd 算子到包接口 |
| `TMANOpPackage/src/ops/TMANFinalize.cpp` | 修改 | 输出从 MainMemory 改为 Tcm |

## 核心设计动机

在此提交之前，推理管线中存在一个性能瓶颈：

1. `TMANLinear` → `TMANFinalize` 的中间结果本应留在 TCM（片上高速缓存，访问延迟 ~1 cycle）。
2. 但后续的 `aten.add`（残差连接）需要将此结果与 Embedding/Gather 的输出相加。
3. Embedding 输出位于 DDR（主存），QNN 标准 Add 算子要求两个输入在同一内存层级。
4. 因此 TMANFinalize 被迫将输出放到 MainMemory，丧失了 TCM 的低延迟优势。

**TMANFinalize.cpp 的注释原文**正记录了这个问题：
> *Tcm("y") results in [ERROR] could not create op: q::Add.tcm*
> *Reason: embedding (Gather) outputs are in MainMemory but TMANLinear outputs are in Tcm*
> *TODO: implement custom TMANOpPackage::Add*

本提交正是实现了这个 TODO。

## 各模块详细分析

### 1. 常量定义 (`qnn_constants.py`)

新增 `OpTMANAdd` dataclass：

```python
@dataclass(init=False, frozen=True)
class OpTMANAdd:
    op_name: str = "TMANAdd"
```

为 Python 端图构建器提供算子名称常量。

### 2. 图构建器分发逻辑 (`op_add.py`)

这是 Python 端变更最大的文件。重构了 `Add` 节点访问器的 `define_node` 方法：

- **新增 `_is_tman_input()`**：检测输入节点是否来自 TMAN 线性算子（`tman.linear.default` 或 `tman.bitnet_linear.default`）。
- **分发逻辑**：
  - 若两个加法输入中**恰好有一个**来自 TMAN 算子 → 走 TMANAdd 路径。
  - 否则（两个都不是，或两个都是）→ 走标准 QNN ElementWiseAdd。
- **TMANAdd 路径**：
  1. 从 TMAN 节点的 `nodes_to_wrappers` 中获取 `wrapper_idx=1` 的中间 fp16 张量。
  2. 如果另一个输入不是 fp16，插入 `Convert` 算子转换为 fp16。
  3. 创建 `TMANAdd` 算子，输入为 TMAN 的 fp16 中间结果和转换后的另一输入。
  4. 如果最终输出不是 fp16，再插入一个 `Convert` 算子做输出类型转换。

### 3. TMAN Linear 适配 (`op_tman_linear.py`)

两个关键改动使 TMANAdd 能正确获取 TMANFinalize 的 fp16 中间输出：

- 将 `intermediate_output_tensor_wrapper` 的 `node_name` 从 `node.name + "_output_converted"` 改为 `node.name`。
- 新增 `wrapper_idx=1` 参数，使该张量以索引 1 存入 `nodes_to_wrappers[node.name]`，供下游 `op_add.py` 通过 `.get(1, None)` 获取。

其余为代码格式化（import 排序、长行拆分）。

### 4. HVX 内核实现 (`TMANAdd.cpp`)

新文件，162 行。实现了 HVX SIMD 优化的 fp16 加法内核：

**内存布局约束**（通过 `DEF_TENSOR_PROPERTIES`）：
- 输入 `a`（来自 TMANFinalize）：**Tcm**（片上缓存）
- 输入 `b`（来自 Embedding/Gather）：**MainMemory**（DDR）
- 输出 `*`：**Tcm**

**性能优化策略**：

| 优化手段 | 细节 |
|----------|------|
| L2 预取 | 初始预取 b 的前 4KB（8 chunks），主循环中滚动预取后续 4KB |
| 4x 循环展开 | 每次迭代处理 256 个 fp16 元素（4 个 HVX 向量） |
| 交错 load/compute | 每对 {a[k], b[k]} 加载后立即计算+存储，降低寄存器压力 |
| HVX fp16 加法 | 使用 `Q6_Vqf16_vadd_VhfVhf`（fp16→qf16 加法→fp16），V75 验证可靠路径 |
| 尾部处理 | 利用 QNN 张量分配的向量对齐 padding 安全过读，通过临时缓冲区 + memcpy 只写有效元素 |

**数据流**：
```
DDR(b) --L2prefetch--> L2 --load--> HVX Reg
                                        |
TCM(a) ----------load---------------> HVX Reg  → vadd → store → TCM(out)
```

### 5. 算子包注册 (`TMANOpPackageInterface.cpp`)

- `sg_opNames` 数组从 3 个扩展到 4 个，添加 `"TMANAdd"`。
- 添加 `DECLARE_PKG_OPS_OPTS_LIST(PKG_TMANAdd)` 声明。
- 在 `TMANOpPackageValidateOpConfig` 中添加 TMANAdd 验证规则：
  - 0 个参数（无标量参数）
  - 2 个输入
  - 1 个输出

其余为代码格式化（缩进统一为 2 空格，花括号位置调整）。

### 6. TMANFinalize 输出位置变更 (`TMANFinalize.cpp`)

**关键改动**：`DEF_TENSOR_PROPERTIES` 中输出 `*` 从 `MainMemory` 改为 `Tcm`：

```cpp
// 修改前
MainMemory("*", "group_size", "bits", "symmetric"),
Tcm("c")

// 修改后
MainMemory("group_size", "bits", "symmetric"),
Tcm("*", "c")
```

这使得 TMANFinalize 的输出留在 TCM，不再写回 DDR。由新的 TMANAdd 算子负责处理与 DDR 侧输入的加法。

其余为代码格式化。

## 整体架构影响

### 修改前的数据流
```
TMANLinear(Tcm) → TMANFinalize → 输出到 DDR → QNN Add(DDR, DDR) → DDR
                                                    ↑
                                          Embedding(DDR) ──┘
```

### 修改后的数据流
```
TMANLinear(Tcm) → TMANFinalize → 输出留在 Tcm → TMANAdd(Tcm, DDR) → Tcm
                                                      ↑
                                            Embedding(DDR) ──┘
```

**性能收益**：
- 避免了 TMANFinalize 输出从 TCM 写回 DDR 的开销。
- TMANAdd 输出留在 TCM，下游算子可直接使用片上数据。
- 通过 L2 预取和交错计算，有效隐藏了 DDR 侧输入的访问延迟。
