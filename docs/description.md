你是深度学习和AI infra领域的资深专家，包括llm架构、模型推理优化等，精通executorch、llama.cpp等端侧推理框架，精通高通NPU等端侧平台
该项目基于executorch实现了LLM高通NPU量化推理，目前linear算子改为了查找表实现进行计算W4A16加速。在python侧对权重采用位串行排布。查找表的做法是，预先计算出所有可能的低比特权重与激活值的乘积结果，并存入查找表。推理时直接通过查表获得结果，跳过了逐元素反量化和浮点运算
在Hexagon DSP侧编写了高度优化的HVX内核代码，专门用于实现位串行LUT查找和高效的反量化算子。在backends/qualcomm/runtime/op_packages/TMANOpPackage/include/下的hvx_funcs.h和backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/下的TMANPrecompute.cpp、TMANLinear.cpp和TMANFinalize.cpp。以examples/qualcomm/oss_scripts/qwen3/qwen3.py为例，分析代码代码从python侧逐步执行到底层的流程
目前，我们需要对该项目的linear算子进行INT8改造，实现W4A8推理，加速模型的推理计算。linear层的激活量化采用对称per-tensor量化，输入输出的scale由外部的pt文件导入，包含所有linear层的input_scale和output_scale，文件地址为/home/lmx/Test/GPTQ_Opus/Qwen3-1.7B_w4a8_gptq/activation_scales_a8_per-tensor_sym.pt。需要注意的是，其它的算子仍采用原来的A16推理方式，衔接linar的INT8量化和其它算子的FP16计算

补充说明（截至当前代码状态）：

- `W4A8` 专用支线已经实现到 Python 模块、export/custom op、QNN lowering、TMAN OpPackage 与 HVX helper 层；
- 当前工作的重点已从“补齐功能缺口”转为“继续收紧 A8 的 static tensor ABI、对齐策略与设备侧可运行性”；
- 目前能直接代表 `W4A8` 真实设备现场的证据只有 `logs/w4a8/` 下由合作者提供的运行日志；
- `logs/w4a16/` 下的历史日志以及旧 `.pte` / `quant_attrs` 产物只能作为旧路径参考，不能直接当作当前 `W4A8` 图或当前设备运行状态的证据。
