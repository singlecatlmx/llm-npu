# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, cast

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch
from executorch.backends.qualcomm.utils.tman_w4a8 import (
    build_w4a8_lut_scratch_packed_shape,
    infer_gptq_tensor_layout,
    pack_w4a8_effective_scales_for_qnn_int32,
    pack_w4a8_weight_lut_indices_for_qnn_int32,
    prepare_w4a8_baseline_runtime_tensors,
    TMAN_A8_TARGET_QGROUPS_PER_GROUP,
    TMAN_A8_TARGET_QGROUP_VECTOR_BLOCKS,
    validate_w4a8_target_config,
)
from executorch.backends.qualcomm.utils.constants import (
    QCOM_DATA,
    QCOM_DTYPE,
    QCOM_OFFSET,
    QCOM_QUANT_MAX,
    QCOM_QUANT_MIN,
    QCOM_SCALE,
)

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import (
    OpConvert,
    OpTMANFinalizeA8,
    OpTMANLinearA8,
    OpTMANPrecomputeA8,
    QNN_OP_PACKAGE_NAME_QTI_AISW,
    QNN_OP_PACKAGE_NAME_TMAN,
)
from .utils import get_parameter


def _make_scale_offset_quant_config(scale: float) -> Dict:
    # 这里构造的是 QNN 的 per-tensor scale/offset 量化配置。
    # W4A8 采用对称 INT8，因此 offset 固定为 0，范围固定在 [-127, 127]。
    return {
        QCOM_SCALE: float(scale),
        QCOM_OFFSET: 0,
        QCOM_DTYPE: torch.int8,
        QCOM_QUANT_MIN: -127,
        QCOM_QUANT_MAX: 127,
    }


def _get_num_rows(shape: torch.Size) -> int:
    num_rows = 1
    for dim in shape[:-1]:
        num_rows *= int(dim)
    return num_rows


@register_node_visitor
class TMANLinearA8(NodeVisitor):
    target = ["tman.linear_a8.default"]

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[str, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        # 这个 lowering 的职责可以概括成一句话：
        # 把 export 图里的 `tman::linear_a8`
        # 变成
        #   FP16 -> INT8 Convert -> TMANPrecomputeA8 -> TMANLinearA8
        #   -> TMANFinalizeA8 -> INT8 -> FP16 Convert
        # 这样一条显式的 QNN 图序列。
        #
        # 其中：
        # - 图边界量化/反量化由 QNN Convert 表达；
        # - TMAN op 只消费已经准备好的 INT8 激活、packed qweight 和 folded scales；
        # - 周围非 linear 图继续保持 FP16 语义。
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        qweight_node = node.args[1]
        qweight_tensor = get_parameter(qweight_node, self.edge_program)
        scales_node = node.args[2]
        scales_tensor = get_parameter(scales_node, self.edge_program)
        qzeros_node = node.args[3]
        qzeros_tensor = get_parameter(qzeros_node, self.edge_program)
        input_scale_node = node.args[7]
        input_scale_tensor = get_parameter(input_scale_node, self.edge_program)
        output_scale_node = node.args[8]
        output_scale_tensor = get_parameter(output_scale_node, self.edge_program)
        group_size = cast(int, node.args[9])
        bits = cast(int, node.args[10])
        symmetric = cast(bool, node.args[11])
        gptq_v2 = cast(bool, node.args[12])

        try:
            validate_w4a8_target_config(bits, group_size, symmetric)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        # 这里补回张量级一致性校验，避免 builder 侧和模块侧对 GPTQ 配置的理解不一致。
        layout = infer_gptq_tensor_layout(
            qweight_tensor,
            scales_tensor,
            qzeros_tensor,
            gptq_v2,
        )
        # 这里不是“重复校验”，而是 builder/runtime 一致性的关键断点。
        # 如果模块层保存的 group_size/bits/symmetric 和实际张量推断结果不一致，
        # 后面的 weight unpack / packed ABI / runtime 校验都会变得不可相信。
        if layout.bits != bits or layout.group_size != group_size or layout.symmetric != symmetric:
            raise RuntimeError(
                "TMANLinearA8 lowering config mismatch: "
                f"layout(bits={layout.bits}, group_size={layout.group_size}, symmetric={layout.symmetric}) "
                f"!= node(bits={bits}, group_size={group_size}, symmetric={symmetric})"
            )

        input_scale = float(input_scale_tensor.detach().reshape(-1)[0].item())
        output_scale = float(output_scale_tensor.detach().reshape(-1)[0].item())
        # 第一阶段先走“正确性优先”的 baseline lowering：
        # - host 侧直接下发 unpack 后的 uint8 权重；
        # - 把 weight_scale、input_scale、output_scale 折叠成 effective_scale；
        # - runtime A8 baseline 再消费这些 static tensor。
        #
        # 后续即使把这里替换成真正的 HVX-friendly packed layout，
        # 也不需要改 export op ABI，只需要替换 builder/runtime 的实现。
        qweight_lut_indices_mgqb, effective_scales_mg = prepare_w4a8_baseline_runtime_tensors(
            qweight_tensor,
            scales_tensor,
            input_scale_tensor,
            output_scale_tensor,
            bits,
            group_size,
        )

        num_groups = qweight_lut_indices_mgqb.shape[1]
        qgroups_per_group = qweight_lut_indices_mgqb.shape[2]
        # 当前 A8 路径已经明确是 `W4G128 sym` 目标专用实现。
        # 因此这里把 group 内 qgroup 数也显式固定下来，防止“看似 shape 对了，
        # 其实 block 组织已经偏掉”的情况悄悄流进 runtime。
        if qgroups_per_group != TMAN_A8_TARGET_QGROUPS_PER_GROUP:
            raise RuntimeError(
                "TMANLinearA8 lowering requires "
                f"qgroups_per_group={TMAN_A8_TARGET_QGROUPS_PER_GROUP}, got {qgroups_per_group}"
            )
        gemm_k = num_groups * qgroups_per_group * 4
        if gemm_k % group_size != 0:
            raise RuntimeError(
                f"TMANLinearA8 lowering requires K divisible by group_size, "
                f"got K={gemm_k}, group_size={group_size}"
            )

        num_rows = _get_num_rows(input_tensor.size())
        qweight_static = pack_w4a8_weight_lut_indices_for_qnn_int32(
            qweight_lut_indices_mgqb,
            bits,
        )
        # W4A8 的 qweight 逻辑上仍是：
        #   [1, M, G, qgroup_vector_blocks * (4 * bits bytes)]
        # 这里只是为了和旧 W4A16 静态权重的 transport 习惯更接近，
        # 把同一串原始字节 reinterpret 成 int32 容器。
        #
        # 因此这里检查的是“每个 [M,G] 行的真实字节数”仍然保持 128B。
        if (
            qweight_static.shape[3] * qweight_static.element_size()
            != TMAN_A8_TARGET_QGROUP_VECTOR_BLOCKS * bits * 4
        ):
            raise RuntimeError(
                "TMANLinearA8 lowering requires "
                "packed qweight bytes-per-[M,G]="
                f"{TMAN_A8_TARGET_QGROUP_VECTOR_BLOCKS * bits * 4}, "
                f"got {qweight_static.shape[3] * qweight_static.element_size()}"
            )
        scales_static = pack_w4a8_effective_scales_for_qnn_int32(
            effective_scales_mg
        )

        qweight_tensor_wrapper = self.define_tensor(
            qweight_node,
            node,
            qweight_static,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
            node_name=node.name + "_qweight_a8",
        )
        scales_tensor_wrapper = self.define_tensor(
            scales_node,
            node,
            scales_static,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
            node_name=node.name + "_effective_scales_a8",
        )

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        no_quant_encoding, no_quant_configs = (
            PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED,
            {},
        )
        scale_offset_encoding = (
            PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET
        )

        input_q_tensor_wrapper = self.define_custom_tensor_wrapper(
            node_name=node.name + "_input_q_a8",
            tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_8,
            quant_encoding=scale_offset_encoding,
            quant_configs=_make_scale_offset_quant_config(input_scale),
            dims=input_tensor.size(),
            tensor=None,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )
        l_tensor_wrapper = self.define_custom_tensor_wrapper(
            node_name=node.name + "_precompute_a8",
            tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_16,
            quant_encoding=no_quant_encoding,
            quant_configs=no_quant_configs,
            dims=build_w4a8_lut_scratch_packed_shape(
                num_rows=num_rows,
                num_groups=num_groups,
                qgroups_per_group=qgroups_per_group,
            ),
            tensor=None,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )
        c_tensor_wrapper = self.define_custom_tensor_wrapper(
            node_name=node.name + "_linear_a8",
            tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
            quant_encoding=no_quant_encoding,
            quant_configs=no_quant_configs,
            dims=output_tensor.size(),
            tensor=None,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )
        output_q_tensor_wrapper = self.define_custom_tensor_wrapper(
            node_name=node.name + "_output_q_a8",
            tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_SFIXED_POINT_8,
            quant_encoding=scale_offset_encoding,
            quant_configs=_make_scale_offset_quant_config(output_scale),
            dims=output_tensor.size(),
            tensor=None,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )

        # A8 路径的图边界设计是：
        # FP16输入 -> Convert到INT8 -> TMAN三段式A8 op -> Convert回FP16输出
        #
        # 这样 linear 内部真正跑的是 A8，
        # 但周围非 linear 图仍然保持当前 FP16 语义不变。
        #
        # 当前 static/scratch 的物理组织进一步显式成 block 形态：
        # - qweight: [1, M, G, 32 int32]  (raw bytes 仍是 128B / [M,G])
        # - scales : [1, M, 1, padded_num_groups] 的 int32 容器，
        #            raw bytes 仍按 float32 effective scales 解释，且每行按 128B 对齐
        # - lut    : [1, R, G, qgroup_vector_blocks * (4 * 16)]
        #
        # 逻辑上仍然是 block 结构，只是对外 ABI 压回 4D packed tensor，
        # 以便和当前 OpPackage XML / HTP 接口约束保持一致。
        #
        # 这点对交付非常关键：
        # - helper kernel 内部仍然保留 block/vector-line 语义；
        # - 但给 QNN 的接口形态保持 4D，减少 compile 阶段 schema/rank 风险。
        input_convert_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name + "_input_convert_a8",
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpConvert.op_name,
        )
        input_convert_op.AddInputTensors([input_tensor_wrapper])
        input_convert_op.AddOutputTensors([input_q_tensor_wrapper])

        precompute_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name + "_precompute_a8",
            QNN_OP_PACKAGE_NAME_TMAN,
            OpTMANPrecomputeA8.op_name,
        )
        # `TMANPrecomputeA8` 的职责很单纯：
        # - 输入已经是 INT8 激活；
        # - 输出是 packed 的 LUT scratch；
        # - 它不关心外部 FP16 语义，也不关心 output_scale。
        precompute_op.AddInputTensors([input_q_tensor_wrapper])
        precompute_op.AddOutputTensors([l_tensor_wrapper])
        precompute_op.AddScalarParam(
            OpTMANPrecomputeA8.param_group_size,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {QCOM_DATA: np.int32(group_size)},
        )
        precompute_op.AddScalarParam(
            OpTMANPrecomputeA8.param_bits,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {QCOM_DATA: np.int32(bits)},
        )
        precompute_op.AddScalarParam(
            OpTMANPrecomputeA8.param_symmetric,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {QCOM_DATA: np.int32(symmetric)},
        )

        linear_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name + "_linear_a8",
            QNN_OP_PACKAGE_NAME_TMAN,
            OpTMANLinearA8.op_name,
        )
        # 这里把 effective_scales 作为 static tensor 传给 runtime。
        # runtime 不再直接感知 input_scale/output_scale，而是消费 host 预折叠后的结果。
        #
        # 换句话说：
        # - input_scale/output_scale 负责图边界上的 quant/dequant；
        # - weight_scale * input_scale / output_scale 的折叠结果，
        #   才是 linear kernel 真正需要的 group-wise 缩放量。
        linear_op.AddInputTensors([l_tensor_wrapper, qweight_tensor_wrapper, scales_tensor_wrapper])
        linear_op.AddOutputTensors([c_tensor_wrapper])
        linear_op.AddScalarParam(
            OpTMANLinearA8.param_group_size,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {QCOM_DATA: np.int32(group_size)},
        )
        linear_op.AddScalarParam(
            OpTMANLinearA8.param_bits,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {QCOM_DATA: np.int32(bits)},
        )
        linear_op.AddScalarParam(
            OpTMANLinearA8.param_symmetric,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {QCOM_DATA: np.int32(symmetric)},
        )

        finalize_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name + "_finalize_a8",
            QNN_OP_PACKAGE_NAME_TMAN,
            OpTMANFinalizeA8.op_name,
        )
        # `TMANFinalizeA8` 的职责是把 float accumulator 收到对称 INT8。
        # 当前实现里：
        # - scalar 路径始终可用；
        # - HVX 路径已经实现了更实质的 round/clamp/s16-sat；
        # - 最后再交给外层 Convert 用 output_scale 反量化回 FP16。
        finalize_op.AddInputTensors([c_tensor_wrapper])
        finalize_op.AddOutputTensors([output_q_tensor_wrapper])
        finalize_op.AddScalarParam(
            OpTMANFinalizeA8.param_group_size,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {QCOM_DATA: np.int32(group_size)},
        )
        finalize_op.AddScalarParam(
            OpTMANFinalizeA8.param_bits,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {QCOM_DATA: np.int32(bits)},
        )
        finalize_op.AddScalarParam(
            OpTMANFinalizeA8.param_symmetric,
            PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
            {QCOM_DATA: np.int32(symmetric)},
        )

        output_convert_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name + "_output_convert_a8",
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpConvert.op_name,
        )
        output_convert_op.AddInputTensors([output_q_tensor_wrapper])
        output_convert_op.AddOutputTensors([output_tensor_wrapper])

        return [
            input_convert_op,
            precompute_op,
            linear_op,
            finalize_op,
            output_convert_op,
        ]
