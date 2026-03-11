# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper
import torch

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import (
    QNN_OP_PACKAGE_NAME_QTI_AISW,
    QNN_OP_PACKAGE_NAME_TMAN,
    OpConvert,
    OpElementWiseAdd,
    OpTMANAdd,
)


@register_node_visitor
class Add(NodeVisitor):
    target = ["aten.add.Tensor"]

    _TMAN_TARGETS = {"tman.linear.default", "tman.bitnet_linear.default"}

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def _is_tman_input(self, node: torch.fx.Node) -> bool:
        return (
            hasattr(node, "target")
            and hasattr(node.target, "__name__")
            and node.target.__name__ in self._TMAN_TARGETS
        )

    def _define_qnn_add(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        out_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            out_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        add_output_tensors = [output_tensor_wrapper]

        add_input_tensors = []
        for index in range(2):
            input_node = node.args[index]
            input_tensor = self.get_tensor(input_node, node)
            input_tensor_wrapper = self.define_tensor(
                input_node,
                node,
                input_tensor,
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                nodes_to_wrappers,
            )
            add_input_tensors.append(input_tensor_wrapper)

        add_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name,
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpElementWiseAdd.op_name,
        )
        add_op.AddInputTensors(add_input_tensors)
        add_op.AddOutputTensors(add_output_tensors)
        return add_op

    def _define_fp16_convert(
        self,
        node_name: str,
        input_wrapper: PyQnnWrapper.TensorWrapper,
        tensor: torch.Tensor,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> List[PyQnnWrapper.PyQnnOpWrapper | PyQnnWrapper.TensorWrapper]:
        no_quant_encoding = (
            PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED
        )
        no_quant_configs = {}
        output_wrapper = self.define_custom_tensor_wrapper(
            node_name=node_name,
            tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            dtype=PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_16,
            quant_encoding=no_quant_encoding,
            quant_configs=no_quant_configs,
            dims=tensor.size(),
            tensor=None,
            is_fake_tensor=True,
            nodes_to_wrappers=nodes_to_wrappers,
        )
        convert_op = PyQnnWrapper.PyQnnOpWrapper(
            node_name + "_convert",
            QNN_OP_PACKAGE_NAME_QTI_AISW,
            OpConvert.op_name,
        )
        convert_op.AddInputTensors([input_wrapper])
        convert_op.AddOutputTensors([output_wrapper])
        return [convert_op, output_wrapper]

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[torch.fx.Node, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_nodes = [node.args[0], node.args[1]]
        tman_flags = [self._is_tman_input(input_node) for input_node in input_nodes]

        if sum(tman_flags) != 1:
            return self._define_qnn_add(node, nodes_to_wrappers)

        tman_index = 0 if tman_flags[0] else 1
        other_index = 1 - tman_index
        tman_node = input_nodes[tman_index]
        other_node = input_nodes[other_index]

        tman_wrapper = nodes_to_wrappers[tman_node.name].get(1, None)
        if tman_wrapper is None:
            return self._define_qnn_add(node, nodes_to_wrappers)

        build_ops = []

        other_tensor = self.get_tensor(other_node, node)
        other_wrapper = self.define_tensor(
            other_node,
            node,
            other_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )
        if other_tensor.dtype != torch.float16:
            convert_op, other_wrapper = self._define_fp16_convert(
                node.name + "_rhs_fp16",
                other_wrapper,
                other_tensor,
                nodes_to_wrappers,
            )
            build_ops.append(convert_op)

        out_tensor = self.get_tensor(node, node)
        if out_tensor.dtype == torch.float16:
            tman_add_output_wrapper = self.define_tensor(
                node,
                node,
                out_tensor,
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                nodes_to_wrappers,
            )
            output_convert_op = None
        else:
            no_quant_encoding = (
                PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED
            )
            no_quant_configs = {}
            tman_add_output_wrapper = self.define_custom_tensor_wrapper(
                node_name=node.name + "_tmanadd_output_fp16",
                tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                dtype=PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_16,
                quant_encoding=no_quant_encoding,
                quant_configs=no_quant_configs,
                dims=out_tensor.size(),
                tensor=None,
                is_fake_tensor=True,
                nodes_to_wrappers=nodes_to_wrappers,
            )
            output_tensor_wrapper = self.define_tensor(
                node,
                node,
                out_tensor,
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                nodes_to_wrappers,
            )
            output_convert_op = PyQnnWrapper.PyQnnOpWrapper(
                node.name + "_output_convert",
                QNN_OP_PACKAGE_NAME_QTI_AISW,
                OpConvert.op_name,
            )
            output_convert_op.AddInputTensors([tman_add_output_wrapper])
            output_convert_op.AddOutputTensors([output_tensor_wrapper])

        tman_add_op = PyQnnWrapper.PyQnnOpWrapper(
            node.name + "_tman_add",
            QNN_OP_PACKAGE_NAME_TMAN,
            OpTMANAdd.op_name,
        )
        tman_add_op.AddInputTensors([tman_wrapper, other_wrapper])
        tman_add_op.AddOutputTensors([tman_add_output_wrapper])
        build_ops.append(tman_add_op)

        if output_convert_op is not None:
            build_ops.append(output_convert_op)

        return build_ops
