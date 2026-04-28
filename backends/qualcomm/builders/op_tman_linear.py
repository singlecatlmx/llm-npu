# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict, List

import executorch.backends.qualcomm.python.PyQnnWrapperAdaptor as PyQnnWrapper

import numpy as np
import torch
from executorch.backends.qualcomm.utils.constants import QCOM_DATA
from executorch.backends.qualcomm.builders.utils import unpack_gptqv2, hvx_preprocess_weights, unpack_weights, precompute_lut1, precompute_eye_tile, precompute_hf_lut, precompute_scales_off
import logging

from .node_visitor import NodeVisitor, register_node_visitor
from .qnn_constants import (
    OpTMANLinear,
    OpTMANPrecompute,
    OpTMANFinalize,
    OpTMANPrefillLinear, # Prefill Linear
    OpConvert,
    QNN_OP_PACKAGE_NAME_TMAN,
    QNN_OP_PACKAGE_NAME_QTI_AISW,
)
from .utils import get_parameter


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _is_fake_tensor(tensor: torch.Tensor) -> bool:
    return isinstance(tensor, torch.Tensor) and (
        type(tensor).__name__ == "FakeTensor"
        or getattr(tensor, "fake_mode", None) is not None
    )


def _static_shape(tensor: torch.Tensor):
    shape = []
    for dim in tensor.shape:
        try:
            shape.append(int(dim))
        except Exception:
            return None
    return tuple(shape)


def _to_numpy_array(tensor: torch.Tensor, target_dtype=None) -> np.ndarray:
    """Safely convert a tensor (including subclasses) to a CPU numpy array."""

    # Detach first so conversion never tracks autograd history.
    data = tensor.detach()

    # Move to CPU when possible.
    try:
        data = data.cpu()
    except Exception:
        pass

    # Unwrap tensor subclasses to a base torch.Tensor when needed.
    if type(data) is not torch.Tensor:
        try:
            data = data.as_subclass(torch.Tensor)
            if data.is_cuda:
                data = data.cpu()
        except Exception:
            # Some tensor subclasses (e.g. FakeTensor) cannot be materialized.
            if not _is_fake_tensor(data):
                data = torch.tensor(data, device="cpu")

    data = data.contiguous()

    try:
        arr = data.numpy()
    except (RuntimeError, TypeError):
        if _is_fake_tensor(data):
            # Partitioning can run in fake mode; create synthetic static arrays to
            # avoid data-dependent symbolic value extraction.
            np_dtype = np.dtype(target_dtype) if target_dtype is not None else np.float32
            shape = _static_shape(data)
            if shape is None:
                shape = (1,)
            arr = np.zeros(shape, dtype=np_dtype)
        else:
        # Prefer DLPack path before the expensive list fallback.
            try:
                if hasattr(np, "from_dlpack"):
                    arr = np.from_dlpack(data)
                else:
                    arr = np.asarray(data.tolist())
            except Exception:
                arr = np.asarray(data.tolist())

    if target_dtype is not None:
        np_dtype = np.dtype(target_dtype)
        if arr.dtype != np_dtype:
            arr = arr.astype(np_dtype, copy=False)

    return np.ascontiguousarray(arr)


def _numpy_dtype_from_torch(dtype: torch.dtype):
    if dtype == torch.float16:
        return np.float16
    if dtype == torch.float32:
        return np.float32
    if dtype == torch.float64:
        return np.float64
    if dtype == torch.bfloat16 and hasattr(np, "bfloat16"):
        return np.dtype("bfloat16")
    return None


def _get_c_size(
    m: int,
    bits: int,
) -> int:
    # float32
    c_size = m * bits
    return c_size * 4


def _get_l_size(
    k: int,
    group_size: int,
    need_dequant: bool = True,
) -> int:
    LUT_G = 4
    LUT_SIZE = 16
    ACT_GROUP_SIZE = 256
    # float16
    x_size = k if need_dequant else 0
    # int16
    l_size = k // LUT_G * LUT_SIZE
    # float32
    ls_size = 1 if (ACT_GROUP_SIZE == -1) else (k // ACT_GROUP_SIZE)
    # float32
    lb_size = 1 if (group_size == 0) else (k // group_size)
    return x_size * 2 + l_size * 2 + max(ls_size * 4, 128) + max(lb_size * 4, 128)


def _decide_tile_size(
    dim_size: int,
    total_size: int,
    vtcm_size_in_mb: int = 8,
    n_threads: int = 6,
    divider: int = 2,
) -> int:
    max_tile_size = vtcm_size_in_mb * 1024 * 1024 // n_threads
    res = dim_size
    success = False
    for s in range(dim_size // divider, 0, -1):
        chunk_size = s * divider  
        if dim_size % chunk_size != 0:
            continue
        res = chunk_size
        # total_size // dim_size * res 每个线程的所需VTCM大小
        if total_size // dim_size * res < max_tile_size:
            success = True
            break
    if not success:
        logger.warning(f"Can't find optimal tile size that is multiple of {divider} and fits in VTCM, use {res} as workaround")
    return res

def _get_output_temp_size(
    m: int,      
):
    return m * 32 * 2

def _get_act_temp_size(

):
    return 512 * 32 * 2

def _get_weight_temp_size(
    m: int
):
    return 128 * 512 * 2

@register_node_visitor
class TMANLinear(NodeVisitor):
    target = ["tman.linear.default", "tman.bitnet_linear.default", "tman.prefill_linear.default"]

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.add_convert = True

    def define_node(
        self,
        node: torch.fx.Node,
        nodes_to_wrappers: Dict[str, PyQnnWrapper.TensorWrapper],
    ) -> PyQnnWrapper.PyQnnOpWrapper:
        input_node = node.args[0]
        input_tensor = self.get_tensor(input_node, node)
        input_tensor_wrapper = self.define_tensor(
            input_node,
            node,
            input_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        if node.target.__name__ == "tman.linear.default":
            qweight_node = node.args[1]
            qweight_tensor = get_parameter(qweight_node, self.edge_program)
            scales_node = node.args[2]
            scales_tensor = get_parameter(scales_node, self.edge_program)
            qzeros_node = node.args[3]
            qzeros_tensor = get_parameter(qzeros_node, self.edge_program)
            group_size = cast(int, node.args[7])
            bits = cast(int, node.args[8])
            symmetric = cast(bool, node.args[9])
            gptq_v2 = cast(bool, node.args[10])

            qweight_np = _to_numpy_array(qweight_tensor, target_dtype=np.int32)
            scales_np = _to_numpy_array(
                scales_tensor,
                target_dtype=_numpy_dtype_from_torch(scales_tensor.dtype),
            )
            qzeros_np = _to_numpy_array(qzeros_tensor, target_dtype=np.int32)


            qweight_repacked, scales_repacked, zeros_repacked, ref_bits, ref_group_size, ref_symmetric = unpack_gptqv2(
                qweight_np,
                scales_np,
                qzeros_np,
                gptq_v2,
            )

            fake_params = _is_fake_tensor(qweight_tensor) or _is_fake_tensor(scales_tensor) or _is_fake_tensor(qzeros_tensor)
            if fake_params:
                assert ref_bits == bits and ref_group_size == group_size, (
                    f"TMANLinear(fake): bits/group_size mismatch, {ref_bits}/{ref_group_size} != {bits}/{group_size}"
                )
            else:
                assert ref_bits == bits and ref_group_size == group_size and ref_symmetric == symmetric, (
                    f"TMANLinear: bits/group_size/symmetric mismatch, {ref_bits}/{ref_group_size}/{ref_symmetric} != {bits}/{group_size}/{symmetric}"
                )
        elif node.target.__name__ == "tman.bitnet_linear.default":
            qweight_node = node.args[1]
            qweight_tensor = get_parameter(qweight_node, self.edge_program)
            scales_node = node.args[2]
            scales_tensor = get_parameter(scales_node, self.edge_program)
            group_size = 0
            bits = 2
            symmetric = True

            qweight_repacked = _to_numpy_array(
                (unpack_weights(qweight_tensor.detach(), dtype=torch.int8) + 2).to(
                    torch.uint8
                )
            )
            scales_repacked = _to_numpy_array(scales_tensor)
            zeros_repacked = None
        elif node.target.__name__ == "tman.prefill_linear.default":
            lut1_node = node.args[1]
            eye_tile_node = node.args[2]
            hf_lut_node = node.args[3]
            scales_off_node = node.args[4]
            qweight_node = node.args[5]
            qweight_tensor = get_parameter(qweight_node, self.edge_program)
            scales_node = node.args[6]
            scales_tensor = get_parameter(scales_node, self.edge_program)
            qzeros_node = node.args[7]
            qzeros_tensor = get_parameter(qzeros_node, self.edge_program)
            group_size = cast(int, node.args[11])
            bits = cast(int, node.args[12])
            symmetric = cast(bool, node.args[13])
            gptq_v2 = cast(bool, node.args[14])

            qweight_np = _to_numpy_array(qweight_tensor, target_dtype=np.int32)
            scales_np = _to_numpy_array(
                scales_tensor,
                target_dtype=_numpy_dtype_from_torch(scales_tensor.dtype),
            )
            qzeros_np = _to_numpy_array(qzeros_tensor, target_dtype=np.int32)

            qweight_repacked, scales_repacked, zeros_repacked, ref_bits, ref_group_size, ref_symmetric = unpack_gptqv2(
                qweight_np,
                scales_np,
                qzeros_np,
                gptq_v2,
            )
            fake_params = _is_fake_tensor(qweight_tensor) or _is_fake_tensor(scales_tensor) or _is_fake_tensor(qzeros_tensor)
            if fake_params:
                assert ref_bits == bits and ref_group_size == group_size, (
                    f"TMANPrefillLinear(fake): bits/group_size mismatch, {ref_bits}/{ref_group_size} != {bits}/{group_size}"
                )
            else:
                assert ref_bits == bits and ref_group_size == group_size and ref_symmetric == symmetric, (
                    f"TMANPrefillLinear: bits/group_size/symmetric mismatch, {ref_bits}/{ref_group_size}/{ref_symmetric} != {bits}/{group_size}/{symmetric}"
                )
            # 由于此处直接重新计算并生成了常量数据，
            # 可以直接调用预计算逻辑，或者保留用作后续封装 Tensor_wrapper 
            lut1_precompute = precompute_lut1()    # 如果你的预计算不需要依赖 PyTorch 的 tensor 参数，直接传需要的参数即可
            eye_tile_precompute = precompute_eye_tile()
            hf_lut = precompute_hf_lut()
            scales_off = precompute_scales_off()

        else:
            raise NotImplementedError(f"Unsupported node target: {node.target.__name__}")

        output_tensor = self.get_tensor(node, node)
        output_tensor_wrapper = self.define_tensor(
            node,
            node,
            output_tensor,
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
            nodes_to_wrappers,
        )

        k = input_tensor.shape[-1]
        m = output_tensor.shape[-1]

        zeros_repacked = zeros_repacked if not symmetric else None
        vec_p = 128
        # total size 是权重、scales、zeros总大小
        total_size = qweight_repacked.nbytes + max((scales_repacked.size + (zeros_repacked.size if zeros_repacked is not None else 0)) * np.dtype("float16").itemsize, 128)
        tile_p = _decide_tile_size(m*bits, total_size, divider=bits*vec_p)
        qweight_repacked, scales_repacked = hvx_preprocess_weights(qweight_repacked, scales_repacked, zeros_repacked, bits, tile_p=tile_p, vec_p=vec_p)
        logger.info(f"TMANLinear: m={m}, k={k}, bits={bits}, tile_p={tile_p}, qweight({qweight_repacked.shape})")
        
        # TODO: the logical tileM is tile_p / Bits

        qweight_tensor_wrapper = self.define_tensor(
            qweight_node,
            node,
            torch.from_numpy(qweight_repacked),
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )

        scales_tensor_wrapper = self.define_tensor(
            scales_node,
            node,
            torch.from_numpy(scales_repacked),
            PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
            nodes_to_wrappers,
        )

        # do not quantize scratch buffer
        no_quant_encoding, no_quant_configs = (
            PyQnnWrapper.Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_UNDEFINED,
            {},
        )

        if node.target.__name__ == "tman.prefill_linear.default":
            lut1_tensor_wrapper = self.define_tensor(
                lut1_node,
                node,
                torch.from_numpy(lut1_precompute),
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
                nodes_to_wrappers,
            )

            eye_tile_wrapper = self.define_tensor(
                eye_tile_node,
                node, 
                torch.from_numpy(eye_tile_precompute),
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
                nodes_to_wrappers,
            )

            hf_lut_wrapper = self.define_tensor(
                hf_lut_node,
                node,
                torch.from_numpy(hf_lut),
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
                nodes_to_wrappers,
            )

            scales_off_wrapper = self.define_tensor(
                scales_off_node,
                node,
                torch.from_numpy(scales_off),
                PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_STATIC,
                nodes_to_wrappers,
            )


        else:
            l_tensor_wrapper = self.define_custom_tensor_wrapper(
                node_name=node.name + "_precompute",
                tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                dtype=PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_8,
                quant_encoding=no_quant_encoding,
                quant_configs=no_quant_configs,
                dims=torch.Size((1, _get_l_size(k, group_size, not self.add_convert))),
                tensor=None,  # Unused when is_fake_tensor is True
                is_fake_tensor=True,
                nodes_to_wrappers=nodes_to_wrappers,
            )
            c_tensor_wrapper = self.define_custom_tensor_wrapper(
                node_name=node.name + "_linear",
                tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                dtype=PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_UINT_8,
                quant_encoding=no_quant_encoding,
                quant_configs=no_quant_configs,
                dims=torch.Size((1, _get_c_size(m, bits))),
                tensor=None,  # Unused when is_fake_tensor is True
                is_fake_tensor=True,
                nodes_to_wrappers=nodes_to_wrappers,
            )

        if self.add_convert:
            intermediate_input_tensor_wrapper = self.define_custom_tensor_wrapper(
                node_name=node.name + "_input_converted",
                tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                dtype=PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_16,
                quant_encoding=no_quant_encoding,
                quant_configs=no_quant_configs,
                dims=input_tensor.size(),
                tensor=None,
                is_fake_tensor=True,
                nodes_to_wrappers=nodes_to_wrappers,
            )
            intermediate_output_tensor_wrapper = self.define_custom_tensor_wrapper(
                node_name=node.name + "_output_converted",
                tensor_type=PyQnnWrapper.Qnn_TensorType_t.QNN_TENSOR_TYPE_NATIVE,
                dtype=PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_FLOAT_16,
                quant_encoding=no_quant_encoding,
                quant_configs=no_quant_configs,
                dims=output_tensor.size(),
                tensor=None,
                is_fake_tensor=True,
                nodes_to_wrappers=nodes_to_wrappers,
            )
            input_convert_op = PyQnnWrapper.PyQnnOpWrapper(
                node.name + "_input_convert",
                QNN_OP_PACKAGE_NAME_QTI_AISW,
                OpConvert.op_name,
            )
            input_convert_op.AddInputTensors([input_tensor_wrapper])
            input_convert_op.AddOutputTensors([intermediate_input_tensor_wrapper])

            output_convert_op = PyQnnWrapper.PyQnnOpWrapper(
                node.name + "_output_convert",
                QNN_OP_PACKAGE_NAME_QTI_AISW,
                OpConvert.op_name,
            )
            output_convert_op.AddInputTensors([intermediate_output_tensor_wrapper])
            output_convert_op.AddOutputTensors([output_tensor_wrapper])

            input_tensor_wrapper = intermediate_input_tensor_wrapper
            output_tensor_wrapper = intermediate_output_tensor_wrapper

        if node.target.__name__ == "tman.prefill_linear.default":
            # prefill linear Op
            prefill_linear_op = PyQnnWrapper.PyQnnOpWrapper(
                node.name + "_prefill_linear",
                QNN_OP_PACKAGE_NAME_TMAN,
                OpTMANPrefillLinear.op_name,
            )
            prefill_linear_op.AddInputTensors([input_tensor_wrapper, lut1_tensor_wrapper, eye_tile_wrapper, hf_lut_wrapper, scales_off_wrapper, qweight_tensor_wrapper,
                                                    scales_tensor_wrapper])
            prefill_linear_op.AddOutputTensors([output_tensor_wrapper])
            prefill_linear_op.AddScalarParam(
                OpTMANPrefillLinear.param_group_size,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
                {QCOM_DATA: np.int32(group_size)},
            )
            prefill_linear_op.AddScalarParam(
                OpTMANPrefillLinear.param_bits,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
                {QCOM_DATA: np.int32(bits)},
            )
            prefill_linear_op.AddScalarParam(
                OpTMANPrefillLinear.param_symmetric,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
                {QCOM_DATA: np.int32(symmetric)},
            )

            if self.add_convert:
                return [input_convert_op, prefill_linear_op, output_convert_op]
            return [prefill_linear_op]
        else:
            # precompute Op
            precompute_op = PyQnnWrapper.PyQnnOpWrapper(
                node.name + "_precompute",
                QNN_OP_PACKAGE_NAME_TMAN,
                OpTMANPrecompute.op_name,
            )
            precompute_op.AddInputTensors([input_tensor_wrapper])
            precompute_op.AddOutputTensors([l_tensor_wrapper])
            precompute_op.AddScalarParam(
                OpTMANPrecompute.param_group_size,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
                {QCOM_DATA: np.int32(group_size)},
            )
            precompute_op.AddScalarParam(
                OpTMANPrecompute.param_bits,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
                {QCOM_DATA: np.int32(bits)},
            )
            precompute_op.AddScalarParam(
                OpTMANPrecompute.param_symmetric,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
                {QCOM_DATA: np.int32(symmetric)},
            )
            # linear Op
            linear_op = PyQnnWrapper.PyQnnOpWrapper(
                node.name,
                QNN_OP_PACKAGE_NAME_TMAN,
                OpTMANLinear.op_name,
            )
            linear_op.AddInputTensors([l_tensor_wrapper, qweight_tensor_wrapper, scales_tensor_wrapper])
            linear_op.AddOutputTensors([c_tensor_wrapper])
            linear_op.AddScalarParam(
                OpTMANLinear.param_group_size,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
                {QCOM_DATA: np.int32(group_size)},
            )
            linear_op.AddScalarParam(
                OpTMANLinear.param_bits,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
                {QCOM_DATA: np.int32(bits)},
            )
            linear_op.AddScalarParam(
                OpTMANLinear.param_symmetric,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
                {QCOM_DATA: np.int32(symmetric)},
            )

            # finalize Op
            finalize_op = PyQnnWrapper.PyQnnOpWrapper(
                node.name + "_finalize",
                QNN_OP_PACKAGE_NAME_TMAN,
                OpTMANFinalize.op_name,
            )
            finalize_op.AddInputTensors([c_tensor_wrapper])
            finalize_op.AddOutputTensors([output_tensor_wrapper])
            finalize_op.AddScalarParam(
                OpTMANFinalize.param_group_size,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
                {QCOM_DATA: np.int32(group_size)},
            )
            finalize_op.AddScalarParam(
                OpTMANFinalize.param_bits,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
                {QCOM_DATA: np.int32(bits)},
            )
            finalize_op.AddScalarParam(
                OpTMANFinalize.param_symmetric,
                PyQnnWrapper.Qnn_DataType_t.QNN_DATATYPE_INT_32,
                {QCOM_DATA: np.int32(symmetric)},
            )

            if self.add_convert:
                return [input_convert_op, precompute_op, linear_op, finalize_op, output_convert_op]
            return [precompute_op, linear_op, finalize_op]
