import torch
from executorch.backends.qualcomm.utils.tman_w4a8 import (
    dequantize_gptq_weight,
    tman_linear_a8_reference,
)
from .utils import unpack_weights


def _dequantize_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    wf_unsqueeze_zero: torch.Tensor,
    wf_unsqueeze_neg_one: torch.Tensor,
    bits: int,
    gptq_v2: bool = True,
) -> torch.Tensor:
    """Compatibility wrapper kept for the existing W4A16 path."""
    return dequantize_gptq_weight(
        qweight,
        scales,
        qzeros,
        g_idx,
        wf_unsqueeze_zero,
        wf_unsqueeze_neg_one,
        bits,
        gptq_v2,
    )


@torch.library.custom_op("tman::linear", mutates_args=())
def tman_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    wf_unsqueeze_zero: torch.Tensor,
    wf_unsqueeze_neg_one: torch.Tensor,
    group_size: int,
    bits: int,
    symmetric: bool,
    gptq_v2: bool,
) -> torch.Tensor:
    out_features = qweight.shape[1]
    out_shape = x.shape[:-1] + (out_features,)
    x = x.reshape(-1, x.shape[-1])
    weights = _dequantize_weight(
        qweight,
        scales,
        qzeros,
        g_idx,
        wf_unsqueeze_zero,
        wf_unsqueeze_neg_one,
        bits,
        gptq_v2,
    ).to(x.dtype)
    out = torch.matmul(x, weights).reshape(out_shape)
    return out.to(x.dtype)


@tman_linear.register_fake
def tman_linear_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    wf_unsqueeze_zero: torch.Tensor,
    wf_unsqueeze_neg_one: torch.Tensor,
    group_size: int,
    bits: int,
    symmetric: bool,
    gptq_v2: bool,
) -> torch.Tensor:
    out_features = qweight.shape[1]
    out_shape = x.shape[:-1] + (out_features,)
    return x.new_zeros(out_shape)


@torch.library.custom_op("tman::linear_a8", mutates_args=())
def tman_linear_a8(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    wf_unsqueeze_zero: torch.Tensor,
    wf_unsqueeze_neg_one: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
    group_size: int,
    bits: int,
    symmetric: bool,
    gptq_v2: bool,
) -> torch.Tensor:
    return tman_linear_a8_reference(
        x,
        qweight,
        scales,
        qzeros,
        g_idx,
        wf_unsqueeze_zero,
        wf_unsqueeze_neg_one,
        input_scale,
        output_scale,
        bits,
        gptq_v2,
    )


@tman_linear_a8.register_fake
def tman_linear_a8_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    wf_unsqueeze_zero: torch.Tensor,
    wf_unsqueeze_neg_one: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
    group_size: int,
    bits: int,
    symmetric: bool,
    gptq_v2: bool,
) -> torch.Tensor:
    out_features = qweight.shape[1]
    out_shape = x.shape[:-1] + (out_features,)
    return x.new_zeros(out_shape)


@torch.library.custom_op("tman::bitnet_linear", mutates_args=())
def tman_bitnet_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    # unpack weights
    w = weight
    w_quant = unpack_weights(w, dtype=x.dtype)
    # activation_quant
    num_bits = 8
    Qn = -(2 ** (num_bits - 1))
    Qp = 2 ** (num_bits - 1) - 1
    scale = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    result = (x * scale).round().clamp(Qn, Qp)
    input_quant, input_scale = result.to(torch.int8), scale
    # linear
    y = torch.nn.functional.linear(input_quant.to(x.dtype), w_quant)
    y = y / input_scale * weight_scale
    return y


@tman_bitnet_linear.register_fake
def tman_bitnet_linear_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    VALUES_PER_ITEM = 4
    out_features = weight.shape[0] * VALUES_PER_ITEM
    out_shape = x.shape[:-1] + (out_features,)
    return x.new_zeros(out_shape)
