import torch
from .utils import unpack_weights


def _dequantize_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    wf_unsqueeze_zero: torch.Tensor,
    wf_unsqueeze_neg_one: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    """
    Based on dequantize_weights in gptqmodel/nn_modules/qlinear/__init__.py
    """
    import torch as t

    num_itr = 1  # desc_act=False
    assert(qweight.dtype == t.int32 and qzeros.dtype == t.int32)
    pack_factor = 32 // bits
    dequant_dtype = t.int16 if bits == 8 else t.int8
    maxq = 2 ** bits - 1

    if bits in [2, 4, 8]:
        zeros = t.bitwise_right_shift(
            t.unsqueeze(qzeros, 2).expand(-1, -1, pack_factor),
            wf_unsqueeze_zero  # wf.unsqueeze(0),
        ).to(dequant_dtype)
        zeros = t.bitwise_and(zeros, maxq).reshape(scales.shape)

        weight = t.bitwise_and(
            t.bitwise_right_shift(
                t.unsqueeze(qweight, 1).expand(-1, pack_factor, -1),
                wf_unsqueeze_neg_one  # wf.unsqueeze(-1)
            ).to(dequant_dtype),
            maxq
        )
    elif bits == 3:
        zeros = qzeros.reshape(qzeros.shape[0], qzeros.shape[1] // 3, 3, 1).expand(
            -1, -1, -1, 12
        )
        zeros = zeros >> wf_unsqueeze_zero  # wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | ((zeros[:, :, 1, 0] << 2) & 0x4)
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | ((zeros[:, :, 2, 0] << 1) & 0x6)
        zeros = zeros & 0x7
        zeros = t.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        ).reshape(scales.shape)

        weight = qweight.reshape(qweight.shape[0] // 3, 3, 1, qweight.shape[1]).expand(
            -1, -1, 12, -1
        )
        weight = (weight >> wf_unsqueeze_neg_one) & 0x7  # wf.unsqueeze(-1)
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = t.cat([weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1)
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    if num_itr == 1:
        weights = scales[g_idx.long()] * (weight - zeros[g_idx.long()])
    else:
        num_dim = g_idx.shape[0] // num_itr
        weights = []
        for i in range(num_itr):
            scale_i = scales[:, i * num_dim: (i + 1) * num_dim]
            weight_i = weight[:, i * num_dim: (i + 1) * num_dim]
            zeros_i = zeros[:, i * num_dim: (i + 1) * num_dim]
            g_idx_i = g_idx[i * num_dim: (i + 1) * num_dim].long()
            weights.append(scale_i[g_idx_i] * (weight_i - zeros_i[g_idx_i]))
        weights = t.cat(weights, dim=1)

    return weights


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
    group_size: int,
    bits: int,
    symmetric: bool,
    gptq_v2: bool,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    """
    这是正式 W4A8 路径在 Python/eager 侧的参考语义。

    作用不是追求性能，而是保证：
    1. 导出前 eager 行为与 runtime 的数学语义一致；
    2. fake tensor / shape propagation 能够顺利通过；
    3. builder 下沉前，图里已经显式携带 input_scale / output_scale。

    当前语义为：
    - 输入按外部 input_scale 做对称 A8 量化，再反量化回 fp；
    - 权重仍按 GPTQ 规则反量化；
    - 输出再按外部 output_scale 做一次量化反量化；
    - 最终对外保持 fp16/fp32 浮点接口。
    """
    out_features = qweight.shape[1]
    out_shape = x.shape[:-1] + (out_features,)
    x_2d = x.reshape(-1, x.shape[-1])
    weights = _dequantize_weight(
        qweight,
        scales,
        qzeros,
        g_idx,
        wf_unsqueeze_zero,
        wf_unsqueeze_neg_one,
        bits,
    ).to(x.dtype)

    # 正式 W4A8 路径使用外部导入的 per-layer input_scale，
    # 不再在 eager/runtime 内部重新估计激活 scale。
    del output_scale
    x_scale = input_scale.reshape(1, 1).to(x_2d.dtype)
    x_scale = x_scale.clamp(min=1.0 / 127.0)
    x_q = (x_2d / x_scale).round().clamp(-127, 127)
    x_dq = x_q * x_scale

    out = torch.matmul(x_dq, weights).reshape(out_shape)
    # 输出端也做一次量化再反量化，模拟 runtime FinalizeA8 的输出误差模型。
    out_scale = output_scale.reshape(1, 1).to(out.dtype).clamp(min=1.0 / 127.0)
    out_q = (out.reshape(-1, out.shape[-1]) / out_scale).round().clamp(-127, 127)
    out_dq = (out_q * out_scale).reshape(out_shape)
    return out_dq.to(x.dtype)


@tman_linear_a8.register_fake
def tman_linear_a8_fake(
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
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
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
