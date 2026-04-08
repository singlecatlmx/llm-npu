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


def _dequantize_activation_per_channel(
    x_q: torch.Tensor,
    input_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    对最后一维做 per-channel 对称反量化。

    这里是 W4A8 Python 参考实现里对输入激活的还原逻辑：
    - x_q      : 图中显式量化后的 8bit 激活
    - input_scale : 外部 pt 文件提供的每个输入通道 scale
    - dtype    : 当前量化存储 dtype

    目前我们沿用 Qualcomm 推荐的 8a8w 激活 dtype 约定，主路径使用
    uint8 + symmetric 语义（零点固定为 128）；为了调试方便，也兼容 int8。
    """
    view_shape = [1] * (x_q.dim() - 1) + [x_q.shape[-1]]
    scale = input_scale.to(device=x_q.device, dtype=torch.float32).view(view_shape)

    if dtype == torch.uint8:
        zero_point = 128
    elif dtype == torch.int8:
        zero_point = 0
    else:
        raise ValueError(f"Unsupported W4A8 activation dtype: {dtype}")

    return (x_q.to(torch.float32) - zero_point) * scale


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


@torch.library.custom_op("tman::linear_w4a8", mutates_args=())
def tman_linear_w4a8(
    x_q: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
    g_idx: torch.Tensor,
    wf_unsqueeze_zero: torch.Tensor,
    wf_unsqueeze_neg_one: torch.Tensor,
    group_size: int,
    bits: int,
    symmetric: bool,
    gptq_v2: bool,
) -> torch.Tensor:
    """
    TMAN W4A8 路径的 Python 参考实现。

    builder 最终会把这个 op lowering 到 TMAN OpPackage 的整数主路径。
    但在 eager/export/PT2E 阶段，仍然需要一个可以直接执行的参考版本，
    所以这里保留一条纯 Python 浮点参考链路：

    1. 用外部 input_scale 把输入 q8 激活反量化回浮点；
    2. 把 GPTQ 的 4bit 权重反量化回浮点；
    3. 用普通 matmul 得到参考输出。

    注意：
    - output_scale 被显式放进 op 签名里，目的是让 builder 在 lowering
      阶段可以直接读到外部校准得到的输出量化参数；
    - 这个 Python 参考实现本身不直接消费 output_scale。
    """
    out_features = qweight.shape[1]
    out_shape = x_q.shape[:-1] + (out_features,)
    x = _dequantize_activation_per_channel(x_q, input_scale, x_q.dtype).reshape(
        -1, x_q.shape[-1]
    )
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


@tman_linear_w4a8.register_fake
def tman_linear_w4a8_fake(
    x_q: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
    g_idx: torch.Tensor,
    wf_unsqueeze_zero: torch.Tensor,
    wf_unsqueeze_neg_one: torch.Tensor,
    group_size: int,
    bits: int,
    symmetric: bool,
    gptq_v2: bool,
) -> torch.Tensor:
    # fake kernel 只需要给出正确形状，便于 export / shape propagation。
    out_features = qweight.shape[1]
    out_shape = x_q.shape[:-1] + (out_features,)
    return input_scale.new_zeros(out_shape, dtype=torch.float32)


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
