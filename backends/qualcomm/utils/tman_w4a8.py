from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch

# 这组常量描述的是当前 A8 路径最核心的“数据组织事实”：
# 1. LUT 的基本粒度是 4 个激活组成一个 group，对应 16 项子集和；
# 2. 每 4 个 qgroup 组成一个 128B block，正好对应 1 条 HVX vector line；
# 3. 当前项目范围已经明确收敛到 W4 / G128 / Sym，因此这里也把目标配置常量化。
TMAN_A8_LUT_GROUP = 4
TMAN_A8_LUT_SIZE = 16
TMAN_A8_BITS = 4
TMAN_A8_QGROUPS_PER_VECTOR = 4
TMAN_A8_LUT_VALUES_PER_VECTOR_BLOCK = TMAN_A8_QGROUPS_PER_VECTOR * TMAN_A8_LUT_SIZE
TMAN_A8_TARGET_GROUP_SIZE = 128
TMAN_A8_TARGET_SYMMETRIC = True
TMAN_A8_TARGET_QGROUPS_PER_GROUP = TMAN_A8_TARGET_GROUP_SIZE // TMAN_A8_LUT_GROUP
TMAN_A8_TARGET_QGROUP_VECTOR_BLOCKS = (
    TMAN_A8_TARGET_QGROUPS_PER_GROUP // TMAN_A8_QGROUPS_PER_VECTOR
)


EXPECTED_QWEN3_LINEAR_NAMES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def validate_w4a8_target_config(
    bits: int,
    group_size: int,
    symmetric: bool,
) -> None:
    # 当前 A8 路径按项目范围只服务：
    # - W4
    # - group_size = 128
    # - symmetric GPTQ
    #
    # 这里统一做一处配置约束，避免 builder / runtime / 工具脚本
    # 各自散落一套相近但不完全一致的判断逻辑。
    if bits != TMAN_A8_BITS:
        raise ValueError(
            f"TMAN W4A8 path currently only supports bits={TMAN_A8_BITS}, got {bits}"
        )
    if group_size != TMAN_A8_TARGET_GROUP_SIZE:
        raise ValueError(
            "TMAN W4A8 path currently only supports "
            f"group_size={TMAN_A8_TARGET_GROUP_SIZE}, got {group_size}"
        )
    # 这里显式转成 Python bool，再做值比较。
    # 这样即使上游传进来的是 numpy.bool_ / torch.bool 标量，也不会因为 `is`
    # 的对象身份比较而产生误判。
    if bool(symmetric) != TMAN_A8_TARGET_SYMMETRIC:
        raise ValueError(
            "TMAN W4A8 path currently only supports "
            f"symmetric={TMAN_A8_TARGET_SYMMETRIC}, got {symmetric}"
        )


def build_ordered_qwen3_linear_paths(num_layers: int) -> list[str]:
    # 返回稳定顺序的目标 linear 路径，方便：
    # 1. 工具脚本做批量审计；
    # 2. 输出日志时保持层次顺序一致，便于人工比对。
    return [
        f"model.layers.{layer_idx}.{linear_name}"
        for layer_idx in range(num_layers)
        for linear_name in EXPECTED_QWEN3_LINEAR_NAMES
    ]


def build_expected_qwen3_linear_paths(num_layers: int) -> set[str]:
    # 单独抽成 helper，方便：
    # 1. registry 内部做覆盖校验；
    # 2. 外部脚本直接复用，不重复拼接规则。
    return {
        f"model.layers.{layer_idx}.{linear_name}"
        for layer_idx in range(num_layers)
        for linear_name in EXPECTED_QWEN3_LINEAR_NAMES
    }


@dataclass(frozen=True)
class LinearActivationScales:
    # 两个量都是 shape=[1] 的标量 tensor。
    # input_scale: 线性层输入从 FP16 量化到 INT8 时使用。
    # output_scale: 线性层输出从 INT8 反量化回 FP16 时使用。
    input_scale: torch.Tensor
    output_scale: torch.Tensor


@dataclass(frozen=True)
class GPTQTensorLayout:
    bits: int
    group_size: int
    k: int
    m: int
    symmetric: bool


def _pack_repeated_lowbit_value(bits: int, repeated_value: int) -> int:
    packed = 0
    shift = 0
    while shift < 32:
        packed |= (repeated_value & ((1 << bits) - 1)) << shift
        shift += bits
    if packed >= (1 << 31):
        packed -= 1 << 32
    return packed


def expected_symmetric_qzeros_pack_value(bits: int, gptq_v2: bool) -> int:
    # 对称 GPTQ 的 qzeros 在不同格式里的 packed 常量不同：
    # - gptq_v2=True  : field_value = 2^(bits-1)
    # - gptq_v2=False : field_value = 2^(bits-1) - 1
    field_value = (1 << (bits - 1)) if gptq_v2 else ((1 << (bits - 1)) - 1)
    return _pack_repeated_lowbit_value(bits, field_value)


def infer_gptq_tensor_layout(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    gptq_v2: bool,
) -> GPTQTensorLayout:
    # 这个 helper 用于在不依赖 gptqmodel 运行时对象的情况下，
    # 直接从 safetensors / state_dict 张量推断：
    # - bits
    # - group_size
    # - K/M
    # - 当前 qzeros 是否满足“对称 GPTQ”的 packed 形式
    if qweight.dim() != 2 or scales.dim() != 2 or qzeros.dim() != 2:
        raise ValueError(
            "Expected qweight/scales/qzeros to be 2D tensors, "
            f"got {tuple(qweight.shape)}, {tuple(scales.shape)}, {tuple(qzeros.shape)}"
        )
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise TypeError(
            f"Expected qweight/qzeros dtype=int32, got {qweight.dtype}, {qzeros.dtype}"
        )

    m = qweight.shape[1]
    if qzeros.shape[0] != scales.shape[0]:
        raise ValueError(
            f"qzeros/scales group dim mismatch: {tuple(qzeros.shape)} vs {tuple(scales.shape)}"
        )
    if scales.shape[1] % qzeros.shape[1] != 0:
        raise ValueError(
            f"Unable to infer GPTQ bits from scales/qzeros shape: {tuple(scales.shape)} vs {tuple(qzeros.shape)}"
        )

    pack_factor = scales.shape[1] // qzeros.shape[1]
    bits = 32 // pack_factor
    if bits not in (2, 4, 8):
        raise ValueError(f"Unsupported inferred GPTQ bits: {bits}")

    k = qweight.shape[0] * (32 // bits)
    if k % scales.shape[0] != 0:
        raise ValueError(
            f"Unable to infer group_size cleanly: K={k}, num_groups={scales.shape[0]}"
        )
    group_size = k // scales.shape[0]

    expected_qzeros = expected_symmetric_qzeros_pack_value(bits, gptq_v2)
    symmetric = bool(torch.all(qzeros == expected_qzeros).item())
    return GPTQTensorLayout(
        bits=bits,
        group_size=group_size,
        k=k,
        m=m,
        symmetric=symmetric,
    )


def slice_gptq_output_channels(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    out_features_slice: int,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 对 GPTQ 张量做“按输出通道切片”时，需要同时裁剪：
    # - qweight: 最后一维直接切
    # - scales : 对应输出通道直接切
    # - qzeros : 最后一维是 pack 后的输出通道，需要按 pack_factor 一起切
    pack_factor = 32 // bits
    if out_features_slice % pack_factor != 0:
        raise ValueError(
            f"out_features_slice must be divisible by pack_factor={pack_factor}, got {out_features_slice}"
        )
    if out_features_slice > qweight.shape[1]:
        raise ValueError(
            f"out_features_slice={out_features_slice} exceeds real out_features={qweight.shape[1]}"
        )

    qweight_sliced = qweight[:, :out_features_slice].contiguous()
    scales_sliced = scales[:, :out_features_slice].contiguous()
    qzeros_sliced = qzeros[:, : out_features_slice // pack_factor].contiguous()
    return qweight_sliced, qzeros_sliced, scales_sliced


def symmetric_int8_quantize_with_scale(
    x: torch.Tensor,
    scale: torch.Tensor,
    quant_min: int = -127,
    quant_max: int = 127,
) -> torch.Tensor:
    # W4A8 路径固定使用“对称 per-tensor INT8”量化。
    # 因此这里不引入 zero-point，只保留一个标量 scale。
    scale = scale.to(torch.float32).reshape(1)
    x_q = torch.round(x.to(torch.float32) / scale).clamp(quant_min, quant_max)
    return x_q.to(torch.int8)


def dequantize_gptq_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    wf_unsqueeze_zero: torch.Tensor,
    wf_unsqueeze_neg_one: torch.Tensor,
    bits: int,
    gptq_v2: bool = True,
) -> torch.Tensor:
    # 这里抽出一份纯 PyTorch 的 GPTQ 权重反量化实现，
    # 目的是给 W4A8 reference 路径和轻量验证脚本复用。
    import torch as t

    num_itr = 1  # desc_act=False
    assert qweight.dtype == t.int32 and qzeros.dtype == t.int32
    pack_factor = 32 // bits
    dequant_dtype = t.int16 if bits == 8 else t.int8
    maxq = 2 ** bits - 1

    if bits in [2, 4, 8]:
        zeros = t.bitwise_right_shift(
            t.unsqueeze(qzeros, 2).expand(-1, -1, pack_factor),
            wf_unsqueeze_zero,
        ).to(dequant_dtype)
        zeros = t.bitwise_and(zeros, maxq).reshape(scales.shape)
        if not gptq_v2:
            # AutoGPTQ / GPTQModel v2 之前的格式里，qzeros 在解包后需要先 +1，
            # 再减去对称中心值，才能恢复真实 zero offset。
            zeros = zeros + 1

        weight = t.bitwise_and(
            t.bitwise_right_shift(
                t.unsqueeze(qweight, 1).expand(-1, pack_factor, -1),
                wf_unsqueeze_neg_one,
            ).to(dequant_dtype),
            maxq,
        )
    elif bits == 3:
        zeros = qzeros.reshape(qzeros.shape[0], qzeros.shape[1] // 3, 3, 1).expand(
            -1, -1, -1, 12
        )
        zeros = zeros >> wf_unsqueeze_zero
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | (
            (zeros[:, :, 1, 0] << 2) & 0x4
        )
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | (
            (zeros[:, :, 2, 0] << 1) & 0x6
        )
        zeros = zeros & 0x7
        zeros = t.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        ).reshape(scales.shape)

        weight = qweight.reshape(qweight.shape[0] // 3, 3, 1, qweight.shape[1]).expand(
            -1, -1, 12, -1
        )
        weight = (weight >> wf_unsqueeze_neg_one) & 0x7
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | (
            (weight[:, 1, 0] << 2) & 0x4
        )
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | (
            (weight[:, 2, 0] << 1) & 0x6
        )
        weight = weight & 0x7
        weight = t.cat(
            [weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1
        )
    else:
        raise ValueError(f"Unsupported GPTQ bits for reference dequantization: {bits}")

    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    if num_itr == 1:
        weights = scales[g_idx.long()] * (weight - zeros[g_idx.long()])
    else:
        num_dim = g_idx.shape[0] // num_itr
        weights = []
        for i in range(num_itr):
            scale_i = scales[:, i * num_dim : (i + 1) * num_dim]
            weight_i = weight[:, i * num_dim : (i + 1) * num_dim]
            zeros_i = zeros[:, i * num_dim : (i + 1) * num_dim]
            g_idx_i = g_idx[i * num_dim : (i + 1) * num_dim].long()
            weights.append(scale_i[g_idx_i] * (weight_i - zeros_i[g_idx_i]))
        weights = t.cat(weights, dim=1)
    return weights


def unpack_gptq_qweight_to_uint8(qweight: torch.Tensor, bits: int) -> torch.Tensor:
    # 把 GPTQ 打包后的 int32 权重展开成 [M, K] 的 uint8 低比特表示。
    # 对称 W4 情况下，每个元素范围在 [0, 15]，后续 runtime baseline
    # 会再减去 zero_bias=8 还原成 [-8, 7]。
    if qweight.dtype != torch.int32:
        raise TypeError(f"Expected int32 qweight, got {qweight.dtype}")
    if bits not in (2, 4, 8):
        raise ValueError(f"Unsupported bits for unpack_gptq_qweight_to_uint8: {bits}")

    pack_factor = 32 // bits
    maxq = (1 << bits) - 1
    pieces = [
        torch.bitwise_and(torch.bitwise_right_shift(qweight, bit_offset), maxq)
        for bit_offset in range(0, 32, bits)
    ]
    unpacked = torch.stack(pieces, dim=1)
    k = qweight.shape[0] * pack_factor
    m = qweight.shape[1]
    return unpacked.reshape(k, m).transpose(0, 1).contiguous().to(torch.uint8)


def normalize_gptq_scales_layout(
    scales: torch.Tensor,
    qweight: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    # GPTQModel 原始 scales 通常是 [num_groups, M]。
    # builder/runtime baseline 更方便使用 [M, num_groups]。
    pack_factor = 32 // bits
    k = qweight.shape[0] * pack_factor
    m = qweight.shape[1]

    if scales.dim() != 2:
        raise ValueError(f"Expected 2D scales tensor, got shape {tuple(scales.shape)}")

    if scales.shape[1] == m:
        return scales.transpose(0, 1).contiguous().to(torch.float32)
    if scales.shape[0] == m:
        return scales.contiguous().to(torch.float32)

    raise ValueError(
        "Unable to normalize GPTQ scales layout: "
        f"qweight implies M={m}, K={k}, but scales shape is {tuple(scales.shape)}"
    )


def fold_weight_scales_for_w4a8(
    weight_scales_mg: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    # W4A8 baseline runtime 把三类 scale 折叠成一类 effective_scale：
    #   effective_scale = weight_scale * input_scale / output_scale
    #
    # 这样 runtime 的 linear op 就只需要消费一份 group-wise float scale。
    return weight_scales_mg.to(torch.float32) * (
        input_scale.to(torch.float32).reshape(1, 1)
        / output_scale.to(torch.float32).reshape(1, 1)
    )


def prepare_w4a8_baseline_runtime_tensors(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 统一生成 runtime baseline 所需的两份 static tensor：
    # 1. [M, G, Qg, bits] 的 LUT 索引张量
    #    其中：
    #    - G  = K / group_size
    #    - Qg = group_size / 4
    #    - bits 维上的每个元素都是一个 0~15 的 nibble 索引
    #
    # 这种“按 weight group 分块”的布局比 [M, Q, bits] 更接近后续
    # runtime / HVX 真正按 group 处理的访问方式。
    # 2. [M, num_groups] 的 effective_scales
    #
    # 注意：
    # - 这里返回的是“逻辑布局”，便于 reference/helper 层按 block 语义思考；
    # - 真正给 QNN / OpPackage 的物理 tensor 形态，会在 builder 里进一步压成 4D packed ABI。
    qweight_u8_mk = unpack_gptq_qweight_to_uint8(qweight, bits)
    qweight_lut_indices_mqb = build_w4a8_weight_lut_indices(qweight_u8_mk, bits)
    qgroups = qweight_lut_indices_mqb.shape[1]
    if qweight_u8_mk.shape[1] % group_size != 0:
        raise ValueError(
            f"K={qweight_u8_mk.shape[1]} must be divisible by group_size={group_size}"
        )
    if group_size % TMAN_A8_LUT_GROUP != 0:
        raise ValueError(f"group_size={group_size} must be divisible by {TMAN_A8_LUT_GROUP}")
    qgroups_per_group = group_size // TMAN_A8_LUT_GROUP
    num_groups = qgroups // qgroups_per_group
    if num_groups * qgroups_per_group != qgroups:
        raise ValueError(
            f"Q={qgroups} is not divisible by qgroups_per_group={qgroups_per_group}"
        )
    qweight_lut_indices_mgqb = qweight_lut_indices_mqb.reshape(
        qweight_lut_indices_mqb.shape[0],
        num_groups,
        qgroups_per_group,
        bits,
    ).contiguous()
    weight_scales_mg = normalize_gptq_scales_layout(scales, qweight, bits)
    effective_scales_mg = fold_weight_scales_for_w4a8(
        weight_scales_mg,
        input_scale,
        output_scale,
    )
    return qweight_lut_indices_mgqb, effective_scales_mg


def pack_w4a8_weight_lut_indices_for_hvx(
    qweight_lut_indices_mgqb: torch.Tensor,
    bits: int,
    qgroups_per_vector: int = TMAN_A8_QGROUPS_PER_VECTOR,
) -> torch.Tensor:
    # 把逻辑上的 [M, G, Qg, bits] 重排成更贴近 runtime/HVX 的 block 视图：
    # [M, G, qgroup_vector_blocks, qgroups_per_vector * bits]
    #
    # 这样每个 block 的最后一维就是 runtime 一次连续消费的一段 packed LUT index。
    if qweight_lut_indices_mgqb.dim() != 4:
        raise ValueError(
            "Expected grouped LUT indices shaped as [M, G, Qg, bits], "
            f"got {tuple(qweight_lut_indices_mgqb.shape)}"
        )
    if qweight_lut_indices_mgqb.shape[-1] != bits:
        raise ValueError(
            f"Grouped LUT indices bits dim={qweight_lut_indices_mgqb.shape[-1]} "
            f"does not match bits={bits}"
        )

    m, num_groups, qgroups_per_group, _ = qweight_lut_indices_mgqb.shape
    if qgroups_per_group % qgroups_per_vector != 0:
        raise ValueError(
            f"qgroups_per_group={qgroups_per_group} must be divisible by "
            f"qgroups_per_vector={qgroups_per_vector}"
        )

    qgroup_vector_blocks = qgroups_per_group // qgroups_per_vector
    return qweight_lut_indices_mgqb.reshape(
        m,
        num_groups,
        qgroup_vector_blocks,
        qgroups_per_vector * bits,
    ).contiguous()


def build_w4a8_lut_scratch_shape(
    num_rows: int,
    num_groups: int,
    qgroups_per_group: int,
    qgroups_per_vector: int = TMAN_A8_QGROUPS_PER_VECTOR,
) -> torch.Size:
    # LUT scratch 按 block 显式组织成：
    # [1, row, group, qgroup_vector_blocks, qgroups_per_vector * 16]
    #
    # 最后一维刚好是一整条 vector line 对应的 LUT block。
    if qgroups_per_group % qgroups_per_vector != 0:
        raise ValueError(
            f"qgroups_per_group={qgroups_per_group} must be divisible by "
            f"qgroups_per_vector={qgroups_per_vector}"
        )

    qgroup_vector_blocks = qgroups_per_group // qgroups_per_vector
    return torch.Size(
        (
            1,
            num_rows,
            num_groups,
            qgroup_vector_blocks,
            qgroups_per_vector * TMAN_A8_LUT_SIZE,
        )
    )


def build_w4a8_lut_scratch_packed_shape(
    num_rows: int,
    num_groups: int,
    qgroups_per_group: int,
    qgroups_per_vector: int = TMAN_A8_QGROUPS_PER_VECTOR,
) -> torch.Size:
    # QNN / OpPackage schema 当前更稳妥的交付接口仍按 4D tensor 描述。
    # 因此对外 ABI 采用 packed 4D：
    # [1, row, group, qgroup_vector_blocks * (qgroups_per_vector * 16)]
    #
    # 这层 helper 的意义是把“逻辑 5D block 布局”和“对外 4D ABI”之间的关系
    # 显式固定下来，避免 builder/runtime/XML 各自手写一份压平规则。
    logical_shape = build_w4a8_lut_scratch_shape(
        num_rows=num_rows,
        num_groups=num_groups,
        qgroups_per_group=qgroups_per_group,
        qgroups_per_vector=qgroups_per_vector,
    )
    return torch.Size(
        (
            logical_shape[0],
            logical_shape[1],
            logical_shape[2],
            logical_shape[3] * logical_shape[4],
        )
    )


def pack_w4a8_weight_lut_indices_for_qnn(
    qweight_lut_indices_mgqb: torch.Tensor,
    bits: int,
    qgroups_per_vector: int = TMAN_A8_QGROUPS_PER_VECTOR,
) -> torch.Tensor:
    # 对外给 QNN 的 static tensor 仍保持 4D packed ABI：
    # [1, M, G, qgroup_vector_blocks * (qgroups_per_vector * bits)]
    #
    # 逻辑上仍然是 block 结构，只是把最后两维压平，避免和当前
    # OpPackage XML / HTP 常见的 4D 约束冲突。
    #
    # 返回值的最后一维就是：
    #   qgroup_vector_blocks * (4 * bits)
    # 这也是 builder/runtime C++ 校验时所使用的 packed ABI 约束。
    packed_hvx = pack_w4a8_weight_lut_indices_for_hvx(
        qweight_lut_indices_mgqb,
        bits,
        qgroups_per_vector=qgroups_per_vector,
    )
    return packed_hvx.reshape(
        1,
        packed_hvx.shape[0],
        packed_hvx.shape[1],
        packed_hvx.shape[2] * packed_hvx.shape[3],
    ).contiguous()


def build_w4a8_weight_lut_indices(
    qweight_u8_mk: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    # 对每个 4-activation 小组，预先计算每个 bit-plane 的 LUT 索引。
    #
    # 例如对于 4 个权重 w0..w3：
    #   idx_bit0 = [w0 bit0, w1 bit0, w2 bit0, w3 bit0] 组成的 4bit 数
    #   idx_bit1 = ...
    #
    # runtime 线性层只需要：
    #   acc_q = -8 * lut[15] + lut[idx0] + 2*lut[idx1] + 4*lut[idx2] + 8*lut[idx3]
    if qweight_u8_mk.dim() != 2:
        raise ValueError(f"Expected [M, K] unpacked qweight, got {tuple(qweight_u8_mk.shape)}")
    if qweight_u8_mk.shape[1] % TMAN_A8_LUT_GROUP != 0:
        raise ValueError(
            f"K={qweight_u8_mk.shape[1]} must be divisible by {TMAN_A8_LUT_GROUP} for LUT grouping"
        )

    m, k = qweight_u8_mk.shape
    qgroups = k // TMAN_A8_LUT_GROUP
    weight_groups = qweight_u8_mk.reshape(m, qgroups, TMAN_A8_LUT_GROUP).to(torch.int32)
    bit_indices = []
    # 向量化构造每个 bit-plane 的 nibble index，避免 Python 三重循环。
    for bit in range(bits):
        bit_plane = (weight_groups >> bit) & 0x1
        lut_idx = (
            bit_plane[..., 0]
            | (bit_plane[..., 1] << 1)
            | (bit_plane[..., 2] << 2)
            | (bit_plane[..., 3] << 3)
        ).to(torch.uint8)
        bit_indices.append(lut_idx)
    return torch.stack(bit_indices, dim=-1).contiguous()


def precompute_w4a8_lut(x_q: torch.Tensor, lut_group_size: int = 4) -> torch.Tensor:
    # 为 A8 baseline 构造 LUT：
    # - 每 4 个激活作为一个 LUT group
    # - 对每个 group 预计算 16 个 subset-sum
    #
    # LUT 定义：
    #   lut[idx] = sum_j bit_j(idx) * x_j
    #
    # 后续 runtime baseline 会对每个 bit-plane 取一个 4bit index，
    # 然后查表恢复当前 group 的整数乘加结果。
    if x_q.dim() != 2:
        raise ValueError(f"Expected 2D quantized activation tensor, got {tuple(x_q.shape)}")
    if x_q.shape[1] % lut_group_size != 0:
        raise ValueError(
            f"Last dim {x_q.shape[1]} must be divisible by lut_group_size={lut_group_size}"
        )

    rows, k = x_q.shape
    qgroups = k // lut_group_size
    x_grouped = x_q.to(torch.int16).reshape(rows, qgroups, lut_group_size)
    lut = torch.zeros((rows, qgroups, 1 << lut_group_size), dtype=torch.int16)
    for idx in range(1 << lut_group_size):
        acc = torch.zeros((rows, qgroups), dtype=torch.int16)
        for lane in range(lut_group_size):
            if (idx >> lane) & 0x1:
                acc = acc + x_grouped[:, :, lane]
        lut[:, :, idx] = acc
    return lut


def w4a8_lut_group_dot(
    lut: torch.Tensor,
    weight_group_u4: torch.Tensor,
    bits: int = 4,
) -> torch.Tensor:
    # 使用 LUT 恢复一个 4-activation group 的整数乘加结果。
    #
    # 对称 W4 权重取值范围为 [0, 15]，对应真实有符号值：
    #   w_signed = w_u4 - 8
    #
    # 那么 group 内的乘加可以分解成：
    #   sum_j x_j * (w_j - 8)
    # = -8 * sum_j x_j + \sum_bit 2^bit * subset_sum(bitplane_idx)
    #
    # 其中：
    #   subset_sum(bitplane_idx) 由 4 个权重在该 bit-plane 上的取值共同决定，
    #   正好可以用 16 项 LUT 查出来。
    if lut.shape[-1] != 16:
        raise ValueError(f"Expected LUT size 16, got {lut.shape[-1]}")
    if weight_group_u4.numel() != 4:
        raise ValueError(f"Expected 4 weights in a LUT group, got {weight_group_u4.numel()}")

    lut_i32 = lut.to(torch.int32)
    weight_group_i32 = weight_group_u4.to(torch.int32).reshape(4)
    acc = -(1 << (bits - 1)) * lut_i32[15]
    for bit in range(bits):
        idx = 0
        for lane in range(4):
            idx |= int(((weight_group_i32[lane] >> bit) & 0x1).item()) << lane
        acc = acc + (1 << bit) * lut_i32[idx]
    return acc


def tman_linear_a8_reference(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    wf_unsqueeze_zero: torch.Tensor,
    wf_unsqueeze_neg_one: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
    bits: int,
    gptq_v2: bool = True,
) -> torch.Tensor:
    y_fp_out, _ = tman_linear_a8_reference_with_quantized_output(
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
    return y_fp_out


def tman_linear_a8_reference_with_quantized_output(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor,
    wf_unsqueeze_zero: torch.Tensor,
    wf_unsqueeze_neg_one: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
    bits: int,
    gptq_v2: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 这份 reference 定义用于统一 export fake 语义和外部轻量验证脚本。
    #
    # 语义如下：
    # 1. x_fp -> x_q
    # 2. W4 GPTQ -> w_fp
    # 3. (x_q * input_scale) @ w_fp
    # 4. y_fp -> y_q (用 output_scale)
    # 5. y_q * output_scale -> 图外最终仍看到的 FP 输出
    #
    # 这份 reference 对应的是“高层数学定义”：
    # - 先把激活按 input_scale 量化到 int8；
    # - 用真实反量化后的浮点权重参与 matmul；
    # - 再按 output_scale 重新量化/反量化。
    #
    # 它不是 runtime 真正逐步执行的实现细节，
    # 但它给 export fake / unit test / 数值审计提供了稳定的高层语义基线。
    out_features = qweight.shape[1]
    out_shape = x.shape[:-1] + (out_features,)
    x_2d = x.reshape(-1, x.shape[-1])
    x_q = symmetric_int8_quantize_with_scale(x_2d, input_scale)
    weights = dequantize_gptq_weight(
        qweight,
        scales,
        qzeros,
        g_idx,
        wf_unsqueeze_zero,
        wf_unsqueeze_neg_one,
        bits,
        gptq_v2,
    ).to(torch.float32)
    y_fp = torch.matmul(x_q.to(torch.float32) * input_scale.reshape(1), weights)
    y_q = symmetric_int8_quantize_with_scale(y_fp, output_scale)
    y_fp_out = y_q.to(torch.float32) * output_scale.reshape(1)
    return y_fp_out.reshape(out_shape).to(x.dtype), y_q.reshape(out_shape)


def tman_linear_a8_grouped_reference(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    y_fp, _ = tman_linear_a8_grouped_reference_with_quantized_output(
        x,
        qweight,
        scales,
        input_scale,
        output_scale,
        bits,
        group_size,
    )
    return y_fp


def tman_linear_a8_grouped_reference_with_quantized_output(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 这份 grouped reference 表示“当前项目目标实现的 group-wise 数学语义”，
    # 但不依赖 LUT index 张量与 runtime 布局。
    #
    # 可以把它理解成：
    # - 已经接受“group 内先做整数乘加、再乘 folded effective_scale”的设计；
    # - 但还没有引入 runtime 的 LUT/index/block 访问方式。
    #
    # 因此它正好位于：
    # high-level reference 与 runtime reference 之间，
    # 是判断“当前 runtime 是否在实现我们想要的 group-wise 数学语义”的关键中间基线。
    qweight_u8_mk = unpack_gptq_qweight_to_uint8(qweight, bits)
    weight_scales_mg = normalize_gptq_scales_layout(scales, qweight, bits)
    effective_scales_mg = fold_weight_scales_for_w4a8(
        weight_scales_mg,
        input_scale,
        output_scale,
    )

    x_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    x_q = symmetric_int8_quantize_with_scale(x_2d, input_scale)
    m, k = qweight_u8_mk.shape
    if k % group_size != 0:
        raise ValueError(f"group_size={group_size} does not divide K={k}")
    if group_size % 4 != 0:
        raise ValueError(f"group_size={group_size} must be divisible by 4")
    num_groups = k // group_size
    qgroups_per_group = group_size // 4

    x_grouped = x_q.reshape(x_q.shape[0], num_groups, qgroups_per_group, 4).to(torch.int32)
    w_grouped = (qweight_u8_mk.to(torch.int32) - (1 << (bits - 1))).reshape(
        m, num_groups, qgroups_per_group, 4
    )
    acc_qgroups = (x_grouped.unsqueeze(1) * w_grouped.unsqueeze(0)).sum(dim=-1).to(torch.float32)
    acc_grouped = acc_qgroups.sum(dim=-1)
    acc = acc_grouped.mul(effective_scales_mg.unsqueeze(0)).sum(dim=-1)

    y_q = symmetric_int8_quantize_with_scale(acc, torch.ones(1, dtype=torch.float32))
    y_fp = y_q.to(torch.float32) * output_scale.reshape(1)
    out_shape = x_shape[:-1] + (m,)
    return y_fp.reshape(out_shape).to(x.dtype), y_q.reshape(out_shape)


def tman_linear_a8_runtime_reference(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    y_fp, _ = tman_linear_a8_runtime_reference_with_quantized_output(
        x,
        qweight,
        scales,
        input_scale,
        output_scale,
        bits,
        group_size,
    )
    return y_fp


def tman_linear_a8_runtime_reference_with_quantized_output(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    input_scale: torch.Tensor,
    output_scale: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 这份 reference 不是“高层数学语义”，而是“当前 runtime baseline 的等价实现”。
    # 它和 tman_linear_a8_reference 的区别在于：
    # - 这里先构造 runtime baseline 真正消费的 static tensor；
    # - 使用 LUT 预构造 + bit-serial 查表 来恢复 group 内整数乘加；
    # - 再按 effective_scale 做 group-wise 缩放；
    # - 最后 round/clamp 成 int8，再按 output_scale 反量化。
    #
    # 这份实现的重点不是简洁，而是尽量贴近当前 runtime 的真实数据流：
    # - qweight 先转换成 grouped LUT index tensor
    # - 激活先构造 16 项 LUT
    # - 再按 block/index 把 group 内整数和恢复出来
    # - 最后乘上 folded effective_scale
    #
    # 因此它是当前项目最贴近 runtime 行为的 Python 参考实现。
    qweight_lut_indices_mgqb, effective_scales_mg = prepare_w4a8_baseline_runtime_tensors(
        qweight,
        scales,
        input_scale,
        output_scale,
        bits,
        group_size,
    )

    x_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    x_q = symmetric_int8_quantize_with_scale(x_2d, input_scale)
    lut = precompute_w4a8_lut(x_q)
    rows = x_q.shape[0]
    m, num_groups, qgroups_per_group, bits_dim = qweight_lut_indices_mgqb.shape
    if bits_dim != bits:
        raise ValueError(
            f"Weight LUT indices bits dim={bits_dim} does not match bits={bits}"
        )
    qgroups = num_groups * qgroups_per_group
    k = qgroups * 4
    if k % group_size != 0:
        raise ValueError(
            f"group_size={group_size} does not divide K={k} in runtime reference"
        )
    if group_size % 4 != 0:
        raise ValueError(f"group_size={group_size} must be divisible by 4 for LUT reference")

    # 把 [R, Q, 16] 的 LUT 和 [M, Q, bits] 的 nibble index 变成
    # [R, M, G, Qg, bits] 的查表结果，然后按 bits 权重和 group 聚合。
    lut_grouped = lut.reshape(rows, num_groups, qgroups_per_group, 16)
    indices = qweight_lut_indices_mgqb.unsqueeze(0).expand(rows, -1, -1, -1, -1).to(torch.int64)
    lut_expanded = lut_grouped.unsqueeze(1).expand(-1, m, -1, -1, -1)
    gathered = torch.gather(lut_expanded, 4, indices).to(torch.int32)

    bit_weights = (
        torch.tensor([1 << bit for bit in range(bits)], dtype=torch.int32)
        .reshape(1, 1, 1, 1, bits)
    )
    lut15 = lut_grouped[..., 15].unsqueeze(1).to(torch.int32)
    acc_qgroups = gathered.mul(bit_weights).sum(dim=-1) - ((1 << (bits - 1)) * lut15)
    acc_qgroups = acc_qgroups.to(torch.float32)

    acc_grouped = acc_qgroups.sum(dim=-1)
    acc = acc_grouped.mul(effective_scales_mg.unsqueeze(0)).sum(dim=-1)

    y_q = symmetric_int8_quantize_with_scale(acc, torch.ones(1, dtype=torch.float32))
    y_fp = y_q.to(torch.float32) * output_scale.reshape(1)
    out_shape = x_shape[:-1] + (m,)
    return y_fp.reshape(out_shape).to(x.dtype), y_q.reshape(out_shape)


class ActivationScaleRegistry:
    def __init__(self, scales: Dict[str, LinearActivationScales]) -> None:
        self._scales = scales

    @staticmethod
    def normalize_qwen3_linear_key(key: str) -> str:
        # 外部 scale 文件来自离线量化流程，key 前缀目前是 model.model.layers.*。
        # 当前静态 Qwen3 结构在导出侧使用的是 model.layers.*。
        # 这里统一把两者规约到同一个“模块路径”命名空间，供后续模块替换和 builder 查询。
        if key.endswith(".input_scale"):
            key = key[: -len(".input_scale")]
        elif key.endswith(".output_scale"):
            key = key[: -len(".output_scale")]

        if key.startswith("model.model.layers."):
            return key.replace("model.model.layers.", "model.layers.", 1)
        if key.startswith("model.layers."):
            return key
        raise KeyError(f"Unsupported activation scale key prefix: {key}")

    @classmethod
    def load_activation_scales_pt(cls, path: str) -> "ActivationScaleRegistry":
        # 这里把原始 pt 文件整理成：
        # {
        #   "model.layers.0.self_attn.q_proj": {
        #       "input_scale": tensor([..]),
        #       "output_scale": tensor([..]),
        #   }
        # }
        #
        # 后续 Python 模块替换层只需要按模块路径查询，不再依赖原始 key 格式。
        raw = torch.load(path, map_location="cpu")
        if not isinstance(raw, dict):
            raise TypeError(
                f"Activation scales file must contain a dict, got {type(raw)}"
            )

        normalized: Dict[str, Dict[str, torch.Tensor]] = {}
        for raw_key, raw_value in raw.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"Activation scale key must be str, got {type(raw_key)}")

            if not torch.is_tensor(raw_value):
                raise TypeError(
                    f"Activation scale value for {raw_key} must be tensor, got {type(raw_value)}"
                )

            if raw_value.numel() != 1:
                raise ValueError(
                    f"Activation scale tensor for {raw_key} must be scalar-like, got shape {tuple(raw_value.shape)}"
                )

            if raw_key.endswith(".input_scale"):
                scale_kind = "input_scale"
            elif raw_key.endswith(".output_scale"):
                scale_kind = "output_scale"
            else:
                raise KeyError(f"Unsupported activation scale suffix: {raw_key}")

            normalized_key = cls.normalize_qwen3_linear_key(raw_key)
            normalized.setdefault(normalized_key, {})[scale_kind] = raw_value.detach().reshape(1).to(torch.float32)

        scales: Dict[str, LinearActivationScales] = {}
        for module_path, values in normalized.items():
            if "input_scale" not in values or "output_scale" not in values:
                raise KeyError(
                    f"Activation scale entry {module_path} must contain both input_scale and output_scale"
                )
            scales[module_path] = LinearActivationScales(
                input_scale=values["input_scale"],
                output_scale=values["output_scale"],
            )
        return cls(scales)

    def get_linear_scales(self, module_path: str) -> LinearActivationScales:
        if module_path not in self._scales:
            raise KeyError(f"Missing activation scales for linear module: {module_path}")
        return self._scales[module_path]

    def has_linear_scales(self, module_path: str) -> bool:
        return module_path in self._scales

    def keys(self) -> Iterable[str]:
        return self._scales.keys()

    def validate_expected_qwen3_linears(self, num_layers: int) -> None:
        # 在正式替换模块前一次性做完整覆盖校验。
        # 这样如果 scale 文件不完整，会在编译早期失败，而不是在某一层替换时才报错。
        expected = build_expected_qwen3_linear_paths(num_layers)
        missing = sorted(expected.difference(self._scales.keys()))
        extra = sorted(set(self._scales.keys()).difference(expected))
        if missing:
            raise KeyError(
                "Activation scales file misses required Qwen3 linears: "
                + ", ".join(missing[:16])
                + (" ..." if len(missing) > 16 else "")
            )
        if extra:
            raise KeyError(
                "Activation scales file contains unexpected Qwen3 linears: "
                + ", ".join(extra[:16])
                + (" ..." if len(extra) > 16 else "")
            )
