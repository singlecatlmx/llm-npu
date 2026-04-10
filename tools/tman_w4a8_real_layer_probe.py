#!/usr/bin/env python3
"""
使用真实 Qwen3 GPTQ safetensors 和外部 activation scale，
对单个 linear 做 W4A8 局部 probe。

这个脚本不依赖完整 ExecuTorch 运行环境，适合当前机器直接做局部数值核对。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys

import torch
from safetensors import safe_open


def _load_tman_w4a8_module():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_path = repo_root / "backends" / "qualcomm" / "utils" / "tman_w4a8.py"
    spec = importlib.util.spec_from_file_location("tman_w4a8_probe_mod", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_tensor_map(weights_file: str, module_prefix: str):
    wanted = {
        "qweight": f"{module_prefix}.qweight",
        "qzeros": f"{module_prefix}.qzeros",
        "scales": f"{module_prefix}.scales",
        "g_idx": f"{module_prefix}.g_idx",
    }
    result = {}
    with safe_open(weights_file, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for name, key in wanted.items():
            if key not in keys:
                raise KeyError(f"Missing tensor in safetensors: {key}")
            result[name] = f.get_tensor(key)
    return result


def _make_module_input(module_path: str, in_features: int, input_scale: float, base_seed: int):
    seed_bytes = hashlib.sha256(module_path.encode("utf-8")).digest()[:8]
    module_seed = int.from_bytes(seed_bytes, "little") % (2**31)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(base_seed + module_seed)
    return torch.randn((1, in_features), dtype=torch.float32, generator=generator) * input_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_file", required=True, type=str)
    parser.add_argument("--scales_pt", required=True, type=str)
    parser.add_argument(
        "--module_path",
        default="model.layers.0.self_attn.q_proj",
        type=str,
        help="Module path such as model.layers.0.self_attn.q_proj",
    )
    parser.add_argument(
        "--out_features_slice",
        default=8,
        type=int,
        help="Only probe the first N output channels to keep the pure Python runtime baseline tractable. Must be divisible by 8 for W4 qzeros packing.",
    )
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    mod = _load_tman_w4a8_module()
    registry = mod.ActivationScaleRegistry.load_activation_scales_pt(args.scales_pt)
    linear_scales = registry.get_linear_scales(args.module_path)

    tensors = _load_tensor_map(args.weights_file, args.module_path)
    qweight = tensors["qweight"]
    qzeros = tensors["qzeros"]
    scales = tensors["scales"]
    g_idx = tensors["g_idx"]

    bits = 4
    weights_path = pathlib.Path(args.weights_file)
    quant_cfg_path = weights_path.with_name("quantize_config.json")
    gptq_v2 = True
    if quant_cfg_path.is_file():
        quant_cfg = json.loads(quant_cfg_path.read_text())
        gptq_v2 = bool(quant_cfg.get("meta", {}).get("v2", True))

    layout = mod.infer_gptq_tensor_layout(qweight, scales, qzeros, gptq_v2)

    qweight, qzeros, scales = mod.slice_gptq_output_channels(
        qweight,
        qzeros,
        scales,
        args.out_features_slice,
        bits,
    )

    group_size = layout.group_size
    wf_unsqueeze_zero = torch.tensor([[0, 4, 8, 12, 16, 20, 24, 28]], dtype=torch.int32)
    wf_unsqueeze_neg_one = torch.tensor(
        [[0], [4], [8], [12], [16], [20], [24], [28]], dtype=torch.int32
    )

    in_features = qweight.shape[0] * (32 // bits)
    x = _make_module_input(
        args.module_path,
        in_features,
        linear_scales.input_scale.item(),
        args.seed,
    )

    y_high, y_high_q = mod.tman_linear_a8_reference_with_quantized_output(
        x,
        qweight,
        scales,
        qzeros,
        g_idx,
        wf_unsqueeze_zero,
        wf_unsqueeze_neg_one,
        linear_scales.input_scale,
        linear_scales.output_scale,
        bits,
        gptq_v2,
    )
    y_grouped, y_grouped_q = mod.tman_linear_a8_grouped_reference_with_quantized_output(
        x,
        qweight,
        scales,
        linear_scales.input_scale,
        linear_scales.output_scale,
        bits,
        group_size,
    )
    y_runtime, y_runtime_q = mod.tman_linear_a8_runtime_reference_with_quantized_output(
        x,
        qweight,
        scales,
        linear_scales.input_scale,
        linear_scales.output_scale,
        bits,
        group_size,
    )
    qweight_u8_mk, effective_scales_mg = mod.prepare_w4a8_baseline_runtime_tensors(
        qweight,
        scales,
        linear_scales.input_scale,
        linear_scales.output_scale,
        bits,
        group_size,
    )

    max_abs_diff_high_runtime = (y_high - y_runtime).abs().max().item()
    max_abs_diff_grouped_runtime = (y_grouped - y_runtime).abs().max().item()
    max_int8_bin_diff_high_runtime = (
        y_high_q.to(torch.int16) - y_runtime_q.to(torch.int16)
    ).abs().max().item()
    max_int8_bin_diff_grouped_runtime = (
        y_grouped_q.to(torch.int16) - y_runtime_q.to(torch.int16)
    ).abs().max().item()
    print(f"module_path: {args.module_path}")
    print(f"in_features: {in_features}")
    print(f"out_features: {qweight.shape[1]}")
    print(f"group_size: {group_size}")
    print(f"gptq_v2: {gptq_v2}")
    print(f"layout_bits: {layout.bits}")
    print(f"layout_k: {layout.k}")
    print(f"layout_m: {layout.m}")
    print(f"layout_symmetric: {layout.symmetric}")
    print(f"input_scale: {float(linear_scales.input_scale.item())}")
    print(f"output_scale: {float(linear_scales.output_scale.item())}")
    print(f"runtime_qweight_shape: {tuple(qweight_u8_mk.shape)}")
    print(f"runtime_effective_scales_shape: {tuple(effective_scales_mg.shape)}")
    print(f"max_abs_diff(high_level, runtime_baseline): {max_abs_diff_high_runtime}")
    print(f"max_abs_diff(grouped, runtime_baseline): {max_abs_diff_grouped_runtime}")
    print(f"max_int8_bin_diff(high_level, runtime_baseline): {int(max_int8_bin_diff_high_runtime)}")
    print(f"max_int8_bin_diff(grouped, runtime_baseline): {int(max_int8_bin_diff_grouped_runtime)}")
    print("high_level_sample:", y_high[0, : min(8, y_high.shape[-1])].tolist())
    print("grouped_sample:", y_grouped[0, : min(8, y_grouped.shape[-1])].tolist())
    print("runtime_sample:", y_runtime[0, : min(8, y_runtime.shape[-1])].tolist())


if __name__ == "__main__":
    main()
