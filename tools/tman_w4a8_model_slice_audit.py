#!/usr/bin/env python3
"""
对真实 Qwen3 GPTQ 权重文件做批量 W4A8 局部审计。

审计内容：
1. activation scale 覆盖是否完整
2. 每个 linear 的 GPTQ 布局是否满足当前 W4A8 假设
3. 对每个 linear 的前 N 个输出通道，比较
   high-level reference 与 runtime baseline reference 的差异

这个脚本适合当前没有完整 ExecuTorch 运行环境的机器做批量静态/局部数值审计。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import time
from collections import defaultdict

import torch
from safetensors import safe_open


def _load_tman_w4a8_module():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_path = repo_root / "backends" / "qualcomm" / "utils" / "tman_w4a8.py"
    spec = importlib.util.spec_from_file_location("tman_w4a8_model_audit_mod", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_tensor_map(reader, module_prefix: str):
    wanted = {
        "qweight": f"{module_prefix}.qweight",
        "qzeros": f"{module_prefix}.qzeros",
        "scales": f"{module_prefix}.scales",
        "g_idx": f"{module_prefix}.g_idx",
    }
    result = {}
    keys = set(reader.keys())
    for name, key in wanted.items():
        if key not in keys:
            raise KeyError(f"Missing tensor in safetensors: {key}")
        result[name] = reader.get_tensor(key)
    return result


def _load_gptq_v2_flag(weights_file: str) -> bool:
    quant_cfg_path = pathlib.Path(weights_file).with_name("quantize_config.json")
    if not quant_cfg_path.is_file():
        return True
    quant_cfg = json.loads(quant_cfg_path.read_text())
    return bool(quant_cfg.get("meta", {}).get("v2", True))


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
    parser.add_argument("--num_layers", default=28, type=int)
    parser.add_argument("--out_features_slice", default=8, type=int)
    parser.add_argument("--start_index", default=0, type=int)
    parser.add_argument(
        "--max_modules",
        default=0,
        type=int,
        help="0 means audit all target modules",
    )
    parser.add_argument(
        "--print_every",
        default=1,
        type=int,
        help="Print one line every N audited modules",
    )
    parser.add_argument(
        "--max_abs_diff_tol",
        default=0.0,
        type=float,
        help="If positive, any module with diff larger than this is counted as mismatch",
    )
    parser.add_argument(
        "--max_int8_bin_diff_tol",
        default=0,
        type=int,
        help="If positive, any module with quantized output bin diff larger than this is counted as mismatch",
    )
    parser.add_argument(
        "--mismatch_mode",
        choices=["float_or_bin", "bin_only"],
        default="float_or_bin",
        type=str,
        help="float_or_bin: use either float diff or int8 bin diff to判定异常; bin_only: only use int8 bin diff tolerance",
    )
    parser.add_argument(
        "--json_out",
        default=None,
        type=str,
        help="Optional path to dump a JSON audit report",
    )
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    mod = _load_tman_w4a8_module()
    registry = mod.ActivationScaleRegistry.load_activation_scales_pt(args.scales_pt)
    registry.validate_expected_qwen3_linears(args.num_layers)
    ordered_modules = mod.build_ordered_qwen3_linear_paths(args.num_layers)
    ordered_modules = ordered_modules[args.start_index :]
    if args.max_modules > 0:
        ordered_modules = ordered_modules[: args.max_modules]

    gptq_v2 = _load_gptq_v2_flag(args.weights_file)
    checked = 0
    max_diff = 0.0
    worst_module = None
    mismatched_modules = []
    per_layer_stats = defaultdict(list)
    report_entries = []
    start_ts = time.time()

    print(f"weights_file: {args.weights_file}")
    print(f"scales_pt: {args.scales_pt}")
    print(f"gptq_v2: {gptq_v2}")
    print(f"start_index: {args.start_index}")
    print(f"modules_to_check: {len(ordered_modules)}")
    print(f"out_features_slice: {args.out_features_slice}")
    print(f"print_every: {args.print_every}")

    with safe_open(args.weights_file, framework="pt", device="cpu") as reader:
        for module_path in ordered_modules:
            module_start_ts = time.time()
            tensors = _load_tensor_map(reader, module_path)
            qweight = tensors["qweight"]
            qzeros = tensors["qzeros"]
            scales = tensors["scales"]
            g_idx = tensors["g_idx"]

            layout = mod.infer_gptq_tensor_layout(qweight, scales, qzeros, gptq_v2)
            if layout.bits != 4 or not layout.symmetric:
                raise RuntimeError(
                    f"{module_path} layout unsupported for current W4A8 assumptions: "
                    f"{layout}"
                )

            qweight, qzeros, scales = mod.slice_gptq_output_channels(
                qweight,
                qzeros,
                scales,
                args.out_features_slice,
                layout.bits,
            )

            linear_scales = registry.get_linear_scales(module_path)
            wf_unsqueeze_zero = torch.tensor([[0, 4, 8, 12, 16, 20, 24, 28]], dtype=torch.int32)
            wf_unsqueeze_neg_one = torch.tensor(
                [[0], [4], [8], [12], [16], [20], [24], [28]], dtype=torch.int32
            )

            x = _make_module_input(
                module_path,
                layout.k,
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
                layout.bits,
                gptq_v2,
            )
            y_grouped, y_grouped_q = mod.tman_linear_a8_grouped_reference_with_quantized_output(
                x,
                qweight,
                scales,
                linear_scales.input_scale,
                linear_scales.output_scale,
                layout.bits,
                layout.group_size,
            )
            y_runtime, y_runtime_q = mod.tman_linear_a8_runtime_reference_with_quantized_output(
                x,
                qweight,
                scales,
                linear_scales.input_scale,
                linear_scales.output_scale,
                layout.bits,
                layout.group_size,
            )
            diff = (y_high - y_runtime).abs().max().item()
            grouped_runtime_diff = (y_grouped - y_runtime).abs().max().item()
            qdiff = (y_high_q.to(torch.int16) - y_runtime_q.to(torch.int16)).abs().max().item()
            qdiff_grouped_runtime = (
                y_grouped_q.to(torch.int16) - y_runtime_q.to(torch.int16)
            ).abs().max().item()
            checked += 1
            if diff > max_diff:
                max_diff = diff
                worst_module = module_path

            layer_name = ".".join(module_path.split(".")[:3])
            per_layer_stats[layer_name].append(diff)
            entry = {
                "module_path": module_path,
                "group_size": layout.group_size,
                "bits": layout.bits,
                "max_abs_diff": diff,
                "max_abs_diff_grouped_runtime": grouped_runtime_diff,
                "max_int8_bin_diff": int(qdiff),
                "max_int8_bin_diff_grouped_runtime": int(qdiff_grouped_runtime),
                "elapsed_sec": time.time() - module_start_ts,
            }
            report_entries.append(entry)
            is_mismatch = False
            if args.mismatch_mode == "float_or_bin":
                is_mismatch = (
                    diff > args.max_abs_diff_tol
                    or qdiff > args.max_int8_bin_diff_tol
                )
            else:
                is_mismatch = qdiff > args.max_int8_bin_diff_tol

            if is_mismatch:
                mismatched_modules.append(entry)

            if checked % max(args.print_every, 1) == 0:
                print(
                    f"[{checked:03d}] {module_path} | "
                    f"group_size={layout.group_size} | "
                    f"max_abs_diff={diff:.10f} | "
                    f"max_abs_diff_grouped_runtime={grouped_runtime_diff:.10f} | "
                    f"max_int8_bin_diff={int(qdiff)} | "
                    f"max_int8_bin_diff_grouped_runtime={int(qdiff_grouped_runtime)} | "
                    f"time={time.time() - module_start_ts:.3f}s"
                )

    print("----")
    print(f"checked_modules: {checked}")
    print(f"worst_module: {worst_module}")
    print(f"max_abs_diff: {max_diff}")
    print(
        f"mismatched_modules(mode={args.mismatch_mode}, "
        f"float_tol={args.max_abs_diff_tol}, "
        f"qdiff_tol={args.max_int8_bin_diff_tol}): {len(mismatched_modules)}"
    )
    for layer_name, diffs in sorted(per_layer_stats.items()):
        print(
            f"layer_summary {layer_name} | "
            f"count={len(diffs)} | max_abs_diff={max(diffs):.10f}"
        )
    print(f"elapsed_sec: {time.time() - start_ts:.3f}")

    if args.json_out:
        report = {
            "weights_file": args.weights_file,
            "scales_pt": args.scales_pt,
            "gptq_v2": gptq_v2,
            "checked_modules": checked,
            "max_abs_diff": max_diff,
            "worst_module": worst_module,
            "mismatch_mode": args.mismatch_mode,
            "max_abs_diff_tol": args.max_abs_diff_tol,
            "max_int8_bin_diff_tol": args.max_int8_bin_diff_tol,
            "mismatched_modules": mismatched_modules,
            "entries": report_entries,
        }
        pathlib.Path(args.json_out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
        print(f"json_report: {args.json_out}")


if __name__ == "__main__":
    main()
