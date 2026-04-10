#!/usr/bin/env python3
"""
一个不依赖完整 ExecuTorch 运行环境的轻量验证脚本。

用途：
1. 验证外部 activation scale 的 key 规范化逻辑；
2. 验证 W4A8 reference 数学语义函数是否能独立运行。

运行方式示例：
    python tools/tman_w4a8_reference_check.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile

import torch


def _load_tman_w4a8_module():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_path = repo_root / "backends" / "qualcomm" / "utils" / "tman_w4a8.py"
    spec = importlib.util.spec_from_file_location("tman_w4a8_standalone", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    mod = _load_tman_w4a8_module()

    print("[1/3] 验证 activation scale key 规范化...")
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        torch.save(
            {
                "model.model.layers.0.self_attn.q_proj.input_scale": torch.tensor([0.25]),
                "model.model.layers.0.self_attn.q_proj.output_scale": torch.tensor([0.5]),
            },
            tmp.name,
        )
        registry = mod.ActivationScaleRegistry.load_activation_scales_pt(tmp.name)
        scales = registry.get_linear_scales("model.layers.0.self_attn.q_proj")
        assert abs(scales.input_scale.item() - 0.25) < 1e-6
        assert abs(scales.output_scale.item() - 0.5) < 1e-6
    print("  通过")

    print("[2/3] 验证 W4A8 high-level reference 数学语义...")
    sym_pack_i32 = -2004318072  # 0x88888888 interpreted as signed int32
    one_pack_i32 = -1717986919  # 0x99999999 interpreted as signed int32
    x = torch.tensor(
        [[0.20, -0.35, 0.10, 0.05, -0.20, 0.30, -0.15, 0.40]], dtype=torch.float32
    )
    qweight = torch.full((1, 8), sym_pack_i32, dtype=torch.int32)
    scales = torch.ones((1, 8), dtype=torch.float32)
    qzeros = torch.full((1, 1), sym_pack_i32, dtype=torch.int32)
    g_idx = torch.zeros((8,), dtype=torch.int32)
    wf_unsqueeze_zero = torch.tensor([[0, 4, 8, 12, 16, 20, 24, 28]], dtype=torch.int32)
    wf_unsqueeze_neg_one = torch.tensor(
        [[0], [4], [8], [12], [16], [20], [24], [28]], dtype=torch.int32
    )
    input_scale = torch.tensor([0.1], dtype=torch.float32)
    output_scale = torch.tensor([0.2], dtype=torch.float32)
    y = mod.tman_linear_a8_reference(
        x,
        qweight,
        scales,
        qzeros,
        g_idx,
        wf_unsqueeze_zero,
        wf_unsqueeze_neg_one,
        input_scale,
        output_scale,
        4,
    )
    assert torch.allclose(y, torch.zeros((1, 8), dtype=torch.float32))
    print("  通过")

    print("[3/3] 验证 runtime baseline reference 与 high-level reference 一致...")
    x2 = torch.tensor(
        [[0.1, 0.2, -0.1, 0.0, 0.1, 0.2, 0.0, -0.1]], dtype=torch.float32
    )
    qweight2 = torch.full((1, 8), one_pack_i32, dtype=torch.int32)
    y_high = mod.tman_linear_a8_reference(
        x2,
        qweight2,
        scales,
        qzeros,
        g_idx,
        wf_unsqueeze_zero,
        wf_unsqueeze_neg_one,
        input_scale,
        output_scale,
        4,
    )
    y_runtime = mod.tman_linear_a8_runtime_reference(
        x2,
        qweight2,
        scales,
        input_scale,
        output_scale,
        4,
        8,
    )
    assert torch.allclose(y_high, y_runtime)
    print("  通过")

    print("W4A8 轻量 reference 校验完成。")


if __name__ == "__main__":
    main()
