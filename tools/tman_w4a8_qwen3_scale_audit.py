#!/usr/bin/env python3
"""
审计 Qwen3 W4A8 外部 activation scale 文件是否覆盖完整。

这个脚本不依赖完整 ExecuTorch 运行环境，只依赖 torch。

示例：
    python tools/tman_w4a8_qwen3_scale_audit.py \
      --scales_pt /home/lmx/Test/GPTQ_Opus/Qwen3-1.7B_w4a8_gptq/activation_scales_a8_per-tensor_sym.pt \
      --num_layers 28
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys


def _load_tman_w4a8_module():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_path = repo_root / "backends" / "qualcomm" / "utils" / "tman_w4a8.py"
    spec = importlib.util.spec_from_file_location("tman_w4a8_audit_mod", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales_pt", required=True, type=str)
    parser.add_argument("--num_layers", default=28, type=int)
    args = parser.parse_args()

    mod = _load_tman_w4a8_module()
    registry = mod.ActivationScaleRegistry.load_activation_scales_pt(args.scales_pt)
    expected = mod.build_expected_qwen3_linear_paths(args.num_layers)
    actual = set(registry.keys())

    missing = sorted(expected.difference(actual))
    extra = sorted(actual.difference(expected))

    print(f"总目标 linear 数: {len(expected)}")
    print(f"scale 文件命中数: {len(actual)}")
    print(f"缺失数: {len(missing)}")
    print(f"额外数: {len(extra)}")
    if missing:
        print("前 16 个缺失项:")
        for item in missing[:16]:
            print(f"  - {item}")
    if extra:
        print("前 16 个额外项:")
        for item in extra[:16]:
            print(f"  - {item}")

    if not missing and not extra:
        print("scale 文件覆盖完整，路径规范化正确。")


if __name__ == "__main__":
    main()
