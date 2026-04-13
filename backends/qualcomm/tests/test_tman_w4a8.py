# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import pathlib
import sys
import tempfile
import unittest

import torch


def _load_tman_w4a8_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    module_path = repo_root / "backends" / "qualcomm" / "utils" / "tman_w4a8.py"
    spec = importlib.util.spec_from_file_location("tman_w4a8_test_mod", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TMAN_W4A8_MOD = _load_tman_w4a8_module()


class TestTMANW4A8Utils(unittest.TestCase):
    def test_precompute_w4a8_lut_subset_sum_definition(self):
        # lut[idx] = sum_j bit_j(idx) * x_j
        # 对输入 [1,2,3,4]：
        # - idx=0b0000 -> 0
        # - idx=0b0101 -> 1 + 3 = 4
        # - idx=0b1111 -> 1 + 2 + 3 + 4 = 10
        x_q = torch.tensor([[1, 2, 3, 4]], dtype=torch.int8)
        lut = TMAN_W4A8_MOD.precompute_w4a8_lut(x_q)
        self.assertEqual(tuple(lut.shape), (1, 1, 16))
        self.assertEqual(int(lut[0, 0, 0].item()), 0)
        self.assertEqual(int(lut[0, 0, 0b0101].item()), 4)
        self.assertEqual(int(lut[0, 0, 0b1111].item()), 10)

    def test_build_w4a8_weight_lut_indices_definition(self):
        # 对 4 个 W4 权重 [1,2,4,8]：
        # - bit0 平面的 nibble index = 0001b = 1
        # - bit1 平面的 nibble index = 0010b = 2
        # - bit2 平面的 nibble index = 0100b = 4
        # - bit3 平面的 nibble index = 1000b = 8
        w = torch.tensor([[1, 2, 4, 8]], dtype=torch.uint8)
        indices = TMAN_W4A8_MOD.build_w4a8_weight_lut_indices(w, 4)
        self.assertEqual(tuple(indices.shape), (1, 1, 4))
        self.assertEqual(indices[0, 0].tolist(), [1, 2, 4, 8])

    def test_pack_w4a8_weight_lut_indices_for_hvx_preserves_block_order(self):
        grouped = torch.arange(32, dtype=torch.uint8).reshape(1, 1, 8, 4)
        packed = TMAN_W4A8_MOD.pack_w4a8_weight_lut_indices_for_hvx(grouped, 4)
        self.assertEqual(tuple(packed.shape), (1, 1, 2, 16))
        self.assertEqual(packed[0, 0, 0].tolist(), list(range(16)))
        self.assertEqual(packed[0, 0, 1].tolist(), list(range(16, 32)))

    def test_build_w4a8_lut_scratch_shape_uses_vector_blocks(self):
        shape = TMAN_W4A8_MOD.build_w4a8_lut_scratch_shape(
            num_rows=2,
            num_groups=3,
            qgroups_per_group=8,
        )
        self.assertEqual(tuple(shape), (1, 2, 3, 2, 64))

    def test_build_w4a8_lut_scratch_packed_shape_flattens_block_axes(self):
        shape = TMAN_W4A8_MOD.build_w4a8_lut_scratch_packed_shape(
            num_rows=2,
            num_groups=3,
            qgroups_per_group=8,
        )
        self.assertEqual(tuple(shape), (1, 2, 3, 128))

    def test_pack_w4a8_weight_lut_indices_for_qnn_flattens_block_axes(self):
        grouped = torch.arange(128, dtype=torch.uint8).reshape(1, 1, 32, 4)
        packed = TMAN_W4A8_MOD.pack_w4a8_weight_lut_indices_for_qnn(grouped, 4)
        self.assertEqual(tuple(packed.shape), (1, 1, 1, 128))
        self.assertEqual(packed.reshape(-1).tolist(), list(range(128)))

    def test_pack_w4a8_weight_lut_indices_for_qnn_int32_preserves_raw_bytes(self):
        grouped = torch.arange(128, dtype=torch.uint8).reshape(1, 1, 32, 4)
        packed = TMAN_W4A8_MOD.pack_w4a8_weight_lut_indices_for_qnn_int32(grouped, 4)
        self.assertEqual(tuple(packed.shape), (1, 1, 1, 32))
        self.assertEqual(packed.dtype, torch.int32)
        self.assertEqual(
            packed.view(torch.uint8).reshape(-1).tolist(),
            list(range(128)),
        )

    def test_pack_w4a8_effective_scales_for_qnn_pads_rows_to_128_bytes(self):
        effective_scales = torch.arange(32, dtype=torch.float32).reshape(2, 16)
        packed = TMAN_W4A8_MOD.pack_w4a8_effective_scales_for_qnn(effective_scales)
        self.assertEqual(tuple(packed.shape), (1, 2, 1, 32))
        self.assertTrue(torch.allclose(packed[0, :, 0, :16], effective_scales))
        self.assertTrue(torch.allclose(packed[0, :, 0, 16:], torch.zeros((2, 16))))

    def test_pack_w4a8_effective_scales_for_qnn_int32_preserves_raw_f32_bits(self):
        effective_scales = torch.arange(32, dtype=torch.float32).reshape(2, 16)
        packed = TMAN_W4A8_MOD.pack_w4a8_effective_scales_for_qnn_int32(
            effective_scales
        )
        self.assertEqual(tuple(packed.shape), (1, 2, 1, 32))
        self.assertEqual(packed.dtype, torch.int32)
        self.assertTrue(
            torch.allclose(
                packed.view(torch.float32)[0, :, 0, :16],
                effective_scales,
            )
        )

    def test_validate_w4a8_target_config_accepts_project_target(self):
        TMAN_W4A8_MOD.validate_w4a8_target_config(
            bits=4,
            group_size=128,
            symmetric=True,
        )

    def test_validate_w4a8_target_config_rejects_non_target_group_size(self):
        with self.assertRaises(ValueError):
            TMAN_W4A8_MOD.validate_w4a8_target_config(
                bits=4,
                group_size=64,
                symmetric=True,
            )

    def test_activation_scale_registry_normalizes_qwen3_keys(self):
        # 这个用例锁定“外部量化流程产出的 key”和“当前静态 Qwen3 模块路径”
        # 之间的规范化规则，避免后续改动路径时把 scale 映射 silently 搞错。
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            torch.save(
                {
                    "model.model.layers.0.self_attn.q_proj.input_scale": torch.tensor([0.25]),
                    "model.model.layers.0.self_attn.q_proj.output_scale": torch.tensor([0.5]),
                },
                tmp.name,
            )
            registry = TMAN_W4A8_MOD.ActivationScaleRegistry.load_activation_scales_pt(tmp.name)
            scales = registry.get_linear_scales("model.layers.0.self_attn.q_proj")
            self.assertAlmostEqual(scales.input_scale.item(), 0.25)
            self.assertAlmostEqual(scales.output_scale.item(), 0.5)

    def test_infer_gptq_tensor_layout_for_v1_symmetric_pack(self):
        qweight = torch.zeros((256, 8), dtype=torch.int32)
        scales = torch.ones((16, 8), dtype=torch.float32)
        qzeros = torch.full((16, 1), 2004318071, dtype=torch.int32)  # 0x77777777
        layout = TMAN_W4A8_MOD.infer_gptq_tensor_layout(qweight, scales, qzeros, False)
        self.assertEqual(layout.bits, 4)
        self.assertEqual(layout.group_size, 128)
        self.assertEqual(layout.k, 2048)
        self.assertEqual(layout.m, 8)
        self.assertTrue(layout.symmetric)

    def test_tman_linear_a8_reference_matches_expected_fp_output(self):
        # 这里构造一个最小 reference case：
        # - 权重反量化后等于 0
        # - 因此无论输入如何，最终输出都应当稳定为 0
        #
        # 这个测试的意义是先锁住 custom op A8 参考语义能跑通，
        # 后面再逐步增加更复杂的数值用例。
        sym_pack_i32 = -2004318072  # 0x88888888 interpreted as signed int32
        x = torch.tensor([[0.20, -0.35, 0.10, 0.05, -0.20, 0.30, -0.15, 0.40]], dtype=torch.float32)
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

        y = TMAN_W4A8_MOD.tman_linear_a8_reference(
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
        expected = torch.zeros((1, 8), dtype=torch.float32)
        self.assertTrue(torch.allclose(y, expected))

    def test_tman_linear_a8_reference_nonzero_case(self):
        # 这里构造一个非零 reference case：
        # - 权重反量化后为全 1
        # - 输入按 input_scale=0.1 量化后得到 [1,2,-1,0,1,2,0,-1]
        # - 单个输出通道的浮点和为 0.4
        # - 再按 output_scale=0.2 量化/反量化后，最终输出应保持为 0.4
        sym_pack_i32 = -2004318072  # 0x88888888 interpreted as signed int32
        one_pack_i32 = -1717986919  # 0x99999999 interpreted as signed int32
        x = torch.tensor([[0.1, 0.2, -0.1, 0.0, 0.1, 0.2, 0.0, -0.1]], dtype=torch.float32)
        qweight = torch.full((1, 8), one_pack_i32, dtype=torch.int32)
        scales = torch.ones((1, 8), dtype=torch.float32)
        qzeros = torch.full((1, 1), sym_pack_i32, dtype=torch.int32)
        g_idx = torch.zeros((8,), dtype=torch.int32)
        wf_unsqueeze_zero = torch.tensor([[0, 4, 8, 12, 16, 20, 24, 28]], dtype=torch.int32)
        wf_unsqueeze_neg_one = torch.tensor(
            [[0], [4], [8], [12], [16], [20], [24], [28]], dtype=torch.int32
        )
        input_scale = torch.tensor([0.1], dtype=torch.float32)
        output_scale = torch.tensor([0.2], dtype=torch.float32)

        y = TMAN_W4A8_MOD.tman_linear_a8_reference(
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
        expected = torch.full((1, 8), 0.4, dtype=torch.float32)
        self.assertTrue(torch.allclose(y, expected))

    def test_runtime_reference_matches_high_level_reference(self):
        # 这个用例校验“当前 runtime baseline 等价实现”和“高层 reference 语义”一致。
        # 它的意义是约束：
        # 1. builder 侧生成的 baseline static tensor 语义没有偏掉；
        # 2. runtime A8 baseline 的 effective_scale 路径和高层定义保持一致。
        sym_pack_i32 = -2004318072
        one_pack_i32 = -1717986919
        x = torch.tensor([[0.1, 0.2, -0.1, 0.0, 0.1, 0.2, 0.0, -0.1]], dtype=torch.float32)
        qweight = torch.full((1, 8), one_pack_i32, dtype=torch.int32)
        scales = torch.ones((1, 8), dtype=torch.float32)
        qzeros = torch.full((1, 1), sym_pack_i32, dtype=torch.int32)
        g_idx = torch.zeros((8,), dtype=torch.int32)
        wf_unsqueeze_zero = torch.tensor([[0, 4, 8, 12, 16, 20, 24, 28]], dtype=torch.int32)
        wf_unsqueeze_neg_one = torch.tensor(
            [[0], [4], [8], [12], [16], [20], [24], [28]], dtype=torch.int32
        )
        input_scale = torch.tensor([0.1], dtype=torch.float32)
        output_scale = torch.tensor([0.2], dtype=torch.float32)

        y_high_level = TMAN_W4A8_MOD.tman_linear_a8_reference(
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
        y_runtime = TMAN_W4A8_MOD.tman_linear_a8_runtime_reference(
            x,
            qweight,
            scales,
            input_scale,
            output_scale,
            4,
            8,
        )
        self.assertTrue(torch.allclose(y_high_level, y_runtime))

    def test_runtime_reference_matches_grouped_reference(self):
        # runtime baseline 是 grouped reference 的 LUT/indexed 实现版本。
        # 两者应该严格一致。
        sym_pack_i32 = -2004318072
        one_pack_i32 = -1717986919
        x = torch.tensor([[0.1, 0.2, -0.1, 0.0, 0.1, 0.2, 0.0, -0.1]], dtype=torch.float32)
        qweight = torch.full((1, 8), one_pack_i32, dtype=torch.int32)
        scales = torch.ones((1, 8), dtype=torch.float32)
        input_scale = torch.tensor([0.1], dtype=torch.float32)
        output_scale = torch.tensor([0.2], dtype=torch.float32)

        y_grouped = TMAN_W4A8_MOD.tman_linear_a8_grouped_reference(
            x,
            qweight,
            scales,
            input_scale,
            output_scale,
            4,
            8,
        )
        y_runtime = TMAN_W4A8_MOD.tman_linear_a8_runtime_reference(
            x,
            qweight,
            scales,
            input_scale,
            output_scale,
            4,
            8,
        )
        self.assertTrue(torch.allclose(y_grouped, y_runtime))

    def test_high_level_reference_supports_gptq_v2_false(self):
        # 这个用例覆盖 gptqmodel v2 之前的 qzeros 表示方式。
        # 对称 W4 情况下，旧格式常见 qzeros pack 为 0x77777777，
        # 只有在解包后执行 +1，才能恢复正确 zero offset。
        qzeros_v1_i32 = 2004318071  # 0x77777777 interpreted as signed int32
        one_pack_i32 = -1717986919  # 0x99999999
        x = torch.tensor([[0.1, 0.2, -0.1, 0.0, 0.1, 0.2, 0.0, -0.1]], dtype=torch.float32)
        qweight = torch.full((1, 8), one_pack_i32, dtype=torch.int32)
        scales = torch.ones((1, 8), dtype=torch.float32)
        qzeros = torch.full((1, 1), qzeros_v1_i32, dtype=torch.int32)
        g_idx = torch.zeros((8,), dtype=torch.int32)
        wf_unsqueeze_zero = torch.tensor([[0, 4, 8, 12, 16, 20, 24, 28]], dtype=torch.int32)
        wf_unsqueeze_neg_one = torch.tensor(
            [[0], [4], [8], [12], [16], [20], [24], [28]], dtype=torch.int32
        )
        input_scale = torch.tensor([0.1], dtype=torch.float32)
        output_scale = torch.tensor([0.2], dtype=torch.float32)

        y = TMAN_W4A8_MOD.tman_linear_a8_reference(
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
            False,
        )
        expected = torch.full((1, 8), 0.4, dtype=torch.float32)
        self.assertTrue(torch.allclose(y, expected))


if __name__ == "__main__":
    unittest.main()
