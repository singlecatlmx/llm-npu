# -*- coding: utf-8 -*-

# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import types
from functools import partial
from typing import Dict, Optional, Set, Tuple

import torch
from executorch.backends.qualcomm._passes import ConvertBmmToMatmul
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
	get_capture_program_passes,
)
from executorch.backends.qualcomm.quantizer.custom_annotation import (
	annotate_matmul_16a8w,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.backends.qualcomm.utils.constants import QCOM_PASS_ACTIVATE_KEY
from executorch.backends.qualcomm.utils.utils import (
	convert_linear_to_conv2d,
	convert_linear_to_qlinear,
	convert_linear_to_tman_prefill_linear,
	generate_htp_compiler_spec,
	generate_qnn_executorch_compiler_spec,
	get_soc_to_chipset_map,
	to_edge_transform_and_lower_to_qnn,
)
from executorch.devtools.backend_debug import print_delegation_info
from executorch.examples.models.llama.rope import precompute_freqs_cis
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from gptqmodel import GPTQModel
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e


FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.getLogger().setLevel(logging.INFO)


# Match qwen3.py import behavior so this script can be run directly.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

from model.configuration_qwen3 import Qwen3Config
from model.static_qwen3 import Qwen3Attention
from utils import make_quantizer, setup_common_args_and_variables


def _get_skip_node_op_set(skip_delegate_node_ops: Optional[str] = None) -> Set[str]:
	skip_node_op_set = {"llama.fallback.default", "aten.rms_norm.default"}
	if skip_delegate_node_ops:
		skip_node_op_set.update(
			op.strip() for op in skip_delegate_node_ops.split(",") if op.strip()
		)
	return skip_node_op_set


def _get_skip_node_id_set(skip_delegate_node_ids: Optional[str] = None) -> Set[str]:
	if not skip_delegate_node_ids:
		return set()
	return {
		node_id.strip()
		for node_id in skip_delegate_node_ids.split(",")
		if node_id.strip()
	}


def _build_prefill_attention_mask(ar_len: int, max_seq_len: int) -> torch.Tensor:
	atten_mask = torch.full((ar_len, ar_len), torch.tensor(-255.0))
	mask_cond = torch.arange(atten_mask.size(-1))
	atten_mask.masked_fill_(
		mask_cond < (mask_cond + 1).view(atten_mask.size(-1), 1),
		0,
	)
	if max_seq_len != ar_len:
		atten_mask = torch.cat(
			[
				torch.ones(ar_len, max_seq_len - ar_len) * -255.0,
				atten_mask,
			],
			dim=-1,
		)
	return atten_mask.unsqueeze(0)


def _build_qlinear_cls_and_state_dict(model_dir: str):
	qmodel = GPTQModel.from_quantized(model_dir)
	state_dict = qmodel.model.state_dict()
	qcfg = qmodel.quantize_config
	if qcfg.desc_act:
		raise RuntimeError("desc_act=True is unsupported right now.")

	try:
		from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
	except ImportError as error:
		raise RuntimeError(f"Failed to import TorchQuantLinear: {error}") from error

	qlinear_cls = partial(
		TorchQuantLinear,
		bits=qcfg.bits,
		group_size=qcfg.group_size,
		desc_act=qcfg.desc_act,
		sym=qcfg.sym,
		pack_dtype=qcfg.pack_dtype,
		device=qcfg.device,
		adapter=qcfg.adapter,
	)
	return qlinear_cls, state_dict


def _extract_attention_state_dict(
	state_dict: Dict[str, torch.Tensor], layer_idx: int
) -> Dict[str, torch.Tensor]:
	prefix_candidates = [
		f"model.layers.{layer_idx}.self_attn.",
		f"layers.{layer_idx}.self_attn.",
		f"model.model.layers.{layer_idx}.self_attn.",
	]

	for prefix in prefix_candidates:
		matched = {
			key[len(prefix) :]: value
			for key, value in state_dict.items()
			if key.startswith(prefix)
		}
		if matched:
			return matched

	dynamic_marker = f"layers.{layer_idx}.self_attn."
	for key in state_dict.keys():
		if dynamic_marker in key:
			prefix = key[: key.index(dynamic_marker)] + dynamic_marker
			return {
				full_key[len(prefix) :]: value
				for full_key, value in state_dict.items()
				if full_key.startswith(prefix)
			}

	attn_keys = [key for key in state_dict.keys() if ".self_attn." in key]
	raise RuntimeError(
		"Failed to find self_attn keys for selected layer. "
		f"layer_idx={layer_idx}, sample_attn_keys={attn_keys[:8]}"
	)


def _normalize_state_dict_dtype_for_tman(
	state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
	converted: Dict[str, torch.Tensor] = {}
	bf16_count = 0
	for key, value in state_dict.items():
		if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
			converted[key] = value.to(dtype=torch.float16)
			bf16_count += 1
		else:
			converted[key] = value
	if bf16_count > 0:
		logging.info("Converted %d BF16 tensors in state_dict to FP16 for TMAN.", bf16_count)
	return converted


def _normalize_module_dtype_for_tman(module: torch.nn.Module) -> None:
	bf16_count = 0
	for sub_module in module.modules():
		for name, parameter in list(sub_module.named_parameters(recurse=False)):
			if parameter is not None and parameter.dtype == torch.bfloat16:
				sub_module._parameters[name] = torch.nn.Parameter(
					parameter.detach().to(dtype=torch.float16),
					requires_grad=parameter.requires_grad,
				)
				bf16_count += 1
		for name, buffer in list(sub_module.named_buffers(recurse=False)):
			if buffer is not None and buffer.dtype == torch.bfloat16:
				sub_module._buffers[name] = buffer.to(dtype=torch.float16)
				bf16_count += 1

	if bf16_count > 0:
		logging.info("Converted %d BF16 tensors in module to FP16 for TMAN.", bf16_count)


def _prepare_torch_qlinear_modules(module: torch.nn.Module) -> None:
	"""
	When TMAN replacement is disabled, TorchQuantLinear still needs post_init()
	to materialize helper buffers used by GPTQ dequantization.
	"""
	try:
		from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
	except ImportError as error:
		raise RuntimeError(f"Failed to import TorchQuantLinear: {error}") from error

	def _unwrap_method(instance: torch.nn.Module, method_name: str) -> bool:
		method = getattr(instance, method_name, None)
		if method is None or not callable(method):
			return False

		wrapped = getattr(method, "__wrapped__", None)
		if wrapped is None:
			return False

		try:
			if getattr(wrapped, "__self__", None) is instance:
				setattr(instance, method_name, wrapped)
			else:
				setattr(instance, method_name, types.MethodType(wrapped, instance))
			return True
		except Exception:
			return False

	prepared = 0
	unwrapped = 0
	for sub_module in module.modules():
		if isinstance(sub_module, TorchQuantLinear):
			sub_module.post_init()
			prepared += 1
			for method_name in ("dequantize_weight", "_forward"):
				if _unwrap_method(sub_module, method_name):
					unwrapped += 1

	if prepared > 0:
		logging.info(
			"Initialized %d TorchQuantLinear modules with post_init() for no_tman path.",
			prepared,
		)
	if unwrapped > 0:
		logging.info(
			"Unwrapped %d TorchQuantLinear dynamo-wrapped methods for export compatibility.",
			unwrapped,
		)


def _count_targets(graph_module: torch.fx.GraphModule, target_name: str) -> int:
	return sum(
		1
		for node in graph_module.graph.nodes
		if target_name in str(node.target)
	)


class SingleAttentionPrefill(torch.nn.Module):
	"""
	Keep exactly one Qwen3 attention module and prefill-style inputs.
	Linear projections are replaced by TMAN prefill linear, while attention
	matmul subgraph stays in the graph for backend lowering.
	"""

	def __init__(self, attention: Qwen3Attention, config: Qwen3Config, max_seq_len: int):
		super().__init__()
		self.attention = attention
		self.max_seq_len = max_seq_len
		self.head_dim = getattr(
			config,
			"head_dim",
			config.hidden_size // config.num_attention_heads,
		)

		freqs_cos, freqs_sin = precompute_freqs_cis(
			self.head_dim,
			self.max_seq_len,
			config.rope_theta,
			False,
			None,
		)
		self.register_buffer("freqs_cos", freqs_cos, persistent=False)
		self.register_buffer("freqs_sin", freqs_sin, persistent=False)

	def forward(self, hidden_states: torch.Tensor, atten_mask: torch.Tensor) -> torch.Tensor:
		seq_len = hidden_states.size(1)
		freqs_cos = self.freqs_cos[:seq_len]
		freqs_sin = self.freqs_sin[:seq_len]
		output, _, _ = self.attention(
			hidden_states,
			freqs_cos=freqs_cos,
			freqs_sin=freqs_sin,
			atten_mask=atten_mask,
			k_caches=None,
			v_caches=None,
		)
		return output


PTQ_MODE = "16a4w"


def _build_parser():
	parser = setup_common_args_and_variables()
	parser.add_argument(
		"-a",
		"--artifact",
		help="Path for generated artifact files.",
		default="./qwen3_attention_only_qnn",
		type=str,
	)
	parser.add_argument(
		"--model_dir",
		help="GPTQ Qwen3 model directory used by GPTQModel.from_quantized.",
		required=True,
		type=str,
	)
	parser.add_argument(
		"--mode",
		choices=["tman", "no_tman", "both"],
		default="both",
		type=str,
		help="Run TMAN profile, no-TMAN profile, or both for A/B.",
	)
	parser.add_argument(
		"--layer_idx",
		help="Attention layer index to extract from GPTQ state_dict.",
		default=0,
		type=int,
	)
	parser.add_argument(
		"--prefill_ar_len",
		help="Prefill AR length for this attention-only graph.",
		default=32,
		type=int,
	)
	parser.add_argument(
		"--max_seq_len",
		help="Maximum sequence length for rope/mask generation.",
		default=128,
		type=int,
	)
	parser.add_argument(
		"--disable_conv_hmx",
		action="store_true",
		help="Disable HTP short-depth-conv-on-HMX path.",
	)
	parser.add_argument(
		"--disable_fold_relu",
		action="store_true",
		help="Disable HTP fold-relu path.",
	)
	parser.add_argument(
		"-v",
		"--verbose",
		action="store_true",
		help="Print delegation info after lowering.",
	)
	parser.add_argument(
		"--seed",
		default=0,
		type=int,
		help="Random seed used for synthetic hidden_states input.",
	)
	return parser


def _validate_args(args):
	if args.max_seq_len < args.prefill_ar_len:
		raise RuntimeError("max_seq_len must be >= prefill_ar_len")


def _build_attention_only_model(
	args,
	use_tman_prefill_linear: bool,
) -> Tuple[torch.nn.Module, Qwen3Config]:
	config = Qwen3Config.from_pretrained(args.model_dir)
	attention = Qwen3Attention(config=config, layer_idx=args.layer_idx)

	qlinear_cls, full_state_dict = _build_qlinear_cls_and_state_dict(args.model_dir)
	convert_linear_to_qlinear(attention, qlinear_cls)

	attention_state_dict = _extract_attention_state_dict(full_state_dict, args.layer_idx)
	attention_state_dict = _normalize_state_dict_dtype_for_tman(attention_state_dict)
	try:
		incompatible = attention.load_state_dict(
			attention_state_dict,
			strict=False,
			assign=True,
		)
	except TypeError:
		incompatible = attention.load_state_dict(attention_state_dict, strict=False)

	logging.info(
		"Attention layer load done: missing=%d unexpected=%d",
		len(incompatible.missing_keys),
		len(incompatible.unexpected_keys),
	)
	if incompatible.unexpected_keys:
		logging.info("Unexpected keys (first 8): %s", incompatible.unexpected_keys[:8])

	_normalize_module_dtype_for_tman(attention)

	if use_tman_prefill_linear:
		convert_linear_to_tman_prefill_linear(attention)
		attention.prepare_tman()
	else:
		_prepare_torch_qlinear_modules(attention)

	model = SingleAttentionPrefill(
		attention=attention,
		config=config,
		max_seq_len=args.max_seq_len,
	)
	model = convert_linear_to_conv2d(model)
	return model.eval(), config


def _get_no_tman_discard_ops(quantizer) -> list:
	# no-TMAN keeps GPTQ unpack chains in graph; these ops frequently touch int8
	# tensors and should not be re-quantized by PT2E.
	candidates = (
		torch.ops.aten.matmul.default,
		torch.ops.aten.bmm.default,
		torch.ops.aten.index.Tensor,
		torch.ops.aten.index_put.default,
		torch.ops.aten.index_put_.default,
		torch.ops.aten.reshape.default,
		torch.ops.aten.view.default,
		torch.ops.aten.unsqueeze.default,
		torch.ops.aten.squeeze.default,
		torch.ops.aten.squeeze.dim,
		torch.ops.aten.expand.default,
		torch.ops.aten.expand_as.default,
		torch.ops.aten.permute.default,
		torch.ops.aten.transpose.int,
		torch.ops.aten.slice.Tensor,
		torch.ops.aten.select.int,
		torch.ops.aten.sub.Tensor,
		torch.ops.aten.mul.Tensor,
	)
	return [op for op in candidates if op in quantizer.quant_ops]


def _maybe_quantize_model(
	model: torch.nn.Module,
	sample_inputs: Tuple[torch.Tensor, torch.Tensor],
	use_tman_prefill_linear: bool,
) -> Tuple[torch.nn.Module, bool]:
	quantizer = make_quantizer(
		quant_dtype=QuantDtype.use_16a4w,
		per_channel_conv=True,
		per_channel_linear=True,
		act_observer=MinMaxObserver,
	)

	if use_tman_prefill_linear:
		quantizer.add_custom_quant_annotations((annotate_matmul_16a8w,))
		logging.info(
			"PTQ profile=tman_16a4w: matmul uses custom 16a8w annotation path."
		)
	else:
		discard_ops = _get_no_tman_discard_ops(quantizer)
		if discard_ops:
			quantizer.add_discard_ops(discard_ops)
		logging.info(
			"PTQ profile=no_tman_16a4w: discarded %d ops for GPTQ-int8-safe subgraph.",
			len(discard_ops),
		)

	with torch.no_grad():
		captured = torch.export.export(model, sample_inputs, strict=False).module()
		prepared = prepare_pt2e(captured, quantizer)
		prepared(*sample_inputs)
		# Keep folding off for both profiles so A/B is controlled and stable.
		quantized = convert_pt2e(prepared, fold_quantize=False)
	return quantized, False


def _build_passes_job(use_tman_prefill_linear: bool):
	if use_tman_prefill_linear:
		return None

	passes_job = get_capture_program_passes()
	if ConvertBmmToMatmul in passes_job:
		passes_job[ConvertBmmToMatmul][QCOM_PASS_ACTIVATE_KEY] = False
		logging.info("Pass profile=no_tman: disabled ConvertBmmToMatmul.")
	return passes_job


def _compile_single_mode(args, mode_tag: str) -> None:
	use_tman_prefill_linear = mode_tag == "tman"
	torch.manual_seed(args.seed)

	logging.info(
		"Running profile=%s with PTQ=%s",
		mode_tag,
		PTQ_MODE,
	)

	model, config = _build_attention_only_model(args, use_tman_prefill_linear)

	hidden_states = torch.randn(
		1,
		args.prefill_ar_len,
		config.hidden_size,
		dtype=torch.float32,
	)
	atten_mask = _build_prefill_attention_mask(
		ar_len=args.prefill_ar_len,
		max_seq_len=args.max_seq_len,
	)
	sample_inputs = (hidden_states, atten_mask)

	lowering_model, use_fp16 = _maybe_quantize_model(
		model,
		sample_inputs,
		use_tman_prefill_linear=use_tman_prefill_linear,
	)
	if hasattr(lowering_model, "graph"):
		matmul_count = _count_targets(lowering_model, "aten.matmul.default")
		tman_prefill_linear_count = _count_targets(
			lowering_model,
			"tman.prefill_linear.default",
		)
		logging.info(
			"Lowering graph nodes (%s): aten.matmul.default=%d, tman.prefill_linear.default=%d",
			mode_tag,
			matmul_count,
			tman_prefill_linear_count,
		)

	backend_options = generate_htp_compiler_spec(
		use_fp16=use_fp16,
		use_multi_contexts=False,
	)
	backend_options.htp_options.use_conv_hmx = not args.disable_conv_hmx
	backend_options.htp_options.use_fold_relu = not args.disable_fold_relu

	compiler_specs = generate_qnn_executorch_compiler_spec(
		soc_model=get_soc_to_chipset_map()[args.model],
		backend_options=backend_options,
		shared_buffer=args.shared_buffer,
		online_prepare=args.online_prepare,
		dump_intermediate_outputs=args.dump_intermediate_outputs,
	)

	skip_node_id_set = _get_skip_node_id_set(args.skip_delegate_node_ids)
	skip_node_op_set = _get_skip_node_op_set(args.skip_delegate_node_ops)
	passes_job = _build_passes_job(use_tman_prefill_linear)

	logging.info("Compiling attention-only prefill repro graph (%s)...", mode_tag)
	edge_prog_mgr = to_edge_transform_and_lower_to_qnn(
		lowering_model,
		sample_inputs,
		compiler_specs,
		passes_job=passes_job,
		skip_node_id_set=skip_node_id_set,
		skip_node_op_set=skip_node_op_set,
	)

	if args.verbose:
		print_delegation_info(edge_prog_mgr.exported_program().graph_module)

	executorch_config = ExecutorchBackendConfig(
		memory_planning_pass=MemoryPlanningPass(
			alloc_graph_input=False,
			alloc_graph_output=False,
		),
		extract_delegate_segments=True,
	)
	exec_prog_mgr = edge_prog_mgr.to_executorch(config=executorch_config)

	pte_path = os.path.join(
		args.artifact,
		f"qwen3_attention_layer{args.layer_idx}_{mode_tag}_{PTQ_MODE}_qnn.pte",
	)
	with open(pte_path, "wb") as file:
		exec_prog_mgr.write_to_file(file)

	logging.info("Done (%s). Wrote %s", mode_tag, pte_path)


def compile_attention_only(args):
	_validate_args(args)
	os.makedirs(args.artifact, exist_ok=True)

	modes = ["tman", "no_tman"] if args.mode == "both" else [args.mode]
	failures = {}
	for mode_tag in modes:
		try:
			_compile_single_mode(args, mode_tag)
		except Exception as error:
			failures[mode_tag] = str(error)
			logging.exception("Profile %s failed.", mode_tag)
			if args.mode != "both":
				raise

	if failures:
		failure_summary = ", ".join(
			f"{mode}: {message}" for mode, message in failures.items()
		)
		raise RuntimeError(f"A/B run completed with failures -> {failure_summary}")


def main():
	parser = _build_parser()
	args = parser.parse_args()
	if not args.compile_only:
		logging.warning("This repro script is compile-only. Continuing with compile flow.")
	compile_attention_only(args)


if __name__ == "__main__":
	main()
