# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check odd TP scale groups against independently decoded MX operands."""

import pytest
import torch
import torch_npu

from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DSDynamicLinearMethod


@pytest.mark.parametrize("width", [288, 320])
@pytest.mark.parametrize("graph_mode", [False, True])
@torch.inference_mode()
def test_deepseek_mxfp8_tp_scale_pairs(width, graph_mode):
    torch.manual_seed(413)
    scheme = object.__new__(AscendW8A8MXFP8DSDynamicLinearMethod)
    scheme.block_size = scheme.group_size = 32
    scheme.dynamic_mx_quant_scale_alg = 0
    weight = torch.randn(64, width).to(torch.float8_e4m3fn)
    scales = torch.exp2(torch.arange(width // 32).float() % 5 - 2).repeat(2, 1)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight.npu(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales.npu(), requires_grad=False)
    layer.prefix = "model.layers.0.mlp.shared_experts.down_proj"
    scheme.process_weights_after_loading(layer)
    x = torch.randn(3, width, dtype=torch.bfloat16, device="npu")

    def reference():
        quantized, exponent = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn, scale_alg=0)
        # Decode E8M0 bytes independently; discard the unused packed tail.
        factors = torch.exp2(exponent.cpu().view(torch.uint8).reshape(3, -1).float() - 127)
        decoded_x = quantized.cpu().float() * factors.repeat_interleave(32, dim=-1)[:, :width]
        decoded_weight = weight.float() * scales.repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
        return decoded_x @ decoded_weight.T

    actual = scheme.apply(layer, x)
    torch.testing.assert_close(actual.cpu().float(), reference(), rtol=0.02, atol=0.04)
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
            replayed = scheme.apply(layer, x)
        x.mul_(1.25)
        graph.replay()
        torch.testing.assert_close(replayed.cpu().float(), reference(), rtol=0.02, atol=0.04)
