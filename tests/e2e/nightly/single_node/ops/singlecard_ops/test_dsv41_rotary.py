# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV4.1 RoPE adapter with native A5 kernels and an independent CPU oracle."""

import pytest
import torch

from vllm_ascend.ops.dsv41_a5.rotary import apply_partial_rotary_inplace
from vllm_ascend.utils import load_custom_op_library


@pytest.mark.parametrize("heads", [None, 64])
@pytest.mark.parametrize("graph", [False, True])
@torch.inference_mode()
def test_dsv41_partial_rotary_forward_and_inverse(heads, graph):
    load_custom_op_library()
    generator = torch.Generator().manual_seed(918)
    tokens, width, rotary_width = 7, 512, 64
    start = width - rotary_width
    shape = (tokens, width) if heads is None else (tokens, heads, width)
    angles = torch.randn(tokens, 1, 1, rotary_width // 2, generator=generator)
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    cos_npu, sin_npu = cos.npu(), sin.npu()
    work = torch.empty(shape, dtype=torch.bfloat16, device="npu")

    for inverse in (False, True):

        def rotate(inverse=inverse):
            return apply_partial_rotary_inplace(work, cos_npu, sin_npu, start=start, end=width, inverse=inverse)

        captured = None
        if graph:
            work.zero_()
            rotate()
            torch.npu.synchronize()
            captured = torch.npu.NPUGraph()
            with torch.npu.graph(captured):
                rotate()

        for replay in range(2):
            source = torch.randn(shape, generator=generator).to(torch.bfloat16) * (replay + 1)
            work.copy_(source.npu())
            if captured is None:
                assert rotate() is work
            else:
                captured.replay()
            actual = work.cpu()
            # Compute complex multiplication directly; do not call an NPU or
            # model RoPE helper in the reference. Non-RoPE bytes must not move.
            expanded = source.reshape(tokens, 1, -1, width).float()
            real, imag = expanded[..., start::2], expanded[..., start + 1 :: 2]
            c, s = cos[..., ::2], sin[..., ::2] * (-1 if inverse else 1)
            expected = expanded.clone()
            expected[..., start::2] = real * c - imag * s
            expected[..., start + 1 :: 2] = imag * c + real * s
            expected = expected.reshape(shape).to(source.dtype)
            torch.testing.assert_close(actual[..., :start], source[..., :start], rtol=0, atol=0)
            torch.testing.assert_close(actual[..., start:], expected[..., start:], rtol=8e-3, atol=8e-3)
