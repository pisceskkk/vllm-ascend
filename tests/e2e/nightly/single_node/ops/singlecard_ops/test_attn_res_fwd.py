# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


@pytest.mark.parametrize(
    "num_tokens,num_blocks,hidden_size",
    [
        (1, 1, 128),
        (7, 3, 256),
        (32, 8, 4096),
        (3, 64, 4096),
        (2, 1, 7168),
    ],
)
@pytest.mark.parametrize("epsilon", [1e-5, 1e-6])
@pytest.mark.parametrize("layout", ["contiguous", "block_slice", "strided", "transpose"])
def test_attn_res_fwd(num_tokens, num_blocks, hidden_size, epsilon, layout):
    generator = torch.Generator().manual_seed(42)
    prefix = torch.randn(num_tokens, hidden_size, generator=generator, dtype=torch.bfloat16)
    blocks = torch.randn(num_tokens, num_blocks, hidden_size, generator=generator, dtype=torch.bfloat16)
    projection = (torch.randn(1, hidden_size, generator=generator) / hidden_size**0.5).bfloat16()
    gamma = torch.randn(hidden_size, generator=generator, dtype=torch.bfloat16)

    values = torch.cat((blocks, prefix.unsqueeze(1)), dim=1).float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)
    logits = (normalized * gamma.float() * projection.float()).sum(-1)
    expected = (logits.softmax(-1).unsqueeze(-1) * values).sum(1).bfloat16()

    inputs = [tensor.npu() for tensor in (prefix, blocks, projection, gamma)]
    if layout == "block_slice":
        storage = inputs[1].new_empty(num_tokens, num_blocks + 2, hidden_size)
        inputs[1] = storage[:, 1 : num_blocks + 1, :]
        inputs[1].copy_(blocks)
    elif layout == "strided":
        for index, tensor in enumerate(inputs):
            storage = tensor.new_empty(*tensor.shape[:-1], 2 * hidden_size)
            inputs[index] = storage[..., 1::2]
            inputs[index].copy_(tensor)
            assert not inputs[index].is_contiguous()
    elif layout == "transpose":
        inputs[0] = inputs[0].t().contiguous().t()

    actual = torch.ops._C_ascend.attn_res_fwd(*inputs, epsilon)

    assert actual.shape == prefix.shape
    assert actual.dtype == prefix.dtype
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("num_tokens,num_blocks", [(1, 0), (1, 1), (4, 4), (8, 8), (512, 8)])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_attn_res_fwd_with_add_preserves_prefix_and_bf16_rounding(num_tokens, num_blocks, strided):
    hidden_size = 7168
    torch.manual_seed(17)
    storage = torch.randn(num_tokens + 2, hidden_size * 2, device="npu", dtype=torch.bfloat16)
    add_storage = torch.randn_like(storage)
    prefix = storage[1:-1, ::2] if strided else storage[1:-1, :hidden_size]
    addend = add_storage[1:-1, ::2] if strided else add_storage[1:-1, :hidden_size]
    saved_prefix = prefix.clone()
    saved_addend = addend.clone()
    blocks = torch.randn(num_tokens, 8, hidden_size, device="npu", dtype=torch.bfloat16)
    blocks[:, num_blocks:] = float("nan")
    valid_blocks = blocks[:, :num_blocks]
    projection = torch.randn(1, hidden_size, device="npu", dtype=torch.bfloat16) * 0.02
    gamma = torch.randn(hidden_size, device="npu", dtype=torch.bfloat16)

    actual, new_prefix = torch.ops._C_ascend.attn_res_fwd_with_add(
        prefix, addend, valid_blocks, projection, gamma, 1e-5
    )
    rounded_prefix = prefix + addend
    expected = (
        torch.ops._C_ascend.attn_res_fwd(rounded_prefix, valid_blocks, projection, gamma, 1e-5)
        if num_blocks
        else rounded_prefix
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(new_prefix, rounded_prefix, rtol=0, atol=0)
    torch.testing.assert_close(prefix, saved_prefix, rtol=0, atol=0)
    torch.testing.assert_close(addend, saved_addend, rtol=0, atol=0)
