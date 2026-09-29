# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.utils import load_custom_op_library

load_custom_op_library()

HC_MULT = 4
HIDDEN_SIZE = 5120
NORM_EPS = 1e-20
HC_EPS = 1e-6
SINKHORN_ITERS = 20


def _hf32(x):
    # HcPre uses HF32 truncation for the coefficient projection.
    return (x.contiguous().view(torch.int32) & ~((1 << 13) - 1)).view(torch.float32)


def _reference(x, weight, scale, bias, previous_pre):
    flat = x.float().flatten(-2)
    mixes = F.linear(_hf32(flat), _hf32(weight))
    mixes *= torch.rsqrt(flat.square().mean(-1, keepdim=True) + NORM_EPS)
    next_pre = torch.sigmoid(mixes[..., :4] * scale[0] + bias[:4]) + HC_EPS
    post = 2 * torch.sigmoid(mixes[..., 4:8] * scale[1] + bias[4:8])
    comb = (mixes[..., 8:] * scale[2] + bias[8:]).unflatten(-1, (4, 4))
    comb = comb.softmax(-1) + HC_EPS
    comb /= comb.sum(-2, keepdim=True) + HC_EPS
    for _ in range(SINKHORN_ITERS - 1):
        comb /= comb.sum(-1, keepdim=True) + HC_EPS
        comb /= comb.sum(-2, keepdim=True) + HC_EPS
    used_pre = next_pre if previous_pre is None else previous_pre
    y = (x.float() * used_pre.unsqueeze(-1)).sum(-2).to(x.dtype)
    return y, post, comb, next_pre


def _inputs(tokens, seed=103, magnitude=1.0):
    generator = torch.Generator().manual_seed(seed)
    x = (torch.randn(tokens, HC_MULT, HIDDEN_SIZE, generator=generator) * magnitude).bfloat16()
    weight = torch.randn(24, HC_MULT * HIDDEN_SIZE, generator=generator) / (HC_MULT * HIDDEN_SIZE) ** 0.5
    scale = torch.tensor([0.3, 0.5, 0.7])
    bias = torch.linspace(-0.8, 0.8, 24)
    previous_pre = torch.rand(tokens, HC_MULT, generator=generator)
    return x, weight, scale, bias, previous_pre


def _run(x, weight, scale, bias, previous_pre):
    return torch.ops._C_ascend.npu_hc_pre_v3(
        x,
        weight,
        scale,
        bias,
        previous_pre,
        hc_mult=HC_MULT,
        hc_sinkhorn_iters=SINKHORN_ITERS,
        norm_eps=NORM_EPS,
        hc_eps=HC_EPS,
    )


def _check(actual, expected):
    for index, (result, reference) in enumerate(zip(actual, expected)):
        assert result.dtype == reference.dtype
        result = result.cpu()
        if index == 0:
            # HF32 coefficient rounding can move a BF16 output across one
            # rounding boundary. Bound that explicitly instead of applying a
            # large relative tolerance to every output and every magnitude.
            up = torch.nextafter(reference, torch.full_like(reference, float("inf")))
            down = torch.nextafter(reference, torch.full_like(reference, -float("inf")))
            ulp = torch.maximum((up.float() - reference.float()).abs(), (down.float() - reference.float()).abs())
            error = (result.float() - reference.float()).abs()
            assert bool((error <= ulp + 5e-3).all()), f"Maximum excess: {(error - ulp - 5e-3).max().item()}"
        else:
            torch.testing.assert_close(result, reference, rtol=8e-4, atol=8e-4)


def _post(x, residual, post, comb):
    return torch.ops._C_ascend.npu_hc_post(
        x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0)
    ).squeeze(0)


@pytest.mark.parametrize("tokens", [1, 7, 96, 257])
@pytest.mark.parametrize("external_pre", [False, True])
@torch.inference_mode()
def test_dsv41_hc_pre_d5120(tokens, external_pre):
    cpu = _inputs(tokens)
    x, weight, scale, bias, previous_pre = cpu
    previous_pre = previous_pre if external_pre else None
    expected = _reference(x, weight, scale, bias, previous_pre)
    actual = _run(x.npu(), weight.npu(), scale.npu(), bias.npu(), None if previous_pre is None else previous_pre.npu())
    _check(actual, expected)
    if external_pre:
        torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=0, atol=0)


@pytest.mark.parametrize("magnitude", [0.0, 1e-12, 1.0])
@torch.inference_mode()
def test_dsv41_hc_pre_one_hot_and_norm_epsilon(magnitude):
    x, weight, scale, bias, previous_pre = _inputs(3, magnitude=magnitude)
    previous_pre.zero_()
    previous_pre[:, 2] = 1
    actual = _run(x.npu(), weight.npu(), scale.npu(), bias.npu(), previous_pre.npu())
    # A one-hot incoming pre must select a residual stream exactly, regardless
    # of the newly computed gate. This fails if pre_next is consumed too early.
    torch.testing.assert_close(actual[0].cpu(), x[:, 2], rtol=0, atol=0)
    _check(actual, _reference(x, weight, scale, bias, previous_pre))


@pytest.mark.parametrize("graph", [False, True])
@torch.inference_mode()
def test_dsv41_two_sublayers_carry_pre(graph):
    x, weight1, scale, bias, previous_pre = _inputs(7)
    _, weight2, _, _, _ = _inputs(7, seed=211)
    npu_inputs = [value.npu() for value in (x, weight1, weight2, scale, bias, previous_pre)]

    def forward():
        residual, w1, w2, gain, base, incoming = npu_inputs
        y1, post1, comb1, next1 = _run(residual, w1, gain, base, incoming)
        attention = (y1.float() * 0.125).tanh().to(y1.dtype)
        residual1 = _post(attention, residual, post1, comb1)
        y2, post2, comb2, next2 = _run(residual1, w2, gain, base, next1)
        ffn = F.silu(y2.float()).to(y2.dtype)
        residual2 = _post(ffn, residual1, post2, comb2)
        collapsed = (residual2.float() * next2.unsqueeze(-1)).sum(-2).to(residual2.dtype)
        return residual1, residual2, next1, next2, collapsed

    if graph:
        for _ in range(2):
            forward()
        torch.npu.synchronize()
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            actual = forward()

    for factor in (1.0, -0.75):
        current_x = (x.float() * factor).bfloat16()
        npu_inputs[0].copy_(current_x.npu())
        if graph:
            captured.replay()
        else:
            actual = forward()
        y1, post1, comb1, next1 = _reference(current_x, weight1, scale, bias, previous_pre)
        attention = (y1.float() * 0.125).tanh().to(x.dtype)
        residual1 = (
            post1.unsqueeze(-1) * attention.float().unsqueeze(-2)
            + torch.matmul(comb1.transpose(-1, -2), current_x.float())
        ).to(x.dtype)
        y2, post2, comb2, next2 = _reference(residual1, weight2, scale, bias, next1)
        ffn = F.silu(y2.float()).to(x.dtype)
        residual2 = (
            post2.unsqueeze(-1) * ffn.float().unsqueeze(-2) + torch.matmul(comb2.transpose(-1, -2), residual1.float())
        ).to(x.dtype)
        collapsed = (residual2.float() * next2.unsqueeze(-1)).sum(-2).to(x.dtype)
        for got, expected in zip(actual, (residual1, residual2, next1, next2, collapsed)):
            torch.testing.assert_close(got.cpu(), expected, rtol=1e-2, atol=1e-2)
