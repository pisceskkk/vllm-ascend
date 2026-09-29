# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU references for the DSV4.1 wire formats and numerical contracts.

These helpers do not import the kernels or use their lookup/packing routines.
"""

import torch

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def pack_fp4(values):
    # argmin selects the lower magnitude on exact halfway cases.
    levels = torch.tensor(FP4_VALUES, dtype=torch.float64)
    values = values.double()
    codes = (values.abs().unsqueeze(-1) - levels).abs().argmin(-1).to(torch.uint8)
    codes |= (values < 0).to(torch.uint8) << 3
    return codes[..., ::2] | (codes[..., 1::2] << 4)


def unpack_fp4(packed):
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    values = torch.tensor(FP4_VALUES)[codes & 7]
    return torch.where((codes & 8) != 0, -values, values)


def quantize_indexer(x):
    x = x.float().reshape(*x.shape[:-1], 4, 32)
    scale = (x.abs().amax(-1).double() / 6).clamp_min(2.0**-126)
    exponent = scale.log2().ceil().clamp(-126, 127)
    data = pack_fp4((x.double() / torch.pow(2.0, exponent).unsqueeze(-1)).flatten(-2))
    return data, (exponent + 127).to(torch.uint8)


def dequantize_indexer(data, scale):
    values = unpack_fp4(data).unflatten(-1, (4, 32))
    return (values * torch.pow(2.0, scale.float() - 127).unsqueeze(-1)).flatten(-2)


def pack_attention_cache(x, fp4):
    """Construct the documented data-then-BF16-scales row layout on CPU."""
    group = 16 if fp4 else 32
    values = x.float().unflatten(-1, (512 // group, group))
    maximum = 6.0 if fp4 else 448.0
    scale = (values.abs().amax(-1) / maximum).clamp_min(2.0**-126)
    scale = torch.pow(2.0, scale.log2().ceil()).to(torch.bfloat16)
    normalized = (values / scale.float().unsqueeze(-1)).flatten(-2)
    data = pack_fp4(normalized) if fp4 else normalized.to(torch.float8_e4m3fn).view(torch.uint8)
    packed = torch.cat((data, scale.view(torch.uint8)), dim=-1)
    decoded = unpack_fp4(data) if fp4 else data.view(torch.float8_e4m3fn).float()
    decoded = (decoded.unflatten(-1, (512 // group, group)) * scale.float().unsqueeze(-1)).flatten(-2)
    return packed, decoded.to(torch.bfloat16).float()


def decode_attention_cache(packed, fp4):
    """Decode bytes by the public row ABI, without calling the consumer."""
    width, group = (256, 16) if fp4 else (512, 32)
    data = packed[..., :width].contiguous()
    scales = packed[..., width:].contiguous().view(torch.bfloat16).float()
    decoded = unpack_fp4(data) if fp4 else data.view(torch.float8_e4m3fn).float()
    return (decoded.unflatten(-1, (512 // group, group)) * scales.unsqueeze(-1)).flatten(-2)


def indexer_scores(q, k, weights):
    """MX matmul, ReLU, and ordered BF16 fused head accumulation."""
    dot = torch.einsum("thd,kd->thk", q.float(), k.float()).to(torch.bfloat16).float().relu()
    weights = weights.to(torch.bfloat16).float()
    scores = torch.zeros((q.shape[0], k.shape[0]), dtype=torch.float32)
    for head in range(q.shape[1]):
        scores = (scores + dot[:, head] * weights[:, head, None]).to(torch.bfloat16).float()
    return scores


def assert_topk(indices, values, scores, eligible, topk=512):
    """Require every strictly better item; permit only exact cutoff ties."""
    indices = indices.cpu().flatten().long()
    values = values.cpu().flatten().float()
    eligible = eligible.bool()
    count = min(topk, int(eligible.sum()))
    valid = indices >= 0
    assert int(valid.sum()) == count
    chosen = indices[valid]
    assert chosen.unique().numel() == count
    assert bool(eligible[chosen].all())
    if not count:
        return
    torch.testing.assert_close(values[valid], scores[chosen], rtol=0, atol=0)
    cutoff = scores[eligible].topk(count).values[-1]
    assert bool((scores[chosen] >= cutoff).all())
    mandatory = torch.nonzero(eligible & (scores > cutoff)).flatten()
    assert bool(torch.isin(mandatory, chosen).all())
