# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.utils import load_custom_op_library

load_custom_op_library()

WIDTH = 512
BLOCK_SIZE = 4
SENTINEL = 83
FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _pack_reference(row, mode, group_size):
    """Encode from numerical format definitions, independently of the kernel."""
    groups = row.float().reshape(-1, group_size)
    amax = groups.abs().amax(-1, keepdim=True)
    if mode == "mxfp8_bf16":
        scale = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-4) / 448)))
        packed = (groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn).flatten().view(torch.uint8)
    else:
        scale = torch.where(amax == 0, 0.0, torch.exp2(torch.floor(torch.log2(amax.clamp_min(2.0**-125))) - 2))
        normalized = groups / torch.where(scale == 0, 1.0, scale)
        distances = (normalized.abs().unsqueeze(-1) - FP4_LEVELS).abs()
        closest = distances == distances.amin(-1, keepdim=True)
        # An exact midpoint chooses the even code, including signed zero.
        priorities = torch.tensor([0, 9, 2, 11, 4, 13, 6, 15])
        code = torch.where(closest, priorities, 100).argmin(-1).to(torch.uint8)
        code |= torch.signbit(normalized).to(torch.uint8) << 3
        # FP4 preserves the input sign even when an all-zero group has scale 0.
        code = code.flatten()
        packed = code[::2] | (code[1::2] << 4)
    return torch.cat((packed, scale.flatten().bfloat16().view(torch.uint8)))


def _input_rows(group_size):
    generator = torch.Generator().manual_seed(671)
    rows = torch.randn(8, WIDTH, generator=generator).bfloat16()
    midpoints = torch.tensor([-6, 6, -5, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, -0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 5])
    rows[0] = midpoints.repeat(WIDTH // midpoints.numel()).bfloat16()
    rows[1, :group_size] = 0
    rows[2] *= 2.0**-20
    rows[3] *= 1024
    return rows


def _expected_cache(initial, rows, slots, mode, group_size, paged, physical_step):
    expected = initial.clone()
    for row, slot in zip(rows, slots.tolist()):
        if not 0 <= slot < 8:
            continue
        payload = _pack_reference(row, mode, group_size)
        if paged:
            block, offset = divmod(slot, BLOCK_SIZE)
            block_bytes = expected[block * physical_step].flatten()
            start = offset * payload.numel()
            block_bytes[start : start + payload.numel()] = payload
        else:
            expected[slot, : payload.numel()] = payload
            aligned_width = (payload.numel() + 31) // 32 * 32
            expected[slot, payload.numel() : aligned_width] = 0
    return expected


@pytest.mark.parametrize("mode,group_size", [("mxfp8_bf16", 32), ("mxfp4_bf16", 16), ("mxfp4_bf16", 32)])
@pytest.mark.parametrize("layout", ["flat", "paged", "strided_paged"])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_kv_compress_epilog_v2_bytes_and_untouched_cache(mode, group_size, layout, slot_dtype):
    rows = _input_rows(group_size)
    slots = torch.tensor([0, 3, 4, 7, -1, -2, 8, 2**30], dtype=slot_dtype)
    if slot_dtype == torch.int64:
        slots[-1] = 2**33
    payload_width = (WIDTH if mode == "mxfp8_bf16" else WIDTH // 2) + 2 * WIDTH // group_size
    cache_width = (payload_width + 31) // 32 * 32 + 32
    paged = layout != "flat"
    step = 2 if layout == "strided_paged" else 1
    shape = (2 * step, BLOCK_SIZE, 1, cache_width) if paged else (8, cache_width)
    initial = torch.full(shape, SENTINEL, dtype=torch.uint8)
    base = initial.npu()
    cache = base[::step] if paged else base
    if mode == "mxfp8_bf16":
        cache = cache.view(torch.float8_e4m3fn)
    torch.ops._C_ascend.kv_compress_epilog_v2(
        cache, rows.npu(), slots.npu(), quant_group_size=group_size, quant_mode=mode
    )
    expected = _expected_cache(initial, rows, slots, mode, group_size, paged, step)
    torch.testing.assert_close(base.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode,group_size", [("mxfp8_bf16", 32), ("mxfp4_bf16", 16)])
@torch.inference_mode()
def test_kv_compress_epilog_v2_graph_replays_new_values(mode, group_size):
    rows = _input_rows(group_size)
    slots = torch.tensor([0, 3, 4, 7, -1, -2, 8, 2**33], dtype=torch.int64)
    payload_width = (WIDTH if mode == "mxfp8_bf16" else WIDTH // 2) + 2 * WIDTH // group_size
    initial = torch.full((4, BLOCK_SIZE, 1, payload_width + 32), SENTINEL, dtype=torch.uint8)
    base, values, indices = initial.npu(), rows.npu(), slots.npu()
    cache = base[::2]
    if mode == "mxfp8_bf16":
        cache = cache.view(torch.float8_e4m3fn)

    def run():
        torch.ops._C_ascend.kv_compress_epilog_v2(cache, values, indices, quant_group_size=group_size, quant_mode=mode)

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for factor in (1.0, -0.5):
        current = (rows.float() * factor).bfloat16()
        base.fill_(SENTINEL)
        values.copy_(current.npu())
        graph.replay()
        expected = _expected_cache(initial, current, slots, mode, group_size, True, 2)
        torch.testing.assert_close(base.cpu(), expected, rtol=0, atol=0)
