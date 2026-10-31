# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident prefill plans must preserve each host page's physical row mapping."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.v1.hisparse.runtime import HiSparseRuntime, build_hisparse_prefill_staging_plan

from vllm_ascend.distributed.kv_transfer.hisparse.backend import AscendHiSparseBackend
from vllm_ascend.distributed.kv_transfer.hisparse.layout import (
    cache_components,
    combine_components,
    split_host_cache,
    split_paged_cache,
)
from vllm_ascend.ops.hisparse import HiSparseNPUOps, allocate_mapped_host


@pytest.mark.parametrize("resident_block_size", [128, 64, 32])
@pytest.mark.parametrize("host_pages", [1, 2])
def test_resident_prefill_plan_shared_layers(resident_block_size, host_pages):
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith(("Ascend950", "Ascend910_93")):
        pytest.skip("HiSparse mapped-host staging requires Ascend 950 or 910_93")
    torch.npu.set_device(0)
    device = "npu:0"
    block_size, width, layers, blocks = 128, 576, 4, 16
    page_elements = block_size * width
    block_stride = 6 * page_elements
    raw = torch.zeros(blocks * block_stride * 2, dtype=torch.int8, device=device)
    # Match the shared HMA slab: create and retain every layer view before use.
    views = [
        split_paged_cache(
            raw,
            dtype=torch.bfloat16,
            row_width=width,
            byte_offset=layer * resident_block_size * width * 2,
            block_stride=block_stride * 2,
            num_blocks=blocks,
            block_size=resident_block_size,
        )
        for layer in range(layers)
    ]
    host = allocate_mapped_host(((host_pages + 1) * block_size, width), torch.bfloat16)
    host_cache = split_host_cache(host)
    for component in cache_components(host_cache):
        component.zero_()
        component[block_size:].fill_(-23)
    ops = HiSparseNPUOps(host._hisparse_registration)
    try:
        subpages = block_size // resident_block_size
        resident_ids = torch.arange(2, 2 + host_pages * subpages, dtype=torch.int32)
        if host_pages == 2:
            # The last subpage is a real host miss; other rows must stay on NPU.
            resident_ids[-1] = 0
        resident_table = resident_ids[None, :].to(device)
        host_table = torch.arange(1, host_pages + 1, dtype=torch.int32, device=device)[None, :]
        plan = build_hisparse_prefill_staging_plan(
            host_table, torch.tensor([host_pages * block_size - 7], device=device), block_size, host_pages
        )
        state = torch.tensor([0], dtype=torch.int32, device=device)
        logical_rows = torch.arange(host_pages * block_size)
        physical = resident_ids.long()[logical_rows // resident_block_size] * resident_block_size
        physical += logical_rows % resident_block_size
        hits = resident_ids[logical_rows // resident_block_size] > 0
        physical[~hits] = -1
        expected_gpu = torch.cat((torch.full((block_size,), -1), physical)).int()
        expected_mask = (expected_gpu < 0).int()
        original_rows = None
        for layer, view in enumerate(views):
            columns = torch.arange(width)[None, :] % 16
            payload = (layer * 16 + logical_rows[:, None].float() / 8 + columns / 16 + 1).bfloat16()
            on_device = payload.to(device)
            device_rows = physical[hits].to(device)
            for component, values in zip(cache_components(view.cache), on_device.split((512, 64), dim=-1), strict=True):
                component[device_rows // resident_block_size, device_rows % resident_block_size] = values[
                    hits.to(device)
                ]
            plan.ensure_gpu_sources(resident_table, state, resident_block_size)
            torch.testing.assert_close(plan.gpu_row_ids.cpu().flatten(), expected_gpu, rtol=0, atol=0)
            torch.testing.assert_close(plan.miss_mask.cpu().flatten(), expected_mask, rtol=0, atol=0)
            if original_rows is None:
                original_rows = plan.gpu_row_ids
            else:
                # Layers in an index group reuse the same resolved page plan.
                assert plan.gpu_row_ids is original_rows
            actual = HiSparseRuntime.gather_prefill_cache(
                SimpleNamespace(cache_ops=ops, backend=AscendHiSparseBackend()),
                combine_components(tuple(c.view(-1, block_size, c.shape[-1]) for c in cache_components(host_cache))),
                plan,
                view.cache,
            )
            for actual_component, host_component, expected_payload in zip(
                cache_components(actual), cache_components(host_cache), payload.split((512, 64), dim=-1), strict=True
            ):
                expected = host_component.clone()
                expected[block_size:][hits] = expected_payload[hits]
                torch.testing.assert_close(actual_component.cpu().flatten(0, 1), expected, rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        host._hisparse_registration.close()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2])
def test_backend_resident_payload_graph_replay(dtype):
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith(("Ascend950", "Ascend910_93")):
        pytest.skip("HiSparse resident gather requires Ascend 950 or 910_93")
    torch.npu.set_device(0)
    width = 656 if dtype != torch.bfloat16 else 576
    stride = 3 * width
    raw = torch.arange(7 * width * dtype.itemsize, dtype=torch.int64).to(torch.uint8).to("npu:0")
    source = raw.view(dtype).as_strided((2, 2, width), (stride, width, 1), storage_offset=width)
    rows = torch.tensor([3, 0, 2, -1], dtype=torch.int32, device="npu:0")
    destination = torch.empty((4, width), dtype=dtype, device="npu:0")
    backend = AscendHiSparseBackend()
    backend.gather_resident_rows(source, destination, rows)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        backend.gather_resident_rows(source, destination, rows)
    for values in ([3, 0, 2, -1], [0, 3, 1, 2]):
        rows.copy_(torch.tensor(values, dtype=torch.int32, device="npu:0"))
        raw.bitwise_xor_(1)
        graph.replay()
        torch.npu.synchronize()
        safe = torch.tensor(values).clamp_min(0)
        expected = source.view(torch.uint8).cpu()[safe // 2, safe % 2]
        assert torch.equal(destination.view(torch.uint8).cpu(), expected)
