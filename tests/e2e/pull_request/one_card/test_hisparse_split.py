# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split planes retain slab isolation and transfer both payload components."""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config import SchedulerConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker
from vllm.v1.hisparse.runtime import (
    HiSparseCacheHandle,
    HiSparseRuntime,
    ResolvedHiSparseConfig,
    build_hisparse_prefill_staging_plan,
)
from vllm.v1.hisparse.types import SparseKVPageTransfer, SparseKVRowMirror
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.distributed.kv_transfer.hisparse.layout import cache_components, combine_components, split_paged_cache
from vllm_ascend.ops.hisparse import allocate_mapped_host


@pytest.fixture
def split_worker():
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith(("Ascend950", "Ascend910_93")):
        pytest.skip("HiSparse split integration requires Ascend 950 or 910_93")
    torch.npu.set_device(0)
    blocks, block_size, width, layers = 8, 128, 576, 2
    stride = layers * block_size * width * 2
    raw = torch.zeros(blocks * stride, dtype=torch.uint8, device="npu:0")
    states = torch.tensor([0], dtype=torch.int32, device="npu:0")
    handles, pools = [], []
    for layer in range(layers):
        runtime = HiSparseRuntime(ResolvedHiSparseConfig(128, 128, 128), 1, width, torch.bfloat16, "npu:0")
        runtime.request_state_indices = states
        runtime.resident_source_index = 0
        host = allocate_mapped_host((blocks, block_size, width), torch.bfloat16)
        host.fill_(-7)
        pools.append(host)
        runtime.bind_source_cache(host, registered_host_pool=host)
        handle = HiSparseCacheHandle(runtime)
        handle.bind_cache(
            raw,
            byte_offset=layer * block_size * width * 2,
            block_stride=stride,
            num_blocks=blocks,
            block_size=block_size,
            block_table=torch.zeros((1, blocks), dtype=torch.int32, device="npu:0"),
            slot_mapping=torch.zeros(block_size, dtype=torch.int64, device="npu:0"),
        )
        handles.append(handle)
    config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=128,
            max_num_seqs=1,
            max_model_len=1024,
            is_encoder_decoder=False,
        )
    )
    worker = HiSparseConnectorWorker(config, KVCacheConfig(num_blocks=blocks, kv_cache_tensors=[], kv_cache_groups=[]))
    worker.initialize(handles, ["layer0", "layer1"], raw, 1, blocks, torch.device("npu:0"), pools)
    try:
        yield worker
    finally:
        torch.npu.synchronize()
        for pool in pools:
            pool._hisparse_registration.close()


def test_split_planes_share_budget_and_isolate_layers(split_worker):
    worker = split_worker
    for layer, handle in enumerate(worker.cache_handles):
        latent, rope = cache_components(handle.view.cache)
        dense = cache_components(handle.view.attention_cache)
        assert all(component.is_contiguous() for component in dense)
        assert latent.untyped_storage().data_ptr() == rope.untyped_storage().data_ptr()
        assert latent.untyped_storage().nbytes() == worker.hot_backing.numel()
        assert handle.view.attention_block_stride == 256
        latent.fill_(layer + 1)
        rope.fill_(layer + 11)
    for layer, handle in enumerate(worker.cache_handles):
        for component, value in zip(cache_components(handle.view.cache), (layer + 1, layer + 11), strict=True):
            assert torch.equal(component.cpu(), torch.full(component.shape, value, dtype=component.dtype))


def test_split_mirror_spill_restore_and_hot_gather(split_worker):
    worker = split_worker
    for layer, handle in enumerate(worker.cache_handles):
        for component, value in zip(cache_components(handle.view.cache), (layer + 1, layer + 11), strict=True):
            component.fill_(value)
    # Partial row mirrors and whole-page spills must copy both planes.
    worker._set_row_mirrors((SparseKVRowMirror((2 * 128 + 9,), 128 + 13, 17),))
    worker._enqueue_row_dma((0, 1))
    worker._enqueue_transfers([SparseKVPageTransfer(1, 3, (4,), False)])
    worker.dma_stream.synchronize()
    for layer, handle in enumerate(worker.cache_handles):
        for component, value in zip(cache_components(handle.runtime.host_cache), (layer + 1, layer + 11), strict=True):
            assert torch.all(component[141:158] == value)
            assert torch.all(component[384:512] == value)
            assert torch.all(component[:128] == -7)
        for component in cache_components(handle.view.cache):
            component[5].zero_()
    worker._restore_pages([SparseKVPageTransfer(2, 3, (5,), False, restore=True)])
    torch.npu.synchronize()
    for layer, handle in enumerate(worker.cache_handles):
        for component, value in zip(cache_components(handle.view.cache), (layer + 1, layer + 11), strict=True):
            assert torch.all(component[5].cpu() == value)
        hot = split_paged_cache(
            worker.hot_backing,
            dtype=torch.bfloat16,
            row_width=576,
            byte_offset=layer * 128 * 576 * 2,
            block_stride=2 * 128 * 576 * 2,
            num_blocks=8,
            block_size=128,
        ).cache
        for component in cache_components(hot):
            component[6].zero_()
        source = torch.tensor([[384, 400, -1]], dtype=torch.int32, device="npu:0")
        destination = torch.tensor([[768, 771, -1]], dtype=torch.int32, device="npu:0")
        counts = torch.tensor([2], dtype=torch.int32, device="npu:0")
        handle.runtime.cache_ops.hisparse_gather_compact(handle.runtime.host_cache, hot, source, destination, counts)
        for component, value in zip(cache_components(hot), (layer + 1, layer + 11), strict=True):
            actual = component[6].cpu()
            assert torch.all(actual[0] == value) and torch.all(actual[3] == value)
            assert torch.all(actual[1:3] == 0)


def test_split_prefill_gathers_resident_hits_and_host_misses(split_worker):
    handle = split_worker.cache_handles[0]
    for component, value in zip(cache_components(handle.view.cache), (3, 13), strict=True):
        component.fill_(value)
    table = torch.tensor([[1, 3]], dtype=torch.int32, device="npu:0")
    resident_table = torch.tensor([[2, 0]], dtype=torch.int32, device="npu:0")
    plan = build_hisparse_prefill_staging_plan(table, torch.tensor([256], device="npu:0"), 128, 2)
    plan.ensure_gpu_sources(resident_table, handle.runtime.request_state_indices, 128)
    host = combine_components(tuple(c.view(-1, 128, c.shape[-1]) for c in cache_components(handle.runtime.host_cache)))
    staged = handle.runtime.gather_prefill_cache(host, plan, handle.view.cache)
    for component, value in zip(cache_components(staged), (3, 13), strict=True):
        actual = component.cpu()
        assert component.is_contiguous()
        assert torch.all(actual[1] == value)
        assert torch.all(actual[2] == -7)
