from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.memfabric_transport import (
    KVPPBufferMetadata,
    KVPPPeerMetadata,
    MemFabricKVPPTransport,
    _append_page_transfers,
)
from vllm_ascend.worker.v2.kvpp import KVPPContext, _active_pages


def test_active_pages_uses_only_pages_covered_by_sequence_lengths():
    block_table = torch.tensor([[7, 2, 9, 0], [4, 8, 0, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([5, 5], dtype=torch.int32)

    pages = _active_pages(block_table, seq_lens, block_size=4, num_blocks=10)

    assert pages == (2, 4, 7, 8)


def test_contiguous_pages_are_coalesced_without_remapping():
    local = KVPPBufferMetadata(1000, 16, 16)
    remote = KVPPBufferMetadata(2000, 16, 16)
    local_addrs: list[int] = []
    remote_addrs: list[int] = []
    lengths: list[int] = []

    _append_page_transfers(
        (2, 3, 7), local, remote, local_addrs, remote_addrs, lengths
    )

    assert local_addrs == [1032, 1112]
    assert remote_addrs == [2032, 2112]
    assert lengths == [32, 16]


def test_padded_page_layout_keeps_one_descriptor_per_page():
    local = KVPPBufferMetadata(1000, 32, 16)
    remote = KVPPBufferMetadata(2000, 32, 16)
    local_addrs: list[int] = []
    remote_addrs: list[int] = []
    lengths: list[int] = []

    _append_page_transfers(
        (2, 3), local, remote, local_addrs, remote_addrs, lengths
    )

    assert local_addrs == [1064, 1096]
    assert remote_addrs == [2064, 2096]
    assert lengths == [16, 16]


class _FakeEngine:
    def __init__(self):
        self.calls = []

    def batch_transfer_async_write_submit(
        self, session_id, source_addrs, destination_addrs, lengths, stream
    ):
        self.calls.append(
            (session_id, source_addrs, destination_addrs, lengths, stream)
        )
        return 0


def test_memfabric_owner_push_targets_original_physical_page_ids():
    group = SimpleNamespace(rank_in_group=0, world_size=2)
    engine = _FakeEngine()
    transport = MemFabricKVPPTransport(
        group=group,
        layer_owners={"layer": 0},
        num_blocks=10,
        engine=engine,
    )
    transport.local_metadata = KVPPPeerMetadata(
        "owner-destination",
        {"layer": (KVPPBufferMetadata(2000, 16, 16),)},
    )
    transport.peer_metadata = [
        transport.local_metadata,
        KVPPPeerMetadata(
            "peer-destination",
            {"layer": (KVPPBufferMetadata(1000, 16, 16),)},
        ),
    ]

    transport.push_active_pages(
        "layer", (2, 3, 7), SimpleNamespace(npu_stream=123)
    )

    assert engine.calls == [
        ("peer-destination", [2032, 2112], [1032, 1112], [32, 16], 123)
    ]


def test_owner_uses_persistent_cache():
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    fake_transport = SimpleNamespace(initialize=lambda caches: None)
    context = KVPPContext(
        group=group,
        layer_owners={"layer": 0},
        num_blocks=10,
        block_size=4,
        transport=fake_transport,
    )
    block_table = torch.tensor([[7, 2]], dtype=torch.int32)
    cache = (torch.zeros((10, 4, 1)),)
    context.prepare_batch(block_table, torch.tensor([5]))

    returned = context.begin_layer("layer", cache)

    assert returned is cache
    context.wait_for_current_layer("layer")


def test_previous_layer_mode_prefetches_layers_in_forward_order():
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    context = KVPPContext(
        group=group,
        layer_owners={"layer.0": 0, "layer.1": 0},
        num_blocks=10,
        block_size=4,
        transport=SimpleNamespace(),
        overlap_mode="previous_layer",
    )
    block_table = torch.tensor([[7, 2]], dtype=torch.int32)
    cache = (torch.zeros((10, 4, 1)),)
    context.prepare_batch(block_table, torch.tensor([5]))

    context.begin_layer("layer.0", cache)
    context.wait_for_current_layer("layer.0")
    context.finish_layer_attention("layer.0")

    assert context._pending_layer == "layer.1"
    context.begin_layer("layer.1", cache)
    context.wait_for_current_layer("layer.1")
    context.finish_layer_attention("layer.1")
    assert context._pending_layer is None


def test_invalid_overlap_mode_is_rejected():
    group = SimpleNamespace(rank_in_group=0, world_size=1)

    with pytest.raises(ValueError, match="ASCEND_KVPP_OVERLAP_MODE"):
        KVPPContext(
            group=group,
            layer_owners={"layer": 0},
            num_blocks=10,
            block_size=4,
            overlap_mode="invalid",
        )
