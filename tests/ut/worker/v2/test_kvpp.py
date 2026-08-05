from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.memfabric_mte_transport import (
    KVPPMTEPeerMetadata,
    MemFabricMTEKVPPTransport,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.memfabric_sdma_transport import (
    KVPPPeerMetadata,
    MemFabricSDMAKVPPTransport,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.kvpp_transport import (
    KVPPBufferMetadata,
    build_kvpp_page_transfer_batch,
    create_kvpp_transport,
    register_kvpp_transport,
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
    transfers = build_kvpp_page_transfer_batch(
        (2, 3, 7), (local,), (remote,)
    )

    assert transfers.source_addrs == (1032, 1112)
    assert transfers.destination_addrs == (2032, 2112)
    assert transfers.lengths == (32, 16)


def test_padded_page_layout_keeps_one_descriptor_per_page():
    local = KVPPBufferMetadata(1000, 32, 16)
    remote = KVPPBufferMetadata(2000, 32, 16)
    transfers = build_kvpp_page_transfer_batch((2, 3), (local,), (remote,))

    assert transfers.source_addrs == (1064, 1096)
    assert transfers.destination_addrs == (2064, 2096)
    assert transfers.lengths == (16, 16)


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


def test_memfabric_owner_push_targets_original_physical_page_ids(monkeypatch):
    class FakeEvent:
        def __init__(self):
            self.recorded_stream = None
            self.synchronized = False

        def record(self, stream):
            self.recorded_stream = stream

        def synchronize(self):
            self.synchronized = True

    event = FakeEvent()
    monkeypatch.setattr(torch.npu, "Event", lambda: event)
    group = SimpleNamespace(rank_in_group=0, world_size=2)
    engine = _FakeEngine()
    transport = MemFabricSDMAKVPPTransport(
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

    transfer_stream = SimpleNamespace(npu_stream=123)
    completion = transport.push_active_pages("layer", (2, 3, 7), transfer_stream)

    assert engine.calls == [
        ("peer-destination", [2032, 2112], [1032, 1112], [32, 16], 123)
    ]
    assert completion.event is event
    assert event.recorded_stream is transfer_stream
    completion.wait()
    assert event.synchronized

    waited_events = []
    completion.wait_on_stream(SimpleNamespace(wait_event=waited_events.append))
    assert waited_events == [event]


def test_transport_factory_accepts_registered_mte_backend(monkeypatch):
    calls = []

    class FakeMTETransport:
        def __init__(self, group, layer_owners, num_blocks):
            calls.append((group, layer_owners, num_blocks))

        def initialize(self, kv_caches):
            pass

        def push_active_pages(self, layer_name, pages, stream):
            pass

        def receive_active_pages(self, layer_name, pages, stream):
            pass

        def close(self):
            pass

    register_kvpp_transport("test-mte", FakeMTETransport)
    monkeypatch.delenv("ASCEND_KVPP_TRANSPORT_CLASS", raising=False)
    monkeypatch.setenv("ASCEND_KVPP_TRANSPORT", "test-mte")
    group = SimpleNamespace(rank_in_group=0, world_size=2)

    transport = create_kvpp_transport(group, {"layer": 0}, num_blocks=10)

    assert isinstance(transport, FakeMTETransport)
    assert calls == [(group, {"layer": 0}, 10)]


def test_unavailable_transport_has_actionable_error(monkeypatch):
    monkeypatch.delenv("ASCEND_KVPP_TRANSPORT_CLASS", raising=False)

    with pytest.raises(RuntimeError, match="ASCEND_KVPP_TRANSPORT_CLASS"):
        create_kvpp_transport(
            SimpleNamespace(), {"layer": 0}, num_blocks=10, backend="unknown"
        )


def test_mte_owner_stages_and_consumer_unpacks_same_active_pages(monkeypatch):
    class FakeEvent:
        def record(self, stream):
            self.stream = stream

        def synchronize(self):
            pass

    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    calls = []

    def copy_op(anchor, sources, destinations, lengths):
        calls.append(
            (
                anchor,
                tuple(sources.tolist()),
                tuple(destinations.tolist()),
                tuple(lengths.tolist()),
            )
        )

    stream = SimpleNamespace()
    owner_anchor = torch.empty(1)
    owner = MemFabricMTEKVPPTransport(
        SimpleNamespace(rank_in_group=0, world_size=2),
        {"layer": 0},
        10,
        copy_op=copy_op,
    )
    owner._layers = {"layer": (KVPPBufferMetadata(2000, 16, 16),)}
    owner._anchors = {"layer": owner_anchor}
    owner._local_metadata = KVPPMTEPeerMetadata(8000, 1024)
    owner._peer_metadata = [
        owner._local_metadata,
        KVPPMTEPeerMetadata(10000, 1024),
    ]

    owner.push_active_pages("layer", (2, 3, 7), stream)
    assert calls == [
        (owner_anchor, (2032, 2112), (10000, 10032), (32, 16))
    ]

    calls.clear()
    consumer_anchor = torch.empty(1)
    consumer = MemFabricMTEKVPPTransport(
        SimpleNamespace(rank_in_group=1, world_size=2),
        {"layer": 0},
        10,
        copy_op=copy_op,
    )
    consumer._layers = {"layer": (KVPPBufferMetadata(1000, 16, 16),)}
    consumer._anchors = {"layer": consumer_anchor}
    consumer._local_metadata = KVPPMTEPeerMetadata(10000, 1024)
    consumer._peer_metadata = [
        KVPPMTEPeerMetadata(8000, 1024),
        consumer._local_metadata,
    ]

    consumer.receive_active_pages("layer", (2, 3, 7), stream)
    assert calls == [
        (consumer_anchor, (10000, 10032), (1032, 1112), (32, 16))
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
