from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_pool.memfabric_mte_transport import (
    KVPPMTEPeerMetadata,
    MemFabricMTEKVPPTransport,
    _MTEDeviceBufferMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.memfabric_sdma_transport import (
    KVPPPeerMetadata,
    MemFabricSDMAKVPPTransport,
    _build_sdma_transfer_batch,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.kvpp_transport import (
    KVPPActivePages,
    KVPPBufferMetadata,
    create_kvpp_transport,
    register_kvpp_transport,
)
from vllm_ascend.worker.v2.kvpp import KVPPContext, _active_pages


def test_active_pages_uses_only_pages_covered_by_sequence_lengths():
    block_table = torch.tensor([[7, 2, 9, 0], [4, 8, 0, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([5, 5], dtype=torch.int32)

    original_block_table = block_table.clone()
    pages = _active_pages(block_table, seq_lens, block_size=4, num_blocks=10)

    assert pages.page_ids.tolist() == [2, 4, 7, 8, 10, 10, 10, 10]
    assert pages.valid_mask.tolist() == [True, True, True, True, False, False,
                                        False, False]
    assert pages.page_ids.device == block_table.device
    assert pages.valid_mask.device == block_table.device
    assert pages.count_upper_bound == 4
    assert torch.equal(block_table, original_block_table)


def _active_page_tensor(*page_ids: int) -> KVPPActivePages:
    pages = torch.tensor(page_ids, dtype=torch.int32)
    return KVPPActivePages(
        pages,
        torch.ones_like(pages, dtype=torch.bool),
        count_upper_bound=len(page_ids),
    )


def test_contiguous_pages_are_coalesced_without_remapping():
    local = KVPPBufferMetadata(1000, 16, 16)
    remote = KVPPBufferMetadata(2000, 16, 16)
    transfers = _build_sdma_transfer_batch(
        _active_page_tensor(2, 3, 7), (local,), (remote,)
    )

    assert transfers.source_addrs == (1032, 1112)
    assert transfers.destination_addrs == (2032, 2112)
    assert transfers.lengths == (32, 16)


def test_padded_page_layout_keeps_one_descriptor_per_page():
    local = KVPPBufferMetadata(1000, 32, 16)
    remote = KVPPBufferMetadata(2000, 32, 16)
    transfers = _build_sdma_transfer_batch(
        _active_page_tensor(2, 3), (local,), (remote,)
    )

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
    completion = transport.push_active_pages(
        "layer", _active_page_tensor(2, 3, 7), transfer_stream
    )

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

    def copy_op(
        anchor,
        local_offsets,
        staging_offsets,
        lengths,
        staging_base,
        source_rank,
        destination_rank,
        shm_id,
    ):
        calls.append(
            (
                anchor,
                tuple(local_offsets.tolist()),
                tuple(staging_offsets.tolist()),
                tuple(lengths.tolist()),
                staging_base,
                source_rank,
                destination_rank,
                shm_id,
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
    owner._device_layers = {
        "layer": _MTEDeviceBufferMetadata(
            torch.tensor([0]), torch.tensor([16]), torch.tensor([16]), 16
        )
    }
    owner._local_metadata = KVPPMTEPeerMetadata(8000, 1024, 0)
    owner._peer_metadata = [
        owner._local_metadata,
        KVPPMTEPeerMetadata(8000, 1024, 1),
    ]

    owner.push_active_pages("layer", _active_page_tensor(2, 3, 7), stream)
    assert calls == [
        (
            owner_anchor,
            (32, 48, 112),
            (0, 16, 32),
            (16, 16, 16),
            8000,
            -1,
            1,
            31,
        )
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
    consumer._device_layers = {
        "layer": _MTEDeviceBufferMetadata(
            torch.tensor([0]), torch.tensor([16]), torch.tensor([16]), 16
        )
    }
    consumer._local_metadata = KVPPMTEPeerMetadata(8000, 1024, 1)
    consumer._peer_metadata = [
        KVPPMTEPeerMetadata(8000, 1024, 0),
        consumer._local_metadata,
    ]

    consumer.receive_active_pages(
        "layer", _active_page_tensor(2, 3, 7), stream
    )
    assert calls == [
        (
            consumer_anchor,
            (32, 48, 112),
            (0, 16, 32),
            (16, 16, 16),
            8000,
            1,
            -1,
            31,
        )
    ]


def test_mte_builds_one_device_batch_for_masked_pages_and_multiple_buffers(
    monkeypatch,
):
    class FakeEvent:
        def record(self, stream):
            self.stream = stream

        def synchronize(self):
            pass

    monkeypatch.setattr(torch.npu, "Event", FakeEvent)
    monkeypatch.setattr(
        torch,
        "_assert_async",
        lambda *args, **kwargs: pytest.fail(
            "MTE capacity validation must not launch a device assertion"
        ),
    )
    calls = []

    def copy_op(anchor, local_offsets, staging_offsets, lengths,
                staging_base, source_rank, destination_rank, shm_id):
        calls.append(
            (
                tuple(local_offsets.tolist()),
                tuple(staging_offsets.tolist()),
                tuple(lengths.tolist()),
                staging_base,
                source_rank,
                destination_rank,
                shm_id,
            )
        )

    transport = MemFabricMTEKVPPTransport(
        SimpleNamespace(rank_in_group=0, world_size=2),
        {"layer": 0},
        10,
        copy_op=copy_op,
    )
    transport._anchors = {"layer": torch.empty(1)}
    transport._device_layers = {
        "layer": _MTEDeviceBufferMetadata(
            torch.tensor([0, 4000]),
            torch.tensor([32, 64]),
            torch.tensor([16, 8]),
            24,
        )
    }
    transport._local_metadata = KVPPMTEPeerMetadata(8000, 240, 0)
    transport._peer_metadata = [
        transport._local_metadata,
        KVPPMTEPeerMetadata(8000, 240, 1),
    ]
    pages = KVPPActivePages(
        torch.tensor([2, 2, 7, 10], dtype=torch.int32),
        torch.tensor([True, False, True, False]),
        count_upper_bound=2,
    )

    transport.push_active_pages("layer", pages, SimpleNamespace())

    assert calls == [
        (
            (64, 64, 224, 320, 4128, 4128, 4448, 4640),
            (0, 0, 16, 16, 160, 160, 168, 168),
            (16, 0, 16, 0, 8, 0, 8, 0),
            8000,
            -1,
            1,
            31,
        )
    ]


def test_mte_rejects_host_upper_bound_larger_than_staging_capacity():
    transport = MemFabricMTEKVPPTransport(
        SimpleNamespace(rank_in_group=0, world_size=2),
        {"layer": 0},
        10,
        copy_op=lambda *args: pytest.fail("copy must not be launched"),
    )
    transport._anchors = {"layer": torch.empty(1)}
    transport._device_layers = {
        "layer": _MTEDeviceBufferMetadata(
            torch.tensor([0]),
            torch.tensor([16]),
            torch.tensor([16]),
            16,
        )
    }
    transport._local_metadata = KVPPMTEPeerMetadata(8000, 32, 0)
    transport._peer_metadata = [
        transport._local_metadata,
        KVPPMTEPeerMetadata(8000, 32, 1),
    ]
    pages = KVPPActivePages(
        torch.tensor([2, 3, 7], dtype=torch.int32),
        torch.tensor([True, True, True]),
        count_upper_bound=3,
    )

    with pytest.raises(
        RuntimeError,
        match="upper_bound=3, capacity=2",
    ):
        transport.push_active_pages("layer", pages, SimpleNamespace())


def test_prepare_batch_preserves_attention_metadata():
    block_table = torch.tensor([[7, 2, 9, 0]], dtype=torch.int32)
    slot_mapping = torch.tensor([28, 29, 8, 9], dtype=torch.int64)
    original_block_table = block_table.clone()
    original_slot_mapping = slot_mapping.clone()
    context = KVPPContext(
        group=SimpleNamespace(rank_in_group=0, world_size=1),
        layer_owners={"layer": 0},
        num_blocks=10,
        block_size=4,
        transport=SimpleNamespace(),
    )

    context.prepare_batch(block_table, torch.tensor([5]))

    assert torch.equal(block_table, original_block_table)
    assert torch.equal(slot_mapping, original_slot_mapping)


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


def test_previous_layer_mode_prefetches_before_current_attention():
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
    assert context._pending_layer == "layer.1"
    context.finish_layer_attention("layer.0")

    assert context._pending_layer == "layer.1"
    context.begin_layer("layer.1", cache)
    context.wait_for_current_layer("layer.1")
    context.finish_layer_attention("layer.1")
    assert context._pending_layer is None


def test_dual_buffer_overlap_is_the_default(monkeypatch):
    monkeypatch.delenv("ASCEND_KVPP_OVERLAP_MODE", raising=False)
    context = KVPPContext(
        group=SimpleNamespace(rank_in_group=0, world_size=1),
        layer_owners={"layer": 0},
        num_blocks=10,
        block_size=4,
        transport=SimpleNamespace(),
    )

    assert context.overlap_mode == "previous_layer"


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
