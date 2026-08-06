# SPDX-License-Identifier: Apache-2.0
"""Stream-ordered MemFabric transport used by KV layer parallelism.

This module intentionally does not depend on MooncakeLayerwiseConnector.  KVPP
has a different lifetime and ownership model: every rank owns a contiguous
bundle of layers and uses one full-size aliased scratch cache for the remaining
layers.  Layer owners therefore push only the active physical pages directly
into the same physical page IDs in every consumer's scratch cache.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.kv_transfer.kv_pool.kvpp_transport import (
    KVPPActivePages,
    KVPPBufferMetadata,
    KVPPCompletion,
    build_kvpp_layer_metadata,
    flatten_kvpp_cache,
)


_REGISTER_MERGE_GAP_BYTES = 4096


@dataclass(frozen=True)
class KVPPPeerMetadata:
    """MemFabric endpoint and cache layout published by one KVPP rank."""

    destination_session_id: str
    layers: dict[str, tuple[KVPPBufferMetadata, ...]]


@dataclass(frozen=True)
class _RegisterRegions:
    ptrs: list[int]
    lengths: list[int]


@dataclass(frozen=True)
class _SDMATransferBatch:
    source_addrs: tuple[int, ...]
    destination_addrs: tuple[int, ...]
    lengths: tuple[int, ...]


def _build_sdma_transfer_batch(
    pages: KVPPActivePages,
    source_buffers: tuple[KVPPBufferMetadata, ...],
    destination_buffers: tuple[KVPPBufferMetadata, ...],
) -> _SDMATransferBatch:
    """Materialize the legacy TransferEngine host descriptor ABI.

    This is deliberately SDMA-private. The common scheduler and MTE data path
    retain the active-page representation on device.
    """
    if len(source_buffers) != len(destination_buffers):
        raise RuntimeError(
            "KVPP owner and destination cache tensor counts differ: "
            f"owner={len(source_buffers)}, "
            f"destination={len(destination_buffers)}."
        )
    page_ids = pages.page_ids[pages.valid_mask].detach().cpu().tolist()
    source_addrs: list[int] = []
    destination_addrs: list[int] = []
    lengths: list[int] = []
    for source, destination in zip(source_buffers, destination_buffers):
        if source.block_bytes != destination.block_bytes:
            raise RuntimeError(
                "KVPP owner and destination cache block sizes differ: "
                f"owner={source.block_bytes}, "
                f"destination={destination.block_bytes}."
            )
        can_coalesce = (
            source.block_stride_bytes == source.block_bytes
            and destination.block_stride_bytes == destination.block_bytes
        )
        run_start = 0
        while run_start < len(page_ids):
            run_end = run_start + 1
            if can_coalesce:
                while (
                    run_end < len(page_ids)
                    and page_ids[run_end] == page_ids[run_end - 1] + 1
                ):
                    run_end += 1
            page = page_ids[run_start]
            source_addrs.append(
                source.base_addr + page * source.block_stride_bytes
            )
            destination_addrs.append(
                destination.base_addr + page * destination.block_stride_bytes
            )
            lengths.append((run_end - run_start) * source.block_bytes)
            run_start = run_end
    return _SDMATransferBatch(
        tuple(source_addrs), tuple(destination_addrs), tuple(lengths)
    )


@dataclass(frozen=True)
class MemFabricSDMACompletion:
    """Completion recorded after SDMA submissions on their NPU stream."""

    event: Any

    @classmethod
    def record(cls, stream: Any) -> "MemFabricSDMACompletion":
        event = torch.npu.Event()
        event.record(stream)
        return cls(event)

    def wait(self) -> None:
        # MemFabric Hybrid 1.2 SDMA stream submission is complete, including
        # remote visibility, when the event following it has completed.
        self.event.synchronize()

    def wait_on_stream(self, stream: Any) -> None:
        stream.wait_event(self.event)


def _storage_key(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().data_ptr()
    except Exception:
        return tensor.data_ptr()


def _collect_register_regions(kv_caches: dict[str, Any]) -> _RegisterRegions:
    """Merge aliased tensor views without importing a Mooncake connector."""
    ranges_by_storage: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for cache in kv_caches.values():
        for tensor in flatten_kvpp_cache(cache):
            if tensor.numel() == 0:
                continue
            start = tensor.data_ptr()
            ranges_by_storage[_storage_key(tensor)].append(
                (start, start + tensor.nbytes)
            )

    ptrs: list[int] = []
    lengths: list[int] = []
    for ranges in ranges_by_storage.values():
        ranges.sort()
        merged_start, merged_end = ranges[0]
        for start, end in ranges[1:]:
            if start <= merged_end + _REGISTER_MERGE_GAP_BYTES:
                merged_end = max(merged_end, end)
            else:
                ptrs.append(merged_start)
                lengths.append(merged_end - merged_start)
                merged_start, merged_end = start, end
        ptrs.append(merged_start)
        lengths.append(merged_end - merged_start)
    return _RegisterRegions(ptrs, lengths)


class MemFabricSDMAKVPPTransport:
    """Register KV caches and enqueue SDMA owner-to-scratch page pushes."""

    def __init__(
        self,
        group: GroupCoordinator,
        layer_owners: dict[str, int],
        num_blocks: int,
        engine: Any | None = None,
    ) -> None:
        self.group = group
        self.layer_owners = layer_owners
        self.num_blocks = num_blocks
        # Production uses two role-specific engines because the current Python
        # binding accepts only Prefill or Decode even though every KVPP rank is
        # both a cache owner and a consumer.  A single injected engine remains
        # useful for transport unit tests.
        self.source_engine = engine
        self.destination_engine = engine
        self.local_metadata: KVPPPeerMetadata | None = None
        self.peer_metadata: list[KVPPPeerMetadata] = []

    def initialize(self, kv_caches: dict[str, Any]) -> None:
        """Initialize MemFabric, register cache storage, and exchange pointers."""
        if self.source_engine is None:
            if not os.getenv("MEMFABRIC_HYBRID_HOME_PATH"):
                raise RuntimeError(
                    "KVPP MemFabric environment is not initialized. Source "
                    "/usr/local/memfabric_hybrid/set_env.sh before launching "
                    "the vLLM service."
                )
            try:
                from memfabric_hybrid import TransferEngine  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "KVPP requires memfabric_hybrid. Install the Ascend "
                    "MemFabric transfer package in the serving image."
                ) from exc
            self.source_engine = TransferEngine()
            self.destination_engine = TransferEngine()
        assert self.source_engine is not None
        assert self.destination_engine is not None

        store_url = os.getenv("ASCEND_MF_STORE_URL")
        if not store_url:
            raise RuntimeError(
                "KVPP MemFabric transport requires ASCEND_MF_STORE_URL."
            )

        device_id = torch.npu.current_device()
        protocol = os.getenv("ASCEND_MF_TRANSFER_PROTOCOL", "sdma").lower()
        if protocol != "sdma":
            raise RuntimeError(
                "KVPP requires ASCEND_MF_TRANSFER_PROTOCOL=sdma: the "
                "memfabric_hybrid 1.2 stream-submit API does not support "
                f"{protocol!r}."
            )

        enum_type = getattr(type(self.source_engine), "TransDataOpType", None)
        trans_op_type = enum_type.SDMA if enum_type is not None else None

        def initialize_engine(engine: Any, role: str) -> str:
            rpc_port = engine.get_rpc_port()
            session_id = f"{get_ip()}:{rpc_port}"
            if trans_op_type is None:
                ret = engine.initialize(store_url, session_id, role, device_id)
            else:
                ret = engine.initialize(
                    store_url, session_id, role, device_id, trans_op_type
                )
            if ret != 0:
                raise RuntimeError(
                    f"KVPP MemFabric {role} initialization failed: error={ret}."
                )
            if rpc_port == 0:
                session_id = f"{get_ip()}:{engine.get_rpc_port()}"
            return session_id

        # An explicitly injected engine is a unit-test seam.  Production
        # always creates a distinct Decode engine above.
        if self.destination_engine is self.source_engine:
            source_session_id = initialize_engine(self.source_engine, "Prefill")
            destination_session_id = source_session_id
        else:
            # MemFabric 1.2 publishes Decode sessions as write destinations.
            # Bring that endpoint up before the local Prefill/source endpoint.
            destination_session_id = initialize_engine(
                self.destination_engine, "Decode"
            )
            source_session_id = initialize_engine(self.source_engine, "Prefill")

        layer_metadata = build_kvpp_layer_metadata(kv_caches, self.num_blocks)
        regions = _collect_register_regions(kv_caches)
        for role, engine in (
            ("Prefill", self.source_engine),
            ("Decode", self.destination_engine),
        ):
            if role == "Decode" and engine is self.source_engine:
                continue
            ret = engine.batch_register_memory(regions.ptrs, regions.lengths)
            if ret != 0:
                raise RuntimeError(
                    f"KVPP MemFabric {role} cache registration failed: "
                    f"error={ret}, regions={len(regions.ptrs)}."
                )

        self.local_metadata = KVPPPeerMetadata(
            destination_session_id=destination_session_id,
            layers=layer_metadata,
        )
        peers: list[KVPPPeerMetadata | None] = [None] * self.group.world_size
        dist.all_gather_object(
            peers, self.local_metadata, group=self.group.cpu_group
        )
        if any(peer is None for peer in peers):
            raise RuntimeError("KVPP did not receive MemFabric metadata from every rank.")
        self.peer_metadata = [peer for peer in peers if peer is not None]
        logger.info(
            "KVPP MemFabric initialized: rank=%d, session=%s, protocol=%s, regions=%d",
            self.group.rank_in_group,
            f"source={source_session_id},destination={destination_session_id}",
            protocol,
            len(regions.ptrs),
        )

    def push_active_pages(
        self,
        layer_name: str,
        pages: KVPPActivePages,
        stream: Any,
    ) -> KVPPCompletion:
        """Push active owner pages into every peer's aliased scratch cache."""
        owner_rank = self.layer_owners[layer_name]
        if owner_rank != self.group.rank_in_group:
            return MemFabricSDMACompletion.record(stream)
        if self.local_metadata is None or not self.peer_metadata:
            raise RuntimeError("KVPP MemFabric transport was not initialized.")

        raw_stream = stream.npu_stream
        assert self.source_engine is not None
        source_buffers = self.local_metadata.layers[layer_name]
        for peer_rank, destination_peer in enumerate(self.peer_metadata):
            if peer_rank == owner_rank:
                continue
            destination_buffers = destination_peer.layers[layer_name]
            transfers = _build_sdma_transfer_batch(
                pages, source_buffers, destination_buffers
            )

            batch_submit = getattr(
                self.source_engine, "batch_transfer_async_write_submit", None
            )
            if batch_submit is not None:
                ret = batch_submit(
                    destination_peer.destination_session_id,
                    list(transfers.source_addrs),
                    list(transfers.destination_addrs),
                    list(transfers.lengths),
                    raw_stream,
                )
                if ret != 0:
                    raise RuntimeError(
                        "KVPP MemFabric batch write submit failed for "
                        f"{layer_name} to rank {peer_rank}: error={ret}."
                    )
                continue

            for source_addr, destination_addr, length in zip(
                transfers.source_addrs,
                transfers.destination_addrs,
                transfers.lengths,
            ):
                ret = self.source_engine.transfer_async_write_submit(
                    destination_peer.destination_session_id,
                    source_addr,
                    destination_addr,
                    length,
                    raw_stream,
                )
                if ret != 0:
                    raise RuntimeError(
                        f"KVPP MemFabric write submit failed for {layer_name} "
                        f"to rank {peer_rank}: error={ret}."
                    )

        return MemFabricSDMACompletion.record(stream)

    def receive_active_pages(
        self,
        layer_name: str,
        pages: KVPPActivePages,
        stream: Any,
    ) -> KVPPCompletion:
        """SDMA writes directly into scratch, so no receive-side unpack exists."""
        return MemFabricSDMACompletion.record(stream)

    def close(self) -> None:
        """Drop backend metadata and TransferEngine references.

        MemFabric Hybrid 1.2 does not expose a stable Python teardown API for
        these sessions.  Their native resources remain process-scoped; clearing
        Python references here prevents KVPP from retaining cache metadata.
        """
        self.peer_metadata.clear()
        self.local_metadata = None
        self.source_engine = None
        self.destination_engine = None


# Compatibility alias for callers created before the transport boundary was
# introduced.  New code should use create_kvpp_transport() or the explicit
# SDMA class name.
MemFabricKVPPTransport = MemFabricSDMAKVPPTransport
