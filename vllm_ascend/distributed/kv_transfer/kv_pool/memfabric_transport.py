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


_REGISTER_MERGE_GAP_BYTES = 4096


@dataclass(frozen=True)
class KVPPBufferMetadata:
    """Address and logical-page layout for one KV cache tensor."""

    base_addr: int
    block_stride_bytes: int
    block_bytes: int


@dataclass(frozen=True)
class KVPPPeerMetadata:
    """MemFabric endpoint and cache layout published by one KVPP rank."""

    destination_session_id: str
    layers: dict[str, tuple[KVPPBufferMetadata, ...]]


@dataclass(frozen=True)
class _RegisterRegions:
    ptrs: list[int]
    lengths: list[int]


def _flatten_tensors(cache: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(cache, torch.Tensor):
        return (cache,)
    if isinstance(cache, (tuple, list)):
        tensors = tuple(cache)
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("KVPP cache tuples may contain only tensors.")
        return tensors
    raise TypeError(f"Unsupported KVPP cache type: {type(cache)!r}.")


def _storage_key(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().data_ptr()
    except Exception:
        return tensor.data_ptr()


def _collect_register_regions(kv_caches: dict[str, Any]) -> _RegisterRegions:
    """Merge aliased tensor views without importing a Mooncake connector."""
    ranges_by_storage: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for cache in kv_caches.values():
        for tensor in _flatten_tensors(cache):
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


def _build_layer_metadata(
    kv_caches: dict[str, Any], num_blocks: int
) -> dict[str, tuple[KVPPBufferMetadata, ...]]:
    layers: dict[str, tuple[KVPPBufferMetadata, ...]] = {}
    for layer_name, cache in kv_caches.items():
        buffers: list[KVPPBufferMetadata] = []
        for tensor in _flatten_tensors(cache):
            if tensor.ndim == 0 or tensor.shape[0] % num_blocks != 0:
                raise RuntimeError(
                    f"KVPP layer {layer_name} cache shape {tuple(tensor.shape)} "
                    f"cannot be divided into {num_blocks} logical blocks."
                )
            block_size_scale = tensor.shape[0] // num_blocks
            block_stride_bytes = (
                tensor.stride(0) * tensor.element_size() * block_size_scale
            )
            logical_block = tensor[0:block_size_scale]
            if not logical_block.is_contiguous():
                raise RuntimeError(
                    f"KVPP layer {layer_name} logical cache block is not "
                    "contiguous and cannot be transferred by address."
                )
            block_bytes = logical_block.numel() * tensor.element_size()
            if block_bytes > block_stride_bytes:
                raise RuntimeError(
                    f"KVPP layer {layer_name} has overlapping logical blocks: "
                    f"payload={block_bytes}, stride={block_stride_bytes}."
                )
            buffers.append(
                KVPPBufferMetadata(
                    base_addr=tensor.data_ptr(),
                    block_stride_bytes=block_stride_bytes,
                    block_bytes=block_bytes,
                )
            )
        layers[layer_name] = tuple(buffers)
    return layers


def _append_page_transfers(
    pages: tuple[int, ...],
    source: KVPPBufferMetadata,
    destination: KVPPBufferMetadata,
    source_addrs: list[int],
    destination_addrs: list[int],
    lengths: list[int],
) -> None:
    if source.block_bytes != destination.block_bytes:
        raise RuntimeError(
            "KVPP owner and destination cache block sizes differ: "
            f"owner={source.block_bytes}, "
            f"destination={destination.block_bytes}."
        )

    # Adjacent pages can be submitted as one region only when neither layout
    # has padding between logical blocks.  Otherwise retain one descriptor per
    # page so no unrelated bytes are transferred.
    can_coalesce = (
        source.block_stride_bytes == source.block_bytes
        and destination.block_stride_bytes == destination.block_bytes
    )
    run_start = 0
    while run_start < len(pages):
        run_end = run_start + 1
        if can_coalesce:
            while run_end < len(pages) and pages[run_end] == pages[run_end - 1] + 1:
                run_end += 1
        page = pages[run_start]
        source_addrs.append(source.base_addr + page * source.block_stride_bytes)
        destination_addrs.append(
            destination.base_addr + page * destination.block_stride_bytes
        )
        lengths.append((run_end - run_start) * source.block_bytes)
        run_start = run_end


class MemFabricKVPPTransport:
    """Register KV caches and enqueue owner-to-scratch page pushes."""

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

        layer_metadata = _build_layer_metadata(kv_caches, self.num_blocks)
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
        pages: tuple[int, ...],
        stream: Any,
    ) -> None:
        """Push active owner pages into every peer's aliased scratch cache."""
        owner_rank = self.layer_owners[layer_name]
        if owner_rank != self.group.rank_in_group or not pages:
            return
        if self.local_metadata is None or not self.peer_metadata:
            raise RuntimeError("KVPP MemFabric transport was not initialized.")

        raw_stream = stream.npu_stream
        assert self.source_engine is not None
        source_buffers = self.local_metadata.layers[layer_name]
        for peer_rank, destination_peer in enumerate(self.peer_metadata):
            if peer_rank == owner_rank:
                continue
            destination_buffers = destination_peer.layers[layer_name]
            if len(source_buffers) != len(destination_buffers):
                raise RuntimeError(
                    f"KVPP layer {layer_name} cache tensor count differs between "
                    f"owner {owner_rank} and destination {peer_rank}."
                )

            source_addrs: list[int] = []
            destination_addrs: list[int] = []
            lengths: list[int] = []
            for source, destination in zip(
                source_buffers, destination_buffers
            ):
                _append_page_transfers(
                    pages,
                    source,
                    destination,
                    source_addrs,
                    destination_addrs,
                    lengths,
                )

            batch_submit = getattr(
                self.source_engine, "batch_transfer_async_write_submit", None
            )
            if batch_submit is not None:
                ret = batch_submit(
                    destination_peer.destination_session_id,
                    source_addrs,
                    destination_addrs,
                    lengths,
                    raw_stream,
                )
                if ret != 0:
                    raise RuntimeError(
                        "KVPP MemFabric batch write submit failed for "
                        f"{layer_name} to rank {peer_rank}: error={ret}."
                    )
                continue

            for source_addr, destination_addr, length in zip(
                source_addrs, destination_addrs, lengths
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
