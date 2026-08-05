# SPDX-License-Identifier: Apache-2.0
"""MemFabric Hybrid MTE backend for KV layer parallelism.

MemFabric Hybrid 1.2 cannot export an existing KV tensor as a remote MTE GVA.
This backend therefore allocates one bounded symmetric active-page staging
segment per rank. Layer owners copy selected persistent pages directly to each
consumer's segment with an AscendC GM->UB->remote-GM kernel. Consumers unpack
the same compact descriptors into their existing scratch cache before
attention. No full layer cache is copied or staged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.kvpp_transport import (
    KVPPBufferMetadata,
    KVPPCompletion,
    KVPPTransferBatch,
    build_kvpp_layer_metadata,
    build_kvpp_page_transfer_batch,
    flatten_kvpp_cache,
)


_DEFAULT_STAGING_BYTES = 256 << 20
_DEFAULT_SHM_ID = 31
_SHM_ID_LIMIT = 64
_SHM_ALIGNMENT = 2 << 20


@dataclass(frozen=True)
class KVPPMTEPeerMetadata:
    """Backend-specific symmetric address published by one rank."""

    staging_addr: int
    staging_bytes: int
    rank: int


@dataclass(frozen=True)
class MemFabricMTECompletion:
    """Stream event recorded after all MTE kernels in one phase."""

    event: Any

    @classmethod
    def record(cls, stream: Any) -> "MemFabricMTECompletion":
        event = torch.npu.Event()
        event.record(stream)
        return cls(event)

    def wait(self) -> None:
        self.event.synchronize()

    def wait_on_stream(self, stream: Any) -> None:
        stream.wait_event(self.event)


def _compact_staging_batch(
    transfer: KVPPTransferBatch, staging_addr: int, staging_bytes: int
) -> KVPPTransferBatch:
    offsets: list[int] = []
    offset = 0
    for length in transfer.lengths:
        offsets.append(staging_addr + offset)
        offset += length
    if offset > staging_bytes:
        raise RuntimeError(
            "KVPP MTE active pages exceed the bounded staging segment: "
            f"required={offset} bytes, capacity={staging_bytes} bytes. "
            "Increase ASCEND_KVPP_MTE_STAGING_BYTES or reduce the batch."
        )
    return KVPPTransferBatch(
        source_addrs=transfer.source_addrs,
        destination_addrs=tuple(offsets),
        lengths=transfer.lengths,
    )


class MemFabricMTEKVPPTransport:
    """Move active physical pages through bounded symmetric MTE staging."""

    def __init__(
        self,
        group: GroupCoordinator,
        layer_owners: dict[str, int],
        num_blocks: int,
        *,
        shm_module: Any | None = None,
        copy_op: Callable[..., None] | None = None,
    ) -> None:
        self.group = group
        self.layer_owners = layer_owners
        self.num_blocks = num_blocks
        self._shm_module = shm_module
        self._copy_op = copy_op
        self._memory: Any | None = None
        self._local_metadata: KVPPMTEPeerMetadata | None = None
        self._peer_metadata: list[KVPPMTEPeerMetadata] = []
        self._layers: dict[str, tuple[KVPPBufferMetadata, ...]] = {}
        self._anchors: dict[str, torch.Tensor] = {}
        self._shm_id = _DEFAULT_SHM_ID

    def initialize(self, kv_caches: dict[str, Any]) -> None:
        if not os.getenv("MEMFABRIC_HYBRID_HOME_PATH"):
            raise RuntimeError(
                "KVPP MTE requires the MemFabric Hybrid environment. Source "
                "/usr/local/memfabric_hybrid/set_env.sh before launching vLLM."
            )
        if self._shm_module is None:
            try:
                from memfabric_hybrid import shm  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "KVPP MTE requires memfabric_hybrid.shm."
                ) from exc
            self._shm_module = shm
        if self._copy_op is None:
            # vllm-ascend loads its native extension lazily to avoid early RTS
            # initialization. KVPP needs the operator during cache setup, so
            # trigger that load before querying the dispatcher namespace.
            import vllm_ascend.vllm_ascend_C  # type: ignore # noqa: F401

            namespace = getattr(torch.ops, "_C_ascend", None)
            self._copy_op = getattr(namespace, "kvpp_mte_copy", None)
            if self._copy_op is None:
                raise RuntimeError(
                    "KVPP MTE custom operator is unavailable. Rebuild "
                    "vllm-ascend after sourcing MemFabric Hybrid 1.2.0."
                )

        store_url = os.getenv("MF_CONFIG_STORE_URL") or os.getenv(
            "ASCEND_MF_STORE_URL"
        )
        if not store_url:
            raise RuntimeError(
                "KVPP MTE requires MF_CONFIG_STORE_URL (or the deprecated "
                "ASCEND_MF_STORE_URL compatibility variable)."
            )
        staging_bytes = int(
            os.getenv("ASCEND_KVPP_MTE_STAGING_BYTES", _DEFAULT_STAGING_BYTES)
        )
        if staging_bytes <= 0 or staging_bytes % _SHM_ALIGNMENT:
            raise ValueError(
                "ASCEND_KVPP_MTE_STAGING_BYTES must be a positive multiple "
                f"of {_SHM_ALIGNMENT} bytes, got {staging_bytes}."
            )
        shm_id = int(os.getenv("ASCEND_KVPP_MTE_SHM_ID", _DEFAULT_SHM_ID))
        if not 0 <= shm_id < _SHM_ID_LIMIT:
            raise ValueError(
                f"ASCEND_KVPP_MTE_SHM_ID must be in [0, {_SHM_ID_LIMIT}), "
                f"got {shm_id}."
            )
        self._shm_id = shm_id

        config = self._shm_module.ShmConfig()
        config.start_store = self.group.rank_in_group == 0
        timeout = int(os.getenv("ASCEND_KVPP_MTE_TIMEOUT_SECONDS", "120"))
        config.init_timeout = timeout
        config.create_timeout = timeout
        config.operation_timeout = timeout
        device_id = torch.npu.current_device()
        ret = self._shm_module.initialize(
            store_url,
            self.group.world_size,
            self.group.rank_in_group,
            device_id,
            config,
        )
        if ret != 0:
            raise RuntimeError(
                f"KVPP MemFabric SHM initialization failed: error={ret}."
            )
        self._memory = self._shm_module.create(
            shm_id,
            self.group.world_size,
            self.group.rank_in_group,
            staging_bytes,
            self._shm_module.ShmDataOpType.MTE,
        )
        if self._memory is None:
            raise RuntimeError("KVPP MemFabric SHM creation returned no memory.")
        operation = int(self._memory.query_support_data_operation())
        if operation != int(self._shm_module.ShmDataOpType.MTE.value):
            raise RuntimeError(
                "KVPP MemFabric SHM does not support MTE: "
                f"reported operation={operation}."
            )

        self._layers = build_kvpp_layer_metadata(kv_caches, self.num_blocks)
        self._anchors = {
            layer_name: flatten_kvpp_cache(cache)[0]
            for layer_name, cache in kv_caches.items()
        }
        # ``gva`` is the common symmetric base. MemFabric may align each
        # rank's segment to an internal symmetric size larger than the local
        # contribution. That size is intentionally queried inside the
        # AscendC kernel; the Python binding does not expose it.
        self._local_metadata = KVPPMTEPeerMetadata(
            staging_addr=int(self._memory.gva),
            staging_bytes=staging_bytes,
            rank=self.group.rank_in_group,
        )
        peers: list[KVPPMTEPeerMetadata | None] = [None] * self.group.world_size
        dist.all_gather_object(
            peers, self._local_metadata, group=self.group.cpu_group
        )
        if any(peer is None for peer in peers):
            raise RuntimeError("KVPP MTE did not receive every peer GVA.")
        self._peer_metadata = [peer for peer in peers if peer is not None]
        logger.info(
            "KVPP MemFabric MTE initialized: rank=%d, gva=%#x, "
            "staging_bytes=%d, shm_id=%d",
            self.group.rank_in_group,
            self._local_metadata.staging_addr,
            staging_bytes,
            shm_id,
        )

    def _local_page_batch(
        self, layer_name: str, pages: tuple[int, ...]
    ) -> KVPPTransferBatch:
        buffers = self._layers[layer_name]
        return build_kvpp_page_transfer_batch(pages, buffers, buffers)

    def _launch(
        self,
        layer_name: str,
        batch: KVPPTransferBatch,
        *,
        source_rank: int = -1,
        destination_rank: int = -1,
    ) -> None:
        assert self._copy_op is not None
        descriptor_count = len(batch.lengths)
        self._copy_op(
            self._anchors[layer_name],
            torch.tensor(batch.source_addrs, dtype=torch.int64),
            torch.tensor(batch.destination_addrs, dtype=torch.int64),
            torch.tensor(batch.lengths, dtype=torch.int64),
            torch.full(
                (descriptor_count,), source_rank, dtype=torch.int32
            ),
            torch.full(
                (descriptor_count,), destination_rank, dtype=torch.int32
            ),
            self._shm_id,
        )

    def push_active_pages(
        self, layer_name: str, pages: tuple[int, ...], stream: Any
    ) -> KVPPCompletion:
        owner_rank = self.layer_owners[layer_name]
        if owner_rank != self.group.rank_in_group or not pages:
            return MemFabricMTECompletion.record(stream)
        if self._local_metadata is None or not self._peer_metadata:
            raise RuntimeError("KVPP MTE transport was not initialized.")

        local_batch = self._local_page_batch(layer_name, pages)
        for peer_rank, peer in enumerate(self._peer_metadata):
            if peer_rank == owner_rank:
                continue
            staged = _compact_staging_batch(
                local_batch, peer.staging_addr, peer.staging_bytes
            )
            self._launch(
                layer_name, staged, destination_rank=peer.rank
            )
        return MemFabricMTECompletion.record(stream)

    def receive_active_pages(
        self, layer_name: str, pages: tuple[int, ...], stream: Any
    ) -> KVPPCompletion:
        owner_rank = self.layer_owners[layer_name]
        if owner_rank == self.group.rank_in_group or not pages:
            return MemFabricMTECompletion.record(stream)
        if self._local_metadata is None:
            raise RuntimeError("KVPP MTE transport was not initialized.")

        local_batch = self._local_page_batch(layer_name, pages)
        staged = _compact_staging_batch(
            local_batch,
            self._local_metadata.staging_addr,
            self._local_metadata.staging_bytes,
        )
        unpack = KVPPTransferBatch(
            source_addrs=staged.destination_addrs,
            destination_addrs=local_batch.destination_addrs,
            lengths=local_batch.lengths,
        )
        self._launch(
            layer_name,
            unpack,
            source_rank=self._local_metadata.rank,
        )
        return MemFabricMTECompletion.record(stream)

    def close(self) -> None:
        if self._memory is not None:
            self._memory.destroy()
            self._memory = None
        self._peer_metadata.clear()
        self._local_metadata = None
        self._layers.clear()
        self._anchors.clear()
