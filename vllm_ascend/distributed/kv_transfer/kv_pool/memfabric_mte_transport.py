# SPDX-License-Identifier: Apache-2.0
"""MemFabric Hybrid MTE backend for KV layer parallelism.

MemFabric Hybrid 1.2 cannot export an existing KV tensor as a remote MTE GVA.
This backend therefore allocates one bounded symmetric active-page staging
segment per rank. Layer owners copy selected persistent pages directly to each
consumer's segment with one batched AscendC GM->UB->remote-GM launch. Consumers
unpack the same device-resident page descriptors into their existing scratch
cache before attention. No full layer cache is copied or staged.
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
    KVPPActivePages,
    KVPPBufferMetadata,
    KVPPCompletion,
    build_kvpp_layer_metadata,
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
    resources: tuple[Any, ...] = ()

    @classmethod
    def record(
        cls, stream: Any, resources: tuple[Any, ...] = ()
    ) -> "MemFabricMTECompletion":
        event = torch.npu.Event()
        event.record(stream)
        return cls(event, resources)

    def wait(self) -> None:
        self.event.synchronize()

    def wait_on_stream(self, stream: Any) -> None:
        stream.wait_event(self.event)


@dataclass(frozen=True)
class _MTEDeviceBufferMetadata:
    local_base_offsets: torch.Tensor
    block_strides: torch.Tensor
    block_bytes: torch.Tensor
    staging_bytes_per_slot: int


@dataclass(frozen=True)
class _MTEDeviceDescriptors:
    local_offsets: torch.Tensor
    staging_offsets: torch.Tensor
    lengths: torch.Tensor
    staging_base: int

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.local_offsets, self.staging_offsets, self.lengths)


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
        self._memfabric_module: Any | None = None
        self._global_initialized = False
        self._copy_op = copy_op
        self._memory: Any | None = None
        self._local_metadata: KVPPMTEPeerMetadata | None = None
        self._peer_metadata: list[KVPPMTEPeerMetadata] = []
        self._layers: dict[str, tuple[KVPPBufferMetadata, ...]] = {}
        self._anchors: dict[str, torch.Tensor] = {}
        self._device_layers: dict[str, _MTEDeviceBufferMetadata] = {}
        self._shm_id = _DEFAULT_SHM_ID

    def initialize(self, kv_caches: dict[str, Any]) -> None:
        if not os.getenv("MEMFABRIC_HYBRID_HOME_PATH"):
            raise RuntimeError(
                "KVPP MTE requires the MemFabric Hybrid environment. Source "
                "/usr/local/memfabric_hybrid/set_env.sh before launching vLLM."
            )
        if self._shm_module is None:
            try:
                import memfabric_hybrid  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "KVPP MTE requires memfabric_hybrid.shm."
                ) from exc
            self._memfabric_module = memfabric_hybrid
            self._shm_module = memfabric_hybrid.shm
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
        if self._memfabric_module is not None:
            ret = self._memfabric_module.initialize(0)
            if ret != 0:
                raise RuntimeError(
                    "KVPP MemFabric global initialization failed: "
                    f"error={ret}."
                )
            self._global_initialized = True
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
        # ``create`` publishes the symmetric address before every rank's
        # device-side mapping is necessarily ready. The SHM barrier completes
        # that setup; a process-group barrier is not an equivalent substitute.
        self._memory.barrier()

        self._layers = build_kvpp_layer_metadata(kv_caches, self.num_blocks)
        self._anchors = {
            layer_name: flatten_kvpp_cache(cache)[0]
            for layer_name, cache in kv_caches.items()
        }
        self._device_layers = {}
        for layer_name, buffers in self._layers.items():
            device = self._anchors[layer_name].device
            anchor_base = self._anchors[layer_name].data_ptr()
            self._device_layers[layer_name] = _MTEDeviceBufferMetadata(
                local_base_offsets=torch.tensor(
                    [buffer.base_addr - anchor_base for buffer in buffers],
                    dtype=torch.int64,
                    device=device,
                ),
                block_strides=torch.tensor(
                    [buffer.block_stride_bytes for buffer in buffers],
                    dtype=torch.int64,
                    device=device,
                ),
                block_bytes=torch.tensor(
                    [buffer.block_bytes for buffer in buffers],
                    dtype=torch.int64,
                    device=device,
                ),
                staging_bytes_per_slot=sum(
                    buffer.block_bytes for buffer in buffers
                ),
            )
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

    def _local_descriptors(
        self, layer_name: str, pages: KVPPActivePages
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metadata = self._device_layers[layer_name]
        page_ids = pages.page_ids.to(dtype=torch.int64)
        offsets = (
            metadata.local_base_offsets[:, None]
            + page_ids[None, :] * metadata.block_strides[:, None]
        )
        lengths = torch.where(
            pages.valid_mask[None, :],
            metadata.block_bytes[:, None],
            torch.zeros((), dtype=torch.int64, device=page_ids.device),
        )
        return offsets.flatten(), lengths.flatten()

    def _staging_offsets(
        self,
        layer_name: str,
        pages: KVPPActivePages,
        staging_bytes: int,
    ) -> torch.Tensor:
        metadata = self._device_layers[layer_name]
        max_active_pages = staging_bytes // metadata.staging_bytes_per_slot
        if max_active_pages == 0:
            raise RuntimeError(
                "KVPP MTE staging segment cannot hold one page across every "
                f"cache buffer: page_bytes={metadata.staging_bytes_per_slot}, "
                f"capacity={staging_bytes}."
            )
        active_ordinals = torch.cumsum(
            pages.valid_mask.to(dtype=torch.int64), dim=0
        ) - 1
        torch._assert_async(
            pages.valid_mask.sum() <= max_active_pages,
            "KVPP MTE active pages exceed ASCEND_KVPP_MTE_STAGING_BYTES",
        )
        per_buffer_capacity = metadata.block_bytes * max_active_pages
        buffer_offsets = torch.cumsum(per_buffer_capacity, dim=0)
        buffer_offsets = buffer_offsets - per_buffer_capacity
        offsets = (
            buffer_offsets[:, None]
            + active_ordinals[None, :] * metadata.block_bytes[:, None]
        )
        return offsets.flatten()

    def _launch(
        self,
        layer_name: str,
        descriptors: _MTEDeviceDescriptors,
        *,
        source_rank: int = -1,
        destination_rank: int = -1,
    ) -> None:
        assert self._copy_op is not None
        self._copy_op(
            self._anchors[layer_name],
            descriptors.local_offsets,
            descriptors.staging_offsets,
            descriptors.lengths,
            descriptors.staging_base,
            source_rank,
            destination_rank,
            self._shm_id,
        )

    def push_active_pages(
        self, layer_name: str, pages: KVPPActivePages, stream: Any
    ) -> KVPPCompletion:
        owner_rank = self.layer_owners[layer_name]
        if owner_rank != self.group.rank_in_group:
            return MemFabricMTECompletion.record(stream)
        if self._local_metadata is None or not self._peer_metadata:
            raise RuntimeError("KVPP MTE transport was not initialized.")

        local_offsets, lengths = self._local_descriptors(layer_name, pages)
        retained: list[torch.Tensor] = [local_offsets, lengths]
        for peer_rank, peer in enumerate(self._peer_metadata):
            if peer_rank == owner_rank:
                continue
            staging_offsets = self._staging_offsets(
                layer_name,
                pages,
                peer.staging_bytes,
            )
            descriptors = _MTEDeviceDescriptors(
                local_offsets, staging_offsets, lengths, peer.staging_addr
            )
            self._launch(
                layer_name, descriptors, destination_rank=peer.rank
            )
            retained.append(staging_offsets)
        return MemFabricMTECompletion.record(stream, tuple(retained))

    def receive_active_pages(
        self, layer_name: str, pages: KVPPActivePages, stream: Any
    ) -> KVPPCompletion:
        owner_rank = self.layer_owners[layer_name]
        if owner_rank == self.group.rank_in_group:
            return MemFabricMTECompletion.record(stream)
        if self._local_metadata is None:
            raise RuntimeError("KVPP MTE transport was not initialized.")

        local_offsets, lengths = self._local_descriptors(layer_name, pages)
        staging_offsets = self._staging_offsets(
            layer_name,
            pages,
            self._local_metadata.staging_bytes,
        )
        descriptors = _MTEDeviceDescriptors(
            local_offsets=local_offsets,
            staging_offsets=staging_offsets,
            lengths=lengths,
            staging_base=self._local_metadata.staging_addr,
        )
        self._launch(
            layer_name,
            descriptors,
            source_rank=self._local_metadata.rank,
        )
        return MemFabricMTECompletion.record(stream, descriptors.tensors())

    def close(self) -> None:
        if self._memory is not None:
            self._memory.destroy()
            self._memory = None
        if self._global_initialized:
            self._shm_module.uninitialize()
            assert self._memfabric_module is not None
            self._memfabric_module.uninitialize()
            self._global_initialized = False
        self._peer_metadata.clear()
        self._local_metadata = None
        self._layers.clear()
        self._anchors.clear()
        self._device_layers.clear()
