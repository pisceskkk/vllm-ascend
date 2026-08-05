# SPDX-License-Identifier: Apache-2.0
"""Transport boundary for KV layer parallelism.

KVPP scheduling owns page selection, layer ordering, scratch lifetime, and
completion synchronization.  A transport backend owns only initialization and
the stream-ordered movement of selected physical pages.  Keeping that boundary
small allows SDMA and MTE implementations to be compared without changing the
model execution path.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import logger


KVPPTransportFactory = Callable[
    [GroupCoordinator, dict[str, int], int], "KVPPTransport"
]


@dataclass(frozen=True)
class KVPPBufferMetadata:
    """Address and logical-page layout for one KV cache tensor."""

    base_addr: int
    block_stride_bytes: int
    block_bytes: int


@dataclass(frozen=True)
class KVPPTransferBatch:
    """One peer's backend-neutral scatter/gather transfer descriptors."""

    source_addrs: tuple[int, ...]
    destination_addrs: tuple[int, ...]
    lengths: tuple[int, ...]


# Compatibility name used by the first transport-boundary prototype.
KVPPPageTransferBatch = KVPPTransferBatch


@runtime_checkable
class KVPPCompletion(Protocol):
    """Completion of one owner-to-peer active-page push.

    ``wait`` must not return until the destination pages are remotely visible.
    ``wait_on_stream`` expresses a local device dependency when the backend can
    do so without blocking the host.  Cross-rank notification remains the
    responsibility of the common KVPP execution layer.
    """

    def wait(self) -> None:
        """Block the caller until all remote destinations are visible."""

    def wait_on_stream(self, stream: Any) -> None:
        """Order a local device stream after this transfer."""


def flatten_kvpp_cache(cache: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(cache, torch.Tensor):
        return (cache,)
    if isinstance(cache, (tuple, list)):
        tensors = tuple(cache)
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("KVPP cache tuples may contain only tensors.")
        return tensors
    raise TypeError(f"Unsupported KVPP cache type: {type(cache)!r}.")


def build_kvpp_layer_metadata(
    kv_caches: dict[str, Any], num_blocks: int
) -> dict[str, tuple[KVPPBufferMetadata, ...]]:
    """Describe logical pages once for use by SDMA and MTE backends."""
    layers: dict[str, tuple[KVPPBufferMetadata, ...]] = {}
    for layer_name, cache in kv_caches.items():
        buffers: list[KVPPBufferMetadata] = []
        for tensor in flatten_kvpp_cache(cache):
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


def build_kvpp_page_transfer_batch(
    pages: tuple[int, ...],
    source_buffers: tuple[KVPPBufferMetadata, ...],
    destination_buffers: tuple[KVPPBufferMetadata, ...],
) -> KVPPTransferBatch:
    """Build identical physical-page descriptors for every data plane."""
    if len(source_buffers) != len(destination_buffers):
        raise RuntimeError(
            "KVPP owner and destination cache tensor counts differ: "
            f"owner={len(source_buffers)}, "
            f"destination={len(destination_buffers)}."
        )

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

        # Coalesce adjacent pages only if neither address space has padding.
        can_coalesce = (
            source.block_stride_bytes == source.block_bytes
            and destination.block_stride_bytes == destination.block_bytes
        )
        run_start = 0
        while run_start < len(pages):
            run_end = run_start + 1
            if can_coalesce:
                while (
                    run_end < len(pages)
                    and pages[run_end] == pages[run_end - 1] + 1
                ):
                    run_end += 1
            page = pages[run_start]
            source_addrs.append(
                source.base_addr + page * source.block_stride_bytes
            )
            destination_addrs.append(
                destination.base_addr + page * destination.block_stride_bytes
            )
            lengths.append((run_end - run_start) * source.block_bytes)
            run_start = run_end

    return KVPPTransferBatch(
        tuple(source_addrs), tuple(destination_addrs), tuple(lengths)
    )


@runtime_checkable
class KVPPTransport(Protocol):
    """Data-plane contract consumed by :class:`KVPPContext`."""

    def initialize(self, kv_caches: dict[str, Any]) -> None:
        """Prepare transport resources for all persistent and scratch caches."""

    def push_active_pages(
        self,
        layer_name: str,
        pages: tuple[int, ...],
        stream: Any,
    ) -> KVPPCompletion:
        """Enqueue selected pages and return remote-visible completion."""

    def close(self) -> None:
        """Release backend-owned sessions and memory metadata."""


_TRANSPORT_FACTORIES: dict[str, KVPPTransportFactory] = {}


def register_kvpp_transport(
    name: str,
    factory: KVPPTransportFactory,
) -> None:
    """Register an out-of-tree or optional KVPP transport implementation."""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("KVPP transport name must not be empty.")
    _TRANSPORT_FACTORIES[normalized] = factory


def _load_transport_factory(path: str) -> KVPPTransportFactory:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            "ASCEND_KVPP_TRANSPORT_CLASS must use 'module:attribute' syntax, "
            f"got {path!r}."
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError(f"KVPP transport factory {path!r} is not callable.")
    return factory


def _sdma_factory(
    group: GroupCoordinator,
    layer_owners: dict[str, int],
    num_blocks: int,
) -> KVPPTransport:
    # Import lazily so importing KVPP metadata does not require MemFabric.
    from vllm_ascend.distributed.kv_transfer.kv_pool.memfabric_sdma_transport import (
        MemFabricSDMAKVPPTransport,
    )

    return MemFabricSDMAKVPPTransport(group, layer_owners, num_blocks)


def create_kvpp_transport(
    group: GroupCoordinator,
    layer_owners: dict[str, int],
    num_blocks: int,
    backend: str | None = None,
) -> KVPPTransport:
    """Create the selected KVPP data plane.

    ``sdma`` is built in.  An MTE implementation can either call
    :func:`register_kvpp_transport` during optional-module initialization or be
    selected explicitly with ``ASCEND_KVPP_TRANSPORT_CLASS=module:attribute``.
    The latter keeps experimental MTE operator bindings outside the scheduler.
    """
    name = (
        backend
        if backend is not None
        else os.getenv("ASCEND_KVPP_TRANSPORT", "sdma")
    ).strip().lower()

    class_path = os.getenv("ASCEND_KVPP_TRANSPORT_CLASS")
    if class_path:
        factory = _load_transport_factory(class_path)
    elif name == "sdma":
        factory = _sdma_factory
    else:
        factory = _TRANSPORT_FACTORIES.get(name)
        if factory is None:
            raise RuntimeError(
                f"KVPP transport {name!r} is not available. Install/register "
                "the optional backend or set ASCEND_KVPP_TRANSPORT_CLASS to "
                "its 'module:attribute' factory."
            )

    transport = factory(group, layer_owners, num_blocks)
    if not isinstance(transport, KVPPTransport):
        raise TypeError(
            f"KVPP transport {name!r} must implement initialize(), "
            "push_active_pages(), and close()."
        )
    logger.info(
        "KVPP transport selected: backend=%s, implementation=%s.%s",
        name,
        type(transport).__module__,
        type(transport).__qualname__,
    )
    return transport
