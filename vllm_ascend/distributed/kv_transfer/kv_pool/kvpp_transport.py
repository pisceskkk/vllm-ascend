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
class KVPPActivePages:
    """Fixed-shape device representation of active physical KV pages.

    ``page_ids`` and ``valid_mask`` stay on the same device as vLLM's original
    block table.  Duplicate and invalid table entries remain as masked slots,
    so transports never need to read a dynamic page count back to the host.
    """

    page_ids: torch.Tensor
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.page_ids.device != self.valid_mask.device:
            raise ValueError("KVPP active page tensors must share one device.")
        if self.page_ids.dim() != 1 or self.valid_mask.dim() != 1:
            raise ValueError("KVPP active page tensors must be one-dimensional.")
        if self.page_ids.numel() != self.valid_mask.numel():
            raise ValueError("KVPP active page tensor lengths must match.")
        if self.page_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("KVPP page_ids must use int32 or int64.")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("KVPP valid_mask must use bool.")


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


@runtime_checkable
class KVPPTransport(Protocol):
    """Data-plane contract consumed by :class:`KVPPContext`."""

    def initialize(self, kv_caches: dict[str, Any]) -> None:
        """Prepare transport resources for all persistent and scratch caches."""

    def push_active_pages(
        self,
        layer_name: str,
        pages: KVPPActivePages,
        stream: Any,
    ) -> KVPPCompletion:
        """Enqueue selected pages and return remote-visible completion."""

    def receive_active_pages(
        self,
        layer_name: str,
        pages: KVPPActivePages,
        stream: Any,
    ) -> KVPPCompletion:
        """Materialize remotely pushed pages in the local scratch cache.

        Direct transports such as SDMA return an already-complete local event.
        Staged transports use this hook to unpack their backend-owned receive
        buffer into the original physical page IDs.
        """

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


def _mte_factory(
    group: GroupCoordinator,
    layer_owners: dict[str, int],
    num_blocks: int,
) -> KVPPTransport:
    from vllm_ascend.distributed.kv_transfer.kv_pool.memfabric_mte_transport import (
        MemFabricMTEKVPPTransport,
    )

    return MemFabricMTEKVPPTransport(group, layer_owners, num_blocks)


def create_kvpp_transport(
    group: GroupCoordinator,
    layer_owners: dict[str, int],
    num_blocks: int,
    backend: str | None = None,
) -> KVPPTransport:
    """Create the selected KVPP data plane.

    ``sdma`` and ``mte`` are built in. Out-of-tree implementations can call
    :func:`register_kvpp_transport` or use
    ``ASCEND_KVPP_TRANSPORT_CLASS=module:attribute``.
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
    elif name == "mte":
        factory = _mte_factory
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
            "push_active_pages(), receive_active_pages(), and close()."
        )
    logger.info(
        "KVPP transport selected: backend=%s, implementation=%s.%s",
        name,
        type(transport).__module__,
        type(transport).__qualname__,
    )
    return transport
