# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports for the renamed KVPP MemFabric SDMA backend."""

from vllm_ascend.distributed.kv_transfer.kv_pool.kvpp_transport import (
    KVPPBufferMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.memfabric_sdma_transport import (
    KVPPPeerMetadata,
    MemFabricKVPPTransport,
    MemFabricSDMACompletion,
    MemFabricSDMAKVPPTransport,
)

__all__ = [
    "KVPPBufferMetadata",
    "KVPPPeerMetadata",
    "MemFabricKVPPTransport",
    "MemFabricSDMACompletion",
    "MemFabricSDMAKVPPTransport",
]
