from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator

from vllm_ascend.distributed.kv_transfer.kv_pool.memfabric_transport import (
    MemFabricKVPPTransport,
)


def _active_pages(
    block_table: torch.Tensor,
    seq_lens: Any,
    block_size: int,
    num_blocks: int,
) -> tuple[int, ...]:
    """Return sorted physical pages read by the current batch.

    MemFabric needs host-side address descriptors. Reuse the physical block
    table already gathered for attention and materialize only the columns
    covered by this batch.
    """
    lengths = seq_lens.tolist()
    max_pages = max(
        ((int(seq_len) + block_size - 1) // block_size for seq_len in lengths),
        default=0,
    )
    table = block_table[: len(lengths), :max_pages].tolist()
    pages: set[int] = set()
    for row, seq_len in zip(table, lengths):
        pages_in_request = (int(seq_len) + block_size - 1) // block_size
        for page in row[:pages_in_request]:
            page = int(page)
            if 0 <= page < num_blocks:
                pages.add(page)
    return tuple(sorted(pages))


@dataclass
class KVPPContext:
    """Layer ownership and stream-ordered active-page transport.

    Owned layers use persistent KV caches. Non-owned layers are already bound
    by vLLM's planner to one full-size shared scratch cache. Active pages are
    pushed into the same physical block IDs, preserving the original block
    table and slot mapping.
    """

    group: GroupCoordinator
    layer_owners: dict[str, int]
    num_blocks: int
    block_size: int
    transport: Any | None = None
    _selected_pages: tuple[int, ...] | None = None
    _comm_stream: Any | None = None
    _transfer_event: Any | None = None
    _transfer_submitted: bool = False
    _current_layer: str | None = None

    def initialize_transport(self, kv_caches: dict[str, Any]) -> None:
        if self.transport is None:
            self.transport = MemFabricKVPPTransport(
                self.group, self.layer_owners, self.num_blocks
            )
        self.transport.initialize(kv_caches)

    def prepare_batch(
        self,
        block_table: torch.Tensor,
        seq_lens: Any,
    ) -> None:
        self._selected_pages = _active_pages(
            block_table, seq_lens, self.block_size, self.num_blocks
        )

    def begin_layer(
        self, layer_name: str, kv_cache: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Start owner page pushes while Q/KV projection runs on compute."""
        if self._selected_pages is None:
            raise RuntimeError("KVPP batch metadata was not prepared before forward.")
        if self.transport is None:
            raise RuntimeError("KVPP MemFabric transport was not initialized.")
        if self._current_layer is not None:
            raise RuntimeError(
                f"KVPP layer {self._current_layer} was not completed before {layer_name}."
            )

        self._current_layer = layer_name
        self._transfer_event = None
        owner_rank = self.layer_owners[layer_name]
        self._transfer_submitted = bool(
            self._selected_pages and self.group.world_size > 1
        )
        if not self._transfer_submitted:
            return kv_cache

        # Every non-owned layer aliases one scratch cache. Before another rank
        # writes into it, every consumer must finish the preceding layer. This
        # blocking rendezvous is the initial correctness implementation; a
        # remote completion primitive can replace it when overlap is optimized.
        current_stream = torch.npu.current_stream()
        current_stream.synchronize()
        dist.barrier(group=self.group.cpu_group)

        if owner_rank == self.group.rank_in_group:
            if self._comm_stream is None:
                self._comm_stream = torch.npu.Stream()
            self._transfer_event = torch.npu.Event()
            with torch.npu.stream(self._comm_stream):
                self.transport.push_active_pages(
                    layer_name, self._selected_pages, self._comm_stream
                )
                self._transfer_event.record(self._comm_stream)
        return kv_cache

    def wait_for_current_layer(self, layer_name: str) -> None:
        """Order current-token KV writes after all owner page pushes."""
        if self._current_layer != layer_name:
            raise RuntimeError(f"No pending KVPP transfer for layer {layer_name}.")
        if self._transfer_submitted:
            if self._transfer_event is not None:
                # A device-side wait is insufficient before the CPU group
                # rendezvous: the owner must publish remote completion first.
                self._transfer_event.synchronize()
            dist.barrier(group=self.group.cpu_group)
        self._transfer_submitted = False
        self._transfer_event = None
        self._current_layer = None
