import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator

from vllm_ascend.distributed.kv_transfer.kv_pool.kvpp_transport import (
    KVPPCompletion,
    KVPPTransport,
    create_kvpp_transport,
)


def _active_pages(
    block_table: torch.Tensor,
    seq_lens: Any,
    block_size: int,
    num_blocks: int,
) -> tuple[int, ...]:
    """Return sorted physical pages read by the current batch.

    Descriptor-based transports need the active physical pages. Reuse the
    block table already gathered for attention and materialize only the
    columns covered by this batch.
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
    transport: KVPPTransport | None = None
    overlap_mode: str | None = None
    _selected_pages: tuple[int, ...] | None = None
    _comm_stream: Any | None = None
    _transfer_completion: KVPPCompletion | None = None
    _transfer_submitted: bool = False
    _current_layer: str | None = None
    _ordered_layers: tuple[str, ...] = field(init=False)
    _layer_indices: dict[str, int] = field(init=False)
    _executor: ThreadPoolExecutor | None = field(
        default=None, init=False, repr=False
    )
    _transfer_future: Future[None] | None = field(
        default=None, init=False, repr=False
    )
    _pending_layer: str | None = field(default=None, init=False)
    _transfer_waited: bool = field(default=False, init=False)
    _device_id: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.overlap_mode is None:
            self.overlap_mode = os.getenv(
                "ASCEND_KVPP_OVERLAP_MODE", "baseline"
            ).lower()
        valid_modes = {"baseline", "current_layer", "previous_layer"}
        if self.overlap_mode not in valid_modes:
            raise ValueError(
                "ASCEND_KVPP_OVERLAP_MODE must be one of "
                f"{sorted(valid_modes)}, got {self.overlap_mode!r}."
            )
        self._ordered_layers = tuple(self.layer_owners)
        self._layer_indices = {
            layer_name: index
            for index, layer_name in enumerate(self._ordered_layers)
        }

    def initialize_transport(self, kv_caches: dict[str, Any]) -> None:
        if self.transport is None:
            self.transport = create_kvpp_transport(
                self.group, self.layer_owners, self.num_blocks
            )
        self.transport.initialize(kv_caches)
        if self.overlap_mode != "baseline" and self.group.world_size > 1:
            self._device_id = torch.npu.current_device()
            self._comm_stream = torch.npu.Stream()
            # One transfer may be in flight. Serializing jobs also preserves
            # point-to-point notification order when layer ownership changes.
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="kvpp-transport"
            )

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
            raise RuntimeError("KVPP transport was not initialized.")
        if self._current_layer is not None:
            raise RuntimeError(
                f"KVPP layer {self._current_layer} was not completed before {layer_name}."
            )

        if self.overlap_mode != "baseline":
            self._current_layer = layer_name
            self._transfer_waited = False
            if self._pending_layer is None:
                self._start_overlap_transfer(layer_name)
            elif self._pending_layer != layer_name:
                raise RuntimeError(
                    f"KVPP prefetched {self._pending_layer}, but forward entered "
                    f"{layer_name}."
                )
            return kv_cache

        self._current_layer = layer_name
        self._transfer_completion = None
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
        layer_index = self._layer_indices[layer_name]
        with torch.profiler.record_function(
            f"kvpp.baseline.scratch_ready.layer_{layer_index}"
        ):
            current_stream.synchronize()
            dist.barrier(group=self.group.cpu_group)

        if self._comm_stream is None:
            self._comm_stream = torch.npu.Stream()
        if owner_rank == self.group.rank_in_group:
            with torch.npu.stream(self._comm_stream):
                self._transfer_completion = self.transport.push_active_pages(
                    layer_name, self._selected_pages, self._comm_stream
                )
        return kv_cache

    def wait_for_current_layer(self, layer_name: str) -> None:
        """Order current-token KV writes after all owner page pushes."""
        if self._current_layer != layer_name:
            raise RuntimeError(f"No pending KVPP transfer for layer {layer_name}.")
        if self.overlap_mode != "baseline":
            if self._pending_layer != layer_name:
                raise RuntimeError(
                    f"No pending KVPP overlap transfer for layer {layer_name}."
                )
            if self._transfer_future is not None:
                # This blocks only for the residual transfer time. In
                # current_layer mode Q/KV projection is already enqueued; in
                # previous_layer mode the transfer starts before o_proj/MoE.
                layer_index = self._layer_indices[layer_name]
                with torch.profiler.record_function(
                    f"kvpp.wait.{self.overlap_mode}.layer_{layer_index}"
                ):
                    self._transfer_future.result()
            self._transfer_waited = True
            return
        if self._transfer_submitted:
            layer_index = self._layer_indices[layer_name]
            with torch.profiler.record_function(
                f"kvpp.baseline.transfer_wait.layer_{layer_index}"
            ):
                if self._transfer_completion is not None:
                    # A local device dependency is insufficient before the CPU
                    # rendezvous: the backend must confirm remote visibility.
                    self._transfer_completion.wait()
                dist.barrier(group=self.group.cpu_group)
                assert self._comm_stream is not None
                assert self._selected_pages is not None
                with torch.npu.stream(self._comm_stream):
                    receive_completion = self.transport.receive_active_pages(
                        layer_name, self._selected_pages, self._comm_stream
                    )
                # SDMA records an immediate event because its push targets the
                # final scratch addresses. MTE waits here for receive-side
                # staging unpack before this rank enters attention.
                receive_completion.wait()
        self._transfer_submitted = False
        self._transfer_completion = None
        self._current_layer = None

    def finish_layer_attention(self, layer_name: str) -> None:
        """Release this layer's scratch and optionally prefetch the next one.

        The call site is after all attention kernels that consume historical
        KV have been submitted, but before o_proj and the layer MLP/MoE. A
        device event recorded here is therefore the earliest safe signal that
        every peer may overwrite the shared scratch cache.
        """
        if self.overlap_mode == "baseline":
            return
        if self._current_layer != layer_name or not self._transfer_waited:
            raise RuntimeError(
                f"KVPP attention for layer {layer_name} finished before its "
                "transfer was consumed."
            )

        self._current_layer = None
        self._pending_layer = None
        self._transfer_future = None
        self._transfer_waited = False

        if self.overlap_mode != "previous_layer":
            return
        layer_index = self._layer_indices[layer_name]
        next_index = layer_index + 1
        if next_index < len(self._ordered_layers):
            self._start_overlap_transfer(self._ordered_layers[next_index])

    def _start_overlap_transfer(self, layer_name: str) -> None:
        if self._pending_layer is not None:
            raise RuntimeError(
                f"KVPP transfer for {self._pending_layer} is still pending."
            )
        self._pending_layer = layer_name
        self._transfer_future = None
        if not self._selected_pages or self.group.world_size <= 1:
            return
        if self._executor is None or self._comm_stream is None:
            raise RuntimeError("KVPP overlap worker was not initialized.")

        # All ranks publish a local safe point. The owner does not write a
        # peer's scratch until that peer reports this event complete.
        scratch_ready = torch.npu.Event()
        scratch_ready.record(torch.npu.current_stream())
        pages = self._selected_pages
        self._transfer_future = self._executor.submit(
            self._run_overlap_transfer, layer_name, pages, scratch_ready
        )

    def _run_overlap_transfer(
        self,
        layer_name: str,
        pages: tuple[int, ...],
        scratch_ready: Any,
    ) -> None:
        """Run safe-point and completion notification off the compute thread."""
        if self.transport is None or self._comm_stream is None:
            raise RuntimeError("KVPP overlap transport is not initialized.")
        if self._device_id is not None:
            torch.npu.set_device(self._device_id)

        owner_rank = self.layer_owners[layer_name]
        local_rank = self.group.rank_in_group
        owner_global_rank = self.group.ranks[owner_rank]
        layer_index = self._layer_indices[layer_name]
        ready_tag = 0x4B560000 + layer_index * 2
        done_tag = ready_tag + 1
        token = torch.ones(1, dtype=torch.uint8, device="cpu")

        with torch.profiler.record_function(
            f"kvpp.comm_total.layer_{layer_index}"
        ):
            with torch.profiler.record_function(
                f"kvpp.scratch_ready.layer_{layer_index}"
            ):
                scratch_ready.synchronize()

            if local_rank != owner_rank:
                with torch.profiler.record_function(
                    f"kvpp.ready_send.layer_{layer_index}"
                ):
                    dist.send(
                        token,
                        dst=owner_global_rank,
                        group=self.group.cpu_group,
                        tag=ready_tag,
                    )
                with torch.profiler.record_function(
                    f"kvpp.done_recv.layer_{layer_index}"
                ):
                    dist.recv(
                        token,
                        src=owner_global_rank,
                        group=self.group.cpu_group,
                        tag=done_tag,
                    )
                with torch.profiler.record_function(
                    f"kvpp.transport_receive.layer_{layer_index}"
                ):
                    with torch.npu.stream(self._comm_stream):
                        receive_completion = (
                            self.transport.receive_active_pages(
                                layer_name, pages, self._comm_stream
                            )
                        )
                    receive_completion.wait()
                return

            with torch.profiler.record_function(
                f"kvpp.ready_recv.layer_{layer_index}"
            ):
                for peer_rank, peer_global_rank in enumerate(self.group.ranks):
                    if peer_rank == owner_rank:
                        continue
                    dist.recv(
                        token,
                        src=peer_global_rank,
                        group=self.group.cpu_group,
                        tag=ready_tag,
                    )

            with torch.profiler.record_function(
                f"kvpp.transport_push.layer_{layer_index}"
            ):
                with torch.npu.stream(self._comm_stream):
                    completion = self.transport.push_active_pages(
                        layer_name, pages, self._comm_stream
                    )
                # Only the communication worker waits on the host. The compute
                # thread continues until this layer first writes/reads its
                # paged KV cache.
                completion.wait()

            with torch.profiler.record_function(
                f"kvpp.done_send.layer_{layer_index}"
            ):
                for peer_rank, peer_global_rank in enumerate(self.group.ranks):
                    if peer_rank == owner_rank:
                        continue
                    dist.send(
                        token,
                        dst=peer_global_rank,
                        group=self.group.cpu_group,
                        tag=done_tag,
                    )

    def close(self) -> None:
        """Drain overlap work and release transport-owned resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self.transport is not None:
            self.transport.close()
            self.transport = None
