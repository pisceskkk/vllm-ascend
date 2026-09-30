# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-address local lookup inputs and per-graph external frontiers."""

import torch


class EngramGraphInputs:
    def __init__(self, tables, capacity, device, *, full_rows=False):
        self.full_rows = full_rows
        self.rows = {
            layer: torch.zeros(
                (capacity, table.n_hash_cols * table.dim)
                if full_rows
                else (capacity, table.part_n_hash_cols, table.dim),
                dtype=torch.bfloat16,
                device=device,
            )
            for layer, table in tables.items()
        }
        self.mask = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.events = {}

    def bindings(self, batch_descriptor, *, prime=False):
        # Each captured graph owns its wait/reset tasks. Sharing an external
        # event between graph descriptors could reset another graph's record.
        if batch_descriptor not in self.events:
            self.events[batch_descriptor] = (
                torch.npu.ExternalEvent(),
                {layer: torch.npu.ExternalEvent() for layer in self.rows},
            )
        mask_ready, ready = self.events[batch_descriptor]
        if prime:
            # Dummy capture consumes initialized zero buffers. Hashing, host
            # routing and event records remain outside the captured graph.
            stream = torch.npu.current_stream()
            mask_ready.record(stream)
            for event in ready.values():
                event.record(stream)
        return {
            "engram_lookups" if self.full_rows else "engram_local_rows": self.rows,
            "engram_mask": self.mask,
            "engram_mask_ready_event": mask_ready,
            "engram_ready_events": ready,
            "engram_graph_events": True,
        }


def wait_engram_event(event, external):
    stream = torch.npu.current_stream()
    if external:
        # Capture wait/reset device tasks, rather than freezing an ordinary
        # Event's capture-time dependency. Replay consumes this step's record.
        event.wait(stream)
        event.reset(stream)
    else:
        stream.wait_event(event)
