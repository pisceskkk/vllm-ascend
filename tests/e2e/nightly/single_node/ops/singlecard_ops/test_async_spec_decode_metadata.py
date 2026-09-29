# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch
import torch_npu
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.spec_decode.utils import update_num_computed_tokens_for_batch_change
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.mark.parametrize("graph_mode", [False, True])
def test_async_accepted_counts_stay_on_device(graph_mode, tmp_path):
    device = torch.device("npu")
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.need_accepted_tokens = False
    runner.num_accepted_tokens_event = torch.npu.Event()
    runner.num_accepted_tokens = CpuGpuBuffer(8, dtype=torch.int32, device=device, pin_memory=True)
    runner.input_batch = SimpleNamespace()
    computed = torch.empty(4, dtype=torch.int32, device=device)
    prev_positions = torch.tensor([2, -1, 0, 1], dtype=torch.int32, device=device)
    prev_drafts = torch.tensor([5, 0, 5, 0], dtype=torch.int32, device=device)
    valid_counts = torch.empty(4, dtype=torch.int64, device=device)
    optimistic_computed = torch.tensor([36, 0, 16, 21], dtype=torch.int32, device=device)

    def correct():
        update_num_computed_tokens_for_batch_change(
            computed,
            runner.num_accepted_tokens.gpu[:4],
            prev_positions,
            valid_counts,
            prev_drafts,
            optimistic_computed,
        )

    graph = None
    if graph_mode:
        computed.copy_(torch.tensor([10, 20, 30, 0], dtype=torch.int32, device=device))
        valid_counts.fill_(1)
        runner._prepare_num_accepted_tokens(4, has_prev_mapping=True)
        correct()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            correct()

    snapshots = []
    # All rejected, partially accepted, and fully accepted drafts. Requests
    # are reordered, one is replaced, and a continuing prefill has no drafts.
    for counts in ([1, 1, 1, 1], [2, 1, 4, 1], [6, 1, 6, 1]):
        computed.copy_(torch.tensor([10, 20, 30, 999], dtype=torch.int32, device=device))
        valid_counts.copy_(torch.tensor(counts, dtype=torch.int64, device=device))
        runner.num_accepted_tokens.gpu.fill_(99)
        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU]) as prof:
            runner._prepare_num_accepted_tokens(4, has_prev_mapping=True)
            if graph is None:
                correct()
            else:
                graph.replay()
        trace_path = tmp_path / f"trace_{counts[0]}.json"
        prof.export_chrome_trace(str(trace_path))
        trace = json.loads(trace_path.read_text())
        names = {event["name"] for event in trace if event.get("cat") == "cpu_op"}
        assert "aten::fill_" in names
        assert "Event::synchronize" not in names
        assert "acl_memcpy_host_to_device" not in names
        assert "acl_memcpy_device_to_host" not in names
        snapshots.append((computed.clone(), runner.num_accepted_tokens.gpu.clone(), counts))

    torch.npu.synchronize()
    for positions, accepted, counts in snapshots:
        torch.testing.assert_close(
            positions.cpu(), torch.tensor([30 + counts[2], 0, 10 + counts[0], 21], dtype=torch.int32)
        )
        torch.testing.assert_close(
            accepted.cpu(), torch.tensor([counts[2], 1, counts[0], 1, 1, 1, 1, 1], dtype=torch.int32)
        )
