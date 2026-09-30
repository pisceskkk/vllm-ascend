# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm_ascend.models.deepseek_v41.engram.graph_inputs import EngramGraphInputs, wait_engram_event
from vllm_ascend.models.deepseek_v41.vl_model import AscendDeepseekV41ForCausalLM


def test_graph_frontiers_stay_bound_to_each_descriptor(monkeypatch):
    calls = []

    class Event:
        def record(self, stream):
            calls.append((self, "record", stream))

        def wait(self, stream):
            calls.append((self, "wait", stream))

        def reset(self, stream):
            calls.append((self, "reset", stream))

    monkeypatch.setattr(torch.npu, "ExternalEvent", Event)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: "main")
    tables = {layer: SimpleNamespace(part_n_hash_cols=3, n_hash_cols=24, dim=4) for layer in (1, 14)}
    inputs = EngramGraphInputs(tables, 192, "cpu")
    first = inputs.bindings("batch96", prime=True)
    second = inputs.bindings("batch192", prime=True)
    replay = inputs.bindings("batch96")
    assert len(calls) == 6
    assert replay["engram_mask_ready_event"] is first["engram_mask_ready_event"]
    assert second["engram_mask_ready_event"] is not first["engram_mask_ready_event"]
    assert replay["engram_local_rows"] is second["engram_local_rows"]
    for layer in (1, 14):
        assert replay["engram_ready_events"][layer] is first["engram_ready_events"][layer]
        assert second["engram_ready_events"][layer] is not first["engram_ready_events"][layer]
    event = first["engram_ready_events"][1]
    wait_engram_event(event, True)
    assert calls[-2:] == [(event, "wait", "main"), (event, "reset", "main")]
    full = EngramGraphInputs(tables, 192, "cpu", full_rows=True).bindings("full")
    assert "engram_local_rows" not in full
    assert full["engram_lookups"][1].shape == (192, 96)


def test_multimodal_wrapper_exposes_graph_producer_contract():
    wrapper = object.__new__(AscendDeepseekV41ForCausalLM)
    torch.nn.Module.__init__(wrapper)
    binding = {"engram_graph_events": True}
    wrapper.language_model = SimpleNamespace(
        engram_multistream_supported=True,
        engram_graph_multistream_supported=True,
        prepare_engram_local_inputs=Mock(return_value=binding),
        prepare_engram_overlap_graph_inputs=Mock(return_value=binding),
        prepare_engram_graph_overlap_inputs=Mock(return_value=binding),
    )
    assert wrapper.engram_multistream_supported and wrapper.engram_graph_multistream_supported
    assert wrapper.prepare_engram_overlap_graph_inputs(96, "batch96", prime=True) is binding
    wrapper.language_model.prepare_engram_overlap_graph_inputs.assert_called_once_with(96, "batch96", prime=True)
    assert wrapper.prepare_engram_local_inputs("ids", "positions", graph_inputs=binding, padded_tokens=96) is binding
    wrapper.language_model.prepare_engram_local_inputs.assert_called_once_with(
        "ids", "positions", None, None, None, None, graph_inputs=binding, padded_tokens=96
    )
    assert (
        wrapper.prepare_engram_graph_overlap_inputs("ids", "positions", None, graph_inputs=binding, padded_tokens=96)
        is binding
    )
    wrapper.language_model.prepare_engram_graph_overlap_inputs.assert_called_once_with(
        "ids", "positions", None, graph_inputs=binding, padded_tokens=96
    )
