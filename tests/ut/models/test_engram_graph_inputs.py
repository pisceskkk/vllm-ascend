# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.models.deepseek_v41.engram.graph_inputs import EngramGraphInputs, wait_engram_event
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model
from vllm_ascend.models.deepseek_v41.vl_model import AscendDeepseekV41ForCausalLM
from vllm_ascend.worker import model_runner_v1 as runner_mod


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
    tables = {layer: SimpleNamespace(n_hash_cols=24, dim=4) for layer in (1, 14)}
    inputs = EngramGraphInputs(tables, 192, "cpu")
    first = inputs.bindings("batch96", prime=True)
    second = inputs.bindings("batch192", prime=True)
    replay = inputs.bindings("batch96")
    assert len(calls) == 6
    assert replay["engram_mask_ready_event"] is first["engram_mask_ready_event"]
    assert second["engram_mask_ready_event"] is not first["engram_mask_ready_event"]
    assert replay["engram_lookups"] is second["engram_lookups"]
    assert replay["engram_lookups"][1].shape == (192, 96)
    for layer in (1, 14):
        assert replay["engram_ready_events"][layer] is first["engram_ready_events"][layer]
        assert second["engram_ready_events"][layer] is not first["engram_ready_events"][layer]
    event = first["engram_ready_events"][1]
    wait_engram_event(event, True)
    assert calls[-2:] == [(event, "wait", "main"), (event, "reset", "main")]

    model = object.__new__(DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.has_engram, model.engram_dp_shared_memory = True, False
    model._engram_graph_inputs, model._engram_max_tokens = None, 192
    model.config = SimpleNamespace(engram_layer_ids=(1, 14))
    model.engram_rotation = torch.eye(32)
    model.layers = [SimpleNamespace(engram=SimpleNamespace(embed_tokens=tables.get(layer))) for layer in range(15)]
    synchronous = model.prepare_engram_graph_inputs()
    bindings = model.prepare_engram_overlap_graph_inputs("aux96", prime=True)
    assert bindings["engram_lookups"] is synchronous["engram_lookups"]
    assert bindings["engram_mask"] is synchronous["engram_mask"]
    assert bindings["engram_lookups"][1].shape == (192, 96)


def test_multimodal_wrapper_exposes_graph_producer_contract():
    wrapper = object.__new__(AscendDeepseekV41ForCausalLM)
    torch.nn.Module.__init__(wrapper)
    binding = {"engram_graph_events": True}
    wrapper.language_model = SimpleNamespace(
        prepare_engram_overlap_inputs=Mock(return_value=binding),
        prepare_engram_overlap_graph_inputs=Mock(return_value=binding),
    )
    assert wrapper.prepare_engram_overlap_graph_inputs("batch96", prime=True) is binding
    wrapper.language_model.prepare_engram_overlap_graph_inputs.assert_called_once_with("batch96", prime=True)
    assert wrapper.prepare_engram_overlap_inputs("ids", "positions", graph_inputs=binding, padded_tokens=96) is binding
    wrapper.language_model.prepare_engram_overlap_inputs.assert_called_once_with(
        "ids", "positions", None, None, None, None, graph_inputs=binding, padded_tokens=96
    )


def test_runner_skips_engram_preparation_when_configuration_is_disabled(monkeypatch):
    runner = object.__new__(runner_mod.NPUModelRunner)
    runner.model = Mock(return_value="hidden")
    runner.vllm_config = SimpleNamespace(engram_config=None)
    runner.enable_enpu = False
    runner._update_full_graph_params_if_needed = Mock()
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    monkeypatch.setattr(runner_mod, "get_forward_context", lambda: context)
    # A global overlap setting must not cause events, streams or metadata
    # preparation for a model whose Engram feature is disabled.
    monkeypatch.setattr(runner_mod, "get_ascend_config", Mock(side_effect=AssertionError("Engram path entered")))
    ids, positions = torch.tensor([1]), torch.tensor([0])
    assert runner._model_forward(1, ids, positions) == "hidden"
    runner.model.assert_called_once_with(
        input_ids=ids,
        positions=positions,
        intermediate_tensors=None,
        inputs_embeds=None,
    )
    runner.model.prepare_engram_inputs.assert_not_called()
    runner.model.prepare_engram_overlap_inputs.assert_not_called()
