# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Auxiliary collective ownership, routing order and idle-rank participation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import model as model_mod
from vllm_ascend.models.deepseek_v41.engram import parallel


@pytest.mark.parametrize("shared", [False, True])
def test_sibling_groups_are_created_once_in_dp_tp_order_and_closed(monkeypatch, shared):
    calls = []

    def coordinator(name):
        return SimpleNamespace(
            world_size=2,
            make_sibling_device_group=lambda **kwargs: calls.append(("create", name)) or name,
        )

    monkeypatch.setattr(parallel, "get_engram_dp_group", lambda: coordinator("dp"))
    monkeypatch.setattr(parallel, "get_tp_group", lambda: coordinator("tp"))
    monkeypatch.setattr(parallel.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel.dist, "destroy_process_group", lambda group: calls.append(("close", group)))
    groups = parallel.EngramAuxGroups(dp_shared_memory=shared)
    assert calls == ([("create", "tp")] if shared else [("create", "dp"), ("create", "tp")])
    groups.close()
    groups.close()
    assert calls[-1 if shared else -2 :] == ([("close", "tp")] if shared else [("close", "tp"), ("close", "dp")])
    with pytest.raises(RuntimeError, match="closed"):
        groups.gather_tp_heads(torch.ones(1, 1, 4))


def test_partial_group_creation_failure_releases_dp_group(monkeypatch):
    destroyed = []
    dp = SimpleNamespace(world_size=2, make_sibling_device_group=Mock(return_value="dp"))
    tp = SimpleNamespace(world_size=2, make_sibling_device_group=Mock(side_effect=RuntimeError("TP init failed")))
    monkeypatch.setattr(parallel, "get_engram_dp_group", lambda: dp)
    monkeypatch.setattr(parallel, "get_tp_group", lambda: tp)
    monkeypatch.setattr(parallel.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel.dist, "destroy_process_group", destroyed.append)
    with pytest.raises(RuntimeError, match="TP init failed"):
        parallel.EngramAuxGroups()
    assert destroyed == ["dp"]


def test_tp_gather_uses_sibling_and_restores_head_order(monkeypatch):
    calls = []
    monkeypatch.setattr(parallel, "get_engram_dp_group", lambda: None)
    monkeypatch.setattr(
        parallel, "get_tp_group", lambda: SimpleNamespace(world_size=2, make_sibling_device_group=lambda **kw: "aux_tp")
    )
    monkeypatch.setattr(parallel.dist, "is_initialized", lambda: False)

    def gather(output, values, *, group):
        calls.append(group)
        assert values.is_contiguous()
        output.copy_(torch.cat((values, values + 100)))

    def select(gathered, rows, source_tokens, token_start, local_width):
        assert source_tokens == rows.shape[0] and token_start == 0
        assert local_width == 6
        rows.copy_(gathered.view(2, source_tokens, 2, 3).permute(1, 0, 2, 3).reshape_as(rows))

    monkeypatch.setattr(parallel.dist, "all_gather_into_tensor", gather)
    monkeypatch.setattr(parallel, "_engram_select_rows", select)
    groups = parallel.EngramAuxGroups()
    values = torch.arange(24).view(2, 2, 6)[:, :, ::2]
    output = groups.gather_tp_heads(values)
    assert torch.equal(output, torch.cat((values, values + 100), dim=1))
    assert calls == ["aux_tp"]
    assert groups.gather_tp_heads(values[:0]).shape == (0, 4, 3)
    assert calls == ["aux_tp"]
    groups.close()


@pytest.mark.parametrize("idle", [False, True])
def test_producer_routes_all_three_dp_gathers_before_table_events(monkeypatch, idle):
    calls = []
    dp = SimpleNamespace(world_size=2, all_gather=Mock(side_effect=AssertionError("main DP group used")))
    monkeypatch.setattr(parallel, "get_engram_dp_group", lambda: dp)
    monkeypatch.setattr(parallel, "engram_gathered_num_tokens", lambda: 2)
    monkeypatch.setattr(model_mod, "get_engram_dp_size", lambda: 2)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: "aux")

    class Event:
        def record(self, stream):
            assert stream == "aux"
            calls.append("ready")

    monkeypatch.setattr(torch.npu, "Event", Event)

    class HashState:
        lookback_depth = 2

        def ensure_cache(self):
            return True

        def __call__(self, tokens, *args):
            calls.append("hash")
            return tokens.new_zeros((tokens.shape[0], 2, 24))

        def dummy_hashes(self, tokens):
            calls.append("dummy")
            return tokens.new_full((tokens.shape[0], 2, 24), -1), tokens.new_zeros(tokens.shape[0], dtype=torch.bool)

    class Groups:
        def gather_dp(self, values):
            calls.append("dp_ids" if values.dtype == torch.int32 else "dp_rows")
            assert values.shape[0] in (2, 4)
            return torch.cat((values, values))

    groups = Groups()

    def table(layer):
        def embed(ids, count, *, aux_groups):
            assert aux_groups is groups
            calls.append(f"lookup{layer}")
            aux_groups.gather_dp(torch.zeros((ids.shape[0], 12, 4), dtype=torch.bfloat16))
            return torch.full((count, 24, 4), layer, dtype=torch.bfloat16)

        return SimpleNamespace(embed_gathered=embed)

    model = object.__new__(model_mod.DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.has_engram, model.engram_dp_shared_memory = True, False
    model._engram_aux_groups = groups
    model.engram_hash = HashState()
    model.config = SimpleNamespace(engram_layer_ids=(1, 14), image_token_id=999)
    model.layers = [SimpleNamespace(engram=SimpleNamespace(embed_tokens=table(layer))) for layer in range(15)]
    count = 1 if idle else 2
    result = model.prepare_engram_local_inputs(
        torch.ones(count, dtype=torch.int32),
        torch.arange(count),
        query_start_loc=torch.tensor([0] if idle else [0, count], dtype=torch.int32),
        block_table=torch.zeros((0 if idle else 1, 1), dtype=torch.int32),
    )
    assert "engram_local_rows" not in result
    assert result["engram_lookups"][1].shape == (count, 96)
    assert result["engram_mask"].tolist() == [not idle] * count
    assert calls == [
        "ready",
        "dummy" if idle else "hash",
        "dp_ids",
        "lookup1",
        "dp_rows",
        "ready",
        "lookup14",
        "dp_rows",
        "ready",
    ]
