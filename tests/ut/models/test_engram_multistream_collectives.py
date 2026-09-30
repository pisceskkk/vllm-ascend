# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram producer exchanges through the existing DP group."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import model as model_mod
from vllm_ascend.models.deepseek_v41.engram import embedding, parallel


@pytest.mark.parametrize("idle", [False, True])
def test_producer_uses_existing_dp_group_before_table_events(monkeypatch, idle):
    calls = []

    def gather(values, dim):
        assert dim == 0 and torch.npu.current_stream() == "aux"
        calls.append("dp_ids" if values.dtype == torch.int32 else "dp_rows")
        assert values.shape[0] in (2, 4)
        return torch.cat((values, values))

    dp = SimpleNamespace(world_size=2, rank_in_group=0, all_gather=gather)
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

    def select(gathered, rows, source_tokens, token_start, local_width):
        assert local_width == 48
        selected = gathered.view(2, source_tokens, 12, 4)[:, token_start : token_start + rows.shape[0]]
        rows.copy_(selected.permute(1, 0, 2, 3).reshape_as(rows))

    monkeypatch.setattr(parallel, "_engram_select_rows", select)

    def table(layer):
        instance = object.__new__(embedding.AscendParallelEngramEmbedding)
        torch.nn.Module.__init__(instance)
        instance.dp_size, instance.tp_size = 2, 1
        instance.n_hash_cols, instance.part_n_hash_cols, instance.dim = 24, 12, 4

        def lookup(ids, out):
            calls.append(f"lookup{layer}")
            out.fill_(layer)

        instance.lookup = lookup
        return instance

    model = object.__new__(model_mod.DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.has_engram, model.engram_dp_shared_memory = True, False
    model.engram_hash = HashState()
    model.config = SimpleNamespace(engram_layer_ids=(1, 14), image_token_id=999)
    model.layers = [SimpleNamespace(engram=SimpleNamespace(embed_tokens=table(layer))) for layer in range(15)]
    count = 1 if idle else 2
    result = model.prepare_engram_overlap_inputs(
        torch.ones(count, dtype=torch.int32),
        torch.arange(count),
        query_start_loc=torch.tensor([0] if idle else [0, count], dtype=torch.int32),
        block_table=torch.zeros((0 if idle else 1, 1), dtype=torch.int32),
    )
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
