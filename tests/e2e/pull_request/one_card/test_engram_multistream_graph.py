# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refresh UVA lookup buffers across external-event graph replays."""

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

# Match worker startup: register ops before importing attention/model modules.
import vllm_ascend.ops  # noqa: F401

# isort: split
from vllm_ascend.models.deepseek_v41.engram.embedding import AscendParallelEngramEmbedding
from vllm_ascend.models.deepseek_v41.engram.npu import HostUvaBuffer
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model


@torch.inference_mode()
def test_engram_external_events_refresh_rows_padding_and_empty_batches():
    torch.npu.set_device(0)
    capacity, heads, width, vocab = 192, 3, 256, 101
    codes = ((torch.arange(vocab * width).view(vocab, width) * 7) % 251 - 125).to(torch.int8)
    scales = torch.ones(vocab, width // 32) * 0.25
    host_codes = HostUvaBuffer(codes.shape, codes.dtype, torch.device("npu:0"))
    host_scales = HostUvaBuffer(scales.shape, scales.dtype, torch.device("npu:0"))
    host_codes.tensor.copy_(codes)
    host_scales.tensor.copy_(scales)

    tables = {}
    for layer in (1, 14):
        table = object.__new__(AscendParallelEngramEmbedding)
        torch.nn.Module.__init__(table)
        table.part_n_hash_cols, table.dim, table.head_start = heads, width, 0
        table.n_hash_cols, table.dp_size, table.tp_size = heads, 1, 1
        table.vocab_start_idx, table.vocab_end_idx = 0, vocab
        table._codes_uva, table._scales_uva = host_codes, host_scales
        tables[layer] = table
    model = object.__new__(DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.has_engram = True
    model.engram_dp_shared_memory = True
    model._engram_aux_groups = None

    # Inject already-known IDs to isolate graph buffer/event correctness from
    # hash arithmetic, which has separate real-cache coverage.
    class HashState:
        lookback_depth = 2

        def ensure_cache(self):
            return True

        def __call__(self, input_ids, *args):
            return input_ids[:, None, None].expand(-1, 2, heads).int()

    model.engram_hash = HashState()
    model.config = SimpleNamespace(engram_layer_ids=(1, 14), image_token_id=999)
    model.layers = [SimpleNamespace(engram=SimpleNamespace(embed_tokens=tables.get(layer))) for layer in range(15)]
    model._engram_input_buffers, model._engram_max_tokens = None, capacity
    model._engram_graph_events = {}
    model.engram_rotation = torch.eye(32, device="npu")
    aux = torch.npu.Stream()
    main = torch.npu.current_stream()
    graphs, outputs = {}, {}
    try:
        with patch("vllm_ascend.models.deepseek_v41.model.gather_engram_hashes", lambda ids, **kwargs: ids):
            for size in (96, 192):
                bindings = model.prepare_engram_overlap_graph_inputs(size, prime=True)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    model._wait_engram_event(bindings["engram_mask_ready_event"], True)
                    output = {}
                    for layer in (1, 14):
                        model._wait_engram_event(bindings["engram_ready_events"][layer], True)
                        output[layer] = torch.where(
                            bindings["engram_mask"][:size, None],
                            bindings["engram_lookups"][layer][:size],
                            0,
                        )
                graphs[size], outputs[size] = graph, output
            # Alternate descriptors and lengths, including padding, invalid
            # IDs and zero request rows. Stale event or buffer reuse must fail
            # an independent CPU row oracle rather than just launch cleanly.
            for phase, (size, count) in enumerate(((192, 168), (96, 1), (192, 192), (96, 0), (192, 17)) * 3):
                ids = (torch.arange(count) + phase * 11) % vocab
                if count > 1:
                    ids[-1] = -1
                positions = torch.arange(count, device="npu")
                ids_device = ids.npu()
                query = torch.tensor([0, count] if count else [0], dtype=torch.int32, device="npu")
                blocks = torch.zeros((1 if count else 0, 1), dtype=torch.int32, device="npu")
                binding = model.prepare_engram_overlap_graph_inputs(size)
                aux.wait_stream(main)
                with torch.npu.stream(aux):
                    model.prepare_engram_overlap_inputs(
                        ids_device,
                        positions,
                        query_start_loc=query,
                        block_table=blocks,
                        graph_inputs=binding,
                        padded_tokens=size,
                    )
                graphs[size].replay()
                main.wait_stream(aux)
                oracle = torch.zeros((size, heads, width), dtype=torch.bfloat16)
                for token, row in enumerate(ids.tolist()):
                    if row >= 0:
                        oracle[token] = (codes[row].float() * 0.25).bfloat16()
                for layer in (1, 14):
                    assert torch.equal(outputs[size][layer].cpu(), oracle.flatten(1)), (phase, size, count, layer)
    finally:
        torch.npu.synchronize()
        host_codes.close()
        host_scales.close()
