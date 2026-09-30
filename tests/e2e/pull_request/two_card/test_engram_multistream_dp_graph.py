# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram DP routing overlaps a graph on a separate EP communicator."""

import socket
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
import torch_npu  # noqa: F401
from vllm.distributed import parallel_state
from vllm.forward_context import override_forward_context

# Match worker startup before importing the model's attention dependencies.
import vllm_ascend.ops  # noqa: F401

# isort: split
from tests.e2e.pull_request.two_card.test_engram_fixed_slots import (
    _CAPACITY,
    _DIM,
    _HEAD_SIZES,
    _context,
    _ids,
    _reference,
)
from vllm_ascend.models.deepseek_v41.engram import parallel
from vllm_ascend.models.deepseek_v41.engram.embedding import AscendParallelEngramEmbedding
from vllm_ascend.models.deepseek_v41.engram.graph_inputs import wait_engram_event
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Model


@torch.inference_mode()
def _worker(rank, port):
    torch.npu.set_device(rank)
    parallel_state.init_distributed_environment(
        world_size=2,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="hccl",
    )
    try:
        parallel_state._DP = parallel_state.init_model_parallel_group(
            [[0, 1]], rank, "hccl", group_name="engram_dp_overlap"
        )
        parallel_state._TP = parallel_state.init_model_parallel_group(
            [[0], [1]], rank, "hccl", group_name="engram_tp_overlap"
        )
        ep = parallel_state.init_model_parallel_group([[0, 1]], rank, "hccl", group_name="engram_ep_overlap")
        rows = sum(_HEAD_SIZES)
        codes = ((torch.arange(rows * _DIM).view(rows, _DIM) * 13) % 251 - 125).to(torch.int8)
        scales = torch.pow(2.0, torch.arange(rows * (_DIM // 32)).view(rows, _DIM // 32) % 5 - 3).float()
        tables = {
            layer: AscendParallelEngramEmbedding(rows, _DIM, _HEAD_SIZES, slot) for slot, layer in enumerate((1, 14))
        }
        for table in tables.values():
            start, stop = table.vocab_start_idx, table.vocab_end_idx
            table.weight.copy_(codes[start:stop].npu())
            table.weight_scale_inv.copy_(scales[start:stop].npu())

        class HashState:
            lookback_depth = 2

            def ensure_cache(self):
                return True

            def __call__(self, *args):
                return self.ids

            def dummy_hashes(self, tokens):
                return tokens.new_full((tokens.shape[0], 2, 24), -1), tokens.new_zeros(
                    tokens.shape[0], dtype=torch.bool
                )

        model = object.__new__(DeepseekV41Model)
        torch.nn.Module.__init__(model)
        model.has_engram, model.engram_dp_shared_memory = True, False
        model.engram_hash = HashState()
        model.config = SimpleNamespace(engram_layer_ids=(1, 14), image_token_id=999)
        model.layers = [SimpleNamespace(engram=SimpleNamespace(embed_tokens=tables.get(layer))) for layer in range(15)]
        model._engram_max_tokens, model._engram_local_graph_inputs = _CAPACITY, None
        model.engram_rotation = torch.eye(32, device="npu")
        main, aux = torch.npu.current_stream(), torch.npu.Stream()
        bucket = (6, 48)[rank]
        with patch.object(parallel, "engram_gathered_num_tokens", return_value=_CAPACITY):
            binding = model.prepare_engram_overlap_graph_inputs(bucket, bucket, prime=True)
            ep_input = torch.full((8,), rank + 1.0, device="npu")
            # Initialize the collective before capturing its fixed addresses.
            ep.all_reduce(ep_input)
            ep_input.fill_(rank + 1.0)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                wait_engram_event(binding["engram_mask_ready_event"], True)
                reduced = ep.all_reduce(ep_input)
                outputs = {}
                for layer in (1, 14):
                    wait_engram_event(binding["engram_ready_events"][layer], True)
                    outputs[layer] = torch.where(
                        binding["engram_mask"][:bucket, None], binding["engram_lookups"][layer][:bucket], 0
                    )
            for phase, counts in enumerate(((2, 31), (5, 3), (0, 36), (1, 0)) * 3):
                count = counts[rank]
                ids_cpu = torch.stack([_ids(count, rank, phase + layer) for layer in range(2)], dim=1)
                model.engram_hash.ids = ids_cpu.npu()
                tokens = torch.ones(count, dtype=torch.int32, device="npu")
                positions = torch.arange(count, device="npu")
                query = torch.tensor([0, count] if count else [0], dtype=torch.int32, device="npu")
                block = torch.zeros((1 if count else 0, 1), dtype=torch.int32, device="npu")
                ep_input.fill_(rank + 1.0)
                aux.wait_stream(main)
                with override_forward_context(_context(bucket)), torch.npu.stream(aux):
                    model.prepare_engram_graph_overlap_inputs(
                        tokens,
                        positions,
                        None,
                        query_start_loc=query,
                        block_table=block,
                        graph_inputs=binding,
                        padded_tokens=bucket,
                    )
                graph.replay()
                main.wait_stream(aux)
                assert torch.equal(reduced.cpu(), torch.full((8,), 3.0))
                for slot, layer in enumerate((1, 14)):
                    expected = torch.zeros((bucket, len(_HEAD_SIZES), _DIM), dtype=torch.bfloat16)
                    expected[:count] = _reference(ids_cpu[:, slot], codes, scales)
                    assert torch.equal(outputs[layer].cpu(), expected.flatten(1)), (rank, phase, layer)
        print(f"ENGRAM_DP_GRAPH_OVERLAP_12_REPLAYS_PASSED rank={rank}", flush=True)
        ep.destroy()
    finally:
        torch.npu.synchronize()
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()


def test_engram_dp_graph_overlap_with_unequal_and_empty_rank_batches():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)


if __name__ == "__main__":
    test_engram_dp_graph_overlap_with_unequal_and_empty_rank_batches()
