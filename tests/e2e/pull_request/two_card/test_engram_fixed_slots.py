# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check fixed Engram slots with real HCCL and independent INT8 row oracles."""

import socket
import statistics
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
import torch_npu  # noqa: F401
from vllm.config import CUDAGraphMode
from vllm.distributed import parallel_state
from vllm.forward_context import DPMetadata, ForwardContext, override_forward_context

from vllm_ascend import utils
from vllm_ascend.models.deepseek_v41.engram import parallel
from vllm_ascend.models.deepseek_v41.engram.embedding import AscendParallelEngramEmbedding
from vllm_ascend.models.deepseek_v41.engram.hash_state import AscendNgramHashState
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

_CAPACITY = 96
_HEAD_SIZES = tuple(11 + 2 * head for head in range(24))
_DIM = 256


def _ids(count, rank, phase):
    columns = []
    start = 0
    for head, size in enumerate(_HEAD_SIZES):
        columns.append(start + (torch.arange(count) * 3 + rank * 7 + phase * 5 + head) % size)
        start += size
    ids = torch.stack(columns, dim=1).to(torch.int32)
    if count:
        ids[0, 0] = -1
    if count > 1:
        ids[-1, -1] = start + 3
    return ids


def _reference(ids, codes, scales):
    # Global-table CPU oracle, independent of production rank/row selection.
    rows = torch.zeros((*ids.shape, codes.shape[1]), dtype=torch.bfloat16)
    start = 0
    for head, size in enumerate(_HEAD_SIZES):
        for token, row in enumerate(ids[:, head].tolist()):
            if start <= row < start + size:
                rows[token, head] = (codes[row].float().reshape(-1, 32) * scales[row, :, None]).flatten()
        start += size
    return rows


def _context(count, *, profile=False, warmup=False, mode=CUDAGraphMode.NONE):
    # Reproduce the runner's rank-local vector when its CPU all-reduce is skipped.
    context = ForwardContext({}, None, {}, dp_metadata=DPMetadata(torch.tensor([count, count])))
    context.in_profile_run = profile
    context.engram_uniform_dp_warmup = warmup
    context.cudagraph_runtime_mode = mode
    return context


def _lookup(tables, hashes):
    # The model gathers both layers' hashes once, then exchanges rows per table.
    gathered = parallel.gather_engram_hashes(hashes)
    return [table.embed_gathered(gathered[:, layer], hashes.shape[0]) for layer, table in enumerate(tables)]


@torch.inference_mode()
def _worker(rank, port):
    torch.npu.set_device(rank)
    init_device_properties_triton()
    parallel_state.init_distributed_environment(
        world_size=2, rank=rank, local_rank=rank, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="hccl"
    )
    try:
        parallel_state._DP = parallel_state.init_model_parallel_group(
            [[0, 1]], rank, "hccl", group_name="engram_dp_test"
        )
        parallel_state._TP = parallel_state.init_model_parallel_group(
            [[0], [1]], rank, "hccl", group_name="engram_tp_test"
        )
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(is_kv_consumer=True, is_kv_producer=False),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=1024, max_num_seqs=16),
            speculative_config=SimpleNamespace(num_speculative_tokens=5),
        )
        ascend = SimpleNamespace(scheduler_config=SimpleNamespace(recompute_scheduler_enable=True))
        total_rows = sum(_HEAD_SIZES)
        codes = ((torch.arange(total_rows * _DIM).reshape(total_rows, _DIM) * 13) % 251 - 125).to(torch.int8)
        scales = torch.pow(2.0, torch.arange(total_rows * (_DIM // 32)).reshape(total_rows, _DIM // 32) % 5 - 3).float()
        tables = [AscendParallelEngramEmbedding(total_rows, _DIM, _HEAD_SIZES, layer) for layer in range(2)]
        for table in tables:
            start, end = table.vocab_start_idx, table.vocab_end_idx
            table.weight.copy_(codes[start:end].npu())
            table.weight_scale_inv.copy_(scales[start:end].npu())
        hash_state = SimpleNamespace(multipliers=torch.empty(2, 4), primes=torch.empty(2, 3, 8))
        calls = []
        dp_group = parallel_state.get_dp_group()
        all_gather = dp_group.all_gather

        def tracked_gather(tensor, dim=0):
            calls.append(tuple(tensor.shape))
            return all_gather(tensor, dim=dim)

        with (
            patch.object(utils, "get_ascend_config", return_value=ascend),
            patch("vllm.config.get_current_vllm_config_or_none", return_value=config),
            patch.object(parallel, "get_current_vllm_config", return_value=config),
            patch.object(parallel, "get_potential_max_tokens", return_value=_CAPACITY),
            patch.object(dp_group, "all_gather", side_effect=tracked_gather),
            patch.object(torch.distributed, "all_reduce", side_effect=AssertionError("runtime CPU synchronization")),
        ):
            # No illegal HCCL baseline: first prove that the old local metadata
            # path chooses different counts, then execute only the fixed path.
            with override_forward_context(_context((42, 48)[rank])):
                ascend.scheduler_config.recompute_scheduler_enable = False
                assert parallel.engram_gathered_num_tokens() == (42, 48)[rank]
                ascend.scheduler_config.recompute_scheduler_enable = True
                assert parallel.engram_gathered_num_tokens() == _CAPACITY

            for phase, counts in enumerate(((1, 5), (7, 2), (0, 3), (3, 0), (96, 42), (2, 95), (1, 48))):
                count = counts[rank]
                idle = phase == 6 and rank == 0
                ids_cpu = torch.stack([_ids(count, rank, phase + layer) for layer in range(2)], dim=1)
                hashes = ids_cpu.npu()
                if idle:
                    hashes, keep = AscendNgramHashState.dummy_hashes(
                        hash_state, torch.zeros(count, device="npu", dtype=torch.int32)
                    )
                    ids_cpu.fill_(-1)
                    assert not keep.any().item()
                calls.clear()
                with override_forward_context(_context(count)):
                    actual = _lookup(tables, hashes)
                assert [shape[0] for shape in calls] == [_CAPACITY, 2 * _CAPACITY, 2 * _CAPACITY]
                for layer, rows in enumerate(actual):
                    torch.testing.assert_close(rows.cpu(), _reference(ids_cpu[:, layer], codes, scales), rtol=0, atol=0)

            # Routing runs before graph replay in the model. Capture local
            # consumers with different buckets, then mix graph/eager ranks.
            for buckets, graph_ranks, idle_first in (
                ((6, 48), (True, True), False),
                ((6, 48), (True, True), True),
                ((6, 37), (True, False), False),
            ):
                bucket = buckets[rank]
                buffers = [
                    torch.empty((bucket, len(_HEAD_SIZES), _DIM), dtype=torch.bfloat16, device="npu") for _ in tables
                ]
                outputs = [torch.empty_like(buffer) for buffer in buffers]

                def consume(outputs=outputs, buffers=buffers):
                    for output, buffer in zip(outputs, buffers):
                        output.copy_(buffer)

                graph = torch.npu.NPUGraph()
                if graph_ranks[rank]:
                    consume()
                    torch.npu.synchronize()
                    with torch.npu.graph(graph):
                        consume()
                for phase, counts in ((20, (2, 31)), (21, (5, 3)), (22, (1, 36))):
                    count = counts[rank]
                    ids_cpu = torch.stack([_ids(count, rank, phase + layer) for layer in range(2)], dim=1)
                    hashes = ids_cpu.npu()
                    if idle_first and rank == 0:
                        hashes, _ = AscendNgramHashState.dummy_hashes(
                            hash_state, torch.zeros(count, device="npu", dtype=torch.int32)
                        )
                        ids_cpu.fill_(-1)
                    mode = CUDAGraphMode.FULL if graph_ranks[rank] else CUDAGraphMode.NONE
                    with override_forward_context(_context(bucket, mode=mode)):
                        rows = _lookup(tables, hashes)
                    for buffer, values in zip(buffers, rows):
                        buffer.zero_()
                        buffer[:count].copy_(values)
                    graph.replay() if graph_ranks[rank] else consume()
                    for layer, output in enumerate(outputs):
                        expected = torch.zeros_like(output, device="cpu")
                        expected[:count] = _reference(ids_cpu[:, layer], codes, scales)
                        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

            # Startup profile and compile warmups deliberately exceed decode capacity.
            for profile, warmup in ((True, False), (False, True)):
                ids_cpu = torch.stack([_ids(128, rank, 30 + layer) for layer in range(2)], dim=1)
                with override_forward_context(_context(128, profile=profile, warmup=warmup)):
                    rows = _lookup(tables, ids_cpu.npu())
                for layer, output in enumerate(rows):
                    torch.testing.assert_close(
                        output.cpu(), _reference(ids_cpu[:, layer], codes, scales), rtol=0, atol=0
                    )

            # Device timing isolates padding cost from CPU metadata synchronization.
            for count in (1, 48, _CAPACITY):
                hashes = torch.stack([_ids(count, rank, layer) for layer in range(2)], dim=1).npu()
                samples = {"fixed": [], "common_slot": []}
                for _ in range(3):
                    for fixed in (False, True):
                        ascend.scheduler_config.recompute_scheduler_enable = fixed
                        with override_forward_context(_context(count)):
                            for _ in range(3):
                                _lookup(tables, hashes)
                            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                            start.record()
                            for _ in range(20):
                                _lookup(tables, hashes)
                            end.record()
                            end.synchronize()
                            samples["fixed" if fixed else "common_slot"].append(start.elapsed_time(end) / 20)
                timings = {name: statistics.median(values) for name, values in samples.items()}
                print(f"ENGRAM_TIMING rank={rank} count={count} milliseconds={timings}", flush=True)
            ascend.scheduler_config.recompute_scheduler_enable = True
        torch.distributed.barrier(group=dp_group.cpu_group)
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()


def test_engram_fixed_slots_unequal_tokens_and_graph_replay():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)
