# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged HiSparse prefill against a causal FP32 reference, including replay."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_paged_prefill_reference(capture, mixed, strided):
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("HiSparse SFA integration currently targets Ascend 950")
    torch.npu.set_device(0)
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    from vllm_ascend.attention.sfa_hisparse import AscendHiSparseImpl

    torch.manual_seed(79)
    device, latent, rope, heads, topk = "npu:0", 512, 64, 8, 2048
    decode = 2 if mixed else 0
    queries, padding = 10, 5
    source = torch.randn(6, 128, latent + rope).bfloat16()
    if strided:
        backing = torch.empty(6, 2, 128, latent + rope, dtype=torch.bfloat16, device=device)
        cache = backing[:, 1]
        cache.copy_(source)
    else:
        cache = source.to(device)
    query = torch.randn(decode + queries + padding, heads, latent).bfloat16()
    query_rope = torch.randn(decode + queries + padding, heads, rope).bfloat16()
    q, qr = query.to(device), query_rope.to(device)
    # Non-contiguous physical pages, two unequal query counts and old context.
    table_cpu = torch.tensor([[4, 1, 0], [3, 5, 2]], dtype=torch.int32)
    table = table_cpu.to(device)
    lengths = [130, 265]
    counts = [3, 7]
    ends = torch.tensor(([0, 1, 2] if mixed else [0]) + [decode + 3, decode + queries], dtype=torch.int32)
    seq_lens = torch.tensor(([1, 1] if mixed else []) + lengths, dtype=torch.int32)
    indices = torch.full((decode + queries + padding, topk), -1, dtype=torch.int32)
    for row in range(queries):
        req = int(row >= counts[0])
        values = torch.randint(lengths[req], (33,), dtype=torch.int32)
        values[1] = values[0]
        values[2::7] = -1
        indices[decode + row, :33] = values
    indices[decode + 1] = -1
    gpu_indices, gpu_ends, gpu_lens = indices.to(device), ends.to(device), seq_lens.to(device)
    metadata = SimpleNamespace(
        req_id_per_token=torch.tensor(
            ([0, 1] if mixed else []) + [decode] * 3 + [decode + 1] * (queries - 3 + padding),
            dtype=torch.int32,
            device=device,
        ),
        query_start_loc=gpu_ends,
        num_decode_tokens=decode,
        num_decodes=decode,
        num_prefills=2,
        num_actual_tokens=decode + queries,
        seq_lens=gpu_lens,
        block_size=128,
        max_seq_len=265,
    )
    group = SimpleNamespace(
        set_logical_topk_ready=lambda _: None,
        stage_prefill_rows=lambda *_: (cache, table, None),
        physical_kv_cache=lambda _: cache,
    )
    impl = SimpleNamespace(
        index_group=group,
        index_group_index=0,
        kv_lora_rank=latent,
        qk_rope_head_dim=rope,
        scale=(latent + rope) ** -0.5,
        hisparse_cache=SimpleNamespace(
            source_block_table=table,
            runtime=SimpleNamespace(host_cache=cache),
            swap_in=lambda *_, **__: gpu_indices[:decode],
        ),
        _attention_physical=lambda query, *_: torch.full_like(query, 7),
    )

    def forward():
        return AscendHiSparseImpl._execute_sparse_flash_attention_process(
            impl, q, qr, None, gpu_indices, metadata, None, None
        )

    graph = None
    try:
        if capture:
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                forward()
                forward()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                actual = forward()
            # Replay changes page payload, request boundary, context and top-k.
            source.neg_()
            cache.copy_(source)
            queries = 11
            counts = [4, 7]
            metadata.num_actual_tokens = decode + queries
            ends[-1] = decode + queries
            indices[decode + queries - 1, :3] = torch.tensor([2, 1, 2])
            lengths = [131, 264]
            ends[-2] = decode + counts[0]
            seq_lens[-2:] = torch.tensor(lengths, dtype=torch.int32)
            indices[decode, :6] = torch.tensor([9, -1, 3, 9, 130, 2])
            gpu_indices.copy_(indices)
            gpu_ends.copy_(ends)
            metadata.req_id_per_token[decode : decode + queries] = torch.tensor(
                [decode] * counts[0] + [decode + 1] * counts[1], dtype=torch.int32, device=device
            )
            gpu_lens.copy_(seq_lens)
            for _ in range(3):
                graph.replay()
        else:
            actual = forward()
        actual = actual.float().cpu()
        expected = torch.zeros_like(query.float())
        expected[:decode] = 7
        for row in range(queries):
            req = int(row >= counts[0])
            offset = row if req == 0 else row - counts[0]
            ids = indices[decode + row].long()
            valid = (ids >= 0) & (ids < lengths[req] - counts[req] + offset + 1)
            ids = ids[valid]
            if ids.numel() == 0:
                continue
            values = source[table_cpu[req, ids // 128].long(), ids % 128].float()
            logits = query[decode + row].float() @ values[:, :latent].T
            logits += query_rope[decode + row].float() @ values[:, latent:].T
            probs = (logits * impl.scale).softmax(dim=-1)
            expected[decode + row] = probs @ values[:, :latent]
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.025)
        assert torch.count_nonzero(actual[decode + queries :]) == 0
        assert torch.count_nonzero(actual[decode + 1]) == 0
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()


@pytest.mark.parametrize("rows,topk,capture", [(0, 128, False), (1, 128, False), (129, 2048, True), (129, 3072, False)])
def test_prefill_index_preparation(rows, topk, capture):
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("HiSparse prefill preparation currently targets Ascend 950")
    torch.npu.set_device(0)
    from vllm_ascend.ops.hisparse import prepare_prefill_attention

    torch.manual_seed(83)
    # Row-strided top-k and a trimmed, row-strided page table.
    backing = torch.randint(-10, 350, (rows, topk + 7), dtype=torch.int32)
    source = backing[:, :topk]
    source[::3] = -1
    if rows > 1:
        source[1, :5] = torch.tensor([7, -1, 3, 7, 300])
    logical = backing.to("npu")[:, :topk]
    requests = torch.full((rows,), 3, dtype=torch.int32)
    requests[:1] = 2
    starts = torch.tensor([0, 1, 2, 3, 2 + rows], dtype=torch.int32)
    lengths = torch.tensor([1, 1, 130, 300], dtype=torch.int32)
    table_backing = torch.arange(2 * 17, dtype=torch.int32).reshape(2, 17)
    table = table_backing.to("npu")[:, :3]
    device_requests, device_starts, device_lengths = requests.to("npu"), starts.to("npu"), lengths.to("npu")

    def forward():
        return prepare_prefill_attention(logical, device_requests, device_starts, device_lengths, table, 2, 2)

    graph = None
    try:
        if capture:
            forward()
            forward()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                outputs = forward()
            source[1, :8] = torch.tensor([-1, 129, 5, -1, 5, 130, 131, 132])
            logical.copy_(source)
            # Repeated replay guards against dynamic-scatter compiler failures.
            for _ in range(200):
                graph.replay()
        else:
            outputs = forward()
        expected = torch.full((rows, 1, topk), -1, dtype=torch.int32)
        counts = torch.ones(rows, dtype=torch.int32)
        for row in range(rows):
            request = requests[row].item()
            end = lengths[request] - starts[request + 1] + 2 + row + 1
            selected = source[row][(source[row] >= 0) & (source[row] < end)]
            expected[row, 0, : selected.numel()] = selected
            counts[row] = max(selected.numel(), 1)
        references = (
            expected,
            torch.arange(1, rows + 1, dtype=torch.int32),
            counts,
            table_backing[requests.long() - 2, :3],
        )
        for actual, reference in zip(outputs, references):
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
