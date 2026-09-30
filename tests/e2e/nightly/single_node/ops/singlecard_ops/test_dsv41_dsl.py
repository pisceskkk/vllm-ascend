# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the in-tree DSL kernels and their metadata on an Ascend A5.

TopK values/indices use an independent MXFP4 decode and ordered BF16 score
reference. Exact cutoff ties may select any tied position; positions strictly
above the cutoff must all be present. Attention uses FP32 softmax with an
explicit sink and independently decoded cache rows (rtol=0.02, atol=0.02).
"""

import importlib
import math
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from tests.e2e.nightly.single_node.ops.singlecard_ops.dsv41_reference import (
    assert_candidate_topk,
    assert_topk,
    decode_attention_cache,
    dequantize_indexer,
    indexer_scores,
    pack_attention_cache,
    quantize_indexer,
)
from vllm_ascend.ops.dsv41_a5.attention import qsmla
from vllm_ascend.ops.dsv41_a5.indexer import run_a5_indexer
from vllm_ascend.ops.dsv41_a5.writers import write_attention_cache
from vllm_ascend.ops.pythondsl import ops
from vllm_ascend.ops.triton.build_window_indices import build_window_indices_triton
from vllm_ascend.ops.triton.fold_indexer_cache import fold_indexer_cache_rows
from vllm_ascend.ops.triton.quantize_mxfp4_indexer import quantize_mxfp4_indexer, write_mxfp4_indexer_cache
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import load_custom_op_library


@pytest.fixture(scope="module", autouse=True)
def initialize_device_properties():
    init_device_properties_triton()


def _assert_indexer_metadata(metadata, expected_tiles, workers=None):
    # The LI ABI contains lexicographic [begin, end) task boundaries.
    # Enumerate logical work independently and require single ownership.
    words = metadata.cpu()
    if workers is None:
        workers = min(32, int(torch.npu.get_device_properties(metadata.device).cube_core_num))
    assert bool((words[workers * 8 : 288] == 0).all())
    assert bool((words[288 + workers * 16 : 864] == 0).all())
    ownership = {tile: 0 for tile in expected_tiles}
    for record in words[:288].reshape(36, 8).tolist():
        if record[0] == 0:
            assert record == [0] * 8
            continue
        assert record[0] == 1
        begin, end = tuple(record[1:4]), tuple(record[4:7])
        assert begin < end
        for tile in ownership:
            if begin <= tile < end:
                ownership[tile] += 1
    assert all(count == 1 for count in ownership.values())
    assert bool((words[864:] == 0).all())


@pytest.mark.parametrize("sparse", [False, True])
@torch.inference_mode()
def test_indexer_metadata_32_worker_abi(sparse):
    """Exercise 32-worker AICPU scheduling without launching 32 Cube workers.

    This verifies the retained LI/LD ABI on any supported device. Numerical
    core execution uses the real physical worker count in the module tests.
    """
    name = "quant_sparse_lightning_indexer" if sparse else "quant_lightning_indexer"
    module = importlib.import_module(f"vllm_ascend.ops.pythondsl.{name}_metadata_dsl")
    cu = torch.tensor([0, 7], dtype=torch.int32, device="npu")
    used = torch.tensor([16449], dtype=torch.int32, device="npu")
    candidates = torch.full((7, 1), 2048, dtype=torch.int32, device="npu")
    metadata = torch.empty(1024, dtype=torch.int32, device="npu")
    groups = 7 if sparse else 2
    scratch = torch.empty(groups * 36, dtype=torch.int64, device="npu")
    _, compiled = module.compiled_metadata()
    compiled.launch(
        module.current_raw_stream(metadata.device.index),
        cu=cu.data_ptr(),
        used_q=0,
        used_k=used.data_ptr(),
        residual=0,
        candidate_length=candidates.data_ptr() if sparse else 0,
        output_offset=0,
        output=metadata.data_ptr(),
        scratch=scratch.data_ptr(),
        has_cu=1,
        has_q=0,
        has_k=1,
        has_residual=0,
        sparse=int(sparse),
        mask=3,
        ratio=1,
        total_q=7,
        batch=1,
        max_tasks=groups,
        capacity=16449,
        splits=8,
        workers=32,
        groups=groups,
        query_rows=1 if sparse else 6,
        heads=32,
        ld=1,
        has_offset=0,
    )
    expected_tiles = [(0, group, tile) for group in range(groups) for tile in range(32 if sparse else 65)]
    _assert_indexer_metadata(metadata, expected_tiles, workers=32)
    words = metadata.cpu()
    li = words[:288].reshape(36, 8).tolist()
    assert all(record[0] == 1 for record in li[:32])
    # Derive each merge's fan-in from independent logical tile ownership,
    # then require single ownership of every query row in that merge.
    owners = [set() for _ in range(groups)]
    for worker, record in enumerate(li[:32]):
        begin, end = tuple(record[1:4]), tuple(record[4:7])
        for tile in expected_tiles:
            if begin <= tile < end:
                owners[tile[1]].add(worker)
    merge_rows = {
        (group, row): 0
        for group in range(groups)
        if len(owners[group]) > 1
        for row in range(1 if sparse else min(6, 7 - group * 6))
    }
    workspace_ranges = {}
    for record in words[288:864].reshape(72, 8).tolist():
        if not record[0]:
            assert record == [0] * 8
            continue
        enabled, batch, group, base, parts, first_row, rows, reserved = record
        assert enabled == 1 and batch == 0 and reserved == 0
        assert parts == len(owners[group]) and parts > 1
        assert base >= 0 and rows > 0
        if group in workspace_ranges:
            assert workspace_ranges[group] == (base, parts)
        workspace_ranges[group] = (base, parts)
        for row in range(first_row, first_row + rows):
            merge_rows[group, row] += 1
    assert merge_rows and all(count == 1 for count in merge_rows.values())
    slots = [slot for base, parts in workspace_ranges.values() for slot in range(base, base + parts)]
    assert sorted(slots) == list(range(len(slots)))


@pytest.mark.parametrize(
    "ratio,candidates_enabled,large_workload",
    [(1, False, False), (1, True, False), (2, True, False), pytest.param(1, True, True, id="candidate-truncation")],
)
@torch.inference_mode()
def test_indexer_metadata_cache_and_candidates(ratio, candidates_enabled, large_workload):
    torch.manual_seed(412)
    lengths = [16449, 513] if large_workload else [769, 145]
    query_lengths = [7, 3] if large_workload else [4, 3]
    total_queries = sum(query_lengths)
    residual = [0, 1] if ratio == 2 else [0, 0]
    cu_cpu = torch.tensor([0, query_lengths[0], total_queries], dtype=torch.int32)
    cu, used = cu_cpu.npu(), torch.tensor(lengths, dtype=torch.int32, device="npu")
    rem = torch.tensor(residual, dtype=torch.int32, device="npu") if ratio != 1 else None
    page_size = 128
    pages_per_batch = math.ceil(max(lengths) / page_size)
    block_table = torch.randperm(2 * pages_per_batch).reshape(2, pages_per_batch).int()
    query = torch.randn(sum(query_lengths), 32, 128).to(torch.bfloat16)
    key = torch.randn(sum(lengths), 128).to(torch.bfloat16)
    weights = torch.randn(sum(query_lengths), 32) / 32
    weights_npu = weights.npu()
    # Signed head weights exercise signed score ordering, including below zero.
    query_data, query_scale = quantize_mxfp4_indexer(query.npu())
    q_ref, qs_ref = quantize_indexer(query)
    k_ref, ks_ref = quantize_indexer(key)
    q_decoded = dequantize_indexer(q_ref, qs_ref)
    k_decoded = dequantize_indexer(k_ref, ks_ref)
    coordinates = []
    for batch, length in enumerate(lengths):
        coordinates.extend(
            [[int(block_table[batch, token // page_size]), token % page_size] for token in range(length)]
        )
    slots = torch.tensor(coordinates, dtype=torch.int32, device="npu")
    data_cache = torch.zeros((2 * pages_per_batch, page_size, 1, 64), dtype=torch.uint8, device="npu")
    scale_cache = torch.ones((2 * pages_per_batch, page_size, 1, 4), dtype=torch.uint8, device="npu")
    folded = torch.zeros((2 * pages_per_batch, page_size // 8, 1, 544), dtype=torch.uint8, device="npu")
    write_mxfp4_indexer_cache(key.npu(), slots, data_cache, scale_cache)
    fold_indexer_cache_rows((data_cache, scale_cache), folded, slots)
    candidate_args = dict(candidate_topk_blocks=2048, candidate_block_size=8) if candidates_enabled else {}
    metadata_args = dict(
        cu_seqlens_q=cu,
        seqused_k=used,
        cmp_residual_k=rem,
        batch_size=2,
        max_seqlen_q=max(query_lengths),
        max_seqlen_k=max(lengths),
        num_heads_q=32,
        num_heads_k=1,
        head_dim=128,
        topk=512,
        mask_mode=3,
        cmp_ratio=ratio,
        layout_q="TND",
        layout_k="PA_BBND",
        **candidate_args,
    )
    metadata = ops.quant_lightning_indexer_metadata(**metadata_args)
    common = dict(
        cu_seqlens_q=cu,
        seqused_k=used,
        cmp_residual_k=rem,
        block_table=block_table.npu(),
        max_seqlen_q=max(query_lengths),
        mask_mode=3,
        cmp_ratio=ratio,
        layout_q="TND",
        layout_k="PA_BBND",
        return_value=True,
    )
    indices, values, candidates, candidate_lengths = ops.quant_lightning_indexer(
        query_data,
        data_cache,
        weights_npu,
        query_scale.unflatten(-1, (2, 2)),
        scale_cache.unflatten(-1, (2, 2)),
        512,
        1,
        metadata=metadata,
        **candidate_args,
        **common,
    )
    scores = []
    visible_lengths = []
    q_offset = k_offset = 0
    for batch, (qlen, klen) in enumerate(zip(query_lengths, lengths)):
        scores.extend(
            indexer_scores(
                q_decoded[q_offset : q_offset + qlen],
                k_decoded[k_offset : k_offset + klen],
                weights[q_offset : q_offset + qlen],
            )
        )
        visible_lengths.extend(
            [min(klen, (klen * ratio + residual[batch] - qlen + row + 1) // ratio) for row in range(qlen)]
        )
        q_offset += qlen
        k_offset += klen
    expected_tiles = []
    for batch, qlen in enumerate(query_lengths):
        for query_tile in range(math.ceil(qlen / 6)):
            last_row = int(cu_cpu[batch]) + min((query_tile + 1) * 6, qlen) - 1
            expected_tiles.extend(
                (batch, query_tile, tile) for tile in range(math.ceil(visible_lengths[last_row] / 256))
            )
    _assert_indexer_metadata(metadata, expected_tiles)
    if large_workload:
        assert len(expected_tiles) > 32
    for row, (score, visible) in enumerate(zip(scores, visible_lengths)):
        eligible = torch.arange(score.numel()) < visible
        assert_topk(indices[row], values[row], score, eligible)
        if candidates_enabled:
            assert_candidate_topk(candidates[row], candidate_lengths[row], score, visible)
    if large_workload:
        # Capture metadata and the LD barrier path. Negating packed Q changes
        # every score while preserving shapes and preallocated graph addresses.
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
            captured_metadata = ops.quant_lightning_indexer_metadata(**metadata_args)
            captured_indices, captured_values, captured_candidates, captured_lengths = ops.quant_lightning_indexer(
                query_data,
                data_cache,
                weights_npu,
                query_scale.unflatten(-1, (2, 2)),
                scale_cache.unflatten(-1, (2, 2)),
                512,
                1,
                metadata=captured_metadata,
                **candidate_args,
                **common,
            )
        for sign in (-1, 1):
            query_data.bitwise_xor_(0x88)
            graph.replay()
            _assert_indexer_metadata(captured_metadata, expected_tiles)
            q_offset = k_offset = 0
            for qlen, klen in zip(query_lengths, lengths):
                replay_scores = indexer_scores(
                    q_decoded[q_offset : q_offset + qlen] * sign,
                    k_decoded[k_offset : k_offset + klen],
                    weights[q_offset : q_offset + qlen],
                )
                for local_row, score in enumerate(replay_scores):
                    row = q_offset + local_row
                    visible = visible_lengths[row]
                    assert_topk(captured_indices[row], captured_values[row], score, torch.arange(klen) < visible)
                    assert_candidate_topk(captured_candidates[row], captured_lengths[row], score, visible)
                q_offset += qlen
                k_offset += klen
    source_metadata = SimpleNamespace(
        qli_metadata=metadata,
        query_start_loc=cu,
        cache_seq_lens=used,
        cmp_residual=rem,
        block_table=common["block_table"],
        num_reqs=2,
    )
    positions = torch.tensor(
        [
            lengths[batch] * ratio + residual[batch] - qlen + row
            for batch, qlen in enumerate(query_lengths)
            for row in range(qlen)
        ],
        dtype=torch.int64,
        device="npu",
    )
    output = torch.empty((total_queries, 512), dtype=torch.int32, device="npu")
    topk_lengths = torch.empty((total_queries, 1), dtype=torch.int32, device="npu")
    adapter_candidate_lengths = torch.empty_like(candidate_lengths)
    adapted, _ = run_a5_indexer(
        query.npu(),
        weights_npu,
        positions,
        (data_cache, scale_cache, folded),
        source_metadata,
        topk=512,
        compress_ratio=ratio,
        is_candidate_source=candidates_enabled,
        uses_candidate_filter=False,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
        candidates=candidates,
        candidate_lengths=adapter_candidate_lengths,
        topk_lengths=topk_lengths,
        indices_output=output,
    )
    assert adapted.data_ptr() == output.data_ptr()
    for row, visible in enumerate(visible_lengths):
        valid_count = min(512, visible)
        assert int(topk_lengths[row].cpu()) == valid_count
        expected = indices[row, 0].cpu()
        expected = torch.where(expected >= 0, expected, torch.iinfo(torch.int32).max).sort().values
        expected[expected == torch.iinfo(torch.int32).max] = -1
        torch.testing.assert_close(adapted[row].cpu(), expected, rtol=0, atol=0)
    if candidates_enabled:
        torch.testing.assert_close(adapter_candidate_lengths, candidate_lengths, rtol=0, atol=0)
    if not candidates_enabled:
        assert candidates.numel() == 0 and candidate_lengths.numel() == 0
        return

    # Consumer layer: first all candidates, then a restricted noncontiguous set
    # with an empty row. Invalid suffix IDs must never be dereferenced.
    for candidate_mode in ("all", "restricted", "empty"):
        restricted = candidate_mode != "all"
        selected_candidates = candidates.clone()
        selected_lengths = candidate_lengths.clone()
        allowed = []
        for row, (score, visible) in enumerate(zip(scores, visible_lengths)):
            count = int(candidate_lengths[row].cpu())
            blocks = candidates[row, 0, :count].cpu().tolist()
            if restricted:
                blocks = sorted(blocks)[::2][::-1] if row else []
                if candidate_mode == "empty":
                    blocks = []
                selected_candidates[row].fill_(2**30)
                if blocks:
                    selected_candidates[row, 0, : len(blocks)].copy_(
                        torch.tensor(blocks, dtype=torch.int32, device="npu")
                    )
                selected_lengths[row, 0] = len(blocks)
            allowed.append(
                (torch.arange(score.numel()) < visible)
                & torch.isin(torch.arange(score.numel()) // 8, torch.tensor(blocks, dtype=torch.int64))
            )
        sparse_metadata_args = dict(
            candidate_block_length=selected_lengths,
            cu_seqlens_q=cu,
            seqused_k=used,
            cmp_residual_k=rem,
            batch_size=2,
            max_seqlen_q=max(query_lengths),
            max_seqlen_k=max(lengths),
            num_heads_q=32,
            num_heads_k=1,
            head_dim=128,
            topk=512,
            quant_mode=1,
            candidate_block_size=8,
            mask_mode=3,
            cmp_ratio=ratio,
            layout_q="TND",
            layout_k="PA_BBND",
        )
        sparse_metadata = ops.quant_sparse_lightning_indexer_metadata(**sparse_metadata_args)
        lengths_cpu = selected_lengths.cpu().flatten().tolist()
        sparse_tiles = []
        for batch, qlen in enumerate(query_lengths):
            for row in range(qlen):
                length = lengths_cpu[int(cu_cpu[batch]) + row]
                sparse_tiles.extend((batch, row, tile) for tile in range(math.ceil(length * 8 / 512)))
        _assert_indexer_metadata(sparse_metadata, sparse_tiles)
        sparse_indices, sparse_values = ops.quant_sparse_lightning_indexer(
            query_data,
            folded.squeeze(2),
            weights_npu,
            query_scale.unflatten(-1, (2, 2)),
            selected_candidates,
            selected_lengths,
            512,
            1,
            8,
            metadata=sparse_metadata,
            **common,
        )
        for row, score in enumerate(scores):
            assert_topk(sparse_indices[row], sparse_values[row], score, allowed[row])
            if not restricted:
                torch.testing.assert_close(
                    sparse_values[row].cpu().sort().values, values[row].cpu().sort().values, rtol=0, atol=0
                )
        if large_workload and candidate_mode == "all":
            sparse_graph = torch.npu.NPUGraph()
            with torch.npu.graph(sparse_graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
                captured_sparse_metadata = ops.quant_sparse_lightning_indexer_metadata(**sparse_metadata_args)
                captured_sparse_indices, captured_sparse_values = ops.quant_sparse_lightning_indexer(
                    query_data,
                    folded.squeeze(2),
                    weights_npu,
                    query_scale.unflatten(-1, (2, 2)),
                    selected_candidates,
                    selected_lengths,
                    512,
                    1,
                    8,
                    metadata=captured_sparse_metadata,
                    **common,
                )
            for sign in (-1, 1):
                query_data.bitwise_xor_(0x88)
                sparse_graph.replay()
                _assert_indexer_metadata(captured_sparse_metadata, sparse_tiles)
                q_offset = k_offset = 0
                for qlen, klen in zip(query_lengths, lengths):
                    replay_scores = indexer_scores(
                        q_decoded[q_offset : q_offset + qlen] * sign,
                        k_decoded[k_offset : k_offset + klen],
                        weights[q_offset : q_offset + qlen],
                    )
                    for local_row, score in enumerate(replay_scores):
                        row = q_offset + local_row
                        assert_topk(captured_sparse_indices[row], captured_sparse_values[row], score, allowed[row])
                    q_offset += qlen
                    k_offset += klen
        adapted, _ = run_a5_indexer(
            query.npu(),
            weights_npu,
            positions,
            (data_cache, scale_cache, folded),
            source_metadata,
            topk=512,
            compress_ratio=ratio,
            is_candidate_source=False,
            uses_candidate_filter=True,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
            candidates=selected_candidates,
            candidate_lengths=selected_lengths,
            topk_lengths=topk_lengths,
            indices_output=output,
        )
        for row in range(total_queries):
            expected = sparse_indices[row, 0].cpu()
            expected = torch.where(expected >= 0, expected, torch.iinfo(torch.int32).max).sort().values
            expected[expected == torch.iinfo(torch.int32).max] = -1
            torch.testing.assert_close(adapted[row].cpu(), expected, rtol=0, atol=0)
            assert int(topk_lengths[row].cpu()) == min(512, int(allowed[row].sum()))


@pytest.mark.parametrize("ratio", [1, 2])
@torch.inference_mode()
def test_sparse_indexer_without_dense_predecessor(ratio):
    """QSLI consumes independently chosen candidates, including an empty row."""
    torch.manual_seed(414)
    lengths, query_lengths = [513, 131], [2, 1]
    residual = [0, 1] if ratio == 2 else [0, 0]
    cu = torch.tensor([0, 2, 3], dtype=torch.int32, device="npu")
    used = torch.tensor(lengths, dtype=torch.int32, device="npu")
    rem = torch.tensor(residual, dtype=torch.int32, device="npu") if ratio != 1 else None
    table = torch.tensor([[6, 2, 8, 1, 4], [0, 7, 3, 5, 9]], dtype=torch.int32)
    query = torch.randn(3, 32, 128).to(torch.bfloat16)
    key = torch.randn(sum(lengths), 128).to(torch.bfloat16)
    weights = torch.randn(3, 32) / 32
    q_ref, qs_ref = quantize_indexer(query)
    k_ref, ks_ref = quantize_indexer(key)
    query_data, query_scale = quantize_mxfp4_indexer(query.npu())
    coordinates = [
        [int(table[batch, token // 128]), token % 128]
        for batch, length in enumerate(lengths)
        for token in range(length)
    ]
    slots = torch.tensor(coordinates, dtype=torch.int32, device="npu")
    data_cache = torch.zeros((10, 128, 1, 64), dtype=torch.uint8, device="npu")
    scale_cache = torch.ones((10, 128, 1, 4), dtype=torch.uint8, device="npu")
    folded = torch.zeros((10, 16, 1, 544), dtype=torch.uint8, device="npu")
    write_mxfp4_indexer_cache(key.npu(), slots, data_cache, scale_cache)
    fold_indexer_cache_rows((data_cache, scale_cache), folded, slots)
    blocks = [list(range(64, -1, -2)), [], [16, 3, 1]]
    candidates = torch.full((3, 1, 2048), 2**30, dtype=torch.int32, device="npu")
    candidate_lengths = torch.tensor([[len(row)] for row in blocks], dtype=torch.int32, device="npu")
    for row, selected in enumerate(blocks):
        if selected:
            candidates[row, 0, : len(selected)] = torch.tensor(selected, dtype=torch.int32, device="npu")
    common = dict(
        cu_seqlens_q=cu,
        seqused_k=used,
        cmp_residual_k=rem,
        max_seqlen_q=2,
        mask_mode=3,
        cmp_ratio=ratio,
        layout_q="TND",
        layout_k="PA_BBND",
    )
    metadata = ops.quant_sparse_lightning_indexer_metadata(
        candidate_lengths,
        batch_size=2,
        max_seqlen_k=max(lengths),
        num_heads_q=32,
        num_heads_k=1,
        head_dim=128,
        topk=512,
        quant_mode=1,
        candidate_block_size=8,
        **common,
    )
    _assert_indexer_metadata(metadata, [(0, 0, 0), (1, 0, 0)])
    indices, values = ops.quant_sparse_lightning_indexer(
        query_data,
        folded.squeeze(2),
        weights.npu(),
        query_scale.unflatten(-1, (2, 2)),
        candidates,
        candidate_lengths,
        512,
        1,
        8,
        metadata=metadata,
        block_table=table.npu(),
        return_value=True,
        **common,
    )
    decoded_q = dequantize_indexer(q_ref, qs_ref)
    decoded_k = dequantize_indexer(k_ref, ks_ref)
    q_offset = k_offset = 0
    for batch, (qlen, klen) in enumerate(zip(query_lengths, lengths)):
        scores = indexer_scores(
            decoded_q[q_offset : q_offset + qlen],
            decoded_k[k_offset : k_offset + klen],
            weights[q_offset : q_offset + qlen],
        )
        for row in range(qlen):
            visible = min(klen, (klen * ratio + residual[batch] - qlen + row + 1) // ratio)
            token_ids = torch.arange(klen)
            eligible = (token_ids < visible) & torch.isin(
                token_ids // 8, torch.tensor(blocks[q_offset + row], dtype=torch.int64)
            )
            assert_topk(indices[q_offset + row], values[q_offset + row], scores[row], eligible)
        q_offset += qlen
        k_offset += klen


def _assert_attention_metadata(metadata, cu):
    # Verify the ABI by recovering the assigned global row intervals, rather
    # than copying the metadata kernel's partition algorithm.
    words = metadata.cpu()
    covered = []
    intervals = []
    for record in words[: 36 * 9].reshape(36, 9):
        if int(record[0]) == 0:
            assert bool((record == 0).all())
            continue
        begin = int(cu[int(record[1])]) + int(record[2])
        end = int(cu[int(record[4])]) + int(record[5])
        assert 0 <= begin < end <= int(cu[-1])
        assert bool((record[[3, 6, 7, 8]] == 0).all())
        covered.extend(range(begin, end))
        intervals.append(end - begin)
    assert sorted(covered) == list(range(int(cu[-1])))
    assert max(intervals) - min(intervals) <= 1
    assert bool((words[324:901] == 0).all())


@pytest.mark.parametrize("has_compressed", [False, True])
@pytest.mark.parametrize("use_cache_writer", [False, True])
@torch.inference_mode()
def test_mixed_attention_metadata_and_window(has_compressed, use_cache_writer):
    torch.manual_seed(413)
    # First request starts at zero; second spans a physical page boundary.
    query = torch.randn(4, 64, 512).to(torch.bfloat16)
    ori = torch.randn(6, 128, 1, 512).to(torch.bfloat16)
    cmp = torch.randn(4, 128, 1, 512).to(torch.bfloat16)
    packed_ori, decoded_ori = pack_attention_cache(ori, fp4=False)
    packed_cmp, decoded_cmp = pack_attention_cache(cmp, fp4=True)
    if use_cache_writer:
        load_custom_op_library()
        for original, packed, fp4 in ((ori, packed_ori, False), (cmp, packed_cmp, True)):
            cache = torch.empty_like(packed, device="npu")
            slots = torch.arange(original.shape[0] * 128, dtype=torch.int64, device="npu")
            write_attention_cache(
                cache,
                slots,
                original.flatten(0, 2).npu(),
                kind="cmp" if fp4 else "win",
            )
            packed.copy_(cache.cpu())
            decoded = decode_attention_cache(packed, fp4)
            # Verify the producer's row layout against its input independently
            # of attention, including FP4's finite-range clipping below 8*scale.
            group = 16 if fp4 else 32
            scales = packed[..., 256 if fp4 else 512 :].contiguous().view(torch.bfloat16).float()
            error = (decoded - original.float()).abs().unflatten(-1, (512 // group, group))
            assert bool((scales > 0).all())
            assert bool((error <= scales.unsqueeze(-1) * (2.0 if fp4 else 16.0)).all())
            if fp4:
                decoded_cmp = decoded.to(torch.bfloat16).float()
            else:
                decoded_ori = decoded.to(torch.bfloat16).float()
    ori_table = torch.tensor([[4, 1], [3, 0]], dtype=torch.int32)
    cmp_table = torch.tensor([[2], [0]], dtype=torch.int32)
    cu_cpu = torch.tensor([0, 2, 4], dtype=torch.int32)
    cu = cu_cpu.npu()
    positions = torch.tensor([0, 1, 128, 129], dtype=torch.int64, device="npu")
    window_indices, window_lengths = build_window_indices_triton(positions, 128)
    cmp_indices_cpu = torch.full((4, 1, 512), -1, dtype=torch.int32)
    cmp_lengths_cpu = torch.tensor([[0], [1], [9], [17]], dtype=torch.int32)
    for row, length in enumerate(cmp_lengths_cpu.flatten().tolist()):
        # Unsorted sparse positions detect accidental contiguous reads.
        cmp_indices_cpu[row, 0, :length] = torch.arange(length - 1, -1, -1, dtype=torch.int32) * 3
    cmp_indices, cmp_lengths = cmp_indices_cpu.npu(), cmp_lengths_cpu.npu()
    sinks = torch.linspace(-1, 2, 64)
    metadata = ops.mixed_quant_sparse_flash_mla_metadata(
        window_lengths,
        cmp_lengths,
        cu_seqlens_q=cu,
        num_heads_q=64,
        num_heads_kv=1,
        head_dim=512,
        quant_mode=1,
        has_ori_kv=True,
        has_cmp_kv=has_compressed,
    )
    _assert_attention_metadata(metadata, cu_cpu)
    call = dict(
        ori_kv=packed_ori.npu(),
        ori_sparse_indices=window_indices,
        ori_block_table=ori_table.npu(),
        cu_seqlens_q=cu,
        ori_topk_length=window_lengths,
        sinks=sinks.npu(),
        metadata=metadata,
        quant_mode=1,
        softmax_scale=512**-0.5,
        return_softmax_lse=True,
    )
    if has_compressed:
        call.update(
            cmp_kv=packed_cmp.npu(),
            cmp_sparse_indices=cmp_indices,
            cmp_block_table=cmp_table.npu(),
            cmp_topk_length=cmp_lengths,
        )
    query_npu = query.npu()
    actual, lse = ops.mixed_quant_sparse_flash_mla(query_npu, **call)

    def reference(q):
        expected, expected_lse = [], []
        for row, position in enumerate(positions.cpu().tolist()):
            batch = 0 if row < 2 else 1
            rows = [
                decoded_ori[ori_table[batch, token // 128], token % 128, 0]
                for token in range(max(0, position - 127), position + 1)
            ]
            if has_compressed:
                rows.extend(
                    [
                        decoded_cmp[cmp_table[batch, 0], token, 0]
                        for token in cmp_indices_cpu[row, 0, : int(cmp_lengths_cpu[row, 0])].tolist()
                    ]
                )
            kv = torch.stack(rows)
            scores = q[row].float() @ kv.T / math.sqrt(512)
            all_scores = torch.cat((scores, sinks[:, None]), dim=-1)
            probabilities = torch.softmax(all_scores, dim=-1)[:, :-1]
            expected.append(probabilities @ kv)
            expected_lse.append(torch.logsumexp(all_scores, dim=-1))
        return torch.stack(expected), torch.stack(expected_lse).unsqueeze(0)

    expected, expected_lse = reference(query)
    torch.testing.assert_close(actual.cpu().float(), expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0.004, atol=0.004)
    if use_cache_writer:
        layer_metadata = SimpleNamespace(
            positions=positions,
            swa=SimpleNamespace(
                ori_sparse_indices=window_indices,
                ori_topk_length=window_lengths,
                smla_metadata=metadata,
                block_table=call["ori_block_table"],
                query_start_loc=cu,
                seq_lens=torch.tensor([2, 130], dtype=torch.int32, device="npu"),
            ),
            attention=SimpleNamespace(
                block_table=cmp_table.npu(), cache_seq_lens=torch.tensor([1, 49], dtype=torch.int32, device="npu")
            ),
        )
        adapted = qsmla(
            query_npu,
            call["ori_kv"],
            call.get("cmp_kv"),
            layer_metadata,
            cmp_indices.squeeze(1),
            window_size=128,
            sinks=sinks.npu(),
            softmax_scale=512**-0.5,
            compressed_lengths=cmp_lengths,
        )
        torch.testing.assert_close(adapted.cpu().float(), expected, rtol=0.02, atol=0.02)
    # Replay uses the registered dispatcher and the same device metadata; a
    # changed query must affect the result without recompiling host wrappers.
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        captured, captured_lse = ops.mixed_quant_sparse_flash_mla(query_npu, **call)
    for factor in (0.5, -1.0):
        changed = (query.float() * factor).to(torch.bfloat16)
        query_npu.copy_(changed.npu())
        graph.replay()
        expected, expected_lse = reference(changed)
        torch.testing.assert_close(captured.cpu().float(), expected, rtol=0.02, atol=0.02)
        torch.testing.assert_close(captured_lse.cpu(), expected_lse, rtol=0.004, atol=0.004)
