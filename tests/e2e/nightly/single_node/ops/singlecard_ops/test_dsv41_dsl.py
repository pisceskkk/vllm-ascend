# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the in-tree DSL kernels and their metadata on an Ascend A5.

TopK values/indices use an independent MXFP4 decode and ordered BF16 score
reference. Exact cutoff ties may select any tied position; positions strictly
above the cutoff must all be present. Attention uses FP32 softmax with an
explicit sink and independently decoded cache rows (rtol=0.02, atol=0.02).
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from tests.e2e.nightly.single_node.ops.singlecard_ops.dsv41_reference import (
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


def _assert_indexer_metadata(metadata, expected_tiles):
    # The LI ABI contains lexicographic [begin, end) task boundaries.
    # Enumerate logical work independently and require single ownership.
    words = metadata.cpu()
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


@pytest.mark.parametrize("ratio,candidates_enabled", [(1, False), (1, True), (2, True)])
@torch.inference_mode()
def test_indexer_metadata_cache_and_candidates(ratio, candidates_enabled):
    torch.manual_seed(412)
    lengths = [769, 145]
    query_lengths = [4, 3]
    residual = [0, 1] if ratio == 2 else [0, 0]
    cu_cpu = torch.tensor([0, 4, 7], dtype=torch.int32)
    cu, used = cu_cpu.npu(), torch.tensor(lengths, dtype=torch.int32, device="npu")
    rem = torch.tensor(residual, dtype=torch.int32, device="npu") if ratio != 1 else None
    page_size = 128
    pages_per_batch = math.ceil(max(lengths) / page_size)
    block_table = torch.randperm(2 * pages_per_batch).reshape(2, pages_per_batch).int()
    query = torch.randn(sum(query_lengths), 32, 128).to(torch.bfloat16)
    key = torch.randn(sum(lengths), 128).to(torch.bfloat16)
    weights = torch.randn(sum(query_lengths), 32) / 32
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
    metadata = ops.quant_lightning_indexer_metadata(
        cu,
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
        weights.npu(),
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
    for row, (score, visible) in enumerate(zip(scores, visible_lengths)):
        eligible = torch.arange(score.numel()) < visible
        assert_topk(indices[row], values[row], score, eligible)
        if candidates_enabled:
            expected_blocks = math.ceil(visible / 8)
            assert int(candidate_lengths[row].cpu()) == expected_blocks
            selected_blocks = candidates[row, 0, :expected_blocks].cpu().sort().values
            torch.testing.assert_close(
                selected_blocks, torch.arange(expected_blocks, dtype=torch.int32), rtol=0, atol=0
            )
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
    output = torch.empty((7, 512), dtype=torch.int32, device="npu")
    topk_lengths = torch.empty((7, 1), dtype=torch.int32, device="npu")
    adapter_candidate_lengths = torch.empty_like(candidate_lengths)
    adapted, _ = run_a5_indexer(
        query.npu(),
        weights.npu(),
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
            blocks = list(range(math.ceil(visible / 8)))
            if restricted:
                blocks = blocks[::2][::-1] if row else []
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
        sparse_metadata = ops.quant_sparse_lightning_indexer_metadata(
            selected_lengths,
            cu,
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
            weights.npu(),
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
        adapted, _ = run_a5_indexer(
            query.npu(),
            weights.npu(),
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
        for row in range(7):
            expected = sparse_indices[row, 0].cpu()
            expected = torch.where(expected >= 0, expected, torch.iinfo(torch.int32).max).sort().values
            expected[expected == torch.iinfo(torch.int32).max] = -1
            torch.testing.assert_close(adapted[row].cpu(), expected, rtol=0, atol=0)
            assert int(topk_lengths[row].cpu()) == min(512, int(allowed[row].sum()))


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
