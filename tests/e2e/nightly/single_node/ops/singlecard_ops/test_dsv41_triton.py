# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from tests.e2e.nightly.single_node.ops.singlecard_ops.dsv41_reference import quantize_indexer
from vllm_ascend.ops.triton.a5_slot_mapping import build_a5_slot_mapping, build_a5_slot_mapping_batch
from vllm_ascend.ops.triton.build_window_indices import build_window_indices_triton
from vllm_ascend.ops.triton.c2_ring_metadata import build_c2_ring_metadata
from vllm_ascend.ops.triton.fold_indexer_cache import fold_indexer_cache_rows
from vllm_ascend.ops.triton.prepare_indexer_indices import prepare_indexer_indices
from vllm_ascend.ops.triton.quantize_mxfp4_indexer import quantize_mxfp4_indexer, write_mxfp4_indexer_cache
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.fixture(scope="module", autouse=True)
def initialize_device_properties():
    init_device_properties_triton()


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("skip", [False, True])
@torch.inference_mode()
def test_slot_mapping_padded_and_int64(ratio, skip):
    positions = torch.tensor([0, 1, 4, 5, 6, 7, 8, 9], dtype=torch.int64)
    slots = torch.tensor([[15, 16, -1, 2**33 + 1, 37, 38, 77, 78], [81, 82, 83, 84, 85, 86, 87, 88]], dtype=torch.int64)
    cu = torch.tensor([0, 2, 6, 8], dtype=torch.int32)
    expected = []
    for group in (1, 0):
        flat = []
        for t, slot in enumerate(slots[group].tolist()):
            live = t < 5 and slot >= 0
            if ratio == 2:
                live &= slot % 2 == 1 and positions[t] % 2 == 1 and not skip
            flat.append(slot // ratio if live else -1)
        expected.append(flat)
    flat_ref = torch.tensor(expected, dtype=torch.int64)
    coords_ref = torch.stack((flat_ref // 16, flat_ref % 16), -1)
    coords_ref[flat_ref < 0] = -1
    coords = torch.empty((2, 8, 2), dtype=torch.int64, device="npu")
    flat = torch.empty((2, 8), dtype=torch.int64, device="npu")
    build_a5_slot_mapping_batch(
        slots.npu(),
        torch.tensor([1, 0], device="npu"),
        positions.npu(),
        cu.npu(),
        8,
        2,
        5,
        16,
        ratio,
        skip_update=skip,
        coordinates_output=coords,
        flat_output=flat,
    )
    torch.testing.assert_close(flat.cpu(), flat_ref, rtol=0, atol=0)
    torch.testing.assert_close(coords.cpu(), coords_ref, rtol=0, atol=0)
    for output_row, group in enumerate((1, 0)):
        build_a5_slot_mapping(
            slots[group].npu(),
            positions.npu(),
            cu.npu(),
            8,
            2,
            5,
            16,
            ratio,
            skip_update=skip,
            coordinates_output=coords[output_row],
            flat_output=flat[output_row],
        )
    torch.testing.assert_close(flat.cpu(), flat_ref, rtol=0, atol=0)
    torch.testing.assert_close(coords.cpu(), coords_ref, rtol=0, atol=0)


@pytest.mark.parametrize("window", [1, 7, 128])
@torch.inference_mode()
def test_window_indices_reuses_output(window):
    positions = torch.tensor([-1, 0, 1, window - 1, window, 255], dtype=torch.int64, device="npu")
    indices = torch.full((6, 1, window), 42, dtype=torch.int32, device="npu")
    lengths = torch.full((6, 1), 42, dtype=torch.int32, device="npu")
    for offset in (0, 3):
        current = positions + offset
        actual, actual_lengths = build_window_indices_triton(
            current, window, indices_output=indices, lengths_output=lengths
        )
        reference = torch.full((6, 1, window), -1, dtype=torch.int32)
        lengths_ref = []
        for row, position in enumerate(current.cpu().tolist()):
            start = max(0, position - window + 1)
            values = list(range(start, position + 1))
            reference[row, 0, : len(values)] = torch.tensor(values, dtype=torch.int32)
            lengths_ref.append(len(values))
        assert actual.data_ptr() == indices.data_ptr()
        assert actual_lengths.data_ptr() == lengths.data_ptr()
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
        torch.testing.assert_close(
            lengths.cpu().flatten(), torch.tensor(lengths_ref, dtype=torch.int32), rtol=0, atol=0
        )


@pytest.mark.parametrize("skip", [False, True])
@torch.inference_mode()
def test_ring_controls_and_rope_request_reuse(skip):
    cu = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
    seq = torch.tensor([8, 3, 12], dtype=torch.int32)
    positions = torch.tensor([5, 6, 7, 1, 2, 9, 10, 11], dtype=torch.int64)
    table = torch.tensor([[7, 8], [3, 4], [11, 12]], dtype=torch.int32)
    cos = torch.arange(16 * 64).reshape(16, 64).to(torch.bfloat16)
    sin = -cos - 2
    ring = torch.empty((5, 3), dtype=torch.int32, device="npu")
    complete = torch.empty(8, dtype=torch.bool, device="npu")
    source = torch.empty(8, dtype=torch.int64, device="npu")
    out_cos = torch.empty((8, 1, 1, 64), dtype=torch.bfloat16, device="npu")
    out_sin = torch.empty_like(out_cos)
    for live_reqs, live_tokens in ((3, 8), (2, 4)):
        # The second invocation retires the third request, then reuses outputs.
        build_c2_ring_metadata(
            cu.npu(),
            seq.npu(),
            positions.npu(),
            table.npu(),
            cos.npu(),
            sin.npu(),
            3,
            8,
            live_reqs,
            live_tokens,
            skip_update=skip,
            ring_metadata_output=ring,
            complete_mask_output=complete,
            source_positions_output=source,
            cos_output=out_cos,
            sin_output=out_sin,
        )
        used = [
            max(min(int(cu[r + 1]), live_tokens) - int(cu[r]), 0) if r < live_reqs and not skip else 0 for r in range(3)
        ]
        reference = torch.stack(
            (
                seq - cu.diff(),
                torch.tensor(used, dtype=torch.int32),
                cu[:-1],
                cu[:-1],
                torch.tensor([int(table[r, 0]) if used[r] else 0 for r in range(3)]),
            )
        )
        reference = reference.int()
        mask_ref = (torch.arange(8) < min(int(cu[live_reqs]), live_tokens)) & (positions % 2 == 1) & (not skip)
        source_ref = torch.where(mask_ref, positions - 1, 0)
        torch.testing.assert_close(ring.cpu(), reference, rtol=0, atol=0)
        torch.testing.assert_close(complete.cpu(), mask_ref, rtol=0, atol=0)
        torch.testing.assert_close(source.cpu(), source_ref, rtol=0, atol=0)
        torch.testing.assert_close(out_cos.cpu().reshape(8, 64), cos[source_ref], rtol=0, atol=0)
        torch.testing.assert_close(out_sin.cpu().reshape(8, 64), sin[source_ref], rtol=0, atol=0)


@torch.inference_mode()
def test_mxfp4_bytes_rounding_and_strided_cache_fold():
    torch.manual_seed(410)
    # Fixed group maximum keeps the scale at one and exposes both sides of
    # every midpoint, negative tiny values, zero, and saturation at six.
    thresholds = torch.tensor([0, -0.0, -0.01, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6])
    x = torch.randn(19, 128).to(torch.bfloat16)
    boundary = torch.cat((thresholds, thresholds[3:10] - 0.015625, thresholds[3:10] + 0.015625))
    x[0, : boundary.numel()] = boundary
    x[0, 31] = 6
    x[1].zero_()
    x[2] *= 128
    x[3] *= 2**-12
    data_ref, scale_ref = quantize_indexer(x)
    data, scale = quantize_mxfp4_indexer(x.npu())
    torch.testing.assert_close(data.cpu(), data_ref, rtol=0, atol=0)
    torch.testing.assert_close(scale.cpu(), scale_ref, rtol=0, atol=0)
    coordinates = torch.tensor([[2, r] for r in range(16)] + [[0, 7], [-1, -1], [1, -1]], dtype=torch.int32)
    # Page gaps are legal; row payloads remain contiguous for DSL consumers.
    data_storage = torch.full((6, 16, 1, 64), 0xA5, dtype=torch.uint8, device="npu")
    scale_storage = torch.full((6, 16, 1, 4), 0x5A, dtype=torch.uint8, device="npu")
    data_cache, scale_cache = data_storage[::2], scale_storage[::2]
    folded = torch.full((3, 2, 1, 544), 0xCC, dtype=torch.uint8, device="npu")
    write_mxfp4_indexer_cache(x.npu(), coordinates.npu(), data_cache, scale_cache)
    fold_indexer_cache_rows((data_cache, scale_cache), folded, coordinates.npu())
    expected_data, expected_scale = data_storage.cpu(), scale_storage.cpu()
    expected_data.fill_(0xA5)
    expected_scale.fill_(0x5A)
    expected_folded = torch.full_like(folded.cpu(), 0xCC)
    for token, (page, row) in enumerate(coordinates.tolist()):
        if page < 0 or row < 0:
            continue
        expected_data[page * 2, row, 0] = data_ref[token]
        expected_scale[page * 2, row, 0] = scale_ref[token]
        expected_folded[page, row // 8, 0, (row % 8) * 64 : (row % 8 + 1) * 64] = data_ref[token]
        expected_folded[page, row // 8, 0, 512 + (row % 8) * 4 : 512 + (row % 8 + 1) * 4] = scale_ref[token]
    torch.testing.assert_close(data_storage.cpu(), expected_data, rtol=0, atol=0)
    torch.testing.assert_close(scale_storage.cpu(), expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(folded.cpu(), expected_folded, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [3, 1025])
@torch.inference_mode()
def test_prepare_indexer_indices_and_lengths(tokens):
    torch.manual_seed(411)
    selected = torch.randint(-3, 2048, (tokens, 512), dtype=torch.int32)
    positions = torch.arange(tokens, dtype=torch.int64) * 3
    valid = (selected >= 0) & (selected < ((positions + 1) // 2).unsqueeze(-1))
    expected = torch.where(valid, selected, torch.iinfo(torch.int32).max).sort(-1).values
    expected[expected == torch.iinfo(torch.int32).max] = -1
    output = torch.empty_like(selected, device="npu")
    lengths = torch.empty((tokens, 1), dtype=torch.int32, device="npu")
    result = prepare_indexer_indices(selected.npu(), positions.npu(), 2, indices_output=output, lengths_output=lengths)
    assert result.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(lengths.cpu(), valid.sum(-1, keepdim=True).int(), rtol=0, atol=0)


@torch.inference_mode()
def test_indexer_writer_and_fold_address_above_int32():
    # Only the two small logical pages are touched. The second page starts
    # above INT32_MAX, so a narrowed byte offset would corrupt another address.
    data = torch.empty_strided((2, 16, 1, 64), (2**31 + 65536, 64, 64, 1), dtype=torch.uint8, device="npu")
    data.fill_(0xA5)
    scales = torch.full((2, 16, 1, 4), 0x5A, dtype=torch.uint8, device="npu")
    folded = torch.full((2, 2, 1, 544), 0xCC, dtype=torch.uint8, device="npu")
    x = torch.linspace(-6, 6, 128).reshape(1, 128).to(torch.bfloat16)
    slots = torch.tensor([[1, 15]], dtype=torch.int64, device="npu")
    write_mxfp4_indexer_cache(x.npu(), slots, data, scales)
    fold_indexer_cache_rows((data, scales), folded, slots.int())
    expected_data, expected_scale = quantize_indexer(x)
    torch.testing.assert_close(data[1, 15, 0].cpu(), expected_data[0], rtol=0, atol=0)
    torch.testing.assert_close(scales[1, 15, 0].cpu(), expected_scale[0], rtol=0, atol=0)
    torch.testing.assert_close(folded[1, 1, 0, 448:512].cpu(), expected_data[0], rtol=0, atol=0)
    torch.testing.assert_close(folded[1, 1, 0, 540:544].cpu(), expected_scale[0], rtol=0, atol=0)
    assert bool((data[0].cpu() == 0xA5).all())
    assert bool((data[1, :15].cpu() == 0xA5).all())
