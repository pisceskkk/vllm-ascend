# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.utils import load_custom_op_library


@pytest.fixture(scope="module", autouse=True)
def _load_ops():
    load_custom_op_library()


def _weights(hidden_size, head_dim, dtype):
    generator = torch.Generator().manual_seed(730)
    return tuple(
        (torch.randn(head_dim, hidden_size, generator=generator) / hidden_size**0.5).to(dtype) for _ in range(2)
    )


def _reference(x, wkv, wgate, state, blocks, offsets, starts, lengths, ratio):
    """FP32 token-by-token ring model, independent of the tiled implementation."""
    state = state.clone()
    head_dim = wkv.shape[0]
    values = x.float() @ wkv.float().T
    scores = x.float() @ wgate.float().T
    rows = []
    written = torch.zeros(state.shape[:2], dtype=torch.bool)
    for block, offset, start, length in zip(blocks, offsets, starts, lengths):
        # Capture history before the operator overwrites its ring slots.
        history = state[block].clone()
        for token in range(length):
            position = start + token
            slot = position % state.shape[1]
            history[slot, :head_dim] = values[offset + token]
            history[slot, head_dim:] = scores[offset + token]
            state[block, slot] = history[slot]
            written[block, slot] = True
            if (position + 1) % ratio == 0:
                slots = torch.arange(position + 1 - ratio, position + 1) % state.shape[1]
                group = history[slots]
                weights = group[:, head_dim:].softmax(dim=0)
                rows.append((weights * group[:, :head_dim]).sum(dim=0))
    output = torch.stack(rows) if rows else torch.empty(0, head_dim)
    return output.to(x.dtype), state, written


def _run(x, wkv, wgate, state, blocks, offsets, starts, lengths, ratio):
    def control(values):
        return torch.tensor(values, dtype=torch.int32, device=x.device)

    return torch.ops._C_ascend.compressor_v2(
        x,
        wkv,
        wgate,
        state,
        control(blocks),
        control([*offsets, x.shape[0]]),
        control(lengths),
        control(starts),
        ratio,
    )


def _assert_result(actual, expected, actual_state, expected_state, initial_state, written):
    # Compare only completed rows; unused packed output capacity is unspecified.
    atol = 8e-3 if actual.dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(actual[: expected.shape[0]].cpu(), expected, rtol=atol, atol=atol)
    torch.testing.assert_close(actual_state.cpu(), expected_state, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(actual_state.cpu()[~written], initial_state[~written], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [128, 512])
@pytest.mark.parametrize("ratio", [2, 4])
def test_compressor_v2_packed_output_and_ring_state(dtype, head_dim, ratio):
    torch.manual_seed(731)
    hidden_size = 5120
    lengths = [7, 0, 5, 9]
    starts = [61, 0, 3, 128]
    offsets = [0, 8, 8, 14]
    blocks = [3, 1, 4, 2]
    x = torch.randn(24, hidden_size).to(dtype)
    wkv, wgate = _weights(hidden_size, head_dim, dtype)
    # A gap after each physical block detects ignored stride and stray writes.
    backing = torch.randn(6, 65, 2 * head_dim, dtype=torch.float32)
    state = backing[:, :64]
    expected, expected_state, written = _reference(x, wkv, wgate, state, blocks, offsets, starts, lengths, ratio)
    device_backing = backing.npu()
    device_state = device_backing[:, :64]
    actual = _run(x.npu(), wkv.npu(), wgate.npu(), device_state, blocks, offsets, starts, lengths, ratio)
    _assert_result(actual, expected, device_state, expected_state, state, written)
    torch.testing.assert_close(device_backing[:, 64].cpu(), backing[:, 64], rtol=0, atol=0)


@pytest.mark.parametrize("chunks", [[3, 4, 1, 7, 5, 6, 7], [1] * 33])
def test_compressor_v2_incremental_matches_full_projection_and_compression(chunks):
    """Cross odd chunk boundaries and wrap the ring several times."""
    torch.manual_seed(732)
    hidden_size, head_dim, ratio = 5120, 128, 2
    x = torch.randn(sum(chunks), hidden_size).to(torch.bfloat16)
    wkv, wgate = _weights(hidden_size, head_dim, x.dtype)
    x_device, wkv_device, wgate_device = x.npu(), wkv.npu(), wgate.npu()
    state = torch.zeros(3, 16, 2 * head_dim)
    state_device = state.npu()
    outputs = []
    offset = 0
    for length in chunks:
        chunk = x[offset : offset + length]
        expected, next_state, written = _reference(chunk, wkv, wgate, state, [2], [0], [offset], [length], ratio)
        # Untouched cells must retain the device's preceding result bit-for-bit;
        # that result may differ slightly from the independent CPU projection.
        previous_device_state = state_device.cpu()
        actual = _run(
            x_device[offset : offset + length],
            wkv_device,
            wgate_device,
            state_device,
            [2],
            [0],
            [offset],
            [length],
            ratio,
        )
        _assert_result(actual, expected, state_device, next_state, previous_device_state, written)
        outputs.append(actual[: expected.shape[0]].cpu())
        state = next_state
        offset += length

    # A separate whole-sequence formula avoids using the ring reference twice.
    values = x.float() @ wkv.float().T
    scores = x.float() @ wgate.float().T
    complete = (x.shape[0] // ratio) * ratio
    weights = scores[:complete].reshape(-1, ratio, head_dim).softmax(dim=1)
    expected_full = (values[:complete].reshape(-1, ratio, head_dim) * weights).sum(dim=1)
    torch.testing.assert_close(torch.cat(outputs), expected_full.to(x.dtype), rtol=8e-3, atol=8e-3)


def test_compressor_v2_graph_replay_updates_inputs_and_reused_request_state():
    torch.manual_seed(733)
    hidden_size, head_dim, ratio = 5120, 128, 2
    wkv, wgate = _weights(hidden_size, head_dim, torch.bfloat16)
    device_wkv, device_wgate = wkv.npu(), wgate.npu()
    x = torch.randn(4, hidden_size).to(torch.bfloat16)
    device_x = x.npu()
    initial = torch.randn(4, 16, 2 * head_dim)
    device_state = initial.npu()
    blocks = torch.tensor([2, 1], dtype=torch.int32, device="npu")
    offsets = torch.tensor([0, 2, 4], dtype=torch.int32, device="npu")
    lengths = torch.tensor([2, 2], dtype=torch.int32, device="npu")
    starts = torch.tensor([15, 3], dtype=torch.int32, device="npu")

    def forward():
        return torch.ops._C_ascend.compressor_v2(
            device_x, device_wkv, device_wgate, device_state, blocks, offsets, lengths, starts, ratio
        )

    forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        actual = forward()

    for batch_blocks, batch_starts in [([2, 1], [15, 3]), ([1, 3], [0, 31])]:
        x = torch.randn_like(x)
        initial = torch.randn_like(initial)
        # New request block 1 starts at zero and must ignore its stale history.
        expected, next_state, written = _reference(
            x, wkv, wgate, initial, batch_blocks, [0, 2], batch_starts, [2, 2], ratio
        )
        device_x.copy_(x)
        device_state.copy_(initial)
        blocks.copy_(torch.tensor(batch_blocks, dtype=torch.int32))
        starts.copy_(torch.tensor(batch_starts, dtype=torch.int32))
        graph.replay()
        _assert_result(actual, expected, device_state, next_state, initial, written)


def test_compressor_v2_adapter_restores_completed_rows_to_original_tokens():
    from vllm_ascend.ops.dsv41_a5.compressor import compressor_v2

    torch.manual_seed(734)
    hidden_size, head_dim = 5120, 128
    wkv, wgate = _weights(hidden_size, head_dim, torch.bfloat16)
    x = torch.randn(7, hidden_size).to(torch.bfloat16)
    state = torch.randn(4, 16, 2 * head_dim)
    blocks, starts, lengths, offsets = [2, 1], [1, 4], [3, 4], [0, 3]
    expected, expected_state, written = _reference(x, wkv, wgate, state, blocks, offsets, starts, lengths, 2)
    # Request 0 finishes groups at tokens 0/2; request 1 at tokens 4/6.
    completed = torch.tensor([True, False, True, False, True, False, True])
    metadata = SimpleNamespace(
        c2_ring_metadata=torch.tensor([starts, lengths, [0, 0], [0, 0], blocks], dtype=torch.int32, device="npu"),
        c2_complete_mask=completed.npu(),
        query_start_loc=torch.tensor([0, 3, 7], dtype=torch.int32, device="npu"),
    )
    device_state = state.npu()
    out = torch.full((7, head_dim), float("nan"), dtype=x.dtype, device="npu")
    actual = compressor_v2(x.npu(), wkv.npu(), wgate.npu(), device_state, metadata, out)
    assert actual.data_ptr() == out.data_ptr()
    aligned = torch.zeros(7, head_dim, dtype=x.dtype)
    aligned[completed] = expected
    torch.testing.assert_close(actual.cpu(), aligned, rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(device_state.cpu(), expected_state, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(device_state.cpu()[~written], state[~written], rtol=0, atol=0)


def test_compressor_v2_long_prefill_large_compression_group():
    torch.manual_seed(735)
    hidden_size, head_dim, ratio = 5120, 512, 128
    lengths, starts, offsets, blocks = [257, 129], [127, 0], [0, 257], [2, 1]
    x = torch.randn(sum(lengths), hidden_size).to(torch.bfloat16)
    wkv, wgate = _weights(hidden_size, head_dim, x.dtype)
    state = torch.randn(4, 512, 2 * head_dim)
    expected, expected_state, written = _reference(x, wkv, wgate, state, blocks, offsets, starts, lengths, ratio)
    device_state = state.npu()
    actual = _run(x.npu(), wkv.npu(), wgate.npu(), device_state, blocks, offsets, starts, lengths, ratio)
    _assert_result(actual, expected, device_state, expected_state, state, written)
