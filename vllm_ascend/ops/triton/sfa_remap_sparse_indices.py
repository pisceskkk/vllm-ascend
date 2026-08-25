# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

triton_sum = tl.sum


@triton.jit
def _sfa_remap_sparse_indices_stage1_kernel(
    input_ptr,
    intermediate_ptr,
    counts_ptr,
    rows,
    top_k: tl.constexpr,
    num_cores: tl.constexpr,
    segment_size: tl.constexpr,
    dcp_size: tl.constexpr,
    dcp_rank: tl.constexpr,
    interleave_size: tl.constexpr,
    interleave_shift: tl.constexpr,
    dcp_interleave_shift: tl.constexpr,
    use_power_of_two: tl.constexpr,
):
    """Stable-compact one input segment per vector core."""
    core = tl.program_id(0)
    segment_start = core * segment_size
    lanes = tl.arange(0, segment_size)
    offsets = segment_start + lanes
    valid_offset = offsets < top_k

    for row in tl.range(0, rows):
        row_base = row * top_k
        values = tl.load(
            input_ptr + row_base + offsets,
            mask=valid_offset,
            other=-1,
        ).to(tl.int32)

        valid_value = valid_offset & (values >= 0)
        if use_power_of_two:
            owner = (values >> interleave_shift) & (dcp_size - 1)
            remapped = ((values >> dcp_interleave_shift) << interleave_shift) | (values & (interleave_size - 1))
        else:
            block = values // interleave_size
            owner = block % dcp_size
            remapped = (block // dcp_size) * interleave_size + values % interleave_size

        is_local = valid_value & (owner == dcp_rank)
        compact_position = tl.cumsum(is_local.to(tl.int32), axis=0) - 1
        segment_count = triton_sum(is_local.to(tl.int32), axis=0)

        intermediate_base = row_base + segment_start
        tl.store(
            intermediate_ptr + intermediate_base + compact_position,
            remapped,
            mask=is_local,
        )
        tl.store(
            intermediate_ptr + intermediate_base + lanes,
            -1,
            mask=valid_offset & (lanes >= segment_count),
        )
        tl.store(counts_ptr + row * num_cores + core, segment_count)


@triton.jit
def _sfa_remap_sparse_indices_stage2_kernel(
    intermediate_ptr,
    output_ptr,
    counts_ptr,
    offsets_ptr,
    top_k: tl.constexpr,
    num_segments: tl.constexpr,
    segment_size: tl.constexpr,
):
    """Copy compacted segment prefixes to their final row positions."""
    program = tl.program_id(0)
    row = program // num_segments
    segment = program % num_segments
    count = tl.load(counts_ptr + program)
    output_base = tl.load(offsets_ptr + program)

    lanes = tl.arange(0, segment_size)
    valid_value = lanes < count
    values = tl.load(
        intermediate_ptr + row * top_k + segment * segment_size + lanes,
        mask=valid_value,
        other=0,
    )
    tl.store(
        output_ptr + row * top_k + output_base + lanes,
        values,
        mask=valid_value,
    )

    if segment == num_segments - 1:
        tail_start = output_base + count
        for tile in tl.range(0, (top_k + segment_size - 1) // segment_size):
            tail_offsets = tail_start + tile * segment_size + lanes
            tl.store(
                output_ptr + row * top_k + tail_offsets,
                -1,
                mask=tail_offsets < top_k,
            )


def remap_sparse_indices_triton(
    indices: torch.Tensor,
    output: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
) -> torch.Tensor:
    """Remap and stable-compact SFA indices with the staged Triton path.

    ``output`` is supplied by the caller so Ascend C and Triton use the same
    output-allocation boundary in controlled comparisons. The Triton path
    still owns its intermediate segment buffers and prefix-sum tensor.
    """
    if indices.dtype != torch.int32:
        raise TypeError(f"indices must have dtype int32, got {indices.dtype}")
    if not indices.is_contiguous():
        raise ValueError("indices must be contiguous")
    if indices.ndim == 0 or indices.shape[-1] <= 0:
        raise ValueError("indices must have a non-empty top-k dimension")
    if dcp_size <= 0 or not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"invalid dcp_size/dcp_rank: {dcp_size}/{dcp_rank}")
    if interleave_size <= 0:
        raise ValueError(f"interleave_size must be positive, got {interleave_size}")
    if output.shape != indices.shape or output.dtype != indices.dtype:
        raise ValueError("output must match indices shape and dtype")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")

    top_k = indices.shape[-1]
    rows = indices.numel() // top_k
    num_cores = get_vectorcore_num()
    segment_size = triton.next_power_of_2(triton.cdiv(top_k, num_cores))
    use_power_of_two = (dcp_size & (dcp_size - 1) == 0) and (interleave_size & (interleave_size - 1) == 0)
    interleave_shift = interleave_size.bit_length() - 1 if use_power_of_two else 0
    dcp_interleave_shift = (dcp_size * interleave_size).bit_length() - 1 if use_power_of_two else 0

    intermediate = torch.empty_like(indices)
    counts = torch.empty((rows, num_cores), dtype=torch.int32, device=indices.device)
    _sfa_remap_sparse_indices_stage1_kernel[(num_cores,)](
        indices,
        intermediate,
        counts,
        rows,
        top_k=top_k,
        num_cores=num_cores,
        segment_size=segment_size,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        interleave_size=interleave_size,
        interleave_shift=interleave_shift,
        dcp_interleave_shift=dcp_interleave_shift,
        use_power_of_two=use_power_of_two,
        multibuffer=False,
    )

    offsets = torch.cumsum(counts, dim=1) - counts
    _sfa_remap_sparse_indices_stage2_kernel[(rows * num_cores,)](
        intermediate,
        output,
        counts,
        offsets,
        top_k=top_k,
        num_segments=num_cores,
        segment_size=segment_size,
        multibuffer=False,
    )
    return output
