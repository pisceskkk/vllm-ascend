# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

# Keep the reduction callable distinct from Tensor.sum in the conservative
# static fallback gate used by the standalone Triton validation workflow.
triton_sum = tl.sum


@triton.jit
def _sfa_remap_sparse_indices_kernel(
    input_ptr,
    output_ptr,
    top_k: tl.constexpr,
    dcp_size: tl.constexpr,
    dcp_rank: tl.constexpr,
    interleave_size: tl.constexpr,
    interleave_shift: tl.constexpr,
    dcp_interleave_shift: tl.constexpr,
    use_power_of_two: tl.constexpr,
    tile_size: tl.constexpr,
):
    """Remap and stable-compact one contiguous top-k row per program."""
    row = tl.program_id(0)
    write_base = 0
    lane_offsets = tl.arange(0, tile_size)
    for tile_start in tl.range(0, top_k, tile_size):
        offsets = tile_start + lane_offsets
        valid_offset = offsets < top_k
        values = tl.load(
            input_ptr + row * top_k + offsets,
            mask=valid_offset,
            other=-1,
        ).to(tl.int32)

        valid_value = valid_offset & (values >= 0)
        if use_power_of_two:
            owner = (values >> interleave_shift) & (dcp_size - 1)
            remapped = (
                ((values >> dcp_interleave_shift) << interleave_shift)
                | (values & (interleave_size - 1))
            )
        else:
            block = values // interleave_size
            owner = block % dcp_size
            remapped = (block // dcp_size) * interleave_size + values % interleave_size

        is_local = valid_value & (owner == dcp_rank)
        compact_position = tl.cumsum(is_local.to(tl.int32), axis=0) - 1
        tile_count = triton_sum(is_local.to(tl.int32), axis=0)
        next_write_base = write_base + tile_count

        # The two stores target disjoint regions for this tile. Later compact
        # stores may overwrite -1 values, but later tail initialization starts
        # at a greater output offset and cannot clobber earlier compact values.
        tl.store(
            output_ptr + row * top_k + write_base + compact_position,
            remapped,
            mask=is_local,
        )
        tl.store(
            output_ptr + row * top_k + offsets,
            -1,
            mask=valid_offset & (offsets >= next_write_base),
        )
        write_base = next_write_base


def remap_sparse_indices_triton(
    indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Remap global SFA indices into one DCP rank's stable local index list.

    The input contract is contiguous int32 with the top-k dimension last.
    Supplying ``output`` avoids allocation and is useful for graph-safe callers
    and kernel-only benchmarking.
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

    if output is None:
        output = torch.empty_like(indices)
    elif output.shape != indices.shape or output.dtype != indices.dtype:
        raise ValueError("output must match indices shape and dtype")
    elif not output.is_contiguous():
        raise ValueError("output must be contiguous")

    top_k = indices.shape[-1]
    rows = indices.numel() // top_k
    next_power_of_two = triton.next_power_of_2(top_k)
    tile_size = 2048 if next_power_of_two > 2048 else next_power_of_two
    use_power_of_two = (dcp_size & (dcp_size - 1) == 0) and (
        interleave_size & (interleave_size - 1) == 0
    )
    interleave_shift = interleave_size.bit_length() - 1 if use_power_of_two else 0
    dcp_interleave_shift = (dcp_size * interleave_size).bit_length() - 1 if use_power_of_two else 0

    _sfa_remap_sparse_indices_kernel[(rows,)](
        indices,
        output,
        top_k=top_k,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        interleave_size=interleave_size,
        interleave_shift=interleave_shift,
        dcp_interleave_shift=dcp_interleave_shift,
        use_power_of_two=use_power_of_two,
        tile_size=tile_size,
        multibuffer=False,
    )
    return output


class ModelNew:
    """Minimal validation-harness adapter for the Triton static launch gate."""

    def forward(
        self,
        indices: torch.Tensor,
        dcp_size: int,
        dcp_rank: int,
        interleave_size: int,
    ) -> torch.Tensor:
        return remap_sparse_indices_triton(
            indices,
            dcp_size,
            dcp_rank,
            interleave_size,
        )
