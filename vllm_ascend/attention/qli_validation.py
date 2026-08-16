# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def _first_true_index(mask: torch.Tensor) -> tuple[int, int] | None:
    indices = torch.nonzero(mask, as_tuple=False)
    if indices.numel() == 0:
        return None
    return int(indices[0, 0].item()), int(indices[0, 1].item())


def _raise_validation_error(source: str, detail: str, context: str) -> None:
    suffix = f", context={context}" if context else ""
    raise RuntimeError(f"QLI block-table validation failed: source={source}, {detail}{suffix}")


def validate_qli_block_table(
    *,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    block_size: int,
    key_scale_cache: torch.Tensor | None = None,
    raw_block_table: torch.Tensor | None = None,
    dcp_size: int = 1,
    blocks_per_phys_block: int = 1,
    context: str = "",
) -> None:
    """Validate only the block-table columns that QLI may dereference.

    The checks distinguish an invalid raw table/cache binding from corruption
    of the SFA-DCP replicated view. They deliberately synchronize the device
    and must only be enabled for diagnosis.
    """
    if key_cache.ndim < 1 or key_cache.shape[0] <= 0:
        _raise_validation_error(
            "indexer_cache_binding",
            f"invalid key_cache.shape={tuple(key_cache.shape)}",
            context,
        )
    if key_scale_cache is not None and key_scale_cache.shape[0] != key_cache.shape[0]:
        _raise_validation_error(
            "indexer_cache_binding",
            f"key_cache.shape[0]={key_cache.shape[0]} differs from key_scale_cache.shape[0]={key_scale_cache.shape[0]}",
            context,
        )
    if block_table.ndim != 2:
        _raise_validation_error(
            "mapped_block_table_metadata",
            f"block_table.shape={tuple(block_table.shape)} is not two-dimensional",
            context,
        )
    if actual_seq_lengths_key.ndim != 1:
        _raise_validation_error(
            "actual_seq_lengths_key",
            f"shape={tuple(actual_seq_lengths_key.shape)} is not one-dimensional",
            context,
        )
    if block_size <= 0:
        _raise_validation_error("actual_seq_lengths_key", f"invalid block_size={block_size}", context)

    num_reqs = block_table.shape[0]
    if actual_seq_lengths_key.shape[0] < num_reqs:
        _raise_validation_error(
            "actual_seq_lengths_key",
            f"length_count={actual_seq_lengths_key.shape[0]} is smaller than block_table rows={num_reqs}",
            context,
        )

    seq_lens = actual_seq_lengths_key[:num_reqs].to(dtype=torch.int64)
    negative_idx = torch.nonzero(seq_lens < 0, as_tuple=False)
    if negative_idx.numel() != 0:
        row = int(negative_idx[0, 0].item())
        _raise_validation_error(
            "actual_seq_lengths_key",
            f"row={row}, seq_len={int(seq_lens[row].item())}",
            context,
        )

    active_cols_per_req = torch.div(seq_lens + block_size - 1, block_size, rounding_mode="floor")
    max_active_cols = int(active_cols_per_req.max().item()) if num_reqs else 0
    if max_active_cols > block_table.shape[1]:
        row = int(torch.argmax(active_cols_per_req).item())
        _raise_validation_error(
            "actual_seq_lengths_key",
            f"row={row}, seq_len={int(seq_lens[row].item())}, active_cols={max_active_cols}, "
            f"block_table_cols={block_table.shape[1]}",
            context,
        )
    if max_active_cols == 0:
        return

    col_idx = torch.arange(max_active_cols, dtype=torch.int64, device=block_table.device)
    active_mask = col_idx.unsqueeze(0) < active_cols_per_req.to(device=block_table.device).unsqueeze(1)
    mapped_active = block_table[:num_reqs, :max_active_cols]

    raw_active = None
    local_col_idx = None
    if raw_block_table is not None:
        if raw_block_table.ndim != 2 or raw_block_table.shape[0] < num_reqs:
            _raise_validation_error(
                "raw_block_table_metadata",
                f"raw_block_table.shape={tuple(raw_block_table.shape)}, required_rows={num_reqs}",
                context,
            )
        if dcp_size <= 1 or blocks_per_phys_block <= 0:
            _raise_validation_error(
                "dcp_replicated_view_metadata",
                f"dcp_size={dcp_size}, blocks_per_phys_block={blocks_per_phys_block}",
                context,
            )

        local_col_idx = (
            col_idx // (dcp_size * blocks_per_phys_block) * blocks_per_phys_block + col_idx % blocks_per_phys_block
        )
        max_local_col = int(local_col_idx.max().item())
        if max_local_col >= raw_block_table.shape[1]:
            _raise_validation_error(
                "actual_seq_lengths_key",
                f"mapped_active_cols={max_active_cols} requires raw_col={max_local_col}, "
                f"raw_block_table_cols={raw_block_table.shape[1]}",
                context,
            )

        rank_in_replicated_view = (col_idx // blocks_per_phys_block) % dcp_size
        raw_active = torch.index_select(raw_block_table[:num_reqs], 1, local_col_idx)
        raw_active_i64 = raw_active.to(dtype=torch.int64)
        if blocks_per_phys_block == 1:
            expected_mapped = raw_active_i64 * dcp_size + rank_in_replicated_view
        else:
            local_sub_blocks = raw_active_i64 % blocks_per_phys_block
            local_phys_blocks = raw_active_i64 // blocks_per_phys_block
            expected_mapped = (
                local_phys_blocks * dcp_size + rank_in_replicated_view
            ) * blocks_per_phys_block + local_sub_blocks

        mismatch_idx = _first_true_index((mapped_active != expected_mapped) & active_mask)
        if mismatch_idx is not None:
            row, col = mismatch_idx
            raw_col = int(local_col_idx[col].item())
            _raise_validation_error(
                "dcp_replicated_view_overwrite",
                f"row={row}, mapped_col={col}, raw_col={raw_col}, "
                f"raw_id={int(raw_active[row, col].item())}, "
                f"expected_mapped_id={int(expected_mapped[row, col].item())}, "
                f"actual_mapped_id={int(mapped_active[row, col].item())}",
                context,
            )

    invalid_range = ((mapped_active < 0) | (mapped_active >= key_cache.shape[0])) & active_mask
    invalid_idx = _first_true_index(invalid_range)
    if invalid_idx is None:
        return

    row, col = invalid_idx
    mapped_id = int(mapped_active[row, col].item())
    if raw_active is not None and local_col_idx is not None:
        raw_col = int(local_col_idx[col].item())
        raw_id = int(raw_active[row, col].item())
        _raise_validation_error(
            "raw_block_table_or_indexer_cache_binding",
            f"row={row}, mapped_col={col}, raw_col={raw_col}, raw_id={raw_id}, "
            f"mapped_id={mapped_id}, valid_mapped_range=[0,{key_cache.shape[0] - 1}], "
            f"seq_len={int(seq_lens[row].item())}",
            context,
        )
    _raise_validation_error(
        "mapped_block_table_or_indexer_cache_binding",
        f"row={row}, mapped_col={col}, mapped_id={mapped_id}, "
        f"valid_mapped_range=[0,{key_cache.shape[0] - 1}], seq_len={int(seq_lens[row].item())}",
        context,
    )
