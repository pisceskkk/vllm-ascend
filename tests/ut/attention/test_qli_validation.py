# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.attention.qli_validation import validate_qli_block_table


def _key_cache(num_blocks: int) -> torch.Tensor:
    return torch.empty((num_blocks, 4, 1, 8), dtype=torch.int8)


def test_validate_qli_block_table_accepts_valid_dcp_active_window() -> None:
    raw = torch.tensor([[2, 3, 99, 99]], dtype=torch.int32)
    mapped = torch.tensor([[4, 5, 6, 7, 198, 199, 198, 199]], dtype=torch.int32)

    validate_qli_block_table(
        key_cache=_key_cache(8),
        block_table=mapped,
        actual_seq_lengths_key=torch.tensor([8], dtype=torch.int32),
        block_size=4,
        raw_block_table=raw,
        dcp_size=2,
    )


def test_validate_qli_block_table_ignores_invalid_padding_columns() -> None:
    validate_qli_block_table(
        key_cache=_key_cache(8),
        block_table=torch.tensor([[4, 5, 1000, 1001]], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([8], dtype=torch.int32),
        block_size=4,
    )


def test_validate_qli_block_table_reports_raw_or_cache_binding() -> None:
    raw = torch.tensor([[2, 3]], dtype=torch.int32)
    mapped = torch.tensor([[4, 5, 6, 7]], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="source=raw_block_table_or_indexer_cache_binding"):
        validate_qli_block_table(
            key_cache=_key_cache(6),
            block_table=mapped,
            actual_seq_lengths_key=torch.tensor([16], dtype=torch.int32),
            block_size=4,
            raw_block_table=raw,
            dcp_size=2,
        )


def test_validate_qli_block_table_reports_replicated_view_overwrite() -> None:
    raw = torch.tensor([[2, 3]], dtype=torch.int32)
    mapped = torch.tensor([[4, 5, 42, 7]], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="source=dcp_replicated_view_overwrite"):
        validate_qli_block_table(
            key_cache=_key_cache(64),
            block_table=mapped,
            actual_seq_lengths_key=torch.tensor([16], dtype=torch.int32),
            block_size=4,
            raw_block_table=raw,
            dcp_size=2,
        )


def test_validate_qli_block_table_reports_seq_len_expansion() -> None:
    with pytest.raises(RuntimeError, match="source=actual_seq_lengths_key"):
        validate_qli_block_table(
            key_cache=_key_cache(8),
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
            actual_seq_lengths_key=torch.tensor([9], dtype=torch.int32),
            block_size=4,
        )


def test_validate_qli_block_table_reports_key_scale_capacity_mismatch() -> None:
    with pytest.raises(RuntimeError, match="source=indexer_cache_binding"):
        validate_qli_block_table(
            key_cache=_key_cache(8),
            key_scale_cache=torch.empty((7, 4, 1, 1), dtype=torch.float16),
            block_table=torch.tensor([[0, 1]], dtype=torch.int32),
            actual_seq_lengths_key=torch.tensor([8], dtype=torch.int32),
            block_size=4,
        )
