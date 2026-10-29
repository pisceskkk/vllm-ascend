# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401


@pytest.mark.parametrize("groups", [1, 2, 4, 7])
def test_fallback_staged_block_ids_reach_slot_mappings(monkeypatch, groups):
    if not torch.npu.is_available():
        pytest.skip("Requires Ascend NPU")
    from vllm_ascend.patch.worker.patch_v2 import patch_uva
    from vllm_ascend.worker.v2.block_table import AscendBlockTables

    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: False)
    monkeypatch.setattr("vllm.v1.worker.gpu.buffer_utils.is_uva_available", lambda: True)
    tables = AscendBlockTables(
        block_sizes=[128] * groups,
        max_num_reqs=16,
        max_num_batched_tokens=16384,
        max_num_blocks_per_group=[8192] * groups,
        device=torch.device("npu"),
        kernel_block_sizes=[128] * groups,
    )
    mapping = torch.tensor([0], dtype=torch.int32, device="npu")
    starts = torch.tensor([0, 1024], dtype=torch.int32, device="npu")
    positions = torch.arange(1024, dtype=torch.int64, device="npu")
    for offset in (9, 109):
        ids = tuple(list(range(offset + 8 * g, offset + 8 * g + 8)) for g in range(groups))
        tables.append_block_ids(0, ids, overwrite=True)
        tables.apply_staged_writes()
        for table, row in zip(tables.block_tables, ids):
            assert table.gpu[0, :8].cpu().tolist() == row
        tables.gather_block_tables(mapping, 1)
        actual = tables.compute_slot_mappings(mapping, starts, positions, 1024).cpu()
        expected = torch.stack(
            [torch.tensor(row).repeat_interleave(128) * 128 + torch.arange(1024) % 128 for row in ids]
        )
        torch.testing.assert_close(actual.to(torch.int64), expected)
