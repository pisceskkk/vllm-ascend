from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.model_states.default import AscendModelState


@pytest.mark.parametrize(
    ("cudagraph_mode", "expected_num_input_tokens"),
    [
        (CUDAGraphMode.NONE, 7),
        (CUDAGraphMode.FULL, 16),
    ],
)
def test_prepare_attn_forwards_selected_token_extent(
    cudagraph_mode: CUDAGraphMode,
    expected_num_input_tokens: int,
):
    state = AscendModelState.__new__(AscendModelState)
    state.max_model_len = 1024

    input_batch = MagicMock()
    input_batch.num_reqs = 1
    input_batch.num_reqs_after_padding = 2
    input_batch.num_tokens = 7
    input_batch.num_tokens_after_padding = 16
    input_batch.num_scheduled_tokens = np.array([7], dtype=np.int32)
    input_batch.query_start_loc_np = np.array([0, 7], dtype=np.int32)
    input_batch.query_start_loc = torch.tensor([0, 7], dtype=torch.int32)
    input_batch.seq_lens = torch.tensor([7], dtype=torch.int32)
    input_batch.dcp_local_seq_lens = None
    input_batch.seq_lens_np = np.array([7], dtype=np.int32)
    input_batch.positions = torch.arange(16, dtype=torch.int64)
    input_batch.attn_state = None

    with patch(
        "vllm_ascend.worker.v2.model_states.default.build_attn_metadata",
        return_value={"metadata": MagicMock()},
    ) as build:
        state.prepare_attn(
            input_batch=input_batch,
            cudagraph_mode=cudagraph_mode,
            block_tables=(torch.zeros((1, 1), dtype=torch.int32),),
            slot_mappings=torch.zeros((16,), dtype=torch.int64),
            attn_groups=[],
            kv_cache_config=MagicMock(),
        )

    assert build.call_args.kwargs["num_input_tokens"] == expected_num_input_tokens
    assert build.call_args.kwargs["positions"] is input_batch.positions
