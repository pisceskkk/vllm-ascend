# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_states/default.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch


class AscendModelState(DefaultModelState):
    """Model state for Ascend NPUs."""

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        """Override prepare_attn method because `build_attn_metadata` is different from vllm."""
        if cudagraph_mode == CUDAGraphMode.FULL:
            # Use padded sizes - padding is handled by model_runner.prepare_attn.
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            # For piecewise cudagraphs and eager, use unpadded sizes.
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = (
            input_batch.query_lens.max().item()
            if input_batch.query_lens is not None
            else input_batch.num_scheduled_tokens.max().item()
        )
        prefill_context_parallel_metadata = None
        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None and getattr(model_runner, "use_cp", False):
            pcp_manager = model_runner.pcp_manager
            assert pcp_manager is not None
            while len(pcp_manager.pcp_padded_slot_mapping_list) < len(block_tables):
                pcp_manager.initialize_slot_mapping()

            updated_block_tables = list(block_tables)
            prefill_context_parallel_metadata, updated_block_tables[0] = pcp_manager.generate_pcp_metadata(
                num_tokens,
                (input_batch.query_lens if input_batch.query_lens is not None
                else torch.from_numpy(input_batch.num_scheduled_tokens)),
                input_batch,
                input_batch.num_scheduled_tokens,
                updated_block_tables[0],
                num_reqs,
                input_batch.num_reqs,
            )
            block_tables = tuple(updated_block_tables)

            if model_runner.pcp_size > 1:
                padded_slot_mappings = []
                for kv_cache_gid, slot_mapping in enumerate(slot_mappings):
                    padded_slot_mappings.append(
                        pcp_manager.get_padded_slot_mapping(
                            num_tokens,
                            input_batch.num_tokens_after_padding,
                            slot_mapping,
                            kv_cache_gid,
                        )
                    )
                slot_mappings = torch.stack(padded_slot_mappings)
        # attn_metadata is needed when update_full_graph_params, but no way can get it now.
        # Temporarily store it in model_state.
        self.attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            # extra attributes for ascend npus.
            seq_lens_np=input_batch.seq_lens_np,
            num_computed_tokens_cpu=input_batch.num_computed_tokens_cpu,
            positions=input_batch.positions,
            attn_state=input_batch.attn_state,
            prefill_context_parallel_metadata=prefill_context_parallel_metadata,
        )
        return self.attn_metadata
