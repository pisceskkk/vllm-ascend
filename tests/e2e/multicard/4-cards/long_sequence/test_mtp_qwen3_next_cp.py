#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
"""Tests for MTP (MultiToken Prediction) with PCP&DCP (Prefill&Decode
Context Parallel) for the qwen3_next model.

Run `pytest tests/e2e/multicard/4-cards/long_sequence/test_mtp_qwen3_next_cp.py`
"""

import os

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

os.environ["HCCL_BUFFSIZE"] = "768"

model = "Qwen/Qwen3-Next-80B-A3B-Instruct"

prompts = [
    "The capital of France is",
    "Hello, my name is Tom, I am",
    "The president of United States is",
    "AI future is",
]


@wait_until_npu_memory_free()
def test_pcp_mtp1_eager():
    with VllmRunner(
        model,
        max_model_len=4096,
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        max_num_batched_tokens=4096,
        block_size=128,
        speculative_config={
            "num_speculative_tokens": 1,
            "method": "qwen3_next_mtp",
        },
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    ) as runner:
        runner.generate_greedy(prompts, 32)


@wait_until_npu_memory_free()
def test_pcp_mtp3_eager():
    with VllmRunner(
        model,
        max_model_len=4096,
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        max_num_batched_tokens=4096,
        block_size=128,
        speculative_config={
            "num_speculative_tokens": 3,
            "method": "qwen3_next_mtp",
        },
        enforce_eager=True,
        gpu_memory_utilization=0.8,
    ) as runner:
        runner.generate_greedy(prompts, 32)
