# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Only the older torch.compile allocator needs index_fill interception."""

import importlib

import torch
from vllm.v1.worker.gpu.spec_decode import adaptive_verification as adaptive

from vllm_ascend.device.device_op import DeviceOperator

MODULE = "vllm_ascend.patch.worker.patch_v2.patch_adaptive_verification"


def test_triton_allocator_is_preserved(monkeypatch):
    module = importlib.import_module(MODULE)
    allocator = adaptive.assign_draft_token_budget
    try:
        with monkeypatch.context() as context:
            context.delattr(adaptive, "_assign_draft_token_budget", raising=False)
            context.delattr(adaptive, "_assign_draft_token_budget_compiled", raising=False)
            importlib.reload(module)
            assert module._original_assign_draft_token_budget is None
            assert adaptive.assign_draft_token_budget is allocator
            assert not hasattr(adaptive, "_assign_draft_token_budget_compiled")
    finally:
        importlib.reload(module)


def test_legacy_allocator_retains_scoped_device_index_fill(monkeypatch):
    module = importlib.import_module(MODULE)

    def allocator(tensor, indices):
        return tensor.index_fill_(0, indices, 3)

    def device_index_fill(tensor, dim, index, value):
        return tensor.scatter_(dim, index, value)

    try:
        with monkeypatch.context() as context:
            context.setattr(adaptive, "_assign_draft_token_budget", allocator, raising=False)
            context.setattr(adaptive, "_assign_draft_token_budget_compiled", object(), raising=False)
            context.setattr(DeviceOperator, "index_fill", device_index_fill)
            importlib.reload(module)
            tensor = torch.zeros(4)
            result = adaptive._assign_draft_token_budget_compiled(tensor, torch.tensor([0, 2]))
            assert result is tensor
            torch.testing.assert_close(tensor, torch.tensor([3.0, 0.0, 3.0, 0.0]), rtol=0, atol=0)
    finally:
        importlib.reload(module)
