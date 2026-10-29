# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import array

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool

from vllm_ascend.patch.worker.patch_v2 import patch_uva


@pytest.mark.parametrize("real_uva", [False, True])
def test_pooled_chunks_refresh_direct_numpy_writes(monkeypatch, real_uva):
    original_zeros = torch.zeros

    def zeros(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", zeros)
    monkeypatch.setattr(patch_uva, "is_uva_available", lambda: False)
    monkeypatch.setattr(patch_uva, "_zeros_on_npu", lambda cpu: torch.zeros_like(cpu))
    monkeypatch.setattr("vllm.v1.worker.gpu.buffer_utils.is_uva_available", lambda: True)
    pool = UvaBufferPool(2, dtype=torch.int32, max_concurrency=2)

    # Growth and ring reuse must copy new data rather than a prior chunk.
    for offset in (0, 100, 200, 300):
        chunks = [np.arange(offset, offset + 5, dtype=np.int64), array.array("i", [offset + 5, offset + 6])]
        if real_uva:
            original_next = pool._next_buf

            def next_buf(n, original_next=original_next):
                buf = original_next(n)
                buf._use_real_uva = True
                buf._uva = buf._cpu
                return buf

            monkeypatch.setattr(pool, "_next_buf", next_buf)
        actual = pool.copy_chunks_to_uva(chunks, 7)
        assert actual.tolist() == list(range(offset, offset + 7))
        assert not pool._uva_bufs[pool._curr]._modified_indices
        if real_uva:
            monkeypatch.setattr(pool, "_next_buf", original_next)
