#!/usr/bin/env python3
"""Minimal msprof-op entry point for compute_slot_mapping."""

from __future__ import annotations

import os

import torch
import torch_npu  # noqa: F401
from compute_slot_mapping_opt_runner import CASES, _launch, _make_inputs


def main() -> int:
    case_id = os.environ.get("SLOT_MAP_CASE", "profile-1r-4096-s8192")
    launches = int(os.environ.get("SLOT_MAP_LAUNCHES", "200"))
    case = CASES[case_id]
    torch.npu.set_device(0)
    _, _, query_start_loc, positions, block_table, slot_mapping = _make_inputs(case)
    for _ in range(launches):
        _launch(case, query_start_loc, positions, block_table, slot_mapping)
    torch.npu.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
