#!/usr/bin/env python3
"""Minimal msprof-op entry point for multi-request slot mapping."""

from __future__ import annotations

import os

import torch
import torch_npu  # noqa: F401
from compute_slot_mapping_opt_runner import (
    CASES,
    _launch,
    _launch_original,
    _make_inputs,
)


def main() -> int:
    case_id = os.environ.get("SLOT_MAP_CASE", "profile-mr-8x512-s8192")
    launch_mode = os.environ.get("SLOT_MAP_LAUNCH_MODE", "current")
    launches = int(os.environ.get("SLOT_MAP_LAUNCHES", "200"))
    if launch_mode not in {"current", "original"}:
        raise ValueError(f"unsupported SLOT_MAP_LAUNCH_MODE={launch_mode!r}")
    case = CASES[case_id]
    torch.npu.set_device(0)
    _, _, query_start_loc, positions, block_table, slot_mapping = _make_inputs(case)
    launch = _launch if launch_mode == "current" else _launch_original
    for _ in range(launches):
        launch(case, query_start_loc, positions, block_table, slot_mapping)
    torch.npu.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
