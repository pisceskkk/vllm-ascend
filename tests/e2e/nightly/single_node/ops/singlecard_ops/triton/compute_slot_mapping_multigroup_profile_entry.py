#!/usr/bin/env python3
"""msprof entry point for fused multi-group slot mapping."""

import os

import torch
import torch_npu  # noqa: F401
from compute_slot_mapping_opt_runner import (  # type: ignore[import-not-found]
    MULTIGROUP_CASES,
    launch_multigroup,
    make_multigroup_inputs,
)

torch.npu.set_device(0)
cases = MULTIGROUP_CASES[os.environ.get("SLOT_MAP_MULTIGROUP_CASE", "multigroup-profile-6x4096")]
inputs = make_multigroup_inputs(cases)
for _ in range(int(os.environ.get("SLOT_MAP_LAUNCHES", "200"))):
    launch_multigroup(cases, inputs)
torch.npu.synchronize()
