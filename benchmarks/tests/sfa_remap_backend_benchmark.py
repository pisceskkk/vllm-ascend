# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.context_parallel.sfa_cp import _remap_sparse_indices_ascendc
from vllm_ascend.ops.triton.sfa_remap_sparse_indices import remap_sparse_indices_triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op

RemapFn = Callable[[torch.Tensor, torch.Tensor, int, int, int], torch.Tensor]

BACKENDS: dict[str, RemapFn] = {
    "ascendc": _remap_sparse_indices_ascendc,
    "triton": remap_sparse_indices_triton,
}


def _reference_remap(
    indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
) -> torch.Tensor:
    indices_cpu = indices.cpu()
    blocks = torch.div(indices_cpu, interleave_size, rounding_mode="floor")
    is_local = (indices_cpu >= 0) & (blocks.remainder(dcp_size) == dcp_rank)
    remapped = torch.div(
        indices_cpu,
        dcp_size * interleave_size,
        rounding_mode="floor",
    ) * interleave_size + indices_cpu.remainder(interleave_size)
    result = torch.full_like(indices_cpu, -1)
    for source_row, local_row, result_row in zip(
        remapped.view(-1, remapped.shape[-1]),
        is_local.view(-1, is_local.shape[-1]),
        result.view(-1, result.shape[-1]),
        strict=True,
    ):
        local_values = source_row[local_row]
        result_row[: local_values.numel()] = local_values
    return result.to(indices.device)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _summarize(samples: list[float]) -> dict[str, Any]:
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p90_ms": _percentile(samples, 0.9),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples_ms": samples,
    }


def _measure(
    fn: Callable[[], torch.Tensor],
    iterations: int,
) -> float:
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def _benchmark_case(
    rows: int,
    top_k: int,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
    warmup: int,
    samples: int,
    iterations: int,
    allocate_output: bool,
) -> dict[str, Any]:
    indices = torch.randint(
        0,
        20_000_000,
        (rows, 1, top_k),
        dtype=torch.int32,
        device="npu",
    )
    indices[..., ::11] = -1
    expected = _reference_remap(indices, dcp_size, dcp_rank, interleave_size)
    outputs = {backend: torch.empty_like(indices) for backend in BACKENDS}

    calls: dict[str, Callable[[], torch.Tensor]] = {}
    for backend, fn in BACKENDS.items():
        if allocate_output:

            def call_with_allocation(fn: RemapFn = fn) -> torch.Tensor:
                return fn(
                    indices,
                    torch.empty_like(indices),
                    dcp_size,
                    dcp_rank,
                    interleave_size,
                )

            calls[backend] = call_with_allocation
        else:
            output = outputs[backend]

            def call_preallocated(fn: RemapFn = fn, output: torch.Tensor = output) -> torch.Tensor:
                return fn(indices, output, dcp_size, dcp_rank, interleave_size)

            calls[backend] = call_preallocated

    for backend, call in calls.items():
        actual = call()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for _ in range(warmup):
            call()
        torch.npu.synchronize()

    timings: dict[str, list[float]] = {backend: [] for backend in BACKENDS}
    backend_order = list(BACKENDS)
    for sample in range(samples):
        order = backend_order if sample % 2 == 0 else list(reversed(backend_order))
        for backend in order:
            timings[backend].append(_measure(calls[backend], iterations))

    summaries = {backend: _summarize(values) for backend, values in timings.items()}
    return {
        "shape": {
            "rows": rows,
            "top_k": top_k,
            "dcp_size": dcp_size,
            "dcp_rank": dcp_rank,
            "interleave_size": interleave_size,
        },
        "correctness": "exact",
        "output_allocation": "per-call" if allocate_output else "preallocated",
        "backends": summaries,
        "triton_speedup_vs_ascendc": (summaries["ascendc"]["median_ms"] / summaries["triton"]["median_ms"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled Ascend C/Triton SFA remap comparison")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 5, 16, 32, 64, 128])
    parser.add_argument("--top-k", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--dcp-size", type=int, default=8)
    parser.add_argument("--dcp-rank", type=int, default=3)
    parser.add_argument("--interleave-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--allocate-output", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    torch.npu.set_device(0)
    init_device_properties_triton()
    if not enable_custom_op() or not hasattr(torch.ops._C_ascend, "sfa_remap_sparse_indices"):
        raise RuntimeError("The Ascend C SFA remap custom operator is unavailable")

    torch.manual_seed(2026)
    cases = [
        _benchmark_case(
            rows,
            top_k,
            args.dcp_size,
            args.dcp_rank,
            args.interleave_size,
            args.warmup,
            args.samples,
            args.iterations,
            args.allocate_output,
        )
        for top_k in args.top_k
        for rows in args.rows
    ]
    result = {
        "measurement": {
            "warmup": args.warmup,
            "samples": args.samples,
            "iterations_per_sample": args.iterations,
            "order": "alternating-ascendc-triton",
            "synchronization": "before-and-after-each-sample",
        },
        "cases": cases,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
