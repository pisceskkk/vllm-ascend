#!/usr/bin/env python3
"""Correctness and latency runner for compute_slot_mapping optimization."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.compute_slot_mapping import (
    _compute_slot_mapping_adaptive_kernel,
    _compute_slot_mapping_fused_groups_kernel,
    _compute_slot_mapping_kernel,
    _compute_slot_mapping_parallel_kernel,
    _next_power_of_2,
)

PAD_ID = -1
TILE_BLOCK_SIZE = 1024


@dataclass(frozen=True)
class Case:
    id: str
    lengths: tuple[int, ...]
    block_table_stride: int
    max_num_tokens: int
    block_size: int = 128
    kv_cache_block_size: int = 128
    max_num_reqs: int = 8
    cp_world_size: int = 1
    cp_rank: int = 0
    cp_interleave: int = 1
    position_order: str = "sequential"


CASES = {
    case.id: case
    for case in (
        Case("profile-1r-4096-s64", (4096,), 64, 4096),
        Case("profile-1r-4096-s2048", (4096,), 2048, 4096),
        Case("profile-1r-4096-s8192", (4096,), 8192, 4096),
        Case("profile-1r-4096-s65536", (4096,), 65536, 4096),
        Case("profile-1r-4096-s131072", (4096,), 131072, 4096),
        Case("profile-2r-4096-s8192", (2048, 2048), 8192, 4096),
        Case("profile-2r-uneven-s8192", (0, 4096), 8192, 4096),
        Case("profile-mr-4x1024-s8192", (1024,) * 4, 8192, 4096),
        Case("profile-mr-8x512-s8192", (512,) * 8, 8192, 4096),
        Case("profile-mr-16x256-s8192", (256,) * 16, 8192, 4096, max_num_reqs=16),
        Case("profile-mr-32x128-s8192", (128,) * 32, 8192, 4096, max_num_reqs=32),
        Case("profile-mr-64x64-s8192", (64,) * 64, 8192, 4096, max_num_reqs=64),
        Case(
            "profile-mr-8-uneven-s8192",
            (2048, 1024, 512, 256, 128, 64, 32, 32),
            8192,
            4096,
        ),
        Case(
            "profile-mr-32-uneven-s8192",
            (4096,) + (0,) * 31,
            8192,
            4096,
            max_num_reqs=32,
        ),
        Case("decode-1r-1-max4096", (1,), 64, 4096),
        Case("decode-64r-64-max4096", (1,) * 64, 64, 4096, max_num_reqs=64),
        Case("tail-1r-1023", (1023,), 64, 4096),
        Case("boundary-1r-1025", (1025,), 64, 4096),
        Case("prefill-8r-8192", (1024,) * 8, 128, 8192),
        Case(
            "hybrid-1r-4096-p128-l32",
            (4096,),
            256,
            4096,
            block_size=32,
            kv_cache_block_size=128,
        ),
        Case(
            "nonmonotonic-1r-257",
            (257,),
            64,
            1024,
            position_order="reverse_chunks",
        ),
        Case("empty-first-2r-4096", (0, 4096), 64, 4096),
        Case("padding-1r-4-max8192", (4,), 64, 8192),
        Case(
            "cp2-rank1-hybrid",
            (4096,),
            64,
            4096,
            block_size=32,
            kv_cache_block_size=128,
            cp_world_size=2,
            cp_rank=1,
            cp_interleave=2,
        ),
    )
}

PERFORMANCE_CASES = (
    "profile-1r-4096-s64",
    "profile-1r-4096-s2048",
    "profile-1r-4096-s8192",
    "profile-1r-4096-s65536",
    "profile-1r-4096-s131072",
    "profile-2r-4096-s8192",
    "profile-2r-uneven-s8192",
    "decode-1r-1-max4096",
    "decode-64r-64-max4096",
    "tail-1r-1023",
    "boundary-1r-1025",
    "prefill-8r-8192",
    "hybrid-1r-4096-p128-l32",
    "padding-1r-4-max8192",
)

MULTIREQUEST_PERFORMANCE_CASES = (
    "profile-2r-4096-s8192",
    "profile-2r-uneven-s8192",
    "profile-mr-4x1024-s8192",
    "profile-mr-8x512-s8192",
    "profile-mr-16x256-s8192",
    "profile-mr-32x128-s8192",
    "profile-mr-64x64-s8192",
    "profile-mr-8-uneven-s8192",
    "profile-mr-32-uneven-s8192",
    "prefill-8r-8192",
    "decode-64r-64-max4096",
)

MULTIGROUP_CASES = {
    "multigroup-profile-6x4096": (
        Case("group-s64", (4096,), 64, 4096),
        Case("group-s2048", (4096,), 2048, 4096),
        Case("group-s8192", (4096,), 8192, 4096),
        Case("group-s65536", (4096,), 65536, 4096),
        Case("group-s131072", (4096,), 131072, 4096),
        Case("group-hybrid", (4096,), 256, 4096, block_size=32, kv_cache_block_size=128),
    ),
    "multigroup-general-2x2048-padding": (
        Case("group-block64", (2048,), 64, 8192, block_size=64, kv_cache_block_size=128),
        Case("group-block128", (2048,), 64, 8192),
    ),
    "multigroup-nonmonotonic-2x4096": (
        Case("group-reverse64", (4096,), 64, 4096, position_order="reverse_chunks"),
        Case(
            "group-reverse64-hybrid",
            (4096,),
            256,
            4096,
            block_size=32,
            kv_cache_block_size=128,
            position_order="reverse_chunks",
        ),
    ),
}


def _positions(case: Case) -> list[int]:
    result: list[int] = []
    for length in case.lengths:
        values = list(range(length))
        if case.position_order == "reverse_chunks":
            values = [value for start in range(0, length, 64) for value in reversed(values[start : start + 64])]
        result.extend(values)
    return result


def _query_start_loc(lengths: tuple[int, ...]) -> list[int]:
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def _reference(case: Case, positions: list[int], block_table: list[list[int]]) -> list[int]:
    output = [PAD_ID] * case.max_num_tokens
    query_start_loc = _query_start_loc(case.lengths)
    blocks_per_kv_block = case.kv_cache_block_size // case.block_size
    for req_idx, (start, end) in enumerate(zip(query_start_loc[:-1], query_start_loc[1:])):
        for token_idx in range(start, end):
            position = positions[token_idx]
            if case.cp_world_size == 1:
                block_idx = position // case.block_size
                slot_offset = position % case.block_size
            else:
                virtual_block_size = case.kv_cache_block_size * case.cp_world_size
                virtual_block_idx = position // virtual_block_size
                virtual_offset = position % virtual_block_size
                is_local = (virtual_offset // case.cp_interleave) % case.cp_world_size == case.cp_rank
                if not is_local:
                    continue
                local_offset = (
                    virtual_offset // (case.cp_world_size * case.cp_interleave) * case.cp_interleave
                    + virtual_offset % case.cp_interleave
                )
                block_idx = virtual_block_idx * blocks_per_kv_block + local_offset // case.block_size
                slot_offset = local_offset % case.block_size
            output[token_idx] = block_table[req_idx][block_idx] * case.block_size + slot_offset
    return output


def _make_inputs(case: Case):
    positions_list = _positions(case)
    query_start_loc_list = _query_start_loc(case.lengths)
    block_table = [
        [req_idx * case.block_table_stride + block_idx for block_idx in range(case.block_table_stride)]
        for req_idx in range(case.max_num_reqs)
    ]
    query_start_loc = torch.tensor(query_start_loc_list, dtype=torch.int32, device="npu")
    positions = torch.tensor(positions_list, dtype=torch.int64, device="npu")
    block_table_tensor = torch.tensor(block_table, dtype=torch.int32, device="npu")
    slot_mapping = torch.full((case.max_num_tokens,), 123456, dtype=torch.int32, device="npu")
    return positions_list, block_table, query_start_loc, positions, block_table_tensor, slot_mapping


def _launch(case: Case, query_start_loc, positions, block_table, slot_mapping) -> None:
    num_reqs = len(case.lengths)
    num_tokens = positions.shape[0]
    tile_block_size = TILE_BLOCK_SIZE
    if num_reqs > 1:
        tokens_per_req = max(math.ceil(num_tokens / num_reqs), 1)
        tile_block_size = min(max(_next_power_of_2(tokens_per_req), 16), tile_block_size)
    parallel_tiles = min(math.ceil(num_tokens / tile_block_size), 4)
    if num_reqs == 1:
        parallel_tiles = parallel_tiles if num_tokens >= 2 * tile_block_size else 1
    elif num_reqs == 2 and num_tokens >= 4 * tile_block_size:
        parallel_tiles = 2
    else:
        parallel_tiles = 1
    common_kernel_kwargs = {
        "KV_CACHE_BLOCK_SIZE": case.kv_cache_block_size,
        "BLOCKS_PER_KV_BLOCK": case.kv_cache_block_size // case.block_size,
        "TOTAL_CP_WORLD_SIZE": case.cp_world_size,
        "TOTAL_CP_RANK": case.cp_rank,
        "CP_KV_CACHE_INTERLEAVE_SIZE": case.cp_interleave,
        "PAD_ID": PAD_ID,
    }
    kernel_kwargs = {
        **common_kernel_kwargs,
        "TILE_BLOCK_SIZE": tile_block_size,
        "BLOCK_TABLE_WINDOW_SIZE": _next_power_of_2(math.ceil(tile_block_size / case.block_size) + 1),
    }
    if num_reqs > 1 and tile_block_size < TILE_BLOCK_SIZE:
        _compute_slot_mapping_adaptive_kernel[(num_reqs + 1,)](
            num_tokens,
            case.max_num_tokens,
            query_start_loc,
            positions,
            block_table,
            block_table.stride(0),
            case.block_size,
            slot_mapping,
            SMALL_TILE_BLOCK_SIZE=tile_block_size,
            SMALL_BLOCK_TABLE_WINDOW_SIZE=kernel_kwargs["BLOCK_TABLE_WINDOW_SIZE"],
            LARGE_BLOCK_TABLE_WINDOW_SIZE=_next_power_of_2(math.ceil(TILE_BLOCK_SIZE / case.block_size) + 1),
            **common_kernel_kwargs,
        )
    elif parallel_tiles > 1:
        _compute_slot_mapping_parallel_kernel[(num_reqs * parallel_tiles + 1,)](
            num_tokens,
            case.max_num_tokens,
            query_start_loc,
            positions,
            block_table,
            block_table.stride(0),
            case.block_size,
            slot_mapping,
            PARALLEL_TILES=parallel_tiles,
            **kernel_kwargs,
        )
    else:
        _compute_slot_mapping_kernel[(num_reqs + 1,)](
            num_tokens,
            case.max_num_tokens,
            query_start_loc,
            positions,
            block_table,
            block_table.stride(0),
            case.block_size,
            slot_mapping,
            **kernel_kwargs,
        )


def _launch_original(case: Case, query_start_loc, positions, block_table, slot_mapping) -> None:
    num_reqs = len(case.lengths)
    num_tokens = positions.shape[0]
    _compute_slot_mapping_kernel[(num_reqs + 1,)](
        num_tokens,
        case.max_num_tokens,
        query_start_loc,
        positions,
        block_table,
        block_table.stride(0),
        case.block_size,
        slot_mapping,
        KV_CACHE_BLOCK_SIZE=case.kv_cache_block_size,
        BLOCKS_PER_KV_BLOCK=case.kv_cache_block_size // case.block_size,
        TOTAL_CP_WORLD_SIZE=case.cp_world_size,
        TOTAL_CP_RANK=case.cp_rank,
        CP_KV_CACHE_INTERLEAVE_SIZE=case.cp_interleave,
        PAD_ID=PAD_ID,
        TILE_BLOCK_SIZE=TILE_BLOCK_SIZE,
        BLOCK_TABLE_WINDOW_SIZE=_next_power_of_2(math.ceil(TILE_BLOCK_SIZE / case.block_size) + 1),
    )


def make_multigroup_inputs(cases: tuple[Case, ...]):
    positions_lists = []
    block_table_lists = []
    block_tables = []
    slot_mappings = []
    positions = None
    for case in cases:
        positions_list, block_table_list, _, current_positions, block_table, slot_mapping = _make_inputs(case)
        if positions is None:
            positions = current_positions
        else:
            assert torch.equal(positions, current_positions)
        positions_lists.append(positions_list)
        block_table_lists.append(block_table_list)
        block_tables.append(block_table)
        slot_mappings.append(slot_mapping)
    assert positions is not None
    block_table_addrs = torch.tensor([tensor.data_ptr() for tensor in block_tables], dtype=torch.uint64, device="npu")
    slot_mapping_addrs = torch.tensor([tensor.data_ptr() for tensor in slot_mappings], dtype=torch.uint64, device="npu")
    block_table_strides = torch.tensor([tensor.stride(0) for tensor in block_tables], dtype=torch.int64, device="npu")
    block_sizes = torch.tensor([case.block_size for case in cases], dtype=torch.int32, device="npu")
    return (
        positions_lists,
        block_table_lists,
        positions,
        block_tables,
        slot_mappings,
        block_table_addrs,
        slot_mapping_addrs,
        block_table_strides,
        block_sizes,
    )


def launch_multigroup(cases: tuple[Case, ...], inputs) -> None:
    (
        _,
        _,
        positions,
        _,
        _,
        block_table_addrs,
        slot_mapping_addrs,
        block_table_strides,
        block_sizes,
    ) = inputs
    num_tokens = positions.shape[0]
    parallel_tiles = min(math.ceil(num_tokens / TILE_BLOCK_SIZE), 4)
    min_block_size = min(case.block_size for case in cases)
    _compute_slot_mapping_fused_groups_kernel[(len(cases) * (parallel_tiles + 1),)](
        num_tokens,
        cases[0].max_num_tokens,
        positions,
        block_table_addrs,
        slot_mapping_addrs,
        block_table_strides,
        block_sizes,
        PAD_ID=PAD_ID,
        TILE_BLOCK_SIZE=TILE_BLOCK_SIZE,
        PARALLEL_TILES=parallel_tiles,
        BLOCK_TABLE_WINDOW_SIZE=_next_power_of_2(math.ceil(TILE_BLOCK_SIZE / min_block_size) + 1),
    )


def validate_multigroup(case_id: str, cases: tuple[Case, ...], *, graph_capture: bool = False) -> dict:
    inputs = make_multigroup_inputs(cases)
    if graph_capture:
        launch_multigroup(cases, inputs)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            launch_multigroup(cases, inputs)
        graph.replay()
    else:
        launch_multigroup(cases, inputs)
    torch.npu.synchronize()
    positions_lists, block_table_lists, _, _, slot_mappings, *_ = inputs
    comparisons = []
    passed = True
    errors = []
    for group_idx, (case, positions_list, block_table_list, actual) in enumerate(
        zip(cases, positions_lists, block_table_lists, slot_mappings)
    ):
        expected = torch.tensor(_reference(case, positions_list, block_table_list), dtype=torch.int32)
        actual_cpu = actual.cpu()
        group_passed = torch.equal(actual_cpu, expected)
        mismatches = int(torch.count_nonzero(actual_cpu != expected).item())
        max_abs = int(torch.max(torch.abs(actual_cpu.to(torch.int64) - expected.to(torch.int64))).item())
        passed = passed and group_passed
        comparisons.append(
            {
                "output": f"slot_mapping_group_{group_idx}",
                "max_abs": max_abs,
                "max_rel": 0.0 if group_passed else float(max_abs),
                "cosine": 1.0 if group_passed else 0.0,
                "mismatches": mismatches,
            }
        )
        if not group_passed:
            errors.append(f"group {group_idx}: {mismatches} mismatches")
    result = {
        "schema_version": 1,
        "case_id": case_id,
        "status": "passed" if passed else "numerical_mismatch",
        "comparisons": comparisons,
        "nan_match": True,
        "inf_match": True,
    }
    if errors:
        result["error"] = "; ".join(errors)
    return result


def validate(case: Case, *, graph_capture: bool = False) -> dict:
    (
        positions_list,
        block_table_list,
        query_start_loc,
        positions,
        block_table,
        slot_mapping,
    ) = _make_inputs(case)
    if graph_capture:
        _launch(case, query_start_loc, positions, block_table, slot_mapping)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            _launch(case, query_start_loc, positions, block_table, slot_mapping)
        graph.replay()
    else:
        _launch(case, query_start_loc, positions, block_table, slot_mapping)
    torch.npu.synchronize()
    expected = torch.tensor(_reference(case, positions_list, block_table_list), dtype=torch.int32)
    actual = slot_mapping.cpu()
    passed = torch.equal(actual, expected)
    mismatches = int(torch.count_nonzero(actual != expected).item())
    max_abs = int(torch.max(torch.abs(actual.to(torch.int64) - expected.to(torch.int64))).item())
    result = {
        "schema_version": 1,
        "case_id": case.id,
        "status": "passed" if passed else "numerical_mismatch",
        "comparisons": [
            {
                "output": "slot_mapping",
                "max_abs": max_abs,
                "max_rel": 0.0 if passed else float(max_abs),
                "cosine": 1.0 if passed else 0.0,
                "mismatches": mismatches,
            }
        ],
        "nan_match": True,
        "inf_match": True,
        "case": asdict(case),
    }
    if not passed:
        first = int(torch.nonzero(actual != expected, as_tuple=False)[0].item())
        result["error"] = (
            f"first mismatch at {first}: actual={actual[first].item()} "
            f"expected={expected[first].item()}, mismatches={mismatches}"
        )
    return result


def benchmark(case: Case, *, warmup: int, samples: int, inner_loops: int) -> dict:
    _, _, query_start_loc, positions, block_table, slot_mapping = _make_inputs(case)
    for _ in range(warmup):
        _launch(case, query_start_loc, positions, block_table, slot_mapping)
    torch.npu.synchronize()

    device_us: list[float] = []
    wrapper_us: list[float] = []
    for _ in range(samples):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        host_start = time.perf_counter_ns()
        start_event.record()
        for _ in range(inner_loops):
            _launch(case, query_start_loc, positions, block_table, slot_mapping)
        end_event.record()
        torch.npu.synchronize()
        host_end = time.perf_counter_ns()
        device_us.append(start_event.elapsed_time(end_event) * 1000.0 / inner_loops)
        wrapper_us.append((host_end - host_start) / 1000.0 / inner_loops)

    return {
        "schema_version": 1,
        "case_id": case.id,
        "case": asdict(case),
        "warmup": warmup,
        "samples": samples,
        "inner_loops": inner_loops,
        "device_us": device_us,
        "wrapper_us": wrapper_us,
        "median_us": statistics.median(device_us),
        "mean_us": statistics.mean(device_us),
        "min_us": min(device_us),
        "max_us": max(device_us),
        "cv": statistics.pstdev(device_us) / statistics.mean(device_us),
        "wrapper_median_us": statistics.median(wrapper_us),
    }


def benchmark_compare(case: Case, *, warmup: int, samples: int, inner_loops: int) -> dict:
    _, _, query_start_loc, positions, block_table, slot_mapping = _make_inputs(case)
    launchers = {"original": _launch_original, "current": _launch}
    for launcher in launchers.values():
        for _ in range(warmup):
            launcher(case, query_start_loc, positions, block_table, slot_mapping)
    torch.npu.synchronize()

    device_us: dict[str, list[float]] = {name: [] for name in launchers}
    wrapper_us: dict[str, list[float]] = {name: [] for name in launchers}
    for sample_idx in range(samples):
        order = ("original", "current") if sample_idx % 2 == 0 else ("current", "original")
        for name in order:
            start_event = torch.npu.Event(enable_timing=True)
            end_event = torch.npu.Event(enable_timing=True)
            torch.npu.synchronize()
            host_start = time.perf_counter_ns()
            start_event.record()
            for _ in range(inner_loops):
                launchers[name](case, query_start_loc, positions, block_table, slot_mapping)
            end_event.record()
            torch.npu.synchronize()
            host_end = time.perf_counter_ns()
            device_us[name].append(start_event.elapsed_time(end_event) * 1000.0 / inner_loops)
            wrapper_us[name].append((host_end - host_start) / 1000.0 / inner_loops)

    measurements = {}
    for name in launchers:
        values = device_us[name]
        measurements[name] = {
            "device_us": values,
            "wrapper_us": wrapper_us[name],
            "median_us": statistics.median(values),
            "mean_us": statistics.mean(values),
            "min_us": min(values),
            "max_us": max(values),
            "cv": statistics.pstdev(values) / statistics.mean(values),
            "wrapper_median_us": statistics.median(wrapper_us[name]),
        }
    original_median = cast(float, measurements["original"]["median_us"])
    current_median = cast(float, measurements["current"]["median_us"])
    return {
        "schema_version": 1,
        "case_id": case.id,
        "case": asdict(case),
        "warmup": warmup,
        "samples": samples,
        "inner_loops": inner_loops,
        "measurements": measurements,
        "relative_improvement": (original_median - current_median) / original_median,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=(
            "validate",
            "validate-all",
            "validate-multigroup",
            "validate-multigroup-graph",
            "benchmark",
            "benchmark-all",
            "benchmark-compare-multirequest",
            "validate-multirequest",
            "validate-multirequest-graph",
        ),
        required=True,
    )
    parser.add_argument("--case", choices=sorted(CASES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--inner-loops", type=int, default=100)
    args = parser.parse_args()

    torch.npu.set_device(0)
    if args.action in {"validate-multigroup", "validate-multigroup-graph"}:
        graph_capture = args.action == "validate-multigroup-graph"
        results = {
            case_id: validate_multigroup(case_id, cases, graph_capture=graph_capture)
            for case_id, cases in MULTIGROUP_CASES.items()
        }
        payload = {"results": results}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "completed", "action": args.action, "output": str(args.output)}))
        return 0
    case_ids: Sequence[str]
    if args.action in {
        "benchmark-compare-multirequest",
        "validate-multirequest",
        "validate-multirequest-graph",
    }:
        case_ids = MULTIREQUEST_PERFORMANCE_CASES
    elif args.action.endswith("-all"):
        case_ids = sorted(CASES) if args.action == "validate-all" else PERFORMANCE_CASES
    else:
        if args.case is None:
            parser.error("--case is required for single-case actions")
        case_ids = (args.case,)

    results = {}
    for case_id in case_ids:
        case = CASES[case_id]
        if args.action.startswith("validate"):
            results[case_id] = validate(
                case,
                graph_capture=args.action == "validate-multirequest-graph",
            )
        elif args.action == "benchmark-compare-multirequest":
            results[case_id] = benchmark_compare(
                case,
                warmup=args.warmup,
                samples=args.samples,
                inner_loops=args.inner_loops,
            )
        else:
            results[case_id] = benchmark(
                case,
                warmup=args.warmup,
                samples=args.samples,
                inner_loops=args.inner_loops,
            )
    payload = results[case_ids[0]] if len(case_ids) == 1 else {"results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "completed", "action": args.action, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
