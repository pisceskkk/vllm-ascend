# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl
from vllm_ascend.ops.fused_moe.router.fused_topk_router import AscendFusedTopKRouter
from vllm_ascend.utils import load_custom_op_library

IMAGE_START = 129257
EXPERTS = 384
TOP_K = 6
SCALING = 1.5


@pytest.fixture(scope="module", autouse=True)
def _load_ops():
    load_custom_op_library()


def _inputs(dtype, route):
    torch.manual_seed(849)
    logits = (torch.randn(8, EXPERTS) * 2).to(dtype)
    # Exercise stable softplus, including very negative and positive logits.
    logits[0, :4] = torch.tensor([-80, -40, 40, 80], dtype=dtype)
    bias = (torch.randn(EXPERTS) * 3).to(dtype)
    bias_vl = torch.flip(bias, dims=[0]) if route == "vision" else None
    ids = torch.tensor([0, 11, 22, 33, 44, 55, 7, 19], dtype=torch.int64)
    if route == "vision":
        # Both sentinel boundaries, plus text immediately outside the range.
        ids = torch.tensor([IMAGE_START - 1, IMAGE_START, IMAGE_START + 4, IMAGE_START + 5, 11, 22, 33, 44])
    table = None
    if route != "dynamic":
        table = torch.arange((IMAGE_START + 6) * TOP_K, dtype=torch.int32)
        table = (table.reshape(-1, TOP_K) * 13 + 7) % EXPERTS
    return logits, bias, bias_vl, ids, table


def _reference(logits, bias, bias_vl, ids, table, image_start=IMAGE_START, image_count=5):
    # FP64 logaddexp remains stable at the large-magnitude inputs above.
    scores = torch.logaddexp(logits.double(), torch.zeros_like(logits, dtype=torch.float64)).sqrt()
    correction = torch.zeros_like(scores) if bias is None else bias.double().expand_as(scores)
    image = (ids >= image_start) & (ids < image_start + image_count)
    if bias_vl is not None:
        correction = torch.where(image[:, None], bias_vl.double(), correction)
    selected = (scores + correction).topk(TOP_K, dim=-1).indices
    if table is not None:
        lookup = torch.where(image, 0, ids) if bias_vl is not None else ids
        hashed = table[lookup].long()
        selected = torch.where(image[:, None], selected, hashed) if bias_vl is not None else hashed
    weights = scores.gather(1, selected)
    weights = weights / weights.sum(dim=-1, keepdim=True) * SCALING
    return weights.float(), selected.int()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("route", ["dynamic", "hash"])
@pytest.mark.parametrize("execution", ["eager", "graph"])
def test_dsv41_text_routing_custom_op(dtype, route, execution):
    logits, bias, bias_vl, ids, table = _inputs(dtype, route)
    expected_weights, expected_ids = _reference(logits, bias, bias_vl, ids, table)
    device_logits, device_bias = logits.npu(), bias.npu()
    device_ids = ids.npu() if table is not None else None
    device_table = table.npu() if table is not None else None

    def run():
        return torch.ops._C_ascend.moe_gating_top_k_hash(
            x=device_logits,
            k=TOP_K,
            bias=device_bias,
            input_ids=device_ids,
            tid2eid=device_table,
            k_group=1,
            group_count=1,
            routed_scaling_factor=SCALING,
            eps=1e-20,
            group_select_mode=1,
            renorm=0,
            norm_type=2,
            out_flag=False,
        )

    if execution == "graph":
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual_weights, actual_ids, _ = run()
        # Replay with changed logits, retaining the same captured allocations.
        logits = logits.roll(1, dims=0)
        device_logits.copy_(logits)
        expected_weights, expected_ids = _reference(logits, bias, bias_vl, ids, table)
        graph.replay()
    else:
        actual_weights, actual_ids, _ = run()
    torch.testing.assert_close(actual_ids.cpu(), expected_ids, rtol=0, atol=0)
    tolerance = 1e-5 if dtype == torch.float32 else 8e-3
    torch.testing.assert_close(actual_weights.cpu().float(), expected_weights, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("with_hash", [False, True])
def test_dsv41_vision_routing_replay_changes_row_kind(dtype, id_dtype, with_hash):
    torch.manual_seed(850)
    # More rows than vector cores exercises each core's pipelined row loop.
    rows, image_start, image_count = 149, 211, 3
    # Unique BF16-representable logits avoid ambiguous unbiased TopK ties.
    logits = (torch.stack([torch.randperm(EXPERTS) for _ in range(rows)]).float() - EXPERTS // 2) / 32
    logits = logits.to(dtype)
    bias_vl = (torch.randn(EXPERTS) * 3).to(dtype)
    # No text bias also covers the distinct unbiased text path.
    ids = torch.tensor([7, image_start - 1, image_start, image_start + 2, image_start + 3, 11], dtype=id_dtype)
    ids = ids.repeat((rows + ids.numel() - 1) // ids.numel())[:rows]
    table = None
    if with_hash:
        table = ((torch.arange((image_start + 4) * TOP_K).reshape(-1, TOP_K) * 13 + 7) % EXPERTS).int()
    device_logits, device_ids, device_bias_vl = logits.npu(), ids.npu(), bias_vl.npu()
    device_table = table.npu() if table is not None else None

    def run():
        return torch.ops._C_ascend.moe_gating_top_k_hash(
            x=device_logits,
            k=TOP_K,
            bias=None,
            input_ids=device_ids,
            tid2eid=device_table,
            k_group=1,
            group_count=1,
            routed_scaling_factor=SCALING,
            eps=1e-20,
            group_select_mode=1,
            renorm=0,
            norm_type=2,
            out_flag=False,
            bias_vl=device_bias_vl,
            image_sentinel_lo=image_start,
            image_sentinel_count=image_count,
        )

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual_weights, actual_ids, _ = run()
    for shift in [0, 1, 3]:
        replay_ids = ids.roll(shift)
        replay_logits = logits.roll(shift, dims=0)
        device_ids.copy_(replay_ids)
        device_logits.copy_(replay_logits)
        graph.replay()
        expected_weights, expected_ids = _reference(
            replay_logits, None, bias_vl, replay_ids, table, image_start, image_count
        )
        torch.testing.assert_close(actual_ids.cpu(), expected_ids, rtol=0, atol=0)
        tolerance = 1e-5 if dtype == torch.float32 else 8e-3
        torch.testing.assert_close(actual_weights.cpu().float(), expected_weights, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("route", ["dynamic", "hash", "vision"])
def test_dsv41_router_and_weighted_experts(route):
    """Run the real router and single-rank ID preparation before expert aggregation."""
    logits, bias, bias_vl, ids, table = _inputs(torch.float32, route)
    expected_weights, expected_ids = _reference(logits, bias, bias_vl, ids, table)
    router = AscendFusedTopKRouter(
        top_k=TOP_K,
        global_num_experts=EXPERTS,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=SCALING,
        e_score_correction_bias=bias.npu(),
        tid2eid=table.npu() if table is not None else None,
        bias_vl=bias_vl.npu() if bias_vl is not None else None,
    )
    hidden = torch.randn(8, 32)
    experts = torch.randn(EXPERTS, 32, 16) / 32**0.5
    config = VllmConfig()
    with set_current_vllm_config(config), set_forward_context(None, config):
        # Communication is real; a one-rank configuration needs no collectives.
        comm = AllGatherCommImpl(
            SimpleNamespace(
                experts_per_token=TOP_K,
                num_experts=EXPERTS,
                num_local_experts=EXPERTS,
                is_sequence_parallel=False,
                dp_size=1,
                pcp_size=1,
            )
        )
        _EXTRA_CTX.moe_comm_type = MoECommType.ALLGATHER
        _EXTRA_CTX.moe_comm_method = comm
        weights, selected = router._compute_routing(hidden.npu(), logits.npu(), torch.int32, input_ids=ids.npu())
    torch.testing.assert_close(selected.cpu(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), expected_weights, rtol=1e-5, atol=1e-5)

    # NPU batched expert execution versus independent CPU per-token accumulation.
    selected_experts = experts.npu()[selected.long()]
    per_expert = torch.matmul(hidden.npu()[:, None, None, :], selected_experts).squeeze(-2)
    actual = (per_expert * weights[..., None]).sum(dim=1)
    expected = torch.zeros(8, 16)
    for token in range(hidden.shape[0]):
        for slot in range(TOP_K):
            expected[token] += (hidden[token] @ experts[expected_ids[token, slot]]) * expected_weights[token, slot]
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=3e-4)
