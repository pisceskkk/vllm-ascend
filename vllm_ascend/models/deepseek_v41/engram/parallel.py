# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Uniform Engram DP exchange, backported from vLLM f84b0c4bce.

For single-node PP=PCP=DCP=1, the existing DP group has exactly the
membership of upstream's node-local Engram DP group. Auxiliary collectives
use sibling device groups with the same membership and head ordering.
"""

import weakref

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_tensor_model_parallel_rank, get_tp_group
from vllm.forward_context import get_forward_context

# Upstream #56741 normalized the V4.1 model package name.
from vllm.models.deepseek_v41.common.engram import DEAD_ID
from vllm.triton_utils import tl, triton

from vllm_ascend.utils import get_potential_max_tokens, is_pd_decode_recompute_scheduler_enabled


def get_engram_dp_group():
    group = get_dp_group()
    return group if group.world_size > 1 else None


def get_engram_dp_size():
    group = get_engram_dp_group()
    return group.world_size if group is not None else 1


def _destroy_engram_aux_groups(groups):
    # A worker's distributed teardown may have already destroyed every group.
    if dist.is_initialized():
        for group in reversed(groups):
            dist.destroy_process_group(group)


class EngramAuxGroups:
    """Model-owned device groups for Engram's auxiliary producer.

    Construct on every world rank during model initialization, in DP then TP
    order, before capture or any rank-local dummy/idle decision. Distinct
    communicators allow the producer's collectives to precede their consumers
    without changing the main graph's attention/MoE collective order.
    """

    def __init__(self, *, dp_shared_memory=False):
        dp, tp = get_engram_dp_group(), get_tp_group()
        self.dp_size = 1 if dp is None or dp_shared_memory else dp.world_size
        self.tp_size = tp.world_size
        self.dp_group = self.tp_group = None
        groups = []
        try:
            if self.dp_size > 1:
                self.dp_group = dp.make_sibling_device_group(group_desc="engram_aux_dp")
                groups.append(self.dp_group)
            if self.tp_size > 1:
                self.tp_group = tp.make_sibling_device_group(group_desc="engram_aux_tp")
                groups.append(self.tp_group)
        except Exception:
            _destroy_engram_aux_groups(groups)
            raise
        self._finalizer = weakref.finalize(self, _destroy_engram_aux_groups, groups)
        # Worker shutdown destroys all process groups. Avoid invoking HCCL
        # after Python/torch shutdown; normal model reclamation still closes.
        self._finalizer.atexit = False

    def close(self):
        """Release groups after all producer and consumer work has retired."""
        self._finalizer()

    def _gather(self, values, group, size):
        if size == 1:
            return values
        values = values.contiguous()
        gathered = values.new_empty((size * values.shape[0], *values.shape[1:]))
        # All members of this group have the same shape. An empty TP replica
        # has no rows to exchange, even while other DP replicas are active.
        if values.numel():
            dist.all_gather_into_tensor(gathered, values, group=group)
        return gathered

    def gather_dp(self, values):
        return self._gather(values, self.dp_group, self.dp_size)

    def gather_tp_heads(self, values):
        if self.tp_size == 1:
            return self._gather(values, self.tp_group, self.tp_size)
        gathered = self._gather(values, self.tp_group, self.tp_size)
        tokens, heads, dim = values.shape
        rows = values.new_empty((tokens, self.tp_size * heads, dim))
        # HCCL writes [rank][token][head]. Reuse the row selector to write
        # [token][rank * head] directly, without a transpose/contiguous pair.
        _engram_select_rows(gathered, rows, tokens, 0, heads * dim)
        return rows


def engram_head_shard_rank() -> int:
    """This rank's slot among the hash-head shards of one engram table.

    TP-major, so the shards a DP gather brings in are contiguous heads and
    the following TP gather completes the head order.
    """
    dp_group = get_engram_dp_group()
    dp_size = dp_group.world_size if dp_group is not None else 1
    dp_rank = dp_group.rank_in_group if dp_group is not None else 0
    return get_tensor_model_parallel_rank() * dp_size + dp_rank


def engram_gathered_num_tokens() -> int:
    """Per-replica token slot for the node-local Engram DP group."""
    context = get_forward_context()
    if (
        not getattr(context, "in_profile_run", False)
        and not getattr(context, "engram_uniform_dp_warmup", False)
        and is_pd_decode_recompute_scheduler_enabled()
    ):
        # Recompute decode can skip the DP metadata all-reduce. Its token
        # vector then contains only local counts, including on idle ranks or
        # ranks using different graph buckets. Size both Engram exchanges
        # from the shared configuration, never from that local vector.
        config = get_current_vllm_config()
        scheduler = config.scheduler_config
        query_len = 1 + config.speculative_config.num_speculative_tokens if config.speculative_config else 1
        return max(
            get_potential_max_tokens(),
            min(scheduler.max_num_batched_tokens, scheduler.max_num_seqs * query_len),
        )
    dp_metadata = context.dp_metadata
    if dp_metadata is None:
        raise RuntimeError("a DP-shared engram table needs DP token metadata")
    group = get_engram_dp_group()
    assert group is not None
    # Engram groups are contiguous slices of the full DP group.
    start = get_dp_group().rank_in_group - group.rank_in_group
    return int(dp_metadata.num_tokens_across_dp_cpu[start : start + group.world_size].max())


def gather_engram_hashes(
    hash_ids: torch.Tensor, *, dp_shared_memory: bool = False, aux_groups: EngramAuxGroups | None = None
) -> torch.Tensor:
    """Collect the n-gram ids of every DP replica sharing one table.

    Replicas are padded to a common token slot, including when recompute
    decode skips DP metadata synchronization and local graph sizes differ.
    """
    dp_group = get_engram_dp_group()
    if dp_group is None or dp_shared_memory:
        return hash_ids
    slot = engram_gathered_num_tokens()
    if hash_ids.shape[0] > slot:
        raise ValueError("Engram token count exceeds the DP token slot")
    if hash_ids.shape[0] < slot:
        pad = hash_ids.new_full((slot - hash_ids.shape[0], *hash_ids.shape[1:]), DEAD_ID)
        hash_ids = torch.cat((hash_ids, pad))
    return dp_group.all_gather(hash_ids, dim=0) if aux_groups is None else aux_groups.gather_dp(hash_ids)


@triton.jit(do_not_specialize=["num_tokens", "token_start", "num_elements"])
def _engram_select_rows_kernel(
    gathered,
    output,
    num_tokens,
    token_start,
    num_elements,
    LOCAL_WIDTH: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tokens = token_start + offsets // WIDTH
    cols = offsets % WIDTH
    source = (cols // LOCAL_WIDTH * num_tokens + tokens) * LOCAL_WIDTH
    source += cols % LOCAL_WIDTH
    values = tl.load(gathered + source, (offsets < num_elements) & (tokens < num_tokens), other=0)
    tl.store(output + offsets, values, offsets < num_elements)


def _engram_select_rows(
    gathered: torch.Tensor,
    output: torch.Tensor,
    source_tokens: int,
    token_start: int,
    local_width: int,
) -> None:
    """Copy one token window out of a rank-major gathered buffer.

    Both gathers land rank-major ([rank][token][local width]); this walks the
    window the rank keeps and lays its ranks out side by side as width.
    """
    if output.numel() == 0:
        return
    _engram_select_rows_kernel[(triton.cdiv(output.numel(), 1024),)](
        gathered,
        output,
        source_tokens,
        token_start,
        output.numel(),
        local_width,
        output.shape[1] * output.shape[2],
        BLOCK_SIZE=1024,
    )


def _gather_engram_rows(
    staged: torch.Tensor, num_tokens: int, *, aux_groups: EngramAuxGroups | None = None
) -> torch.Tensor:
    """Exchange DP tokens for heads, retaining only this replica's tokens."""
    dp_group = get_engram_dp_group()
    assert dp_group is not None
    slot, remainder = divmod(staged.shape[0], dp_group.world_size)
    assert remainder == 0 and 0 <= num_tokens <= slot
    gathered = dp_group.all_gather(staged, dim=0) if aux_groups is None else aux_groups.gather_dp(staged)
    local_heads, dim = staged.shape[1:]
    rows = staged.new_empty((num_tokens, dp_group.world_size * local_heads, dim))
    _engram_select_rows(
        gathered,
        rows,
        staged.shape[0],
        dp_group.rank_in_group * slot,
        local_heads * dim,
    )
    return rows
