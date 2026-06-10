from typing import TypeVar

import numpy as np
import torch
import torch_npu
from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.common_cp import AscendPCPMetadata
from vllm_ascend.attention.mla_v1 import MAX_O_PROJ_PREFETCH_SIZE, MLAPO_MAX_SUPPORTED_TOKENS
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadata, AscendSFAMetadataBuilder
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    enabling_mlapo,
    maybe_save_kv_layer_to_connector,
    split_decodes_and_prefills,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.utils import all_gather_async
from vllm_ascend.ops.layer_shard_linear import is_hidden_layer, reach_layer_for_shard_weight_series
from vllm_ascend.ops.triton.rope import rope_forward_triton_siso
from vllm_ascend.utils import (
    AscendDeviceType,
    enable_dsa_cp_with_pcp_shard,
    get_ascend_device_type,
    get_weight_prefetch_method,
)

M = TypeVar("M", bound=AscendSFAMetadata)
sfa_ag_stream = torch_npu.npu.Stream()


class AscendSFACPMetadataBuilder(AscendSFAMetadataBuilder):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, metadata_cls, supports_dcp_with_varlen)

        # In sfa, pcp prefill does not support mlapo
        self.enable_mlapo = enabling_mlapo(self.vllm_config)

        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.pcp_group = get_pcp_group().device_group if self.pcp_size > 1 else None

        self.dcp_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_size > 1 else 0
        self.dcp_group = get_dcp_group().device_group if self.dcp_size > 1 else None
        self.cp_local_block_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
        self.cp_virtual_block_size = self.cp_local_block_size * self.dcp_size * self.pcp_size
        self.block_size = (self.block_size * self.cp_virtual_block_size) // np.gcd(
            self.block_size, self.cp_virtual_block_size
        )
        self.slot_mapping_buf = torch.empty(
            (
                vllm_config.scheduler_config.max_num_batched_tokens
                + 2 * self.pcp_size * vllm_config.scheduler_config.max_num_seqs,
            ),
            dtype=torch.int32,
            device=device,
        )
        self.block_arange_buffer = torch.arange(self.pcp_size * self.dcp_size, dtype=torch.int32, device=device)

    def _compact_varlen_decode_slot_mapping(
        self,
        decode_slot_mapping: torch.Tensor,
        decode_query_lens: torch.Tensor,
    ) -> None:
        device = decode_slot_mapping.device
        decode_query_lens_cpu = decode_query_lens.to(device="cpu", dtype=torch.int64, non_blocking=True)
        total_valid_tokens = int(decode_query_lens_cpu.sum().item())
        if total_valid_tokens == 0:
            return
        decode_query_lens = decode_query_lens_cpu.to(device=device, dtype=torch.int64, non_blocking=True)

        req_spans = decode_query_lens * self.pcp_size
        req_starts = torch.cumsum(req_spans, dim=0) - req_spans

        token_offsets = torch.arange(total_valid_tokens, device=device, dtype=torch.int64)
        token_base = torch.cumsum(decode_query_lens, dim=0) - decode_query_lens
        token_offsets = token_offsets - torch.repeat_interleave(token_base, decode_query_lens)

        expanded_req_starts = torch.repeat_interleave(req_starts, decode_query_lens)
        valid_in_idx = expanded_req_starts + token_offsets * self.pcp_size
        valid_out_idx = expanded_req_starts + token_offsets

        valid_slots = decode_slot_mapping[valid_in_idx]
        decode_slot_mapping.fill_(-1)
        decode_slot_mapping.index_copy_(0, valid_out_idx, valid_slots)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendSFAMetadata:
        metadata_cls = super().build(common_prefix_len, common_attn_metadata, fast_build)
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.decode_threshold
        )
        num_reqs = common_attn_metadata.num_reqs
        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == common_attn_metadata.num_actual_tokens

        sfa_cp_metadata = self.build_cp_metadata(
            self.block_arange_buffer, metadata_cls.seq_lens, common_attn_metadata, num_decodes
        )
        metadata_cls.num_decode_tokens = num_decode_tokens
        metadata_cls.num_decodes = num_decodes
        metadata_cls.num_prefills = num_prefills
        actual_seq_lengths_query = metadata_cls.cum_query_lens
        if num_prefills > 0:
            assert sfa_cp_metadata is not None
            # Prefill uses a compact block view so it can all-gather only the
            # real KV blocks it needs instead of the request-scoped decode view.
            valid_block_ids, block_table_cp = self.build_prefill_compact_block_metadata(
                metadata_cls.block_table, num_decodes
            )
            sfa_cp_metadata.valid_block_ids = valid_block_ids
            sfa_cp_metadata.block_table_cp = block_table_cp
            sfa_cp_metadata.block_table_cp_repeat = block_table_cp.repeat(2, 1)
            # Mixed batches store decode requests first, so prefill cumulative
            # query lengths must be rebased to the prefill-only token range.
            if num_decode_tokens > 0:
                prefill_q_cum_seqlens = (
                    actual_seq_lengths_query[num_decodes:] - actual_seq_lengths_query[num_decodes - 1]
                )
            else:
                prefill_q_cum_seqlens = actual_seq_lengths_query
            assert sfa_cp_metadata is not None
            sfa_cp_metadata.prefill_q_cum_seqlens = prefill_q_cum_seqlens

        if self.pcp_size > 1:
            long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
            assert long_seq_metadata is not None
            num_actual_tokens_pcp_padded = long_seq_metadata.num_actual_tokens_pcp_padded
            self.slot_mapping_buf[:num_actual_tokens_pcp_padded].copy_(
                common_attn_metadata.slot_mapping[:num_actual_tokens_pcp_padded], non_blocking=True
            )
            self.slot_mapping_buf[:num_decode_tokens] = self.slot_mapping_buf[
                : num_decode_tokens * self.pcp_size : self.pcp_size
            ]
            self.slot_mapping_buf[num_decode_tokens : num_decode_tokens * self.pcp_size].fill_(-1)

            if self.speculative_config is not None and num_decodes > 0:
                # when mtp, pcp_allgather_restore_idx=[696,-1,697,-1,560,-1,561,-1,100,101,102],
                # slot_mapping should be [696,697,-1,-1,560,561,-1,-1,100,101,102]
                # corner case: decode requests in the same MTP batch can have
                # different query lengths when some drafts are clipped near
                # max_model_len, so compact slot_mapping by per-request length
                # instead of assuming each request has decode_threshold tokens.
                decode_query_lens = long_seq_metadata.query_lens_pcp_full_cpu[:num_decodes]
                decode_slot_mapping = self.slot_mapping_buf[: num_decode_tokens * self.pcp_size]
                self._compact_varlen_decode_slot_mapping(
                    decode_slot_mapping,
                    decode_query_lens,
                )
            # prefill slot mapping
            self.prefill_slot_mapping = None
            if num_prefills > 0:
                restore_idx = long_seq_metadata.pcp_allgather_restore_idx
                num_tokens_pcp_padded = len(restore_idx)

                inverse_idx = torch.empty_like(restore_idx)
                inverse_idx.scatter_(
                    0,
                    restore_idx,
                    torch.arange(num_tokens_pcp_padded, device=restore_idx.device, dtype=restore_idx.dtype),
                )

                prefill_start_in_slot_mapping = num_decode_tokens * self.pcp_size
                prefill_slots = self.slot_mapping_buf[prefill_start_in_slot_mapping:].clone()

                self.prefill_slot_mapping = torch.where(
                    inverse_idx < num_decode_tokens * self.pcp_size,
                    torch.tensor(-1, device=restore_idx.device, dtype=torch.long),
                    prefill_slots[inverse_idx - num_decode_tokens * self.pcp_size],
                )
            metadata_cls.slot_mapping = self.slot_mapping_buf[:num_actual_tokens_pcp_padded]
            metadata_cls.prefill_slot_mapping = (
                self.prefill_slot_mapping if hasattr(self, "prefill_slot_mapping") else None
            )
        metadata_cls.sfa_cp_metadata = sfa_cp_metadata
        return metadata_cls

    def build_prefill_compact_block_metadata(
        self, block_table: torch.Tensor, num_decodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefill_block_table = block_table[num_decodes:]
        valid_block_ids, new_block_table = prefill_block_table.flatten().unique(return_inverse=True)
        num_blocks = valid_block_ids.shape[0]
        # Remap prefill block ids to the compact KV buffer after CP all-gather.
        block_table_cp = (
            new_block_table.unsqueeze(-1).to(prefill_block_table)
            + (self.block_arange_buffer * num_blocks).view(1, 1, -1).to(prefill_block_table)
        ).reshape(prefill_block_table.shape[0], -1)
        return valid_block_ids, block_table_cp

    def build_cp_metadata(
            self,
            block_arange: torch.Tensor,
            seq_lens: torch.Tensor,
            common_attn_metadata: AscendCommonAttentionMetadata,
            num_decodes: int,
    ) -> AscendPCPMetadata | None:
        common_long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
        assert common_long_seq_metadata is not None
        num_reqs = common_attn_metadata.num_reqs
        query_lens = (
            common_attn_metadata.query_start_loc_cpu[1 : num_reqs + 1]
            - common_attn_metadata.query_start_loc_cpu[:num_reqs]
        ).to(seq_lens.device)
        if common_attn_metadata.num_computed_tokens_cpu is not None:
            num_computed_tokens = common_attn_metadata.num_computed_tokens_cpu.to(seq_lens.device)
        else:
            # In async spec decode mode, num_computed_tokens_cpu is None
            # (CPU values are optimistic, GPU tensors are authoritative).
            # Compute from query_start_loc_cpu and seq_lens instead, matching
            # the approach in attention_cp.py and mla_v1.py.
            num_computed_tokens = seq_lens - query_lens.to(seq_lens.device)
        q_head_kv_lens = (query_lens // 2) * (self.pcp_rank + 1) + num_computed_tokens
        q_tail_kv_lens = query_lens * self.pcp_size - (query_lens // 2) * self.pcp_rank + num_computed_tokens
        full_overall_attn_seq_lens = torch.cat([q_head_kv_lens[num_decodes:], q_tail_kv_lens[num_decodes:]], dim=0)
        attn_mask_full_seqlens = torch.tensor(
            common_long_seq_metadata.attn_mask_full_seqlens,
            device=full_overall_attn_seq_lens.device,
            dtype=full_overall_attn_seq_lens.dtype,
        )

        return AscendPCPMetadata(
            q_head_idx=common_long_seq_metadata.q_head_idx_tensor,
            q_tail_idx=common_long_seq_metadata.q_tail_idx_tensor,
            q_head_tail_idx=common_long_seq_metadata.q_head_tail_idx_tensor,
            q_full_idx=common_long_seq_metadata.q_full_idx,
            attn_mask_full_seqlens=attn_mask_full_seqlens,
            head_attn_nomask_seqlens=q_head_kv_lens,
            tail_attn_nomask_seqlens=q_tail_kv_lens,
            full_overall_attn_seq_lens=full_overall_attn_seq_lens,
            pcp_allgather_restore_idx=common_long_seq_metadata.pcp_allgather_restore_idx,
            block_arange=block_arange,
        )


class AscendSFACPImpl(AscendSFAImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # Supports forward using the all-gather o_proj weight when PCP shard is enabled.
    o_proj_pcp_full_pool: torch.Tensor | None = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **kwargs,
    ):
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **kwargs,
        )
        # In sfa, pcp prefill does not support mlapo
        self.enable_mlapo = enabling_mlapo(self.vllm_config)
        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.pcp_group = get_pcp_group().device_group if self.pcp_size > 1 else None

        self.dcp_size = get_dcp_group().world_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_size > 1 else 0
        self.dcp_group = get_dcp_group().device_group if self.dcp_size > 1 else None
        self.local_num_heads = self.num_heads

        # use original PCP o_proj weight in PD mix stage, and full gather
        # for o_proj weight for prefill stage.
        self.enable_dsa_cp_with_pcp_shard = enable_dsa_cp_with_pcp_shard()
        if self.enable_dsa_cp_with_pcp_shard:
            self.pcp_shard_size = get_pcp_group().world_size

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        if self.enable_dsa_cp_with_pcp_shard:
            self._init_o_proj_pcp_shard_params()

    def _init_o_proj_pcp_shard_params(self):
        """Initialize shard-mode and Full-mode parameters for o_proj weight in PCP shard mode.

        Each PCP rank holds 1/pcp_shard_size of o_proj weight (sharded along input_dim).
        At compute time, all-gather across PCP ranks to reconstruct full weight.
        """
        if AscendSFACPImpl.o_proj_pcp_full_pool is None:
            sample = self.o_proj.weight
            AscendSFACPImpl.o_proj_pcp_full_pool = torch.empty(
                (sample.shape[0] * self.pcp_shard_size, sample.shape[1]),
                dtype=sample.dtype,
                device=sample.device,
            )

        # Save shard-mode parameters (PCP-sharded weights)
        self.o_proj_pcp_shard_weight = self.o_proj.weight.clone().detach()
        self.o_proj_pcp_shard_aclnn_input_scale = self.o_proj.aclnn_input_scale.clone().detach()
        self.o_proj_pcp_shard_aclnn_input_scale_reciprocal = self.o_proj.aclnn_input_scale_reciprocal.clone().detach()
        self.o_proj_pcp_shard_aclnn_input_offset = self.o_proj.aclnn_input_offset.clone().detach()

        # Initially switch to PCP mode for graph capture
        self.o_proj.weight.set_(self.o_proj_pcp_shard_weight)
        self.o_proj.aclnn_input_scale.set_(self.o_proj_pcp_shard_aclnn_input_scale)
        self.o_proj.aclnn_input_scale_reciprocal.set_(self.o_proj_pcp_shard_aclnn_input_scale_reciprocal)
        self.o_proj.aclnn_input_offset.set_(self.o_proj_pcp_shard_aclnn_input_offset)

        # Precompute Full-mode quantization parameters by repeating shard parameters across all PCP ranks
        self.o_proj_pcp_full_aclnn_input_scale = self.o_proj.aclnn_input_scale.repeat(self.pcp_shard_size)
        self.o_proj_pcp_full_aclnn_input_scale_reciprocal = self.o_proj.aclnn_input_scale_reciprocal.repeat(
            self.pcp_shard_size
        )
        self.o_proj_pcp_full_aclnn_input_offset = self.o_proj.aclnn_input_offset.repeat(self.pcp_shard_size)

    def _execute_sparse_flash_attention_cp_process(
        self,
        ql_nope,
        q_pe,
        decode_kv,
        decode_block_num,
        prefill_kv,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
    ):
        if decode_kv is not None and decode_kv[0] is not None:
            assert len(decode_kv) == 2
            (decode_k_nope, decode_k_rope) = decode_kv
        if prefill_kv is not None and prefill_kv[0] is not None:
            assert len(prefill_kv) == 2
            (prefill_k_nope, prefill_k_rope) = prefill_kv

        assert attn_metadata.sfa_cp_metadata is not None
        sfa_cp_metadata = attn_metadata.sfa_cp_metadata
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefills = attn_metadata.num_prefills
        decode_attn_out = None
        if num_decode_tokens > 0:
            decode_block_table_src = attn_metadata.block_table[:num_decodes]
            decode_block_table = self.gather_block_table(
                decode_block_num, decode_block_table_src, sfa_cp_metadata.block_arange
            )
            decode_attn_out = self._execute_sparse_flash_attention(
                ql_nope[:num_decode_tokens],
                q_pe[:num_decode_tokens],
                decode_k_nope,
                decode_k_rope,
                decode_block_table,
                topk_indices[:num_decode_tokens],
                actual_seq_lengths_query[:num_decodes],
                actual_seq_lengths_key[:num_decodes],
            )

        if num_prefills < 1:
            return self._align_to_graph_bucket_tokens(decode_attn_out, attn_metadata)

        prefill_valid_block_ids = sfa_cp_metadata.valid_block_ids
        prefill_block_table = sfa_cp_metadata.block_table_cp
        prefill_block_table_cp = sfa_cp_metadata.block_table_cp_repeat
        assert prefill_valid_block_ids is not None and prefill_block_table is not None
        prefill_ql_nope = ql_nope[num_decode_tokens:]
        prefill_q_pe = q_pe[num_decode_tokens:]
        prefill_topk_indices = topk_indices[num_decode_tokens:]
        prefill_actual_seq_lengths_key = actual_seq_lengths_key[num_decodes:]
        if attn_metadata.prefill_allgather_kv_event is not None:
            attn_metadata.prefill_allgather_kv_event.wait()
        if self.pcp_size == 1:
            prefill_attn_out = self._execute_sparse_flash_attention(
                prefill_ql_nope,
                prefill_q_pe,
                prefill_k_nope,
                prefill_k_rope,
                prefill_block_table,
                prefill_topk_indices,
                sfa_cp_metadata.prefill_q_cum_seqlens,
                prefill_actual_seq_lengths_key,
            )
            if decode_attn_out is not None:
                prefill_attn_out = torch.cat([decode_attn_out, prefill_attn_out], dim=0)
            return self._align_to_graph_bucket_tokens(prefill_attn_out, attn_metadata)

        # q split for head and tail
        q_head_tail_idx = sfa_cp_metadata.q_head_tail_idx

        # q head + tail compute
        full_overall_attn_seq_lens = sfa_cp_metadata.full_overall_attn_seq_lens
        q_head_tail_topk_indices = self._execute_sparse_flash_attention(
            torch.index_select(prefill_ql_nope, 0, q_head_tail_idx),
            torch.index_select(prefill_q_pe, 0, q_head_tail_idx),
            prefill_k_nope,
            prefill_k_rope,
            prefill_block_table_cp,
            torch.index_select(prefill_topk_indices, 0, q_head_tail_idx),
            sfa_cp_metadata.attn_mask_full_seqlens,
            full_overall_attn_seq_lens,
        )

        q_full_idx = sfa_cp_metadata.q_full_idx
        attn_output = torch.index_select(q_head_tail_topk_indices, 0, q_full_idx)

        if decode_attn_out is not None:
            attn_output = torch.cat([decode_attn_out, attn_output], dim=0)
        return self._align_to_graph_bucket_tokens(attn_output, attn_metadata)

    def _align_to_graph_bucket_tokens(self, attn_output: torch.Tensor | None, attn_metadata: M) -> torch.Tensor | None:
        if attn_output is None or self.pcp_size == 1:
            return attn_output
        # In graph mode, output buffer uses graph bucket token size
        # (forward_context.num_tokens), while PCP path may compute only valid
        # tokens. Align to the larger one to avoid later write-back mismatch.
        forward_context = get_forward_context()
        target_tokens = max(
            attn_metadata.num_input_tokens,
            forward_context.num_tokens if forward_context is not None else 0,
        )

        if attn_output.shape[0] == target_tokens:
            return attn_output
        aligned = torch.zeros(
            (target_tokens, *attn_output.shape[1:]),
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        valid_tokens = min(attn_output.shape[0], target_tokens)
        aligned[:valid_tokens] = attn_output[:valid_tokens]
        return aligned

    def _execute_sparse_flash_attention(
        self, ql_nope, q_pe, kv, key_rope, block_table, topk_indices, actual_seq_lengths_query, actual_seq_lengths_key
    ):
        attn_output, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=ql_nope,
            key=kv,
            value=kv,
            sparse_indices=topk_indices,
            scale_value=self.scale,
            sparse_block_size=1,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_key,
            query_rope=q_pe,
            key_rope=key_rope,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
        )
        return attn_output

    def gather_kv_cross_cp(self, kv_cache: torch.Tensor, block_tables: torch.Tensor) -> tuple[torch.Tensor, int]:
        # Note(qcs): we need set kv_cache_interleave_size = block_size for sfa!!!
        # Decode path uses request-scoped KV: first select the blocks referenced
        # by its block table, then all-gather only that request-local view.
        req_kv_cache = torch.index_select(kv_cache, 0, block_tables.flatten())
        block_num = req_kv_cache.shape[0]
        if self.dcp_size > 1:
            req_kv_cache = get_dcp_group().all_gather(req_kv_cache, 0)
        if self.pcp_size > 1:
            req_kv_cache = get_pcp_group().all_gather(req_kv_cache, 0)
        return req_kv_cache, block_num

    def gather_kv_cross_cp_compact(self, kv_cache: torch.Tensor, valid_block_ids: torch.Tensor) -> torch.Tensor:
        # prefill path uses compact KV: valid_block_ids
        kv_cache = torch.index_select(kv_cache, 0, valid_block_ids)
        if self.dcp_size > 1:
            kv_cache = get_dcp_group().all_gather(kv_cache, 0)
        if self.pcp_size > 1:
            kv_cache = get_pcp_group().all_gather(kv_cache, 0)
        return kv_cache

    def gather_block_table(self, block_num: int, block_tables: torch.Tensor, block_arange: torch.Tensor):
        # Remap original block ids to positions in the request-scoped KV buffer
        # generated by gather_kv_cross_cp().
        new_block_tables = torch.arange(block_tables.numel(), device=block_tables.device).view(block_tables.shape)
        block_tables = (
            (new_block_tables.unsqueeze(-1) + (block_arange * block_num).view(1, 1, -1).to(block_tables))
            .reshape(block_tables.shape[0], -1)
            .to(block_tables.dtype)
        )
        return block_tables

    def indexer_select_pre_process(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        kw, _ = self.wk_weights_proj(x)
        k_li = kw[:, : self.head_dim]
        weight = kw[:, self.head_dim :]
        k_li = self.k_norm(k_li).unsqueeze(1)
        k_li = k_li.view(-1, 1, self.head_dim)

        if HAS_TRITON:
            cos = cos.view(-1, self.qk_rope_head_dim)
            sin = sin.view(-1, self.qk_rope_head_dim)
            k_li = rope_forward_triton_siso(
                k_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        else:
            k_li_pe, k_li_nope = torch.split(
                k_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
            )

            cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
            sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

            k_li_pe = k_li_pe.unsqueeze(2)
            k_li_pe = torch_npu.npu_rotary_mul(k_li_pe, cos, sin)
            k_li_pe = k_li_pe.squeeze(2)

            k_li = torch.cat([k_li_pe, k_li_nope], dim=-1)  # [b*s,128]

        if self.use_sparse_c8_indexer:
            k_li = k_li @ AscendSFAImpl.k_hadamard
            k_li, k_li_scale = torch_npu.npu_dynamic_quant(k_li.view(-1, self.head_dim), dst_type=self.c8_k_cache_dtype)
            k_li_scale = k_li_scale.to(self.c8_k_scale_cache_dtype)  # [b*s,]
            k_li_scale = k_li_scale.unsqueeze(-1)  # [b*s,1]
        else:
            k_li_scale = None

        return k_li, k_li_scale, weight

    def indexer_select_cp_post_process(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        q_c: torch.Tensor,
        prefill_kvcache: torch.Tensor,
        decode_kvcache: torch.Tensor,
        decode_block_num: int,
        attn_metadata: M,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ):
        if isinstance(q_c, tuple):
            # MLAPO in C8 scenario already quantizes q_c to MXFP8 (fp8) with a per-token scale.
            # Skip wq_b's internal npu_dynamic_mx_quant and directly use the
            # pre-quantized values in npu_quant_matmul to avoid double quantization.
            q_c_tensor, q_c_scale = q_c
            q_c_tensor = q_c_tensor.view(-1, q_c_tensor.shape[-1])
            if q_c_scale.dim() == 2:
                q_c_scale = q_c_scale.view(q_c_scale.shape[0], -1, 2)
            q_li = torch_npu.npu_quant_matmul(
                q_c_tensor,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                pertoken_scale=q_c_scale,
                pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
                bias=None,
                output_dtype=x.dtype,
                group_sizes=[1, 1, getattr(self.wq_b.quant_method.quant_method, "group_size", 32)],
            )
        else:
            q_li, _ = self.wq_b(q_c)  # [b,s,1536] @ [1536,64*128] = [b,s,64*128]
        q_li = q_li.view(-1, self.n_head, self.head_dim)  # [n_toks,64,128]
        if HAS_TRITON:
            q_li = rope_forward_triton_siso(
                q_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        else:
            q_li_pe, q_li_nope = torch.split(
                q_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
            )  # [b,s,64,64+64]

            q_li_pe = q_li_pe.unsqueeze(2)
            q_li_pe = torch_npu.npu_rotary_mul(q_li_pe, cos, sin)
            q_li_pe = q_li_pe.squeeze(2)
            q_li = torch.cat([q_li_pe, q_li_nope], dim=-1)  # [b*s,64,128]

        assert attn_metadata.sfa_cp_metadata is not None
        sfa_cp_metadata = attn_metadata.sfa_cp_metadata
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefills = attn_metadata.num_prefills

        q_li_scale = None
        if self.use_sparse_c8_indexer:
            q_li_shape_ori = q_li.shape
            q_li = q_li @ AscendSFAImpl.q_hadamard
            q_li, q_li_scale = torch_npu.npu_dynamic_quant(q_li.view(-1, self.head_dim), dst_type=self.c8_k_cache_dtype)
            q_li_scale = q_li_scale.to(self.c8_k_scale_cache_dtype)  # [b*s,]
            q_li = q_li.view(q_li_shape_ori)
            q_li_scale = q_li_scale.view(q_li_shape_ori[:-1])

        decode_topk_indices = None
        if num_decode_tokens > 0:
            decode_block_table_src = attn_metadata.block_table[:num_decodes]
            decode_block_table = self.gather_block_table(
                decode_block_num, decode_block_table_src, sfa_cp_metadata.block_arange
            )
            decode_topk_indices = DeviceOperator.indexer_select_post_process(
                self,
                q_li[:num_decode_tokens],
                q_li_scale[:num_decode_tokens] if q_li_scale is not None else None,
                weights[:num_decode_tokens],
                decode_kvcache,
                decode_block_table,
                actual_seq_lengths_query[:num_decodes],
                actual_seq_lengths_key[:num_decodes],
                self.use_sparse_c8_indexer,
                self.use_torch_npu_lightning_indexer,
            )
        # prefill compute
        if num_prefills == 0:
            return decode_topk_indices

        prefill_block_table = sfa_cp_metadata.block_table_cp
        prefill_block_table_cp = sfa_cp_metadata.block_table_cp_repeat
        prefill_q_li = q_li[num_decode_tokens:]
        prefill_q_li_scale = q_li_scale[num_decode_tokens:] if q_li_scale is not None else None
        prefill_weights = weights[num_decode_tokens:]
        prefill_actual_seq_lengths_key = actual_seq_lengths_key[num_decodes:]
        if attn_metadata.prefill_allgather_kli_event is not None:
            attn_metadata.prefill_allgather_kli_event.wait()
        if self.pcp_size == 1:
            prefill_topk_indices = DeviceOperator.indexer_select_post_process(
                self,
                prefill_q_li,
                prefill_q_li_scale,
                prefill_weights,
                prefill_kvcache,
                prefill_block_table,
                sfa_cp_metadata.prefill_q_cum_seqlens,
                prefill_actual_seq_lengths_key,
                self.use_sparse_c8_indexer,
                self.use_torch_npu_lightning_indexer,
            )
            if decode_topk_indices is not None:
                prefill_topk_indices = torch.cat([decode_topk_indices, prefill_topk_indices], dim=0)
            return prefill_topk_indices

        # pcp split for head and tail
        q_head_tail_idx = sfa_cp_metadata.q_head_tail_idx

        # q head/tail compute
        full_overall_attn_seq_lens = sfa_cp_metadata.full_overall_attn_seq_lens
        prefill_q_li = torch.index_select(prefill_q_li, 0, q_head_tail_idx)
        prefill_q_li_scale = torch.index_select(prefill_q_li_scale, 0, q_head_tail_idx) if prefill_q_li_scale is not None else None
        q_head_tail_topk_indices = DeviceOperator.indexer_select_post_process(
            self,
            prefill_q_li,
            prefill_q_li_scale,
            torch.index_select(prefill_weights, 0, q_head_tail_idx),
            prefill_kvcache,
            prefill_block_table_cp,
            sfa_cp_metadata.attn_mask_full_seqlens,
            full_overall_attn_seq_lens,
            self.use_sparse_c8_indexer,
            self.use_torch_npu_lightning_indexer,
        )
        q_full_idx = sfa_cp_metadata.q_full_idx
        topk_indices = torch.index_select(q_head_tail_topk_indices, 0, q_full_idx)

        if decode_topk_indices is not None:
            topk_indices = torch.cat([decode_topk_indices, topk_indices], dim=0)
        return topk_indices

    def exec_kv_pre(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        attn_metadata: M,
    ):
        kv_c, k_pe = kv_no_split.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())  # type: ignore[misc]
        assert len(kv_cache) > 1, "the number of kv cache should be greater than 1, namely (nope_cache and rope_cache)"
        assert attn_metadata.sfa_cp_metadata is not None
        kv_c_normed = kv_c_normed.view([kv_c_normed.shape[0], self.num_kv_heads, -1])
        k_pe = k_pe.unsqueeze(1)
        k_pe = self.rope_single(k_pe, cos, sin)
        return kv_c_normed, k_pe

    def exec_kv_decode(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: tuple,
        attn_metadata: M,
    ):
        k_pe = k_pe.view(k_pe.shape[0], self.num_kv_heads, -1)
        kv_c_normed = kv_c_normed.view([kv_c_normed.shape[0], self.num_kv_heads, -1])
        num_decode_tokens = attn_metadata.num_decode_tokens
        torch_npu._npu_reshape_and_cache(
            key=kv_c_normed[:num_decode_tokens],
            value=k_pe[:num_decode_tokens],
            key_cache=kv_cache[0],
            value_cache=kv_cache[1],
            slot_indices=attn_metadata.slot_mapping[:num_decode_tokens],
        )
        return None, None

    def exec_kv_prefill(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: tuple,
        attn_metadata: M,
    ):
        assert attn_metadata.prefill_slot_mapping is not None
        k_pe = k_pe.view(k_pe.shape[0], self.num_kv_heads, -1)
        kv_c_normed = kv_c_normed.view([kv_c_normed.shape[0], self.num_kv_heads, -1])
        torch_npu._npu_reshape_and_cache(
            key=kv_c_normed,
            value=k_pe,
            key_cache=kv_cache[0],
            value_cache=kv_cache[1],
            slot_indices=attn_metadata.prefill_slot_mapping,
        )
        return None, None

    def _handle_o_proj_weight_switch_and_forward(
        self,
        attn_output: torch.Tensor,
        output: torch.Tensor,
        o_proj_pcp_handle: torch.distributed.Work | None,
        should_shard_weight: bool,
    ) -> tuple[torch.Tensor, bool]:
        """
        Handle o_proj weight switching between TP-mode and Full-mode, and execute forward computation.
        """
        # Gather o_proj weight from all TP ranks for Full-mode computation
        if should_shard_weight:
            if o_proj_pcp_handle is not None:
                o_proj_pcp_handle.wait()
            # row split
            # Switch o_proj to Full-mode (gathered weight from all PCP ranks)
            self.o_proj.weight.set_(AscendSFACPImpl.o_proj_pcp_full_pool)
            self.o_proj.aclnn_input_scale.set_(self.o_proj_pcp_full_aclnn_input_scale)
            self.o_proj.aclnn_input_scale_reciprocal.set_(self.o_proj_pcp_full_aclnn_input_scale_reciprocal)
            self.o_proj.aclnn_input_offset.set_(self.o_proj_pcp_full_aclnn_input_offset)

            # Apply quantization method and execute forward computation
            output[...] = self.o_proj.quant_method.quant_method.apply(self.o_proj, attn_output)

            # Switch o_proj back to shard-mode
            self.o_proj.weight.set_(self.o_proj_pcp_shard_weight)
            self.o_proj.aclnn_input_scale.set_(self.o_proj_pcp_shard_aclnn_input_scale)
            self.o_proj.aclnn_input_scale_reciprocal.set_(self.o_proj_pcp_shard_aclnn_input_scale_reciprocal)
            self.o_proj.aclnn_input_offset.set_(self.o_proj_pcp_shard_aclnn_input_offset)

            return output, False
        else:
            split_size = self.num_heads * self.v_head_dim // self.pcp_shard_size
            attn_output = attn_output[:, self.pcp_rank * split_size : (self.pcp_rank + 1) * split_size]
            return attn_output, True

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            if self.enable_dsa_cp_with_layer_shard and not _EXTRA_CTX.in_profile_run:
                for layer in self.layer_sharding_kwargs or []:
                    if is_hidden_layer(layer):
                        reach_layer_for_shard_weight_series(layer)
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        cos = attn_metadata.cos
        sin = attn_metadata.sin
        slot_mapping = attn_metadata.slot_mapping
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        if self.enable_sp:
            assert attn_metadata.dsa_cp_context is not None
            cos_cp = attn_metadata.dsa_cp_context.cos_cp
            sin_cp = attn_metadata.dsa_cp_context.sin_cp
        else:
            cos_cp, sin_cp = cos, sin
        actual_seq_lengths_query = attn_metadata.cum_query_lens
        actual_seq_lengths_key = attn_metadata.seq_lens

        # Inputs and outputs may be padded for CUDA graphs
        num_input_tokens = attn_metadata.num_input_tokens
        output_padded = output

        # all-gather o_proj weight for prefill stage of PD mix node
        o_proj_pcp_handle = None
        # if is PD mix stage, using original TP o_proj weight, and also need to full gather for o_proj
        # weight for prefill stage.
        full_gather_o_proj_enabled = self.enable_dsa_cp_with_pcp_shard and attn_metadata.attn_state not in {
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.SpecDecoding,
        }
        if attn_metadata.num_prefills > 0:
            attn_metadata.prefill_allgather_kli_event = torch.npu.Event()
            attn_metadata.prefill_allgather_kv_event = torch.npu.Event()
            attn_metadata.prefill_kv_cache_event = torch.npu.Event()
            attn_metadata.prefill_kli_cache_event = torch.npu.Event()

        # run mlapo ops when dsa-cp is disabled, and ensure that num_tokens satisfies the count limitation
        if self.enable_mlapo and num_input_tokens <= MLAPO_MAX_SUPPORTED_TOKENS:
            hidden_states, ql_nope, q_pe, q_c = self._sfa_preprocess_with_mlapo(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                cos=cos,
                sin=sin,
                slot_mapping=slot_mapping,
                num_input_tokens=num_input_tokens,
            )
            k_li, k_li_scale, weight = self.indexer_select_pre_process(x=hidden_states, cos=cos, sin=sin)
            wait_for_kv_layer_from_connector(layer_name)
        # native
        else:
            assert self.fused_qkv_a_proj is not None, "q lora is required for DSA."
            weight_prefetch_method = get_weight_prefetch_method()
            weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
                inputs=self.fused_qkv_a_proj.weight, dependency=hidden_states
            )
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_no_split = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            assert self.q_a_layernorm is not None, "q_a_layernorm must be initialized"
            q_c = self.q_a_layernorm(q_c)

            k_li, k_li_scale, weight = self.indexer_select_pre_process(x=hidden_states, cos=cos, sin=sin)

            wait_for_kv_layer_from_connector(layer_name)
            k_nope, k_pe = self.exec_kv_pre(kv_no_split, cos, sin, kv_cache, attn_metadata)

            if self.enable_sp:
                # sp AG
                fused_kv_no_split = torch.cat(
                    [k_nope.view(-1, k_nope.shape[-1]), k_pe.view(-1, k_pe.shape[-1]), k_li.view(-1, k_li.shape[-1])],
                    dim=1,
                )
                fused_kv_no_split = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                    fused_kv_no_split.contiguous(), need_gather_q_kv
                )
                k_nope, k_pe, k_li = fused_kv_no_split.split(
                    [self.kv_lora_rank, self.qk_rope_head_dim, self.head_dim], dim=-1
                )
                weight = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(weight.contiguous(), need_gather_q_kv)
                q_c = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(q_c.contiguous(), need_gather_q_kv)

            if attn_metadata.num_decodes > 0:
                _, _ = self.exec_kv_decode(k_nope, k_pe, kv_cache, attn_metadata)

            if not self.use_sparse_c8_indexer:
                fused_kv_no_split = torch.cat([k_nope.view(-1, k_nope.shape[-1]), k_pe.view(-1, k_pe.shape[-1]), k_li.view(-1, k_li.shape[-1]),], dim=1)[:num_actual_tokens]
                fused_kv_no_split, kv_pcp_ag_handle = all_gather_async(fused_kv_no_split, get_pcp_group())
            else:
                assert k_li_scale is not None
                fused_kv_no_split = torch.cat(
                    [
                        k_nope.view(-1, k_nope.shape[-1]),
                        k_pe.view(-1, k_pe.shape[-1]),
                    ],
                    dim=1,
                )[:num_actual_tokens]
                fused_kv_no_split, _ = all_gather_async(fused_kv_no_split, get_pcp_group())
                k_li, _ = all_gather_async(k_li.view(-1, k_li.shape[-1])[:num_actual_tokens], get_pcp_group())
                k_li_scale, kv_pcp_ag_handle = all_gather_async(k_li_scale.view(-1, k_li_scale.shape[-1])[:num_actual_tokens], get_pcp_group())

            ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
            q_pe = self.rope_single(q_pe, cos_cp, sin_cp)
            if kv_pcp_ag_handle is not None:
                kv_pcp_ag_handle.wait()
            if not self.use_sparse_c8_indexer:
                k_nope, k_pe, k_li = fused_kv_no_split.split([self.kv_lora_rank, self.qk_rope_head_dim, self.head_dim], dim=-1)
            else:
                k_nope, k_pe = fused_kv_no_split.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            if attn_metadata.num_prefills > 0:
                _, _ = self.exec_kv_prefill(k_nope, k_pe, kv_cache, attn_metadata)
                attn_metadata.prefill_kv_cache_event.record()
            k_li = self._get_full_kv(k_li, attn_metadata)

        if kv_cache is not None:
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
            if self.use_sparse_c8_indexer and get_ascend_device_type() == AscendDeviceType.A5:
                dsa_k_cache_idx = 1
                dsa_k_scale_cache_idx = 2
            else:
                dsa_k_cache_idx = 2
                dsa_k_scale_cache_idx = 3
            if num_decode_tokens > 0:
                torch_npu.npu_scatter_nd_update_(
                    kv_cache[dsa_k_cache_idx].view(-1, k_li.shape[-1]), slot_mapping[:num_decode_tokens].view(-1, 1), k_li[:num_decode_tokens].view(-1, k_li.shape[-1])
                )  # b, s, n, d
                if self.use_sparse_c8_indexer:
                    if get_ascend_device_type() == AscendDeviceType.A5:
                        assert len(kv_cache) == 3
                    else:
                        assert len(kv_cache) == 4
                    assert k_li_scale is not None
                    torch_npu.npu_scatter_nd_update_(
                        kv_cache[dsa_k_scale_cache_idx].view(-1, k_li_scale.shape[-1]),
                        slot_mapping[:num_decode_tokens].view(-1, 1),
                        k_li_scale[:num_decode_tokens].view(-1, k_li_scale.shape[-1]),
                    )
            if attn_metadata.num_prefills > 0:
                assert attn_metadata.prefill_slot_mapping is not None
                slot_mapping = attn_metadata.prefill_slot_mapping
                torch_npu.npu_scatter_nd_update_(
                    kv_cache[dsa_k_cache_idx].view(-1, k_li.shape[-1]), slot_mapping.view(-1, 1), k_li.view(-1, k_li.shape[-1])
                )  # b, s, n, d
                if self.use_sparse_c8_indexer:
                    if get_ascend_device_type() == AscendDeviceType.A5:
                        assert len(kv_cache) == 3
                    else:
                        assert len(kv_cache) == 4
                    assert k_li_scale is not None
                    torch_npu.npu_scatter_nd_update_(
                        kv_cache[dsa_k_scale_cache_idx].view(-1, k_li_scale.shape[-1]),
                        slot_mapping.view(-1, 1),
                        k_li_scale.view(-1, k_li_scale.shape[-1]),
                    )
                attn_metadata.prefill_kli_cache_event.record()
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()

        # allgather kv_cache block
        decode_key, decode_key_scale, decode_block_num, prefill_key, prefill_key_scale = None, None, None, None, None
        decode_kv, prefill_kv = (None, None), (None, None)
        if attn_metadata.num_decodes > 0:
            decode_block_table_src = attn_metadata.block_table[:attn_metadata.num_decodes]
            decode_key, decode_block_num = self.gather_kv_cross_cp(kv_cache[2], decode_block_table_src) # k_li
            if self.use_sparse_c8_indexer:
                assert len(kv_cache) == 4
                decode_key_scale, _ = self.gather_kv_cross_cp(kv_cache[3], decode_block_table_src) # k_li
            decode_k_nope, _ = self.gather_kv_cross_cp(kv_cache[0], decode_block_table_src) # k_nope
            decode_k_rope, _ = self.gather_kv_cross_cp(kv_cache[1], decode_block_table_src) # k_rope
            decode_kv = (decode_k_nope, decode_k_rope)
        if attn_metadata.num_prefills > 0:
            assert attn_metadata.sfa_cp_metadata is not None
            with torch_npu.npu.stream(sfa_ag_stream):
                prefill_valid_block_ids = attn_metadata.sfa_cp_metadata.valid_block_ids
                prefill_block_table = attn_metadata.sfa_cp_metadata.block_table_cp
                assert prefill_valid_block_ids is not None and prefill_block_table is not None
                attn_metadata.prefill_kli_cache_event.wait()
                prefill_key = self.gather_kv_cross_cp_compact(kv_cache[2], prefill_valid_block_ids) # k_li
                if self.use_sparse_c8_indexer:
                    assert len(kv_cache) == 4
                    prefill_key_scale = self.gather_kv_cross_cp_compact(kv_cache[3], prefill_valid_block_ids) # k_li
                attn_metadata.prefill_allgather_kli_event.record()
                attn_metadata.prefill_kv_cache_event.wait()
                prefill_k_nope = self.gather_kv_cross_cp_compact(kv_cache[0], prefill_valid_block_ids)  # k_nope
                prefill_k_rope = self.gather_kv_cross_cp_compact(kv_cache[1], prefill_valid_block_ids)  # k_rope
                attn_metadata.prefill_allgather_kv_event.record()
                prefill_kv = (prefill_k_nope, prefill_k_rope)

        # o-proj weight allgather
        if full_gather_o_proj_enabled:
            _, o_proj_pcp_handle = all_gather_async(
                self.o_proj_pcp_shard_weight,
                get_pcp_group(),
                output=AscendSFACPImpl.o_proj_pcp_full_pool,
            )

        topk_num_tokens = num_input_tokens or hidden_states.shape[0]
        if self.skip_topk:
            topk_indices = self._get_indexcache_topk_indices(topk_num_tokens)
        else:
            prefill_kvcache = (*prefill_kv, prefill_key, prefill_key_scale) if self.use_sparse_c8_indexer else (*prefill_kv, prefill_key)
            decode_kvcache = (*decode_kv, decode_key, decode_key_scale) if self.use_sparse_c8_indexer else (*decode_kv, decode_key)
            topk_indices = self.indexer_select_cp_post_process(
                x=hidden_states,
                weights=weight,
                q_c=q_c,
                prefill_kvcache=prefill_kvcache,
                decode_kvcache=decode_kvcache,
                decode_block_num=decode_block_num,
                attn_metadata=attn_metadata,
                cos=cos_cp,
                sin=sin_cp,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
            )
            if self.use_index_cache:
                self._update_indexcache_topk_indices(topk_indices)

        attn_output = self._execute_sparse_flash_attention_cp_process(
            ql_nope, q_pe, decode_kv, decode_block_num, prefill_kv, topk_indices, attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key
        )

        attn_output = self._v_up_proj(attn_output)
        weight_prefetch_method = get_weight_prefetch_method()
        weight_prefetch_method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
            inputs=self.o_proj.weight,
            dependency=attn_output,
            max_size=MAX_O_PROJ_PREFETCH_SIZE,
            linear_layer=self.o_proj,
        )

        if self.enable_dsa_cp_with_pcp_shard:
            # 1. prefill: o_proj is a TP(actually PCP) weight, we need to all-gather o_proj weight to switch TP=1.
            # 2. decode: all-to-all the hidden_state before the o_proj forward.
            result, require_o_proj_forward = self._handle_o_proj_weight_switch_and_forward(
                attn_output=attn_output,
                output=output,
                o_proj_pcp_handle=o_proj_pcp_handle,
                should_shard_weight=full_gather_o_proj_enabled,
            )
            if not require_o_proj_forward:
                return result
            attn_output = result

        if self.enable_dsa_cp_with_pcp_shard:
            quant_method = getattr(self.o_proj.quant_method, "quant_method", self.o_proj.quant_method)
            outputo = quant_method.apply(self.o_proj, attn_output, tp_rank=self.pcp_rank)
            outputo = get_pcp_group().all_reduce(outputo)
        else:
            outputo = self.o_proj(attn_output)[0]

        output[...] = outputo

        maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))

        return output_padded
