# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend SFA execution over upstream HiSparse's resident and hot cache pool."""

from dataclasses import dataclass

import torch
import torch_npu
from vllm.v1.attention.backends.mla.index_group import HiSparseMLAIndexGroup, SparseMLAIndexGroupBuilder
from vllm.v1.hisparse.runtime import HiSparsePrefillStagingPlan, build_hisparse_prefill_staging_plan
from vllm.v1.worker.workspace import current_workspace_manager

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadata, AscendSFAMetadataBuilder
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

# Bound the temporary D2D packing workspace independently of prefill length.
_ATTENTION_CHUNK_ROWS = 32


@dataclass
class AscendHiSparsePrefillMetadata:
    host_staging_plan: HiSparsePrefillStagingPlan


@dataclass
class AscendHiSparseMetadata(AscendSFAMetadata):
    req_id_per_token: torch.Tensor | None = None
    num_reqs: int = 0
    prefill: AscendHiSparsePrefillMetadata | None = None


class AscendHiSparseMetadataBuilder(AscendSFAMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, **kwargs):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, metadata_cls=AscendHiSparseMetadata, **kwargs)
        self._request_row_buffers: dict[int | None, torch.Tensor] = {}
        self._row_capacity = max(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )

    def _build(self, common_attn_metadata, draft_index=None, pcp_context=None):
        metadata = super()._build(common_attn_metadata, draft_index, pcp_context)
        metadata.num_reqs = common_attn_metadata.num_reqs
        rows = torch.arange(metadata.num_input_tokens, device=self.device, dtype=torch.int32)
        if draft_index not in self._request_row_buffers:
            self._request_row_buffers[draft_index] = torch.empty(
                self._row_capacity, device=self.device, dtype=torch.int32
            )
        metadata.req_id_per_token = self._request_row_buffers[draft_index][: metadata.num_input_tokens]
        metadata.req_id_per_token.copy_(torch.searchsorted(metadata.query_start_loc[1:].contiguous(), rows, right=True))
        metadata.req_id_per_token.clamp_(max=max(metadata.num_reqs - 1, 0))
        if metadata.num_prefills:
            seq_lens_cpu = metadata.seq_lens_cpu
            if seq_lens_cpu is None:
                # Capacity may use an upper bound; the staging plan receives
                # current device lengths to decide the actual copied rows.
                seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            if seq_lens_cpu is None:
                raise RuntimeError("HiSparse prefill requires scheduler sequence-length bounds on CPU")
            prefill_lens = seq_lens_cpu[metadata.num_decodes :]
            capacity = int(((prefill_lens + metadata.block_size - 1) // metadata.block_size).sum())
            metadata.prefill = AscendHiSparsePrefillMetadata(
                build_hisparse_prefill_staging_plan(
                    metadata.block_table[metadata.num_decodes :],
                    metadata.seq_lens[metadata.num_decodes :],
                    metadata.block_size,
                    capacity,
                )
            )
        return metadata


class AscendHiSparseImpl(AscendSFAImpl):
    """Keep HiSparse residency decisions on device, with an SFA layout bridge."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.enable_sparse_sfa_c8 or self.enable_sparse_sfa_turboquant:
            raise ValueError("Ascend HiSparse currently requires an unquantized main MLA KV cache")
        parallel = self.vllm_config.parallel_config
        if parallel.prefill_context_parallel_size > 1 or parallel.decode_context_parallel_size > 1:
            raise ValueError("Ascend HiSparse does not support context parallelism")
        builder = kwargs.get("index_group_builder")
        if builder is None:
            if self.topk_indices_buffer is None:
                raise ValueError("Ascend HiSparse requires a model sparse-index group")
            builder = SparseMLAIndexGroupBuilder(self.topk_indices_buffer)
        self.index_group, self.index_group_index = builder.register_layer(
            self.has_indexer and not self.skip_topk,
            self.vllm_config,
            row_width=self.kv_lora_rank + self.qk_rope_head_dim,
            kv_cache_dtype=self.vllm_config.cache_config.cache_dtype,
        )
        if not isinstance(self.index_group, HiSparseMLAIndexGroup):
            raise RuntimeError("Ascend HiSparse was initialized without a HiSparse index group")
        self.hisparse_cache = self.index_group.cache(self.index_group_index)
        self.topk = self.index_group.logical_topk_indices.shape[-1]
        if self.topk % 128:
            raise ValueError("Ascend HiSparse SFA requires top-k to be divisible by 128")
        max_rows = max(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            self.vllm_config.compilation_config.max_cudagraph_capture_size or 0,
        )
        self._kv_staging_capacity = (max_rows + 127) // 128 * 128
        self._packed_resident_supported = (
            get_current_hardware_profile().supports(HardwareCapability.SFA_PACKED_KV)
            and self.kv_lora_rank == 512
            and self.qk_rope_head_dim == 64
        )

    def _fused_preprocess_type(self):
        # Fused prologs write the ordinary split cache directly. HiSparse owns
        # both the resident destination and optional host-mirror staging. This
        # also keeps quantized linear weight layout under its normal owner.
        return None

    def _compose_sfa_kv_cache(self, kv_cache):
        return (kv_cache,)

    def _workspace(self, dtype):
        key = ("ascend_hisparse_attention", self.topk, self.kv_lora_rank, self.qk_rope_head_dim, dtype)

        def allocate():
            device = self.index_group.logical_topk_indices.device
            shape = (_ATTENTION_CHUNK_ROWS, self.topk, 1)
            return (
                torch.empty((*shape, self.kv_lora_rank), dtype=dtype, device=device),
                torch.empty((*shape, self.qk_rope_head_dim), dtype=dtype, device=device),
                torch.arange(_ATTENTION_CHUNK_ROWS * self.topk // 128, dtype=torch.int32, device=device).view(
                    _ATTENTION_CHUNK_ROWS, self.topk // 128
                ),
                torch.arange(self.topk, dtype=torch.int32, device=device).view(1, 1, self.topk),
                torch.arange(1, _ATTENTION_CHUNK_ROWS + 1, dtype=torch.int32, device=device),
                torch.full((_ATTENTION_CHUNK_ROWS,), self.topk, dtype=torch.int32, device=device),
            )

        return current_workspace_manager().get_persistent_resource(key, allocate)

    def _kv_workspace(self, dtype):
        key = (
            "ascend_hisparse_kv_preprocess",
            self._kv_staging_capacity,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            dtype,
        )

        def allocate():
            device = self.index_group.logical_topk_indices.device
            shape = (self._kv_staging_capacity // 128, 128, 1)
            return (
                torch.empty((*shape, self.kv_lora_rank), dtype=dtype, device=device),
                torch.empty((*shape, self.qk_rope_head_dim), dtype=dtype, device=device),
                torch.arange(self._kv_staging_capacity, dtype=torch.int64, device=device),
            )

        return current_workspace_manager().get_persistent_resource(key, allocate)

    def forward(self, layer_name, hidden_states, kv_cache, attn_metadata, output=None):
        self._workspace(hidden_states.dtype)
        if self.qk_rope_head_dim:
            # Include preprocessing scratch in profiling and allocate it before
            # graph capture, including dummy forwards without attention metadata.
            self._kv_workspace(hidden_states.dtype)
        self.index_group.prepare_for_batch(self.index_group_index, attn_metadata)
        return super().forward(layer_name, hidden_states, kv_cache, attn_metadata, output)

    def exec_kv(self, kv_no_split, cos, sin, kv_cache, slots, attn_metadata):
        from vllm_ascend.distributed.kv_transfer.hisparse.layout import cache_components
        from vllm_ascend.ops.hisparse import scatter_packed_kv

        assert self.kv_a_layernorm is not None
        values = kv_no_split.reshape(-1, self.kv_lora_rank + self.qk_rope_head_dim)
        num_tokens = values.shape[0]
        cache = self.hisparse_cache
        if getattr(cache, "view", None) is not None and len(cache_components(cache.view.attention_cache)) == 2:
            _, mapping, num_rows = cache.write_target(num_tokens, slots.numel())
            latent, rope = cache_components(cache.view.attention_cache)
            block_size = cache.view.block_size
            valid = torch.arange(num_rows, device=mapping.device) < attn_metadata.query_start_loc[-1]
            physical_slots = torch.where(
                (mapping >= 0) & valid,
                mapping // block_size * cache.view.attention_block_stride + mapping % block_size,
                -1,
            ).to(torch.int64)
            torch_npu.npu_kv_rmsnorm_rope_cache(
                values[:num_rows].view(num_rows, self.num_kv_heads, 1, self.kv_lora_rank + self.qk_rope_head_dim),
                self.kv_a_layernorm.weight,
                cos[:num_rows],
                sin[:num_rows],
                physical_slots,
                rope.unsqueeze(2),
                latent.unsqueeze(2),
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA",
            )
            cache.finish_kv_update()
            return None, None
        if self.qk_rope_head_dim:
            if num_tokens > self._kv_staging_capacity:
                raise ValueError("HiSparse KV preprocessing exceeds its profiled token capacity")
            latent_cache, rope_cache, local_slots = self._kv_workspace(values.dtype)
            # Match ordinary SFA's fused RMSNorm/RoPE rounding. Separate RMSNorm
            # can differ by a BF16 ULP near a rounding midpoint. Local slots
            # isolate padding from the real resident/hot destination below.
            torch_npu.npu_kv_rmsnorm_rope_cache(
                values.view(num_tokens, self.num_kv_heads, 1, self.kv_lora_rank + self.qk_rope_head_dim),
                self.kv_a_layernorm.weight,
                cos,
                sin,
                local_slots[:num_tokens],
                rope_cache,
                latent_cache,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA",
            )
            latent = latent_cache.view(-1, self.kv_lora_rank)[:num_tokens]
            rope = rope_cache.view(-1, self.qk_rope_head_dim)[:num_tokens]
        else:
            # The native NoPE path uses the layer's normalization implementation.
            latent = self.kv_a_layernorm(values)
            rope = values[:, self.kv_lora_rank :]
        cache = self.hisparse_cache
        destination, mapping, num_rows = cache.write_target(values.shape[0], slots.numel())
        num_valid_rows = attn_metadata.query_start_loc[-1:]
        scatter_packed_kv(latent[:num_rows], rope[:num_rows], destination, mapping, num_valid_rows)
        # The common connector mirrors resident rows after this write. Draft
        # rows are deferred until the next step, and FULL graphs are mirrored
        # by finish_forward after replay.
        cache.finish_kv_update()
        return rope, latent

    def _attention_physical(self, query, query_rope, packed_cache, physical_topk):
        from vllm_ascend.distributed.kv_transfer.hisparse.layout import cache_components
        from vllm_ascend.ops.hisparse import pack_attention_rows

        latent, rope, block_table, indices, query_ends, seq_lens = self._workspace(query.dtype)
        positions = indices.view(1, -1).to(torch.float32)
        outputs = []
        for start in range(0, query.shape[0], _ATTENTION_CHUNK_ROWS):
            end = min(start + _ATTENTION_CHUNK_ROWS, query.shape[0])
            count = end - start
            # SFA's causal mode derives the valid selection length from KV
            # lengths. Move invalid entries to the tail while preserving the
            # selected order and duplicate multiplicity. Reordering valid KV
            # changes BF16 reductions. Unique FP32 position keys are exact for
            # supported top-k sizes and avoid integer ArgSort on AiCPU.
            physical = physical_topk[start:end]
            keys = positions + (physical < 0).to(torch.float32) * physical.shape[-1]
            physical = physical.gather(1, keys.sort(dim=-1).indices)
            valid_lengths = (physical >= 0).sum(dim=-1, dtype=torch.int32)
            selected_latent, selected_rope = latent[:count], rope[:count]
            components = cache_components(packed_cache)
            if len(components) == 2:
                row_ids = physical.clamp_min(0).to(torch.int64).view(-1)
                for source, destination in zip(components, (selected_latent, selected_rope), strict=True):
                    destination.copy_(source.view(-1, source.shape[-1]).index_select(0, row_ids).view_as(destination))
                    destination.masked_fill_((physical < 0)[:, :, None, None], 0)
            else:
                pack_attention_rows(packed_cache, physical, selected_latent, selected_rope, self.kv_lora_rank)
            selected_indices = torch.where(physical[:, None, :] >= 0, indices, -1)
            result = torch.ops._C_ascend.npu_sparse_flash_attention(
                query=query[start:end],
                key=selected_latent.view(-1, 128, 1, self.kv_lora_rank),
                value=selected_latent.view(-1, 128, 1, self.kv_lora_rank),
                sparse_indices=selected_indices,
                scale_value=self.scale,
                sparse_block_size=1,
                block_table=block_table[:count],
                actual_seq_lengths_query=query_ends[:count],
                actual_seq_lengths_kv=valid_lengths,
                query_rope=query_rope[start:end] if self.qk_rope_head_dim else None,
                key_rope=selected_rope.view(-1, 128, 1, self.qk_rope_head_dim) if self.qk_rope_head_dim else None,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
            )
            if isinstance(result, tuple):
                result = result[0]
            result = result.masked_fill(~(physical >= 0).any(dim=-1)[:, None, None], 0)
            outputs.append(result)
        return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table=None,
    ):
        metadata = attn_metadata
        logical = topk_indices.squeeze(1) if topk_indices.ndim == 3 else topk_indices
        assert metadata.req_id_per_token is not None
        assert metadata.query_start_loc is not None
        rows = torch.arange(logical.shape[0], dtype=torch.int32, device=logical.device)
        logical = torch.where(rows[:, None] < metadata.query_start_loc[-1], logical, -1)
        self.index_group.set_logical_topk_ready(self.index_group_index)
        cache = self.hisparse_cache
        from vllm_ascend.distributed.kv_transfer.hisparse.layout import cache_components, combine_components

        if (
            metadata.num_prefills
            and (
                self._packed_resident_supported
                or (getattr(cache, "view", None) is not None and len(cache_components(cache.view.attention_cache)) == 2)
            )
            and cache.all_context_pages_resident
            and cache.view is not None
            and cache.view.block_size == metadata.block_size
            and cache.view.attention_cache.is_contiguous()
        ):
            from vllm_ascend.ops.hisparse import prepare_prefill_attention

            # The coordinator's resident flag covers every request in the
            # current batch. Read the original packed cache, including mixed
            # decode rows, without gathering or allocating a split KV mirror.
            assert cache.block_table is not None
            states = cache.runtime.request_state_indices
            assert states is not None
            width = (metadata.max_seq_len + metadata.block_size - 1) // metadata.block_size
            table = cache.block_table[:, :width].index_select(0, states[: metadata.num_reqs])
            table = table * (cache.view.attention_block_stride // metadata.block_size)
            selected, query_ends, valid_lengths, expanded_table = prepare_prefill_attention(
                logical,
                metadata.req_id_per_token,
                metadata.query_start_loc,
                metadata.seq_lens,
                table,
                0,
                0,
                selection_lengths=get_current_hardware_profile().supports(HardwareCapability.SFA_SELECTION_LENGTHS),
            )
            components = cache_components(cache.view.attention_cache)
            key = components[0].unsqueeze(2)
            key_rope = components[-1].unsqueeze(2)
            result = torch.ops._C_ascend.npu_sparse_flash_attention(
                query=ql_nope,
                key=key,
                value=key,
                sparse_indices=selected,
                scale_value=self.scale,
                sparse_block_size=1,
                block_table=expanded_table,
                actual_seq_lengths_query=query_ends,
                actual_seq_lengths_kv=valid_lengths,
                query_rope=q_pe,
                key_rope=key_rope,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
            )
            if isinstance(result, tuple):
                result = result[0]
            return result.masked_fill(~(selected >= 0).any(dim=-1)[:, :, None], 0)
        outputs = []
        num_decode = metadata.num_decode_tokens
        if num_decode:
            cache = self.hisparse_cache
            assert cache.source_block_table is not None
            # Keep mixed prefill/decode batches on the NPU residency resolver.
            # The generic index-group resident shortcut uses the CUDA sparse
            # index converter when a batch also contains prefill tokens.
            physical = cache.swap_in(
                metadata.req_id_per_token[:num_decode],
                block_table=cache.source_block_table,
                logical_topk_indices=logical[:num_decode],
                block_size=metadata.block_size,
                num_valid_rows=metadata.query_start_loc[-1:],
            )
            outputs.append(
                self._attention_physical(
                    ql_nope[:num_decode],
                    q_pe[:num_decode] if q_pe is not None else None,
                    self.index_group.physical_kv_cache(self.index_group_index),
                    physical,
                )
            )
        if metadata.num_prefills:
            host_cache = self.hisparse_cache.runtime.host_cache
            host_cache = combine_components(
                tuple(
                    component.view(-1, metadata.block_size, component.shape[-1])
                    for component in cache_components(host_cache)
                )
            )
            staged, staged_table, _ = self.index_group.stage_prefill_rows(
                self.index_group_index,
                host_cache,
                metadata,
            )
            from vllm_ascend.ops.hisparse import prepare_prefill_attention

            # Split each staged row once. Per-query page tables and selection
            # lengths avoid duplicating top-k KV payload and repeated SFA calls.
            # Limit table copies to the CPU context upper bound, independently
            # of the configured maximum model length.
            components = cache_components(staged)
            if len(components) == 2:
                latent, rope = (component.unsqueeze(2) for component in components)
            else:
                latent = staged[..., : self.kv_lora_rank].contiguous().unsqueeze(2)
                rope = staged[..., self.kv_lora_rank :].contiguous().unsqueeze(2)
            query_end = logical.shape[0]
            table_width = (metadata.max_seq_len + metadata.block_size - 1) // metadata.block_size
            selected, query_ends, valid_lengths, expanded_table = prepare_prefill_attention(
                logical[num_decode:query_end],
                metadata.req_id_per_token[num_decode:query_end],
                metadata.query_start_loc,
                metadata.seq_lens,
                staged_table[:, :table_width],
                num_decode,
                metadata.num_decodes,
                selection_lengths=get_current_hardware_profile().supports(HardwareCapability.SFA_SELECTION_LENGTHS),
            )
            result = torch.ops._C_ascend.npu_sparse_flash_attention(
                query=ql_nope[num_decode:query_end],
                key=latent,
                value=latent,
                sparse_indices=selected,
                scale_value=self.scale,
                sparse_block_size=1,
                block_table=expanded_table,
                actual_seq_lengths_query=query_ends,
                actual_seq_lengths_kv=valid_lengths,
                query_rope=q_pe[num_decode:query_end] if self.qk_rope_head_dim else None,
                key_rope=rope if self.qk_rope_head_dim else None,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
            )
            if isinstance(result, tuple):
                result = result[0]
            outputs.append(result.masked_fill(~(selected >= 0).any(dim=-1)[:, :, None], 0))
        if not outputs:
            return torch.zeros_like(ql_nope)
        result = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
        if result.shape[0] < ql_nope.shape[0]:
            result = torch.cat((result, torch.zeros_like(ql_nope[result.shape[0] :])), dim=0)
        return result
