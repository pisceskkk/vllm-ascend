import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional, Tuple, Type, TypeVar

import torch
import torch.nn.functional as F
import torch_npu
import vllm.envs as envs_vllm
from vllm.v1.attention.backend import AttentionBackend
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.attention.backend import (AttentionCGSupport,
                                        AttentionMetadataBuilder)
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

from vllm_ascend.attention.abstract import DSAAttentionImpl
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import rotate_activation 
from vllm_ascend.attention.utils import (AscendCommonAttentionMetadata,
                                         split_decodes_and_prefills)
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import (AscendDeviceType,
                               enable_dsa_cp,
                               get_ascend_device_type, get_dsv4_compress_ratio, extract_dsv4_layer_index,
                               olora_tp_enable)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_ascend.ops.triton.rms_norm import triton_q_rms

if HAS_TRITON:
    from vllm_ascend.ops.triton.rms_norm import triton_q_rms  # noqa: F811
else:
    triton_q_rms = None  # type: ignore


@dataclass
class AscendDSAReqMetadata:
    """Unified per-request metadata — combines fields formerly split into
    prefill and decode sub-structures.

    All methods (builder, forward) operate on this single metadata,
    without distinguishing prefill vs decode request types.
    """
    input_positions: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    compress_sin: torch.Tensor = None
    compress_cos: torch.Tensor = None
    start_pos: torch.Tensor = None
    sas_metadata: torch.Tensor = None
    qli_metadata: torch.Tensor = None
    cu_c4_cmp_seqlen_list: torch.Tensor = None
    cu_c128_cmp_seqlen_list: torch.Tensor = None
    attn_mask: Optional[torch.Tensor] = None


@dataclass
class DSACPContext:
    """Context parallelism split info for DSA CP.

    Each TP rank processes a different chunk of the input sequence.
    This dataclass holds the per-rank metadata needed for local
    Q computation and full KV processing.
    """
    num_tokens: int
    num_tokens_pad: int
    local_start: int
    local_end: int
    local_end_with_pad: int
    actual_seq_lengths_query: torch.Tensor
    actual_seq_lengths_key: torch.Tensor


@dataclass
class AscendDSAMetadata:
    """Metadata for MLACommon.
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    query_lens: Optional[list[int]] = None
    # The dimension of the attention heads
    head_dim: Optional[int] = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    req_metadata: Optional[AscendDSAReqMetadata] = None
    reshape_cache_event: torch.npu.Event = None

    # metadata for dsv4 indexer

    hadamard: Optional[torch.Tensor] = None

    start_pos: Optional[torch.Tensor] = None

    # Context parallelism sequence sharding context
    dsa_cp_context: Optional[DSACPContext] = None

    def __post_init__(self):
        pass


M = TypeVar("M", bound=AscendDSAMetadata)


class AscendDSACPMetadataBuilder(AttentionMetadataBuilder[AscendDSAMetadata]):
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_BATCH
    hadamard = None
    start_pos_prefill: Optional[torch.Tensor] = None
    block_size: Optional[int] = 128
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec: MLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendDSAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.metadata_cls = (metadata_cls if metadata_cls is not None else
                             AscendDSAMetadata)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        scheduler_config = vllm_config.scheduler_config
        # self.block_size = vllm_config.cache_config.block_size
        self.max_blocks = (vllm_config.model_config.max_model_len +
                           self.block_size - 1) // self.block_size

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"

        self.reorder_batch_threshold = self.decode_threshold
        self.rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
        self.cos_cache = None
        self.sin_cache = None

        self.cu_seq_lens_cpu: torch.Tensor = None
        self.num_decodes = 0
        self.num_prefills = 0
        self.num_decode_tokens = 0
        self.num_prefill_tokens = 0
        self.context_lens_cpu: torch.Tensor = None
        self.num_actual_tokens: Optional[int] = None
        self.block_table: torch.Tensor = None
        self.slot_mapping: torch.Tensor = None
        self.graph_pad_size = 0
        self.query_lens: torch.Tensor = None
        self.seq_lens: torch.Tensor = None
        self.attn_mask_builder = AttentionMaskBuilder(self.device)
        
        self.compressor_ratio = getattr(kv_cache_spec, 'compress_ratio', 0)
        hf_config = self.model_config.hf_config

        self.enable_dsa_cp = enable_dsa_cp()
        if self.enable_dsa_cp:
            max_num_reqs = vllm_config.scheduler_config.max_num_seqs
            self.actual_seq_lengths_query = torch.zeros(max_num_reqs + 1,
                                                        dtype=torch.int32,
                                                        device=self.device)
            self.actual_seq_lengths_key = torch.empty_like(
                self.actual_seq_lengths_query)
        else:
            self.actual_seq_lengths_query = torch.tensor([])
            self.actual_seq_lengths_key = torch.tensor([])

        if AscendDSACPMetadataBuilder.hadamard is None:
            if hf_config.model_type == 'deepseek_v4':
                indexer_head_dim = hf_config.index_head_dim
                try:
                    from scipy.linalg import hadamard
                except ImportError as e:
                    raise ImportError("Please install scipy") from e
                log_dim = math.ceil(math.log2(indexer_head_dim))
                dim_padded = 2**log_dim
                AscendDSACPMetadataBuilder.hadamard = torch.tensor(
                    hadamard(dim_padded, dtype=float),
                    dtype=torch.float,
                    device=self.device).to(torch.bfloat16)
        self.start_pos_prefill = torch.zeros(scheduler_config.max_num_seqs,
                                             dtype=torch.int32,
                                             device=self.device)
        self.cu_seqlens_ori_kv = torch.tensor([], device=self.device)
        self.cu_seqlens_cmp_kv = torch.tensor([], device=self.device)
        self.seqused_q = torch.tensor([], device=self.device)
        # Note(qcs): we use two dimension slot_mapping for kvcache with shape [block_nums, block_size, head_num, head_dim]
        self.slot_mapping = torch.zeros((vllm_config.scheduler_config.max_num_batched_tokens, 2), dtype=torch.int32, device=self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendDSACPMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.UNIFORM_BATCH

    def reorder_batch(self, input_batch: "NPUInputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        # We now want to reorder the batch so that the "decode" requests are at
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound, TODO(lucas): figure out a
        # better naming here)
        decodes = []
        prefills = []

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if num_tokens <= self.decode_threshold:
                decodes.append(i)
            else:
                prefills.append(i)

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        first_prefill = 0
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            if decodes[num_decodes - i] >= num_decodes:
                input_batch.swap_states(prefills[first_prefill],
                                        decodes[num_decodes - i])
                first_prefill += 1
                modified_batch = True
            else:
                break

        # Save for next `build` call
        # TODO(lucas): this is a bit of a hack, we should probably have a
        # better way of doing this
        return modified_batch

    def set_num_actual_tokens(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        num_reqs_actual = kwargs.get("num_reqs_actual", None)
        self.block_size = kwargs.get("block_size", 128)

        self.common_ratio_to_sas_metadata = kwargs.get("common_ratio_to_sas_metadata", None)

        if self.common_ratio_to_sas_metadata.get("num_decodes", None) is None:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = \
                split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)
            self.common_ratio_to_sas_metadata["num_decodes"] = self.num_decodes
            self.common_ratio_to_sas_metadata["num_prefills"] = self.num_prefills
            self.common_ratio_to_sas_metadata["num_decode_tokens"] = self.num_decode_tokens
            self.common_ratio_to_sas_metadata["num_prefill_tokens"] = self.num_prefill_tokens
            self.set_num_actual_tokens(common_attn_metadata)
            assert self.num_decodes + self.num_prefills == num_reqs
            assert self.num_decode_tokens + self.num_prefill_tokens == common_attn_metadata.num_actual_tokens
            num_input_tokens = common_attn_metadata.num_input_tokens
            input_positions = common_attn_metadata.positions[:
                                                         num_input_tokens].long(
                                                         )
            self.common_ratio_to_sas_metadata["input_positions"] = input_positions
            cos, sin = get_cos_and_sin_dsa(input_positions,
                                           use_cache=(self.num_prefills == 0))
            self.common_ratio_to_sas_metadata["cos"] = cos
            self.common_ratio_to_sas_metadata["sin"] = sin
            self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
            self.common_ratio_to_sas_metadata["seq_lens"] = self.seq_lens

            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            query_seq_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            self.query_lens = query_seq_lens_cpu[:num_reqs]
            self.common_ratio_to_sas_metadata["query_lens"] = self.query_lens
        else:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = \
                self.common_ratio_to_sas_metadata["num_decodes"], \
                self.common_ratio_to_sas_metadata["num_prefills"], \
                self.common_ratio_to_sas_metadata["num_decode_tokens"], \
                self.common_ratio_to_sas_metadata["num_prefill_tokens"]
            self.set_num_actual_tokens(common_attn_metadata)
            num_input_tokens = common_attn_metadata.num_input_tokens
            input_positions = self.common_ratio_to_sas_metadata["input_positions"]
            cos, sin = self.common_ratio_to_sas_metadata["cos"], self.common_ratio_to_sas_metadata["sin"]
            self.seq_lens = self.common_ratio_to_sas_metadata["seq_lens"]
            self.query_lens = self.common_ratio_to_sas_metadata["query_lens"]

        # NOTE: Currently, MTP-fullgraph is incompatibility pcp
        slot_mapping = common_attn_metadata.slot_mapping[:
                                                              num_input_tokens]
        self.slot_mapping[:num_input_tokens] = torch.stack([slot_mapping // self.block_size, slot_mapping % self.block_size], axis=-1)

        self.graph_pad_size = common_attn_metadata.graph_pad_size
        self.block_table = common_attn_metadata.block_table_tensor[:num_reqs]

        req_metadata = self.build_req_metadata(
            common_prefix_len, common_attn_metadata, num_reqs_actual)

        # --- Build DSACPContext for sequence sharding ---
        if self.enable_dsa_cp:
            global_tp_size = get_tp_group().world_size
            num_tokens = num_input_tokens
            num_tokens_pad = cdiv(num_tokens, global_tp_size) * global_tp_size
            num_tokens_per_device = num_tokens_pad // global_tp_size
            local_start = get_tp_group().rank_in_group * num_tokens_per_device
            local_end_with_pad = local_start + num_tokens_per_device
            local_end = min(local_end_with_pad, self.num_actual_tokens)

            # actual_seq_lengths_query/key: per-sequence token counting
            # actual_seq_lengths_query follows TND convention: [0, cum_len_1, ..., total]
            seq_lens = self.seq_lens[:num_reqs]
            num_segs = query_start_loc.shape[0] - 1
            last_token = 0
            cum = 0
            for i in range(num_segs):
                global_start = last_token
                global_end = query_start_loc[i + 1].item()
                last_token = global_end

                req_local_start = max(global_start, local_start)
                req_local_end = min(global_end, local_end_with_pad)
                num_local_tokens = max(0, req_local_end - req_local_start)

                if num_local_tokens > 0:
                    cum += num_local_tokens
                    self.actual_seq_lengths_query[i + 1] = cum
                    # KV is full on all ranks (from FC1 all-gather), so use full seq_lens
                    self.actual_seq_lengths_key[i] = seq_lens[i].item()
                else:
                    self.actual_seq_lengths_query[i + 1] = cum
                    # KV is full on all ranks even if rank has no local query tokens
                    self.actual_seq_lengths_key[i] = seq_lens[i].item()

            dsa_cp_context = DSACPContext(
                num_tokens=num_tokens,
                num_tokens_pad=num_tokens_pad,
                local_start=local_start,
                local_end=local_end,
                local_end_with_pad=local_end_with_pad,
                actual_seq_lengths_query=self.actual_seq_lengths_query[:num_reqs + 1],
                actual_seq_lengths_key=self.actual_seq_lengths_key[:num_reqs],
            )
        else:
            dsa_cp_context = None

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=self.num_actual_tokens,
            query_lens=self.query_lens,
            slot_mapping=None,
            head_dim=self.model_config.get_head_size(),
            num_decodes=self.num_decodes,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            attn_mask=None,
            attn_state=common_attn_metadata.attn_state,
            req_metadata=req_metadata,
            query_start_loc=query_start_loc,
            block_tables=None,
            seq_lens=self.seq_lens,
            cos=cos,
            sin=sin,
            hadamard=AscendDSACPMetadataBuilder.hadamard,
            dsa_cp_context=dsa_cp_context,
        )

    def build_req_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        num_reqs_actual: Optional[int],
    ) -> AscendDSAReqMetadata:
        """Build a single unified metadata for all requests (prefill + decode)."""
        num_reqs = self.num_decodes + self.num_prefills
        query_start_loc = common_attn_metadata.query_start_loc[:num_reqs + 1]
        cache = self.common_ratio_to_sas_metadata

        input_positions = common_attn_metadata.positions[:self.num_actual_tokens].long()

        # _seq_lens_cpu for max_seq_lens
        if common_attn_metadata._seq_lens_cpu is not None:
            _seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        elif common_attn_metadata.seq_lens_cpu is not None:
            _seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        else:
            _seq_lens_cpu = common_attn_metadata.seq_lens.cpu()

        max_query_len = self.query_lens.max().item()
        max_seq_lens = _seq_lens_cpu[:num_reqs].max().item()

        # cos/sin for all tokens
        cos, sin = get_cos_and_sin_dsa(input_positions,
                                       use_cache=(self.num_prefills == 0))

        # start_pos: context length before current query
        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
        start_pos = self.seq_lens[:num_reqs] - seq_lens_q

        assert self.start_pos_prefill is not None
        self.start_pos_prefill.fill_(0)
        self.start_pos_prefill[:num_reqs] = start_pos

        if num_reqs_actual is not None and num_reqs_actual < num_reqs:
            self.start_pos_prefill[num_reqs_actual:].fill_(0)
            self.block_table[num_reqs_actual:num_reqs, ...].fill_(0)

        # --- Compressed positions ---
        compress_cos, compress_sin = None, None
        cu_c4_cmp_seqlen_list = None
        cu_c128_cmp_seqlen_list = None

        if self.compressor_ratio > 1:
            layer_name = f"c{self.compressor_ratio}"
            compressed_input_positions = self._get_padded_compressed_position(
                input_positions, self.compressor_ratio)
            compress_cos, compress_sin = get_cos_and_sin_dsa(
                {layer_name: compressed_input_positions},
                use_cache=(self.num_prefills == 0))

            if self.compressor_ratio == 4:
                cu_c4_cmp_seqlen_list = self._get_cmp_seq_lens(
                    self.seq_lens[:num_reqs], self.compressor_ratio)
            else:
                cu_c128_cmp_seqlen_list = self._get_cmp_seq_lens(
                    self.seq_lens[:num_reqs], self.compressor_ratio)

        compressed_tokens_start, compressed_tokens_end = \
            self._get_compressed_token_start_and_end(
                input_positions, self.compressor_ratio)
        slot_mapping = self.slot_mapping[
            compressed_tokens_start:compressed_tokens_end + compressed_tokens_start]

        # --- SAS metadata (all requests combined) ---
        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size
        index_topk = self.model_config.hf_config.index_topk

        sas_metadata = self._build_sas_metadata(
            n_local_heads=n_local_heads,
            query_start_loc=query_start_loc,
            seq_lens=self.seq_lens[:num_reqs],
            max_query_len=max_query_len,
            max_seq_lens=max_seq_lens,
            index_topk=index_topk,
            num_reqs=num_reqs,
            cu_c4_cmp_seqlen_list=cu_c4_cmp_seqlen_list,
            cu_c128_cmp_seqlen_list=cu_c128_cmp_seqlen_list)

        # --- QLI metadata (all requests combined) ---
        qli_metadata = self._build_qli_metadata(
            query_start_loc=query_start_loc,
            seq_lens=self.seq_lens[:num_reqs],
            num_reqs=num_reqs)

        return AscendDSAReqMetadata(
            input_positions=input_positions,
            block_table=self.block_table[:num_reqs, ...],
            slot_mapping=slot_mapping,
            seq_lens=self.seq_lens[:num_reqs],
            query_start_loc=query_start_loc,
            sin=sin,
            cos=cos,
            compress_sin=compress_sin,
            compress_cos=compress_cos,
            start_pos=self.start_pos_prefill[:num_reqs],
            sas_metadata=sas_metadata,
            qli_metadata=qli_metadata,
            cu_c4_cmp_seqlen_list=cu_c4_cmp_seqlen_list,
            cu_c128_cmp_seqlen_list=cu_c128_cmp_seqlen_list)

    # --- helper: padded compressed positions ---
    def _get_padded_compressed_position(self, input_positions,
                                        compress_ratio):
        if compress_ratio <= 1:
            return input_positions
        mask = ((input_positions + 1) % compress_ratio) == 0
        pos = input_positions[mask]
        pos = (pos + 1) - compress_ratio
        num_tokens = self.num_prefill_tokens + self.num_decode_tokens
        num_reqs = self.num_prefills + self.num_decodes
        target_shape = (min(num_tokens,
                            num_tokens // compress_ratio + num_reqs),)
        pad_right = target_shape[0] - pos.shape[0]
        return F.pad(pos, (0, pad_right), value=0.0)

    # --- helper: compressed seq lens ---
    def _get_cmp_seq_lens(self, seq_lens, compress_ratio):
        _cmp = seq_lens // compress_ratio if compress_ratio >= 1 else seq_lens
        return torch.concat(
            (torch.tensor([0], device=_cmp.device),
             torch.cumsum(_cmp, -1)), dim=-1)

    # --- helper: compressed token start / end ---
    def _get_compressed_token_start_and_end(self, input_positions,
                                            compress_ratio):
        if compress_ratio <= 1:
            return 0, 0
        if compress_ratio == 0:
            compress_ratio = 1
        mask = ((input_positions + 1) % compress_ratio) == 0
        compressed_num = mask.sum()
        num_tokens = self.num_prefill_tokens + self.num_decode_tokens
        end = min(num_tokens, num_tokens // compress_ratio +
                  self.num_prefills + self.num_decodes)
        return compressed_num, end

    # --- helper: build SAS metadata ---
    def _build_sas_metadata(self, n_local_heads, query_start_loc, seq_lens,
                            max_query_len, max_seq_lens, index_topk,
                            num_reqs, cu_c4_cmp_seqlen_list,
                            cu_c128_cmp_seqlen_list):
        cache_key = f"sas_c{self.compressor_ratio}"
        if self.common_ratio_to_sas_metadata.get(cache_key) is not None:
            return self.common_ratio_to_sas_metadata[cache_key]

        cu_seqlens_ori_kv = self.cu_seqlens_ori_kv  # empty placeholder
        kw = dict(
            num_heads_q=n_local_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=query_start_loc,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            seqused_q=self.seqused_q,
            seqused_kv=seq_lens,
            max_seqlen_q=max_query_len,
            max_seqlen_kv=max_seq_lens,
            batch_size=num_reqs,
            ori_mask_mode=4,
            ori_win_left=self.model_config.hf_config.sliding_window - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            device=str(self.seqused_q.device))

        if self.compressor_ratio > 1:
            kw['has_cmp_kv'] = True
            if self.compressor_ratio == 4:
                kw['cmp_ratio'] = 4
                kw['cmp_mask_mode'] = 3
                kw['cmp_topk'] = index_topk
                kw['cu_seqlens_cmp_kv'] = cu_c4_cmp_seqlen_list
            else:
                kw['cmp_ratio'] = 128
                kw['cmp_mask_mode'] = 3
                kw['cu_seqlens_cmp_kv'] = cu_c128_cmp_seqlen_list
        else:
            kw['cmp_ratio'] = 1
            kw['has_cmp_kv'] = False

        metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(**kw)
        self.common_ratio_to_sas_metadata[cache_key] = metadata
        return metadata

    # --- helper: build QLI metadata ---
    def _build_qli_metadata(self, query_start_loc, seq_lens, num_reqs):
        cache_key = "qli"
        if self.common_ratio_to_sas_metadata.get(cache_key) is not None:
            return self.common_ratio_to_sas_metadata[cache_key]

        if self.compressor_ratio != 4:
            return None

        metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_metadata(
            actual_seq_lengths_query=query_start_loc[1:].clone(),
            actual_seq_lengths_key=seq_lens.clone(),
            num_heads_q=self.model_config.hf_config.index_n_heads,
            num_heads_k=1,
            head_dim=self.model_config.hf_config.index_head_dim,
            query_quant_mode=0,
            key_quant_mode=0,
            batch_size=num_reqs,
            max_seqlen_q=query_start_loc[1:].max().item(),
            max_seqlen_k=seq_lens.max().item(),
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.model_config.hf_config.index_topk,
            sparse_mode=3,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            device=str(self.seqused_q.device))
        self.common_ratio_to_sas_metadata[cache_key] = metadata
        return metadata

    def build_for_graph_capture(
            self,
            common_attn_metadata: AscendCommonAttentionMetadata,
            attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
            **kwargs):
        if attn_state in {
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding
        }:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and SpecDecoding state"
            )

        assert attn_metadata is not None
        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendDSACPImpl(DSAAttentionImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
        **kwargs,
    ):
        self.num_heads = n_heads
        self.n_local_heads = n_local_heads
        self.scale = scale
        self.o_lora_rank = o_lora_rank
        self.nope_head_dim = nope_head_dim
        self.rope_head_dim = rope_head_dim
        self.head_dim = head_dim
        self.n_group = n_groups
        self.n_local_groups = n_local_groups
        self.window_size = window_size
        self.q_lora_rank = q_lora_rank
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.head_dim**-0.5
        self.enable_dsa_cp = enable_dsa_cp()

        # MLA Args
        self.wq_a = kwargs['wq_a']
        self.wq_b = kwargs['wq_b']
        self.wkv = kwargs['wkv']
        self.q_norm = kwargs['q_norm']
        self.kv_norm = kwargs['kv_norm']

        self.indexer = kwargs.get('indexer', None)
        self.compressor = kwargs.get('compressor', None)

        self.wo_a = kwargs['wo_a']
        self.wo_b = kwargs['wo_b']

        self.eps = kwargs['eps']

        self.attn_sink = kwargs['attn_sink']

        self.vllm_config = get_current_vllm_config()

        # indexer param
        if self.indexer is not None:
            self.indexer_heads: int = self.indexer.n_heads
            self.inderxer_dim: int = self.indexer.head_dim
            self.inderxer_wq_b = self.indexer.wq_b
            self.weights_proj = self.indexer.weights_proj
            self.indexer_softmax_scale = self.inderxer_dim**-0.5

            self.indexer_compress = self.indexer.compressor

            # indexer_compressor
            self.indexcom_ape = self.indexer.compressor.ape
            self.indexcom_wkv = self.indexer.compressor.wkv
            self.indexcom_wgate = self.indexer.compressor.wgate
            self.indexcom_norm = self.indexer.compressor.norm

            self.indexcom_head_dim = self.indexer.compressor.head_dim
            self.indexcom_rotate = self.indexer.compressor.rotate
            self.index_topk = self.indexer.index_topk

        # compress param
        if self.compressor is not None:
            self.compressor_head_dim = self.compressor.head_dim
            self.compressor_overlap = self.compressor.overlap
            self.compressor_rotate = self.compressor.rotate

            self.compressor_ape = self.compressor.ape
            self.compressor_wkv = self.compressor.wkv
            self.compressor_wgate = self.compressor.wgate
            self.compressor_norm = self.compressor.norm
            self.compressor_norm_eps = self.compressor.norm_eps

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if self.enable_dsa_cp:
            tp_group = get_tp_group().device_group
            tp_size = get_tensor_model_parallel_world_size()

            # All-gather wq_b weight across TP ranks to restore full heads
            w = self.wq_b.weight.data
            full_w = torch.empty(w.shape[0] * tp_size, *w.shape[1:],
                                 dtype=w.dtype, device=w.device)
            torch.distributed.all_gather_into_tensor(full_w, w.contiguous(), group=tp_group)
            self.wq_b.weight = torch.nn.Parameter(full_w)

            # All-gather quantized weight_scale if present
            if (hasattr(self.wq_b, 'weight_scale')
                    and self.wq_b.weight_scale is not None):
                ws = self.wq_b.weight_scale.data
                full_ws = torch.empty(ws.shape[0] * tp_size, *ws.shape[1:],
                                      dtype=ws.dtype, device=ws.device)
                torch.distributed.all_gather_into_tensor(full_ws, ws.contiguous(), group=tp_group)
                self.wq_b.weight_scale = full_ws

            # All-gather bias if present
            if hasattr(self.wq_b, 'bias') and self.wq_b.bias is not None:
                b = self.wq_b.bias.data
                full_b = torch.empty(b.shape[0] * tp_size, *b.shape[1:],
                                     dtype=b.dtype, device=b.device)
                torch.distributed.all_gather_into_tensor(full_b, b.contiguous(), group=tp_group)
                self.wq_b.bias = torch.nn.Parameter(full_b)

    # TODO: cast to bfloat16 to speed up
    def rope_single(self, x, cos, sin, inverse=False):
        if inverse:
            sin = -sin
        tnd_layout = 1
        if len(x.shape) == 3:
            num_tokens, num_heads, rotary_dim = x.shape
        else:
            tnd_layout = 0
            _, num_tokens, num_heads, rotary_dim = x.shape
        x_rot = torch_npu.npu_rotary_mul(x.reshape(num_tokens, num_heads, 1,
                                                   rotary_dim),
                                         cos,
                                         sin,
                                         rotary_mode="interleave")
        if tnd_layout:
            x = x_rot.reshape(num_tokens, -1, rotary_dim)
        else:
            x = x_rot.reshape(1, num_tokens, -1, rotary_dim)
        return x

    def forward(  # type: ignore[override]
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: Tuple[torch.Tensor],
        attn_metadata: list[M],
        need_gather_q_kv: bool = False,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)
        if not isinstance(attn_metadata, list):
            attn_metadata = [attn_metadata]
        output_padded = output
        attn_heads = self.num_heads if self.enable_dsa_cp else self.n_local_heads
        o_proj_input = self._forward(layer_name, hidden_states,
                                    kv_cache, attn_metadata)

        cos = attn_metadata[0].cos[layer_name]
        sin = attn_metadata[0].sin[layer_name]
        # In CP mode, slice cos/sin to local token range for inverse RoPE
        if self.enable_dsa_cp and attn_metadata[0].dsa_cp_context is not None:
            cp_ctx = attn_metadata[0].dsa_cp_context
            cos = cos[cp_ctx.local_start:cp_ctx.local_end_with_pad]
            sin = sin[cp_ctx.local_start:cp_ctx.local_end_with_pad]
        num_tokens = o_proj_input.shape[0]

        # In CP mode, _forward uses full heads; extract local heads for O proj.
        if self.enable_dsa_cp:
            tp_rank = get_tp_group().rank_in_group
            start_h = tp_rank * self.n_local_heads
            end_h = start_h + self.n_local_heads
            o_proj_input = o_proj_input[:, start_h:end_h, :]

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            o_proj_input.unsqueeze(1),
            cos,
            -sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # o
        o_proj_input = o_proj_input.view(num_tokens, self.n_local_groups, -1)
        if olora_tp_enable():
            o_proj_input = self.wo_a(o_proj_input)
        else:
            # wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            # o = torch.einsum("tgd,grd->tgr", o, wo_a)
            o_proj_input = torch_npu.npu_transpose_batchmatmul(
                o_proj_input,
                self.wo_a.weight,
                bias=None,
                scale=None,
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
                batch_split_factor=1)
            o_proj_input = o_proj_input.reshape(num_tokens, -1)
        output[...] = self.wo_b(o_proj_input)

        return output_padded

    def _forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: Tuple,
        attn_metadata: AscendDSAMetadata,
    ):
        """Unified forward for all requests — no prefill/decode distinction,
        no multi-stream computation."""
        compress_common_attn_metadata = None

        if self.compress_ratio == 4:
            (compress_kv_cache, swa_kv_cache, state_cache, _, _, _) = kv_cache
            (compressor_attn_metadata, compressor_kv_state_metadata, _, _, swa_metadata) = attn_metadata
            compress_common_attn_metadata = compressor_attn_metadata
        elif self.compress_ratio == 128:
            (compress_kv_cache, swa_kv_cache, state_cache, _, _, _) = kv_cache
            (compressor_attn_metadata, compressor_kv_state_metadata, swa_metadata) = attn_metadata
            compress_common_attn_metadata = compressor_attn_metadata
        else:
            (_, swa_kv_cache, _, _, _, _) = kv_cache
            (swa_metadata,) = attn_metadata
            compress_common_attn_metadata = swa_metadata

        assert compress_common_attn_metadata.req_metadata is not None
        cos = compress_common_attn_metadata.req_metadata.cos[layer_name]
        sin = compress_common_attn_metadata.req_metadata.sin[layer_name]
        actual_seq_lengths_query_full = compress_common_attn_metadata.req_metadata.query_start_loc
        actual_seq_lengths_key = compress_common_attn_metadata.req_metadata.seq_lens

        # In CP mode: FC1 all-gather gives full hidden_states.
        # Q uses local tokens (slice hidden_states), KV uses full hidden_states.
        if self.enable_dsa_cp:
            cp_ctx = compress_common_attn_metadata.dsa_cp_context
            hidden_states_q = hidden_states[cp_ctx.local_start:cp_ctx.local_end_with_pad]
            cos_q = cos[cp_ctx.local_start:cp_ctx.local_end_with_pad]
            sin_q = sin[cp_ctx.local_start:cp_ctx.local_end_with_pad]
            actual_seq_lengths_query = cp_ctx.actual_seq_lengths_query
        else:
            hidden_states_q = hidden_states
            cos_q = cos
            sin_q = sin
            actual_seq_lengths_query = actual_seq_lengths_query_full

        # --- q (local tokens in CP mode) ---
        q_heads = self.num_heads if self.enable_dsa_cp else self.n_local_heads
        if (not isinstance(self.wq_b.quant_method, AscendUnquantizedLinearMethod)) and \
                isinstance(self.wq_b.quant_method.quant_method, AscendW8A8DynamicLinearMethod):
            q_a = self.wq_a(hidden_states_q)
            qr, qr_pertoken_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                q_a, self.q_norm.weight, epsilon=self.eps)
            q = torch_npu.npu_quant_matmul(
                qr,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=qr_pertoken_scale,
                bias=self.wq_b.bias,
                output_dtype=hidden_states.dtype,
            ).unflatten(-1, (q_heads, self.head_dim))
        else:
            qr = self.q_norm(self.wq_a(hidden_states_q))
            q = self.wq_b(qr).unflatten(-1, (q_heads, self.head_dim))
            qr_pertoken_scale = None

        q = triton_q_rms(q, self.eps)

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos_q,
            sin_q,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # --- kv (full hidden_states in both CP and non-CP) ---
        kv = self.wkv(hidden_states)
        kv = self.kv_norm(kv)
        assert self.rope_head_dim is not None
        kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # swa exec kv
        torch.ops._C_ascend.npu_scatter_nd_update_v2(
            swa_kv_cache,
            swa_metadata.req_metadata.slot_mapping, kv)

        # --- compressor + indexer ---
        if self.compress_ratio > 1:
            compress_cos = compress_common_attn_metadata.req_metadata.compress_cos[layer_name]
            compress_sin = compress_common_attn_metadata.req_metadata.compress_sin[layer_name]
            compress_topk_idxs = None
            if self.compress_ratio == 4:
                compress_topk_idxs = self.indexer_select_qli(
                    x=hidden_states,
                    qr=qr,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    cos=cos_q,
                    sin=sin_q,
                    compressed_cos=compress_cos,
                    compressed_sin=compress_sin,
                    # Pass FULL query_start_loc for KV compressor inside indexer
                    actual_seq_lengths_query=actual_seq_lengths_query_full,
                    actual_seq_lengths_key=actual_seq_lengths_key,
                    qr_pertoken_scale=qr_pertoken_scale)

            coff = 2 if self.compressor_overlap else 1

            compressed_kv = torch.ops._C_ascend.compressor(
                hidden_states,
                self.compressor_wkv.weight,
                self.compressor_wgate.weight,
                state_cache.squeeze(-2),
                self.compressor_ape,
                self.compressor_norm.weight,
                compress_sin.view(-1, compress_sin.shape[-1]),
                compress_cos.view(-1, compress_cos.shape[-1]),
                state_block_table=compressor_kv_state_metadata.req_metadata.block_table,
                cu_seqlens=actual_seq_lengths_query_full,
                seqused=None,
                start_pos=compress_common_attn_metadata.req_metadata.start_pos,
                rope_head_dim=self.rope_head_dim,
                cmp_ratio=self.compress_ratio,
                coff=coff,
                norm_eps=self.compressor_norm_eps,
                rotary_mode=2,
                cache_mode=1)

            if compressed_kv.numel() == 0:
                compressed_kv = None

            torch.ops._C_ascend.npu_scatter_nd_update_v2(
                compress_kv_cache,
                compressor_attn_metadata.req_metadata.slot_mapping,
                compressed_kv)

        # --- sparse attention ---
        if self.compress_ratio <= 1:
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_kv_cache,
                ori_block_table=swa_metadata.req_metadata.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=swa_metadata.req_metadata.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND")[0]
        elif self.compress_ratio == 4:
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                cmp_sparse_indices=compress_topk_idxs,
                ori_block_table=swa_metadata.req_metadata.block_table,
                cmp_block_table=compressor_attn_metadata.req_metadata.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                cu_seqlens_cmp_kv=compress_common_attn_metadata.req_metadata.
                cu_c4_cmp_seqlen_list,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=compress_common_attn_metadata.req_metadata.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND")[0]
        else:
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                ori_block_table=swa_metadata.req_metadata.block_table,
                cmp_block_table=compressor_attn_metadata.req_metadata.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                cu_seqlens_cmp_kv=compress_common_attn_metadata.req_metadata.
                cu_c128_cmp_seqlen_list,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=compressor_attn_metadata.req_metadata.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND")[0]
        return attn_output

    def indexer_select_qli(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: list[M],
        cos: torch.Tensor,
        sin: torch.Tensor,
        compressed_cos: torch.Tensor,
        compressed_sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        qr_pertoken_scale: torch.Tensor = None,
    ):
        (_, _, _, indexer_state_cache, indexer_k_cache, indexer_scale_cache) = kv_cache
        # sorted keys: [attn, compressor.state_cache, indexer.compressor.state_cache, indexer.k_cache, swa_cache]
        (_, _, indexer_kv_state_metadata, indexer_kv_scale_metadata, _) = attn_metadata

        if (not isinstance(self.inderxer_wq_b.quant_method, AscendUnquantizedLinearMethod)) and \
            isinstance(self.inderxer_wq_b.quant_method.quant_method, AscendW8A8DynamicLinearMethod) and \
            qr_pertoken_scale is not None:
            q = torch_npu.npu_quant_matmul(
                qr,
                self.inderxer_wq_b.weight,
                self.inderxer_wq_b.weight_scale,
                pertoken_scale=qr_pertoken_scale,
                bias=self.inderxer_wq_b.bias,
                output_dtype=x.dtype,
            )
        else:
            q = self.inderxer_wq_b(qr)
        q = q.view(-1, self.indexer_heads, self.indexcom_head_dim)  # [T, N, D]

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[
                self.indexcom_head_dim - self.rope_head_dim,
                self.indexcom_head_dim
            ],
        )

        q = rotate_activation(q, indexer_kv_scale_metadata.hadamard)
        coff = 2 if self.compressor_overlap else 1

        assert indexer_kv_scale_metadata.req_metadata is not None
        kv_block_table = indexer_kv_state_metadata.req_metadata.block_table
        start_pos = indexer_kv_scale_metadata.req_metadata.start_pos

        kv = torch.ops._C_ascend.compressor(
            x,
            self.indexcom_wkv.weight,
            self.indexcom_wgate.weight,
            indexer_state_cache.squeeze(-2),
            self.indexcom_ape,
            self.indexcom_norm.weight,
            compressed_sin.view(-1, compressed_sin.shape[-1]),
            compressed_cos.view(-1, compressed_cos.shape[-1]),
            state_block_table=kv_block_table,
            cu_seqlens=actual_seq_lengths_query,
            seqused=None,
            start_pos=start_pos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            coff=coff,
            norm_eps=self.compressor_norm_eps,
            rotary_mode=2,
            cache_mode=1)

        if kv.numel() == 0:
            kv = None
        elif self.indexer.compressor.rotate:
            kv = rotate_activation(kv, indexer_kv_scale_metadata.hadamard)

        weights = self.weights_proj(x) * (self.indexer_softmax_scale *
                                          self.indexer_heads ** -0.5)

        soc_version = get_ascend_device_type()
        dst_type = torch.float8_e4m3fn if soc_version in {AscendDeviceType.A5
                                                          } else torch.int8

        q, q_scale = torch_npu.npu_dynamic_quant(q, dst_type=dst_type)
        if kv is not None:
            kv, kv_scale = torch_npu.npu_dynamic_quant(kv, dst_type=dst_type)
            kv_scale = kv_scale.unsqueeze(-1)

        if soc_version not in {AscendDeviceType.A5}:
            q_scale = q_scale.to(torch.float16)
            if kv is not None:
                kv_scale = kv_scale.to(torch.float16)
                kv_scale = kv_scale.unsqueeze(-1)

        if kv is not None:
            torch.ops._C_ascend.npu_scatter_nd_update_v2(
                indexer_k_cache,
                indexer_kv_scale_metadata.req_metadata.slot_mapping,
                kv)
            if kv_scale is not None:
                torch.ops._C_ascend.npu_scatter_nd_update_v2(
                    indexer_scale_cache,
                    indexer_kv_scale_metadata.req_metadata.slot_mapping,
                    kv_scale)

        assert indexer_kv_scale_metadata.req_metadata is not None
        # In CP mode, use local cumulative query lengths for the indexer attention
        if self.enable_dsa_cp and indexer_kv_scale_metadata.dsa_cp_context is not None:
            cp_ctx = indexer_kv_scale_metadata.dsa_cp_context
            qlens = cp_ctx.actual_seq_lengths_query[1:]
        else:
            qlens = indexer_kv_scale_metadata.req_metadata.query_start_loc[1:]
        kvlens = indexer_kv_scale_metadata.req_metadata.seq_lens
        block_table = indexer_kv_scale_metadata.req_metadata.block_table
        qli_metadata = indexer_kv_scale_metadata.req_metadata.qli_metadata

        topk_idxs, _ = torch.ops._C_ascend.npu_quant_lightning_indexer(
            query=q,
            key=indexer_k_cache,
            weights=weights.to(torch.float16),
            query_dequant_scale=q_scale,
            key_dequant_scale=indexer_scale_cache.squeeze(-2),
            actual_seq_lengths_query=qlens,
            actual_seq_lengths_key=kvlens,
            block_table=block_table,
            metadata=qli_metadata,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            return_value=False)
        return topk_idxs
