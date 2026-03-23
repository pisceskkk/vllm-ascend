# Context Parallel Attention Metadata 开发指南

本文档梳理 Ascend 后端中 **Context Parallel（CP，包含 PCP / DCP）** 相关的 `attention_metadata` 变量结构与用途，供开发者理解 CP 注意力元数据的构建链路、字段含义以及 PCP/DCP 场景下的差异。

> **相关文档**：用户使用指南见 [Context Parallel User Guide](../../user_guide/feature_guide/context_parallel.md)；CP 原理概述见 [Context Parallel 开发指南](context_parallel.md)。

---

## 目录

1. [概览](#概览)
2. [构建链路](#构建链路)
3. [CommonAttentionMetadata 基类字段](#commonattentionmetadata-基类字段)
4. [AscendCommonAttentionMetadata 扩展字段](#ascendcommonattentionmetadata-扩展字段)
5. [AscendPrefillContextParallelMetadata 字段详解](#ascendprefillcontextparallelmetadata-字段详解)
6. [下游结构：AscendPCPMetadata 与 AscendMetadata](#下游结构ascendpcpmetadata-与-ascendmetadata)
7. [PCP 与 DCP 场景差异](#pcp-与-dcp-场景差异)
8. [与通用字段的关系](#与通用字段的关系)
9. [关键代码位置索引](#关键代码位置索引)

---

## 概览

在 Ascend 后端的 CP 实现中，`attention_metadata` 按层级组织：

```
AscendCommonAttentionMetadata          ← 跨层共享、按 batch 构建
    ├─ (继承自) CommonAttentionMetadata    ← vLLM 上游基类
    ├─ seq_lens_cpu / num_computed_tokens_cpu / positions / ...  ← Ascend 扩展字段
    └─ prefill_context_parallel_metadata: AscendPrefillContextParallelMetadata
           ├─ pcp_allgather_restore_idx        ← AllGather 后恢复顺序的索引
           ├─ num_actual_tokens_pcp_padded     ← 含 padding 的 token 总数
           ├─ num_computed_tokens_of_pcp_dcp   ← 各 (pcp, dcp) rank 的本地 KV 长度
           ├─ q_head_idx_tensor / q_tail_idx_tensor  ← PCP 序列 head/tail 分块索引
           ├─ kv_with_q_head_*/kv_with_q_tail_*      ← nomask/mask 区域 KV 索引
           ├─ attn_mask_seqlens / *_nomask_seqlens    ← Attention 分段 seqlens
           ├─ q_full_idx                       ← head+tail 合并后的排序索引
           ├─ query_lens_pcp_full_cpu          ← PCP 分割前的原始 query 长度（MTP 用）
           ├─ max_query_len_pcp_full           ← PCP 分割前的最大 query 长度
           ├─ pcp_use_hybrid_attn             ← 是否使用 hybrid-attn（qwen3_next）
           ├─ pcp_unpad_mask                  ← 标记 padded allgather buffer 中真实 token
           ├─ pcp_fa_query_idx                ← hybrid-attn: FA 阶段 query 重排索引
           ├─ pcp_enter_fa_restore_idx        ← hybrid-attn: 线性注意力→FA 切换时的恢复索引
           └─ pcp_padded_tokens_fla           ← hybrid-attn: 线性注意力阶段的 padding token 数

然后由 AscendAttentionCPMetadataBuilder.build() 进一步生成：
AscendMetadata
    ├─ prefill: AscendMetadataForPrefill
    │       └─ pcp_metadata: AscendPCPMetadata   ← 复制 PCP 索引字段
    └─ decode_meta: AscendMetadataForDecode
            └─ num_computed_tokens_of_pcp_dcp    ← 各 rank 本地 KV 长度（numpy）
```

---

## 构建链路

### 1. `build_attn_metadata`（入口）

**文件**：`vllm_ascend/worker/v2/attn_utils.py::build_attn_metadata`

该函数是 Attention Metadata 的顶层入口，由 ModelRunner 在每次 forward 前调用。它接收以下参数并构建 `AscendCommonAttentionMetadata`：

| 参数 | 说明 |
|------|------|
| `query_start_loc_gpu` / `query_start_loc_cpu` | 每个请求 query 在 token 序列中的起始偏移（分 GPU/CPU 版本） |
| `seq_lens` / `max_seq_len` | 各请求的序列长度（已计算 + 新 token） |
| `block_tables` / `slot_mappings` | KV cache 的物理块表和槽位映射 |
| `num_computed_tokens_cpu` | 各请求已计算的 token 数（CPU tensor） |
| `positions` | 各 token 在序列中的位置（用于 RoPE） |
| `prefill_context_parallel_metadata` | PCP 元数据（由 `PCPManager.generate_pcp_metadata` 生成） |

构建完 `AscendCommonAttentionMetadata` 后，调用 `attn_metadata_builder.build()` 生成每层的 `AscendMetadata`：

- **无 CP**：使用 `AscendAttentionMetadataBuilder.build()`
- **有 CP**：使用 `AscendAttentionCPMetadataBuilder.build()`（在 `attention_cp.py` 中）

### 2. `PCPManager.generate_pcp_metadata`（PCP 元数据生成）

**文件**：`vllm_ascend/worker/pcp_utils.py::PCPManager.generate_pcp_metadata`

该函数构建 `AscendPrefillContextParallelMetadata`，包含 PCP 的所有分块索引。调用链：

```
ModelRunner._prepare_inputs()
  └─ pcp_manager.update_tokens_for_pcp()     # 计算分块位置，填充 pcp_allgather_restore_idx
  └─ pcp_manager.generate_pcp_metadata()     # 构建 AscendPrefillContextParallelMetadata
        ├─ _get_cp_local_seq_lens()          # 计算各 (pcp, dcp) rank 的本地 KV 长度
        ├─ 计算 q_head_idx/q_tail_idx/kv 索引
        └─ 填充 long_seq_metadata 所有字段
  └─ build_attn_metadata()                   # 构建 AscendCommonAttentionMetadata
        └─ AscendAttentionCPMetadataBuilder.build()  # 构建每层 AscendMetadata
```

### 3. `AscendAttentionCPMetadataBuilder.build`（每层元数据）

**文件**：`vllm_ascend/attention/context_parallel/attention_cp.py::AscendAttentionCPMetadataBuilder.build`

从 `AscendCommonAttentionMetadata` 中读取 `prefill_context_parallel_metadata`，将字段分发给：
- `AscendMetadataForPrefill.pcp_metadata`（prefill 阶段，GQA backend）
- `AscendMetadataForDecode`（decode 阶段）

---

## CommonAttentionMetadata 基类字段

**定义**：`vllm.v1.attention.backends.utils.CommonAttentionMetadata`（vLLM 上游）

这些字段被 `AscendCommonAttentionMetadata` 继承，是所有后端共用的基础字段：

| 字段名 | 类型 / Shape | 位置 | 用途 |
|--------|-------------|------|------|
| `query_start_loc` | `torch.Tensor` (int32, GPU), shape `[num_reqs + 1]` | GPU | 每个请求 query 在 token 序列的起始偏移（累积和），用于 FA kernel |
| `query_start_loc_cpu` | `torch.Tensor` (int32, CPU), shape `[num_reqs + 1]` | CPU | 同上，CPU 版本，用于 metadata builder 中的计算 |
| `seq_lens` | `torch.Tensor` (int32, GPU), shape `[num_reqs]` | GPU | 每个请求的完整序列长度（已计算 + 新 token） |
| `num_reqs` | `int` | - | batch 中请求数 |
| `num_actual_tokens` | `int` | - | 实际 token 数（不含 CUDA Graph padding） |
| `max_query_len` | `int` | - | batch 中最长 query 长度 |
| `block_table_tensor` | `torch.Tensor` (int32, GPU), shape `[num_reqs, max_blocks]` | GPU | 每个请求的物理 KV block 表 |
| `slot_mapping` | `torch.Tensor` (int32, GPU), shape `[num_tokens]` | GPU | 每个 token 对应 KV cache 的物理 slot 索引 |
| `causal` | `bool` | - | 是否使用因果 attention mask |
| `max_seq_len` | `int` | - | batch 中最大序列长度 |

---

## AscendCommonAttentionMetadata 扩展字段

**定义**：`vllm_ascend/attention/utils.py::AscendCommonAttentionMetadata`

继承自 `CommonAttentionMetadata`，新增以下 Ascend 特有字段：

| 字段名 | 类型 / Shape | CPU/GPU | 用途 |
|--------|-------------|---------|------|
| `seq_lens_cpu` | `torch.Tensor` (int32), shape `[num_reqs]` | CPU | 序列长度的 CPU 副本，用于 metadata builder 中无需 D2H 的计算（如 `seq_lens - query_lens` 得到 `num_computed_tokens`） |
| `num_computed_tokens_cpu` | `torch.Tensor` (int32), shape `[num_reqs]` | CPU | 每请求已计算 token 数；spec decode 场景下在 eagle/mtp proposer 中更新（`+= 1`） |
| `decode_token_per_req` | `int` | - | 每请求 decode token 数（正常 decode=1；spec decode=num_speculative_tokens+1） |
| `actual_seq_lengths_q` | `list[int]`, len `num_actual_tokens` | CPU | 每个 token 对应的 query 累积序列长度（供 FA 使用的 actual_seq_lengths 参数） |
| `positions` | `torch.Tensor` (int64, GPU), shape `[num_tokens]` | GPU | 各 token 在序列中的位置索引（用于 RoPE 计算）；PCP 下由 `update_tokens_for_pcp` 计算分块位置 |
| `attn_state` | `AscendAttentionState` | - | 当前 batch 的 attention 状态（PrefillNoCache / PrefillCacheHit / DecodeOnly / ChunkedPrefill / SpecDecoding） |
| `graph_pad_size` | `int` | - | CUDA Graph padding 大小，`-1` 表示非 graph 模式 |
| `num_input_tokens` | `int` | - | 含 padding 的输入 token 总数 |
| `prefill_context_parallel_metadata` | `AscendPrefillContextParallelMetadata \| None` | - | PCP 元数据，仅在 pcp_size > 1 或 dcp_size > 1 时非 None |

---

## AscendPrefillContextParallelMetadata 字段详解

**定义**：`vllm_ascend/attention/utils.py::AscendPrefillContextParallelMetadata`

**构造位置**：`vllm_ascend/worker/pcp_utils.py::PCPManager.generate_pcp_metadata`

下表列出所有字段，标注推断类型时注明依据：

### 通用字段（PCP 和 DCP 均使用）

| 字段名 | 类型 / Shape | 在哪构造 | 在哪使用 | 用途简述 | 备注 |
|--------|-------------|---------|---------|---------|------|
| `num_actual_tokens_pcp_padded` | `int` | `generate_pcp_metadata` | `AscendAttentionCPMetadataBuilder.build`; `eagle_proposer` | `total_num_scheduled_tokens * pcp_world_size`，表示 AllGather 后（含 padding）的 token 总数 | 用于截取 `slot_mapping` 长度 |
| `num_computed_tokens_of_pcp_dcp` | `numpy.ndarray`，shape `[num_reqs_flatten * decode_threshold, pcp_size, dcp_size]` (int32) | `generate_pcp_metadata::_get_cp_local_seq_lens` | `AscendAttentionCPMetadataBuilder.build`→`AscendMetadataForDecode`；CP decode 阶段 FA kernel | 每个 (pcp_rank, dcp_rank) 的本地 KV 长度（即该 rank 存储的已计算 token 数）；decode 阶段用作 `actual_seq_lengths_kv` | spec decode 下展平 decode 请求 |
| `pcp_unpad_mask` | `torch.Tensor` (bool), shape `[num_padded_tokens]` | `update_tokens_for_pcp`，`generate_pcp_metadata` 打包 | `AscendAttentionCPMetadataBuilder.build` | 标记 padded allgather buffer 中哪些位置是真实 token（非 pad）；DualChunkSwap 对齐后部分位置为 False | |
| `pcp_use_hybrid_attn` | `bool` | `PCPManager.__init__` | 各处判断分支 | 是否使用 hybrid-attn 模式（目前仅 `qwen3_next` 模型为 True） | |
| `pcp_padded_tokens_fla` | `int` | `update_tokens_for_pcp`（hybrid 路径） | `get_restore_hidden_states` | hybrid-attn 下线性注意力阶段的 padding token 数 | 仅 hybrid-attn 路径有效 |

### PCP prefill 阶段字段（pcp_world_size > 1，非 hybrid-attn）

| 字段名 | 类型 / Shape | 在哪构造 | 在哪使用 | 用途简述 | 备注 |
|--------|-------------|---------|---------|---------|------|
| `pcp_allgather_restore_idx` | `torch.Tensor` (int64, GPU), shape `[num_actual_tokens_pcp_padded]` | `update_tokens_for_pcp::pcp_allgather_restore_idx.np` → GPU | prefill FA 后、`get_restore_hidden_states`; chunked prefill KV 恢复 | AllGather 后用于恢复 token 原始顺序的索引；`hidden_states[restore_idx]` 即还原顺序 | 每个 pcp rank 对应的 head/tail chunk 拼合后，通过 argsort 得到恢复索引 |
| `q_head_idx_tensor` | `torch.Tensor` (int32, GPU), shape `[num_prefill_chunk_tokens]`（推断） | `generate_pcp_metadata` | `AscendPCPMetadata.q_head_idx` | prefill 时各请求 head chunk 中 query token 在本 rank token 序列中的索引 | DualChunkSwap：每请求 query 分为 head（前半）和 tail（后半） |
| `q_tail_idx_tensor` | `torch.Tensor` (int32, GPU), shape `[num_prefill_chunk_tokens]`（推断） | `generate_pcp_metadata` | `AscendPCPMetadata.q_tail_idx` | prefill 时各请求 tail chunk 中 query token 的索引 | |
| `kv_with_q_head_nomask_idx_tensor` | `torch.Tensor` (int32, GPU) | `generate_pcp_metadata` | `AscendPCPMetadata.kv_with_q_head_nomask_idx` | head chunk query 对应的"无需 causal mask"区域 KV token 索引（即 chunk_id < q_head_chunk_id 的部分） | GQA backend prefill FA 的 nomask 计算输入 |
| `kv_with_q_head_mask_idx_tensor` | `torch.Tensor` (int32, GPU) | `generate_pcp_metadata` | `AscendPCPMetadata.kv_with_q_head_mask_idx` | head chunk query 对应的"需要 causal mask"区域 KV token 索引 | 与 `kv_with_q_head_nomask_idx` 合并覆盖 head chunk 的完整 KV 范围 |
| `kv_with_q_tail_nomask_idx_tensor` | `torch.Tensor` (int32, GPU) | `generate_pcp_metadata` | `AscendPCPMetadata.kv_with_q_tail_nomask_idx` | tail chunk query 对应的无 mask KV 索引 | |
| `kv_with_q_tail_mask_idx_tensor` | `torch.Tensor` (int32, GPU) | `generate_pcp_metadata` | `AscendPCPMetadata.kv_with_q_tail_mask_idx` | tail chunk query 对应的需 mask KV 索引 | |
| `attn_mask_seqlens` | `torch.Tensor` (int32), shape `[2, num_prefill_reqs]` → 在 builder 中转为 `list[int]` | `generate_pcp_metadata` | `AscendAttentionCPMetadataBuilder.build` → FA seqlens 参数 | head/tail chunk 各自的 query 长度列表（`[chunk_seqlens, chunk_seqlens]`），用于 FA 带 causal mask 的计算 | 在 builder 中 `cumsum` 后转为累积 seqlens 列表 |
| `head_attn_nomask_seqlens` | `torch.Tensor` (int32), shape `[2, num_prefill_reqs]` → `list[int]` | `generate_pcp_metadata` | FA nomask 路径的 kv seqlens | head chunk query 对应的 nomask KV 长度（`kv_with_q_head_nomask_seqlens`），用于无 mask FA 计算 | |
| `tail_attn_nomask_seqlens` | `torch.Tensor` (int32), shape `[2, num_prefill_reqs]` → `list[int]` | `generate_pcp_metadata` | FA nomask 路径的 kv seqlens | tail chunk query 对应的 nomask KV 长度 | |
| `q_full_idx` | `torch.Tensor` (int32, GPU), shape `[num_prefill_tokens]` | `generate_pcp_metadata`：`argsort([q_head_idx, q_tail_idx])` | prefill FA 后恢复 head/tail 分割前的 query 顺序 | 将按 head/tail 分割后的 query 输出恢复原始 token 顺序 | `torch.cat([q_head_idx, q_tail_idx]).argsort()` |

### MTP（Speculative Decode）相关字段

| 字段名 | 类型 / Shape | 在哪构造 | 在哪使用 | 用途简述 | 备注 |
|--------|-------------|---------|---------|---------|------|
| `query_lens_pcp_full_cpu` | `torch.Tensor` (int32, CPU), shape `[num_reqs_padded]` | `generate_pcp_metadata`（仅 spec decode 且 pcp > 1） | `split_decodes_and_prefills`；MTP proposer | PCP 分割**前**的原始 query 长度，用于区分 decode/prefill 请求 | 仅在 speculative_config 非 None 且 pcp_world_size > 1 时有效；spec decode 下 query_lens 经过 PCP 分割，需用原始值判断 |
| `max_query_len_pcp_full` | `int` | `generate_pcp_metadata` | `split_decodes_and_prefills` | PCP 分割前的最大 query 长度 | 同上 |

### Hybrid-Attn 专用字段（pcp_use_hybrid_attn=True，即 qwen3_next）

| 字段名 | 类型 / Shape | 在哪构造 | 在哪使用 | 用途简述 | 备注 |
|--------|-------------|---------|---------|---------|------|
| `pcp_fa_query_idx` | `torch.Tensor` (int32, GPU), shape `[num_prefill_tokens_rank]`（推断） | `update_tokens_for_pcp`（hybrid 路径）→ `generate_pcp_metadata` 截取 | `AscendPCPMetadata.pcp_fa_query_idx` | hybrid-attn 下 FA 阶段各 rank 的 query token 重排索引（线性注意力输出→FA 输入的顺序） | qwen3_next 专用：先线性注意力（FLA）再 FlashAttention |
| `pcp_enter_fa_restore_idx` | `torch.Tensor` (int32, GPU), shape 动态 | `update_tokens_for_pcp`（hybrid 路径）→ `generate_pcp_metadata` 截取 | `AscendPCPMetadata.pcp_enter_fa_restore_idx`；`get_restore_hidden_states`（hybrid 路径） | 从线性注意力阶段切换到 FA 阶段时，AllGather 后恢复跨 rank token 全局顺序的索引 | 包含 decode 部分（对应 `enter_fa_decode_restore_idx`）和 prefill 部分（`enter_fa_prefill_restore_idx`） |

---

## 下游结构：AscendPCPMetadata 与 AscendMetadata

### AscendPCPMetadata

**定义**：`vllm_ascend/attention/context_parallel/common_cp.py::AscendPCPMetadata`

由 `AscendAttentionCPMetadataBuilder.build` 从 `AscendPrefillContextParallelMetadata` 复制字段而来，作为 `AscendMetadataForPrefill.pcp_metadata` 的内容，供 GQA backend（`attention_cp.py`）在 prefill 时使用：

| 字段名 | 来源字段 | 用途 |
|--------|---------|------|
| `q_head_idx` | `q_head_idx_tensor` | head chunk query 索引 |
| `q_tail_idx` | `q_tail_idx_tensor` | tail chunk query 索引 |
| `kv_with_q_head_nomask_idx` | `kv_with_q_head_nomask_idx_tensor` | head chunk nomask KV 索引 |
| `kv_with_q_head_mask_idx` | `kv_with_q_head_mask_idx_tensor` | head chunk mask KV 索引 |
| `kv_with_q_tail_nomask_idx` | `kv_with_q_tail_nomask_idx_tensor` | tail chunk nomask KV 索引 |
| `kv_with_q_tail_mask_idx` | `kv_with_q_tail_mask_idx_tensor` | tail chunk mask KV 索引 |
| `attn_mask_seqlens` | `attn_mask_seqlens`（cumsum 后） | FA mask 计算的 seqlens |
| `head_attn_nomask_seqlens` | `head_attn_nomask_seqlens`（cumsum 后） | head chunk nomask FA 的 kv seqlens |
| `tail_attn_nomask_seqlens` | `tail_attn_nomask_seqlens`（cumsum 后） | tail chunk nomask FA 的 kv seqlens |
| `q_full_idx` | `q_full_idx` | 恢复 head/tail 分割前顺序 |
| `pcp_allgather_restore_idx` | `pcp_allgather_restore_idx` | AllGather 后恢复 token 顺序 |
| `pcp_unpad_mask` | `pcp_unpad_mask` | 标记真实 token 的 mask |
| `pcp_use_hybrid_attn` | `pcp_use_hybrid_attn` | 是否 hybrid-attn |
| `pcp_fa_query_idx` | `pcp_fa_query_idx` | hybrid-attn FA query 重排索引 |
| `pcp_enter_fa_restore_idx` | `pcp_enter_fa_restore_idx` | hybrid-attn 线性→FA 恢复索引 |
| `pcp_padded_tokens_fla` | `pcp_padded_tokens_fla` | hybrid-attn FLA padding 数 |

### AscendMetadata

**定义**：`vllm_ascend/attention/attention_v1.py::AscendMetadata`

这是每层 attention 使用的最终元数据，在 CP 场景下包含以下关键字段：

| 字段名 | 用途 |
|--------|------|
| `num_actual_tokens_pcp_padded` | PCP AllGather 后的 token 总数（含 padding），用于截取 slot_mapping |
| `num_decodes_flatten` | spec decode 下展平后的 decode token 总数（`query_lens_decode.sum()`） |
| `prefill: AscendMetadataForPrefill` | prefill 阶段元数据，含 `pcp_metadata` 和 chunked context metadata |
| `decode_meta: AscendMetadataForDecode` | decode 阶段元数据，含 `num_computed_tokens_of_pcp_dcp` (numpy array) |

---

## PCP 与 DCP 场景差异

### PCP（Prefill Context Parallel）

PCP 对 prefill 序列按 DualChunkSwap 方式分割，每个 rank 处理序列的 head+tail 各 `1/(2*pcp_size)` 部分：

1. **Token 分割**（`update_tokens_for_pcp`）：
   - 序列长度填充至 `2 * pcp_size` 的倍数
   - 分为 `2 * pcp_size` 段，rank `r` 取第 `r` 段（head）和第 `2*pcp_size-1-r` 段（tail）
   - decode 请求不分割，在所有 rank 上复制

2. **AllGather 恢复**：prefill FA 计算后，通过 `pcp_allgather_restore_idx` 恢复 token 原始顺序（`get_restore_hidden_states`）

3. **专有元数据**：`q_head_idx_tensor`、`q_tail_idx_tensor`、`kv_with_q_*_idx_tensor`、`attn_mask_seqlens`、`*_nomask_seqlens`、`q_full_idx`

4. **slot_mapping 扩展**（`get_padded_slot_mapping`）：KV cache 按 CP 方式分片，slot_mapping 需填充至 `num_tokens * pcp_world_size`，并由 `pcp_unpad_mask` 标记真实位置

### DCP（Decode Context Parallel）

DCP 复用 TP 通信域，主要影响 decode 和 chunked prefill 阶段的 KV cache 访问：

1. **本地 KV 长度**：`num_computed_tokens_of_pcp_dcp[:, pcp_rank, dcp_rank]` 给出该 rank 实际存储的 KV token 数，按 `cp_kv_cache_interleave_size` 粒度交织分布

2. **block_table / slot_mapping**：CP（PCP + DCP）共同影响 KV cache 的物理 slot 计算（见 `vllm_ascend/worker/block_table.py`），slot_mapping 的值已经根据 cp_rank 和 interleave 调整

3. **无专有 prefill 索引**：DCP 无 PCP 的 head/tail 分割索引；decode 阶段的 Q AllGather（head 维度）和 `cp_lse_ag_out_rs` 由 attention impl 内部处理

4. **`num_computed_tokens_of_pcp_dcp`**：该字段对 PCP 和 DCP 均有效，shape 为 `[num_reqs_flatten, pcp_size, dcp_size]`，通过 `_get_cp_local_seq_lens` 计算各 rank 的本地序列长度

### 混合模式（PCP + DCP）

两者同时启用时：
- `cp_size = pcp_size * dcp_size`，`cp_rank = pcp_rank * dcp_size + dcp_rank`
- block table 中虚拟块大小 = `block_size * cp_size`
- `num_computed_tokens_of_pcp_dcp` 的第二、三维分别对应 pcp 和 dcp 维度

---

## 与通用字段的关系

### slot_mapping 与 CP

`slot_mapping`（来自 `CommonAttentionMetadata`）在 CP 场景下经过调整：

- **构造位置**：`vllm_ascend/worker/block_table.py::compute_slot_mapping`（CP 版本）
- **PCP 扩展**：`PCPManager.get_padded_slot_mapping` 将 slot_mapping 扩展为 `num_tokens_padded * pcp_world_size`，配合 `pcp_unpad_mask` 在真实 token 位置填充原始 slot，pad 位置填 `-1`，用于 AllGather 后的 KV cache 写入
- **DCP 调整**：slot 值按 CP interleave 规则调整，使 token 的 KV 存储到对应 rank 的物理 block

### block_tables 与 CP

- `block_tables`（`CommonAttentionMetadata.block_table_tensor`）在 spec decode + PCP 场景下被**展平**（flatten）：decode 请求的 block table 行按 `num_spec_tokens` 复制展开（`generate_pcp_metadata` 中处理）
- `AscendMetadataForPrefill.block_tables` 和 `AscendMetadataForDecode.block_tables` 分别持有 prefill 和 decode 部分的切片

### seq_lens / num_computed_tokens 与 CP

- `seq_lens_cpu`（全局序列长度）在 CP 下**不变**，仍表示完整序列长度
- `num_computed_tokens_of_pcp_dcp` 是各 rank 的**局部**已计算 token 数，由 `_get_cp_local_seq_lens` 从全局 `num_computed_tokens` 推导
- 在 decode FA 中，`actual_seq_lengths_kv` 使用的是 `num_computed_tokens_of_pcp_dcp[:, pcp_rank, dcp_rank]`

### query_start_loc 与 PCP 分割

- PCP 分割后，`query_start_loc` 中的 query 长度是**分割后**的长度（约为原来的 `1 / pcp_size`）
- `query_lens_pcp_full_cpu` 保存分割**前**的原始长度，`split_decodes_and_prefills` 在 PCP > 1 时使用此字段来正确区分 decode/prefill 请求

---

## 关键代码位置索引

| 文件 | 描述 |
|------|------|
| `vllm_ascend/attention/utils.py` | `AscendPrefillContextParallelMetadata`、`AscendCommonAttentionMetadata` 定义；`split_decodes_and_prefills` |
| `vllm_ascend/worker/v2/attn_utils.py` | `build_attn_metadata`（顶层入口）；`build_attn_state` |
| `vllm_ascend/worker/pcp_utils.py` | `PCPManager`：`update_tokens_for_pcp`（token 分割与 restore 索引计算）；`generate_pcp_metadata`（完整 PCP 元数据构建）；`_get_cp_local_seq_lens`（各 rank KV 长度计算）；`get_padded_slot_mapping`；`get_restore_hidden_states` |
| `vllm_ascend/attention/attention_v1.py` | `AscendAttentionMetadataBuilder.build`（非 CP 路径）；`AscendMetadata` 定义 |
| `vllm_ascend/attention/context_parallel/attention_cp.py` | `AscendAttentionCPMetadataBuilder.build`（CP 路径 builder）；`AscendAttentionCPImpl`（CP attention 实现） |
| `vllm_ascend/attention/context_parallel/common_cp.py` | `AscendPCPMetadata`；`AscendMetadataForPrefill`；`AscendMetadataForDecode`；`_process_attn_out_lse`；`_npu_attention_update` |
| `vllm_ascend/attention/context_parallel/mla_cp.py` | MLA backend 的 CP 实现（AllGatherKV 策略） |
| `vllm_ascend/worker/block_table.py` | CP 场景下 slot_mapping 的计算（CP-aware block table） |
| `vllm_ascend/spec_decode/eagle_proposer.py` | EAGLE proposer 中 CP 下 slot_mapping 的多步更新；`pcp_allgather_restore_idx` 在 spec decode 中的应用 |
| `vllm_ascend/worker/model_runner_v1.py` | ModelRunner 中调用 `update_tokens_for_pcp` 和 `generate_pcp_metadata` 的位置 |
