# DSA-CP Design for DeepSeek V4

## Overview

DSA-CP (DSA with Context Parallelism) removes TP head sharding from the
attention computation. Each TP rank computes attention for a local token chunk
with the full set of attention heads, while attending to a full-sequence KV
cache.

The design assumes sequence parallelism provides full `hidden_states` to each
rank before the attention stage. DSA-CP then repartitions work by tokens rather
than by attention heads.

## Design Principles

### 1. Query Uses Token Partitioning

- `hidden_states` are available on every rank before attention.
- Query tokens are sliced as `hidden_states[local_start:local_end_with_pad]`.
- The token count is padded to a multiple of TP size so every rank receives the
  same number of query tokens.
- Padding tokens are included for shape consistency and excluded from effective
  attention/output semantics.

### 2. KV Uses Full Sequence

- KV is produced from full-sequence `hidden_states`.
- KV cache updates use the full `slot_mapping` so every rank can attend to the
  same logical KV sequence.
- `wkv` is treated as a replicated projection, so no TP head sharding is applied
  to KV generation.

### 3. Communication Reuses TP Group

- DSA-CP uses the TP communication domain for token partitioning, KV sharing,
  weight gathering, reduce-scatter, and all-to-all exchange.
- No additional CP-only process group is required for the base design.

### 4. Attention Uses Full Heads and Local Tokens

- Query projection produces `num_heads` heads per rank, not `n_local_heads`.
- `wq_b.weight` is represented as full-head weight with shape
  `[q_lora_rank, num_heads * head_dim]`.
- RoPE cos/sin are sliced by local token range and applied consistently to Q and
  KV.
- Sparse attention is invoked with local query tokens and full-sequence KV
  metadata.

### 5. Output Restores TP Layout

- Attention output is first produced as local tokens with full heads.
- Before the TP-sharded output projection, the output is transformed back to the
  per-rank TP head layout.
- The final output projection returns the local token shard expected by the
  surrounding sequence-parallel pipeline.

## Token Metadata

For total token count `T` and TP size `tp_size`:

```text
T_pad = round_up(T, tp_size)
tokens_per_rank = T_pad / tp_size
local_start = tp_rank * tokens_per_rank
local_end_with_pad = local_start + tokens_per_rank
local_end = min(local_end_with_pad, T)
```

Metadata derived from the local token range:

- `slot_mapping_cp`: `slot_mapping[local_start:local_end_with_pad]`
- `cos_cp` / `sin_cp`: RoPE cache sliced by the same local token range
- `actual_seq_lengths_query`: cumulative valid query tokens per request after
  local slicing
- `actual_seq_lengths_key`: effective KV length visible to each local query
  segment

## Data Flow: Activation All-to-All

This design keeps O-proj weights TP-sharded and exchanges activations before the
output projection.

```text
hidden_states_full
  -> local token slice
  -> Q projection with full heads
     [T/tp, num_heads, head_dim]

hidden_states_full
  -> KV projection
  -> full KV cache update
     [T, 1, head_dim]

Q local tokens + full KV cache
  -> sparse attention
     [T/tp, num_heads, head_dim]

attention output
  -> inverse RoPE on output path
  -> activation all-to-all across TP group
     [T, n_local_heads, head_dim]

TP-sharded V-up / O-proj path
  -> reduce-scatter or TP-row output
     [T/tp, hidden_dim]
```

## Data Flow: Full O-Proj Weights

This alternative gathers O-proj weights and executes the output projection in
full-weight mode.

```text
attention output
  -> inverse RoPE on output path
  -> full-head V-up / O-proj with gathered weights
     [T/tp, hidden_dim]
```

Design properties:

- `wo_a` and `wo_b` weights are gathered across TP ranks before the projection.
- `wo_b.reduce_results` is disabled because the projection already uses full
  output weights.
- No activation all-to-all is required before O-proj.

## Weight Requirements

| Parameter | DSA-CP Requirement | Rationale |
|-----------|--------------------|-----------|
| `attn_sink` | Full-head shape `[num_heads]` | Sparse attention runs with full heads on each rank. |
| `wq_b.weight` | Full-head shape `[q_lora_rank, num_heads * head_dim]` | Q projection must produce all heads locally. |
| `wkv.weight` | Replicated | KV generation is full-sequence and not head-sharded. |
| `wo_a.weight` | TP-sharded for activation all-to-all; full for full-weight O-proj | Matches the selected output restoration strategy. |
| `wo_b.weight` | TP-sharded for activation all-to-all; full for full-weight O-proj | Matches the selected output restoration strategy. |

## Correctness Constraints

- Local query slicing must preserve request boundaries in
  `actual_seq_lengths_query`.
- Effective KV lengths must account for the offset between global request end and
  local query end.
- Padding tokens must not write invalid KV slots or contribute to user-visible
  outputs.
- RoPE cache slicing must use the same local token range as Q and KV cache
  updates.
- The output layout after all-to-all or full-weight O-proj must match the
  sequence-parallel contract expected by downstream layers.
