# DSA-CP Design for DeepSeek V4

## Overview

DSA-CP (DSA with Context Parallelism) eliminates TP head sharding during the attention stage, so each rank computes full-head attention on its local token chunk against the full KV cache. FC1 all-gathers `hidden_states` across TP ranks (SP); each rank then independently runs attention with all heads and local tokens.

## Key Principles

### 1. Q reuses FC1's token split
- FC1 `tensor_model_parallel_all_gather` → all ranks have full `hidden_states`
- Q slices: `hidden_states[local_start:local_end_with_pad]` (same `ceil(T/tp)` per rank)
- Token count padded to TP multiple so every rank gets identical workload — no empty tensors

### 2. KV — full sequence, no weight changes needed
- KV from FULL `hidden_states`: `kv = self.wkv(hidden_states)` → `[T, 1, head_dim]`
- `wkv` is `ReplicatedLinear` — weights already fully replicated across TP ranks

### 3. CP reuses TP communication domain
- `enable_dsa_cp()` = `enable_sp()` AND (deepseek_v4 or deepseek_v32)
- `get_tp_group()` for all sharding computation and weight all-gather

### 4. Attention — full heads, two O-proj strategies

Attention operates on `[T/tp, n_heads, head_dim]` (local tokens, FULL heads).

Two correct strategies exist for O-proj (after attention):

---

## Solution A: Full O-proj Weights (prefill / PD-disaggregated P-side)

### Principle
All-gather `wo_a` and `wo_b` so each rank processes all heads through O-proj without communication. `wo_b.reduce_results = False` skips the redundant all-reduce.

### Data flow
```
Q:  hidden_states[local_slice]  → [T/tp, n_heads, head_dim]    (full heads)
KV: wkv(hidden_states_full)     → [T, 1, head_dim]             (full tokens)
Attn:                            → [T/tp, n_heads, head_dim]   (full heads)
V-up proj:                       → [T/tp, n_heads, v_head_dim]
  ↓  NO head slice — keep all heads
wo_a (full weight)              → [T/tp, n_groups, o_lora_rank]
wo_b (full weight, no AR)       → [T/tp, dim]
```

### Weight all-gather (in process_weights_after_loading, called from load_weights)

| Parameter | Shard size | Gather on dim |
|-----------|-----------|---------------|
| `wq_b.weight` | `n_local_heads * head_dim` | dim with that size |
| `wq_b.weight_scale` | `n_local_heads * head_dim` | dim 0 (1D) |
| `wq_b.bias` | `n_local_heads * head_dim` | dim 0 (1D) |
| `wo_a.weight` | `n_local_groups * o_lora_rank` | dim with that size |
| `wo_b.weight` | `n_local_groups * o_lora_rank` | dim with that size |

**Weight layout handling**: W8A8 quant post-processing transposes weights to NZ format `[in, out_shard]`. The `_all_gather_weight` helper detects the shard size on dim 0 or dim 1 and all-gathers correctly regardless of layout.

### Activation (no head slice)
```python
# Solution A: NO slice — keep full heads through O-proj
# o_proj_input stays [T/tp, n_heads, head_dim]
o_proj_input = o_proj_input.view(num_tokens, self.n_group, -1)  # full groups
wo_b_out = self.wo_b(o_proj_input)  # full weight, reduce_results=False
output[...] = wo_b_out  # [T/tp, dim] — correct per-token output with all heads
```

---

## Solution B: Activation All-to-All (decode / PD-mixed)

### Principle
After attention, all-to-all reshapes activation from `[T/tp, n_heads, head_dim]` to `[T, n_local_heads, head_dim]`. Then standard TP-sharded O-proj works unchanged.

### Data flow
```
Q:  hidden_states[local_slice]  → [T/tp, n_heads, head_dim]    (full heads)
KV: wkv(hidden_states_full)     → [T, 1, head_dim]             (full tokens)
Attn:                            → [T/tp, n_heads, head_dim]   (full heads)
  ↓
All-to-All:                      → [T, n_local_heads, head_dim] (shard by heads)
  ↓
V-up proj + O-proj (TP weights)  → [T/tp, dim]                 (standard)
```

### Activation communication
```python
tp_size = get_tp_group().world_size
send = (o_proj_input
        .view(-1, tp_size, n_local_heads, head_dim)
        .permute(1, 0, 2, 3)
        .reshape(-1, n_local_heads * head_dim))  # [T, n_local_heads*head_dim]
o_proj_input = torch.empty_like(send)
torch.distributed.all_to_all_single(o_proj_input, send, group=tp_group.device_group)
# Result: [T, n_local_heads, head_dim]
```

### O-proj output
```python
# Standard TP-sharded O-proj
wo_b_out = self.wo_b(o_proj_input)
output[...] = tensor_model_parallel_reduce_scatter(wo_b_out, dim=0)
```

### Weight handling — none needed
Solution B uses original TP-sharded weights; no all-gather required.

---

## Strategy Selection

`enable_dsa_cp_with_o_proj_tp()` (`utils.py:1386`) controls the choice:

| Flag value | O-proj strategy | Use case |
|-----------|----------------|---------|
| `False` | Solution A (full weight) | PD-disaggregated P-side, standalone prefill |
| `True` (default when PD-mixed) | Solution B (all-to-all) | PD-mixed decode |

---

## Weight Loading Call Chain

`DSAAttention` does NOT inherit from vllm's `Attention`/`MLAAttention`, so vllm's `process_weights_after_loading` does not call it. Instead:

1. `AscendDeepseekV4ForCausalLM.load_weights()` loads all raw weights (TP-sharded)
2. At end of `load_weights`: iterates all `DSAAttention` modules, calls `module.process_weights_after_loading(act_dtype)` — this all-gathers `wq_b`, `wo_a`, `wo_b` in standard layout (before quant post-processing)
3. vllm's `process_weights_after_loading()` runs: quant post-processing transposes + NZ-formats the now-full weights
4. Mark `_dsa_cp_weights_processed = True` to make the call idempotent

---

## Metadata Builder

`AscendDSACPMetadataBuilder` builds `AscendDSAMetadata` with `DSACPContext` containing per-rank token shard info. It is a sibling class (not subclass) of `AscendDSAMetadataBuilder`.

Model runner `_build_attn_group_metadata` checks:
```python
isinstance(builder, (AscendDSAMetadataBuilder, AscendDSACPMetadataBuilder))
```
to populate `extra_attn_metadata_args` for both builder types.

---

## Files

| File | Role |
|------|------|
| `attention/context_parallel/dsa_cp.py` | AscendDSACPImpl (forward), AscendDSACPMetadataBuilder (metadata) |
| `attention/dsa_v1.py` | AscendDSABackend (backend routing), AscendDSAImpl (base) |
| `models/deepseek_v4.py` | Model init (sink size), load_weights (sink loading + DSAAttention post-process) |
| `worker/model_runner_v1.py` | Metadata builder dispatch |
| `models/layer/attention/layer.py` | DSAAttention (wraps impl, delegates process_weights_after_loading) |
| `utils.py` | `enable_dsa_cp()`, `enable_dsa_cp_with_o_proj_tp()` |

## Appendix: Token Padding

FC1 `tensor_model_parallel_all_gather` pads input to TP multiple, then UNPADs via `x[:-pad_size]`. For DSA-CP, we RE-PAD `hidden_states` to `num_tokens_pad = ceil(T/tp) * tp` before Q slicing so each rank gets exactly `num_tokens_per_device` tokens. KV is computed from original (unpadded) hidden_states. Padding tokens produce zero attention output and do not affect results.

```python
num_pad = cp_ctx.num_tokens_pad - hidden_states.shape[0]
if num_pad > 0:
    hs_for_q = F.pad(hidden_states, (0, 0, 0, num_pad))
hidden_states_q = hs_for_q[cp_ctx.local_start:cp_ctx.local_end_with_pad]
```
