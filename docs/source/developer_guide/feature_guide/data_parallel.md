# Data Parallel (DP) 处理流程

本文档梳理了 vLLM Ascend 中 Data Parallel（DP）的完整处理流程及涉及的设备间通信操作。

---

## 目录

- [DP 概念说明](#dp-概念说明)
- [DP 配置与拓扑](#dp-配置与拓扑)
- [DP 处理流程](#dp-处理流程)
  - [1. 引擎层 DP 调度](#1-引擎层-dp-调度)
  - [2. Batch 元数据同步](#2-batch-元数据同步)
  - [3. MoE 前向传播中的 DP 通信](#3-moe-前向传播中的-dp-通信)
- [DP 通信汇总](#dp-通信汇总)
- [DP 通信跳过条件](#dp-通信跳过条件)
- [关键源文件索引](#关键源文件索引)

---

## DP 概念说明

vLLM Ascend 中存在两类语境下的 DP：

### 1. 外部 DP（ExternalDP）

服务级别的数据并行，多个独立的模型副本分别处理不同的请求集合，常用于 verl 等训练集成场景。通过 `data_parallel_size` 参数配置，每个 DP rank 运行独立的模型副本，并通过 `DPEngineCoreProc` / `BalanceDPEngineCoreProc` 进行协调。

### 2. MoE 内部 DP

在 MoE（Mixture-of-Experts）模型的前向传播中，不同 DP rank 持有不同的 token 切片。在进入专家网络之前，各 DP rank 之间需要做通信，把所有 rank 的 token 聚合（All-Gather）后再路由到对应的专家，并在专家计算后把结果散回（Reduce-Scatter）各 rank。这一过程与 EP（Expert Parallelism）深度结合。

> **注意**："DCP"（Decode Context Parallel）是另一个概念，指解码阶段的序列并行，与本文的 DP 不同，但在 model runner 初始化时会一并处理。

---

## DP 配置与拓扑

### 关键配置参数（`ParallelConfig`）

| 参数 | 说明 |
|------|------|
| `data_parallel_size` | 全局 DP 总数 |
| `data_parallel_rank` | 当前进程的 DP rank 编号 |
| `data_parallel_size_local` | 本节点上的 DP 数量 |
| `data_parallel_backend` | DP 后端（`"ray"` 或 `"external_launcher"` 等） |

### 通信组布局

在 `vllm_ascend/distributed/parallel_state.py` 中，`init_ascend_model_parallel()` 建立如下拓扑：

```
vllm_all_ranks shape: [ExternalDP, dp_size, pp_size, pcp_size, tp_size]

MC2 组 (mc2):
  all_ranks shape: [ExternalDP, dp_size * pcp_size * tp_size]
  → 每一行构成一个 MC2 group，覆盖 DP * PCP * TP 所有 rank
```

MC2 group 是 vLLM Ascend 中用于 MoE token dispatch/combine 的关键通信组，它合并了 DP 与 EP 的通信范围。

---

## DP 处理流程

### 1. 引擎层 DP 调度

**文件**：`vllm_ascend/patch/platform/patch_balance_schedule.py`

`BalanceDPEngineCoreProc` 继承自上游 `DPEngineCoreProc`，其 `run_busy_loop()` 是 DP 场景下引擎的主循环：

```
┌─────────────────────────────────────────────────────────┐
│ BalanceDPEngineCoreProc.run_busy_loop()                  │
│                                                           │
│  while True:                                              │
│   1. _process_input_queue()  ← 处理新请求                │
│   2. _process_engine_step()  ← 调度 & 执行模型           │
│      └── 若无任务，execute_dummy_batch()                  │
│          ← 确保所有 DP rank 都调用 DP 通信原语，避免挂起  │
│   3. _has_global_unfinished_reqs()  ← CPU All-Reduce    │
│      ← 跨 DP ranks 同步"是否还有未完成请求"              │
│   4. scheduler.balance_gather(dp_group)  ← CPU All-Gather│
│      ← 各 rank 广播自己的 running 队列长度，用于负载均衡  │
└─────────────────────────────────────────────────────────┘
```

**关键设计**：当某 DP rank 没有实际请求可执行时，必须调用 `execute_dummy_batch()`，否则其他 rank 在 `_has_global_unfinished_reqs()` 的 All-Reduce 处会永久等待，造成死锁。

同样地，当 `num_scheduled_tokens == 0` 但 `data_parallel_size > 1` 时，`execute_model()` 中也会主动调用 `_dummy_run(1)` 来保证所有 DP rank 同步：

```python
# vllm_ascend/worker/model_runner_v1.py
if not num_scheduled_tokens:
    if (
        self.parallel_config.distributed_executor_backend == "external_launcher"
        and self.parallel_config.data_parallel_size > 1
    ):
        self._dummy_run(1)
```

---

### 2. Batch 元数据同步

**文件**：`vllm_ascend/worker/model_runner_v1.py`

在每次模型 forward 之前，需要在所有 DP ranks 之间同步 batch 的元数据，以确保所有 rank 使用相同的 token 数量（以便 padding 一致）和相同的 cudagraph 运行模式。

#### 2a. `_sync_metadata_across_dp()`：同步 token 数与 prefill 标志

```
每个 DP rank 独立 → 调用本函数 → 全局对齐

num_tokens_tensor: [0, ..., num_tokens_i, ..., 0]  (只填自己的位置)
flags_tensor:      [int(with_prefill)]

packed_tensor = torch.cat([num_tokens_tensor, flags_tensor])  # 伪代码：拼接两段张量

dist.all_reduce(packed_tensor, group=get_dp_group().cpu_group)
   ↑ CPU 侧 All-Reduce（求和），在 CPU 上异步进行，不阻塞 NPU 计算

结果：
  max_tokens_across_dp = max(packed_tensor[:-1])
  global_with_prefill  = bool(packed_tensor[-1])
  num_tokens_after_padding = [max_tokens_across_dp] * dp_size
```

> **为什么用 CPU group**：CPU 侧的通信可以与 NPU 上的其他计算重叠，避免不必要的同步开销。

#### 2b. `_sync_batch_across_dp()`：同步 cudagraph 模式与 padded token 数

```
tensor[0, dp_rank] = num_tokens_padded
tensor[1, dp_rank] = cudagraph_mode.value

dist.all_reduce(tensor, group=get_dp_group().cpu_group)
   ↑ CPU 侧 All-Reduce

结果：
  num_tokens_across_dp = tensor[0, :]
  max_num_tokens       = max(num_tokens_across_dp)
  synced_cudagraph_mode = post_process(tensor[1, :])  ← 取各 rank 的最小模式
```

`_determine_batch_execution_and_padding()` 整合了上述逻辑，流程如下：

```
_determine_batch_execution_and_padding()
│
├── _pad_for_sequence_parallelism()     ← TP/SP padding
├── dispatch_cudagraph()                ← 选择 cudagraph 模式
│
└── if dp_size > 1:
    ├── _sync_batch_across_dp()         ← CPU All-Reduce（同步 padded tokens 和 cudagraph 模式）
    └── 用 max 值重新 dispatch_cudagraph()  ← 确保所有 rank 用相同的 padded token 数
```

#### 2c. `max_tokens_across_dp` 传递到 Forward Context

同步后的 `max_tokens_across_dp` 被写入 `set_ascend_forward_context()` / `set_additional_forward_context()` 的 forward context，后续 MoE 层的 prepare/finalize 会从中读取用于 padding：

```python
# ascend_forward_context.py
forward_context.max_tokens_across_dp = max_tokens_across_dp
# 用于 MC2 mask 的 padded token 数
forward_context.padded_num_tokens = ceil(max_tokens_across_dp / tp_size) * tp_size
```

---

### 3. MoE 前向传播中的 DP 通信

**文件**：`vllm_ascend/ops/fused_moe/prepare_finalize.py`、`vllm_ascend/ops/fused_moe/moe_comm_method.py`、`vllm_ascend/ops/fused_moe/token_dispatcher.py`

MoE 前向传播在进入专家计算前后，分别有 `prepare()` 和 `finalize()` 两个阶段。根据硬件代际、token 数量等条件，系统会选择不同的通信方式（由 `select_moe_comm_method()` 决定）。

#### 通信方式选择逻辑（`select_moe_comm_method()`）

```
if NOT MoE model:
    return None

if NOT enable_expert_parallel OR ep_size == 1:
    → ALLGATHER

elif A2 (Ascend 910B):
    if num_tokens ≤ mc2_capacity AND world_size_across_dp ≥ 16:
        → MC2  # 16 为 A2 硬件上 MC2 生效的最小规模阈值（需足够多的 EP rank 分摊通信开销）
    else:
        → ALLGATHER

elif A3 (Ascend 910C):
    if num_tokens ≤ mc2_capacity:
        → FUSED_MC2 (quant w8a8_dynamic) 或 MC2
    else:
        → FUSED_MC2 (prefill quant) 或 ALLTOALL

elif 310P:
    → ALLGATHER
```

#### 方式一：AllGather（`PrepareAndFinalizeWithAllGather`）

适用于不启用 EP 或 AllGather 场景，整体数据流为：

```
Attn → TP All-Reduce → DP All-Gather → MoE → DP Reduce-Scatter → TP All-Reduce

Prepare:
  if dp_size > 1:
    pad hidden_states to max_tokens_across_dp
    hidden_states = dp_group.all_gather(hidden_states, dim=0)    ← NPU All-Gather
    router_logits = dp_group.all_gather(router_logits, dim=0)    ← NPU All-Gather

  if pcp_size > 1:
    pad to max_tokens_across_pcp
    hidden_states = pcp_group.all_gather(hidden_states, dim=0)   ← NPU All-Gather (PCP)
    router_logits = pcp_group.all_gather(router_logits, dim=0)   ← NPU All-Gather (PCP)

Finalize:
  if dp_size > 1 and not shared_expert_dp:
    hidden_states = dp_group.reduce_scatter(hidden_states, dim=0) ← NPU Reduce-Scatter
    hidden_states = hidden_states[:num_tokens]                     ← 截断 padding

  if pcp_size > 1:
    hidden_states = pcp_group.reduce_scatter(hidden_states, dim=0) ← NPU Reduce-Scatter (PCP)
```

启用序列并行（SP）时，TP All-Gather + DP All-Gather 被合并为 EP All-Gather，TP Reduce-Scatter + DP Reduce-Scatter 合并为 EP Reduce-Scatter，进一步优化通信：

```
TP All-Gather → Attn → TP Reduce-Scatter → EP All-Gather → MoE → EP Reduce-Scatter
```

#### 方式二：All2All（`PrepareAndFinalizeWithAll2All`）

适用于 EP 启用、token 数超过 MC2 容量的大 batch 场景：

```
Prepare:
  pad hidden_states/router_logits 到 TP 的整数倍（本地操作）
  if tp_size > 1:
    split hidden_states → 取本 TP rank 的切片（本地操作，无 NPU 通信）

Finalize（token combine 后）:
  if tp_size > 1:
    dist.all_gather(split_hidden_states, hidden_states,
                    group=tp_group.device_group)                  ← NPU All-Gather (TP)
  hidden_states = hidden_states[:num_tokens]                     ← 截断
```

> All2All 中的 DP 通信被 EP 内部的 All-to-All（在 `TokenDispatcherWithAll2AllV` 中完成）吸收，不再显式地做 DP All-Gather/Reduce-Scatter。

#### 方式三：MC2（`PrepareAndFinalizeWithMC2` + `TokenDispatcherWithMC2`）

MC2 是 Ascend 设备上专为 MoE 通信优化的方法，将 DP+EP 的 token dispatch/combine 合并为一个 NPU 原语。

**通信组**：`mc2_group` 覆盖 `ExternalDP × EP` 所有 rank（即 `DP * PCP * TP` 行内的 rank）。

```
Prepare（PrepareAndFinalizeWithMC2）:
  mc2_mask = forward_context.mc2_mask[: padded_num_tokens]
  if tp_size > 1:
    mc2_mask   = mc2_mask[tp_rank_slice]                         ← 本地切片
    hidden_states = hidden_states[tp_rank_slice]                 ← 本地切片
    router_logits = router_logits[tp_rank_slice]                 ← 本地切片

TokenDispatch（npu_moe_distribute_dispatch）:
  torch_npu.npu_moe_distribute_dispatch(
      x=hidden_states,
      expert_ids=topk_ids,
      global_bs=ep_world_size × max_tokens_per_tp_rank,
      group_ep=mc2_group_name,
      ep_world_size=ep_world_size,
      ep_rank_id=ep_rank_id,
      ...
  )
  → 输出: expand_x, expert_token_nums, ep_recv_counts, assist_info_for_combine
  → 本质：All-to-All within MC2 group（DP+EP 联合通信，通信与计算重叠）

  [Expert 计算（GMM）]

TokenCombine（npu_moe_distribute_combine）:
  torch_npu.npu_moe_distribute_combine(
      expand_x=hidden_states,
      ep_send_counts=ep_recv_counts,
      group_ep=mc2_group_name,
      ep_world_size=ep_world_size,
      ep_rank_id=ep_rank_id,
      ...
  )
  → 本质：All-to-All within MC2 group（逆向合并）

Finalize（PrepareAndFinalizeWithAll2All，与 MC2 共用）:
  if tp_size > 1:
    dist.all_gather(gathered_states, hidden_states,
                    group=tp_group.device_group)                  ← NPU All-Gather (TP)
  hidden_states = hidden_states[:num_tokens]
```

**MC2 与普通 AllGather 的核心差异**：
- AllGather 方式将 DP 通信（All-Gather/Reduce-Scatter）与 Expert 计算串行执行
- MC2 方式通过 `npu_moe_distribute_dispatch/combine` 将 DP+EP 的 token 路由通信与 Expert 计算流水并行（comm-compute overlap），在大 batch 场景下显著提升吞吐量

#### Fused MC2（`FusedMC2CommImpl`）

在 A3 设备 + w8a8_dynamic 量化 + EP size ≤ 32 时启用。在 MC2 基础上将 dispatch_ffn_combine 进一步融合，进一步减少通信延迟。

---

## DP 通信汇总

| 通信操作 | 通信组 | 方向 | 触发时机 | 说明 |
|---------|-------|------|---------|------|
| `dist.all_reduce` (CPU) | `dp_group.cpu_group` | 所有 DP rank | Batch 元数据同步（`_sync_metadata_across_dp` / `_sync_batch_across_dp`） | 同步 num_tokens、with_prefill、cudagraph_mode |
| `dist.all_reduce` (CPU) | `dp_group.cpu_group` | 所有 DP rank | BalanceDPEngineCoreProc `_has_global_unfinished_reqs` | 判断是否还有全局未完成请求 |
| `dist.all_gather` (CPU) | `dp_group` | 所有 DP rank | `BalanceScheduler.balance_gather` | 负载均衡：汇总各 rank 的 running 队列长度 |
| `all_gather` (NPU) | `dp_group.device_group` | 所有 DP rank | MoE Prepare，AllGather 方式 | 聚合各 DP rank 的 hidden_states |
| `reduce_scatter` (NPU) | `dp_group.device_group` | 所有 DP rank | MoE Finalize，AllGather 方式 | 散回各 DP rank 的 hidden_states |
| `npu_moe_distribute_dispatch` | `mc2_group`（DP*EP） | 所有 EP rank | MoE token dispatch，MC2/FusedMC2 方式 | Ascend 专用 All-to-All，comm-compute 重叠 |
| `npu_moe_distribute_combine` | `mc2_group`（DP*EP） | 所有 EP rank | MoE token combine，MC2/FusedMC2 方式 | Ascend 专用 All-to-All，逆向合并 |
| `dist.all_gather` (NPU) | `tp_group.device_group` | 同 TP 组 | MoE Finalize，All2All/MC2 方式 | 重建完整的 hidden_states（TP 维度） |
| `all_gather` (NPU) | `pcp_group` | PCP 组 | MoE Prepare，PCP 场景 | 序列并行时聚合各 PCP rank 的 tokens |
| `reduce_scatter` (NPU) | `pcp_group` | PCP 组 | MoE Finalize，PCP 场景 | 序列并行时散回各 PCP rank |

---

## DP 通信跳过条件

`_skip_all_reduce_across_dp_group()` 控制是否跳过 DP 级别的元数据同步 All-Reduce：

```python
def _skip_all_reduce_across_dp_group(self, is_draft_model=False) -> bool:
    # 1. 非 MoE 模型（Dense 模型）：无需 DP 同步，直接跳过
    if not is_moe_model:
        return True

    # 2. 非 KV Consumer：不跳过（需要正常同步）
    if not self.is_kv_consumer:
        return False

    # 3. MoE 模型 + KV Consumer：
    #    当 decode 必须用 MC2（因为 MC2 内部已包含 DP 通信）
    #    且（prefill 也必须用 MC2，或启用了 recompute_scheduler），才跳过
    decode_must_use_mc2 = needs_mc2(max_decode_tokens)
    prefill_must_use_mc2 = needs_mc2(max_prefill_tokens)
    return decode_must_use_mc2 and (prefill_must_use_mc2 or recompute_scheduler_enable)
```

**跳过的含义**：当所有场景下都使用 MC2 时，MC2 的 `npu_moe_distribute_dispatch/combine` 已经在内部完成了跨 DP 的 token 路由，不再需要额外的 batch 元数据 All-Reduce（只需要让 `num_tokens_after_padding` 等于本 rank 自己的 token 数即可）。

---

## 关键源文件索引

| 文件 | 作用 |
|------|------|
| `vllm_ascend/distributed/parallel_state.py` | 初始化 MC2、DP 等各类通信组 |
| `vllm_ascend/worker/model_runner_v1.py` | `_sync_metadata_across_dp`、`_sync_batch_across_dp`、`_dummy_run` |
| `vllm_ascend/ascend_forward_context.py` | `set_ascend_forward_context`、`select_moe_comm_method`、`max_tokens_across_dp` |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py` | AllGather/All2All/MC2 的 prepare/finalize 实现 |
| `vllm_ascend/ops/fused_moe/token_dispatcher.py` | MC2/AllGather/All2AllV token dispatch/combine |
| `vllm_ascend/ops/fused_moe/moe_comm_method.py` | `AllGatherCommImpl`、`MC2CommImpl`、`AlltoAllCommImpl`、`FusedMC2CommImpl` |
| `vllm_ascend/patch/platform/patch_balance_schedule.py` | `BalanceDPEngineCoreProc`、`BalanceScheduler` |
