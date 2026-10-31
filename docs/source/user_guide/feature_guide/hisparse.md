# HiSparse on Ascend

## Overview

HiSparse keeps sparse MLA KV pages in host memory and uses NPU memory for resident pages and a per-request hot cache. During decode, the sparse indexer selects logical tokens. The NPU resolves those tokens to resident or hot-cache rows and reads missing rows from mapped host memory. This allows the main KV history to exceed the resident device cache.

This experimental implementation requires the matching vLLM platform-backend changes. It has no released version yet. The development environment uses Ascend 950 and 910_93, CANN 9.2, torch-npu 2.10, and Triton-Ascend 3.6 development builds.

```text
Sparse indexer -> logical top-k tokens -> NPU residency resolver
                                             |
                               resident row / hot-cache hit / host miss
                                             |
                              mapped-host gather for missing KV rows
                                             |
                              pack latent and RoPE rows -> sparse attention
```

The resolver protects the union of tokens selected by all MTP query rows before choosing eviction slots. It uses device sorting and Triton-Ascend kernels with persistent scratch buffers. Decode does not copy top-k indices to the CPU for transfer planning.

The adapter reuses upstream scheduling, prefix-cache ownership, page spills, and host-mirror completion. Host, resident, and hot caches use separate 512-dimensional latent and 64-dimensional RoPE planes. Both planes share one logical page allocator and the original KV memory budget. Runtime layer views retain shared-slab gaps; attention views are contiguous and use a common scaled page table. Indexer pages retain their own contiguous format within the latent plane, with initialization checks for slab capacity and layer offsets.

KV preprocessing uses native fused RMSNorm/RoPE to write the two destination planes directly. Slot translation masks invalid slots and inactive rows. Resident mirrors, page spills, restores, and host copies transfer both components in one batched DMA submission. This avoids a preprocessing scratch cache and a separate payload scatter.

When the coordinator confirms that all context pages in the batch remain resident, prefill and mixed decode rows read the resident planes directly, skipping prefill staging. Other prefill batches gather resident and host rows into two contiguous staging tensors. A device kernel filters selections to the causal context, stable-compacts valid indices, and builds per-query page tables and lengths for one SFA call. Table width follows the current context upper bound. Empty and graph-padding query rows produce zero output.

Decode retains the resident/hot-cache resolver and a bounded selected-row workspace. Both paths preserve valid selection order and duplicates. SFA uses its existing 512/64 input interface on Ascend 950 and 910_93; this adapter requires no changes to its tiling or device kernels. Staging and device copying remain execution costs; measure throughput and latency for the intended workload.

## Requirements and limits

| Area | Requirement |
| --- | --- |
| Hardware | Ascend 950 or 910_93 with AIV access to registered host memory; runtime capability is checked |
| Runner | Model Runner V2 |
| Model path | Sparse MLA/DSA with a sparse-index group |
| KV formats | BF16 indexer KV and unquantized main KV; model-weight quantization is independent |
| Context parallelism | Prefill and decode context-parallel sizes must both be 1 |
| MLA dimensions | 512 latent and 64 RoPE |
| Sparse selection | SFA requires top-k divisible by 128 |
| Hot cache | Representation supports at most 32768 aligned rows; NPU tests cover capacities through 10240. Capacity must cover the complete query-row union |
| Host allocation | Private pool per worker rank; account for the sum across TP ranks |
| Native extension | Build the Ascend extension for page DMA and sparse attention |

Mapped-host allocations remain registered until work using their addresses has completed. Host pools must remain alive across graph replay. Initialization fails if the hardware lacks the required mapping capability.

On Ascend 910_93, SFA receives the full causal context length and compacts valid selections before attention. Eager operator checks cover resident and staged prefill, mixed decode, host misses and LRU state. Graph capture remains unvalidated on this platform; use `--enforce-eager` for the current A3 development path.

## Configuration

Set `VLLM_USE_V2_MODEL_RUNNER=1`, then add these options to a compatible model's launch command:

```bash
--block-size 128 \
--kv-cache-dtype bfloat16 \
--attention-config '{"indexer_kv_dtype":"bf16","hisparse_config":{"eager_host_mirror":true}}' \
--kv-transfer-config '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":4}}'
```

The example requests 4 GiB of host pool per rank, or 32 GiB across TP8. Size the pool for the model's layers and intended context capacity. Select the model's usual weight-quantization option separately.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `kv_connector_extra_config.host_pool_gib` | Required, positive | Host pool capacity in GiB per rank |
| `attention_config.indexer_kv_dtype` | Model dependent | Set `bf16` for this implementation |
| `attention_config.hisparse_config.device_buffer_size` | Derived from top-k and query-row count | Per-request hot-cache capacity, rounded to the cache block size |
| `attention_config.hisparse_config.eager_host_mirror` | `true` | Mirror decode KV writes during forward; prefill writes are always mirrored |

Omit `device_buffer_size` initially so the upstream allocator accounts for the number of MTP query rows. Changing MTP length changes the required union capacity.

## Operator validation

From the source checkout, with the Ascend environment and project virtual environment active:

```bash
cd /path/to/vllm-ascend
ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest \
  --confcutdir=tests/e2e/pull_request/one_card \
  tests/e2e/pull_request/one_card/test_hisparse_ops.py \
  tests/e2e/pull_request/one_card/test_hisparse_attention.py \
  tests/e2e/pull_request/one_card/test_hisparse_paged_prefill.py \
  tests/e2e/pull_request/one_card/test_hisparse_prefill_staging.py \
  tests/e2e/pull_request/one_card/test_hisparse_selection_order.py \
  tests/e2e/pull_request/one_card/test_hisparse_kv_preprocess.py \
  tests/e2e/pull_request/one_card/test_hisparse_split.py -v
```

These tests compare state transitions and KV payloads with a CPU reference. They cover resident hits, hot-cache hits, host misses, reordered request state, padded rows, physical cache strides, compact gathers, invalidation after KV writes, and ACL graph replay. Large cases include top-k 2048 and four query rows with an 8192-row union. The attention tests independently compute QK, masked softmax, and weighted values in FP32 on CPU, including duplicate selections and empty rows. `--confcutdir` keeps these tests independent of the model-download fixtures. Passing operator tests alone does not establish full-model accuracy or performance.

The prefill staging tests check resident and host rows across shared-layer cache views and different resident page sizes. The selection-order tests check interleaved padding, duplicates, empty rows, and changed selections during graph replay.

The KV preprocessing tests include a fixed BF16 activation row that exposes a rounding difference between standalone and fused RMSNorm. They require native fused results to match exactly, including split planes in shared backing, page-boundary writes, invalid slots, padding, and graph replay with changed inputs.

## Platform backend dependency

The platform-backend interface from vLLM PR
[#62592](https://github.com/vllm-project/vllm/pull/62592) is merged upstream.
This integration requires the component-cache backend hooks in vLLM `611502d66a`
and Ascend main `e8a6393dda93e085cd8210011c6ec12027acb8a0`.
Pin the paired revisions and integration changes when reproducing validation.

`NPUPlatform.get_hisparse_backend_cls()` selects `AscendHiSparseBackend`.
The common runtime owns scheduling, TP coordination, allocation rollback and
synchronization before Host release. The plugin owns NPU operations and view
adaptation through `validate_kv_cache_config()` and `prepare_kv_cache_views()`.
Indexer views are prepared before common handle binding, inside its rollback scope.

`gather_resident_rows()` copies complete payloads while preserving page strides
and layer offsets. FP8-typed payloads use `uint8` views before indexing, avoiding
unsupported NPU FP8 indexing without numerical conversion. This primitive does
not add end-to-end attention C8 support to this BF16 feature PR. The CUDA `fp8_ds_mla`
prefill conversion path is explicitly rejected rather than reinterpreting its
layout as Ascend storage.

Host pools remain private per worker rank. Local event creation accepts
`interprocess=False`; NPU IPC event creation/export/import reject calls until
that protocol is implemented and validated. Existing context-parallel and model
restrictions remain.

Resolver scratch is shared by the layers of one sparse-index group and is
released when that group and its cache operations are released. Independent
groups use independent workspaces. The common runtime owns streams and events.

The paired upstream version does not expose speculative group prefetch, so the
backend does not opt into that capability. Tentative KV writes retain the
upstream host-mirror behavior. Operator coverage of multiple query rows alone
does not establish end-to-end MTP accuracy. Full-model accuracy and performance
must be evaluated separately for the intended checkpoint and workload.
