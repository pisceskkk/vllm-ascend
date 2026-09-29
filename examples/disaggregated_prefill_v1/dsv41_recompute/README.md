# DeepSeek V4.1 PD with decode recompute

This example starts one prefill node and one decode node, each with eight Ascend 950DT devices. It uses vLLM 0.30.0, the V1 runner and the V4.1 fixed Engram DP-slot change. Each node runs DP8/TP1 with expert parallelism; no external DP coordinator or manual rank loop is needed.

The launch configuration keeps async scheduling, five DSpark draft tokens with an eager drafter, target `FULL_DECODE_ONLY`, Engram in HBM, a 1M context limit, 1024 batched tokens, 16 sequences and disabled prefix caching. Decode recompute is enabled only on the decode node. Both roles use `enable_fused_mc2=0`.

## Runtime prerequisites

This complete-package example requires [PR #3](https://github.com/linfeng-yuan/vllm-ascend/pull/3), [PR #4](https://github.com/linfeng-yuan/vllm-ascend/pull/4) and [PR #5](https://github.com/linfeng-yuan/vllm-ascend/pull/5) applied together to `0929_950DT_vllm0300`. PR #3 supplies the async metadata optimization; PR #4 supplies the in-tree V4.1 operators and their native call paths; PR #5 supplies fixed Engram DP slots. PR #5 retains that target branch as its base: checking out PR #5 alone does not include the operator migration from PR #4.

Use the same compatible vLLM, torch-npu and CANN installation on both nodes, with the combined source and complete 31-operator package plus extension:

| Variable | Required location |
| --- | --- |
| `VLLM_ASCEND_ROOT` | Installed vllm-ascend checkout, including `vllm_ascend/vllm_ascend_C*.so` and `vllm_ascend/_cann_ops_custom/vendors/custom_transformer` |
| `CANN_HOME` | Toolkit root containing `set_env.sh`; defaults to `/usr/local/Ascend/ascend-toolkit` |
| `VLLM_BIN` | vLLM executable from the prepared Python environment; defaults to `vllm` on `PATH` |

Use the same complete operator artifacts on both nodes. The script clears inherited operator paths and `LD_PRELOAD`, then loads the toolkit environment and selects only this checkout's bundled vendor in `ASCEND_CUSTOM_OPP_PATH`. The custom API library is resolved through the normal local `dlopen` path; do not preload custom or toolkit operator libraries. PythonDSL operators come from the checkout, without the legacy arena selector. The selected source is prepended to `PYTHONPATH` while retaining the toolkit's pyACL paths.

Reuse validated artifacts built for this native source, hardware and CANN version when they are already available; the Python-only fixed-slot change does not require recompilation. Otherwise, prepare the repository's build dependencies and build the complete package through the normal entry point, with at least 256-way build parallelism on the Ascend 950DT build host:

```bash
cd "$VLLM_ASCEND_ROOT"
source "${CANN_HOME:-/usr/local/Ascend/ascend-toolkit}/set_env.sh"
unset LD_PRELOAD ASCEND_CUSTOM_OPP_PATH
export MAX_JOBS=256 CMAKE_BUILD_PARALLEL_LEVEL=256 COMPILE_CUSTOM_KERNELS=1
export SOC_VERSION=ascend950dt_9572
export PYTHONPATH="$PWD${PYTHONPATH:+:${PYTHONPATH}}"
python setup.py build_ext --inplace
```

Both nodes need the complete checkpoint, eight visible devices and network access to each other's HTTP and Mooncake/HCCL communication ports. `ROLE_HOST` must be reachable from the other node and correspond to `COMM_IFNAME`. Run each command inside its prepared container or environment. No credentials, container setup or operator build is included.

## Launch

On the prefill node, replace the example paths and network values:

```bash
export MODEL_PATH=/models/DeepSeek-V4.1-Flash
export VLLM_ASCEND_ROOT=/workspace/vllm-ascend
export ROLE_HOST=192.0.2.10
export COMM_IFNAME=eth0
export WORK_DIR=/var/tmp/dsv41-pd
API_PORT=18121 KV_PORT=33100 bash run_role.sh prefill
```

On the decode node, set the same model/operator variables with that node's paths, then:

```bash
export ROLE_HOST=192.0.2.11
export COMM_IFNAME=eth0
export WORK_DIR=/var/tmp/dsv41-pd
API_PORT=18123 KV_PORT=33100 bash run_role.sh decode
```

The scripts run in the foreground and write `$WORK_DIR/logs/prefill.log` or `decode.log`. Override `LOG_DIR` and `CACHE_DIR` to choose independent log and compilation-cache locations. `API_HOST` defaults to `0.0.0.0`; `SERVED_MODEL_NAME` defaults to `dsv41-recompute-audit`. Each role uses its own engine ID and the same DP8/TP1 connector topology. HCCL deterministic mode and launch blocking are unset, matching the PD validation. No profiler or diagnostic code is enabled.

Wait for both services to become ready before sending requests:

```bash
curl --fail http://192.0.2.10:18121/health
curl --fail http://192.0.2.11:18123/health
```

## Two-leg smoke test

The client sends the prompt to the producer with `max_tokens=1`, `min_tokens=1` and `do_remote_decode=true`. It then sends the same prompt to the consumer with the producer's complete `kv_transfer_params`, including every KV group, unchanged. Both legs share `X-Request-Id`. No generic `remote_bootstrap_addr` protocol or load-balancing proxy is used.

Run a short serial check, then a concurrent mix of short and long responses:

```bash
python smoke_pd.py \
  --prefill http://192.0.2.10:18121 --decode http://192.0.2.11:18123 \
  --requests 8 --concurrency 1 --suite short --max-tokens 16 \
  --output results/serial

python smoke_pd.py \
  --prefill http://192.0.2.10:18121 --decode http://192.0.2.11:18123 \
  --requests 64 --concurrency 64 --suite mixed --max-tokens 64 192 16 16 \
  --output results/mixed64
```

`--max-tokens` accepts one or several budgets, cycled across requests. `--suite burst --max-tokens 192` makes every request count from 1 to 50. Logprobs and raw token IDs are requested by default; `--no-logprobs` disables logprobs. Set `--model` if the launch model name was changed. Use a new output directory for every run.

Each request directory immediately saves its producer and consumer response bytes, including HTTP error bodies, plus `record.json` with payloads, HTTP status, text, token IDs, usage and errors. `summary.json` lists failures and responses stopped by the length limit. The process exits nonzero for an HTTP/transport failure, missing transferable KV, empty text/tokens, all-zero output token IDs, non-finite selected-token logprobs or a `recomputed` stop. A `length` finish is explicitly reported and does not prove a complete semantic answer. Review the saved text for accuracy; this is a smoke test, not an accuracy benchmark.

The client does not retry or silently restart a preempted request. A production PD proxy must handle `stop_reason=recomputed` by reissuing prefill; that recovery is outside this small reproducer.
