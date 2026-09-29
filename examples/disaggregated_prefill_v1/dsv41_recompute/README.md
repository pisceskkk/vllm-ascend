# DeepSeek V4.1 PD with decode recompute

This example starts one prefill node and one decode node, each with eight Ascend 950DT devices. It uses vLLM 0.30.0, the V1 runner and the V4.1 fixed Engram DP-slot change. Each node runs DP8/TP1 with expert parallelism; no external DP coordinator or manual rank loop is needed.

The launch configuration keeps async scheduling, five DSpark draft tokens with an eager drafter, target `FULL_DECODE_ONLY`, Engram in HBM, a 1M context limit, 1024 batched tokens, 16 sequences and disabled prefix caching. Decode recompute is enabled only on the decode node. Both roles use `enable_fused_mc2=0`.

## Runtime prerequisites

Use the same compatible vLLM, torch-npu, CANN and V4.1 operator installation on both nodes, with this source installed. The launch script reproduces the validated PD environment's **three-vendor operator stack**:

| Variable | Required location |
| --- | --- |
| `VLLM_ASCEND_ROOT` | Installed vllm-ascend checkout, including `vllm_ascend/_cann_ops_custom/vendors/custom_transformer` |
| `CUSTOMIZE_OPP` | CANN `customize` vendor; defaults to `$CANN_HOME/latest/opp/vendors/customize` |
| `DSV41_LEGACY_OPP` | Compatible legacy `custom_transformer` vendor |
| `CANN_HOME` | Toolkit root containing `set_env.sh`; defaults to `/usr/local/Ascend/ascend-toolkit` |
| `VLLM_BIN` | vLLM executable from the prepared Python environment; defaults to `vllm` on `PATH` |

All three vendors must already be installed. The script preserves their `ASCEND_CUSTOM_OPP_PATH`, library search and `LD_PRELOAD` ordering, and selects the installed arena PythonDSL implementation. These loader settings apply to this operator stack; they are not the loader recipe for the separately built, complete 31-operator package. The script prepends the selected source checkout to `PYTHONPATH` while retaining the toolkit's pyACL paths.

Both nodes need the complete checkpoint, eight visible devices and network access to each other's HTTP and Mooncake/HCCL communication ports. `ROLE_HOST` must be reachable from the other node and correspond to `COMM_IFNAME`. Run each command inside its prepared container or environment. No credentials, container setup or operator build is included.

## Launch

On the prefill node, replace the example paths and network values:

```bash
export MODEL_PATH=/models/DeepSeek-V4.1-Flash
export VLLM_ASCEND_ROOT=/workspace/vllm-ascend
export DSV41_LEGACY_OPP=/opt/dsv41/legacy_opp/vendors/custom_transformer
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
