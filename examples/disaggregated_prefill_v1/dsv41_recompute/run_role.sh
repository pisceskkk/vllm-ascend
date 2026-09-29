#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -Eeuo pipefail

case "${1:-}" in
  prefill) role=prefill; kv_role=kv_producer; recompute=false; default_port=18121; engine_id=0 ;;
  decode) role=decode; kv_role=kv_consumer; recompute=true; default_port=18123; engine_id=1 ;;
  *) echo "Usage: MODEL_PATH=... ROLE_HOST=... COMM_IFNAME=... VLLM_ASCEND_ROOT=... DSV41_LEGACY_OPP=... bash $0 {prefill|decode}" >&2; exit 2 ;;
esac

: "${MODEL_PATH:?Set MODEL_PATH to the DeepSeek-V4.1-Flash checkpoint}"
: "${ROLE_HOST:?Set ROLE_HOST to the communication IP address of this node}"
: "${COMM_IFNAME:?Set COMM_IFNAME to the communication network interface}"
: "${VLLM_ASCEND_ROOT:?Set VLLM_ASCEND_ROOT to the installed vllm-ascend checkout}"
: "${DSV41_LEGACY_OPP:?Set DSV41_LEGACY_OPP to the validated legacy custom_transformer vendor directory}"
CANN_HOME="${CANN_HOME:-/usr/local/Ascend/ascend-toolkit}"
CUSTOMIZE_OPP="${CUSTOMIZE_OPP:-${CANN_HOME}/latest/opp/vendors/customize}"
WORK_DIR="${WORK_DIR:-${PWD}/dsv41-recompute-pd}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/logs}"
CACHE_DIR="${CACHE_DIR:-${WORK_DIR}/cache/${role}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dsv41-recompute-audit}"
VLLM_BIN="${VLLM_BIN:-vllm}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-${default_port}}"
KV_PORT="${KV_PORT:-33100}"

export VLLM_VERSION=0.30.0
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_HOST_IP="${ROLE_HOST}"
export HCCL_IF_IP="${ROLE_HOST}"
export GLOO_SOCKET_IFNAME="${COMM_IFNAME}"
export TP_SOCKET_IFNAME="${COMM_IFNAME}"
export HCCL_SOCKET_IFNAME="${COMM_IFNAME}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_BUFFSIZE=1600
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
unset HCCL_DETERMINISTIC VLLM_COMPUTE_NANS_IN_LOGITS VLLM_RAISE_ON_LOGIT_NANS

set +u
# shellcheck disable=SC1091
source "${CANN_HOME}/set_env.sh"
set -u
unset ATB_HOME_PATH ASDOPS_HOME_PATH ATB_SPEED_HOME_PATH ASCEND_LAUNCH_BLOCKING
export PYTHONPATH="${VLLM_ASCEND_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Preserve the three-vendor loader order of the validated PD environment.
export VLLM_ASCEND_BUNDLED_OPP="${VLLM_ASCEND_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export ASCEND_CUSTOM_OPP_PATH="${CUSTOMIZE_OPP}:${VLLM_ASCEND_BUNDLED_OPP}:${DSV41_LEGACY_OPP}"
export LD_LIBRARY_PATH="${DSV41_LEGACY_OPP}/op_api/lib:${CUSTOMIZE_OPP}/op_api/lib:${VLLM_ASCEND_BUNDLED_OPP}/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_PRELOAD="${CUSTOMIZE_OPP}/op_api/lib/libcust_opapi.so:${VLLM_ASCEND_BUNDLED_OPP}/op_api/lib/libcust_opapi.so:${DSV41_LEGACY_OPP}/op_api/lib/libcust_opapi.so:${CANN_HOME}/latest/lib64/libopapi_nn.so"
export DSV41_A5_DSL_SOURCE=arena
export CANNBOTDSL_NATIVE_BINARY_MODE=prefer

mkdir -p "${LOG_DIR}" "${CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}"
exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" --host "${API_HOST}" --port "${API_PORT}" \
  --api-server-count 1 --data-parallel-size 8 --tensor-parallel-size 1 \
  --enable-expert-parallel --enable-ep-weight-filter --seed 1024 \
  --max-model-len 1048576 --max-num-batched-tokens 1024 --max-num-seqs 16 \
  --block-size 128 --no-enable-prefix-caching --limit-mm-per-prompt '{"image":0}' \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --safetensors-load-strategy lazy --trust-remote-code \
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --gpu-memory-utilization 0.90 --quantization deepseek_v4_fp8 --async-scheduling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --engram-config '{"cpu_offload":false}' \
  --kv-transfer-config "{\"kv_connector\":\"MooncakeHybridConnector\",\"kv_role\":\"${kv_role}\",\"kv_port\":${KV_PORT},\"engine_id\":\"${engine_id}\",\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":8,\"tp_size\":1},\"decode\":{\"dp_size\":8,\"tp_size\":1}}}" \
  --additional-config "{\"enable_cpu_binding\":true,\"multistream_overlap_shared_expert\":true,\"multistream_dsv4_dsa_overlap\":false,\"enable_fused_mc2\":0,\"scheduler_config\":{\"recompute_scheduler_enable\":${recompute}}}" \
  >"${LOG_DIR}/${role}.log" 2>&1
