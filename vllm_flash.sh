export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10  
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export VLLM_USE_V1=1
export VLLM_VERSION=0.20.1
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
#export HCCL_DETERMINISTIC=true
# vllm serve /mnt/share/z00951268/results/dsv4-pro/DeepSeek-V4-Pro-w4a8-fixmtp \
vllm serve /mnt/share/z00919641/v4_w8a8 \
    --enable-prefix-caching \
    --max_model_len 16384 \
    --max-num-batched-tokens 16384 \
    --served-model-name dsv4 \
    --gpu-memory-utilization 0.95 \
    --api-server-count 1 \
    --max-num-seqs 256 \
    --data-parallel-size 1 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --safetensors-load-strategy 'prefetch' \
    --quantization ascend \
    --port 8091 \
    --block-size 128 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'\
    --async-scheduling \
    --enforce-eager \
    --additional-config '
    {"ascend_compilation_config":{
        "enable_npugraph_ex":true,
        "enable_static_kernel":false
        },
    "enable_cpu_binding": "true",
    "multistream_overlap_shared_expert":false,
    "multistream_dsa_preprocess":false}' \
    2>&1 | tee flash.log
    # --speculative-config '{"num_speculative_tokens": 3,"method": "deepseek_mtp", "enforce_eager": "true"}' \
