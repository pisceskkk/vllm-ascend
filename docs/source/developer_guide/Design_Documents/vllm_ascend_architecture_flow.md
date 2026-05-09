# vLLM + vLLM-Ascend 服务启动与推理全流程架构分析

本文档基于 vLLM 和 vLLM-Ascend 源码，梳理 vLLM 服务启动、请求调度、推理准备、模型推理、后端处理的全流程结构，帮助初学者熟悉 vLLM 整体框架及 vLLM-Ascend 插件化能力。

---

## 一、整体架构概览

vLLM V1 采用**多进程架构**，核心进程包括：

| 进程 | 数量 | 职责 |
|------|------|------|
| **API Server** | 1（或 dp_size 个） | HTTP 请求处理、Tokenization、结果流式返回 |
| **Engine Core** | 1（或 dp_size 个） | 调度、KV Cache 管理、协调 Worker |
| **GPU/NPU Worker** | tp_size × pp_size 个 | 模型加载、前向推理 |
| **DP Coordinator** | 1（dp>1 时） | 数据并行协调 |

进程间通过 **ZMQ Socket** 通信。

以 `vllm serve -tp=2 -dp=4`（8 卡）为例，进程拓扑为：

- 4 个 API Server + 4 个 Engine Core + 8 个 GPU Worker + 1 个 DP Coordinator = **17 个进程**

---

## 二、全流程结构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           vLLM 服务启动 & 推理全流程                               │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 阶段 1: 服务启动 (Service Startup)                                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  CLI 入口: vllm serve <model>                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ vllm/entrypoints/cli/main.py                                             │   │
│  │   └─> vllm/entrypoints/cli/serve.py::ServeSubcommand.cmd()               │   │
│  │         └─> run_headless() / 主流程                                       │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 1.1 参数解析 & 配置构建                                                    │   │
│  │   AsyncEngineArgs.from_cli_args(args)                                     │   │
│  │     └─> create_engine_config() → VllmConfig                               │   │
│  │         ├─ ModelConfig (模型路径、dtype、max_model_len...)                  │   │
│  │         ├─ CacheConfig (block_size、KV cache 配置)                         │   │
│  │         ├─ ParallelConfig (tp_size、pp_size、dp_size...)                   │   │
│  │         ├─ SchedulerConfig (max_num_seqs、max_num_batched_tokens...)       │   │
│  │         ├─ DeviceConfig (设备类型)                                         │   │
│  │         └─ CompilationConfig (cudagraph/npugraph 配置)                     │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 1.2 平台插件发现 & 加载 (Platform Plugin Discovery)                        │   │
│  │                                                                           │   │
│  │   vllm/platforms/__init__.py::resolve_current_platform_cls_qualname()     │   │
│  │     │                                                                     │   │
│  │     ├─ 扫描内置平台: cuda / rocm / xpu / cpu / tpu                        │   │
│  │     │                                                                     │   │
│  │     └─ 扫描 entry_points "vllm.platform_plugins"                          │   │
│  │           │                                                               │   │
│  │           ▼  ★ vllm-ascend 插件入口 ★                                     │   │
│  │           vllm_ascend/__init__.py::register()                             │   │
│  │             └─> return "vllm_ascend.platform.NPUPlatform"                 │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 1.3 NPUPlatform 初始化 (vllm-ascend 核心)                                  │   │
│  │                                                                           │   │
│  │   vllm_ascend/platform.py::NPUPlatform                                   │   │
│  │     │                                                                     │   │
│  │     ├─ pre_register_and_update()  ★ 全局 Patch 注入 ★                     │   │
│  │     │   ├─ adapt_patch(is_global_patch=True)  → 修改 vllm 原生行为        │   │
│  │     │   ├─ 注册 ascend 量化方法 (--quantization ascend)                    │   │
│  │     │   └─ 注册 AscendCompressedTensorsConfig                             │   │
│  │     │                                                                     │   │
│  │     ├─ check_and_update_config()  ★ 配置校验 & 修正 ★                     │   │
│  │     │   ├─ 自动检测量化方法                                                │   │
│  │     │   ├─ 修正不兼容配置                                                  │   │
│  │     │   ├─ 初始化 ascend_config (soc_version, 算子配置...)                 │   │
│  │     │   └─ 设置 ACL Graph / NPU Graph 相关参数                             │   │
│  │     │                                                                     │   │
│  │     ├─ get_attn_backend_cls()  → 返回 AscendAttentionBackend              │   │
│  │     ├─ get_compile_backend()   → AscendCompiler (图编译后端)               │   │
│  │     ├─ get_static_graph_wrapper_cls() → ACLGraphWrapper (ACL图模式)       │   │
│  │     ├─ get_device_communicator_cls() → NPUCommunicator (HCCL通信)         │   │
│  │     └─ import_kernels() → 加载 AscendC 自定义算子                          │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 1.4 启动 Engine Core 进程                                                  │   │
│  │                                                                           │   │
│  │   launch_core_engines(vllm_config, executor_class, ...)                   │   │
│  │     │                                                                     │   │
│  │     ├─ Executor.get_class(vllm_config) → MultiprocExecutor / RayExecutor  │   │
│  │     │                                                                     │   │
│  │     └─ EngineCore.__init__()                                              │   │
│  │         ├─ load_general_plugins()  ★ 加载通用插件 ★                       │   │
│  │         │   └─ vllm_ascend:register_connector (KV传输)                     │   │
│  │         │   └─ vllm_ascend:register_model_loader (模型加载)                │   │
│  │         │   └─ vllm_ascend:register_service_profiling (性能分析)           │   │
│  │         ├─ executor_class(vllm_config) → 初始化 Executor                  │   │
│  │         ├─ _initialize_kv_caches() → 初始化 KV Cache                      │   │
│  │         └─ Scheduler.__init__() → 初始化调度器                             │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 1.5 Worker 进程初始化 (vllm-ascend NPUWorker)                              │   │
│  │                                                                           │   │
│  │   NPUWorker.__init__()                                                    │   │
│  │     │                                                                     │   │
│  │     ├─ adapt_patch()  ★ Worker 级 Patch 注入 ★                            │   │
│  │     │   └─ 修改 vllm worker 行为以适配 NPU                                │   │
│  │     │                                                                     │   │
│  │     ├─ ops.register_dummy_fusion_op()  ★ 注册融合算子 ★                   │   │
│  │     ├─ _register_atb_extensions()  ★ 注册 ATB 扩展 ★                      │   │
│  │     ├─ register_ascend_customop()  ★ 注册 AscendC 自定义算子 ★            │   │
│  │     ├─ init_ascend_config() → 初始化 Ascend 配置                           │   │
│  │     │                                                                     │   │
│  │     ├─ super().__init__() → WorkerBase.__init__()                         │   │
│  │     │                                                                     │   │
│  │     └─ init_device()                                                      │   │
│  │         ├─ torch.npu.set_device() → 设置 NPU 设备                          │   │
│  │         ├─ init_workspace_manager() → 初始化 workspace                     │   │
│  │         └─ NPUModelRunner(vllm_config, device)  ★ 创建 NPU ModelRunner ★  │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 1.6 模型加载 & 预热                                                        │   │
│  │                                                                           │   │
│  │   NPUModelRunner.__init__()                                               │   │
│  │     │                                                                     │   │
│  │     ├─ get_model() → 加载 HuggingFace 模型权重到 NPU                       │   │
│  │     ├─ 初始化 KV Cache 相关 buffer                                        │   │
│  │     ├─ 初始化 Attention Backend (AscendAttentionBackend)                   │   │
│  │     ├─ 初始化 ACLGraphWrapper (ACL 图模式)                                 │   │
│  │     ├─ 初始化 NPU Graph (类似 CUDA Graph)                                  │   │
│  │     └─ profile_run() → 预热 & 内存分析                                     │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 1.7 启动 API Server (FastAPI + Uvicorn)                                   │   │
│  │                                                                           │   │
│  │   APIServerProcessManager → 启动 HTTP 服务                                 │   │
│  │     └─ run_server() → serve_http() → uvicorn.run()                        │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 阶段 2: 请求调度 (Request Scheduling)                                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 2.1 HTTP 请求到达 API Server                                              │   │
│  │                                                                           │   │
│  │   POST /v1/chat/completions  (OpenAI 兼容 API)                            │   │
│  │     │                                                                     │   │
│  │     ├─ FastAPI 路由 → OpenAIServingChat                                   │   │
│  │     ├─ Tokenization (InputProcessor)                                      │   │
│  │     │   └─ tokenizer.encode(prompt) → token_ids                          │   │
│  │     ├─ 多模态数据加载 (如有图片/视频/音频)                                   │   │
│  │     └─ 构建 EngineCoreRequest                                             │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼  (ZMQ Socket)                             │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 2.2 Engine Core 接收请求                                                   │   │
│  │                                                                           │   │
│  │   EngineCore.run_busy_loop()  ★ 核心忙等循环 ★                            │   │
│  │     │                                                                     │   │
│  │     ├─ _process_input_queue()                                             │   │
│  │     │   └─ input_queue.get() → _handle_client_request(*req)               │   │
│  │     │       └─ scheduler.add_request(request)                             │   │
│  │     │                                                                     │   │
│  │     └─ _process_engine_step()                                             │   │
│  │         └─ step_fn() → step()                                             │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 2.3 Scheduler.schedule() ★ 核心调度逻辑 ★                                 │   │
│  │                                                                           │   │
│  │   Scheduler (vllm/v1/core/sched/scheduler.py)                             │   │
│  │     │                                                                     │   │
│  │     ├─ 调度策略: FCFS (先来先服务) / PRIORITY (优先级)                      │   │
│  │     │                                                                     │   │
│  │     ├─ _schedule_new_requests()                                           │   │
│  │     │   ├─ 从 waiting 队列取请求                                           │   │
│  │     │   ├─ 检查 KV Cache 是否有足够空间                                     │   │
│  │     │   ├─ 分配 KV Cache blocks (KVCacheManager)                          │   │
│  │     │   └─ 构建 NewRequestData (token_ids, mm_features...)                │   │
│  │     │                                                                     │   │
│  │     ├─ _schedule_running_requests()                                       │   │
│  │     │   ├─ 为 running 请求分配新 token (decode 阶段)                       │   │
│  │     │   └─ 构建 CachedRequestData                                         │   │
│  │     │                                                                     │   │
│  │     ├─ Chunked Prefill: 大请求分块调度                                      │   │
│  │     │   └─ max_num_batched_tokens 限制每步 token 数                        │   │
│  │     │                                                                     │   │
│  │     └─ 输出 SchedulerOutput                                               │   │
│  │         ├─ scheduled_new_reqs (新请求数据)                                 │   │
│  │         ├─ scheduled_cached_reqs (运行中请求数据)                           │   │
│  │         ├─ num_scheduled_tokens (每个请求调度的 token 数)                   │   │
│  │         ├─ scheduled_encoder_inputs (多模态编码器输入)                      │   │
│  │         └─ finished_req_ids (已完成的请求)                                  │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 阶段 3: 推理准备 (Inference Preparation)                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 3.1 EngineCore.step() 调度→执行→输出                                       │   │
│  │                                                                           │   │
│  │   def step(self):                                                         │   │
│  │     scheduler_output = self.scheduler.schedule()     # 调度               │   │
│  │     future = self.model_executor.execute_model(      # 异步执行模型        │   │
│  │         scheduler_output, non_block=True)                                 │   │
│  │     grammar_output = self.scheduler.get_grammar_bitmask(...)  # 语法约束  │   │
│  │     model_output = future.result()                   # 等待结果           │   │
│  │     engine_core_outputs = self.scheduler.update_from_output(              │   │
│  │         scheduler_output, model_output)              # 更新调度器状态      │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 3.2 Executor → WorkerBase.execute_model()                                 │   │
│  │                                                                           │   │
│  │   MultiprocExecutor / RayExecutor                                        │   │
│  │     └─ 将 SchedulerOutput 分发到各 Worker 进程                             │   │
│  │         └─ WorkerBase.execute_model(scheduler_output)                     │   │
│  │             ├─ _apply_mm_cache() → 多模态缓存处理                          │   │
│  │             └─ worker.execute_model(scheduler_output)                     │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 3.3 NPUWorker.execute_model() (vllm-ascend)                               │   │
│  │                                                                           │   │
│  │   NPUWorker.execute_model(scheduler_output)                               │   │
│  │     │                                                                     │   │
│  │     ├─ Pipeline Parallelism 处理 (send/recv intermediate_tensors)         │   │
│  │     └─ self.model_runner.execute_model(scheduler_output)                  │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 3.4 NPUModelRunner 输入准备                                                │   │
│  │                                                                           │   │
│  │   NPUModelRunner.execute_model(scheduler_output)                          │   │
│  │     │                                                                     │   │
│  │     ├─ _determine_batch_execution_and_padding()                           │   │
│  │     │   ├─ 区分 Prefill vs Decode 请求                                     │   │
│  │     │   ├─ 计算 batch 大小和 padding                                       │   │
│  │     │   └─ 构建 BatchDescriptor                                           │   │
│  │     │                                                                     │   │
│  │     ├─ prepare_inputs() → NPUInputBatch                                   │   │
│  │     │   ├─ 构建 input_ids tensor (NPU)                                    │   │
│  │     │   ├─ 构建 positions tensor                                          │   │
│  │     │   ├─ 构建 block_tables (KV Cache 索引)                               │   │
│  │     │   ├─ 构建 slot_mappings (KV Cache slot 映射)                         │   │
│  │     │   ├─ 构建 query_start_loc (ragged batch 起始位置)                    │   │
│  │     │   └─ 构建 seq_lens (每个请求的序列长度)                               │   │
│  │     │                                                                     │   │
│  │     ├─ 准备 Attention Metadata                                            │   │
│  │     │   └─ AscendAttentionMetadataBuilder.build()                         │   │
│  │     │       ├─ 区分 PrefillNoCache / DecodeOnly / ChunkedPrefill          │   │
│  │     │       └─ 构建 AscendMetadata (block_tables, seq_lens, ...)          │   │
│  │     │                                                                     │   │
│  │     └─ 设置 ForwardContext (LoRA, 量化参数...)                             │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 阶段 4: 模型推理 (Model Inference)                                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 4.1 模型 Forward Pass                                                     │   │
│  │                                                                           │   │
│  │   model.forward(input_ids, positions, attn_metadata)                      │   │
│  │     │                                                                     │   │
│  │     ├─ Embedding Layer                                                    │   │
│  │     │   └─ VocabParallelEmbedding (TP 分片)                               │   │
│  │     │                                                                     │   │
│  │     ├─ for each DecoderLayer:                                             │   │
│  │     │   │                                                                 │   │
│  │     │   ├─ Input LayerNorm / RMSNorm                                      │   │
│  │     │   │                                                                 │   │
│  │     │   ├─ ★ Self-Attention ★                                             │   │
│  │     │   │   │                                                             │   │
│  │     │   │   ├─ Q/K/V Projection (ColumnParallelLinear)                    │   │
│  │     │   │   ├─ RoPE (Rotary Position Embedding)                           │   │
│  │     │   │   └─ Attention.forward()                                        │   │
│  │     │   │       │                                                         │   │
│  │     │   │       └─ AscendAttentionBackendImpl.forward_impl()              │   │
│  │     │   │           │                                                     │   │
│  │     │   │           ├─ DecodeOnly:                                        │   │
│  │     │   │           │   └─ forward_paged_attention()                      │   │
│  │     │   │           │       └─ torch_npu._npu_paged_attention() ★NPU算子★ │   │
│  │     │   │           │           (PagedAttention NPU 实现)                  │   │
│  │     │   │           │                                                     │   │
│  │     │   │           ├─ PrefillNoCache:                                    │   │
│  │     │   │           │   └─ forward_fused_infer_attention()                │   │
│  │     │   │           │       └─ torch_npu._npu_fused_infer_attention_score()│   │
│  │     │   │           │           (FusedInferAttention NPU 实现)             │   │
│  │     │   │           │                                                     │   │
│  │     │   │           └─ ChunkedPrefill:                                    │   │
│  │     │   │               └─ forward_fused_infer_attention()                │   │
│  │     │   │                                                                 │   │
│  │     │   ├─ O Projection (RowParallelLinear)                               │   │
│  │     │   │                                                                 │   │
│  │     │   ├─ ★ MLP / MoE ★                                                  │   │
│  │     │   │   ├─ Gate + Up Projection                                        │   │
│  │     │   │   ├─ Activation (SiLU / GELU)                                   │   │
│  │     │   │   ├─ Down Projection                                             │   │
│  │     │   │   └─ (MoE) Expert Routing + FusedMoE ★NPU融合算子★              │   │
│  │     │   │                                                                 │   │
│  │     │   └─ Residual Connection                                             │   │
│  │     │                                                                     │   │
│  │     └─ Final LayerNorm → LM Head → logits                                 │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 4.2 NPU Graph 加速 (ACLGraph / NPUGraph)                                  │   │
│  │                                                                           │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │   │ NPU Graph 模式 (类似 CUDA Graph):                                 │    │   │
│  │   │   - 首次运行: Capture 计算图 → NPU Graph                          │    │   │
│  │   │   - 后续运行: Replay NPU Graph (减少 Launch Overhead)             │    │   │
│  │   │                                                                   │    │   │
│  │   │ ACL Graph 模式 (Ascend 特有):                                     │    │   │
│  │   │   - ACLGraphWrapper: 将模型编译为 ACL 静态图                       │    │   │
│  │   │   - 支持 Piecewise Graph (分段图)                                 │    │   │
│  │   │   - 支持 LoRA + ACLGraph                                          │    │   │
│  │   │                                                                   │    │   │
│  │   │ Graph Fusion Pass Manager:                                        │    │   │
│  │   │   - AllReduce + RMSNorm Fusion                                    │    │   │
│  │   │   - Norm + Quant Fusion                                           │    │   │
│  │   │   - QKNorm + RoPE Fusion                                          │    │   │
│  │   └─────────────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 4.3 Sampling (采样)                                                       │   │
│  │                                                                           │   │
│  │   model_runner.sample_tokens(grammar_output)                              │   │
│  │     │                                                                     │   │
│  │     ├─ LogitsProcessor (temperature, top_k, top_p, min_p...)              │   │
│  │     ├─ Structured Output (grammar bitmask via xgrammar)                   │   │
│  │     ├─ AscendSampler.sample() ★ NPU 采样 ★                                │   │
│  │     │   └─ torch_npu._npu_multinomial() / topk_topp_sampler               │   │
│  │     └─ 输出 SamplerOutput (sampled_token_ids, logprobs)                   │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 阶段 5: 后端处理 & 结果返回 (Backend Processing)                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 5.1 Scheduler 状态更新                                                     │   │
│  │                                                                           │   │
│  │   scheduler.update_from_output(scheduler_output, model_output)            │   │
│  │     │                                                                     │   │
│  │     ├─ 更新每个 Request 的 num_computed_tokens                            │   │
│  │     ├─ 将新 token 追加到 Request                                          │   │
│  │     ├─ 检查 Request 是否完成 (EOS / stop_string / max_tokens)             │   │
│  │     ├─ 释放已完成请求的 KV Cache blocks                                    │   │
│  │     └─ 返回 EngineCoreOutputs (包含每个请求的新 token)                     │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼  (ZMQ Socket)                             │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 5.2 AsyncLLM Output Handler (API Server 进程)                             │   │
│  │                                                                           │   │
│  │   output_handler() async task:                                            │   │
│  │     │                                                                     │   │
│  │     ├─ engine_core.get_output_async() → EngineCoreOutputs                 │   │
│  │     ├─ output_processor.process_outputs()                                 │   │
│  │     │   ├─ Detokenization (token_ids → text)                              │   │
│  │     │   ├─ Stop string 检测                                               │   │
│  │     │   └─ 构建 RequestOutput                                             │   │
│  │     ├─ 推入 per-request queue                                             │   │
│  │     └─ 日志记录 (Prometheus metrics, stat loggers)                        │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                           │
│                                      ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ 5.3 流式返回给客户端                                                       │   │
│  │                                                                           │   │
│  │   AsyncLLM.generate() → AsyncGenerator[RequestOutput]                     │   │
│  │     │                                                                     │   │
│  │     └─ FastAPI StreamingResponse → SSE (Server-Sent Events)               │   │
│  │         └─ data: {"choices": [{"delta": {"content": "你好"}}], ...}       │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、vLLM-Ascend 插件化能力详解

vLLM-Ascend 通过 **Python entry_points** 机制实现插件化，核心注册在 `setup.py`：

```python
entry_points={
    "vllm.platform_plugins": ["ascend = vllm_ascend:register"],
    "vllm.general_plugins": [
        "ascend_kv_connector = vllm_ascend:register_connector",
        "ascend_model_loader = vllm_ascend:register_model_loader",
        "ascend_service_profiling = vllm_ascend:register_service_profiling",
    ],
}
```

### 3.1 插件加载流程

```
vllm/platforms/__init__.py::resolve_current_platform_cls_qualname()
  │
  ├─ 1. 扫描内置平台 (cuda/rocm/xpu/cpu/tpu)
  │     └─ 调用各平台的注册函数，返回支持的平台类名
  │
  └─ 2. 扫描 entry_points "vllm.platform_plugins"
        └─ vllm_ascend:register() → "vllm_ascend.platform.NPUPlatform"
              │
              └─ NPUPlatform 被设为 current_platform
                    │
                    └─ 全局可用: from vllm.platforms import current_platform
```

### 3.2 NPUPlatform 提供的插件化接口

| 接口方法 | 作用 |
|---------|------|
| `pre_register_and_update()` | 全局 Patch 注入、注册量化方法 |
| `check_and_update_config()` | 配置校验修正、初始化 Ascend 配置 |
| `get_attn_backend_cls()` | 返回 AscendAttentionBackend |
| `get_compile_backend()` | 返回 AscendCompiler (图编译) |
| `get_static_graph_wrapper_cls()` | 返回 ACLGraphWrapper |
| `get_device_communicator_cls()` | 返回 NPUCommunicator (HCCL) |
| `get_pass_manager_cls()` | 返回 GraphFusionPassManager |
| `import_kernels()` | 加载 AscendC 自定义算子 |
| `opaque_attention_op()` | 返回 True (Attention 作为不透明算子) |

### 3.3 Patch 机制

vLLM-Ascend 通过 **Monkey Patch** 修改 vLLM 原生行为，分为两级：

| Patch 级别 | 触发时机 | 作用范围 |
|-----------|---------|---------|
| **Platform Patch** | `NPUPlatform.pre_register_and_update()` | Engine Core 进程 |
| **Worker Patch** | `NPUWorker.__init__()` | 每个 Worker 进程 |

Patch 原则：
1. **Less is more** — 非必要不 Patch
2. **有进有出** — 每个 Patch 需描述未来移除计划
3. **随时清理** — 欢迎清理过时 Patch

### 3.4 Attention 后端路由

```
vllm/v1/attention/selector.py::get_attn_backend()
  │
  └─ current_platform.get_attn_backend_cls()
        │
        └─ NPUPlatform → AscendAttentionBackend (CUSTOM)
              │
              ├─ AscendAttentionBackendImpl (标准 Attention)
              │   ├─ forward_paged_attention() → torch_npu._npu_paged_attention()
              │   └─ forward_fused_infer_attention() → torch_npu._npu_fused_infer_attention_score()
              │
              ├─ AscendAttentionCPImpl (Context Parallel)
              │
              └─ AscendMLABackend (MLA Attention, DeepSeek-V2/V3)
```

### 3.5 模型适配

vLLM-Ascend 支持两种模型适配方式：

1. **复用 vLLM 原生模型**：通过 `from vllm.attention import Attention` 自动路由到 Ascend Attention 后端
2. **Ascend 专用模型**：在 `vllm_ascend/models/` 下实现，通过 `ModelRegistry.register_model()` 注册

### 3.6 关键特性矩阵

| 特性 | 说明 |
|------|------|
| **ACL Graph** | 静态图编译加速 |
| **NPU Graph** | 动态图 Capture/Replay |
| **Graph Fusion** | 算子融合 Pass (AllReduce+Norm, Norm+Quant, QKNorm+RoPE) |
| **AscendC Custom Ops** | 自定义高性能算子 (PagedAttention, FusedMoE, RMSNorm 等) |
| **量化 (W8A8/W4A8/FP8)** | Ascend 量化方案 |
| **LoRA** | 支持 ACLGraph + LoRA |
| **Context Parallel** | CP 注意力实现 |
| **Expert Parallel** | MoE EP + EPLB 动态负载均衡 |
| **Speculative Decoding** | 投机解码 (Eagle/Medusa/Ngram) |
| **Sequence Parallelism** | SP 优化 |
| **KV Pool / EPD** | KV Cache 池化 & 分离式推理 |

---

## 四、关键代码路径速查表

| 功能 | vLLM 原生 | vLLM-Ascend 插件 |
|------|----------|-----------------|
| **平台注册** | `vllm/platforms/__init__.py` | `vllm_ascend/__init__.py` |
| **平台实现** | `vllm/platforms/cuda.py` | `vllm_ascend/platform.py` |
| **Worker** | `vllm/v1/worker/gpu_worker.py` | `vllm_ascend/worker/worker.py` |
| **Model Runner** | `vllm/v1/worker/gpu/model_runner.py` | `vllm_ascend/worker/model_runner_v1.py` |
| **Attention** | `vllm/v1/attention/backends/` | `vllm_ascend/attention/attention_v1.py` |
| **Scheduler** | `vllm/v1/core/sched/scheduler.py` | (复用 vLLM) |
| **Engine Core** | `vllm/v1/engine/core.py` | (复用 vLLM) |
| **AsyncLLM** | `vllm/v1/engine/async_llm.py` | (复用 vLLM) |
| **API Server** | `vllm/entrypoints/openai/api_server.py` | (复用 vLLM) |
| **CLI 入口** | `vllm/entrypoints/cli/serve.py` | (复用 vLLM) |
| **Patch 机制** | — | `vllm_ascend/patch/__init__.py` |
| **量化** | `vllm/model_executor/layers/quantization/` | `vllm_ascend/quantization/` |
| **图编译** | `vllm/compilation/` | `vllm_ascend/compilation/` |

---

## 五、总结

vLLM-Ascend 的插件化设计核心思路是：

1. **通过 Python entry_points 自动发现**：安装 `vllm-ascend` 包后，vLLM 启动时自动加载 NPUPlatform
2. **通过 Platform 抽象层注入差异化实现**：Attention Backend、Compiler、Communicator、Graph Wrapper 等全部通过 Platform 接口替换
3. **通过 Patch 机制兜底**：对于无法通过接口替换的硬编码逻辑，使用 Monkey Patch 临时适配
4. **Worker/ModelRunner 继承复用**：NPUWorker 继承 WorkerBase，NPUModelRunner 继承 GPUModelRunner，最大化复用 vLLM 调度、KV Cache 管理等核心逻辑
5. **AscendC 自定义算子 + ACL Graph 提供极致性能**：PagedAttention、FusedMoE、RMSNorm 等关键算子均有 NPU 定制实现