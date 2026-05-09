# vLLM + vLLM-Ascend 服务启动与推理全流程架构分析

本文档基于 vLLM 和 vLLM-Ascend 源码，梳理 vLLM 服务启动、请求调度、推理准备、模型推理、后端处理的全流程结构，帮助初学者熟悉 vLLM 整体框架及 vLLM-Ascend 插件化能力。

---

## 一、整体架构概览

vLLM V1 采用**多进程架构**，核心进程包括：

| 进程 | 数量 | 职责 |
|------|------|------|
| **API Server** | api-server-count | HTTP 请求处理、Tokenization、结果流式返回 |
| **Engine Core** | 1（或 dp_size 个） | 调度、KV Cache 管理、协调 Worker |
| **GPU/NPU Worker** | dp_size × tp_size × pp_size 个 | 模型加载、前向推理 |
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

## 五、核心数据结构

理解 vLLM 中关键数据结构的流转，是读懂源码的基础。以下按请求生命周期顺序梳理各阶段的核心数据结构。

### 5.1 请求全生命周期数据结构

```
客户端 HTTP 请求
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ EngineCoreRequest  (API Server → Engine Core, 通过 ZMQ 传输)     │
│   ├─ request_id: str                                            │
│   ├─ prompt_token_ids: list[int]                                │
│   ├─ mm_features: list[MultiModalFeatureSpec]                   │
│   ├─ sampling_params: SamplingParams                            │
│   ├─ arrival_time: float                                        │
│   ├─ priority: int                                              │
│   └─ lora_request: LoRARequest | None                           │
└─────────────────────────────────────────────────────────────────┘
  │  Scheduler.add_request()
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ Request  (Scheduler 内部管理)                                    │
│   ├─ request_id: str                                            │
│   ├─ prompt_token_ids: list[int]                                │
│   ├─ status: RequestStatus  (WAITING / RUNNING / FINISHED_*)    │
│   ├─ num_computed_tokens: int  (已计算的 token 数)               │
│   ├─ num_tokens: int  (prompt + 已生成的 token 总数)             │
│   ├─ block_ids: list[int]  (分配的 KV Cache block 列表)          │
│   ├─ sampling_params: SamplingParams                            │
│   ├─ max_tokens: int                                            │
│   ├─ stop_token_ids: set[int]                                   │
│   ├─ output_token_ids: list[int]  (已生成的 token)               │
│   └─ kv_transfer_params: dict | None  (分离式推理 KV 传输参数)    │
└─────────────────────────────────────────────────────────────────┘
```

**RequestStatus 状态机**：

```
                    ┌──────────┐
        add_request │ WAITING  │
        ───────────>│          │
                    └────┬─────┘
                         │ schedule() 分配 KV Cache 成功
                         ▼
                    ┌──────────┐
                    │ RUNNING  │◄──────────────┐
                    └────┬─────┘               │
                         │                    │ 未完成，继续
              ┌──────────┼──────────┐         │
              ▼          ▼          ▼         │
         EOS/stop   max_tokens   abort        │
              │          │          │         │
              ▼          ▼          ▼         │
         FINISHED_  FINISHED_  FINISHED_      │
         STOPPED    LENGTH_    ABORTED        │
                    CAPPED                    │
                                              │
         WAITING_FOR_REMOTE_KVS ──────────────┘
         (分离式推理：等待远端 KV Cache 到达后转为 RUNNING)
```

### 5.2 调度阶段数据结构

```
Scheduler.schedule() 输出:

┌─────────────────────────────────────────────────────────────────┐
│ SchedulerOutput                                                 │
│   ├─ scheduled_new_reqs: list[NewRequestData]                   │
│   │     ├─ req_id: str                                          │
│   │     ├─ prompt_token_ids: list[int]                          │
│   │     ├─ mm_features: list[MultiModalFeatureSpec]             │
│   │     ├─ sampling_params: SamplingParams                      │
│   │     ├─ block_ids: list[int]  (新分配的 KV Cache blocks)      │
│   │     ├─ num_computed_tokens: int                             │
│   │     ├─ lora_request: LoRARequest | None                     │
│   │     └─ kv_transfer_params: dict | None                      │
│   │                                                             │
│   ├─ scheduled_cached_reqs: CachedRequestData                   │
│   │     ├─ req_ids: list[str]                                   │
│   │     ├─ resumed_req_ids: list[str]  (从 waiting 恢复的请求)   │
│   │     ├─ num_computed_tokens: list[int]                       │
│   │     ├─ new_block_ids: list[list[int]]  (新分配的 blocks)     │
│   │     ├─ resumed_block_ids: list[list[int]]                   │
│   │     └─ num_common_prefix_blocks: list[int]  (前缀缓存命中)   │
│   │                                                             │
│   ├─ num_scheduled_tokens: dict[str, int]                       │
│   │     └─ {req_id: 本步调度的 token 数}                         │
│   │                                                             │
│   ├─ total_num_scheduled_tokens: int                            │
│   ├─ scheduled_spec_decode_tokens: dict[str, list[int]]         │
│   ├─ scheduled_encoder_inputs: dict[str, list[int]]             │
│   └─ finished_req_ids: set[str]                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 模型执行阶段数据结构

```
Worker 输入准备:

┌─────────────────────────────────────────────────────────────────┐
│ NPUInputBatch  (ModelRunner 构建，传入 model.forward())          │
│   ├─ input_ids: Tensor  [num_tokens]                            │
│   ├─ positions: Tensor  [num_tokens]                            │
│   ├─ query_start_loc: Tensor  [batch_size + 1]  (ragged batch)  │
│   ├─ seq_lens: Tensor  [batch_size]                             │
│   ├─ block_tables: Tensor  [batch_size, max_blocks_per_seq]     │
│   ├─ slot_mappings: Tensor  [num_tokens]                        │
│   ├─ is_token_ids: bool                                         │
│   └─ num_prefills: int  (prefill 请求数)                         │
└─────────────────────────────────────────────────────────────────┘

Attention Metadata:

┌─────────────────────────────────────────────────────────────────┐
│ AscendMetadata  (AscendAttentionMetadataBuilder 构建)            │
│   ├─ block_tables: Tensor  [batch_size, max_blocks_per_seq]     │
│   ├─ seq_lens: Tensor  [batch_size]                             │
│   ├─ context_lens: Tensor  [batch_size]                         │
│   ├─ slot_mappings: Tensor  [num_tokens]                        │
│   ├─ query_start_loc: Tensor  [batch_size + 1]                  │
│   ├─ max_query_len: int                                         │
│   ├─ max_seq_len: int                                           │
│   ├─ num_prefills: int                                          │
│   ├─ num_decode_tokens: int                                     │
│   ├─ num_prefill_tokens: int                                    │
│   └─ attn_type: str  (PrefillNoCache / DecodeOnly / ChunkedPrefill)│
└─────────────────────────────────────────────────────────────────┘

模型输出:

┌─────────────────────────────────────────────────────────────────┐
│ ModelRunnerOutput  (Worker → Engine Core)                       │
│   ├─ req_ids: list[str]                                         │
│   ├─ req_id_to_index: dict[str, int]                            │
│   ├─ sampled_token_ids: list[list[int]]  (每个请求采样的 token)  │
│   ├─ logprobs: LogprobsLists | None                             │
│   ├─ prompt_logprobs: LogprobsLists | None                      │
│   ├─ kv_connector_output: KVConnectorOutput | None              │
│   └─ spec_decoding_output: SpecDecodingOutput | None            │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 结果返回阶段数据结构

```
┌─────────────────────────────────────────────────────────────────┐
│ EngineCoreOutput  (Engine Core → API Server, 通过 ZMQ 传输)      │
│   ├─ request_id: str                                            │
│   ├─ outputs: list[EngineCoreOutputItem]                        │
│   │     ├─ token_ids: list[int]  (新生成的 token)                │
│   │     ├─ logprobs: LogprobsLists | None                       │
│   │     └─ finish_reason: FinishReason | None                   │
│   ├─ finished: bool                                             │
│   ├─ kv_transfer_params: dict | None                            │
│   └─ trace_headers: dict | None                                 │
└─────────────────────────────────────────────────────────────────┘
  │  OutputProcessor.process_outputs()
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ RequestOutput  (返回给客户端)                                    │
│   ├─ request_id: str                                            │
│   ├─ prompt: str | None                                         │
│   ├─ outputs: list[CompletionOutput]                            │
│   │     ├─ text: str  (detokenized)                             │
│   │     ├─ token_ids: list[int]                                 │
│   │     ├─ logprobs: Logprobs | None                            │
│   │     └─ finish_reason: str | None                            │
│   ├─ finished: bool                                             │
│   └─ metrics: RequestMetrics | None                             │
└─────────────────────────────────────────────────────────────────┘
```

### 5.5 KV Cache 相关数据结构

```
┌─────────────────────────────────────────────────────────────────┐
│ KVCacheBlock  (单个 KV Cache Block)                              │
│   ├─ block_id: int  (全局唯一 ID)                                │
│   ├─ ref_cnt: int  (引用计数，前缀缓存共享时 > 1)                 │
│   ├─ is_null: bool  (是否为空 block)                             │
│   └─ block_hash: int  (用于前缀缓存的哈希值)                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ BlockTable  (每个 Request 的 KV Cache 映射表)                    │
│   └─ block_ids: list[int]  (逻辑 block → 物理 block 的映射)      │
│                                                                  │
│   示例: Request 有 12 个 token, block_size=4                     │
│   block_ids = [7, 3, 15]                                        │
│   token 0-3  → block 7                                          │
│   token 4-7  → block 3                                          │
│   token 8-11 → block 15                                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ SlotMapping  (每个 token → KV Cache 物理位置)                    │
│   └─ slot_mappings[i] = block_id * block_size + block_offset    │
│                                                                  │
│   示例: token 5 在 block 3 的 offset 1                           │
│   slot_mappings[5] = 3 * 4 + 1 = 13                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ KVCacheManager  (全局 KV Cache 管理器)                           │
│   ├─ block_pool: BlockPool  (所有 block 的池)                    │
│   │     ├─ blocks: list[KVCacheBlock]                           │
│   │     ├─ free_block_queue: FreeBlockQueue                     │
│   │     └─ num_total_blocks: int                                │
│   ├─ prefix_cache: PrefixCachingScheduler                       │
│   │     └─ 基于 block_hash 的 LRU 缓存                           │
│   └─ allocate_slots(req_id, num_tokens) → list[int]             │
└─────────────────────────────────────────────────────────────────┘
```

### 5.6 配置体系数据结构

```
┌─────────────────────────────────────────────────────────────────┐
│ VllmConfig  (全局配置根对象，贯穿整个调用链)                       │
│   ├─ model_config: ModelConfig                                  │
│   │     ├─ model: str  (模型路径或 HuggingFace ID)               │
│   │     ├─ dtype: str  (float16 / bfloat16 / ...)               │
│   │     ├─ max_model_len: int                                   │
│   │     ├─ tokenizer: str                                       │
│   │     ├─ trust_remote_code: bool                              │
│   │     └─ hf_config: PretrainedConfig                          │
│   │                                                             │
│   ├─ cache_config: CacheConfig                                  │
│   │     ├─ block_size: int  (通常 16)                            │
│   │     ├─ gpu_memory_utilization: float  (通常 0.9)             │
│   │     ├─ enable_prefix_caching: bool                          │
│   │     └─ kv_cache_dtype: str  (auto / fp8 / ...)              │
│   │                                                             │
│   ├─ parallel_config: ParallelConfig                            │
│   │     ├─ tensor_parallel_size: int  (tp)                      │
│   │     ├─ pipeline_parallel_size: int  (pp)                    │
│   │     ├─ data_parallel_size: int  (dp)                        │
│   │     └─ context_parallel_size: int  (cp)                     │
│   │                                                             │
│   ├─ scheduler_config: SchedulerConfig                          │
│   │     ├─ max_num_seqs: int  (最大并发请求数)                   │
│   │     ├─ max_num_batched_tokens: int  (每步最大 token 数)      │
│   │     ├─ max_model_len: int                                   │
│   │     └─ policy: str  (fcfs / priority)                       │
│   │                                                             │
│   ├─ compilation_config: CompilationConfig                      │
│   │     ├─ mode: CompilationMode  (0/1/2/3)                     │
│   │     ├─ cudagraph_capture_sizes: list[int]                   │
│   │     └─ cudagraph_mode: CUDAGraphMode                        │
│   │                                                             │
│   ├─ lora_config: LoRAConfig | None                             │
│   ├─ speculative_config: SpeculativeConfig | None               │
│   ├─ observability_config: ObservabilityConfig                  │
│   ├─ kv_transfer_config: KVTransferConfig | None                │
│   ├─ quant_config: QuantizationConfig | None                    │
│   ├─ additional_config: dict  ★ vllm-ascend 在此注入 ascend_config ★│
│   └─ instance_id: str                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 5.7 数据流全景图

```
API Server 进程                    Engine Core 进程                  Worker 进程
─────────────                      ────────────────                  ───────────

HTTP Request                        ZMQ Input Queue
  │                                    │
  ▼                                    ▼
EngineCoreRequest ────ZMQ────>  Scheduler.add_request()
                                     │
                                     ▼
                                  Request (WAITING)
                                     │
                                     ▼ schedule()
                                  SchedulerOutput ──────ZMQ────>  NPUWorker.execute_model()
                                     │                                │
                                     │                                ▼
                                     │                          NPUInputBatch
                                     │                                │
                                     │                                ▼
                                     │                          model.forward()
                                     │                                │
                                     │                                ▼
                                  ModelRunnerOutput <────ZMQ───  ModelRunnerOutput
                                     │
                                     ▼ update_from_output()
                                  EngineCoreOutput ────ZMQ────>  OutputProcessor
                                                                     │
                                                                     ▼
                                                                  RequestOutput ──> SSE Stream
```

---

## 六、总结

vLLM-Ascend 的插件化设计核心思路是：

1. **通过 Python entry_points 自动发现**：安装 `vllm-ascend` 包后，vLLM 启动时自动加载 NPUPlatform
2. **通过 Platform 抽象层注入差异化实现**：Attention Backend、Compiler、Communicator、Graph Wrapper 等全部通过 Platform 接口替换
3. **通过 Patch 机制兜底**：对于无法通过接口替换的硬编码逻辑，使用 Monkey Patch 临时适配
4. **Worker/ModelRunner 继承复用**：NPUWorker 继承 WorkerBase，NPUModelRunner 继承 GPUModelRunner，最大化复用 vLLM 调度、KV Cache 管理等核心逻辑
5. **AscendC 自定义算子 + ACL Graph 提供极致性能**：PagedAttention、FusedMoE、RMSNorm 等关键算子均有 NPU 定制实现

---

## 七、关键文档索引

### 7.1 vLLM 官方文档

| 文档 | 路径 | 说明 |
|------|------|------|
| 架构概览 | `vllm/docs/design/arch_overview.md` | V1 多进程架构、Engine/Worker/ModelRunner 层次结构 |
| 插件系统 | `vllm/docs/design/plugin_system.md` | Platform Plugin、General Plugin 机制详解 |
| PagedAttention | `vllm/docs/design/paged_attention.md` | KV Cache 分页管理、Block Table 设计 |
| Prefix Caching | `vllm/docs/design/prefix_caching.md` | 前缀缓存自动命中与复用机制 |
| Hybrid KV Cache | `vllm/docs/design/hybrid_kv_cache_manager.md` | 混合 KV Cache 管理器设计 |
| HuggingFace 集成 | `vllm/docs/design/huggingface_integration.md` | 模型加载、Config 解析、权重映射 |
| 多模态支持 | `vllm/docs/design/multimodal/multimodal.md` | 多模态数据处理流程 |
| 配置选项 | `vllm/docs/configuration/engine_args.md` | Engine 启动参数完整列表 |
| 环境变量 | `vllm/docs/configuration/env_vars.md` | 所有环境变量说明 |
| OpenAI 兼容 API | `vllm/docs/serving/openai_compatible_server.md` | API 接口规范 |
| 离线推理 | `vllm/docs/design/offline_inference.md` | LLM 类离线推理流程 |
| 量化 | `vllm/docs/features/quantization/` | 量化方案总览 |
| LoRA | `vllm/docs/features/lora.md` | LoRA 适配器机制 |
| 投机解码 | `vllm/docs/features/spec_decode.md` | 投机解码设计 |
| 分离式推理 | `vllm/docs/features/disagg_prefill.md` | Prefill/Decode 分离架构 |
| 添加新模型 | `vllm/docs/contributing/model/basic.md` | 如何在 vLLM 中添加新模型 |

### 7.2 vLLM-Ascend 文档

| 文档 | 路径 | 说明 |
|------|------|------|
| 全流程架构（本文档） | `vllm-ascend/docs/source/developer_guide/Design_Documents/vllm_ascend_architecture_flow.md` | 服务启动、推理全流程 + 核心数据结构 |
| Patch 机制 | `vllm-ascend/docs/source/developer_guide/Design_Documents/patch.md` | Patch 原则、分类、清理指南 |
| 添加新模型 | `vllm-ascend/docs/source/developer_guide/modeling/adding_a_new_model.md` | Ascend 模型适配开发指南 |
| 量化指南 | `vllm-ascend/docs/source/developer_guide/Design_Documents/quantization.md` | Ascend 量化方案 (W8A8/W4A8/FP8) |
| FP8 on NPU | `vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/fp8-on-npu-lessons.md` | FP8 量化在 NPU 上的实践经验 |
| 多模态 EP + ACLGraph | `vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/multimodal-ep-aclgraph-lessons.md` | 多模态 + Expert Parallel + ACLGraph 组合经验 |
| 故障排查 | `vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/troubleshooting.md` | 常见问题排查指南 |
| 开发工作流 | `vllm-ascend/.agents/skills/vllm-ascend-model-adapter/references/workflow-checklist.md` | 模型适配开发检查清单 |
| LoRA 特性 | `vllm-ascend/docs/source/user_guide/feature_guide/lora.md` | Ascend LoRA 使用指南 |
| 安装指南 | `vllm-ascend/docs/source/user_guide/installation.md` | 环境安装与依赖配置 |
| 发布说明 | `vllm-ascend/docs/source/release_notes/` | 各版本 Release Notes |

### 7.3 关键源码入口

| 模块 | 文件 | 关键类/函数 |
|------|------|-----------|
| CLI 入口 | `vllm/entrypoints/cli/serve.py` | `ServeSubcommand.cmd()` |
| API Server | `vllm/entrypoints/openai/api_server.py` | `run_server()` → `serve_http()` |
| AsyncLLM | `vllm/v1/engine/async_llm.py` | `AsyncLLM.generate()`, `output_handler()` |
| Engine Core | `vllm/v1/engine/core.py` | `EngineCore.run_busy_loop()`, `step()` |
| Scheduler | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule()`, `update_from_output()` |
| KV Cache Manager | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.allocate_slots()` |
| Block Pool | `vllm/v1/core/block_pool.py` | `BlockPool`, `KVCacheBlock` |
| Worker Base | `vllm/v1/worker/worker_base.py` | `WorkerBase.execute_model()` |
| GPU Worker | `vllm/v1/worker/gpu_worker.py` | `Worker` |
| GPU ModelRunner | `vllm/v1/worker/gpu/model_runner.py` | `GPUModelRunner.execute_model()` |
| Attention Selector | `vllm/v1/attention/selector.py` | `get_attn_backend()` |
| Platform 发现 | `vllm/platforms/__init__.py` | `resolve_current_platform_cls_qualname()` |
| VllmConfig | `vllm/config/vllm.py` | `VllmConfig` |
| **Ascend 插件入口** | `vllm_ascend/__init__.py` | `register()`, `register_connector()` 等 |
| **Ascend Platform** | `vllm_ascend/platform.py` | `NPUPlatform` |
| **Ascend Worker** | `vllm_ascend/worker/worker.py` | `NPUWorker` |
| **Ascend ModelRunner** | `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner` |
| **Ascend Attention** | `vllm_ascend/attention/attention_v1.py` | `AscendAttentionBackendImpl` |
| **Ascend Patch** | `vllm_ascend/patch/__init__.py` | `adapt_patch()` |
| **Ascend 量化** | `vllm_ascend/quantization/` | Ascend 量化方案实现 |
| **Ascend 图编译** | `vllm_ascend/compilation/` | `ACLGraphWrapper`, `AscendCompiler` |
| **Ascend 模型** | `vllm_ascend/models/` | Ascend 专用模型实现 |