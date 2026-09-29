# DSV4.1 PythonDSL operators

Import `vllm_ascend.ops.pythondsl.ops` to register these PyTorch operators:

- `vllm_ascend::mixed_quant_sparse_flash_mla` and its metadata operator.
- `vllm_ascend::quant_lightning_indexer` and its metadata operator.
- `vllm_ascend::quant_sparse_lightning_indexer` and its metadata operator.

The implementations are the in-tree PythonDSL sources from the 2026-09-23
DSV4.1 delivery. They require the compatible `cannbotdsl` compiler/runtime
and an Ascend A5 toolchain. Neither `cannbot-arena-net-ops` nor
`cann_ops_transformer` is imported. Registration and FakeTensor execution do
not import the compiler; the first real call compiles the kernel during warmup.
No exported wheel binaries or alternate operator namespaces are used.

The imported CANN sources retain their source notices. The CANN Open Software
License Agreement Version 2.0 text is included in [LICENSE](LICENSE), copied
from the source `cann-recipes-infer/ops/LICENSE`; references to that agreement
refer to this local copy rather than the repository's top-level Apache license.

QLI and QSLI consume packed E2M1 data with E8M0 scales. QSLI's paged cache
contains eight key rows followed by eight scale rows in each 544-byte group.
Mixed attention consumes the separate 544-byte MXFP8/BF16-scale and 320-byte
MXFP4/BF16-scale cache rows. All three metadata operators return INT32[1024].
Supply device prefix sums, sequence lengths, and metadata explicitly in model
execution; optional sequence defaults are for standalone use.

The indexers use `min(32, device Cube core count)` workers for both metadata
scheduling and kernel launch. The limit of 32 is the workspace/tuning capacity,
not a device assumption: a 32-core A5 keeps 32 workers, and a 28-core A5 uses 28.
The fused LD merge contains a global barrier, so every launched worker must fit
in one resident wave. Device properties are cached on the host during warmup;
choosing the worker count does not read a device tensor or synchronize a stream.

The source kernels retain their compiler allocation and synchronization order.
Some DSL buffer declarations intentionally have no subsequent Python reference;
the AICPU `zeros` expression is a compiler intrinsic. Local lint annotations
preserve those statements. The attention kernel also retains the delivery's
FP4 cast lowering shim required by the compatible compiler. Native-wheel export
registration and profiling switches are omitted; address vectorization retains
its default enabled state and shape-based fallback.

Numerical tests are in `tests/e2e/nightly/single_node/ops/singlecard_ops/`:
`test_dsv41_triton.py` covers metadata, quantization, cache writes and folding;
`test_dsv41_dsl.py` runs the registered operators, metadata-to-indexer and
writer-to-attention paths, independent numerical references, and attention
graph replay. These require an A5 and the freshly built native cache writer.
