# SFA remap Ascend C/Triton A/B

This branch keeps the CANN-compatible Ascend C operator from PR 14373 and
adds the staged Triton candidate as a selectable SFA DCP remap backend.

Source provenance:

- Ascend C baseline: `upstream/pr/14373` at
  `85acaa6f68cd53968d50cae0a33195d0c5022ebe`.
- Supplied Triton candidate SHA256:
  `1be6baeab9ea80509cff830f890fe0a3a176b12ca1b2d958cafe2d7f44689de1`.

The service defaults to Ascend C. Select exactly one backend before service
startup:

```bash
export VLLM_ASCEND_SFA_REMAP_BACKEND=ascendc
# or
export VLLM_ASCEND_SFA_REMAP_BACKEND=triton
```

All other serving parameters should remain identical between runs. The
backend is resolved once during `AscendSFADCPImpl` initialization and bound to
a callable, so the remap hot path does not parse the environment variable.

For isolated operator comparison, run:

```bash
python benchmarks/tests/sfa_remap_backend_benchmark.py \
  --rows 1 5 16 32 64 128 \
  --top-k 2048 8192 \
  --dcp-size 8 \
  --dcp-rank 3 \
  --interleave-size 128 \
  --json-out /tmp/sfa-remap-ab.json
```

The benchmark checks both backends against the same CPU integer reference,
requires exact output equality, alternates backend order between samples, and
excludes warmup. By default the output tensor is preallocated for both
backends; pass `--allocate-output` to include per-call output allocation.
Triton's intermediate buffers and prefix-sum operation are included in both
measurement modes.
