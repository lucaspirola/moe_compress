# Multi-arch vLLM + FlashInfer cache rebuild recipe

This recipe rebuilds the cold-start cache used by `build_self_traces_calib_vllm.py`
for **any future calibration re-run**, across the four GPU architectures we
target (A100, H100/H200, B200, RTX 6000 Pro Blackwell). The cache eliminates
the ~30-45 min JIT pass at every vLLM cold-start.

The recipe is procedural and deterministic — if the published cache ever goes
stale (vLLM, torch, transformers, flashinfer, or the model's `config.json`
changes), regenerate it with these steps.

## When to rebuild

Trigger conditions, in order of likelihood:

1. **Pinned dependency bumps** in `requirements.txt` — vLLM, torch,
   transformers, datasets, or flashinfer version change.
2. **Teacher model changes** — pin update on Qwen3.6-35B-A3B (config.json
   hash) or swap to a different teacher.
3. **New GPU arch added** to the supported list — e.g. adding sm_11.0 or
   Blackwell-Pro variants.
4. **Calibration config change** that alters generation shapes — different
   `max_model_len`, `dtype`, or `reasoning_parser`.

Cache invalidation is keyed on all of the above. The Hugging Face bucket
artifact MUST carry an accompanying `requirements.txt` snapshot inside the
tarball so downstream consumers can validate version match before extract.

## Where the cache lives

After a clean vLLM cold-start the relevant directories are:

```
~/.cache/flashinfer/0.6.8.post1/<sm_arch>/cached_ops/   # ~250 MB / arch
~/.cache/vllm/torch_compile_cache/                       # ~50 MB
```

`<sm_arch>` is one of: `sm_80` (A100), `sm_90a` (H100/H200),
`sm_100` (B200), `sm_120` (RTX 6000 Pro Blackwell, Workstation).

## The four target archs (CUDA Compute Capability)

| Arch | GPUs (representative) | Why included |
|------|----------------------|--------------|
| 8.0 | A100 40/80 GB | Cheapest large-VRAM cloud option; default fallback |
| 9.0a | H100, H200 SXM5 | The throughput sweet spot we used for the canon run |
| 10.0 | B200 SXM6 | Next-gen, ~1.5× H200 throughput, becoming widely available |
| 12.0 | RTX 6000 Pro Workstation (Blackwell) | Owner-class hardware for self-hosted re-runs |

## Recipe — single host, multi-arch nvcc compile

Pre-requisite: **a Linux host with one of the listed GPUs** (any one will do
— nvcc compiles for *all* listed archs from a single source via
`TORCH_CUDA_ARCH_LIST`, but vLLM's `LLM(...)` warmup must instantiate on a
real CUDA device to dispatch the kernels we want compiled).

A CPU-only box does NOT work: vLLM's offline `LLM()` constructor opens a CUDA
context and allocates a GPU pool. Without a device, it raises at startup.

The cheapest practical path is a $0.30-0.50/hr vast.ai or DataCrunch GPU rental
for ~75 min wall-clock total. Or — if a calibration run has just completed on a
sufficient box — run the warmup on the same machine before destroying it.

### Step-by-step

```bash
# 1. Pin the EXACT versions used in the canon calibration run.
#    Source: max_quality/requirements.txt at the commit that produced the
#    canon self-traces dataset (currently pinned: 9c5abe2 or descendant).
python3 -m venv ~/venv-vllm-build && source ~/venv-vllm-build/bin/activate
pip install --upgrade pip wheel
pip install \
    vllm==0.21.0 \
    torch==2.11.0 \
    transformers==5.9.0 \
    datasets==4.8.5
# verify
python -c "import vllm, torch, transformers, datasets, flashinfer; \
print(f'vllm={vllm.__version__} torch={torch.__version__} \
transformers={transformers.__version__} datasets={datasets.__version__} \
flashinfer={flashinfer.__version__}')"

# 2. Set the multi-arch list. nvcc compiles for ALL listed archs in one shot.
export TORCH_CUDA_ARCH_LIST="8.0;9.0a;10.0;12.0"

# 3. (Optional) Pre-download the teacher to avoid an HF cache miss timing
#    artifact during warmup.
hf download Qwen/Qwen3.6-35B-A3B --local-dir /tmp/teacher

# 4. Trigger the JIT compiles via a single warmup invocation.
#    vLLM constructs the model on GPU, runs internal warmup forwards, and
#    every Triton/CUDA kernel that fires gets compiled for every listed arch.
#    Expect 30-45 min total wall-clock on a single H200.
python - <<'PY'
from vllm import LLM
llm = LLM(
    model="Qwen/Qwen3.6-35B-A3B",
    dtype="bfloat16",
    reasoning_parser="qwen3",
    max_model_len=20480,
    trust_remote_code=True,
    gpu_memory_utilization=0.90,
    max_logprobs=50,
)
# Probe one tiny generation to exercise the generate path too:
from vllm import SamplingParams
out = llm.generate(["Hi."], SamplingParams(temperature=0, max_tokens=4, logprobs=50))
print("warmup done:", out[0].outputs[0].text)
PY

# 5. Pack the cache. Include a requirements.txt snapshot for validation.
pip freeze > /tmp/requirements.txt
tar czf vllm_cache_multiarch.tgz \
    -C ~ .cache/flashinfer .cache/vllm \
    -C /tmp requirements.txt

# 6. Upload to HF (canonical artifact for all future vLLM cold-starts).
hf upload pirola/qwen3-6-35b-a3b-vllm-cache vllm_cache_multiarch.tgz
```

## Cold-start recipe (consumer side)

Future calibration re-runs on a fresh box can skip the JIT pass:

```bash
hf download pirola/qwen3-6-35b-a3b-vllm-cache vllm_cache_multiarch.tgz
# verify versions match before extract:
tar xzf vllm_cache_multiarch.tgz -C /tmp requirements.txt
diff /tmp/requirements.txt max_quality/requirements.txt  # must be clean
tar xzf vllm_cache_multiarch.tgz -C ~/   # extracts .cache/flashinfer + .cache/vllm
# vLLM startup now skips the ~45-min JIT pass on the target arch
```

If `diff` shows pinned-version drift, **do not extract** — the cache is stale.
Either rebuild it (this recipe) or pin your venv to the cache's versions
before running.

## Cost / time accounting

- One-shot rebuild on a single H200: ~75 min wall-clock + tar/upload time.
- At $4.615/hr (Spheron H200): ~$6 amortized.
- Saved per consumer cold-start: ~45 min × N runs × (cloud GPU $/hr).

The cache is small (~750 MB tarball) — HF dataset bucket pricing is
negligible.

## Where in this repo the trigger lives

The vLLM warmup invocation in step 4 above is functionally identical to the
first model-load that `max_quality/scripts/build_self_traces_calib_vllm.py`
performs at startup. So in practice, if you run the calibration script
end-to-end on a fresh box with `TORCH_CUDA_ARCH_LIST="8.0;9.0a;10.0;12.0"`
exported, **calibration itself becomes the cache-builder** — and the tar
step can run after the calibration completes, before tearing down the host.

This is the "queue on H200" pattern: don't pay for a separate compile rental
when a calibration run is about to happen on the right hardware anyway.

## Related

- `[[vllm-multiarch-cache-plan]]` — agent memory note covering the same
  recipe at the planning level.
- `[[disk-pressure-lever-is-uploads]]` — the cache is small enough that
  disk pressure is never an issue here.
