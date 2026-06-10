# RUNBOOK — Calibration Covariance Re-capture (gate/up + down_proj)

**Goal:** re-capture the MoE input covariances with the newly-patched vLLM
wheel **d20260609**, capturing BOTH the gate/up input covariance
(`--capture-input-covariance`) **and** the post-SwiGLU down_proj input
covariance (`--capture-input-covariance-down`, landed main `b4f882c`) into a
single `covariance.pt` in one windowed offload pass.

All paths below are relative to the repo root `/home/lucas/ai/moe_compress`.
Driver: `max_quality/scripts/build_self_traces_calib_vllm.py`.

---

## 0. TL;DR — verified constraints (read before running)

These are enforced **in code**, not optional:

- The down_proj group is **NOT** an independent capture mode. Per argparse
  (`build_self_traces_calib_vllm.py:1576-1578`) it *"Rides the
  `--input-cov-offload` windowed path (down-only without `--input-cov-offload`
  is unsupported)."* → you MUST pass `--input-cov-offload`.
- `--input-cov-offload` is *"Only valid in `--replay-from` mode"*
  (`:1598-1600`) and the input-cov group(s) must be the **SOLE** capture
  flag(s) — the per-window early-exit truncates any other full-forward signal.
  Validated at `:2156-2182` (`return 1` if any other `--capture-*` is on).
- Triton MoE backend only — the down SYRK lives inside `TritonExperts.apply`;
  FlashInfer's monolithic path does not expose the intermediate
  (`:1574-1576`). The driver hard-forces `moe_backend="triton"` in the replay
  path anyway (`:468-486`).
- Both groups ride **one** pass and each layer integrates the full corpus
  exactly once (`:3835`). The down group is exempt from the "sole flag" check
  by construction — it is an offload rider, not a `_CAPTURE_WRITER_MODULES`
  entry (`:2162-2164`).

---

## 1. EXACT command line

Run this from inside the prepared GPU box, repo at `${REPO_ROOT}`, venv
active, **after** Section 2's wheel install. `${TRACES_JSONL}` is the existing
v2 self-traces JSONL (the `--replay-from` source — generation is skipped):

```bash
python max_quality/scripts/build_self_traces_calib_vllm.py \
  --teacher Qwen/Qwen3.6-35B-A3B \
  --teacher-revision main \
  --replay-from "${TRACES_JSONL}" \
  --capture-input-covariance \
  --capture-input-covariance-down \
  --input-cov-offload \
  --moe-backend triton \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.45 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 256 \
  --max-model-len 20480 \
  --input-cov-checkpoint-every-chunks 1 \
  --input-cov-staging-dir "${STAGING_VOL}/_cov_staging" \
  --seed 1337
```

### Flag provenance (argparse, `build_self_traces_calib_vllm.py`)

| Flag | Line | Notes |
|---|---|---|
| `--teacher` | `:1387` | default `Qwen/Qwen3.6-35B-A3B` (the calibrated teacher). |
| `--teacher-revision` | `:1388` | default `main`. |
| `--replay-from` | `:1903` | v3 forward-only replay; **required** for offload. Path to the existing self-traces JSONL. |
| `--capture-input-covariance` | `:1547` | gate_proj/up_proj group (up aliased to gate_proj). Auto-sets `VLLM_CALIB_CAPTURE_INPUT_COV=1` + `..._EXPERT=1`. |
| `--capture-input-covariance-down` | `:1562` | down_proj group (2nd in-graph grouped-SYRK). Auto-sets `VLLM_CALIB_CAPTURE_INPUT_COV_DOWN=1` (+ `..._EXPERT=1`, `VLLM_CALIB_INPUT_COV_MODE=resident`) — see env wiring `:1967-1997`. |
| `--input-cov-offload` | `:1588` | windowed CPU-offload path (REQUIRED here — all-resident Gram ~172 GB OOMs). Forces `VLLM_CALIB_INPUT_COV_MODE=resident`. |
| `--moe-backend` | `:1432` | `triton` (only backend that exposes the down intermediate; driver force-sets it anyway). |
| `--dtype` | `:1407` | `bfloat16` (shards/cov written bf16 — fp16 overflow fix, main `1878988`). |
| `--gpu-memory-utilization` | `:1412` | default 0.90 — **lower it** (0.45 suggested). vLLM otherwise reserves most VRAM for KV and the windowed Gram has no room (`:3881-3883`, abort note `:4000-4007`). |
| `--max-num-batched-tokens` | `:1426` | sets `buf_rows` for the compact SYRK scratch (`:3963`); default falls back to 2048 in offload (`:3884`). |
| `--max-num-seqs` | `:1418` | default falls back to 256 in offload (`:3885`). |
| `--max-model-len` | `:1414` | default 20480; rows over this are skipped (`:4028-4033`). |
| `--input-cov-checkpoint-every-chunks` | `:1579` | default 1; resume hydrates `<jsonl>.input_cov.ckpt`. |
| `--input-cov-staging-dir` | `:1624` | per-layer bf16 Gram shards. Default = `<jsonl>/sidecars/<stem>/_covariance_staging`. Point at a **large volume** if the JSONL lives on a small OS disk. |
| `--input-cov-window-size` | `:1617` | 0 = auto-size from free VRAM after model load (`:3985-3999`). Leave 0 unless tuning peak VRAM. |
| `--input-cov-max-rows` | `:1605` | 0 = all rows. Optional: a seeded random subset (~2-3k) cuts per-window cost 3-4x with negligible SVD/EoRA loss (`:4035-4046`). Use only if wall-time-bound. |
| `--seed` | `:1396` | default 1337; selects the `--input-cov-max-rows` subset and is part of the staging fingerprint. |

> The offload makes **one corpus pass per window** of MoE layers. If wall-time
> is tight, add `--input-cov-max-rows 3000` (seeded random subset, not a head
> slice — `:4037-4038`).

---

## 2. Env / wheel setup

The wheel is pinned via `VLLM_WHEEL_FILE` (default in both harnesses):

- `max_quality/scripts/v2_validation_harness.sh:44`
- `max_quality/scripts/l1_validation_harness.sh:28`

```
VLLM_WHEEL_FILE="${VLLM_WHEEL_FILE:-vllm-0.21.1.dev0+gad7125a43.d20260609-cp312-cp312-linux_x86_64.whl}"
VLLM_WHEEL_REPO="${VLLM_WHEEL_REPO:-pirola/vllm-patched-calib}"   # v2_validation_harness.sh:43
```

**Bare CUDA-13 image needs a full env build** (per prior vast/DataCrunch
runs — no Python/torch preinstalled). Mirror the harness's
`v2_validation_harness.sh` Phases 1-4:

1. **Phase 1** apt: `git python3 python3-pip python3-venv python3-dev curl ca-certificates`.
2. **Phase 2** venv + torch: `python3 -m venv /tmp/venv && . /tmp/venv/bin/activate`; `pip install "numpy<2.0"`; `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130`.
3. **Phase 3** wheel pull + install (`v2_validation_harness.sh:65-83`):
   ```bash
   pip install "huggingface_hub>=1.16"
   python - <<'PY'
   import os; from huggingface_hub import hf_hub_download
   p = hf_hub_download(repo_id="pirola/vllm-patched-calib",
                       filename=os.environ["VLLM_WHEEL_FILE"],
                       local_dir="/tmp/wheels", token=os.environ.get("HF_TOKEN"))
   print("downloaded ->", p)
   PY
   pip install "/tmp/wheels/${VLLM_WHEEL_FILE}"
   python -c "import vllm; print('vllm:', vllm.__version__)"
   ```
4. **Phase 4**: `pip install "transformers>=4.51.0" "accelerate>=0.30.0" "datasets>=2.20.0"`.
5. Clone/checkout the repo at the recapture commit (main with `b4f882c` +
   `5fdc541` present) and `pip install -e max_quality` (or set `PYTHONPATH` to
   `max_quality/src` so `moe_compress.utils.input_cov_offload` imports).

> The patched wheel exposes `vllm.calibration_hooks`,
> `vllm.calibration_input_cov`, and the down-group buffers
> (`_INPUT_COV_DOWN_GPU` etc., used at `:3941-3945`). A stale wheel (≤d20260608)
> has **no down buffers** → the run aborts in discovery.

---

## 3. Calibration data / prompts / mix

In **replay mode the corpus is the `--replay-from` JSONL itself** — generation
is skipped, each row's `(prompt+answer)` is replayed as a single prefill-only
forward (`--replay-from` help `:1903-1911`). So `--prompts` / `--num-prompts`
are **ignored** here.

- Source JSONL = the existing **v2 self-traces** mix (the one that produced the
  85.9 GB prior `covariance.pt`):
  **HF `pirola/calib-v2-self-traces`**, file under
  `sidecars/self_traces_489ee0e1b17b43b0/` namespace — i.e. the
  `self_traces_<cache_key>.jsonl` with stem `self_traces_489ee0e1b17b43b0`.
  Download it (or the whole sidecar tree) to `${TRACES_JSONL}` before running.
- For reference, the **generate-mode** default mix is `qwen3-pretrain-mix`
  (`--prompts`, `:1389`); v2 is `qwen3-pretrain-mix-v2` (12 subsets, hybrid
  GENERATE + TEACHER_FORCED — `tasks/CALIBRATION_MIX_V2_PLAN.md`). Override
  with `--prompts <name|path.jsonl>`. **Not needed for this recapture** since we
  replay the already-generated v2 traces.

---

## 4. Expected outputs + locations

**Local (during run):**
- Per-layer bf16 Gram shards: `${STAGING_VOL}/_cov_staging/_covariance_staging/layer_<NNNNN>_{gate_proj,down_proj}.pt`
  (the `matrix_name` suffix lets gate+down for the same layer coexist —
  `input_cov_offload.py:56-63`). Resume is at **layer granularity** guarded by
  a topology+seed fingerprint `_meta.pt` (`:4059-4069`).
- Resume checkpoint: `<jsonl>.input_cov.ckpt`.

**Local (final, assembled):**
- `${TRACES_JSONL_DIR}/sidecars/self_traces_489ee0e1b17b43b0/covariance.pt`
  — single canonical sidecar (path = `sidecar_path(jsonl, "covariance")`,
  namespaced by JSONL stem, `cached_calibration_signals.py:151`). bf16.
  Assembled by `assemble_covariance` (`input_cov_offload.py:135-184`):
  `sigma_in[(layer, expert, "gate_proj"|"down_proj")]` +
  `token_counts[...]`, wrapped in `CovariancePayload`.

> ⚠ Assembly is **not memory-bounded** (`input_cov_offload.py:147-155`): the
> full ~80 GB `sigma_in` dict is resident in host RAM before the final write.
> Run on a box with enough RAM, or as a separate assemble-only process after
> vLLM exits.

**HF upload target:**
- Push the assembled `covariance.pt` (and shard tree if you want resumable
  state) back to **`pirola/calib-v2-self-traces`** under
  `sidecars/self_traces_489ee0e1b17b43b0/covariance.pt` — same repo/path that
  held the prior 85.9 GB bf16 covariance. (The driver does **not** auto-upload
  the cov sidecar; upload manually with `huggingface_hub.upload_file` /
  `upload_large_folder`.)

**Consumers (downstream, no action here):**
- Stage 3: `max_quality/src/moe_compress/stage3/plugins/input_cov_cache.py`
- Stage 4: `max_quality/src/moe_compress/stage4/plugins/input_cov_cache.py`
  + `eora_inputs.py`

Key schema both consume: `(layer_rank, expert_idx, matrix_name)` with
`matrix_name ∈ {"gate_proj", "down_proj"}`, up_proj aliased to gate_proj
(`cached_calibration_signals.py:460-461`).

---

## 5. VERIFY — content, not file size

After assembly, load the sidecar and **assert down_proj keys exist and are
finite** (the prior gate-only `covariance.pt` had NO down keys — file size
alone proves nothing):

```bash
COV="${TRACES_JSONL_DIR}/sidecars/self_traces_489ee0e1b17b43b0/covariance.pt"
python - "$COV" <<'PY'
import sys, torch
p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
# Unwrap CovariancePayload or raw dict.
sigma = getattr(p, "sigma_in", None) or (p.get("sigma_in") if isinstance(p, dict) else None) or p
keys = list(sigma.keys())
assert keys, "EMPTY covariance.pt — capture failed"
mats = {k[2] for k in keys if isinstance(k, tuple) and len(k) == 3}
print("n_entries:", len(keys), "matrices:", sorted(mats))
assert "gate_proj" in mats, "MISSING gate_proj keys"
assert "down_proj" in mats, "MISSING down_proj keys — down capture did NOT fire"
# Finiteness + shape sanity on one of each.
for want in ("gate_proj", "down_proj"):
    k = next(k for k in keys if k[2] == want)
    t = sigma[k]
    assert torch.isfinite(t.float()).all(), f"{want} {k} has inf/nan"
    print(f"  {want} sample key={k} shape={tuple(t.shape)} dtype={t.dtype} "
          f"trace={t.float().diagonal().sum().item():.3e}")
n_gate = sum(1 for k in keys if k[2] == "gate_proj")
n_down = sum(1 for k in keys if k[2] == "down_proj")
print(f"PASS — gate_proj entries={n_gate}, down_proj entries={n_down}")
PY
```

Expected: non-empty, `matrices: ['down_proj', 'gate_proj']`, both finite,
`gate_proj` shape `[d_in, d_in]` (≈2048²) and `down_proj` shape
`[d_in_down, d_in_down]` (the intermediate dim), per-expert traces > 0.

---

## 6. Resource notes

- **1×H200, tp=1.** The driver is single-GPU; the replay/offload path runs one
  vLLM engine. (Prior calib-v3 ran on a single H200; the multi-GPU path is not
  used here.)
- **172 GB all-resident Gram wall.** The full `[n_layers, E, H, H]` fp32 Gram
  is ~172 GB and OOMs (`--input-cov-offload` help `:1591-1593`). The offload
  windows MoE layers (`window_size` layers resident at a time), early-exits the
  forward after the window's top layer (`set_calibration_max_layer`), streams
  each window's Gram to a bf16 disk shard, frees the GPU slice, advances — one
  corpus pass per window (`:3826-3839`). The down group **doubles** the
  per-layer Gram footprint (one Gram per active group, `:3966-3968`), so auto
  window-sizing will pick **smaller** windows ⇒ more passes. Budget wall-time
  accordingly, or use `--input-cov-max-rows`.
- **Lower `--gpu-memory-utilization`** (0.45 suggested): vLLM reserves most
  VRAM for KV cache; the windowed Gram + scratch need the rest. If you see
  *"not enough free VRAM for even ONE layer's Gram"* (`:4000-4007`), drop GPU
  util or `--max-num-batched-tokens` further.
- **Staging volume.** Point `--input-cov-staging-dir` at the large data volume
  (not the OS root) — the shards + final ~80 GB `covariance.pt` must fit. Shards
  are durable and resume the run at layer granularity.
- **Host RAM for assembly** ≥ ~80 GB (unbounded final-dict assembly, §4 ⚠).
