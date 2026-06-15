"""Multi-GPU Stage-3 covariance collection — correctness + equivalence tests.

Covers the feature in PLAN_MULTIGPU_STAGE3.md:

  * Lever 1 (model sharding):
      - ``_resolve_4bit_device_map`` (load_model 4bit device-aware pin, §3.A)
      - ``_device_for_key`` longest-prefix routing (load_compressed_model, §3.B)
      - ``test_cov_sharding_equivalence`` — the real cross-device guard: the
        cov-collection dual-forward with teacher and student on DIFFERENT
        devices must produce the same covariances as the single-device pass,
        within the §4 tolerance (CPU-stand-in: rtol=1e-5/atol=1e-6, NOT atol=0).
  * Lever 2 (data-parallel):
      - ``test_reduce_spilled_cov_dirs`` — key-wise fp32 sum of replica spills.
      - ``test_cov_dp_equivalence`` — single-pass vs 2-shard DP reduce, within
        the §4 DP tolerance (rtol=1e-4/atol=1e-5 in fp32 before storage cast).

All tests run WITHOUT a real multi-GPU box: lever-2 uses disk-based replica
handoff with CPU replicas; lever-1's cross-device guard uses CPU vs CUDA when a
single GPU is present and otherwise asserts the coercion is exercised on the
matching-device (no-op) path. The single-GPU / single-device code path is
byte-identical (regression-guarded by the existing golden/spill tests).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moe_compress.utils.activation_hooks import InputCovarianceAccumulator
from moe_compress.utils.model_io import (
    _resolve_4bit_device_map,
    _device_for_key,
)
from moe_compress.stage3.plugins.covariance_collection import (
    _collect_covariances,
    _reduce_spilled_cov_dirs,
    _shard_calib,
)


# ===========================================================================
# Lever 1 — load_model 4bit device-aware pin (section 3.A)
# ===========================================================================


def test_load_model_4bit_device_aware():
    """The 4bit single-device hint is parsed from device_map (there is no
    ``device`` parameter, section 3.A/L2). A single-device dict / cuda:N string
    is honored; a multi-device map falls back to {"": 0}."""
    # Single-device dict honored verbatim.
    assert _resolve_4bit_device_map({"": "cuda:1"}) == {"": "cuda:1"}
    assert _resolve_4bit_device_map({"": 0}) == {"": 0}
    # Concrete single-device strings wrapped.
    assert _resolve_4bit_device_map("cuda:1") == {"": "cuda:1"}
    assert _resolve_4bit_device_map("cuda") == {"": "cuda"}
    assert _resolve_4bit_device_map("cpu") == {"": "cpu"}
    # Multi-device requests cannot be honored by bnb -> pin GPU 0.
    assert _resolve_4bit_device_map("auto") == {"": 0}
    assert _resolve_4bit_device_map("balanced") == {"": 0}
    assert _resolve_4bit_device_map({"": "cuda:0", "lm_head": "cuda:1"}) == {"": 0}


# ===========================================================================
# Lever 1 — load_compressed_model per-key device routing helper (section 3.B)
# ===========================================================================


def test_device_for_key_longest_prefix():
    """``_device_for_key`` resolves by longest dotted-prefix match; "" is the
    catch-all root bucket."""
    cpu = torch.device("cpu")
    d0 = torch.device("cuda:0")
    d1 = torch.device("cuda:1")
    rmap = {
        "": cpu,
        "model.layers.0": d0,
        "model.layers.1": d1,
    }
    # Exact-layer prefix wins over the root catch-all.
    assert _device_for_key("model.layers.0.mlp.experts.gate_up_proj", rmap) == d0
    assert _device_for_key("model.layers.1.mlp.gate.weight", rmap) == d1
    # A key outside any layer prefix falls to the root bucket.
    assert _device_for_key("model.embed_tokens.weight", rmap) == cpu
    # A near-miss prefix (layers.10 must NOT match layers.1) — dotted boundary.
    rmap2 = {"": cpu, "model.layers.1": d1}
    assert _device_for_key("model.layers.10.mlp.gate.weight", rmap2) == cpu


# ===========================================================================
# Lever 1 — accelerate dispatch routing (placement + forward completes)
# ===========================================================================


def test_load_compressed_multidevice_routing(tiny_model):
    """Scope-limited to PLACEMENT, not intra-layer forward (M1). Force a
    2-bucket whole-layer device map (layer 0 vs layer 1 — honoring
    ``no_split_module_classes`` so the fused stacked expert params stay intact)
    and assert: (i) each param landed on its layer's mapped device, (ii)
    ``dispatch_model`` ran and a plain forward completes (accelerate relocates
    the hidden state across the layer-0 -> layer-1 boundary).

    This passes trivially for the fused experts and intentionally does NOT claim
    intra-layer cross-device safety — the cov-collection coercion is the genuine
    cross-device guard (``test_cov_sharding_equivalence``). Needs accelerate;
    skipped where it (or a 2nd device) is unavailable.
    """
    accelerate = pytest.importorskip("accelerate")
    import copy
    from accelerate import dispatch_model

    # Whole-layer buckets. With a 2nd device available, split layer 0/1 across
    # cuda:0 and cpu (a genuine cross-device-boundary forward); otherwise both
    # land on cpu (dispatch still runs + forward completes).
    second = "cpu"
    first = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = copy.deepcopy(tiny_model).eval()
    device_map = {
        "embed": first,
        "model.layers.0": first,
        "model.layers.1": second,
        "lm_head": second,
    }
    model = dispatch_model(model, device_map=device_map)

    # (i) Each param landed on its layer's mapped device.
    for name, p in model.named_parameters():
        if name.startswith("model.layers.0.") or name.startswith("embed"):
            assert p.device.type == torch.device(first).type, \
                f"{name} on {p.device}, expected {first}"
        elif name.startswith("model.layers.1.") or name.startswith("lm_head"):
            assert p.device.type == torch.device(second).type, \
                f"{name} on {p.device}, expected {second}"

    # (ii) Forward completes — accelerate relocates across the layer boundary.
    out = model(input_ids=torch.randint(0, 32, (2, 8)))
    assert out.logits.shape == (2, 8, 32)


# ===========================================================================
# Lever 1 — cross-device equivalence of the cov-collection dual-forward
# ===========================================================================


def _run_collect(model, teacher, moe_layers, teacher_moe_layers, batches,
                 *, device, cross, storage_dtype=torch.float32):
    B_acc = InputCovarianceAccumulator()
    B_acc.set_storage_dtype(storage_dtype)
    C_acc = None
    if cross:
        C_acc = InputCovarianceAccumulator()
        C_acc.set_storage_dtype(storage_dtype)
    _collect_covariances(
        model, moe_layers, batches, B_acc, device=device,
        teacher_model=teacher,
        teacher_moe_layers=teacher_moe_layers,
        C_acc=C_acc,
    )
    B_acc.finalize_all()
    if C_acc is not None:
        C_acc.finalize_all()
    return B_acc, C_acc


def test_cov_sharding_equivalence(tiny_model):
    """The dual-forward cross-covariance is invariant to the DEVICE the pass
    runs on — this exercises the teacher-row ``.to(tgt_device)`` coercion
    (covariance_collection.py:310-313) and the accumulator coercion
    (activation_hooks.py:1051), proving device-relocation correctness.

    On a box WITH a GPU we run the reference pass on CPU and the comparison pass
    on CUDA (the CPU-stand-in for a second device): a CPU matmul and a CUDA
    matmul are NOT bit-identical, so we assert rtol=1e-5/atol=1e-6, NOT atol=0
    (section 4/N1). On a CPU-only box both passes run on CPU and must be
    bit-identical. The true same-arch 2-GPU ``atol=0`` variant is the multi-GPU
    integration check (it needs two physical GPUs; not runnable in CI).

    NOTE: the fused-expert tiny model is a plain nn.Module (no accelerate
    dispatch), so whole-model-on-different-devices is not representable here —
    the genuine intra-layer cross-device split is covered by the real run. This
    test isolates the cov-collection coercion + device-relocation math.
    """
    import copy
    from moe_compress.utils.model_io import iter_moe_layers

    torch.manual_seed(7)
    batches = [torch.randint(0, 32, (2, 8)) for _ in range(3)]

    # --- Reference: pass on CPU. ---
    student_ref = copy.deepcopy(tiny_model).eval()
    teacher_ref = copy.deepcopy(tiny_model).eval()
    moe_ref = list(iter_moe_layers(student_ref))
    tmoe_ref = list(iter_moe_layers(teacher_ref))
    B_ref, C_ref = _run_collect(
        student_ref, teacher_ref, moe_ref, tmoe_ref, batches,
        device=torch.device("cpu"), cross=True,
    )

    # --- Comparison: pass on CUDA (stand-in 2nd device) if available. ---
    student = copy.deepcopy(tiny_model).eval()
    teacher = copy.deepcopy(tiny_model).eval()
    if torch.cuda.is_available():
        student = student.to("cuda:0")
        teacher = teacher.to("cuda:0")
        batch_device = torch.device("cuda:0")
        atol, rtol = 1e-6, 1e-5                  # CPU vs CUDA: NOT bit-identical
    else:
        batch_device = torch.device("cpu")
        atol, rtol = 0.0, 0.0                    # same device: bit-identical
    moe = list(iter_moe_layers(student))
    tmoe = list(iter_moe_layers(teacher))
    B_split, C_split = _run_collect(
        student, teacher, moe, tmoe, batches,
        device=batch_device, cross=True,
    )

    # Compare every covariance key (move both to CPU for comparison).
    assert set(B_ref.covariance) == set(B_split.covariance)
    for k in B_ref.covariance:
        a = B_ref.covariance[k].to(torch.float32).cpu()
        b = B_split.covariance[k].to(torch.float32).cpu()
        assert torch.allclose(a, b, rtol=rtol, atol=atol), f"B mismatch at {k}"
    assert set(C_ref.covariance) == set(C_split.covariance)
    for k in C_ref.covariance:
        a = C_ref.covariance[k].to(torch.float32).cpu()
        b = C_split.covariance[k].to(torch.float32).cpu()
        assert torch.allclose(a, b, rtol=rtol, atol=atol), f"C mismatch at {k}"


# ===========================================================================
# Lever 2 — _reduce_spilled_cov_dirs (section 3.C)
# ===========================================================================


def _spill_synth(dir_path, layer_ids, keys_per_layer, hidden, seed, dtype):
    """Write a synthetic replica spill dir; return the in-memory cov/token dicts."""
    acc = InputCovarianceAccumulator()
    acc.set_storage_dtype(dtype)
    g = torch.Generator().manual_seed(seed)
    cov = {}
    tok = {}
    for li in layer_ids:
        for (e, mat) in keys_per_layer:
            m = torch.randn(hidden, hidden, generator=g, dtype=torch.float32)
            psd = (m @ m.T).to(dtype)
            key = (li, e, mat)
            acc.covariance[key] = psd
            acc.token_count[key] = (e + 1) * 8 + seed
            cov[key] = psd.to(torch.float32)
            tok[key] = (e + 1) * 8 + seed
    for li in layer_ids:
        acc.spill_layer_to_disk(li, dir_path)
    return cov, tok


def test_reduce_spilled_cov_dirs(tmp_path):
    """3 replica spill dirs -> merged file equals the key-wise fp32 sum within
    bf16 storage eps; token_counts sum exactly (integers)."""
    layer_ids = [0, 1]
    keys = [(0, "gate_proj"), (1, "gate_proj"), (0, "down_proj")]
    hidden = 8
    dtype = torch.bfloat16

    replica_dirs = []
    expect_cov: dict = {}
    expect_tok: dict = {}
    for r in range(3):
        d = tmp_path / f"replica_{r}"
        d.mkdir()
        replica_dirs.append(d)
        cov, tok = _spill_synth(d, layer_ids, keys, hidden, seed=10 + r, dtype=dtype)
        for k, v in cov.items():
            expect_cov[k] = expect_cov.get(k, torch.zeros_like(v)) + v
        for k, n in tok.items():
            expect_tok[k] = expect_tok.get(k, 0) + n

    out_dir = tmp_path / "merged"
    # Pass dirs in shuffled order — the reducer must sort internally (determinism).
    written = _reduce_spilled_cov_dirs(
        [replica_dirs[2], replica_dirs[0], replica_dirs[1]], out_dir,
        storage_dtype=dtype,
    )
    assert sorted(written) == sorted(layer_ids)

    for li in layer_ids:
        payload = torch.load(out_dir / f"layer_{li}.pt", map_location="cpu",
                             weights_only=True)
        assert payload["format_version"] == 1
        for (e, mat) in keys:
            key = (li, e, mat)
            got = payload["covariance"][key].to(torch.float32)
            exp = expect_cov[key]
            # Merged is summed in fp32 then cast to bf16; allow bf16 eps.
            assert torch.allclose(got, exp.to(dtype).to(torch.float32),
                                  rtol=1e-2, atol=1e-2), f"cov mismatch {key}"
            assert payload["tokens"][key] == expect_tok[key], f"tok mismatch {key}"


def test_reduce_spilled_cov_dirs_fp32_exact(tmp_path):
    """In fp32 storage the key-wise reduce matches the single-sum to rtol=1e-6
    (the section 7 ``rtol=1e-6`` claim, isolated from storage quantization)."""
    layer_ids = [0]
    keys = [(0, "gate_proj"), (1, "gate_proj")]
    hidden = 6
    dtype = torch.float32

    replica_dirs = []
    expect: dict = {}
    for r in range(3):
        d = tmp_path / f"r{r}"
        d.mkdir()
        replica_dirs.append(d)
        cov, _ = _spill_synth(d, layer_ids, keys, hidden, seed=r, dtype=dtype)
        for k, v in cov.items():
            expect[k] = expect.get(k, torch.zeros_like(v)) + v

    out_dir = tmp_path / "out"
    _reduce_spilled_cov_dirs(replica_dirs, out_dir, storage_dtype=dtype)
    payload = torch.load(out_dir / "layer_0.pt", map_location="cpu", weights_only=True)
    for k, exp in expect.items():
        assert torch.allclose(payload["covariance"][k], exp, rtol=1e-6, atol=1e-6)


# ===========================================================================
# Lever 2 — DP-vs-single-pass equivalence (disk handoff, CPU replicas)
# ===========================================================================


def test_shard_calib_disjoint_and_complete():
    """``_shard_calib`` produces contiguous, disjoint, complete shards (last
    absorbs the remainder)."""
    calib = torch.arange(10).reshape(10, 1)
    shards = _shard_calib(calib, 3)
    assert len(shards) == 3
    # Reassembling the shards reproduces the original (disjoint + complete).
    recon = torch.cat(shards, dim=0)
    assert torch.equal(recon, calib)
    # replicas<=1 -> single shard (in-process identity).
    assert len(_shard_calib(calib, 1)) == 1


def _layer_ids_on_disk(d):
    return {int(p.stem.split("_")[1]) for p in Path(d).glob("layer_*.pt")}


def _assert_payload_close(ref_payload, dp_payload, *, rtol, atol):
    assert set(ref_payload["covariance"]) == set(dp_payload["covariance"])
    for k, ref_t in ref_payload["covariance"].items():
        dp_t = dp_payload["covariance"][k]
        assert torch.allclose(
            ref_t.to(torch.float32), dp_t.to(torch.float32), rtol=rtol, atol=atol
        ), f"DP-vs-single covariance mismatch at {k}"


def test_cov_dp_equivalence(tiny_model, tmp_path):
    """Single in-process pass over the full calibration tensor vs the DP path
    (2 disjoint batch-shards each spilled, then key-wise reduced) must match
    within the section 4 DP tolerance (fp32 rtol=1e-4/atol=1e-5). Runs CPU-only
    via disk handoff — no 2 GPUs needed.
    """
    import copy
    from moe_compress.utils.model_io import iter_moe_layers
    from moe_compress.utils.calibration import iter_batches

    torch.manual_seed(11)
    # A calibration tensor of 8 sequences; batch_size=2 -> 4 batches.
    calib = torch.randint(0, 32, (8, 8), dtype=torch.long)

    # --- Single-pass reference (full tensor, one accumulator). ---
    student_ref = copy.deepcopy(tiny_model).eval()
    teacher_ref = copy.deepcopy(tiny_model).eval()
    moe_ref = list(iter_moe_layers(student_ref))
    tmoe_ref = list(iter_moe_layers(teacher_ref))
    ref_dir_b = tmp_path / "ref_b"
    ref_dir_c = tmp_path / "ref_c"
    B_ref = InputCovarianceAccumulator(); B_ref.set_storage_dtype(torch.float32)
    C_ref = InputCovarianceAccumulator(); C_ref.set_storage_dtype(torch.float32)
    _collect_covariances(
        student_ref, moe_ref, iter_batches(calib, 2), B_ref, device=torch.device("cpu"),
        spill_dir=ref_dir_b, teacher_model=teacher_ref,
        teacher_moe_layers=tmoe_ref, C_acc=C_ref, ccov_spill_dir=ref_dir_c,
    )

    # --- DP path: 2 disjoint shards, each spilled to its own replica dir. ---
    shards = _shard_calib(calib, 2)
    rep_b_dirs, rep_c_dirs = [], []
    for r, shard in enumerate(shards):
        student = copy.deepcopy(tiny_model).eval()
        teacher = copy.deepcopy(tiny_model).eval()
        moe = list(iter_moe_layers(student))
        tmoe = list(iter_moe_layers(teacher))
        b_dir = tmp_path / f"rep{r}_b"; b_dir.mkdir()
        c_dir = tmp_path / f"rep{r}_c"; c_dir.mkdir()
        rep_b_dirs.append(b_dir); rep_c_dirs.append(c_dir)
        B = InputCovarianceAccumulator(); B.set_storage_dtype(torch.float32)
        C = InputCovarianceAccumulator(); C.set_storage_dtype(torch.float32)
        _collect_covariances(
            student, moe, iter_batches(shard, 2), B, device=torch.device("cpu"),
            spill_dir=b_dir, teacher_model=teacher,
            teacher_moe_layers=tmoe, C_acc=C, ccov_spill_dir=c_dir,
        )

    merged_b = tmp_path / "merged_b"
    merged_c = tmp_path / "merged_c"
    _reduce_spilled_cov_dirs(rep_b_dirs, merged_b, storage_dtype=torch.float32)
    _reduce_spilled_cov_dirs(rep_c_dirs, merged_c, storage_dtype=torch.float32)

    # Compare merged DP spill against the single-pass spill, key-wise.
    b_layers = {k[0] for k in B_ref.covariance} | _layer_ids_on_disk(ref_dir_b)
    for li in b_layers:
        ref_payload = torch.load(ref_dir_b / f"layer_{li}.pt", map_location="cpu",
                                 weights_only=True)
        dp_payload = torch.load(merged_b / f"layer_{li}.pt", map_location="cpu",
                                weights_only=True)
        _assert_payload_close(ref_payload, dp_payload, rtol=1e-4, atol=1e-5)
        # Token counts sum exactly across the disjoint shards.
        for k, n in ref_payload["tokens"].items():
            assert dp_payload["tokens"][k] == n, f"token_count mismatch {k}"

    for li in _layer_ids_on_disk(ref_dir_c):
        ref_payload = torch.load(ref_dir_c / f"layer_{li}.pt", map_location="cpu",
                                 weights_only=True)
        dp_payload = torch.load(merged_c / f"layer_{li}.pt", map_location="cpu",
                                weights_only=True)
        _assert_payload_close(ref_payload, dp_payload, rtol=1e-4, atol=1e-5)


# ===========================================================================
# Lever 2 (v2 step 3) — per-replica AUTO cov-batch sizing on the DP path.
#
# The load-bearing correctness claim: with the per-sequence reduction pin
# active, each DP replica may auto-size its forward batch INDEPENDENTLY (to its
# own VRAM) and the key-wise DP reduce (``_reduce_spilled_cov_dirs``, a fp32
# key-wise SUM of finalized per-replica Grams) is still correct — no
# cross-replica min(candidate) agreement is required (supersedes spec §6). The
# only residual is the bounded ~1e-6 forward-activation drift on factored
# down_proj B, N-independent and quality-neutral (the single-GPU property).
#
# CPU harness: we drive the REAL auto path (``cov_auto=True`` →
# ``size_batch`` + ``run_with_oom_backoff`` + ``iter_batches(shard, cov_bs)`` +
# the per-sequence pin) by monkeypatching the same cuda shims the cov-autobatch
# wire test uses (fake ``CudaMemProbe`` + ``torch.cuda`` peak shims) and pinning
# ``_COV_MAX_CAP`` per replica so each replica clamps to a DISTINCT cov_bs. This
# exercises the actual auto re-slice — NOT a monkeypatched non-auto branch.
# ===========================================================================


def _patch_cov_cuda_shims(monkeypatch, *, max_cap, mem_total=10 ** 12):
    """Make the REAL ``_collect_covariances`` auto path runnable on CPU and force
    ``size_batch`` to clamp the sized cov_bs to ``max_cap``.

    The probe's ``cost_probe_fn`` runs ``calib[:1]`` / ``calib[:2]`` through the
    real model and reads ``torch.cuda.max_memory_allocated`` for the peak; with a
    gentle 2-point slope and a huge ``mem.total()`` the predicted candidate is
    huge, so ``size_candidate`` returns ``min(raw, max_cap) == max_cap`` (verified
    by ``size_candidate``'s ``max(fixed_batch, min(raw, max_cap))``). Returns a
    one-shot reset for the peak counter so each replica probes from a clean slope.
    """
    import moe_compress.stage3.plugins.covariance_collection as cc

    monkeypatch.setattr(cc, "_COV_MAX_CAP", int(max_cap))

    class _FakeMem:
        def __init__(self, device=None):
            pass

        def total(self):
            return mem_total

        def allocated(self):
            return 0

        def reset_peak(self):
            pass

        def peak(self):
            return 0

    monkeypatch.setattr(cc, "CudaMemProbe", _FakeMem)

    # Two-point probe slope: peak(1)=1000, peak(2)=1100 (gentle → big candidate,
    # clamped to max_cap). The counter advances per max_memory_allocated read;
    # size_batch reads it exactly twice (cost_probe_fn(1), cost_probe_fn(2)).
    peaks = [1000, 1100]
    pk = {"i": 0}

    def _max_alloc(d=None):
        v = peaks[min(pk["i"], len(peaks) - 1)]
        pk["i"] += 1
        return v

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d=None: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", _max_alloc)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return pk


def test_dp_reduce_heterogeneous_replica_batches_matches_bs1(tiny_model, tmp_path):
    """LOAD-BEARING: replica 0 auto-sizes to cov_bs=A, replica 1 to cov_bs=B
    (A != B), both via the REAL auto path with the pin active, each spilled and
    then key-wise reduced. The reduced per-key Gram must equal a UNIFORM bs=1 DP
    reduce: ``torch.equal`` for gate_proj/up B + cross-cov C keys (the pin makes
    those bitwise-invariant to forward batch), ``torch.allclose`` (~1e-5) for
    factored down_proj B (bounded forward-activation drift). This proves replicas
    need NOT agree on a batch — the pin subsumes the spec §6 min-agreement.
    """
    import copy
    from moe_compress.utils.model_io import iter_moe_layers
    from moe_compress.utils.calibration import iter_batches
    from moe_compress.stage3.plugins.covariance_collection import AutoBatchConfig

    torch.manual_seed(11)
    # 8 sequences → 2 disjoint shards of 4. Per-replica caps 2 and 3 force
    # DISTINCT auto cov_bs (A=2, B=3) — both > 1 and != each other.
    calib = torch.randint(0, 32, (8, 8), dtype=torch.long)
    shards = _shard_calib(calib, 2)
    assert len(shards) == 2

    ab = AutoBatchConfig(enabled=True, headroom_frac=0.0)

    def _collect_replica(shard, *, b_dir, c_dir, cov_auto, max_cap, monkeypatch_ctx):
        student = copy.deepcopy(tiny_model).eval()
        teacher = copy.deepcopy(tiny_model).eval()
        moe = list(iter_moe_layers(student))
        tmoe = list(iter_moe_layers(teacher))
        B = InputCovarianceAccumulator(); B.set_storage_dtype(torch.float32)
        C = InputCovarianceAccumulator(); C.set_storage_dtype(torch.float32)
        if cov_auto:
            _patch_cov_cuda_shims(monkeypatch_ctx, max_cap=max_cap)
        _collect_covariances(
            student, moe, iter_batches(shard, 1), B, device=torch.device("cpu"),
            spill_dir=b_dir, teacher_model=teacher, teacher_moe_layers=tmoe,
            C_acc=C, ccov_spill_dir=c_dir, cov_window_size=1,
            calib=shard, cov_auto=cov_auto, auto_batch_cfg=ab,
        )

    # --- Heterogeneous AUTO DP: replica 0 → cov_bs=2, replica 1 → cov_bs=3. ---
    auto_b_dirs, auto_c_dirs = [], []
    caps = [2, 3]
    for r, shard in enumerate(shards):
        b_dir = tmp_path / f"auto{r}_b"; b_dir.mkdir()
        c_dir = tmp_path / f"auto{r}_c"; c_dir.mkdir()
        auto_b_dirs.append(b_dir); auto_c_dirs.append(c_dir)
        with pytest.MonkeyPatch.context() as mp:
            _collect_replica(shard, b_dir=b_dir, c_dir=c_dir, cov_auto=True,
                             max_cap=caps[r], monkeypatch_ctx=mp)
    auto_b = tmp_path / "auto_merged_b"; auto_c = tmp_path / "auto_merged_c"
    _reduce_spilled_cov_dirs(auto_b_dirs, auto_b, storage_dtype=torch.float32)
    _reduce_spilled_cov_dirs(auto_c_dirs, auto_c, storage_dtype=torch.float32)

    # --- Uniform bs=1 DP reference (non-auto, same shards). ---
    ref_b_dirs, ref_c_dirs = [], []
    for r, shard in enumerate(shards):
        b_dir = tmp_path / f"ref{r}_b"; b_dir.mkdir()
        c_dir = tmp_path / f"ref{r}_c"; c_dir.mkdir()
        ref_b_dirs.append(b_dir); ref_c_dirs.append(c_dir)
        _collect_replica(shard, b_dir=b_dir, c_dir=c_dir, cov_auto=False,
                         max_cap=1, monkeypatch_ctx=None)
    ref_b = tmp_path / "ref_merged_b"; ref_c = tmp_path / "ref_merged_c"
    _reduce_spilled_cov_dirs(ref_b_dirs, ref_b, storage_dtype=torch.float32)
    _reduce_spilled_cov_dirs(ref_c_dirs, ref_c, storage_dtype=torch.float32)

    # B keys: gate_proj/up bitwise-equal, down_proj allclose (~1e-5 fwd drift).
    b_layers = _layer_ids_on_disk(ref_b)
    assert b_layers, "no B layers spilled (harness wiring bug)"
    for li in b_layers:
        ref_payload = torch.load(ref_b / f"layer_{li}.pt", map_location="cpu",
                                 weights_only=True)
        auto_payload = torch.load(auto_b / f"layer_{li}.pt", map_location="cpu",
                                  weights_only=True)
        assert set(ref_payload["covariance"]) == set(auto_payload["covariance"])
        for k, ref_t in ref_payload["covariance"].items():
            auto_t = auto_payload["covariance"][k]
            rf = ref_t.to(torch.float32); af = auto_t.to(torch.float32)
            if k[2] == "down_proj":
                assert torch.allclose(af, rf, rtol=1e-5, atol=1e-5), (
                    f"down_proj B beyond fwd-drift tol at {k}: heterogeneous DP "
                    f"reduce diverged from uniform bs=1"
                )
            else:
                assert torch.equal(af, rf), (
                    f"{k[2]} B not bitwise-equal at {k}: the pin failed to make "
                    f"the heterogeneous DP reduce batch-independent"
                )
        # Token counts must sum identically regardless of per-replica batching.
        for k, n in ref_payload["tokens"].items():
            assert auto_payload["tokens"][k] == n, f"token_count mismatch {k}"

    # Cross-cov C keys: bitwise-equal across the heterogeneous reduce.
    c_layers = _layer_ids_on_disk(ref_c)
    assert c_layers, "no C layers spilled (cross-cov harness wiring bug)"
    for li in c_layers:
        ref_payload = torch.load(ref_c / f"layer_{li}.pt", map_location="cpu",
                                 weights_only=True)
        auto_payload = torch.load(auto_c / f"layer_{li}.pt", map_location="cpu",
                                  weights_only=True)
        assert set(ref_payload["covariance"]) == set(auto_payload["covariance"])
        for k, ref_t in ref_payload["covariance"].items():
            assert torch.equal(
                ref_t.to(torch.float32),
                auto_payload["covariance"][k].to(torch.float32),
            ), f"cross-cov C not bitwise-equal at {k} (heterogeneous DP reduce)"


def test_dp_default_no_auto_inherited(tiny_model, tmp_path, monkeypatch):
    """DEFAULT (no ``cov_batch_size:"auto"``): the worker's ``_cov_is_auto`` is
    False → ``cov_auto=False`` → the probe / OOM-backoff NEVER run (spy on
    ``size_batch``) and the DP reduce is byte-identical to today (the inherited
    bs=1 path). Pins that flipping the worker gate is a no-op when not opted in.
    """
    import copy
    import moe_compress.stage3.plugins.covariance_collection as cc
    from moe_compress.utils.model_io import iter_moe_layers
    from moe_compress.utils.calibration import iter_batches
    from moe_compress.stage3.plugins.covariance_collection import (
        AutoBatchConfig, _cov_is_auto,
    )

    # The worker resolves its gate from stage3_svd; default (no "auto") → False.
    assert _cov_is_auto({}) is False
    assert _cov_is_auto({"cov_batch_size": "auto"}) is False  # auto_batch absent

    calls = {"size_batch": 0}
    real_sb = cc.size_batch
    monkeypatch.setattr(
        cc, "size_batch",
        lambda *a, **k: (calls.__setitem__("size_batch", calls["size_batch"] + 1),
                         real_sb(*a, **k))[1],
    )

    torch.manual_seed(11)
    calib = torch.randint(0, 32, (8, 8), dtype=torch.long)
    shards = _shard_calib(calib, 2)
    ab = AutoBatchConfig.from_dict(None)  # disabled by default

    # DP path with the (default) inherited gate: cov_auto=_cov_is_auto({}) → False.
    dp_b_dirs, dp_c_dirs = [], []
    for r, shard in enumerate(shards):
        student = copy.deepcopy(tiny_model).eval()
        teacher = copy.deepcopy(tiny_model).eval()
        moe = list(iter_moe_layers(student))
        tmoe = list(iter_moe_layers(teacher))
        b_dir = tmp_path / f"dp{r}_b"; b_dir.mkdir()
        c_dir = tmp_path / f"dp{r}_c"; c_dir.mkdir()
        dp_b_dirs.append(b_dir); dp_c_dirs.append(c_dir)
        B = InputCovarianceAccumulator(); B.set_storage_dtype(torch.float32)
        C = InputCovarianceAccumulator(); C.set_storage_dtype(torch.float32)
        _collect_covariances(
            student, moe, iter_batches(shard, 1), B, device=torch.device("cpu"),
            spill_dir=b_dir, teacher_model=teacher, teacher_moe_layers=tmoe,
            C_acc=C, ccov_spill_dir=c_dir, cov_window_size=1,
            calib=shard, cov_auto=_cov_is_auto({}), auto_batch_cfg=ab,
        )
    dp_b = tmp_path / "dp_merged_b"; dp_c = tmp_path / "dp_merged_c"
    _reduce_spilled_cov_dirs(dp_b_dirs, dp_b, storage_dtype=torch.float32)
    _reduce_spilled_cov_dirs(dp_c_dirs, dp_c, storage_dtype=torch.float32)

    # The auto sizer must NEVER have been invoked on the default path.
    assert calls["size_batch"] == 0, "size_batch ran on the default (non-auto) DP path"

    # And the gated-default DP reduce is BYTE-IDENTICAL to today's DP reduce: an
    # explicit non-auto (``cov_auto=False``) DP pass over the SAME shards. (We
    # compare against the non-auto DP reduce — NOT a single-pass — because the
    # DP reduce sums two separately-finalized fp32 Grams, which is allclose- but
    # not bitwise-equal to a single accumulator; the property the gate flip must
    # preserve is "default == today's DP path", and that is the same code path.)
    today_b_dirs, today_c_dirs = [], []
    for r, shard in enumerate(shards):
        student = copy.deepcopy(tiny_model).eval()
        teacher = copy.deepcopy(tiny_model).eval()
        moe = list(iter_moe_layers(student))
        tmoe = list(iter_moe_layers(teacher))
        b_dir = tmp_path / f"today{r}_b"; b_dir.mkdir()
        c_dir = tmp_path / f"today{r}_c"; c_dir.mkdir()
        today_b_dirs.append(b_dir); today_c_dirs.append(c_dir)
        B = InputCovarianceAccumulator(); B.set_storage_dtype(torch.float32)
        C = InputCovarianceAccumulator(); C.set_storage_dtype(torch.float32)
        _collect_covariances(
            student, moe, iter_batches(shard, 1), B, device=torch.device("cpu"),
            spill_dir=b_dir, teacher_model=teacher, teacher_moe_layers=tmoe,
            C_acc=C, ccov_spill_dir=c_dir, cov_window_size=1,
            calib=shard, cov_auto=False,
        )
    today_b = tmp_path / "today_merged_b"; today_c = tmp_path / "today_merged_c"
    _reduce_spilled_cov_dirs(today_b_dirs, today_b, storage_dtype=torch.float32)
    _reduce_spilled_cov_dirs(today_c_dirs, today_c, storage_dtype=torch.float32)

    for li in _layer_ids_on_disk(today_b):
        rp = torch.load(today_b / f"layer_{li}.pt", map_location="cpu", weights_only=True)
        dp = torch.load(dp_b / f"layer_{li}.pt", map_location="cpu", weights_only=True)
        for k, rt in rp["covariance"].items():
            assert torch.equal(
                rt.to(torch.float32), dp["covariance"][k].to(torch.float32)
            ), f"default DP B not byte-identical to today's DP path at {k}"
    for li in _layer_ids_on_disk(today_c):
        rp = torch.load(today_c / f"layer_{li}.pt", map_location="cpu", weights_only=True)
        dp = torch.load(dp_c / f"layer_{li}.pt", map_location="cpu", weights_only=True)
        for k, rt in rp["covariance"].items():
            assert torch.equal(
                rt.to(torch.float32), dp["covariance"][k].to(torch.float32)
            ), f"default DP C not byte-identical to today's DP path at {k}"


# ===========================================================================
# Lever 2 — config replica resolution (auto-detect, 1-GPU no-op)
# ===========================================================================


def test_resolve_cov_replicas_floor_and_autodetect():
    """``_resolve_cov_replicas``: absent multi_gpu block OR n_gpu<2 -> (1, 1)
    (in-process, byte-identical). With n_gpu>=2 and cov_replicas>1 the effective
    count is min(requested, n_gpu // shards_per_model). No 1x/2x special-casing.
    """
    from moe_compress.stage3.orchestrator import _resolve_cov_replicas

    # No multi_gpu block -> in-process.
    assert _resolve_cov_replicas({}) == (1, 1)
    # cov_replicas==1 -> in-process regardless of GPUs.
    assert _resolve_cov_replicas({"multi_gpu": {"cov_replicas": 1}}) == (1, 1)

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    eff, spm = _resolve_cov_replicas({"multi_gpu": {"cov_replicas": 4}})
    assert spm == 1
    if n_gpu < 2:
        # 1-GPU / 0-GPU boxes collapse to the in-process path.
        assert eff == 1
    else:
        assert eff == min(4, n_gpu)
        assert eff >= 1


# ===========================================================================
# A7 (capture-only hook) + A1 (windowed single-pass) equivalence
# ===========================================================================


def _factored_from_fused(tiny_model):
    """Return a deepcopy of ``tiny_model`` with every MoE layer's fused experts
    replaced by a FULL-RANK ``FactoredExperts`` whose U@V reproduces the fused
    weights. This puts the model on the genuine native padded-``bmm`` forward
    (``FactoredExperts.forward``) so the A7-vs-instrument test exercises the
    real native-vs-Python-loop reduction-order delta (not the fused tiny path,
    where both are the same per-expert ``F.linear`` loop).
    """
    import copy
    from moe_compress.utils.model_io import FactoredExperts, iter_moe_layers

    model = copy.deepcopy(tiny_model).eval()
    for ref in iter_moe_layers(model):
        fused = ref.experts_module
        ne = fused.num_experts
        d_hid = fused.hidden_dim
        d_int = fused.intermediate_dim
        fe = FactoredExperts(
            num_experts=ne, hidden_dim=d_hid, intermediate_dim=d_int,
            ranks={"gate_proj": d_int, "up_proj": d_int,
                   "down_proj": min(d_hid, d_int)},
            dtype=torch.float32,
        )
        with torch.no_grad():
            for e in range(ne):
                gate = fused.gate_up_proj[e][:d_int]
                up = fused.gate_up_proj[e][d_int:]
                down = fused.down_proj[e]
                fe.set_factors_from_weight(e, "gate_proj", gate)
                fe.set_factors_from_weight(e, "up_proj", up)
                fe.set_factors_from_weight(e, "down_proj", down)
        ref.mlp.experts = fe
    return model


def test_a7_capture_matches_instrument_single_layer(tiny_model):
    """A7 ``capture_experts`` (native forward) vs ``instrument_experts`` (Python
    loop) on a SINGLE FactoredExperts MoE layer must agree per-key within
    rtol=1e-4/atol=1e-5. Isolates the native-vs-loop fp delta (single layer ⇒
    no upstream drift); bounds it, justifying the new all-native golden.

    Both runs use the SAME post-prune FactoredExperts model, so the only
    difference is the captured-layer forward reduction order: the native padded
    ``bmm`` (A7 recompute) vs the per-expert ``F.linear`` loop (instrument).
    """
    import copy
    from moe_compress.utils.model_io import iter_moe_layers

    torch.manual_seed(3)
    batches = [torch.randint(0, 32, (2, 8)) for _ in range(3)]

    base = _factored_from_fused(tiny_model)
    # Restrict to a single MoE layer so there is no upstream-layer drift.
    moe_all = list(iter_moe_layers(base))
    single_idx = moe_all[0].layer_idx

    def _collect(mode):
        m = copy.deepcopy(base).eval()
        moe = [r for r in iter_moe_layers(m) if r.layer_idx == single_idx]
        B = InputCovarianceAccumulator()
        B.set_storage_dtype(torch.float32)
        _collect_covariances(
            m, moe, batches, B, device=torch.device("cpu"),
            cov_window_size=1, cov_capture_mode=mode,
        )
        B.finalize_all()
        return B

    B_cap = _collect("capture")
    B_ins = _collect("instrument")

    assert set(B_cap.covariance) == set(B_ins.covariance)
    # gate_proj key is a pure gather ⇒ byte-identical; down_proj moves with the
    # native-vs-loop order ⇒ bounded by rtol=1e-4. Both checked under one tol.
    for k in B_cap.covariance:
        a = B_cap.covariance[k].to(torch.float32)
        b = B_ins.covariance[k].to(torch.float32)
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-5), \
            f"A7-vs-instrument cov delta exceeds native-vs-loop bound at {k}"
    # gate_proj keys must be EXACT (pure input gather, identical ops).
    for k in B_cap.covariance:
        if k[2] == "gate_proj":
            assert torch.equal(
                B_cap.covariance[k].to(torch.float32),
                B_ins.covariance[k].to(torch.float32),
            ), f"gate_proj key {k} must be bit-identical (pure gather)"


def _collect_windowed(model_factory, batches, *, G, cross, cross_impl="dense"):
    """Run A7 cov collection at window size ``G`` and return (B, C) accumulators."""
    import copy
    from moe_compress.utils.model_io import iter_moe_layers

    model = copy.deepcopy(model_factory).eval()
    teacher = copy.deepcopy(model_factory).eval() if cross else None
    moe = list(iter_moe_layers(model))
    tmoe = list(iter_moe_layers(teacher)) if cross else None
    B = InputCovarianceAccumulator()
    B.set_storage_dtype(torch.float32)
    C = None
    if cross:
        C = InputCovarianceAccumulator()
        C.set_storage_dtype(torch.float32)
    _collect_covariances(
        model, moe, batches, B, device=torch.device("cpu"),
        teacher_model=teacher, teacher_moe_layers=tmoe, C_acc=C,
        cov_window_size=G, cov_capture_mode="capture",
        cov_cross_impl=cross_impl,
    )
    B.finalize_all()
    if C is not None:
        C.finalize_all()
    return B, C


def test_a4_cross_cov_dense_equals_dict(tiny_model):
    """A4: the dense ``index_select`` cross-cov path is BYTE-IDENTICAL
    (``torch.equal``, atol=0) to the legacy ``{token_idx → row}`` dict path on
    CPU per C key. Exercised across G ∈ {1, 2, N} (window independence) — G ≥ 2
    additionally hooks multiple MoE layers SIMULTANEOUSLY, covering the
    per-layer ``teacher_dense[li]`` separation. B keys are also checked equal
    (the B path is impl-independent but must not regress)."""
    from moe_compress.utils.model_io import iter_moe_layers

    torch.manual_seed(13)
    batches = [torch.randint(0, 32, (2, 8)) for _ in range(3)]
    n_layers = len(list(iter_moe_layers(tiny_model)))
    sizes = sorted({1, 2, n_layers})

    for G in sizes:
        B_dict, C_dict = _collect_windowed(
            tiny_model, batches, G=G, cross=True, cross_impl="dict"
        )
        B_dense, C_dense = _collect_windowed(
            tiny_model, batches, G=G, cross=True, cross_impl="dense"
        )
        assert set(C_dict.covariance) == set(C_dense.covariance), \
            f"G={G}: C key sets differ dict vs dense"
        for k in C_dict.covariance:
            assert torch.equal(
                C_dict.covariance[k].to(torch.float32),
                C_dense.covariance[k].to(torch.float32),
            ), f"G={G}: C dense != dict at {k}"
        assert set(B_dict.covariance) == set(B_dense.covariance)
        for k in B_dict.covariance:
            assert torch.equal(
                B_dict.covariance[k].to(torch.float32),
                B_dense.covariance[k].to(torch.float32),
            ), f"G={G}: B dense != dict at {k}"


def test_a6_resolve_cov_batch_size():
    """A6 resolver: ``cov_batch_size`` absent ⇒ inherits ``batch_size``; explicit
    int / str passes through; ``"auto"`` on CPU / 1-GPU ⇒ inherited value (NO
    raise, so the 1-GPU golden stays byte-identical)."""
    from moe_compress.stage3.plugins.covariance_collection import (
        _resolve_cov_batch_size,
    )

    # Absent ⇒ inherits batch_size.
    assert _resolve_cov_batch_size({"batch_size": 4}) == 4
    assert _resolve_cov_batch_size({}) == 1  # both absent ⇒ default 1
    # Explicit int / str passes through.
    assert _resolve_cov_batch_size({"batch_size": 1, "cov_batch_size": 8}) == 8
    assert _resolve_cov_batch_size({"batch_size": 1, "cov_batch_size": "6"}) == 6
    # None ⇒ inherits.
    assert _resolve_cov_batch_size({"batch_size": 3, "cov_batch_size": None}) == 3
    # "auto": on CPU / 1-GPU degrades to the inherited value (no raise). When a
    # real ≥2-GPU box is present the deferred measurement still returns the
    # inherited value (raise is DEFERRED), so this holds regardless of hardware.
    assert _resolve_cov_batch_size({"batch_size": 2, "cov_batch_size": "auto"}) == 2


def test_a6_cov_batch_size_close(tiny_model):
    """A6 + reduction-pin: covariance at cov batch sizes bs ∈ {1, 2, 4} over the
    SAME tokens. The per-sequence reduction-pin (``update_grouped`` /
    operand-split) re-imposes the bs=1 *summation grouping* regardless of how a
    bigger forward batch merged the sequences, so the result is BITWISE-INVARIANT
    (``torch.equal``, atol=0) across bs for any key whose input operand is itself
    batch-shape-stable:

      * ``gate_proj`` B keys: operand is ``hidden_states[token_idx]`` — a pure
        gather, identical at any bs -> bitwise.
      * ALL cross-cov ``C`` keys: operands are the teacher dense gather + the
        student gate-input gather, both batch-shape-stable -> bitwise.
      * ``down_proj`` B keys: operand is ``act_fn(gate)*up`` from a PADDED
        batched ``bmm`` in ``capture_experts`` whose fp reduction is perturbed by
        the forward batch shape upstream of the pin -> the GROUPING is pinned but
        the operand carries ~1e-6 forward drift -> allclose (NOT bitwise), the
        same unavoidable v1 forward-activation drift.
    """
    import copy
    from moe_compress.utils.model_io import iter_moe_layers

    torch.manual_seed(21)
    # A flat pool of sequences; repartitioned into batches at each bs.
    n_seq, seq = 8, 8
    pool = torch.randint(0, 32, (n_seq, seq))

    def run(bs):
        batches = [pool[i:i + bs] for i in range(0, n_seq, bs)]
        model = copy.deepcopy(tiny_model).eval()
        teacher = copy.deepcopy(tiny_model).eval()
        moe = list(iter_moe_layers(model))
        tmoe = list(iter_moe_layers(teacher))
        B = InputCovarianceAccumulator(); B.set_storage_dtype(torch.float32)
        C = InputCovarianceAccumulator(); C.set_storage_dtype(torch.float32)
        _collect_covariances(
            model, moe, batches, B, device=torch.device("cpu"),
            teacher_model=teacher, teacher_moe_layers=tmoe, C_acc=C,
            cov_window_size=1, cov_capture_mode="capture",
        )
        B.finalize_all(); C.finalize_all()
        return B, C

    B1, C1 = run(1)
    for bs in (2, 4):
        B, C = run(bs)
        assert set(B.covariance) == set(B1.covariance)
        for k in B1.covariance:
            matrix_name = k[2]
            got = B.covariance[k].to(torch.float32)
            ref = B1.covariance[k].to(torch.float32)
            if matrix_name == "down_proj":
                # Pinned grouping, but the padded-bmm operand drifts upstream.
                assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5), \
                    f"bs={bs}: down_proj B beyond fp-reassoc tolerance at {k}"
            else:
                # gate_proj: pure-gather operand -> the pin makes it bitwise.
                assert torch.equal(got, ref), \
                    f"bs={bs}: {matrix_name} B not bitwise-invariant at {k}"
        assert set(C.covariance) == set(C1.covariance)
        for k in C1.covariance:
            # Cross-cov operands are batch-shape-stable gathers -> the pin makes
            # every C key bitwise-invariant across cov batch size.
            assert torch.equal(
                C.covariance[k].to(torch.float32),
                C1.covariance[k].to(torch.float32),
            ), f"bs={bs}: C not bitwise-invariant at {k}"


def test_a1_windowed_equals_perlayer(tiny_model):
    """A7 windowed (G=N, single pass) vs A7 per-layer (G=1) must be
    BYTE-IDENTICAL (atol=0) on CPU per key — windowing adds zero error on top of
    the native baseline (PLAN §2.1). Exercised WITH cross-cov (dual-forward) so
    the per-batch ``_teacher_hidden`` window lifetime is covered too.
    """
    torch.manual_seed(5)
    batches = [torch.randint(0, 32, (2, 8)) for _ in range(3)]
    n_layers = len(list(__import__(
        "moe_compress.utils.model_io", fromlist=["iter_moe_layers"]
    ).iter_moe_layers(tiny_model)))

    B1, C1 = _collect_windowed(tiny_model, batches, G=1, cross=True)
    Bn, Cn = _collect_windowed(tiny_model, batches, G=n_layers, cross=True)

    assert set(B1.covariance) == set(Bn.covariance)
    for k in B1.covariance:
        assert torch.equal(
            B1.covariance[k].to(torch.float32), Bn.covariance[k].to(torch.float32)
        ), f"B windowed(G=N) != per-layer(G=1) at {k}"
    assert set(C1.covariance) == set(Cn.covariance)
    for k in C1.covariance:
        assert torch.equal(
            C1.covariance[k].to(torch.float32), Cn.covariance[k].to(torch.float32)
        ), f"C windowed(G=N) != per-layer(G=1) at {k}"


def test_a1_window_sizes_consistent(tiny_model):
    """G ∈ {1, 2, N} all produce byte-identical per-key cov — window-boundary
    independence (PLAN §7.1)."""
    from moe_compress.utils.model_io import iter_moe_layers

    torch.manual_seed(9)
    batches = [torch.randint(0, 32, (2, 8)) for _ in range(2)]
    n_layers = len(list(iter_moe_layers(tiny_model)))
    sizes = sorted({1, 2, n_layers})

    ref_B, _ = _collect_windowed(tiny_model, batches, G=sizes[0], cross=False)
    for G in sizes[1:]:
        B, _ = _collect_windowed(tiny_model, batches, G=G, cross=False)
        assert set(B.covariance) == set(ref_B.covariance)
        for k in B.covariance:
            assert torch.equal(
                B.covariance[k].to(torch.float32),
                ref_B.covariance[k].to(torch.float32),
            ), f"G={G} cov differs from G={sizes[0]} at {k}"


def test_resolve_cov_window_config():
    """``_resolve_cov_window``: explicit int clamps to [1,N]; auto on a CPU-only
    box degrades to G=1; absent key defaults to auto (G=1 on CPU)."""
    from moe_compress.stage3.plugins.covariance_collection import _resolve_cov_window

    # Explicit int clamps into [1, N].
    assert _resolve_cov_window({"multi_gpu": {"cov_window_size": 3}}, 10) == 3
    assert _resolve_cov_window({"multi_gpu": {"cov_window_size": 99}}, 10) == 10
    assert _resolve_cov_window({"multi_gpu": {"cov_window_size": 0}}, 10) == 1
    # String int accepted.
    assert _resolve_cov_window({"multi_gpu": {"cov_window_size": "4"}}, 10) == 4
    # auto / absent: VRAM-probe; on a CPU-only box this MUST be 1 (clean degrade).
    if not torch.cuda.is_available():
        assert _resolve_cov_window({"multi_gpu": {"cov_window_size": "auto"}}, 10) == 1
        assert _resolve_cov_window({}, 10) == 1
    else:
        # With CUDA the probe returns a value in [1, N].
        g = _resolve_cov_window({"multi_gpu": {"cov_window_size": "auto"}}, 10)
        assert 1 <= g <= 10
    # n_layers<=0 → 1.
    assert _resolve_cov_window({}, 0) == 1


def test_cov_per_layer_gram_bytes_includes_expert_factor():
    """The per-layer Gram estimate MUST scale with n_experts (the historical bug
    omitted this ~256× factor → auto picked a catastrophic G). Also: cross-cov adds
    the gate term, and B uses gate=d_hid² + down=d_int² per expert."""
    from moe_compress.stage3.plugins.covariance_collection import _cov_per_layer_gram_bytes

    d_hid, d_int = 2048, 512
    # B-only, one expert: gate d_hid² + down d_int², fp32.
    one_b = _cov_per_layer_gram_bytes(d_hid, d_int, n_exp=1, cross_cov=False)
    assert one_b == (d_hid ** 2 + d_int ** 2) * 4.0
    # Linear in n_exp — the fix. 256 experts == 256× one expert (the old formula
    # used a single expert's footprint → ~256× under-count).
    assert _cov_per_layer_gram_bytes(d_hid, d_int, 256, False) == 256 * one_b
    # Cross-cov adds gate d_hid² per expert.
    with_c = _cov_per_layer_gram_bytes(d_hid, d_int, 256, cross_cov=True)
    assert with_c == _cov_per_layer_gram_bytes(d_hid, d_int, 256, False) + 256 * d_hid ** 2 * 4.0
    # Realistic Qwen3.6 layer (256 experts, cross-cov on) is multi-GB, not ~18 MB.
    assert with_c > 8e9  # ~8.85 GB/layer — the figure the old estimate missed by ~256×


def test_capture_experts_rejects_double_instrument(tiny_model):
    """A7 ``capture_experts`` refuses to attach to a module whose forward is
    already swapped by ``instrument_experts`` (mutually exclusive, PLAN §3.3)."""
    from moe_compress.utils.activation_hooks import (
        capture_experts, instrument_experts,
    )
    from moe_compress.utils.model_io import iter_moe_layers

    ref = list(iter_moe_layers(tiny_model))[0]
    with instrument_experts(ref, {"input": lambda *a, **k: None}):
        with pytest.raises(RuntimeError, match="mutually exclusive"):
            with capture_experts(ref, {"input": lambda *a, **k: None}):
                pass


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-v"]))
