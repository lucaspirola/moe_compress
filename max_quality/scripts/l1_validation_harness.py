"""L1 validation harness -- tests the 3 architectural blockers that
scope-cut L1 of the calibration-v2 campaign.

VALIDATION HARNESS -- NOT PART OF THE PRODUCTION CODEBASE.

Background
==========
L1 was scope-cut (commit 9647362) on the basis of three suspected
architectural blockers that we lacked GPU access to verify. This
harness probes each blocker on a TINY Qwen3-MoE model (PrimeIntellect/
qwen3-moe-tiny, 670 M params, 16 experts, top-k 4) so a single
A10G-small ($1/hr) Job can render a verdict in 30-60 min.

The three tests
===============
1. ``test_update_weights_cuda_graph`` -- does an in-place ``.copy_()``
   into the vLLM model's expert weight tensor survive vLLM's captured
   CUDA graph? Verdict via timing: post-update forward time should
   match the warm (graph-replay) baseline, not the cold
   (graph-capture) baseline, AND the generated output must change.

2. ``test_per_expert_hook_mapping`` -- vLLM's patched ``expert_in``
   hook fires with ``(hidden_states, topk_ids)`` BEFORE per-expert
   dispatch. Stage 2's HF capture path hooks the per-expert
   ``gate_proj`` AFTER dispatch + gather. This test scatters vLLM's
   pre-dispatch hidden_states by topk_ids and compares the
   reconstruction to HF's per-expert ``gate_proj`` input on the same
   prompt. Verdict: max-abs-diff < tolerance per (layer, expert) cell.

3. ``test_weight_layout_adapter`` -- HF stores 16 separate
   ``nn.Linear`` experts per layer (each ``[I,H]`` for gate / up /
   down). vLLM stacks them into ``w13_weight[E, 2I, H]`` (gate ||
   up) and ``w2_weight[E, H, I]``. This test (a) translates HF -> vLLM
   stacked form and checks the translation matches vLLM's actually-
   loaded weights byte-for-byte, then (b) modifies one expert in HF,
   re-translates, pushes via direct ``.copy_()``, and verifies the
   model's generation output changed accordingly.

Results
=======
Writes ``/tmp/l1_validation_results.json``; the wrapper shell script
uploads it to the dataset ``pirola/l1-validation-results``.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

# Print env up front so the Job log captures it.
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] === L1 validation harness ===", flush=True)
print(f"  python   : {sys.version.split()[0]}", flush=True)
print(f"  torch    : {torch.__version__}", flush=True)
print(f"  cuda     : {torch.version.cuda}", flush=True)
print(f"  device   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
if torch.cuda.is_available():
    print(f"  vram     : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB", flush=True)

import vllm  # noqa: E402
print(f"  vllm     : {vllm.__version__}", flush=True)
from vllm import LLM, SamplingParams  # noqa: E402
import vllm.calibration_hooks as ch  # patched module  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# Model choice locked in: PrimeIntellect/qwen3-moe-tiny.
# - Architecture: Qwen3MoeForCausalLM (exact target of the patch)
# - 24 hidden layers (layer 0 = dense MLP, layers 1..23 = MoE)
# - 16 experts, top-k 4, hidden=1024, intermediate=2048, moe_intermediate=256
# - 670 M params float32 (~2.7 GB on disk) -> ~1.4 GB in bf16
# - Fits trivially in 24 GB with BOTH vLLM AND HF transformers loaded.
MODEL_ID = "PrimeIntellect/qwen3-moe-tiny"

# First MoE layer index (skip layer 0 which is a dense MLP).
# Verified from config: mlp_only_layers = [0].
MOE_LAYER_IDX = 1

# Tolerances. fp32 -> bf16 round-trip noise is generously bounded by 1e-2
# in absolute value at hidden_size=1024 magnitudes; tighter would be
# brittle. Translation-correctness (Test 3a) is a pure copy so 1e-4.
SCATTER_DIFF_TOL = 1e-2
TRANSLATION_DIFF_TOL = 1e-4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _vllm_model(llm: LLM):
    """Reach into the LLM to grab the live nn.Module on the driver worker.

    The exact attribute chain varies slightly across vLLM versions; this
    is the public-ish path for vllm 0.21.x with the default engine.
    """
    return llm.llm_engine.model_executor.driver_worker.model_runner.model


def _hf_moe_layer(hf_model, layer_idx: int):
    """Return the HF Qwen3MoeSparseMoeBlock at layer_idx."""
    return hf_model.model.layers[layer_idx].mlp


def _vllm_moe_layer(vllm_model, layer_idx: int):
    """Return the vLLM Qwen3MoeSparseMoeBlock at layer_idx."""
    return vllm_model.model.layers[layer_idx].mlp


def _time_generate(llm: LLM, prompt: str, sp: SamplingParams) -> tuple[float, str]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = llm.generate([prompt], sp, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    text = out[0].outputs[0].text
    return dt, text


# ---------------------------------------------------------------------------
# Test 1 -- update_weights survives CUDA graph
# ---------------------------------------------------------------------------


def test_update_weights_cuda_graph(llm: LLM) -> dict[str, Any]:
    """Timing + output verification that an in-place tensor write into
    the vLLM-loaded MoE expert weights propagates through the captured
    CUDA graph and changes generation output.

    Strategy:
      1. Warmup #1 -> cold (graph capture). Record time + output.
      2. Warmup #2 -> warm (graph replay). Record time + output.
      3. In-place zero of one expert of one MoE layer's w13_weight.
      4. Forward #3 -> verdict. If time ~ warm and output changed,
         in-place worked. If time ~ cold, graph re-captured. If output
         did NOT change, vLLM is reading from a different buffer than
         the one we wrote to (worst case for L1).
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] -- Test 1: update_weights + CUDA graph --", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=20, seed=0)
    prompt = "The quick brown fox jumps over"

    cold_time, cold_text = _time_generate(llm, prompt, sp)
    print(f"  cold forward : {cold_time*1000:.0f} ms | {cold_text!r}", flush=True)

    warm_time, warm_text = _time_generate(llm, prompt, sp)
    print(f"  warm forward : {warm_time*1000:.0f} ms | {warm_text!r}", flush=True)

    vllm_model = _vllm_model(llm)
    vllm_mlp = _vllm_moe_layer(vllm_model, MOE_LAYER_IDX)
    w13 = vllm_mlp.experts.w13_weight  # [E, 2*I, H]
    original_w13 = w13.detach().clone()

    # Aggressive but cleanly reversible: zero expert 0's gate half.
    intermediate = w13.shape[1] // 2  # gate || up convention
    update_ok = False
    update_error: str | None = None
    try:
        with torch.no_grad():
            w13[0, :intermediate, :].zero_()
        update_ok = True
    except Exception as e:  # pragma: no cover
        update_error = repr(e)
        print(f"  ! in-place write raised: {update_error}", flush=True)

    if not update_ok:
        return {
            "model": MODEL_ID,
            "cold_time_s": cold_time,
            "warm_time_s": warm_time,
            "post_update_time_s": None,
            "in_place_write_raised": True,
            "update_error": update_error,
            "output_changed_after_update": None,
            "verdict_in_place_works": False,
            "verdict_output_changed": False,
        }

    post_time, post_text = _time_generate(llm, prompt, sp)
    print(f"  post-update  : {post_time*1000:.0f} ms | {post_text!r}", flush=True)

    # Restore so subsequent tests run on the clean model.
    with torch.no_grad():
        w13.copy_(original_w13)
    restored_time, restored_text = _time_generate(llm, prompt, sp)
    print(f"  restored     : {restored_time*1000:.0f} ms | {restored_text!r}", flush=True)

    in_place_works = post_time < (cold_time + warm_time) / 2
    output_changed = post_text != warm_text

    return {
        "model": MODEL_ID,
        "cold_time_s": cold_time,
        "warm_time_s": warm_time,
        "post_update_time_s": post_time,
        "restored_time_s": restored_time,
        "cold_text": cold_text,
        "warm_text": warm_text,
        "post_update_text": post_text,
        "restored_text": restored_text,
        "in_place_write_raised": False,
        "verdict_in_place_works": in_place_works,
        "verdict_output_changed": output_changed,
        "verdict_full_pass": in_place_works and output_changed,
    }


# ---------------------------------------------------------------------------
# Test 2 -- per-expert hook signal mapping
# ---------------------------------------------------------------------------


def test_per_expert_hook_mapping(llm: LLM, hf_model, tokenizer) -> dict[str, Any]:
    """Verify that scatter-by-topk_ids on vLLM's pre-dispatch
    (hidden_states, topk_ids) reproduces HF's post-gather per-expert
    gate_proj input.
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] -- Test 2: per-expert hook signal mapping --", flush=True)

    # Ensure expert_in capture gate is on. Setting env after import
    # wouldn't take effect; flip the module attr directly. The dispatch
    # fast-path is a plain attribute read so this is sufficient.
    ch._CAPTURE_EXPERT = True

    # Drop any pre-existing callback for expert_in to ensure a clean slate.
    ch.register_callback("expert_in", None)

    vllm_captures: list[dict[str, Any]] = []

    def on_expert_in(layer_idx: int, hidden_states, topk_ids, **_kw):
        if layer_idx != MOE_LAYER_IDX:
            return
        vllm_captures.append({
            "layer_idx": int(layer_idx),
            "hidden_states": hidden_states.detach().to("cpu", torch.float32).clone(),
            "topk_ids": topk_ids.detach().to("cpu", torch.long).clone(),
        })

    ch.register_callback("expert_in", on_expert_in)

    hf_layer = _hf_moe_layer(hf_model, MOE_LAYER_IDX)
    n_experts = hf_model.config.num_experts
    hf_captures: dict[int, list[torch.Tensor]] = {e: [] for e in range(n_experts)}
    hf_hook_handles = []

    def make_hf_hook(expert_idx: int):
        def hook(_module, input_, _output):
            x = input_[0] if isinstance(input_, tuple) else input_
            hf_captures[expert_idx].append(x.detach().to("cpu", torch.float32).clone())
        return hook

    for e in range(n_experts):
        h = hf_layer.experts[e].gate_proj.register_forward_hook(make_hf_hook(e))
        hf_hook_handles.append(h)

    prompt = "The quick brown fox jumps over"
    sp = SamplingParams(temperature=0.0, max_tokens=1, seed=0)

    print("  running vLLM forward...", flush=True)
    _ = llm.generate([prompt], sp, use_tqdm=False)
    print(f"  captured vllm dispatches @ layer {MOE_LAYER_IDX}: {len(vllm_captures)}", flush=True)

    print("  running HF forward...", flush=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(hf_model.device)
    with torch.no_grad():
        _ = hf_model(input_ids)
    n_hf = sum(len(v) for v in hf_captures.values())
    print(f"  captured hf per-expert dispatches: {n_hf}", flush=True)

    for h in hf_hook_handles:
        h.remove()
    ch.register_callback("expert_in", None)

    if not vllm_captures:
        return {
            "model": MODEL_ID,
            "moe_layer_idx": MOE_LAYER_IDX,
            "vllm_capture_count": 0,
            "error": "no vllm expert_in callbacks fired; patched wheel may not be wiring the hook",
            "verdict_scatter_reconstruction_works": False,
        }

    # vLLM gets one prefill batch (single forward) -> use the first capture.
    cap = vllm_captures[0]
    hs = cap["hidden_states"]      # [n_tok, hidden]
    topk_ids = cap["topk_ids"]     # [n_tok, top_k]

    per_expert_diffs: dict[str, dict[str, Any]] = {}
    for e in range(n_experts):
        mask = (topk_ids == e).any(dim=-1)
        vllm_per_expert = hs[mask]  # [n_routed_e, hidden]

        hf_lst = hf_captures.get(e, [])
        hf_per_expert = hf_lst[0] if hf_lst else None

        rec: dict[str, Any] = {
            "vllm_n_tokens": int(vllm_per_expert.shape[0]),
            "hf_n_tokens": int(hf_per_expert.shape[0]) if hf_per_expert is not None else 0,
            "shape_match": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "compared": False,
        }
        if hf_per_expert is None or vllm_per_expert.numel() == 0:
            per_expert_diffs[f"L{MOE_LAYER_IDX}_E{e}"] = rec
            continue
        if vllm_per_expert.shape == hf_per_expert.shape:
            rec["shape_match"] = True
            # Sets may be unordered between the two stacks -- stable sort
            # by the leading 8 hidden dims before diffing.
            def _sort_rows(t: torch.Tensor) -> torch.Tensor:
                if t.shape[0] == 0:
                    return t
                key = t[:, : min(8, t.shape[1])]
                idx = torch.arange(t.shape[0])
                for c in range(key.shape[1] - 1, -1, -1):
                    idx = idx[torch.argsort(key[idx, c], stable=True)]
                return t[idx]
            v_sorted = _sort_rows(vllm_per_expert)
            h_sorted = _sort_rows(hf_per_expert)
            diff = (v_sorted - h_sorted).abs()
            rec["max_abs_diff"] = float(diff.max().item())
            rec["mean_abs_diff"] = float(diff.mean().item())
            rec["compared"] = True
        per_expert_diffs[f"L{MOE_LAYER_IDX}_E{e}"] = rec

    compared = [d["max_abs_diff"] for d in per_expert_diffs.values() if d["compared"]]
    overall_max = max(compared) if compared else None
    reconstruction_works = (
        bool(compared)
        and overall_max is not None
        and overall_max < SCATTER_DIFF_TOL
    )

    return {
        "model": MODEL_ID,
        "moe_layer_idx": MOE_LAYER_IDX,
        "vllm_capture_count": len(vllm_captures),
        "per_expert": per_expert_diffs,
        "max_diff_overall": overall_max,
        "tolerance": SCATTER_DIFF_TOL,
        "verdict_scatter_reconstruction_works": bool(reconstruction_works),
    }


# ---------------------------------------------------------------------------
# Test 3 -- HF -> vLLM weight layout adapter
# ---------------------------------------------------------------------------


def test_weight_layout_adapter(llm: LLM, hf_model, tokenizer) -> dict[str, Any]:
    """Translate HF per-expert {gate, up, down} weights into vLLM's
    stacked w13_weight/w2_weight layout, compare to vLLM's actual
    tensors, then modify one expert in HF, re-translate, push via
    in-place .copy_, and verify the model output changed.
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] -- Test 3: HF -> vLLM weight layout adapter --", flush=True)

    hf_layer = _hf_moe_layer(hf_model, MOE_LAYER_IDX)
    vllm_model = _vllm_model(llm)
    vllm_mlp = _vllm_moe_layer(vllm_model, MOE_LAYER_IDX)
    w13 = vllm_mlp.experts.w13_weight  # [E, 2*I, H]
    w2 = vllm_mlp.experts.w2_weight    # [E, H, I]

    n_experts = hf_model.config.num_experts
    moe_inter = hf_model.config.moe_intermediate_size

    print(f"  vllm w13 shape: {tuple(w13.shape)} dtype={w13.dtype}", flush=True)
    print(f"  vllm w2  shape: {tuple(w2.shape)}  dtype={w2.dtype}", flush=True)
    g0 = hf_layer.experts[0].gate_proj.weight
    u0 = hf_layer.experts[0].up_proj.weight
    d0 = hf_layer.experts[0].down_proj.weight
    print(f"  hf  gate[0]: {tuple(g0.shape)} | up[0]: {tuple(u0.shape)} | down[0]: {tuple(d0.shape)}", flush=True)

    # Assumed conventions:
    #   w13_weight[e] = concat([gate, up], dim=0) -> [2I, H]
    #   w2_weight[e]  = down                       -> [H, I]
    # If the order is up || gate instead we detect via diff and pick.
    def translate(hf_layer_ref, swap_gate_up: bool) -> tuple[torch.Tensor, torch.Tensor]:
        tw13 = torch.zeros_like(w13)
        tw2 = torch.zeros_like(w2)
        for e in range(n_experts):
            gate = hf_layer_ref.experts[e].gate_proj.weight.to(w13.dtype).to(w13.device)
            up = hf_layer_ref.experts[e].up_proj.weight.to(w13.dtype).to(w13.device)
            down = hf_layer_ref.experts[e].down_proj.weight.to(w2.dtype).to(w2.device)
            if swap_gate_up:
                tw13[e, :moe_inter, :].copy_(up)
                tw13[e, moe_inter:, :].copy_(gate)
            else:
                tw13[e, :moe_inter, :].copy_(gate)
                tw13[e, moe_inter:, :].copy_(up)
            tw2[e].copy_(down)
        return tw13, tw2

    tw13_a, tw2_a = translate(hf_layer, swap_gate_up=False)
    pre_diff_a_w13 = (tw13_a - w13).abs().max().item()
    pre_diff_a_w2 = (tw2_a - w2).abs().max().item()
    print(f"  gate||up    : w13 max-diff={pre_diff_a_w13:.3e}  w2 max-diff={pre_diff_a_w2:.3e}", flush=True)

    tw13_b, _tw2_b = translate(hf_layer, swap_gate_up=True)
    pre_diff_b_w13 = (tw13_b - w13).abs().max().item()
    print(f"  up||gate    : w13 max-diff={pre_diff_b_w13:.3e}  (w2 unchanged)", flush=True)

    if pre_diff_a_w13 <= pre_diff_b_w13:
        chosen_order = "gate||up"
        chosen_pre_diff_w13 = pre_diff_a_w13
        swap = False
    else:
        chosen_order = "up||gate"
        chosen_pre_diff_w13 = pre_diff_b_w13
        swap = True
    chosen_pre_diff_w2 = pre_diff_a_w2
    translation_correct = (
        chosen_pre_diff_w13 < TRANSLATION_DIFF_TOL
        and chosen_pre_diff_w2 < TRANSLATION_DIFF_TOL
    )

    sp = SamplingParams(temperature=0.0, max_tokens=20, seed=0)
    prompt = "The quick brown fox jumps over"
    _, baseline_text = _time_generate(llm, prompt, sp)
    print(f"  baseline gen : {baseline_text!r}", flush=True)

    orig_w13 = w13.detach().clone()
    orig_w2 = w2.detach().clone()

    # Zero gate_proj of expert 0 in HF -- a strong, clean perturbation.
    hf_layer.experts[0].gate_proj.weight.data.zero_()

    new_w13, new_w2 = translate(hf_layer, swap_gate_up=swap)

    push_ok = False
    push_error: str | None = None
    try:
        with torch.no_grad():
            w13.copy_(new_w13)
            w2.copy_(new_w2)
        push_ok = True
    except Exception as e:  # pragma: no cover
        push_error = repr(e)

    if push_ok:
        _, after_text = _time_generate(llm, prompt, sp)
        print(f"  after push   : {after_text!r}", flush=True)
        output_changed = after_text != baseline_text
    else:
        after_text = None
        output_changed = False

    # Restore vLLM tensors (HF is left perturbed since this is the last test).
    with torch.no_grad():
        w13.copy_(orig_w13)
        w2.copy_(orig_w2)

    return {
        "model": MODEL_ID,
        "moe_layer_idx": MOE_LAYER_IDX,
        "n_experts": n_experts,
        "moe_intermediate_size": moe_inter,
        "vllm_w13_shape": list(w13.shape),
        "vllm_w2_shape": list(w2.shape),
        "vllm_w13_dtype": str(w13.dtype),
        "vllm_w2_dtype": str(w2.dtype),
        "hf_gate_shape": list(g0.shape),
        "hf_up_shape": list(u0.shape),
        "hf_down_shape": list(d0.shape),
        "pre_translation_diff_w13_gate_up": pre_diff_a_w13,
        "pre_translation_diff_w13_up_gate": pre_diff_b_w13,
        "pre_translation_diff_w2": pre_diff_a_w2,
        "chosen_order": chosen_order,
        "chosen_pre_translation_diff_w13": chosen_pre_diff_w13,
        "chosen_pre_translation_diff_w2": chosen_pre_diff_w2,
        "tolerance": TRANSLATION_DIFF_TOL,
        "verdict_translation_correct": bool(translation_correct),
        "baseline_text": baseline_text,
        "after_push_text": after_text,
        "push_raised": not push_ok,
        "push_error": push_error,
        "verdict_output_changed_after_push": bool(output_changed),
        "verdict_full_pass": bool(translation_correct and push_ok and output_changed),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    results: dict[str, Any] = {
        "harness_version": 1,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": MODEL_ID,
        "moe_layer_idx": MOE_LAYER_IDX,
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_vram_gib": (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
            if torch.cuda.is_available() else None
        ),
        "tests": {},
        "errors": {},
    }

    print(f"\n[{time.strftime('%H:%M:%S')}] === loading vLLM ===", flush=True)
    # enforce_eager=False so CUDA graph capture IS exercised (point of Test 1).
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        enforce_eager=False,
        gpu_memory_utilization=0.45,  # leave headroom for HF + activations
        max_model_len=2048,
        trust_remote_code=False,
    )

    print(f"\n[{time.strftime('%H:%M:%S')}] === loading HF model ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=False,
    )
    hf_model.eval()

    for name, fn in [
        ("test_update_weights_cuda_graph", lambda: test_update_weights_cuda_graph(llm)),
        ("test_per_expert_hook_mapping", lambda: test_per_expert_hook_mapping(llm, hf_model, tokenizer)),
        ("test_weight_layout_adapter", lambda: test_weight_layout_adapter(llm, hf_model, tokenizer)),
    ]:
        try:
            results["tests"][name] = fn()
        except Exception as e:
            tb = traceback.format_exc()
            print(f"\n!! {name} CRASHED:\n{tb}", flush=True)
            results["errors"][name] = {"error": repr(e), "traceback": tb}

    def _pass(d, key="verdict_full_pass"):
        if d is None:
            return "ERROR"
        v = d.get(key)
        if v is True:
            return "PASS"
        if v is False:
            return "FAIL"
        return "?"

    print("\n=== L1 VALIDATION SUMMARY ===", flush=True)
    print(f"  Test 1 (update_weights + CUDA graph): "
          f"{_pass(results['tests'].get('test_update_weights_cuda_graph'))}", flush=True)
    t2 = results["tests"].get("test_per_expert_hook_mapping")
    print(f"  Test 2 (per-expert hook mapping)    : "
          f"{_pass(t2, 'verdict_scatter_reconstruction_works')}", flush=True)
    print(f"  Test 3 (HF -> vLLM weight adapter)  : "
          f"{_pass(results['tests'].get('test_weight_layout_adapter'))}", flush=True)

    out_path = Path(os.environ.get("L1_RESULTS_PATH", "/tmp/l1_validation_results.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nresults written to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
