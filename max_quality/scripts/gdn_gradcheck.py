#!/usr/bin/env python3
"""GDN-backward gate: prove fla's gated-DeltaNet backward does NOT raise on
Hopper (H200) and yields finite grads — i.e. tilelang is correctly providing the
Hopper backward (fla PR #827 makes the pure-Triton path RAISE without it).

CRITICAL test-bug lesson: the GDN gate ``g`` is LOG-SPACE (<= 0). Building it
with a positive tensor (e.g. torch.rand) overflows exp() -> bf16 inf -> a FALSE
"non-finite grads" failure that looks like a kernel bug but is a test bug. Use
logsigmoid (or -softplus) so g <= 0.

Exit 0 = backward ran + all grads finite (env is correct for Router-KD training).
Any raise / non-finite = exit 1 (loud, before any GPU-hours are spent).
"""
import sys
import torch


def main() -> int:
    assert torch.cuda.is_available(), "CUDA not available"
    dev = "cuda"
    dt = torch.bfloat16
    cap = torch.cuda.get_device_capability(0)
    print(f"[gdn] device={torch.cuda.get_device_name(0)} cap={cap} torch={torch.__version__} cuda={torch.version.cuda}")

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    B, T, H, D = 2, 256, 4, 128
    g_ = torch.Generator(device=dev).manual_seed(1337)
    q = torch.randn(B, T, H, D, device=dev, dtype=dt, generator=g_, requires_grad=True)
    k = torch.randn(B, T, H, D, device=dev, dtype=dt, generator=g_, requires_grad=True)
    v = torch.randn(B, T, H, D, device=dev, dtype=dt, generator=g_, requires_grad=True)
    # LOG-SPACE decay: logsigmoid(rand) in (-inf, 0], stable. (NOT positive rand.)
    g = torch.nn.functional.logsigmoid(
        torch.randn(B, T, H, device=dev, dtype=torch.float32, generator=g_)
    ).requires_grad_(True)
    beta = torch.rand(B, T, H, device=dev, dtype=dt, generator=g_).sigmoid().requires_grad_(True)

    try:
        # Current fla signature: (q,k,v,g,beta,...) with [B,T,H,D] layout (no head_first kwarg).
        # use_qk_l2norm_in_kernel=True: GDN needs L2-normalized q,k or the recurrence
        # blows up in bf16 (the model always normalizes — raw randn here would not).
        out = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
        o = out[0] if isinstance(out, (tuple, list)) else out
        loss = o.float().square().mean()
        loss.backward()
    except Exception as exc:  # noqa: BLE001
        print(f"[gdn] FAIL — backward raised: {type(exc).__name__}: {exc}")
        return 1

    grads = {"q": q.grad, "k": k.grad, "v": v.grad, "g": g.grad, "beta": beta.grad}
    missing = [n for n, gr in grads.items() if gr is None]
    nonfinite = [n for n, gr in grads.items() if gr is not None and not torch.isfinite(gr).all()]
    if missing or nonfinite:
        print(f"[gdn] FAIL — missing={missing} non-finite={nonfinite}")
        return 1
    print(f"[gdn] PASS — loss={loss.item():.4e}, all grads finite "
          f"({', '.join(f'{n}:{tuple(gr.shape)}' for n, gr in grads.items())})")
    print("GDN_GRADCHECK_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
