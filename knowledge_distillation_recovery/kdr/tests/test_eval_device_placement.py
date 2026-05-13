"""LLR-0062: eval-PPL input tensor is constructed on accelerator.device.

# REQ: LLR-0062
# VERIFIES: LLR-0062
"""

from __future__ import annotations

import inspect

import torch

from kdr.eval import quick


def test_wikitext2_uses_device_kwarg_not_trailing_to_call() -> None:
    """LLR-0062 AC: source check — the `torch.tensor(...)` call passes
    `device=accelerator.device` directly, and there is no separate
    `.to(accelerator.device)` chained onto it."""
    src = inspect.getsource(quick.wikitext2_ppl)
    assert "device=accelerator.device" in src, (
        "expected `device=accelerator.device` in the construction span; "
        "the LLR-0062 cleanup is not in place"
    )
    # Locate the multi-line `inp = torch.tensor(` construction and verify
    # no chained `.to(accelerator.device)` follows it.
    lines = src.splitlines()
    start_idx = next(
        (i for i, ln in enumerate(lines) if "inp = torch.tensor(" in ln),
        -1,
    )
    assert start_idx >= 0, "could not locate `inp = torch.tensor(` in source"
    window = "\n".join(lines[start_idx:start_idx + 6])
    assert "device=accelerator.device" in window, (
        f"expected `device=accelerator.device` in the inp construction "
        f"span; got:\n{window}"
    )
    assert ".to(accelerator.device)" not in window, (
        f"found legacy `.to(accelerator.device)` chained onto inp; "
        f"LLR-0062 cleanup incomplete:\n{window}"
    )


def test_torch_tensor_device_kwarg_lands_on_meta() -> None:
    """LLR-0062 AC: behavioural sanity — confirm that
    `torch.tensor(list, dtype=long, device=meta)` does land on the meta
    device (as the production line now does). Pinned for CPU-only CI."""
    inp = torch.tensor(
        [1, 2, 3, 4], dtype=torch.long, device=torch.device("meta")
    ).view(2, 2)
    assert inp.device == torch.device("meta")
    assert inp.shape == (2, 2)
    assert inp.dtype == torch.long
