"""Tests for `kdr.training.loop.run_recovery` mode dispatch (LLR-0007).

LLR-0007 AC #1: exactly one if/else branch on `cfg.mode` in the loop.
LLR-0007 AC #2: in bf16 mode there are zero modelopt calls.
LLR-0007 AC #4: in da_qad mode `partition_and_dispatch` is called INSIDE
    the activate_zero3_init context (LLR-0048 step 4).

Phase 3b only validates the bf16 path — da_qad's `partition_and_dispatch`
is a Phase 4 stub. We verify the structural invariants and the dispatch
guard rather than running an actual training step.

# VERIFIES: LLR-0007
"""

from __future__ import annotations

import inspect

from kdr.training import loop


def test_loop_has_single_mode_branch() -> None:
    """LLR-0007 AC #1: exactly one `if config.mode == ...` / `if cfg.mode ==`
    in the loop module."""
    src = inspect.getsource(loop.run_recovery)
    # Count occurrences of the mode comparison.
    n = src.count('config.mode == "da_qad"') + src.count('config.mode == "bf16"')
    assert n == 1, f"Expected exactly one mode-equality branch, found {n}."


def test_da_qad_branch_inside_zero3_context() -> None:
    """LLR-0007 AC #4 / LLR-0048 step 4: the partition_and_dispatch call
    SHALL occur INSIDE the `activate_zero3_init` context. Structural check
    against the source — assert that the `partition_and_dispatch` import
    appears before the context exits (i.e. inside the `with` block)."""
    src = inspect.getsource(loop.run_recovery)
    # Find the `with activate_zero3_init` line and the next dedent.
    lines = src.splitlines()
    idx_with = next(
        i for i, ln in enumerate(lines) if "activate_zero3_init" in ln and "with " in ln
    )
    # The dispatch must appear after `with` and before its enclosing block ends.
    # We simplify by asserting the dispatch line is strictly after the `with`.
    idx_dispatch = next(
        (i for i, ln in enumerate(lines) if "partition_and_dispatch" in ln),
        -1,
    )
    assert idx_dispatch > idx_with, (
        "partition_and_dispatch must appear after `with activate_zero3_init`."
    )

    # Verify the dispatch line is indented MORE than the `with` line — i.e.
    # nested inside the context, not at the parent block's indent.
    with_indent = len(lines[idx_with]) - len(lines[idx_with].lstrip())
    dispatch_indent = len(lines[idx_dispatch]) - len(lines[idx_dispatch].lstrip())
    assert dispatch_indent > with_indent, (
        "partition_and_dispatch must be nested inside the with-block "
        "(LLR-0048 step 4)."
    )


def test_bf16_mode_does_not_import_modelopt() -> None:
    """LLR-0007 AC #2: bf16 mode does NOT call modelopt. The
    `partition_and_dispatch` import lives inside the da_qad branch — verify
    that nothing at function-top-level pulls it in unconditionally."""
    src = inspect.getsource(loop.run_recovery)
    # The line `from ..quant.factory import partition_and_dispatch` MUST be
    # under a `da_qad` guard, not at top-level of run_recovery.
    lines = src.splitlines()
    import_lines = [
        i for i, ln in enumerate(lines) if "import partition_and_dispatch" in ln
    ]
    assert len(import_lines) == 1, "Expected exactly one import of partition_and_dispatch."
    import_line = lines[import_lines[0]]
    # Indent must be > the function's def indent + 4 (i.e. deep nested,
    # inside both the with-block AND the if-da_qad branch).
    indent = len(import_line) - len(import_line.lstrip())
    assert indent >= 8, (
        "partition_and_dispatch import must be inside the da_qad branch, "
        f"not at the loop's top-level (got indent={indent})."
    )


def test_accelerator_prepare_outside_zero3_context() -> None:
    """LLR-0048 step 6: `accelerator.prepare(...)` is called OUTSIDE the
    activate_zero3_init context. Structural check via source inspection."""
    src = inspect.getsource(loop.run_recovery)
    lines = src.splitlines()
    # Find the `with activate_zero3_init` line.
    idx_with = next(
        i for i, ln in enumerate(lines) if "activate_zero3_init" in ln and "with " in ln
    )
    with_indent = len(lines[idx_with]) - len(lines[idx_with].lstrip())
    idx_prepare = next(
        i for i, ln in enumerate(lines) if "accelerator.prepare(" in ln
    )
    prepare_indent = len(lines[idx_prepare]) - len(lines[idx_prepare].lstrip())
    # `prepare` must be at the same or shallower indent than `with`, i.e.
    # outside the with-block.
    assert prepare_indent <= with_indent, (
        "accelerator.prepare must be OUTSIDE the activate_zero3_init "
        "context (LLR-0048 step 6)."
    )
    assert idx_prepare > idx_with
