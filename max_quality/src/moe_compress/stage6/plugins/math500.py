"""MATH-500 accuracy generative eval (S6-4 of the Stage 6 plugin refactor).

Paper / dataset
----------------
MATH-500 — Hendrycks et al. 2021 (arXiv:2103.03874) original MATH
benchmark; the 500-problem subset is Lightman et al. 2024 (OpenAI's
``prm800k`` curation). Scoring: ``\boxed{...}`` extraction +
SymPy symbolic equivalence.

Stage 6 implementation note: each problem is wrapped in the model's
chat template, the model generates chain-of-thought +
``\boxed{answer}``, and the gate is computed over all 500 prompts.

Reference code
--------------
No single canonical implementation. The ``\boxed{}`` extraction +
SymPy grading follows the canonical math-eval recipe (PRM800k /
DeepSeek-Math / Qwen-Math implementations all converge on this
form).

Home of the Stage 6 MATH-500 concern, extracted from the legacy
``stage6_validate.py`` monolith. MATH-500 is the math-reasoning half of the
Stage 6 generative gate: each problem is wrapped in the model's chat template,
the student/teacher generates a chain-of-thought + ``\\boxed{answer}``, and the
answer is graded by ``\\boxed{}`` extraction + SymPy symbolic equivalence
(Spec §9 -- the gate is computed over all 500 prompts).

Pattern A vs Pattern B
----------------------
S6-4's MATH-500 slice covers a MIXED pattern (mirror of S6-3):

* **Pattern A -- relocated verbatim**: ``_math500``, ``_extract_boxed``,
  ``_last_numeric``, ``_check_math``, ``_math_fallback_extract`` AND the
  module-level optional-SymPy ``try/except`` guard below are character-
  identical copies of the monolith definitions. ``stage6_validate.py``
  re-imports the public-facing helpers (a ``# noqa: F401`` block) so ``run()``
  and external callers/tests keep their original import path. The shared
  batched-generation / chat-format primitives ``_math500`` calls live in
  ``tools/eval_harness`` (also extracted by S6-4) and are imported from there.
  The SymPy guard is relocated (not re-imported) because the monolith no
  longer references ``_SYMPY_AVAILABLE`` / ``simplify`` / ``sympify`` /
  ``_parse_latex`` once ``_check_math`` / ``_math_fallback_extract`` leave it.
* **Pattern B -- reproduced in an inert hook**: the ``run()`` student-side
  MATH-500 *call site* (the ``if "math500" in s6["generative"]`` gate + the
  ``_math500(...)`` invocation that lands the result in
  ``results["student"]["math500_accuracy"]``) is INLINE ``run()`` code in the
  monolith -- there is nothing standalone to relocate. The ``eval_task`` hook
  below REPRODUCES that inline call faithfully; the monolith ``run()`` is NOT
  modified for it. This is an intentional, temporary logic duplication that
  resolves at S6-8 when the monolith ``run()`` is deleted and this hook is
  wired live.

Circular-import contract (mirror of ``stage6/plugins/wikitext_ppl.py``): this
module imports only from ``..context`` / ``...tools.eval_harness`` / stdlib /
optional SymPy -- NEVER from ``stage6_validate``, ``stage6.orchestrator`` or
``orchestrator`` at any scope (module-top OR function-local). The monolith
re-imports *this* module at load time, so a ``from ..stage6_validate import
...`` here would deadlock the import; nothing in this module does that.

``Math500Plugin`` is registered-but-INERT at S6-4 -- no orchestrator walk or
test invokes its ``eval_task`` hook. S6-8 plugs the hook into the live Stage 6
plugin sequencer and deletes the monolith ``run()``.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from ..context import PipelineContext
from ...tools.eval_harness import (
    _chat_format_prompts,
    _generate_batched,
    _stage6_enable_thinking,
)

log = logging.getLogger(__name__)


# F-C-H-1: Spec F-S-M-1 mandates eager attention for both teacher and student
# during the Stage 6 gate run. Constant -- never override at call sites. This
# is a module-LOCAL copy of the monolith's ``_STAGE6_ATTN_IMPLEMENTATION``: the
# monolith keeps its own definition and is NOT imported here (circular-import
# contract). Both copies must stay in sync until S6-8 collapses the monolith.
_STAGE6_ATTN_IMPLEMENTATION: str = "eager"


# N3: sympy is optional — imported once at module level to avoid repeated
# import overhead inside _check_math (which is called per-problem).
try:
    from sympy import simplify, sympify
    from sympy.parsing.latex import parse_latex as _parse_latex
    _SYMPY_AVAILABLE = True
except Exception:  # noqa: BLE001
    _SYMPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Generative -- MATH-500 accuracy
# ---------------------------------------------------------------------------


def _math500(model, tokenizer, cfg: dict, *, device=None, collect=None,
             batch_size: int = 8,
             dataset_revisions: dict[str, str | None] | None = None) -> float:
    try:
        from datasets import load_dataset
    except Exception as err:           # noqa: BLE001
        log.warning("datasets not available (%s); skipping MATH-500.", err)
        return float("nan")
    revision = (dataset_revisions or {}).get("math500")
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test", revision=revision)
    except Exception as err:           # noqa: BLE001
        log.warning("MATH-500 dataset load failed (%s); skipping.", err)
        return float("nan")

    # Bumped default 1024 → 4096 for thinking-mode (math reasoning traces
    # are routinely 2-4k tokens before the boxed answer).
    max_new = int(os.environ.get("STAGE6_MAX_NEW_MATH", cfg.get("max_new_tokens", 4096)) or 4096)
    n = int(cfg.get("num_samples", 500))
    if n > len(ds):
        # N5: Warn explicitly rather than silently clamping, so config errors are visible.
        log.warning(
            "MATH-500: num_samples=%d exceeds dataset size=%d; clamping to %d",
            n, len(ds), len(ds),
        )
    n_total = min(n, len(ds))
    # Spec §9 mandates "MATH-500 (500 prompts)" — the gate is computed over
    # the full benchmark, not a subset. Refuse to under-sample silently.
    if n_total < len(ds):
        raise ValueError(
            f"MATH-500: configured num_samples={n} produces n_total={n_total} but "
            f"the spec gate requires all {len(ds)} prompts. Either set "
            "generative.math500.num_samples=500 (or omit) or update the spec to "
            "permit subset evaluation."
        )

    selected = ds.select(range(n_total))
    raw_problems = [row["problem"] for row in selected]
    answers = [row.get("answer", "") for row in selected]

    if collect is not None:
        collect.extend(raw_problems)

    # Wrap each problem with the model's chat template — thinking-mode chat
    # models produce CoT reasoning + a \boxed{answer}. The downstream
    # _check_math uses _extract_boxed / _last_numeric which work on
    # free-form text. Same harness fix as HumanEval; see
    # project_a0_student_diagnosis_2026_05_15.md.
    _enable_thinking = _stage6_enable_thinking()
    prompts = _chat_format_prompts(
        tokenizer, raw_problems,
        system=(
            "Solve the math problem. Show your reasoning, then write the "
            "final answer inside \\boxed{...}."
        ),
    )
    log.info("Stage 6 MATH-500: %d problems, batch_size=%d, max_new=%d, "
             "enable_thinking=%s, chat_template=on",
             n_total, batch_size, max_new, _enable_thinking)
    completions = _generate_batched(
        model, tokenizer, prompts, max_new=max_new,
        device=device, batch_size=batch_size,
    )

    correct = 0
    for i, (completion, answer) in enumerate(zip(completions, answers)):
        if _check_math(completion, answer):
            correct += 1
        if (i + 1) % 25 == 0:
            log.info("  MATH-500 eval %d/%d (correct=%d)", i + 1, n_total, correct)
    log.info("  MATH-500 final: %d/%d = %.3f", correct, n_total, correct / max(n_total, 1))
    return correct / max(n_total, 1)


def _extract_boxed(s: str) -> str | None:
    """Extract the last \\boxed{...} value from s using balanced-brace scanning.

    Handles nested braces (e.g. \\boxed{\\frac{1}{2}}). Pure function; defined at
    module level to avoid re-allocation on every _check_math call.
    """
    results = []
    idx = 0
    while True:
        m = re.search(r'\\boxed\{', s[idx:])
        if not m:
            break
        start = idx + m.end()
        depth = 1
        i = start
        while i < len(s) and depth > 0:
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            results.append(s[start:i - 1])
            idx = i  # advance past the closing '}'
        else:
            # Unclosed \boxed{ — truncated output; stop scanning to avoid
            # misidentifying nested \boxed{} inside the open group as top-level.
            break
    return results[-1] if results else None


def _last_numeric(s: str) -> str | None:
    """Return the last numeric token in s (integer, float, or scientific notation).

    Pure function; defined at module level to avoid re-allocation on every call.
    """
    nums = re.findall(r"-?\d*\.?\d+(?:[eE][+-]?\d+)?", s)
    return nums[-1] if nums else None


def _check_math(completion: str, reference: str) -> bool:
    comp_answer = _extract_boxed(completion)
    ref_answer = _extract_boxed(reference)
    # F-iter4-HIGH-2: when \boxed{...} is absent from the reference, do NOT
    # fall straight to _last_numeric — that returns "2" for "\\frac{1}{2}",
    # silently truncating the rational to its denominator. Try SymPy LaTeX
    # parsing first; only fall back to _last_numeric if LaTeX parsing fails
    # too. (Comp answers are model output and are scored by symbolic
    # equivalence below, so the same fallback policy applies.)
    if comp_answer is None:
        comp_answer = _math_fallback_extract(completion)
    if ref_answer is None:
        ref_answer = _math_fallback_extract(reference)

    if comp_answer is None or ref_answer is None:
        return False
    if comp_answer.strip() == ref_answer.strip():
        return True

    # N3: sympy is imported once at module level; skip symbolic check if unavailable.
    if _SYMPY_AVAILABLE:
        try:
            try:
                a = _parse_latex(comp_answer)
            except Exception:
                a = sympify(comp_answer)
            try:
                b = _parse_latex(ref_answer)
            except Exception:
                b = sympify(ref_answer)
            return bool(simplify(a - b) == 0)
        except Exception:
            pass

    a = _last_numeric(comp_answer)
    b = _last_numeric(ref_answer)
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def _math_fallback_extract(s: str) -> str | None:
    """Return a candidate answer string from ``s`` when no \\boxed{} is found.

    F-iter4-HIGH-2: prefer SymPy LaTeX parsing of the full string before
    collapsing to the last numeric token. ``_last_numeric`` returns "2" for
    "\\frac{1}{2}" — incorrect. If LaTeX parsing succeeds the original LaTeX
    string is returned (the caller's symbolic-equivalence check then re-parses
    it the same way); only when LaTeX parsing fails do we fall through to
    ``_last_numeric``.
    """
    if _SYMPY_AVAILABLE:
        try:
            _ = _parse_latex(s.strip())
            return s.strip()
        except Exception:  # noqa: BLE001
            pass
    return _last_numeric(s)


class Math500Plugin:
    """Stage 6 MATH-500 accuracy plugin (S6-4 -- registered-but-INERT).

    Owns the Stage 6 MATH-500 sub-metric: the relocated ``_math500`` driver and
    its ``_extract_boxed`` / ``_last_numeric`` / ``_check_math`` /
    ``_math_fallback_extract`` grading helpers (Pattern A) plus an inert
    ``eval_task`` hook (Pattern B) that reproduces the monolith's inline
    student-side MATH-500 call site.

    S6-4 wires this class into the plugin registry as metadata only -- no
    orchestrator walk or test invokes ``eval_task``. S6-8 plugs the hook into
    the live Stage 6 plugin sequencer and deletes the monolith ``run()``.
    """

    name = "math500"
    paper = "MATH-500 — Hendrycks et al. 2021 arXiv:2103.03874 (subset Lightman et al. 2024); SymPy-graded \boxed extraction. See module docstring."
    config_key = "stage6_validate.generative.enabled"
    reads: tuple[str, ...] = ("model", "tokenizer", "config", "dataset_revisions")
    writes: tuple[str, ...] = ("eval_results",)
    # eval_results is a shared collector the orchestrator pre-creates per side
    # and every eval plugin appends to; it is NOT a calibration-pass accumulator,
    # so it belongs in `writes`, not `provides`. (S6-8 wires the collector.)
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Gate on ``generative.enabled`` AND a ``math500`` sub-key.

        Mirrors the monolith ``run()``'s two-level guard: the outer
        ``if s6["generative"]["enabled"]`` block and the inner
        ``if "math500" in s6["generative"]`` gate. Uses ``.get()`` chains so a
        missing ``stage6_validate`` or ``generative`` subdict resolves to
        disabled rather than raising.
        """
        generative = (config.get("stage6_validate", {}) or {}).get("generative", {}) or {}
        return bool(generative.get("enabled", False)) and "math500" in generative

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def eval_task(self, ctx: PipelineContext) -> None:
        """Phase hook -- Stage 6 MATH-500 eval (S6-8 wiring surface).

        INERT at S6-4: no orchestrator walk or test invokes this hook. S6-8
        replaces the Stage 6 orchestrator body with the plugin sequencer and
        dispatches this hook in place of the monolith's inline ``run()``
        MATH-500 block. The body below reproduces that inline call site
        faithfully -- it is dead code at S6-4 but S6-8 relies on it once the
        monolith ``run()`` is deleted.

        Reproduces the monolith ``run()``'s student-side call:

            results["student"]["math500_accuracy"] = _math500(
                model, tokenizer, s6["generative"]["math500"], device=device,
                collect=eval_text_concat, batch_size=gen_batch_size,
                dataset_revisions=dataset_revisions,
            )

        The result lands in the pre-existing ``eval_results`` ctx slot (the
        analogue of the monolith's ``results["student"]`` dict) under the
        ``math500_accuracy`` key. This hook does NOT ``ctx.set``
        ``eval_results`` -- it mutates the dict another plugin/the orchestrator
        already created.

        The monolith parses ``gen_batch_size`` from ``s6.get("gen_batch_size",
        8)`` and passes the run-scoped ``device`` / ``eval_text_concat``
        side-channel; the hook reproduces the ``gen_batch_size`` default + its
        positive-int validation and threads ``device`` / ``collect`` from
        optional ctx slots so the call shape matches even though those
        side-channels are not S6-4's concern.
        """
        model = ctx.get("model")
        tokenizer = ctx.get("tokenizer")
        config = ctx.get("config")
        dataset_revisions = ctx.get("dataset_revisions")
        s6 = config["stage6_validate"]

        # Reproduces the monolith's gen_batch_size parse/validation block.
        gen_batch_size = int(s6.get("gen_batch_size", 8))
        if gen_batch_size <= 0:
            raise ValueError(
                f"stage6_validate.gen_batch_size must be a positive int; got {gen_batch_size!r}."
            )

        # The run-scoped `device` / `eval_text_concat` are optional context
        # side-channels in the plugin world (the monolith threads them through
        # run()); default to None when a wiring stage has not provided them.
        device = ctx.get("device") if ctx.has("device") else None
        collect = ctx.get("eval_text_concat") if ctx.has("eval_text_concat") else None

        log.info("Stage 6: MATH-500 (student), batch_size=%d", int(gen_batch_size))
        eval_results = ctx.get("eval_results")
        eval_results["math500_accuracy"] = _math500(
            model, tokenizer, s6["generative"]["math500"], device=device,
            collect=collect, batch_size=gen_batch_size,
            dataset_revisions=dataset_revisions,
        )


__all__ = [
    "_math500",
    "_extract_boxed",
    "_last_numeric",
    "_check_math",
    "_math_fallback_extract",
    "Math500Plugin",
]
