"""HumanEval pass@1 generative eval (S6-4 of the Stage 6 plugin refactor).

Paper / dataset
----------------
HumanEval — Chen et al. 2021 "Evaluating Large Language Models Trained
on Code" (arXiv:2107.03374). pass@1 metric: each problem stub is
wrapped in the model's chat template, the model generates a
completion, and the completion is scored by exec'ing the dataset's
reference unit test.

Deviations
----------

**D-humaneval-greedy** — Paper protocol: **stochastic** pass@1
estimated from ``n=10`` samples per problem at ``T=0.2, top_p=0.95``,
then unbiased pass@k formula; **exec-based scoring runs each problem
in a subprocess sandbox**.

This plugin's protocol:

  (a) **Greedy decoding** pass@1: ``do_sample=False, n=1``, no
      temperature, no top_p, single sample per problem.
  (b) Scoring runs in a **subprocess sandbox** (a
      ``ProcessPoolExecutor`` forced onto the ``spawn`` start-method):
      each problem's reference test runs in a CHILD PROCESS, so
      ``sys.modules`` / signal-handler / ``os.environ`` mutations
      cannot leak across problems, and a hung/runaway test is
      hard-terminated at a shared deadline (no interpreter-lifetime
      daemon-thread leak). Both teacher and student score with the
      same isolated worker so the relative-to-teacher gate is not
      biased.

Rationale: greedy is lower-variance and reproducible across runs
without seed plumbing, sufficient for **relative-to-teacher gating**
(the gate is a 3pp absolute drop vs the same-protocol teacher score,
not against published baselines). **Absolute pass@1 numbers will not
match published Chen et al. 2021 baselines** and must not be compared
to them. The ``VALIDATED_STRATEGIES`` #3/#4 references cover the
batched-generate PLUMBING (left-padding, EOS-truncation, eager-attn
pin), NOT numerical identity of the metric: the generative metrics
``humaneval_pass_at_1`` / ``math500_accuracy`` are batch-geometry-
dependent (bf16 + left-pad reduction drift flips near-tied argmax
across batch sizes) and are pinned via ``gen_batch_size`` /
``tools.eval_harness.PINNED_GEN_BATCH_SIZE`` for reproducibility. Only
the forward/PPL/loglikelihood path is batch-invariant.

Subprocess isolation (Item-2) replaces the legacy in-process
daemon-thread scoring: the model code now runs in a ``spawn``
ProcessPool child, so the documented daemon-thread LEAK (a timed-out
thread that ran until interpreter exit) is gone — a stuck worker is
terminated at the shared deadline. The remaining caveats are weaker:
the child is still not a full sandbox (no seccomp / landlock /
container), so this is best-effort isolation for trusted inputs. See
``stage6_validate.py`` module docstring "Known limitations" subsection;
that note's daemon-thread-leakage / no-syscall-interruption wording is
superseded here by the subprocess-worker design.

**D-humaneval-maxnew** — Paper / Chen 2021 §2.2 uses comparatively
small decode budgets (the original eval scores standalone completions
that rarely exceed a few hundred tokens). This plugin defaults
``max_new_tokens=2048`` (was 512) to leave room for thinking-mode
traces on Qwen3.5/3.6-class chat models, where ``<think>...</think>``
blocks routinely consume 1-2k tokens before the actual code emerges.
Override via ``cfg.max_new_tokens`` or env ``STAGE6_MAX_NEW_HE`` for
non-thinking models. Has no effect on relative-to-teacher gating
(both sides decode with the same budget) but further widens the gap
from Chen et al. 2021 absolute pass@1 numbers.

**D-humaneval-limit** — When ``HUMANEVAL_LIMIT`` env (or ``cfg.limit``)
is set to a positive integer less than 164, prompts/tests/entry_points
are truncated in lockstep and pass@1 is rebased to the subset count.
This is a **smoke-subset** mode (segfault-fix / wiring smoke runs);
any non-zero ``HUMANEVAL_LIMIT`` invalidates direct comparison against
164-problem published baselines OR against full-eval prior runs.
Relative-to-teacher gating remains valid only if teacher and student
share the same limit.

Home of the Stage 6 HumanEval concern, extracted from the legacy
``stage6_validate.py`` monolith. HumanEval is the code-generation half of the
Stage 6 generative gate: each problem stub is wrapped in the model's chat
template, the student/teacher generates a completion, and the completion is
scored by exec'ing the dataset's reference unit test (Spec D-humaneval-greedy:
greedy single-sample pass@1).

Pattern A vs Pattern B
----------------------
S6-4's HumanEval slice covers a MIXED pattern (mirror of S6-3):

* **Pattern A -- relocated verbatim**: ``_humaneval`` and ``_check_humaneval``
  below are character-identical copies of the monolith bodies.
  ``stage6_validate.py`` re-imports them (a ``# noqa: F401`` block) so
  ``run()`` and external callers/tests keep their original import path. The
  shared batched-generation / chat-format primitives those two functions call
  live in ``tools/eval_harness`` (also extracted by S6-4) and are imported
  from there.
* **Pattern B -- reproduced in an inert hook**: the ``run()`` student-side
  HumanEval *call site* (the ``if "humaneval" in s6["generative"]`` gate, the
  ``num_samples_per_task`` guard, and the ``_humaneval(...)`` invocation that
  lands the result in ``results["student"]["humaneval_pass_at_1"]``) is INLINE
  ``run()`` code in the monolith -- there is nothing standalone to relocate.
  The ``eval_task`` hook below REPRODUCES that inline call faithfully; the
  monolith ``run()`` is NOT modified for it. This is an intentional, temporary
  logic duplication that resolves at S6-8 when the monolith ``run()`` is
  deleted and this hook is wired live.

Circular-import contract (mirror of ``stage6/plugins/wikitext_ppl.py``): this
module imports only from ``..context`` / ``...tools.eval_harness`` / sibling
plugin modules (``eval_environment``, ``_humaneval_worker`` — both torch-free
or context-only) / stdlib --
NEVER from ``stage6_validate``, ``stage6.orchestrator`` or ``orchestrator`` at
any scope (module-top OR function-local). The monolith re-imports *this* module
at load time, so a ``from ..stage6_validate import ...`` here would deadlock
the import; nothing in this module does that.

**Security note -- HumanEval code execution (H1):** scoring runs
model-generated Python in a ``spawn`` ProcessPool CHILD PROCESS (Item-2), with
a shared wall-clock deadline and hard termination of stuck workers. This is
subprocess isolation -- strictly stronger than the legacy in-process daemon
thread -- but still best-effort: no seccomp / landlock / container boundary.
See the monolith docstring for the full caveat list. Use only in trusted
environments.

``HumanEvalPlugin`` is registered-but-INERT at S6-4 -- no orchestrator walk or
test invokes its ``eval_task`` hook. S6-8 plugs the hook into the live Stage 6
plugin sequencer and deletes the monolith ``run()``.
"""
from __future__ import annotations

import concurrent.futures
import logging
import multiprocessing
import os
import time
from typing import Any

from ..context import PipelineContext
from ...tools.eval_harness import (
    PINNED_GEN_BATCH_SIZE,
    _chat_format_prompts,
    _generate_batched,
    _stage6_enable_thinking,
)
# C1: imported at module top so the eval_task hook can call the canonical
# experts-impl shim without a function-local import (which would be
# re-resolved per call and complicate static analysis). The eval_environment
# plugin module only imports from ..context / stdlib / model internals, so
# this is safe wrt the circular-import contract documented in the module
# docstring.
from .eval_environment import _set_experts_implementation_s6
# Item-2: torch-free leaf worker submitted to the spawn ProcessPool (see the
# scoring loop in _humaneval). Importing it here is contract-safe: the worker
# imports only stdlib + re (no torch, no stage modules).
from ._humaneval_worker import _score_humaneval_one

log = logging.getLogger(__name__)


# F-C-H-1: Spec F-S-M-1 mandates eager attention for both teacher and student
# during the Stage 6 gate run. Constant -- never override at call sites. This
# is a module-LOCAL copy of the monolith's ``_STAGE6_ATTN_IMPLEMENTATION``: the
# monolith keeps its own definition and is NOT imported here (circular-import
# contract). Both copies must stay in sync until S6-8 collapses the monolith.
# N1: Currently unreferenced *inside this module* (the attention selection is
# enforced by eval_environment.py before this plugin runs); retained so tests
# pinning the constant's existence/value and the future S6-8 plug-in collapse
# both find it here. Do NOT remove without coordinated test + monolith edits.
_STAGE6_ATTN_IMPLEMENTATION: str = "eager"  # noqa: F841 — see N1 comment above


# ---------------------------------------------------------------------------
# Generative -- HumanEval pass@1
# ---------------------------------------------------------------------------


def _humaneval(model, tokenizer, cfg: dict, *, device=None, collect=None,
               batch_size: int = 8,
               dataset_revisions: dict[str, str | None] | None = None) -> float:
    try:
        from datasets import load_dataset
    except Exception as err:           # noqa: BLE001
        log.warning("datasets not available (%s); skipping HumanEval.", err)
        return float("nan")
    revision = (dataset_revisions or {}).get("humaneval")
    # F-iter4-LOW-3: prefer the namespaced HF id ("openai/openai_humaneval");
    # fall back to the legacy unnamespaced id for HF datasets versions that
    # haven't migrated yet.
    ds = None
    last_err: Exception | None = None
    for ds_id in ("openai/openai_humaneval", "openai_humaneval"):
        try:
            ds = load_dataset(ds_id, split="test", revision=revision)
            break
        except Exception as err:           # noqa: BLE001
            last_err = err
            log.debug("HumanEval load via %r failed (%s); will try fallback.", ds_id, err)
    if ds is None:
        log.warning("HumanEval dataset load failed (%s); skipping.", last_err)
        return float("nan")

    # Bumped default 512 → 2048 to leave room for thinking-mode traces
    # (Qwen3.5/3.6 think blocks are routinely 1-2k tokens). Override via
    # cfg.max_new_tokens or env STAGE6_MAX_NEW_HE if needed.
    _he_max = int(os.environ.get("STAGE6_MAX_NEW_HE", cfg.get("max_new_tokens", 2048)) or 2048)
    max_new = _he_max
    exec_timeout_secs = int(cfg.get("exec_timeout_secs", 10))

    raw_prompts = [row["prompt"] for row in ds]
    tests = [row["test"] for row in ds]
    entry_points = [row["entry_point"] for row in ds]

    # Optional problem-count cap for smoke testing the segfault-fix path
    # without burning 164 generates. Env var wins over cfg. Value 0/unset =
    # no cap. Truncate prompts/tests/entry_points in lockstep so pass@1
    # arithmetic still divides by the actual count.
    _he_limit = int(os.environ.get("HUMANEVAL_LIMIT", cfg.get("limit", 0)) or 0)
    if _he_limit > 0 and _he_limit < len(raw_prompts):
        log.warning(
            "Stage 6 HumanEval: HUMANEVAL_LIMIT=%d active — truncating from "
            "%d → %d problems (smoke mode; pass@1 computed on subset)",
            _he_limit, len(raw_prompts), _he_limit,
        )
        raw_prompts = raw_prompts[:_he_limit]
        tests = tests[:_he_limit]
        entry_points = entry_points[:_he_limit]

    if collect is not None:
        collect.extend(raw_prompts)

    # Wrap each HumanEval stub with the model's chat template so the chat-
    # tuned student/teacher actually engage their normal response behavior.
    # Sending raw stubs to a thinking-mode-default chat model produced 0/164
    # for the teacher and ~28% for the student in the prior run — the model
    # decoded `<think>...` filler past max_new_tokens because EOS never fires
    # on a raw-prompt continuation. (project_a0_student_diagnosis_2026_05_15.md)
    _enable_thinking = _stage6_enable_thinking()
    prompts = _chat_format_prompts(
        tokenizer, raw_prompts,
        system=(
            "Complete the Python function. Reply with the full function "
            "definition inside a ```python code block."
        ),
    )
    log.info("Stage 6 HumanEval: %d problems, batch_size=%d, max_new=%d, "
             "enable_thinking=%s, chat_template=on",
             len(prompts), batch_size, max_new, _enable_thinking)
    # H-1 — Security note (emitted once for the full eval, not per problem):
    # scoring runs model-generated Python in spawn ProcessPool CHILD PROCESSES
    # (Item-2) with a shared wall-clock deadline and hard termination of stuck
    # workers.  This is subprocess isolation (stronger than the legacy in-process
    # daemon thread) but still best-effort — no seccomp / landlock / container
    # boundary.  Runaway or malicious generated code can still access the
    # filesystem, network, and its own interpreter state.  Use only in trusted
    # environments or behind an external sandbox.
    log.warning(
        "HumanEval: running model-generated code for %d problems in spawn "
        "ProcessPool workers with a %.0fs shared scoring deadline; subprocess "
        "isolation only (no seccomp / landlock / container).",
        len(prompts), exec_timeout_secs,
    )
    completions = _generate_batched(
        model, tokenizer, prompts, max_new=max_new,
        device=device, batch_size=batch_size,
    )

    total = len(prompts)
    # Pair RAW stub (for the fallback src construction) with the model's
    # chat-formatted reply. `_score_humaneval_one` prefers code extracted from
    # the reply when it contains `def <entry_point>(...)`; else falls back to
    # stub + reply (identical to the legacy `_check_humaneval` path).
    passes = _score_all_humaneval(
        raw_prompts, completions, tests, entry_points,
        exec_timeout_secs=exec_timeout_secs,
    )
    log.info("  HumanEval final: %d/%d = %.3f", passes, total, passes / max(total, 1))
    return passes / max(total, 1)


def _score_all_humaneval(
    raw_prompts: list[str], completions: list[str],
    tests: list[str], entry_points: list[str],
    *, exec_timeout_secs: int,
) -> int:
    """Score every HumanEval problem in a spawn ProcessPool; return pass count.

    Item-2: replaces the legacy serial daemon-thread loop. Each problem is
    scored by ``_humaneval_worker._score_humaneval_one`` in a CHILD PROCESS
    (the process is the isolation + kill boundary), so a hanging/runaway
    completion no longer leaks a daemon thread that lives until interpreter
    exit. All problems are submitted up front and waited against a SINGLE
    SHARED deadline (``exec_timeout_secs`` total, NOT per-future) so total
    scoring wall-time is capped at ~one timeout instead of N x timeout in the
    timeout-heavy regime. Unfinished futures at the deadline score False
    (matches the legacy timeout->False contract) and their workers are
    hard-terminated via ``shutdown(wait=False, cancel_futures=True)``.

    The pass tally is an order-independent ``sum`` over finished futures, so
    completion order does not affect the result (greedy decode makes each
    score a pure function of its inputs).
    """
    total = len(raw_prompts)
    if total == 0:
        return 0
    # FORCE spawn (host default may be fork -- fork-after-CUDA-init can deadlock
    # the child). The worker leaf module is torch-free, so spawn import cost is
    # small + one-time.
    ctx = multiprocessing.get_context("spawn")
    max_workers = max(1, min(os.cpu_count() or 1, total))
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers, mp_context=ctx,
    )
    fut_to_idx: dict[concurrent.futures.Future, int] = {}
    try:
        for i, (raw_stub, completion, test, ep) in enumerate(
            zip(raw_prompts, completions, tests, entry_points)
        ):
            fut = executor.submit(
                _score_humaneval_one, raw_stub, completion, test, ep,
            )
            fut_to_idx[fut] = i

        # Shared deadline across the whole batch (NOT per-future) so the timeout
        # does not re-serialize to N x exec_timeout_secs.
        deadline = time.monotonic() + exec_timeout_secs
        pending = set(fut_to_idx)
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            _done, pending = concurrent.futures.wait(
                pending, timeout=remaining,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

        # Order-independent tally over FINISHED futures; unfinished -> False.
        passes = 0
        terminated = 0
        for fut, i in fut_to_idx.items():
            if fut.done() and not fut.cancelled():
                try:
                    if fut.result() is True:
                        passes += 1
                except Exception as exc:           # noqa: BLE001
                    # A worker that crashed (segfault->BrokenProcessPool, etc.)
                    # scores False, matching the legacy exception->False path.
                    log.warning(
                        "HumanEval problem %d: worker raised (%s); scoring False.",
                        i, exc,
                    )
            else:
                terminated += 1
        if terminated:
            log.warning(
                "HumanEval: %d problems exceeded the %ds shared scoring deadline "
                "and were terminated (subprocess workers; scored as failures).",
                terminated, exec_timeout_secs,
            )
        return passes
    finally:
        # Hard-terminate any still-running workers. cancel_futures drops
        # queued-but-unstarted work; but a worker already RUNNING a runaway
        # snippet (e.g. `while True: pass`) is NOT stopped by
        # shutdown(wait=False) alone -- the pool's atexit join would then HANG
        # the whole interpreter (the very failure mode this design exists to
        # kill). The public API has no "terminate running children" call, so we
        # reach into the documented-but-private `_processes` mapping to send
        # SIGTERM (then SIGKILL) to each live worker. This is a CPython
        # ProcessPoolExecutor implementation detail (stable across 3.9-3.13:
        # `_processes` is a {pid: Process} dict); if a future CPython renames
        # it the getattr-guarded fallback degrades to shutdown-only (still
        # correct for non-runaway workers, only re-introducing the hang risk
        # for a truly stuck child). A terminated process reclaims runaway work
        # immediately -- unlike the legacy daemon thread that ran until
        # interpreter exit.
        procs = getattr(executor, "_processes", None)
        if procs:
            for _proc in list(procs.values()):
                if _proc.is_alive():
                    _proc.terminate()      # SIGTERM
            for _proc in list(procs.values()):
                _proc.join(timeout=1.0)
                if _proc.is_alive():
                    _proc.kill()           # SIGKILL — stuck in C ext / ignoring TERM
                    _proc.join(timeout=1.0)
        executor.shutdown(wait=False, cancel_futures=True)


def _check_humaneval(
    prompt: str, completion: str, test_src: str, entry_point: str,
    *, exec_timeout_secs: int = 10,
    _leaked_counter: list | None = None,
    _problem_index: int = 0,
) -> bool:
    """Score a single HumanEval problem; True iff its reference test passes.

    Item-2: this is now a CONTRACT-PRESERVING WRAPPER that delegates to the
    torch-free leaf worker ``_humaneval_worker._score_humaneval_one``. The
    batch scoring path (``_humaneval`` -> ``_score_all_humaneval``) runs that
    same worker in a spawn ProcessPool with a shared deadline and hard
    termination of stuck workers — the subprocess IS the timeout + isolation
    boundary, replacing the legacy in-process daemon thread (which leaked a
    runaway thread that lived until interpreter exit).

    The signature is unchanged so the existing in-process unit tests keep
    pinning it positionally (``_check_humaneval(prompt, completion, test_src,
    entry_point) is True/False``) and ``stage6_validate`` keeps re-importing
    this symbol from here (the is-identity pin). ``exec_timeout_secs`` /
    ``_leaked_counter`` / ``_problem_index`` are accepted for backwards
    compatibility but the wall-clock timeout + leak accounting now live in the
    batch ProcessPool path, NOT in this single-problem call (a direct call runs
    the worker in-process and returns its bool exactly).
    """
    log.debug("HumanEval problem %d: scoring (delegated to worker).",
              _problem_index)
    return _score_humaneval_one(prompt, completion, test_src, entry_point)


class HumanEvalPlugin:
    """Stage 6 HumanEval pass@1 plugin (S6-4 -- registered-but-INERT).

    Owns the Stage 6 HumanEval sub-metric: the relocated ``_humaneval`` and
    ``_check_humaneval`` helpers (Pattern A) plus an inert ``eval_task`` hook
    (Pattern B) that reproduces the monolith's inline student-side HumanEval
    call site.

    S6-4 wires this class into the plugin registry as metadata only -- no
    orchestrator walk or test invokes ``eval_task``. S6-8 plugs the hook into
    the live Stage 6 plugin sequencer and deletes the monolith ``run()``.
    """

    name = "humaneval"
    # N2: split citation from deviation list, mirroring sibling Stage 6
    # plugins. ``paper`` is the canonical citation only; ``deviation`` is the
    # one-line summary pointing to the module docstring for the full list.
    paper = "HumanEval pass@1 — Chen et al. 2021 arXiv:2107.03374."
    deviation = (
        "D-humaneval-greedy (greedy n=1 + in-process exec; relative-to-teacher "
        "gating only), D-humaneval-maxnew (max_new_tokens=2048 for thinking-mode "
        "traces), D-humaneval-limit (HUMANEVAL_LIMIT rebases pass@1 to subset "
        "count). See module docstring for full Deviations section."
    )
    config_key = "stage6_validate.generative.enabled"
    # M1: reads tuple includes the run-scoped ctx side-channels the body
    # actually consumes — device, eval_text_concat, eval_results — plus the
    # mandatory generative-side restore handles published by
    # eval_environment.py (pre_compile_forward, experts_implementation_generative).
    # Failure to declare these here would let the ctx-validation layer reject
    # the slots even though the body legitimately reads them under ``ctx.has(...)``
    # guards.
    reads: tuple[str, ...] = (
        "model",
        "tokenizer",
        "config",
        "dataset_revisions",
        "device",
        "eval_text_concat",
        "eval_results",
        "pre_compile_forward",
        "experts_implementation_generative",
    )
    writes: tuple[str, ...] = ("eval_results",)
    # eval_results is a shared collector the orchestrator pre-creates per side
    # and every eval plugin appends to; it is NOT a calibration-pass accumulator,
    # so it belongs in `writes`, not `provides`. (S6-8 wires the collector.)
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Gate on ``generative.enabled`` AND a ``humaneval`` sub-key.

        Mirrors the monolith ``run()``'s two-level guard: the outer
        ``if s6["generative"]["enabled"]`` block and the inner
        ``if "humaneval" in s6["generative"]`` gate. Uses ``.get()`` chains so
        a missing ``stage6_validate`` or ``generative`` subdict resolves to
        disabled rather than raising.
        """
        generative = (config.get("stage6_validate", {}) or {}).get("generative", {}) or {}
        return bool(generative.get("enabled", False)) and "humaneval" in generative

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def eval_task(self, ctx: PipelineContext) -> None:
        """Phase hook -- Stage 6 HumanEval eval (S6-8 wiring surface).

        INERT at S6-4: no orchestrator walk or test invokes this hook. S6-8
        replaces the Stage 6 orchestrator body with the plugin sequencer and
        dispatches this hook in place of the monolith's inline ``run()``
        HumanEval block. The body below reproduces that inline call site
        faithfully -- it is dead code at S6-4 but S6-8 relies on it once the
        monolith ``run()`` is deleted.

        Reproduces the monolith ``run()``'s student-side call:

            _humaneval_cfg = s6["generative"]["humaneval"]
            _nspt = _humaneval_cfg.get("num_samples_per_task", 1)
            if int(_nspt) != 1:
                raise ValueError(...)            # spec D-humaneval-greedy
            results["student"]["humaneval_pass_at_1"] = _humaneval(
                model, tokenizer, s6["generative"]["humaneval"], device=device,
                collect=eval_text_concat, batch_size=gen_batch_size,
                dataset_revisions=dataset_revisions,
            )

        The result lands in the pre-existing ``eval_results`` ctx slot (the
        analogue of the monolith's ``results["student"]`` dict) under the
        ``humaneval_pass_at_1`` key. This hook does NOT ``ctx.set``
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
        # Item-1: generative metrics (humaneval_pass_at_1, math500_accuracy) are
        # batch-geometry-dependent (bf16 + left-pad reduction drift). Pin the
        # geometry so reported numbers are reproducible run-to-run. Advisory
        # only — smoke runs (HUMANEVAL_LIMIT) legitimately use other geometries,
        # so do NOT raise here.
        if gen_batch_size != PINNED_GEN_BATCH_SIZE:
            log.warning(
                "gen_batch_size=%d differs from the pinned generative geometry "
                "%d; humaneval/math500 numbers are NOT comparable to pinned runs.",
                gen_batch_size, PINNED_GEN_BATCH_SIZE,
            )

        # F-CR2-L-1: schema preservation -- accept `num_samples_per_task` for
        # future operators who may want to ablate, but assert it equals 1.
        # Spec D-humaneval-greedy mandates greedy single-sample pass@1 (NOT
        # Chen-2021-style stochastic pass@1 that requires k>=10 samples).
        _humaneval_cfg = s6["generative"]["humaneval"]
        _nspt = _humaneval_cfg.get("num_samples_per_task", 1)
        if int(_nspt) != 1:
            raise ValueError(
                f"stage6_validate.generative.humaneval.num_samples_per_task must be 1 "
                f"(spec D-humaneval-greedy: greedy single-sample pass@1); got {_nspt}. "
                f"Stochastic pass@1 (Chen 2021) requires a different harness — not supported here."
            )

        # The run-scoped `device` / `eval_text_concat` are optional context
        # side-channels in the plugin world (the monolith threads them through
        # run()); default to None when a wiring stage has not provided them.
        device = ctx.get("device") if ctx.has("device") else None
        collect = ctx.get("eval_text_concat") if ctx.has("eval_text_concat") else None

        log.info("Stage 6: HumanEval (student), batch_size=%d", int(gen_batch_size))

        # C1: Honor the downstream contract published by eval_environment.py
        # (commit 6cca08f, reaffirmed in 7f53280 / 2f4465bc; canonical code
        # block at eval_environment.py L496-524). Before any model.generate(...)
        # on the generative side we MUST:
        #   (1) restore the uncompiled forward (avoids cu130 Inductor recompile
        #       storm on growing cache_position + decode-shape codegen crashes
        #       under torch 2.11+cu130 / Triton 3.4 on Hopper);
        #   (2) switch experts_implementation to the generative impl (the
        #       PPL/lm_eval default grouped_mm crashes on B=1 decode-shape on
        #       cu130 — torch._grouped_mm requires the prefill batch geometry).
        # Mirror of the teacher-side block at teacher_provider.py L602-616.
        # Both slots are optional ctx publications (pre-S6-8 wiring may omit
        # them in tests); fall through quietly when absent.
        _pre = ctx.get("pre_compile_forward") if ctx.has("pre_compile_forward") else None
        if _pre is not None:
            model.forward = _pre
            log.info(
                "Stage 6 HumanEval: restored uncompiled model.forward for "
                "generative block (keep PPL/lm_eval compiled, generative eager)"
            )
        _gen_impl = (
            ctx.get("experts_implementation_generative")
            if ctx.has("experts_implementation_generative")
            else None
        )
        if _gen_impl is not None:
            _cfg = getattr(model, "_orig_mod", model).config
            _current_impl = getattr(_cfg, "_experts_implementation", None)
            if _gen_impl != _current_impl:
                log.info(
                    "Stage 6 HumanEval: switching experts_implementation "
                    "%r → %r for generative block",
                    _current_impl, _gen_impl,
                )
                _set_experts_implementation_s6(model, _gen_impl)

        eval_results = ctx.get("eval_results")
        eval_results["humaneval_pass_at_1"] = _humaneval(
            model, tokenizer, s6["generative"]["humaneval"], device=device,
            collect=collect, batch_size=gen_batch_size,
            dataset_revisions=dataset_revisions,
        )


__all__ = ["_humaneval", "_check_humaneval", "HumanEvalPlugin"]
