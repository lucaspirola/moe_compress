"""HumanEval pass@1 generative eval (S6-4 of the Stage 6 plugin refactor).

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
module imports only from ``..context`` / ``...tools.eval_harness`` / stdlib --
NEVER from ``stage6_validate``, ``stage6.orchestrator`` or ``orchestrator`` at
any scope (module-top OR function-local). The monolith re-imports *this* module
at load time, so a ``from ..stage6_validate import ...`` here would deadlock
the import; nothing in this module does that.

**Security note -- HumanEval code execution (H1):** ``_check_humaneval`` runs
model-generated Python inside a daemon thread with a wall-clock timeout. This
is best-effort sandboxing only -- no process isolation. See the monolith
docstring for the full caveat list. Use only in trusted environments.

``HumanEvalPlugin`` is registered-but-INERT at S6-4 -- no orchestrator walk or
test invokes its ``eval_task`` hook. S6-8 plugs the hook into the live Stage 6
plugin sequencer and deletes the monolith ``run()``.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

from ..context import PipelineContext
from ...tools.eval_harness import (
    _chat_format_prompts,
    _extract_code_from_chat_response,
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
    # _check_humaneval executes model-generated Python via exec() in a daemon thread
    # with a wall-clock timeout.  This is best-effort sandboxing only — no process
    # isolation (no subprocess, no seccomp, no container boundary).  Runaway or
    # malicious generated code can access the filesystem, network, and interpreter
    # state.  Use only in trusted environments or behind an external sandbox.
    log.warning(
        "HumanEval: executing model-generated code via exec() for %d problems "
        "— best-effort sandboxed via daemon threads with %.0fs timeout each; "
        "no process isolation.",
        len(prompts), exec_timeout_secs,
    )
    completions = _generate_batched(
        model, tokenizer, prompts, max_new=max_new,
        device=device, batch_size=batch_size,
    )

    passes = 0
    total = len(prompts)
    leaked_counter = [0]  # mutable box so _check_humaneval can increment it
    # Pair RAW stub (for exec fallback) with the model's chat-formatted reply.
    # `_check_humaneval` will prefer code extracted from the reply when the
    # reply contains `def <entry_point>(...)`; else falls back to stub + reply.
    for i, (raw_stub, completion, test, ep) in enumerate(
        zip(raw_prompts, completions, tests, entry_points)
    ):
        if _check_humaneval(
            raw_stub, completion, test, ep,
            exec_timeout_secs=exec_timeout_secs,
            _leaked_counter=leaked_counter,
            _problem_index=i,
        ):
            passes += 1
        if (i + 1) % 16 == 0:
            log.info("  HumanEval eval %d/%d (pass=%d)", i + 1, total, passes)
    if leaked_counter[0]:
        log.warning(
            "HumanEval: %d exec threads leaked (daemon threads; will be killed at interpreter exit)",
            leaked_counter[0],
        )
    log.info("  HumanEval final: %d/%d = %.3f", passes, total, passes / max(total, 1))
    return passes / max(total, 1)


def _check_humaneval(
    prompt: str, completion: str, test_src: str, entry_point: str,
    *, exec_timeout_secs: int = 10,
    _leaked_counter: list | None = None,
    _problem_index: int = 0,
) -> bool:
    # H1 — Security note (debug-level; outer _humaneval emits one WARNING for
    # the full eval before the loop starts).  exec() is used here inside a
    # daemon thread; the outer function owns the one-time security log.
    log.debug(
        "HumanEval problem %d: running in daemon thread, timeout=%.0fs",
        _problem_index, exec_timeout_secs,
    )
    # Chat-formatted completions wrap reasoning + code in markdown / think
    # tags. _extract_code_from_chat_response strips <think>...</think> and
    # prefers ```python fences. If the extracted code already contains a full
    # `def <entry_point>(`, run it standalone; otherwise fall back to the
    # legacy `stub + body` concatenation (preserves the raw-completion path
    # used by non-chat models in earlier runs).
    code = _extract_code_from_chat_response(completion, entry_point)
    if code and f"def {entry_point}(" in code:
        src = code + "\n" + test_src + f"\ncheck({entry_point})\n"
    else:
        src = prompt + completion + "\n" + test_src + f"\ncheck({entry_point})\n"
    ns: dict = {}
    _exc_holder: list = []

    def _exec_target() -> None:
        try:
            exec(src, ns, ns)           # noqa: S102 — controlled benchmark use
        except Exception as _e:         # noqa: BLE001
            _exc_holder.append(_e)

    _t = threading.Thread(target=_exec_target, daemon=True)
    _t.start()
    _t.join(timeout=exec_timeout_secs)
    if _t.is_alive():
        # Thread leaked (daemon — will die with process); count as failure.
        if _leaked_counter is not None:
            _leaked_counter[0] += 1
            log.warning(
                "HumanEval exec timed out for problem %d (%d leaked threads total)",
                _problem_index, _leaked_counter[0],
            )
        return False
    if _exc_holder:
        return False
    return True


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
    paper = (
        "HumanEval pass@1 -- Chen et al. 2021, Evaluating Large Language "
        "Models Trained on Code (arXiv:2107.03374); Stage 6 validation gate, "
        "generative half, Spec D-humaneval-greedy (greedy single-sample)."
    )
    config_key = "stage6_validate.generative.enabled"
    reads: tuple[str, ...] = ("model", "tokenizer", "config", "dataset_revisions")
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
        eval_results = ctx.get("eval_results")
        eval_results["humaneval_pass_at_1"] = _humaneval(
            model, tokenizer, s6["generative"]["humaneval"], device=device,
            collect=collect, batch_size=gen_batch_size,
            dataset_revisions=dataset_revisions,
        )


__all__ = ["_humaneval", "_check_humaneval", "HumanEvalPlugin"]
