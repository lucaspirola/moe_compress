"""Torch-free HumanEval scoring worker (Item-2 of the Stage-6 eval refactor).

This is a LEAF module: it imports only stdlib (``re``) and NOTHING that pulls
``torch`` (no stage modules, no ``tools.eval_harness``). That property is
load-bearing: ``stage6/plugins/humaneval.py`` runs HumanEval scoring in a
``ProcessPoolExecutor`` forced onto the ``spawn`` start-method. Under ``spawn``
each child re-imports the DEFINING module of the submitted callable to unpickle
it. If the worker lived in (or imported from) ``tools.eval_harness`` -- which
does ``import torch`` at module top -- every spawned child would re-import
torch, paying a large one-time cost and re-triggering eval_harness module-level
side effects. Keeping the worker here, torch-free, means spawn children pay
only the stdlib + ``re`` import cost.

The pure regex/string code-extraction logic below is an INDEPENDENT COPY of
``tools.eval_harness._extract_code_from_chat_response`` (and its three regex
constants). The two copies must stay in sync -- same duplication discipline as
``_STAGE6_ATTN_IMPLEMENTATION``. Keep in sync with
``tools/eval_harness._extract_code_from_chat_response``.

Security: ``_score_humaneval_one`` runs model-generated Python via the builtin
code-execution primitive. It is invoked only inside a ProcessPool child
(subprocess isolation), which is strictly more isolated than the legacy
in-process daemon thread, but it is NOT a sandbox (no seccomp / landlock /
container). Use only in trusted environments. See ``humaneval.py`` for the
one-time security WARNING.
"""
from __future__ import annotations

import re


# --- COPY of tools/eval_harness regex constants (keep in sync) -------------
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_PY_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
# After a `def <name>(...):` body, the function ends at the first line that
# is non-blank, not indented, and not a `def`/`class`/`async def`/decorator/
# import. That line (and everything after) is trailing prose like
# "This function works by..." -- must be trimmed before running. The negative
# lookahead lets us keep continuations (another top-level def, a decorator,
# imports needed by the body).
_TRAILING_PROSE_RE = re.compile(
    r"\n(?=[A-Za-z][^\n]*$)"            # next line starts with a letter (not indent, not symbol)
    r"(?!def |class |async def |@|from |import )",  # but is not Python top-level construct
    re.MULTILINE,
)


def _extract_code_from_chat_response(text: str, entry_point: str) -> str:
    """Pull executable Python out of a chat-formatted completion.

    COPY of ``tools/eval_harness._extract_code_from_chat_response`` -- kept here
    so the spawn worker stays torch-free (see module docstring). Keep in sync.

    Steps:
      1. Strip any <think>...</think> reasoning block (Qwen thinking-mode).
      2. Prefer a ```python ... ``` fenced block (most common).
      3. Fall back to text starting at the first `def <entry_point>(`,
         trimmed at the first top-level prose line so trailing explanation
         ("This function works by...") doesn't break the run.
      4. Return empty string if no code can be located -> scoring fails.
    """
    s = _THINK_BLOCK_RE.sub("", text)
    m = _PY_FENCE_RE.search(s)
    if m:
        return m.group(1)
    needle = f"def {entry_point}("
    idx = s.find(needle)
    if idx >= 0:
        body = s[idx:]
        trim = _TRAILING_PROSE_RE.search(body)
        return body[:trim.start()] if trim else body
    return ""


def _score_humaneval_one(
    prompt: str, completion: str, test_src: str, entry_point: str,
) -> bool:
    """Score a single HumanEval problem; returns True iff the test passes.

    Pure function of its (picklable str) arguments -- greedy decode makes the
    completion deterministic, so the score is reproducible. Runs inside a
    ProcessPool child: the process IS the isolation + kill boundary, so there
    is NO threading here (the parent enforces the timeout by terminating the
    worker). Mirrors the legacy ``_check_humaneval`` ``src`` construction and
    exception->False semantics exactly.
    """
    code = _extract_code_from_chat_response(completion, entry_point)
    if code and f"def {entry_point}(" in code:
        src = code + "\n" + test_src + f"\ncheck({entry_point})\n"
    else:
        src = prompt + completion + "\n" + test_src + f"\ncheck({entry_point})\n"
    ns: dict = {}
    try:
        exec(src, ns, ns)           # noqa: S102 -- controlled benchmark use
    except Exception:               # noqa: BLE001
        return False
    return True
