"""Stage 6 — Validation (plugin architecture)."""
# Item-2: lazy re-exports (PEP 562). Eagerly doing
# ``from .orchestrator import run`` here pulled torch (and the whole Stage-6
# orchestrator graph) into ANY import of a stage6 submodule — including the
# torch-free ``stage6.plugins._humaneval_worker`` leaf that the HumanEval
# ProcessPool submits under the ``spawn`` start-method. Under spawn each child
# re-imports the worker's defining module by its fully-qualified name, which
# runs THIS package ``__init__``; an eager torch import here would defeat the
# torch-free-worker optimization (H1) and re-pay torch's import cost in every
# child. Deferring the orchestrator import to first attribute access keeps
# ``moe_compress.stage6.run`` / ``...STAGE6`` working exactly as before while
# leaving plain submodule imports torch-free.
from typing import TYPE_CHECKING

__all__ = ["run", "STAGE6"]

if TYPE_CHECKING:  # static analysers / IDEs still see the real symbols
    from .orchestrator import run
    from .stage import STAGE6


def __getattr__(name: str):
    if name == "run":
        from .orchestrator import run
        return run
    if name == "STAGE6":
        from .stage import STAGE6
        return STAGE6
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
