"""Meta-test: every Pydantic config class has `strict=True` and `extra='forbid'`
(HLR-0012 AC #2).

Walks `kdr.config`, `kdr.quant.specs`, and `kdr.quant.interface` and asserts
the property on every `BaseModel` subclass. Adding a new model without the
correct `ConfigDict` would silently pass mypy and the per-model tests but
fail this gate.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest
from pydantic import BaseModel

# Modules to walk. `kdr` itself is the package root; pkgutil walks submodules.
import kdr


def _all_basemodel_subclasses() -> list[type[BaseModel]]:
    """Discover every BaseModel subclass under `kdr.*`.

    Uses pkgutil to walk submodules — adding a new model file is automatically
    included without test edits.
    """
    found: list[type[BaseModel]] = []
    seen: set[str] = set()
    for module_info in pkgutil.walk_packages(kdr.__path__, prefix="kdr."):
        try:
            module = importlib.import_module(module_info.name)
        except Exception:  # pragma: no cover — skip stubs that NotImplementedError on import
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(cls, BaseModel)
                and cls is not BaseModel
                and cls.__module__.startswith("kdr.")
                and cls.__qualname__ not in seen
            ):
                found.append(cls)
                seen.add(cls.__qualname__)
    return found


@pytest.mark.parametrize("model_cls", _all_basemodel_subclasses(), ids=lambda c: c.__qualname__)
def test_every_basemodel_has_strict_and_extra_forbid(model_cls: type[BaseModel]) -> None:
    """Per HLR-0012 AC #2: every Pydantic config class in `kdr.*` SHALL set
    both `strict=True` AND `extra='forbid'`. Either alone is insufficient
    (Pydantic's `strict` does not imply `extra='forbid'` and vice versa).
    """
    config = model_cls.model_config
    assert config.get("strict") is True, (
        f"{model_cls.__qualname__} missing `strict=True` in `model_config`"
    )
    assert config.get("extra") == "forbid", (
        f"{model_cls.__qualname__} missing `extra='forbid'` in `model_config`; "
        f"got extra={config.get('extra')!r}"
    )


def test_at_least_one_basemodel_subclass_was_discovered() -> None:
    """Guard against the discovery walking zero classes (would silently make
    the parametrized test above a no-op).
    """
    classes = _all_basemodel_subclasses()
    assert len(classes) >= 5, f"discovered too few BaseModel subclasses: {classes}"
