"""Ordered registry of Stage 2 plugins.

Insertion order is the execution order. ``active(cfg)`` instantiates the
subset whose ``is_enabled(cfg)`` returns True. ``dispatch_first`` is the
slot-style helper for hooks like ``compute_cost`` / ``solve_assignment``
where exactly one plugin should win per layer.
"""
from __future__ import annotations

from typing import Any

from .base import Stage2Plugin


class PluginRegistry:
    """Mutable, ordered list of plugin classes."""

    def __init__(self) -> None:
        self._classes: list[type[Stage2Plugin]] = []

    def register(self, plugin_cls: type[Stage2Plugin]) -> type[Stage2Plugin]:
        """Append a plugin class. Returns the class so this can be used as a decorator."""
        if not isinstance(plugin_cls, type) or not issubclass(plugin_cls, Stage2Plugin):
            raise TypeError(
                f"register() expects a Stage2Plugin subclass, got {plugin_cls!r}"
            )
        self._classes.append(plugin_cls)
        return plugin_cls

    def classes(self) -> list[type[Stage2Plugin]]:
        """Return the registered classes in insertion order (defensive copy)."""
        return list(self._classes)

    def active(self, cfg: dict[str, Any]) -> list[Stage2Plugin]:
        """Instantiate plugins whose ``is_enabled(cfg)`` returns True, preserving order."""
        return [cls() for cls in self._classes if cls.is_enabled(cfg)]

    @staticmethod
    def dispatch_first(
        plugins: list[Stage2Plugin],
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any | None:
        """Call ``method_name`` on each plugin in order; return the first non-None result."""
        for plugin in plugins:
            hook = getattr(plugin, method_name)
            result = hook(*args, **kwargs)
            if result is not None:
                return result
        return None
