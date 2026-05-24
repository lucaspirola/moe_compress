"""Plugin-shared logic package.

This package holds logic that is *reused by ≥2 plugins or ≥2 stages* but is
not part of the pipeline framework mechanism itself. It sits between two
neighbours:

* ``pipeline/`` — the framework *mechanism* (the ``PipelinePlugin`` Protocol,
  ``PipelineContext``, ``PluginRegistry``, the stage orchestrator). ``tools/``
  may import ``pipeline/``; ``pipeline/`` must never import ``tools/``. The
  import direction is one-way (``tools/`` → ``pipeline/``).
* ``utils/`` — the low-level substrate (tensor helpers, I/O, hooks). ``tools/``
  is higher-level: it composes framework + substrate into reusable plugin
  building blocks.

The package is filled incrementally by refactor tasks F-5..F-8. Task F-5 lands
``phase_walker.py`` (the reflective phase scheduler) and ``artifact_builder.py``
(the schema-parametric artifact assembler). This ``__init__`` deliberately
re-exports nothing — callers import the concrete modules directly.
"""
