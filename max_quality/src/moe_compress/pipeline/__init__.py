"""Universal pipeline framework package.

This package holds the mechanism shared by all 7 compression stages: a single
plugin-based pipeline framework that every stage configures rather than
re-implements. The planned modules — ``plugin.py`` (the ``PipelinePlugin``
Protocol + ``BasePlugin``), ``context.py`` (the ``PipelineContext`` data
carrier), ``registry.py`` (the ``PluginRegistry``), and ``stage.py`` (the stage
orchestrator) — are added incrementally by refactor tasks F-1..F-4, so this
``__init__`` deliberately re-exports nothing while the package is partial.
"""
