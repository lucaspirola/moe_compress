"""Stage 4 plugin implementations.

The Stage 4 EoRA algorithm — residual input collection and √Λ-weighted
residual compensation — is extracted from the legacy ``stage4_eora.py``
monolith into focused plugins here:

* ``eora_inputs`` — ``EoraInputsPlugin``: the EoRA input load (A-cov /
  Stage-3 originals load, file-deleted double-widen guard, ``stage3_ranks``
  snapshot, crash-resume partial-dir setup);
* ``eora_compensation`` — ``EoraCompensationPlugin``: the √Λ-weighted
  per-matrix residual compensation and the in-process double-widen
  ``assert``.

Both plugins are LIVE: ``stage4/orchestrator.run`` (S4-4) builds a
``PluginRegistry`` of the two and drives the ``load_eora_inputs`` →
LOOP[``compensate_layer``] → finalize schedule; ``stage4_eora.run`` is a
thin shim delegating to it.
"""
