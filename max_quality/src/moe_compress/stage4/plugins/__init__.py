"""Stage 4 plugin implementations.

The Stage 4 EoRA algorithm — residual input collection and √Λ-weighted
residual compensation — is extracted from the legacy ``stage4_eora.py``
monolith into focused plugins here by tasks S4-2..S4-3:

* ``eora_inputs`` (S4-2, done) — ``EoraInputsPlugin``: the EoRA input load
  (A-cov / Stage-3 originals load, file-deleted double-widen guard,
  ``stage3_ranks`` snapshot, crash-resume partial-dir setup);
* ``eora_compensation`` (S4-3, planned) — the √Λ-weighted per-matrix
  residual compensation and the in-process double-widen ``assert``.

No plugin manifest exists yet — the plugins are registered-but-INERT until
S4-4 wires the live Stage 4 plugin sequencer and deletes the monolith.
"""
