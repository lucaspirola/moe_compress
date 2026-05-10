"""Model-family adapters for kdr.

Public surface:
* `kdr.adapters.base`: `ModelAdapter` Protocol (LLR-0022).
* `kdr.adapters.zaya1_8b`: ZAYA1-8B adapter (LLR-0023).

Adding a new model family is purely additive: drop a new module here that
implements `ModelAdapter` (no edits required to the trainer, the QuantBackend,
or the save logic).
"""
