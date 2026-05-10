"""Tests for `kdr.io.run_id` (LLR-0031).

# VERIFIES: LLR-0031
"""

from __future__ import annotations

import yaml

from kdr.config import Config
from kdr.io.run_id import canonical_yaml_dump, derive_run_id

_BASE_YAML = """
mode: bf16
teacher:
  name_or_path: Zyphra/ZAYA1-reasoning-base
  revision: main
  torch_dtype: bfloat16
  attn_implementation: sdpa
student:
  source: Zyphra/ZAYA1-reasoning-base
  torch_dtype: bfloat16
  attn_implementation: sdpa
calibration:
  source: nvidia-cascade
  dataset: nvidia/Nemotron-Cascade-2-SFT-Data
  seed: 1337
  num_sequences: 13000
  sequence_length: 4096
  subset_weights: { math: 0.21, code: 0.79 }
  ptq_subset_size: 256
distillation:
  loss: forward_kld
  temperature: 1.0
  optimizer: adamw_bnb_8bit
  learning_rate: 1.0e-5
  min_learning_rate: 1.0e-6
  weight_decay: 0.0
  betas: [0.9, 0.95]
  grad_clip_norm: 1.0
  warmup_steps: 50
  total_tokens: 50000000
  per_device_batch_size: 1
  gradient_accumulation: 4
  sequence_length: 4096
  log_every_n_steps: 10
  eval_every_n_steps: 100
  save_every_n_steps: 100
  trainable_scope: full
  use_gradient_checkpointing: true
"""


def _config(**overrides: object) -> Config:
    raw = yaml.safe_load(_BASE_YAML)
    # Apply dotted-path overrides: `distillation.total_tokens=...` etc.
    for key, val in overrides.items():
        parts = key.split(".")
        target = raw
        for p in parts[:-1]:
            target = target[p]
        target[parts[-1]] = val
    return Config.model_validate(raw)


# REQ: VERIFIES: LLR-0031
def test_derive_run_id_returns_16_hex_chars() -> None:
    """LLR-0031 hash format: first 16 hex chars of sha256."""
    rid = derive_run_id(_config(), "deadbeef" * 5, "bf16")
    assert len(rid) == 16
    assert all(c in "0123456789abcdef" for c in rid)


# REQ: VERIFIES: LLR-0031
def test_derive_run_id_is_deterministic_across_calls() -> None:
    """Same inputs → same hash. (Stronger: same machine; LLR-0031 AC #3 also
    requires this across machines, which holds because Pydantic v2's class-
    declaration-order JSON dump is stable.)"""
    cfg = _config()
    sha = "abc123def456" * 3
    a = derive_run_id(cfg, sha, "da_qad")
    b = derive_run_id(cfg, sha, "da_qad")
    assert a == b


# REQ: VERIFIES: LLR-0031
def test_derive_run_id_changes_on_mode() -> None:
    """A mode change must yield a different run_id (LLR-0031 prevents cross-
    mode resume contamination)."""
    cfg = _config()
    a = derive_run_id(cfg, "x" * 40, "bf16")
    b = derive_run_id(cfg, "x" * 40, "da_qad")
    assert a != b


# REQ: VERIFIES: LLR-0031
def test_derive_run_id_changes_on_student_sha() -> None:
    """A student_sha change → different run_id (different student fingerprint)."""
    cfg = _config()
    a = derive_run_id(cfg, "x" * 40, "bf16")
    b = derive_run_id(cfg, "y" * 40, "bf16")
    assert a != b


# REQ: VERIFIES: LLR-0031
def test_derive_run_id_changes_on_semantic_config_field() -> None:
    """LLR-0031 AC #2: changing `total_tokens` (or `bits`, etc.) yields a
    different hash. The hash must encode every semantic field — otherwise
    two semantically distinct jobs could share a run_id and partials."""
    a = derive_run_id(_config(), "x" * 40, "bf16")
    b = derive_run_id(
        _config(**{"distillation.total_tokens": 100_000_000}),
        "x" * 40,
        "bf16",
    )
    assert a != b


# REQ: VERIFIES: LLR-0031
def test_derive_run_id_invariant_to_yaml_whitespace() -> None:
    """LLR-0031 AC #1: whitespace and key-order changes in YAML do NOT change
    the hash because canonical_yaml_dump operates on the parsed model."""
    yaml_a = """
    mode: bf16
    teacher:
      name_or_path: Zyphra/ZAYA1-reasoning-base
      revision: main
      torch_dtype: bfloat16
      attn_implementation: sdpa
    student: { source: Zyphra/ZAYA1-reasoning-base, torch_dtype: bfloat16, attn_implementation: sdpa }
    calibration:
      source: nvidia-cascade
      dataset: nvidia/Nemotron-Cascade-2-SFT-Data
      seed: 1337
      num_sequences: 13000
      sequence_length: 4096
      subset_weights: { math: 0.21, code: 0.79 }
      ptq_subset_size: 256
    distillation:
      loss: forward_kld
      temperature: 1.0
      optimizer: adamw_bnb_8bit
      learning_rate: 1.0e-5
      min_learning_rate: 1.0e-6
      weight_decay: 0.0
      betas: [0.9, 0.95]
      grad_clip_norm: 1.0
      warmup_steps: 50
      total_tokens: 50000000
      per_device_batch_size: 1
      gradient_accumulation: 4
      sequence_length: 4096
      log_every_n_steps: 10
      eval_every_n_steps: 100
      save_every_n_steps: 100
      trainable_scope: full
      use_gradient_checkpointing: true
    """
    cfg_a = Config.model_validate(yaml.safe_load(yaml_a))
    cfg_b = _config()  # Base form, same semantics.
    sha = "x" * 40
    assert derive_run_id(cfg_a, sha, "bf16") == derive_run_id(cfg_b, sha, "bf16")


# REQ: VERIFIES: LLR-0031
def test_derive_run_id_field_separator_blocks_alias_collision() -> None:
    """LLR-0031 AC #4: the \\x00 separator prevents the collision class
    where two (yaml, sha, mode) tuples concatenate to the same byte string.

    We construct two configs that share the same JSON dump *prefix-suffix*
    (semantically different) and verify their hashes differ. With a naive
    concat (no separator), `("ab", "cde", "f")` and `("abc", "de", "f")`
    would produce identical concatenations; with `\\x00` they don't.
    """
    cfg = _config()
    # Synthetic: simulate the AC's "abcd" + "ef" + "g" vs "abc" + "def" + "g"
    # collision class by varying the SHA strings with overlapping prefixes
    # — without the separator, the concatenations would be `abcdefg` for
    # both. With the separator, the embedded `\x00`s break the alias.
    a = derive_run_id(cfg, "abcd", "ef")
    b = derive_run_id(cfg, "abc", "def")
    # Note: "ef" and "def" are not valid `Mode` literals at the type level
    # but `derive_run_id` accepts arbitrary strings — the test exercises
    # the hash-input separator, not mode validation.
    assert a != b
    # And the canonical_yaml_dump itself does not contain a literal `\x00`,
    # so the separators in the hash input are exactly two.
    dump = canonical_yaml_dump(cfg)
    assert "\x00" not in dump


# REQ: VERIFIES: LLR-0031
def test_canonical_yaml_dump_is_compact_single_line() -> None:
    """The dump uses indent=None → single line, no trailing whitespace."""
    out = canonical_yaml_dump(_config())
    assert "\n" not in out
    assert out == out.strip()


# REQ: VERIFIES: LLR-0031
def test_derive_run_id_cross_machine_golden() -> None:
    """LLR-0031 AC #3 (verbatim): "Same (config, sha, mode) triple → same hash
    on different machines (verified by a deterministic unit test with
    hard-coded inputs and a hard-coded expected hash)."

    The golden value was computed once on this machine; if Pydantic v2's
    ``model_dump_json`` field ordering ever drifts (e.g., across a Pydantic
    upgrade or a Config field reorder), this test surfaces the regression
    before partials get orphaned by the new run_id.

    Updating the golden requires intentional acknowledgement that all
    in-flight runs against the prior hash are losing their resume seeds.
    """
    rid = derive_run_id(_config(), "a" * 40, "bf16")
    assert rid == "0842bde6e350eb3b", (
        f"run_id drift detected — got {rid!r}. If intentional (Config "
        "schema change), update this golden value AND verify no in-flight "
        "vast.ai runs against the prior hash will lose their partials."
    )
