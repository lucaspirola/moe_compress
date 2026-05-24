"""S4-3 — EoRA residual-compensation plugin extraction tests.

Verifies the S4-3 ``EoraCompensationPlugin`` scaffolding in
``stage4/plugins/eora_compensation.py``:

* the plugin class and the two relocated standalone functions
  (``_compute_eora_factors``, ``_spill_layer``) import from the plugin module;
* ``EoraCompensationPlugin`` satisfies the universal ``PipelinePlugin``
  Protocol, carries correct metadata, is unconditionally enabled, and exposes
  the (S4-4) ``compensate_layer`` phase hook;
* the module never imports the ``stage4_eora`` monolith or
  ``stage4.orchestrator`` at any scope (the circular-import contract);
* the ``stage4_eora`` monolith re-exports the relocated symbols (the
  ``# noqa: F401`` re-import block) so ``run()`` + external callers/tests keep
  their existing import paths;
* the dtype noise-floor table ``_NOISE_FLOOR_BY_DTYPE`` is relocated to
  ``tools/dtype_noise_floor`` and re-exported by both stage 3 and stage 4.

S4-3 is a MIXED relocation: ``_compute_eora_factors`` / ``_spill_layer`` are
relocated verbatim and the monolith re-imports them (S3-2/S3-3 pattern); the
per-matrix budget + widen loop is reproduced in the inert ``compensate_layer``
hook with the monolith ``run()`` left byte-identical (S4-2 pattern). The
byte-identical behavioral gate is the S4-0 golden snapshot
(``test_stage4_golden_snapshot.py``) plus ``test_eora_bf16_A.py``; this file
checks the relocation plumbing and adds a couple of light kernel sanity
checks that do not duplicate the deep numerics already pinned by
``test_eora_bf16_A.py``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import torch


def test_eora_compensation_module_imports():
    """Plugin class + the two relocated functions import from the plugin module."""
    from moe_compress.stage4.plugins.eora_compensation import (
        EoraCompensationPlugin,
        _compute_eora_factors,
        _spill_layer,
    )

    assert isinstance(EoraCompensationPlugin, type)
    assert callable(_compute_eora_factors)
    assert callable(_spill_layer)


def test_plugin_satisfies_protocol():
    """``EoraCompensationPlugin`` structurally satisfies ``PipelinePlugin``."""
    from moe_compress.pipeline.plugin import PipelinePlugin
    from moe_compress.stage4.plugins.eora_compensation import EoraCompensationPlugin

    assert isinstance(EoraCompensationPlugin(), PipelinePlugin)


def test_plugin_metadata():
    """Plugin metadata — name / paper id / config_key / tuple-typed fields."""
    from moe_compress.stage4.plugins.eora_compensation import EoraCompensationPlugin

    plugin = EoraCompensationPlugin()
    assert plugin.name == "eora_compensation"
    assert "2410.21271" in plugin.paper
    assert plugin.config_key == "stage4_eora.compensation_budget_pct"
    assert isinstance(plugin.reads, tuple)
    assert isinstance(plugin.writes, tuple)
    assert isinstance(plugin.provides, tuple)
    # The hook mutates the shared rank_map dict + advances compensated_params.
    assert plugin.writes == ("rank_map", "compensated_params")


def test_plugin_is_enabled_unconditional():
    """EoRA compensation is UNCONDITIONAL — ``is_enabled`` always True.

    The per-matrix budget calc may compute ``r<=0`` and ``continue`` past an
    individual matrix internally, but the plugin as a whole always runs;
    ``config_key`` only parametrises the per-matrix budget, never the plugin.
    """
    from moe_compress.stage4.plugins.eora_compensation import EoraCompensationPlugin

    plugin = EoraCompensationPlugin()
    assert plugin.is_enabled({}) is True
    assert plugin.is_enabled(
        {"stage4_eora": {"compensation_budget_pct": 0.25,
                         "eigenspace_rank_cap": 16}}
    ) is True


def test_plugin_has_compensate_layer_hook():
    """The S4-4 phase hook ``compensate_layer`` is present and callable."""
    from moe_compress.stage4.plugins.eora_compensation import EoraCompensationPlugin

    plugin = EoraCompensationPlugin()
    assert callable(getattr(plugin, "compensate_layer", None))


def test_no_monolith_import_at_any_scope():
    """The module never imports ``stage4_eora`` / ``stage4.orchestrator``.

    The module docstring's contract says NEVER import the monolith (or the
    orchestrator that S4-4 will make import this module) at any scope —
    module-top OR function-local — since either would risk an import cycle.
    Parse the source with ``ast`` and walk the FULL tree so a function-local
    ``import stage4_eora`` cannot slip past. Assert no ``Import`` /
    ``ImportFrom`` names the forbidden modules at any nesting level.
    """
    from moe_compress.stage4.plugins import eora_compensation as mod

    src = Path(mod.__file__).read_text()
    tree = ast.parse(src)
    forbidden = ("stage4_eora", "stage4.orchestrator")
    for node in ast.walk(tree):  # any nesting level, not just module-top
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(f in alias.name for f in forbidden), (
                    f"forbidden import at any scope: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            assert not any(f in mod_name for f in forbidden), (
                f"forbidden import-from at any scope: {mod_name}"
            )


def test_monolith_reexports_relocated_symbols():
    """The ``stage4_eora`` monolith re-exports the relocated symbols.

    ``run()`` and external callers/tests keep their ``stage4_eora`` import
    paths via the S4-3 ``# noqa: F401`` re-import block — and the re-exported
    objects must be the SAME objects defined in the plugin module.
    """
    import moe_compress.stage4_eora as monolith
    import moe_compress.stage4.plugins.eora_compensation as plugin

    assert monolith._compute_eora_factors is plugin._compute_eora_factors
    assert monolith._spill_layer is plugin._spill_layer


def test_dtype_noise_floor_relocated():
    """``_NOISE_FLOOR_BY_DTYPE`` relocated to ``tools/dtype_noise_floor``.

    It is a dict, and both stage 3 (``aa_svd_factor``) and stage 4
    (``stage3_svd``'s S3-5 re-export) resolve the SAME object.
    """
    from moe_compress.tools.dtype_noise_floor import _NOISE_FLOOR_BY_DTYPE

    assert isinstance(_NOISE_FLOOR_BY_DTYPE, dict)

    import moe_compress.stage3.plugins.aa_svd_factor as aa
    import moe_compress.stage3_svd as s3
    import moe_compress.stage4.plugins.eora_compensation as eora

    assert aa._NOISE_FLOOR_BY_DTYPE is _NOISE_FLOOR_BY_DTYPE
    assert s3._NOISE_FLOOR_BY_DTYPE is _NOISE_FLOOR_BY_DTYPE
    # The stage-4 access path — the module this relocation directly serves.
    assert eora._NOISE_FLOOR_BY_DTYPE is _NOISE_FLOOR_BY_DTYPE


def test_compute_eora_factors_r_zero_early_return():
    """``_compute_eora_factors`` with ``r<=0`` returns empty zero-width factors.

    The early-return path (``r <= 0``) yields ``U`` of shape ``[d_out, 0]``,
    ``V`` of shape ``[0, d_in]`` and ``take_eff == 0`` — no SVD is taken.
    """
    from moe_compress.stage4.plugins.eora_compensation import _compute_eora_factors

    delta = torch.randn(12, 20)
    U, V, take_eff = _compute_eora_factors(delta, None, r=0, device="cpu")
    assert U.shape == (12, 0)
    assert V.shape == (0, 20)
    assert take_eff == 0


def test_compute_eora_factors_shape_stable():
    """``_compute_eora_factors`` returns factors padded to the requested rank.

    The relocated kernel must keep the caller's pre-allocated-tensor contract:
    ``U`` is ``[d_out, r]`` and ``V`` is ``[r, d_in]`` regardless of the
    effective rank, on both the isotropic (A is None) and A-weighted paths.
    Deep numerics are pinned by ``test_eora_bf16_A.py``; this is a light
    shape-stability sanity check on the relocated symbol.
    """
    from moe_compress.stage4.plugins.eora_compensation import _compute_eora_factors

    torch.manual_seed(0)
    d_out, d_in, r = 16, 24, 4
    delta = torch.randn(d_out, d_in)
    A = torch.randn(d_in, d_in)
    A = A @ A.T  # SPD covariance

    U_iso, V_iso, eff_iso = _compute_eora_factors(delta, None, r, "cpu")
    assert U_iso.shape == (d_out, r)
    assert V_iso.shape == (r, d_in)
    assert 0 <= eff_iso <= r

    U_a, V_a, eff_a = _compute_eora_factors(delta, A, r, "cpu")
    assert U_a.shape == (d_out, r)
    assert V_a.shape == (r, d_in)
    assert 0 <= eff_a <= r
