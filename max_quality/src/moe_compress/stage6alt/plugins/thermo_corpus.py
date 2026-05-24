"""Thermometer calibration-corpus build (S6A-2 of the Stage 6alt plugin-architecture refactor).

Paper / dataset
----------------
Stage 6alt thermometer corpus: ONE of two evaluation corpora —

  - **Nemotron held-out slice** (default) — same Nemotron-Cascade-2-SFT-Data
    distribution as Stage-2 / 2.5 / 5 calibration (D11 — owner
    :mod:`stage2.plugins.reap_scoring`), drawn with a distinct
    ``seed_offset`` so the thermometer eval is disjoint from
    calibration.
  - **WikiText-2 test split** — Merity et al. 2017 arXiv:1609.07843,
    same source as :mod:`stage6.plugins.wikitext_ppl` but the
    **test** split (Stage 6 uses test too; the thermometer uses the
    same chunked form).

Project-original sweep harness; no upstream thermometer paper.

Home of the Stage 6alt thermometer calibration-corpus concern, extracted
from the legacy ``stage6alt_thermometer.py`` monolith. The thermometer
selects ONE of two evaluation corpora — the nemotron held-out slice
(default) or the WikiText-2 test split — via the ``thermometer.corpus``
config key, and builds the ``(num_seqs, seq_len)`` int64 tensor that
``_bpt_from_nll`` scores.

Pattern A vs Pattern B
----------------------
S6A-2's corpus slice covers a MIXED pattern:

* **Pattern A — relocated verbatim**: the five standalone symbols below
  (``THERMO_SEED_OFFSET``, ``_DEFAULT_SUBSET_WEIGHTS``,
  ``_thermo_corpus_spec``, ``_thermo_wikitext_tensor``,
  ``_build_thermo_corpus``) are character-identical copies of the monolith
  bodies. ``stage6alt_thermometer.py`` re-imports them (the ``# noqa: F401``
  block) so ``run()`` and external callers/tests (notably
  ``stage2/orchestrator.py``'s xD calibration path that imports
  ``_thermo_wikitext_tensor``) keep their existing import paths.
* **Pattern B — reproduced in an inert hook**: the monolith ``run()``'s
  inline corpus-build call site (the ``_build_thermo_corpus(config,
  tokenizer, artifacts_dir)`` call that returns
  ``(calib, corpus_meta, corpus_id)``) is reproduced in the inert
  ``build_corpus`` hook below. The monolith ``run()`` is NOT modified for
  it. This is an intentional, temporary logic duplication that resolves at
  S6A-6 when the orchestrator flip wires this hook live and the monolith
  ``run()`` becomes a thin shim.

Circular-import contract (mirror of ``stage6/plugins/wikitext_ppl.py``):
this module imports only from ``..context`` / ``...utils.calibration`` /
stdlib / torch — NEVER from ``stage6alt_thermometer`` or
``stage6alt.orchestrator`` at any scope (module-top OR function-local).
The monolith re-imports *this* module's symbols at load time, so a
``from ..stage6alt_thermometer import ...`` here would deadlock the
import; nothing in this module does that.

``ThermoCorpusPlugin`` is registered-but-INERT at S6A-2 — no orchestrator
walk or test invokes its ``build_corpus`` hook. S6A-6 plugs the hook into
the live Stage 6alt plugin sequencer.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

from ..context import PipelineContext
from ...utils.calibration import (
    CalibrationSpec,
    build_calibration_tensor,
    shared_calibration_cache_dir,
    spec_from_config,
)

log = logging.getLogger(__name__)


# Held-out draw: shifts the calibration seed so the thermometer's eval
# sequences do not overlap the Stage 2/2.5 training draw. Bumping this value
# changes the effective seed inside CalibrationSpec.cache_key, which in turn
# changes _thermo_teacher_cache_key — so the teacher cache auto-invalidates.
THERMO_SEED_OFFSET = 715


# Default eval subset mix — reasoning-heavy, independent of the chat-dominant
# calibration.subset_weights used for compression. Overridable via the
# `thermometer.subset_weights` config key.
_DEFAULT_SUBSET_WEIGHTS = {"math": 0.35, "swe": 0.25, "chat": 0.25, "science": 0.15}


# ---------------------------------------------------------------------------
# Corpus spec
# ---------------------------------------------------------------------------


def _thermo_corpus_spec(config: dict) -> CalibrationSpec:
    """Build the held-out CalibrationSpec for the thermometer corpus.

    Copies `config["calibration"]`, overlays the thermometer's own
    `subset_weights` (reasoning-heavy, not the chat-dominant training mix),
    and applies `THERMO_SEED_OFFSET` so the draw is disjoint from Stage 2/2.5.
    Never mutates `config["calibration"]` — Stage 2/2.5 read it.
    """
    therm = config.get("stage6_validate", {}).get("thermometer", {}) or {}
    cal_cfg = dict(config["calibration"])  # shallow copy — we replace one key
    cal_cfg["subset_weights"] = dict(
        therm.get("subset_weights") or _DEFAULT_SUBSET_WEIGHTS
    )
    return spec_from_config(
        cal_cfg,
        num_sequences_override=int(therm.get("num_sequences", 64)),
        sequence_length_override=int(therm.get("sequence_length", 2048)),
        seed_offset=THERMO_SEED_OFFSET,
    )


def _thermo_wikitext_tensor(tokenizer, *, num_sequences: int,
                            sequence_length: int, dataset: str, subset: str,
                            split: str) -> torch.Tensor:
    """Build the first `num_sequences` full-length chunks of WikiText.

    Mirrors `stage6_validate._wikitext2_ppl`'s tokenization exactly: rows are
    concatenated with "\\n\\n", the whole corpus is tokenized in one call with
    `add_special_tokens=True` (BOS applied once), then chunked into
    `sequence_length`-token rows. The chunk order is fixed by the dataset, so
    the draw is fully deterministic. WikiText test text is not in the Stage 2/
    2.5 training distribution, so no seed-offset disjointness logic is needed.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, subset, split=split)
    concatenated = "\n\n".join(row.get("text", "") for row in ds)
    all_ids = tokenizer(
        concatenated, add_special_tokens=True, return_tensors=None,
    )["input_ids"]
    n_full = len(all_ids) // sequence_length
    if n_full == 0:
        raise RuntimeError(
            f"thermometer wikitext corpus: {dataset}/{subset}:{split} has no "
            f"full {sequence_length}-token sequence."
        )
    take = min(num_sequences, n_full)
    if take < num_sequences:
        log.warning("thermometer wikitext: only %d full sequences available "
                    "(< %d requested) — using %d", n_full, num_sequences, take)
    return torch.tensor(
        all_ids[: take * sequence_length], dtype=torch.long
    ).view(take, sequence_length)


def _build_thermo_corpus(config: dict, tokenizer, artifacts_dir: Path):
    """Build the thermometer's evaluation corpus.

    Returns `(calib_ids, corpus_meta, corpus_id)`:
      - `calib_ids`: `(num_seqs, seq_len)` int64 tensor for `_bpt_from_nll`.
      - `corpus_meta`: JSON-able dict recorded in `stage6alt_eval.json`.
      - `corpus_id`: stable string folded into the teacher cache key so a
        corpus switch (nemotron <-> wikitext, or a spec change) auto-
        invalidates the sweep-shared teacher cache.

    Selected by `thermometer.corpus` ("nemotron" default, or "wikitext").
    See the module docstring for why the choice changes how `bpt_gap` is read.
    """
    therm = config.get("stage6_validate", {}).get("thermometer", {}) or {}
    corpus = str(therm.get("corpus", "nemotron")).lower()
    seq_len = int(therm.get("sequence_length", 2048))
    n_seq = int(therm.get("num_sequences", 64))
    # Class-qualified fallback so a tokenizer that lacks name_or_path (e.g. an
    # in-memory instance) doesn't yield a tokenizer-blind corpus_id — mirrors
    # build_calibration_tensor's defensive identity.
    tok_id = (getattr(tokenizer, "name_or_path", None)
              or f"{tokenizer.__class__.__module__}."
                 f"{tokenizer.__class__.__name__}")

    if corpus == "wikitext":
        wt = therm.get("wikitext", {}) or {}
        dataset = wt.get("dataset", "wikitext")
        subset = wt.get("subset", "wikitext-2-raw-v1")
        split = wt.get("split", "test")
        calib = _thermo_wikitext_tensor(
            tokenizer, num_sequences=n_seq, sequence_length=seq_len,
            dataset=dataset, subset=subset, split=split,
        )
        corpus_meta = {
            "name": "wikitext", "dataset": dataset, "subset": subset,
            "split": split, "num_sequences": int(calib.shape[0]),
            "sequence_length": seq_len,
        }
        corpus_id = (f"wikitext:{dataset}:{subset}:{split}:"
                     f"{calib.shape[0]}x{seq_len}:{tok_id}")
        log.info("Stage 6alt corpus: wikitext (%s/%s:%s) %d x %d",
                 dataset, subset, split, calib.shape[0], seq_len)
        return calib, corpus_meta, corpus_id

    if corpus == "nemotron":
        spec = _thermo_corpus_spec(config)
        calib = build_calibration_tensor(
            tokenizer, spec,
            cache_dir=(os.environ.get("MOE_CALIB_CACHE_DIR") or shared_calibration_cache_dir(artifacts_dir)),
        )
        corpus_meta = {
            "name": "nemotron",
            "num_sequences": spec.num_sequences,
            "sequence_length": spec.sequence_length,
            "effective_seed": spec.seed,
            "seed_offset": THERMO_SEED_OFFSET,
            "subset_weights": spec.subset_weights,
        }
        corpus_id = f"nemotron:{spec.cache_key(tok_id)}"
        log.info("Stage 6alt corpus: nemotron (held-out slice) %d x %d "
                 "— bpt_gap is RANKING-ONLY (Stage-2.5 adaptation confound)",
                 spec.num_sequences, spec.sequence_length)
        return calib, corpus_meta, corpus_id

    raise ValueError(
        f"thermometer.corpus must be 'nemotron' or 'wikitext', got {corpus!r}"
    )


class ThermoCorpusPlugin:
    """Stage 6alt thermometer calibration-corpus plugin (S6A-2 — registered-but-INERT).

    Owns the Stage 6alt corpus-build concern: the relocated
    ``_thermo_corpus_spec`` / ``_thermo_wikitext_tensor`` / ``_build_thermo_corpus``
    helpers (Pattern A) plus an inert ``build_corpus`` hook (Pattern B) that
    reproduces the monolith's inline corpus-build call site.

    S6A-2 wires this class into the plugin registry as metadata only — no
    orchestrator walk or test invokes ``build_corpus``. S6A-6 plugs the hook
    into the live Stage 6alt plugin sequencer.
    """

    name = "thermo_corpus"
    paper = "Stage 6alt thermometer corpus build — Nemotron held-out or WikiText-2 test (project-original sweep harness; D11 calibration deviation owner is :mod:`stage2.plugins.reap_scoring`). See module docstring."
    config_key = "stage6_validate.thermometer.corpus"
    reads: tuple[str, ...] = (
        "model", "tokenizer", "config", "artifacts_dir",
    )
    writes: tuple[str, ...] = ("calib_ids", "corpus_meta", "corpus_id")
    # No calibration-pass accumulator — `_build_thermo_corpus` already
    # produces the tensor in one call, no per-layer sweep is needed.
    provides: tuple[str, ...] = ()

    writes: tuple[str, ...] = ("calib_ids", "corpus_meta", "corpus_id")
    # No calibration-pass accumulator — `_build_thermo_corpus` already
    # produces the tensor in one call, no per-layer sweep is needed.
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — every thermometer run must build an eval corpus.

        ``config_key`` only names *which* corpus is built (nemotron vs
        wikitext); it never gates the plugin as a whole.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def build_corpus(self, ctx: PipelineContext) -> None:
        """Phase hook — Stage 6alt thermometer corpus build (S6A-6 wiring surface).

        INERT at S6A-2: no orchestrator walk or test invokes this hook. S6A-6
        replaces the Stage 6alt orchestrator body with the plugin sequencer
        and dispatches this hook in place of the monolith ``run()``'s inline
        ``_build_thermo_corpus`` call. The body below reproduces that inline
        call faithfully — it is dead code at S6A-2 but S6A-6 relies on it
        once the monolith ``run()`` becomes a thin shim.

        Reproduces the monolith ``run()``'s student-side call:

            calib, corpus_meta, corpus_id = _build_thermo_corpus(
                config, tokenizer, artifacts_dir,
            )

        The three return values are written to ``calib_ids`` / ``corpus_meta``
        / ``corpus_id`` ctx slots.
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise.
        config = ctx.get("config")
        tokenizer = ctx.get("tokenizer")
        artifacts_dir = ctx.get("artifacts_dir")

        calib, meta, cid = _build_thermo_corpus(config, tokenizer, artifacts_dir)

        ctx.set("calib_ids", calib)
        ctx.set("corpus_meta", meta)
        ctx.set("corpus_id", cid)


__all__ = [
    "THERMO_SEED_OFFSET",
    "_DEFAULT_SUBSET_WEIGHTS",
    "_thermo_corpus_spec",
    "_thermo_wikitext_tensor",
    "_build_thermo_corpus",
    "ThermoCorpusPlugin",
]
