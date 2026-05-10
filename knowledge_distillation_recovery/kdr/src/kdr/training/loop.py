"""Mode-aware FKLD training loop.

Direct port from `structural_recovery/distillation.py:353-719`, simplified
where structural_recovery has accumulated max_quality-specific concerns
(MoE-factored topology, `iter_moe_layers`-based scope toggling).

Mode dispatch (LLR-0007):
  * `bf16`: skip the QuantBackend factory entirely; the student stays in
    BF16 and the loop is structural_recovery's existing behaviour.
  * `da_qad`: call `quant.factory.partition_and_dispatch(...)` exactly once,
    INSIDE the `activate_zero3_init` context that's still open from the
    teacher/student load (LLR-0048 step 4) and BEFORE the optimizer is
    constructed.

Required call order under ZeRO-3 (LLR-0048, all enforced here):
  1. open `activate_zero3_init(accelerator)` context
  2. inside: adapter.load_teacher_and_student
  3. inside: (da_qad only) partition_and_dispatch
  4. exit context
  5. build_optimizer → accelerator.prepare(student, optim)
  6. iterate the FKLD loop
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase

from ..adapters.base import ModelAdapter
from ..adapters.router_replay import (
    NoOpReplayContextManager,
    RouterReplayHookProtocol,
)
from ..config import Config
from ..eval.quick import run as eval_run
from ..io.save import SAVE_COMPLETE_SENTINEL, save_kdr_artifact, save_partial
from ..kd_loss import forward_kld_loss
from ..quant.interface import QuantBackend
from .optim import build_optimizer, cosine_with_warmup, set_lr
from .zero3_init import activate_zero3_init, is_deepspeed

log = logging.getLogger(__name__)


# REQ: LLR-0007
def run_recovery(
    config: Config,
    adapter: ModelAdapter,
    accelerator: Accelerator,
    artifacts_dir: Path,
    *,
    batches: list[torch.Tensor],
    resume_step: int = 0,
    source_metadata_path: Path | None = None,
) -> Path:
    """Top-level kdr training entrypoint.

    Args:
        config: validated YAML.
        adapter: ModelAdapter implementation for the model family.
        accelerator: pre-built `Accelerator` (the CLI constructs it; tests
            pass a fake).
        artifacts_dir: where partial + final dirs are written.
        batches: pre-tokenized calibration micro-batches, each a `[B, T]`
            `torch.long` tensor. Decoupled from `moe_compress.utils.calibration`
            so kdr is testable without that sibling project; the CLI builds
            this list via the production calibration path.
        resume_step: 0-based optimizer step to resume from. The caller
            (CLI) discovered this via `find_latest_partial`.
        source_metadata_path: optional path to the input student's
            `compressed_metadata.json`, preserved verbatim into every saved
            dir per HLR-0005.

    Returns the final-checkpoint directory.
    """
    # ── Stage 1: load teacher + student under the ZeRO-3 init context ─────
    with activate_zero3_init(accelerator):
        teacher, student, tokenizer = adapter.load_teacher_and_student(
            accelerator,
            teacher_cfg=config.teacher,
            student_cfg=config.student,
        )

        # REQ: LLR-0007
        active_backends: list[QuantBackend] = []
        # Fetched once and cached so the same list flows into both
        # `partition_and_dispatch` (where it shapes ModelOpt's `ignore` config)
        # and Stage 6's `save_kdr_artifact` (where it shapes config.json's
        # `ignore` list per LLR-0020 AC #3). Re-querying the adapter at
        # Stage 6 would inspect the post-`accelerator.prepare` (DDP-wrapped)
        # student, whose dotted module paths carry a `module.` prefix and
        # would diverge from the patterns ModelOpt was configured with at
        # `apply_quant` time.
        cached_fp32_carve_outs: list[str] = []
        if config.mode == "da_qad":
            if config.quant is None:
                raise ValueError(
                    "Config.quant is required in da_qad mode (HLR-0003)."
                )
            # Phase 4 implementation: real dispatch with calibration batches
            # and adapter-supplied carve-outs / attention paths.
            from ..quant.factory import partition_and_dispatch

            cached_fp32_carve_outs = adapter.fp32_carve_outs(student)
            active_backends = partition_and_dispatch(
                student,
                config.quant,
                calibration_batches=batches,
                ptq_subset_size=config.calibration.ptq_subset_size,
                fp32_carve_outs=cached_fp32_carve_outs,
                attention_module_paths=adapter.attention_module_paths(student),
                kv_quant_exempt_indices=adapter.kv_quant_exempt_indices(student),
            )

    # ── Stage 2: trainable scope + optimizer + accelerator.prepare ────────
    _enable_trainable_scope(student, scope=config.distillation.trainable_scope)
    optim = build_optimizer(student, config.distillation)
    student, optim = accelerator.prepare(student, optim)

    # ── Stage 3: place the teacher correctly for the chosen distributed type
    if not is_deepspeed(accelerator):
        # Non-DS path (single-GPU / DDP): replicate on each rank's device.
        # Under DS3 the teacher was already sharded by HfDeepSpeedConfig.
        teacher = teacher.to(accelerator.device)

    # ── Stage 4: gradient checkpointing toggled BEFORE the loop runs ──────
    if config.distillation.use_gradient_checkpointing:
        # `gradient_checkpointing_enable` exists on every HF PreTrainedModel
        # but is not in `nn.Module`'s typed surface — `getattr` keeps mypy
        # happy without a stub-wide type ignore.
        gc_enable = getattr(student, "gradient_checkpointing_enable", None)
        if gc_enable is not None:
            gc_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── Stage 5: run the FKLD loop ────────────────────────────────────────
    # Router-replay hook (LLR-0025) — pinned ON for da_qad mode so the
    # student's MoE expert assignments match the teacher's despite precision
    # drift. For bf16 we use a no-op hook (zero overhead) so the loop's
    # microbatch step stays polymorphic — preserves LLR-0007 AC #1's
    # single-mode-branch invariant by gating on `active_backends` rather
    # than re-checking config.mode.
    replay_hook: RouterReplayHookProtocol
    if active_backends:
        # Walk the unwrapped student so the hook addresses raw module objects
        # (DDP / DeepSpeed wrappers don't change the inner module identity,
        # but `accelerator.unwrap_model` keeps the named_modules paths
        # consistent with `attention_module_paths`).
        replay_hook = adapter.router_replay_hook(
            teacher, accelerator.unwrap_model(student)
        )
    else:
        replay_hook = NoOpReplayContextManager()

    with replay_hook:
        _LoopState(
            config=config,
            accelerator=accelerator,
            artifacts_dir=artifacts_dir,
            teacher=teacher,
            student=student,
            tokenizer=tokenizer,
            optim=optim,
            batches=batches,
            resume_step=resume_step,
            source_metadata_path=source_metadata_path,
            replay_hook=replay_hook,
        ).run()

    # ── Stage 6: final save ───────────────────────────────────────────────
    # Branching on `active_backends` (rather than re-checking the mode flag)
    # preserves LLR-0007 AC #1's single-mode-branch invariant: the only
    # mode-equality site is in Stage 1 above, where it populates
    # `active_backends`. A non-empty list ⇒ da_qad path took effect, an
    # empty list ⇒ bf16.
    final_step = max(0, _read_step_from_metadata(artifacts_dir, config.mode))
    if active_backends:
        # da_qad: compressed-tensors save via the active backend's converter
        # (LLR-0018 / LLR-0021). Final dir is sibling to partials.
        assert config.quant is not None  # Guarded in Stage 1.
        final_dir = artifacts_dir / f"kdr_{config.mode}_recovered"
        if accelerator.is_main_process:
            save_kdr_artifact(
                accelerator.unwrap_model(student),
                final_dir,
                backends=active_backends,
                quant_block=config.quant,
                fp32_carve_outs=cached_fp32_carve_outs,
                tokenizer=tokenizer,
                source_metadata_path=source_metadata_path,
            )
        accelerator.wait_for_everyone()
        return final_dir
    # bf16: vanilla save_partial(..., partial=False).
    return save_partial(
        student,
        tokenizer,
        accelerator,
        artifacts_dir=artifacts_dir,
        mode=config.mode,
        step=final_step,
        source_metadata_path=source_metadata_path,
        partial=False,
    )


# ---------------------------------------------------------------------------
# Trainable-scope toggling
# ---------------------------------------------------------------------------


def _enable_trainable_scope(student: nn.Module, *, scope: str) -> int:
    """v0 supports only `scope="full"`. The `experts_only` and
    `factored_only` scopes from structural_recovery require
    `moe_compress.utils.iter_moe_layers` which is not a kdr dependency;
    Phase 5+ may reintroduce those if needed."""
    if scope != "full":
        raise NotImplementedError(
            f"trainable_scope={scope!r} is not implemented in kdr v0. "
            "Only 'full' is supported (the experts_only/factored_only "
            "scopes from structural_recovery require moe_compress.utils "
            "which kdr does not depend on)."
        )
    n_trainable = 0
    for p in student.parameters():
        p.requires_grad_(True)
        n_trainable += p.numel()
    log.info("trainable_scope=%s -> %.3fB params trainable", scope, n_trainable / 1e9)
    return n_trainable


# ---------------------------------------------------------------------------
# Loop state — kept in a dataclass-shaped class for clarity
# ---------------------------------------------------------------------------


class _LoopState:
    """Owns the per-run mutable state (counters, NaN flags, partial micros).

    Encapsulating this state in an instance keeps the public `run_recovery`
    signature small and lets tests inject fakes for individual sub-steps.
    """

    def __init__(
        self,
        *,
        config: Config,
        accelerator: Accelerator,
        artifacts_dir: Path,
        teacher: nn.Module,
        student: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        optim: torch.optim.Optimizer,
        batches: Iterable[torch.Tensor],
        resume_step: int,
        source_metadata_path: Path | None,
        replay_hook: RouterReplayHookProtocol,
    ) -> None:
        self.config = config
        self.dconf = config.distillation
        self.accelerator = accelerator
        self.artifacts_dir = artifacts_dir
        # `replay_hook` is the LLR-0025 router-replay context manager
        # (already entered by the caller). NoOp instance for bf16 mode.
        self._replay_hook: RouterReplayHookProtocol = replay_hook
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.optim = optim
        self.batches = list(batches)
        self.resume_step = resume_step
        self.source_metadata_path = source_metadata_path

        # Schedule
        self.world = max(1, self.accelerator.num_processes)
        self.grad_accum = self.dconf.gradient_accumulation
        self.micro_bsz = self.dconf.per_device_batch_size
        self.seq_len = self.dconf.sequence_length
        self.tokens_per_step = (
            self.micro_bsz * self.world * self.grad_accum * self.seq_len
        )
        self.total_steps = max(
            1, self.dconf.total_tokens // self.tokens_per_step
        )
        self.warmup = self.dconf.warmup_steps
        self.lr_max = self.dconf.learning_rate
        self.lr_min = self.dconf.min_learning_rate
        if self.warmup >= self.total_steps:
            raise ValueError(
                f"warmup_steps={self.warmup} must be < "
                f"total_steps={self.total_steps}. Increase total_tokens or "
                "reduce warmup_steps."
            )

        # Token counters (4-way split per structural_recovery's invariant:
        # consumed = with_grad + skipped_nan + dropped_window).
        self.tokens_consumed = 0
        self.tokens_with_grad = 0
        self.tokens_skipped_nan = 0
        self.tokens_dropped_window = 0
        self.pending_window_tokens = 0

        # NaN escalation state
        self.nan_threshold = 5
        self.consecutive_nan_windows = 0
        self.nan_in_current_window = False

        # Per-step state
        self.step = resume_step
        self.micro_in_window = 0
        self.last_real_loss: float | None = None

    # REQ: LLR-0043
    def run(self) -> None:
        """Iterate calibration batches and step the optimizer.

        Per-rank shard, then trim to a multiple of (world * grad_accum) so
        ranks finish in lock-step (otherwise NCCL hangs on the last
        unbalanced micro-batch).
        """
        all_batches = self.batches
        truncate_to = (
            len(all_batches) // (self.world * self.grad_accum)
        ) * (self.world * self.grad_accum)
        if truncate_to < len(all_batches):
            log.info(
                "Truncating calibration: %d -> %d batches (multiple of %d*%d)",
                len(all_batches),
                truncate_to,
                self.world,
                self.grad_accum,
            )
            all_batches = all_batches[:truncate_to]

        local_batches = self._shard_per_rank(all_batches)
        log.info(
            "rank %d/%d sees %d local batches (of %d global)",
            self.accelerator.process_index,
            self.world,
            len(local_batches),
            len(all_batches),
        )

        # Resume: skip the micros consumed by the prior run. Calibration is
        # deterministic from the cached tensor, so slicing == replay+discard.
        if self.resume_step > 0:
            skip_micros = self.resume_step * self.grad_accum
            if skip_micros < len(local_batches):
                local_batches = local_batches[skip_micros:]
                log.info(
                    "Resuming from step=%d: skipped %d local micro-batches.",
                    self.resume_step,
                    skip_micros,
                )
            else:
                log.warning(
                    "resume_step=%d would skip all %d local batches; "
                    "starting from scratch.",
                    self.resume_step,
                    len(local_batches),
                )
                self.resume_step = 0
                self.step = 0

        self.optim.zero_grad(set_to_none=True)
        self.student.train()

        remaining_micros = max(0, (self.total_steps - self.step) * self.grad_accum)
        for consumed_micros, batch in enumerate(local_batches):
            if consumed_micros >= remaining_micros:
                break
            self._step_one_micro(batch)
            if self.step >= self.total_steps:
                break

        # On exit, any uncommitted micros are dropped.
        self.tokens_dropped_window += self.pending_window_tokens
        self.pending_window_tokens = 0

    def _shard_per_rank(self, batches: list[torch.Tensor]) -> list[torch.Tensor]:
        """Strided slice so each rank consumes a disjoint subset.

        Under DS3 the params are model-sharded; the data must still be
        data-parallel for any compute speedup. Without this slice every rank
        would compute the same loss on the same tokens — correctness-OK but
        ~world-x wasteful.
        """
        if self.world <= 1:
            return list(batches)
        pi = self.accelerator.process_index
        return list(batches[pi :: self.world])

    def _step_one_micro(self, batch: torch.Tensor) -> None:
        """Forward teacher+student, compute loss, accumulate / step."""
        ids = batch.to(self.accelerator.device, non_blocking=True)
        # REQ: LLR-0025
        # Reset router-replay state at the top of every microbatch so the
        # teacher's freshly-captured assignments drive THIS microbatch's
        # student forward (and not a stale capture from the previous one).
        self._replay_hook.start_microbatch()
        with torch.no_grad():
            t_logits = self.teacher(input_ids=ids).logits
        s_logits = self.student(input_ids=ids).logits
        loss = forward_kld_loss(s_logits, t_logits, temperature=self.dconf.temperature)

        this_micro_tokens = int(ids.numel()) * self.world
        self.tokens_consumed += this_micro_tokens
        self.pending_window_tokens += this_micro_tokens

        finite = self._all_finite(loss)
        if not finite:
            log.warning(
                "step=%d micro=%d non-finite loss; substituting zero.",
                self.step,
                self.micro_in_window,
            )
            self.tokens_skipped_nan += this_micro_tokens
            self.pending_window_tokens -= this_micro_tokens
            self.nan_in_current_window = True
            if is_deepspeed(self.accelerator):
                # Build a graph-connected zero loss so DS's micro-batch
                # counter advances and backward touches engine bookkeeping.
                cleaned = torch.nan_to_num(s_logits, nan=0.0, posinf=0.0, neginf=0.0)
                loss = (cleaned * 0.0).sum()
            else:
                self.optim.zero_grad(set_to_none=True)
                self.tokens_dropped_window += self.pending_window_tokens
                self.pending_window_tokens = 0
                self.micro_in_window = 0
                self.nan_in_current_window = False
                return

        self.accelerator.backward(loss / self.grad_accum)
        if finite:
            self.last_real_loss = float(loss.detach().item())
        del t_logits, s_logits
        self.micro_in_window += 1

        if self.micro_in_window % self.grad_accum == 0:
            self._commit_window()

    def _commit_window(self) -> None:
        """Run optimizer step, advance counters, fire eval/save callbacks."""
        if not is_deepspeed(self.accelerator) and self.dconf.grad_clip_norm > 0:
            self.accelerator.clip_grad_norm_(
                [p for p in self.student.parameters() if p.requires_grad],
                self.dconf.grad_clip_norm,
            )
        set_lr(
            self.optim,
            cosine_with_warmup(
                self.step,
                warmup_steps=self.warmup,
                total_steps=self.total_steps,
                lr_max=self.lr_max,
                lr_min=self.lr_min,
            ),
        )
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)
        self.tokens_with_grad += self.pending_window_tokens
        self.pending_window_tokens = 0
        self.step += 1
        self.micro_in_window = 0

        if self.nan_in_current_window:
            self.consecutive_nan_windows += 1
            if self.consecutive_nan_windows >= self.nan_threshold:
                raise RuntimeError(
                    f"NaN circuit breaker tripped: {self.consecutive_nan_windows} "
                    f"consecutive windows with non-finite loss "
                    f"(threshold={self.nan_threshold}). Likely root causes: "
                    "corrupted student weights, incompatible teacher/student "
                    "tokenisation, or sharded-init mismatch."
                )
        else:
            self.consecutive_nan_windows = 0
        self.nan_in_current_window = False

        if (
            self.accelerator.is_main_process
            and self.step % self.dconf.log_every_n_steps == 0
        ):
            log.info(
                "step=%d/%d lr=%.3e loss=%.6f tok=%.2fB/%.2fB nan_skip=%.2fM",
                self.step,
                self.total_steps,
                self.optim.param_groups[0]["lr"],
                self.last_real_loss
                if self.last_real_loss is not None
                else float("nan"),
                self.tokens_with_grad / 1e9,
                self.dconf.total_tokens / 1e9,
                self.tokens_skipped_nan / 1e6,
            )

        # REQ: LLR-0049
        # Eval cadence: step > 0 AND step % eval_every_n_steps == 0.
        # Step 0 does NOT trigger eval (a baseline eval is a separate concern).
        if (
            self.step > 0
            and self.step % self.dconf.eval_every_n_steps == 0
        ):
            eval_run(self.student, self.tokenizer, self.config.eval, self.accelerator)

        # REQ: LLR-0027
        # Save cadence: 0 means no partial saves (final-only).
        if (
            self.dconf.save_every_n_steps > 0
            and self.step % self.dconf.save_every_n_steps == 0
        ):
            save_partial(
                self.student,
                self.tokenizer,
                self.accelerator,
                artifacts_dir=self.artifacts_dir,
                mode=self.config.mode,
                step=self.step,
                source_metadata_path=self.source_metadata_path,
                extra_metadata=self._snapshot_run_metadata(),
                partial=True,
            )

    def _all_finite(self, loss: torch.Tensor) -> bool:
        """Collective NaN/Inf check — True iff loss is finite on EVERY rank.

        Critical under DeepSpeed: a rank-local skip would mismatch the next
        backward's reductions and hang NCCL. We collectively detect →
        collectively skip.
        """
        is_finite_local = 1.0 if torch.isfinite(loss).all() else 0.0
        flag = torch.tensor(is_finite_local, device=self.accelerator.device)
        if self.accelerator.num_processes > 1:
            all_flags = self.accelerator.gather(flag.unsqueeze(0))
            if (
                self.accelerator.is_main_process
                and (all_flags < 0.5).any()
            ):
                bad = (all_flags < 0.5).nonzero(as_tuple=True)[0].tolist()
                log.warning(
                    "non-finite loss on ranks %s — investigate hardware if "
                    "recurrent on the same rank.",
                    bad,
                )
            flag = all_flags.min()
        return bool(flag.item() >= 0.5)

    def _snapshot_run_metadata(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "mode": self.config.mode,
            "tokens_consumed": self.tokens_consumed,
            "tokens_with_grad": self.tokens_with_grad,
            "tokens_skipped_nan": self.tokens_skipped_nan,
            "tokens_dropped_window": self.tokens_dropped_window,
            "total_steps_planned": self.total_steps,
        }


def _read_step_from_metadata(artifacts_dir: Path, mode: str) -> int:
    """Recover the latest committed step from any partial dir's
    `kdr_run_metadata.json` so the final save name carries an honest
    step number even when partial saves were enabled.

    Mirrors `find_latest_partial`'s sentinel rule (LLR-0029): only dirs
    whose `_SAVE_COMPLETE` is present are considered. A crash between
    the atomic rename and the sentinel `touch` would otherwise leave a
    `kdr_run_metadata.json` visible in a structurally-incomplete dir,
    making this scan disagree with `find_latest_partial`.
    """
    pattern = f"kdr_{mode}_partial_step*"
    best = -1
    for p in artifacts_dir.glob(pattern):
        if not p.is_dir() or not (p / SAVE_COMPLETE_SENTINEL).exists():
            continue
        meta = p / "kdr_run_metadata.json"
        if not meta.exists():
            continue
        try:
            payload = json.loads(meta.read_text())
            best = max(best, int(payload.get("step", -1)))
        except (OSError, json.JSONDecodeError):
            continue
    return best
