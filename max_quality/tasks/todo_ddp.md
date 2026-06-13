# Stage-5 Router-KD DDP — TDD execution (COMPLETE)

- [x] T0  ddp_config.py — DdpConfig.from_config + tests
- [x] T1  orchestrator dispatch fork (_run_single_process / _spawn_ddp_workers)
- [x] T2  _unwrap.py unwrap_student + migrate student sites
- [x] T3  ddp_runtime.py bootstrap/teardown/spawn
- [x] T4  per-step row split + teacher (live no-double-slice / cache rank offset)
- [x] T5  DDP wrap + no_sync + finiteness all-reduce (M1)
- [x] T6  sync early-stop/EMA + rank-0 I/O + best.pt broadcast + resume (M4)
- [x] T7  teacher VRAM precondition validation
- [x] T8  _run_ddp_worker student materialization (H1)
- [x] T9  DEFAULT-PATH GOLDEN GUARDRAIL — green, no MOE_REGEN_GOLDEN
- [x] T10 RESULT-PRESERVATION GATE — gloo 1 vs 2 within rtol=1e-5/atol=1e-7
        (achieved: loss/raw_kl 2.98e-08, router weight 3.73e-09)
- [x] T11 deadlock/failure modes (M1 no-hang, early-stop no-desync, watchdog)
- [x] T12 merge-repair+DDP guard (not-yet-supported error)

DEFERRED (no hardware): live multi-GPU NCCL ≥2-GPU validation. All tests gloo/CPU.
</content>
