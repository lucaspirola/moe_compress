# Stage-5 Router-KD DDP — TDD execution

- [ ] T0  ddp_config.py — DdpConfig.from_config + tests
- [ ] T1  orchestrator dispatch fork (_run_single_process / _spawn_ddp_workers)
- [ ] T2  _unwrap.py unwrap_student + migrate student sites
- [ ] T3  ddp_runtime.py bootstrap/teardown/spawn
- [ ] T4  per-step row split + teacher (live no-double-slice / cache rank offset)
- [ ] T5  DDP wrap + no_sync + finiteness all-reduce
- [ ] T6  sync early-stop/EMA + rank-0 I/O + best.pt broadcast + resume
- [ ] T7  teacher VRAM precondition validation
- [ ] T8  _run_ddp_worker student materialization
- [ ] T9  DEFAULT-PATH GOLDEN GUARDRAIL (no change)
- [ ] T10 RESULT-PRESERVATION GATE (gloo 1 vs 2)
- [ ] T11 deadlock/failure modes
- [ ] T12 merge-repair+DDP guard (not-yet-supported error)
</content>
