# kdr Docker / vast.ai bootstrap

**Phase 2 placeholder.** This directory will hold the vast.ai operator runbook
and `bootstrap.sh` script in Phase 6.

The image itself is shared with `max_quality` — kdr does NOT ship its own
Dockerfile. We pull `ghcr.io/lucaspirola/moe-compress:latest` and run
`bootstrap.sh` inside the container, which:

1. Validates env vars (`HF_TOKEN`, `STUDENT_REPO`, `CACHE_MOUNT`).
2. Clones `moe_compress`.
3. Installs Zyphra's transformers fork over the base image's stock transformers
   (required for ZAYA1's `ZayaForCausalLM`).
4. Authenticates with HF.
5. Snapshot-downloads teacher + student into `/cache`.
6. Computes `run_id` from `(config_hash, student_repo_sha, mode)` and queries
   the HF Hub partials repo for resume state.
7. Invokes `python -m kdr.cli.train ...`.
8. Synchronously uploads each partial save to HF Hub.
9. Uploads the final compressed-tensors artifact.

See requirements `HLR-0008`, `LLR-0031`..`LLR-0035` for the contract.
