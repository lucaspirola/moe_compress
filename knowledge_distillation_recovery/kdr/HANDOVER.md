# kdr Phase 7.1 — Handover

Current state of the Phase 7.1 ZAYA1-8B BF16 KD smoke after a long debug
loop on 2026-05-11. Written for an agent picking up the work with only a
fresh git clone of `lucaspirola/moe_compress` and no on-disk state.

## What Phase 7.1 is

A pure plumbing validation of the kdr training loop against `Zyphra/ZAYA1-8B`:
load teacher + student, run 200 optimizer steps of BF16 forward-KLD
self-distillation, save partials at 50/100/150/200, upload to HF Hub, and
load the final artifact back in a fresh Python process for a forward-pass
sanity. Acceptance is "no NaN/OOM/NCCL crash + load-back returns finite
logits", not a numerical-correctness check. Self-distillation, so the loss
is ≈0 throughout — see `feedback_kdr_bf16_self_distillation` memory.

The config lives at `knowledge_distillation_recovery/kdr/configs/zaya1_8b_bf16_smoke200.yaml`.
The orchestrator is `knowledge_distillation_recovery/kdr/docker/bootstrap.sh`.

## What's been built so far

The kdr loop runs end-to-end against the local model structure (verified via
`inspect_shapes.py`). The path to actual GPU training was blocked by a series
of incompatibilities between Zyphra's published `transformers@zaya1` fork and
the released `ZAYA1-8B` checkpoint. We forked Zyphra's repo to
`lucaspirola/transformers@zaya1-patches` and have applied three patches there:

1. **commit `7be4d6c`** — relax `huggingface-hub<1.0` upper bound in
   `dependency_versions_table.py`. The base docker image ships
   `transformers>=5.8` which pulls hf-hub 1.x; the fork's pin made import-time
   `dependency_versions_check` fail.
2. **commit `b479098`** — guard `past_key_values` against `bool` in
   `ZayaConv.forward` (lines 358/408 of modeling_zaya.py). Training paths
   pass a bool through where the fork expects `ZayaDynamicCache | None`.
3. **commit `219cd23`** — fix positional-arg order in
   `ZayaModel.forward`'s `_gradient_checkpointing_func` call. The original
   had `cca_mask` and `position_ids` swapped (and a cascade of misalignment
   downstream); `position_embeddings` ended up receiving
   `prev_router_hidden_states` (None on first layer), tripping
   `cos, sin = position_embeddings` with `TypeError: cannot unpack non-iterable NoneType object`.

The kdr trainer also got a `use_cache=False` fix
(`knowledge_distillation_recovery/kdr/src/kdr/training/loop.py` —
commit `bd5299c` on this repo) so the model never tries to thread a cache
through training paths in the first place.

The kdr docker bootstrap has a background watchdog
(commit `24acd74`) that prints a `[watchdog HH:MM:SS] RAM=... VRAM=... disk=... python_rss=...`
line every 10 seconds while the trainer runs. The watchdog is what unblocked
the recent debugging: it kept the host's stdout active and ruled out OOM as
the cause of the early "host vanishes" failure mode (cgroup limit was 242 GB;
we were nowhere near it).

## How to resume the smoke

The last known failure was the gradient-checkpointing positional-arg mismatch
fixed in `219cd23`. That patch is pushed but **was not yet validated on
GPU** — the instance running the prior attempt (36555362, Oklahoma, $0.74/hr)
was destroyed before re-running the bootstrap with the new fork. The next
launch will be the first one carrying all three fork patches plus the
trainer `use_cache=False` fix.

Local credentials and tools on the user's machine (the user is `lucas` on a
WSL2 box; their auto-memory has the exact paths):

- vast.ai CLI: `/home/lucas/.venv-vastai/bin/vastai`, API key persisted at
  `~/.config/vastai/vast_api_key`. Just call the CLI; auth is automatic.
- HF token (write scope): `~/.cache/huggingface/token`. Inject at launch
  via `--env "-e HF_TOKEN=$(cat ~/.cache/huggingface/token) ..."`.
- Local fork checkout (for iteration): `/home/lucas/ai/transformers-zaya1/`
  (clone of Zyphra/transformers @ zaya1 with origin=Zyphra, fork=lucaspirola).
  Already editable-installed into kdr's venv. If you're on a fresh machine,
  you'll need to clone it and `uv pip install -e .` into the kdr venv.
- Local model snapshot (for `inspect_shapes.py`):
  `/mnt/d/models/Zyphra/ZAYA1-8B/` — config + safetensors only, no tokenizer.
  Used only for structure verification, not training.

## Standing pre-authorizations (read carefully)

The user has issued several **standing** authorizations that remain active
across messages until they explicitly revoke. Apply them automatically; do
not re-ask:

- **vast.ai auto-launch** when an offer meets ALL of: A100-80GB SXM4 or PCIE,
  single-GPU, verified, rentable, gpu_ram≥80, disk_space≥100, direct_port_count≥1,
  dph_total < $1.50, reliability > 0.98, inet_down ≥ 500 Mbps. Pick the
  cheapest match. Fire **immediately** when the conditions become true —
  not contingent on a scheduled poll. (See `feedback_standing_auth_means_act`.)
- **destroy + diagnose + fix + relaunch** is pre-authorized for kdr smoke
  iterations. Per the user's latest correction, prefer to **KEEP the
  instance alive** when a fixable Python error surfaces — patch + push the
  fix, then re-run the bootstrap on the SAME live host via SSH. Destroying
  burns the lease and the next bidder may take the offer.
- **$2/hr hard ceiling** on offer selection at all times.
- **Never leave an idle billed instance running.** If you can't fix it in a
  reasonable window, destroy.

## Known-bad hosts (DO NOT rent)

- `36311484` (Georgia US, $1.375/hr A100) — broken outbound to GitHub;
  `git clone` returns HTTP 500 from inside the container.
- `21050975` (Sweden, $0.87/hr A100 SXM4) — host vanished mid-run with no
  diagnosable failure. (Now suspected to be the same trainer crash hitting
  via OOM-of-ssh-server or similar, but unverified — leave blacklisted.)
- `29019375` (Czechia, $1.07/hr A100 SXM4) — same as Sweden.

## Other agents on the same vast.ai account

A different agent runs `max_quality` ablations on H200 instances (typically
$3.40–$3.80/hr, contracts in the `3654xxxx` range, image is the same
`ghcr.io/lucaspirola/moe-compress:latest`). **Do NOT destroy max_quality
instances.** Distinguish by env vars: max_quality launches set
`ONLY`/`PREFLIGHT_ONLY`/`UPLOAD_ON_SUCCESS`; kdr launches set `STUDENT_REPO`/
`KDR_CONFIG`/`KDR_MODE`.

## The structural debugger (`inspect_shapes.py`)

`tools/inspect_shapes.py` builds the model on meta-device using the patched
local fork and diffs the constructed parameter names + shapes against the
checkpoint's safetensors headers. No weights downloaded, runs in seconds,
uses <100 MB RAM. Use this to catch structural fork-vs-checkpoint mismatches
locally before paying for a GPU launch.

```bash
# Against local snapshot:
/home/lucas/ai/moe_compress/knowledge_distillation_recovery/kdr/.venv-kdr/bin/python \
  knowledge_distillation_recovery/kdr/tools/inspect_shapes.py /mnt/d/models/Zyphra/ZAYA1-8B

# Against HF Hub (downloads headers only, no weights):
/home/lucas/ai/moe_compress/knowledge_distillation_recovery/kdr/.venv-kdr/bin/python \
  knowledge_distillation_recovery/kdr/tools/inspect_shapes.py
# (defaults to Zyphra/ZAYA1-reasoning-base — change DEFAULT_REPO in source
#  to switch to ZAYA1-8B for the smoke target)
```

For the script to work, the kdr venv needs the local fork editable-installed
(see "Local credentials and tools" above). The patches in
`lucaspirola/transformers@zaya1-patches` are the canonical patched fork;
if you set up a fresh local clone, check out that branch.

## Vast.ai launch command (for reference)

```bash
HF_TOKEN=$(cat ~/.cache/huggingface/token)
/home/lucas/.venv-vastai/bin/vastai create instance <OFFER_ID> \
  --image ghcr.io/lucaspirola/moe-compress:latest \
  --disk 200 \
  --ssh --direct \
  --env "-e HF_TOKEN=$HF_TOKEN -e STUDENT_REPO=Zyphra/ZAYA1-8B \
         -e CACHE_MOUNT=/workspace \
         -e KDR_CONFIG=knowledge_distillation_recovery/kdr/configs/zaya1_8b_bf16_smoke200.yaml \
         -e KDR_MODE=bf16" \
  --onstart-cmd "curl -sSL https://raw.githubusercontent.com/lucaspirola/moe_compress/main/knowledge_distillation_recovery/kdr/docker/bootstrap.sh | bash"
```

The bootstrap.sh is curl-piped from raw.githubusercontent.com on every
launch, so changes to it (or to the patched fork URL it pip-installs) take
effect immediately — no docker image rebuild needed for code or fork changes.
The docker image is only rebuilt on `max_quality/requirements.txt` or
`max_quality/docker/**` edits, via `.github/workflows/docker-build.yml`.

## Monitoring a live run

Primary channel: `vastai logs <id>`. Grep for `[watchdog ` to see the
10-second resource trace, and for `>>>` to see bootstrap milestone markers.

If `vastai logs` becomes spammy (it can fill with SSH port-forward retries on
some hosts), SSH in:

```bash
SSHURL=$(/home/lucas/.venv-vastai/bin/vastai ssh-url <id>)
HOST=$(echo "$SSHURL" | sed -E 's|ssh://root@([^:]+):.*|\1|')
PORT=$(echo "$SSHURL" | sed -E 's|.*:([0-9]+)$|\1|')
ssh -o StrictHostKeyChecking=accept-new -p $PORT root@$HOST \
    'tail -80 /var/log/onstart.log; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
```

If the bootstrap is re-run via SSH (after a code fix), the new log lives at
`/var/log/onstart-rerun.log`, NOT `onstart.log`.

## To rerun the bootstrap on a live instance after a fix

```bash
SSHURL=$(/home/lucas/.venv-vastai/bin/vastai ssh-url <id>)
HOST=$(echo "$SSHURL" | sed -E 's|ssh://root@([^:]+):.*|\1|')
PORT=$(echo "$SSHURL" | sed -E 's|.*:([0-9]+)$|\1|')
ssh -o StrictHostKeyChecking=no -p $PORT root@$HOST 'bash -c "
  set -a
  source <(cat /proc/1/environ | tr \"\\0\" \"\\n\" | grep -E \"^(HF_TOKEN|STUDENT_REPO|CACHE_MOUNT|KDR_CONFIG|KDR_MODE)=\")
  set +a
  nohup bash -c \"curl -sSL https://raw.githubusercontent.com/lucaspirola/moe_compress/main/knowledge_distillation_recovery/kdr/docker/bootstrap.sh | bash\" > /var/log/onstart-rerun.log 2>&1 &
  echo started PID \$!
"'
```

## Phase 7.1 expected marker sequence

A successful run produces these in `/var/log/onstart.log`:

```
>>> Cloning ... moe_compress
>>> Installing patched Zyphra transformers fork (lucaspirola/transformers@zaya1-patches)
>>> Snapshot-downloading student Zyphra/ZAYA1-8B
>>> Resolving student HF Hub SHA
>>> Deriving run_id from (config, student_sha=1396e81e, mode=bf16)
>>> run_id=88f04a9c064109de
>>> Querying pirola/kdr-partials-88f04a9c064109de for resume seed
>>> No prior partials; starting from scratch
>>> Resource snapshot pre-trainer:        ← bootstrap's debug print
    cgroup memory.max: <bytes>
    host RAM (MB): used=… total=…
    GPU VRAM (MB): used=… total=…
    /workspace disk: used=… total=…
>>> Invoking kdr.cli.train (mode=bf16)
INFO __main__ :: World size: 1 device: cuda
INFO moe_compress.utils.calibration :: Building calibration tensor: 13000 x 4096
INFO moe_compress.utils.calibration :: Streaming math samples …  (then science, chat, instruction_following, conversational_agent, swe, terminal_agent)
INFO moe_compress.utils.calibration :: Cached calibration tensor: artifacts/_calibration_cache/calib_…
INFO kdr.adapters.zaya1_8b :: Loading TEACHER Zyphra/ZAYA1-8B
INFO kdr.adapters.zaya1_8b :: Loading STUDENT /workspace/student
INFO kdr.training.loop :: trainable_scope=full -> 8.840B params trainable
INFO kdr.training.loop :: Truncating calibration: 13000 -> 12992 batches
INFO kdr.training.loop :: rank 0/1 sees 12992 local batches
[here be training step output — the new frontier the prior runs never reached]
… save partials @ steps 50, 100, 150, 200 …
>>> Uploading final artifact to pirola/kdr-recovered-88f04a9c064109de
>>> Load-back round-trip: pulling pirola/kdr-recovered-88f04a9c064109de in a fresh Python process
load-back OK: logits shape=(1, …, 262272), dtype=torch.bfloat16, finite ✓
kdr bootstrap complete.
```

`run_id` is deterministic from the (canonical-config-dump, student_sha, mode)
triple — `88f04a9c064109de` for the current smoke config. Re-runs hit the same
HF Hub repos and can resume from partials.

## What to do next

1. **Resume the polling loop** for an under-$1.50/hr A100-80GB offer (skip the
   3 blacklisted hosts). Fire on first match per the standing pre-auth.
2. Once the bootstrap reaches the training step output (the prior frontier),
   monitor via the watchdog. If it crashes at a new fork bug, **keep the
   instance alive**, patch the fork, push to `zaya1-patches`, re-run bootstrap
   on the live host via SSH.
3. On success, capture the partials count + final HF Hub URL + total cost,
   destroy the instance, and report.
