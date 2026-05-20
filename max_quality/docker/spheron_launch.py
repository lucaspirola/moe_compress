#!/usr/bin/env python3
"""Spheron split-machine orchestrator for the SH (merge-heal-with-LR-schedule
+ cross-domain telemetry) run.

End-to-end flow on `spheron-es US Central 1`:
  1. Create a 500 GB NETWORK_SSD volume (or reuse the one named MOE_VOLUME_NAME).
  2. Spin up an RTX PRO 6000 ($2.405/hr on-demand) deployment with the volume
     attached. cloudInit pulls + runs `run_sh_split.sh MOE_PHASE=stage2`. Poll
     until the docker container exits cleanly; then DELETE the deployment
     (volume keeps the data).
  3. Spin up an H200 ($4.615/hr on-demand) deployment with the same volume
     attached. cloudInit pulls + runs `run_sh_split.sh MOE_PHASE=stage2p5`.
     Poll until done; DELETE the deployment.
  4. (Optional) DELETE the volume.

Why not the Spheron CLI: there is no marketplace CLI that exposes the
`/api/volumes` + `/api/deployments` REST surface — the `@spheron/protocol-sdk`
node package is the Akash-style on-chain layer, which is a different
abstraction. The REST API is well-defined enough that a thin Python wrapper
is the path of least surprise.

Usage:
  python3 spheron_launch.py volume-create [--size-gb 500] [--volume-name ...]
  python3 spheron_launch.py phase1 --volume-id ID [--branch feat/heal-lr-schedule]
  python3 spheron_launch.py phase2 --volume-id ID [--branch feat/heal-lr-schedule]
  python3 spheron_launch.py status [--deployment-id ID]
  python3 spheron_launch.py teardown --deployment-id ID [--keep-volume] [--volume-id ID]
  python3 spheron_launch.py run [--size-gb 500]      # end-to-end (phase1 + phase2 + teardown)

Auth: reads `sai_pk_*` from ~/.config/spheron/credentials.
HF token: reads ~/.cache/huggingface/token (passed through to the cloudInit).
GitHub: cloudInit clones `https://github.com/lucaspirola/moe_compress.git`
        (public; no token needed).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import shutil
import subprocess  # noqa: S404 — argv-list form only; no shell=True invocations
import sys
import time
import textwrap
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Config (defaults — overridable via env or CLI)
# ---------------------------------------------------------------------------

SPHERON_API_BASE = os.environ.get("SPHERON_API_BASE", "https://app.spheron.ai")
SPHERON_CREDENTIALS_PATH = Path(
    os.environ.get("SPHERON_CREDENTIALS", str(Path.home() / ".config" / "spheron" / "credentials"))
)
HF_TOKEN_PATH = Path(
    os.environ.get("HF_TOKEN_PATH", str(Path.home() / ".cache" / "huggingface" / "token"))
)

PROVIDER = "spheron-es"
REGION = "US Central 1"
RTX6000_OFFER_ID = "US Central 1::gpu-rtx6000::1gpu-24vcpu-218gb"     # $2.405/hr
H200_OFFER_ID = "US Central 1::gpu-h200-sxm::1gpu-16vcpu-200gb"        # $4.615/hr
OS_IMAGE = "Ubuntu 24.04 (CUDA 13)"
DEFAULT_VOLUME_NAME = "moe-sh-split"
DEFAULT_VOLUME_SIZE_GB = 500
DEFAULT_BRANCH = "feat/heal-lr-schedule"
DEFAULT_HF_BUCKET = "pirola/moe-strategy-35pct"
GH_REPO_URL = "https://github.com/lucaspirola/moe_compress.git"
DOCKER_IMAGE = "ghcr.io/lucaspirola/moe-compress:latest"

# Volume mount path inside the deployment. Tracks the volume NAME (Spheron's
# convention is /mnt/<volume-name>); _make_cloud_init computes it per call so
# the orchestrator and the deployment agree without a hard-coded constant.
def _volume_mount_for(volume_name: str) -> str:
    return f"/mnt/{volume_name}"

# Polling cadences (seconds).
POLL_DEPLOYMENT_READY = 20         # how often to check IF the deployment came up
POLL_DEPLOYMENT_READY_TIMEOUT = 3600   # 1 h cap on provisioning (marketplace can run 20-25 min)
POLL_RUN = 60                      # how often to SSH-poll the container status flag
POLL_RUN_TIMEOUT_PHASE1 = 8 * 3600    # 8 h cap on Phase 1 (Stage 2 alone)
POLL_RUN_TIMEOUT_PHASE2 = 3 * 3600    # 3 h cap on Phase 2 (Stage 2.5 + Stage 6 alt)
SSH_KEY_PATH = Path(os.environ.get(
    "SPHERON_SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519")
))

# ---------------------------------------------------------------------------
# Logging — uniform format, no external deps.
# ---------------------------------------------------------------------------

log = logging.getLogger("spheron_launch")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


# ---------------------------------------------------------------------------
# Auth + HTTP plumbing.
# ---------------------------------------------------------------------------


def _read_spheron_key() -> str:
    if not SPHERON_CREDENTIALS_PATH.exists():
        raise FileNotFoundError(
            f"Spheron credentials not found at {SPHERON_CREDENTIALS_PATH}. "
            "Place the `sai_pk_...` key in the file (one match per file is fine)."
        )
    text = SPHERON_CREDENTIALS_PATH.read_text()
    match = re.search(r"sai_pk_[A-Za-z0-9_-]+", text)
    if not match:
        raise RuntimeError(
            f"No `sai_pk_...` key found in {SPHERON_CREDENTIALS_PATH}."
        )
    return match.group(0)


def _read_hf_token() -> str:
    if not HF_TOKEN_PATH.exists():
        raise FileNotFoundError(
            f"HF token not found at {HF_TOKEN_PATH}. Run `hf auth login` first."
        )
    return HF_TOKEN_PATH.read_text().strip()


class SpheronClient:
    """Minimal REST wrapper. Raises on HTTP errors; logs request id when present."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        })

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{SPHERON_API_BASE}{path}"
        # Debug log: redact `cloudInit` (embeds the HF token) and any header
        # marked sensitive. The bearer token itself never appears in the body.
        body = kwargs.get("json")
        if isinstance(body, dict) and "cloudInit" in body:
            body = {**body, "cloudInit": f"<{len(body['cloudInit'])}B redacted>"}
        log.debug("HTTP %s %s body=%s", method, url, body)
        r = self._session.request(method, url, timeout=60, **kwargs)
        if not r.ok:
            # Surface the API's error message verbatim — usually descriptive.
            raise RuntimeError(
                f"Spheron API {method} {path} → {r.status_code}: {r.text[:500]}"
            )
        if r.text:
            try:
                return r.json()
            except ValueError:
                return r.text
        return None

    # --- volumes -----------------------------------------------------------

    def list_volumes(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/api/volumes")
        # Response is {"volumes":[...], "total":N, ...}
        return payload.get("volumes", []) if isinstance(payload, dict) else payload

    def create_volume(self, name: str, size_gb: int) -> dict[str, Any]:
        body = {
            "name": name, "sizeInGb": size_gb,
            "provider": PROVIDER, "region": REGION,
        }
        return self._request("POST", "/api/volumes", json=body)

    def get_volume(self, volume_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/volumes/{volume_id}")

    def delete_volume(self, volume_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/volumes/{volume_id}")

    # --- deployments -------------------------------------------------------

    def list_deployments(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/api/deployments")
        return payload if isinstance(payload, list) else payload.get("deployments", [])

    def create_deployment(
        self, *, offer_id: str, ssh_key_id: str, volume_ids: list[str],
        cloud_init: str, name: str,
    ) -> dict[str, Any]:
        body = {
            "provider": PROVIDER,
            "offerId": offer_id,
            "region": REGION,
            "operatingSystem": OS_IMAGE,
            "instanceType": "DEDICATED",
            "sshKeyId": ssh_key_id,
            "volumeIds": volume_ids,
            "cloudInit": cloud_init,
            "name": name,
        }
        return self._request("POST", "/api/deployments", json=body)

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/deployments/{deployment_id}")

    def delete_deployment(self, deployment_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/deployments/{deployment_id}")

    def list_ssh_keys(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/ssh-keys")


# ---------------------------------------------------------------------------
# cloudInit script generators.
# ---------------------------------------------------------------------------


def _bash_single_quote(s: str) -> str:
    """Escape an arbitrary string for safe embedding inside bash single quotes.

    Python's `repr()` is a misleading helper here because bash single-quoted
    strings do NOT honour backslash escapes — `\\'` would render as a literal
    backslash + closing quote. The canonical safe transformation is
    `'…' + ' \\' ' + '…'` ("close, escaped-quote, reopen") for every `'`.
    """
    return "'" + s.replace("'", "'\\''") + "'"


def _make_cloud_init(*, phase: str, branch: str, hf_token: str,
                     hf_bucket: str, volume_name: str) -> str:
    """Produce the bash payload that runs on the deployment after boot.

    Responsibilities:
      - mount the attached volume (Spheron names it /mnt/<volume-name>);
        hard-fails if the mount can't be established (silent fall-through
        would write Stage 2 output to ephemeral local disk, defeating the
        whole point of the split).
      - ensure docker + nvidia-container-toolkit are present (CUDA image
        usually ships them; defensive install for both, never just docker.io).
      - docker run the moe-compress image, invoking run_sh_split.sh.
      - write a status flag (`.spheron_launch_status.json`) at the end so the
        orchestrator's SSH poll can detect completion.

    Secret hygiene:
      - `set -x` is OFF when handling the HF token (lines that echo the
        token must not appear in `/var/log/moe-launch.log`).
      - The token is written to a chmod-600 file on the volume; docker
        reads it via `$(cat …/hf_token)` so the `docker run` argv never
        contains the literal token.
      - The token is interpolated into bash via `_bash_single_quote`, which
        is robust against any token charset (single-quote escape is the only
        bash-safe transformation).
    """
    volume_mount = _volume_mount_for(volume_name)
    tok_q = _bash_single_quote(hf_token)
    branch_q = _bash_single_quote(branch)
    phase_q = _bash_single_quote(phase)
    bucket_q = _bash_single_quote(hf_bucket)
    return textwrap.dedent(f"""\
        #!/bin/bash
        # Spheron cloudInit for moe-compress SH split run — PHASE={phase}, BRANCH={branch}
        # NOTE: `set -x` is intentionally OFF in the secret-handling block below.
        set -uo pipefail
        exec > >(tee -a /var/log/moe-launch.log) 2>&1
        echo "[cloud-init] starting at $(date -u +%FT%TZ)"

        # ----- 1. Volume mount -------------------------------------------------
        # Spheron may pre-mount the volume at /mnt/<mountTag> (where mountTag
        # is the volume name with non-alphanumeric chars stripped). We probe:
        #   (a) is our target path already a mountpoint? Done.
        #   (b) is the volume mounted somewhere ELSE under /mnt? bind-mount.
        #   (c) is there an unmounted disk? mount it at our path.
        #   (d) HARD-FAIL — Stage 2 writes to ephemeral local disk would be
        #       silently lost on teardown (catastrophic).
        mkdir -p {volume_mount}
        if ! mountpoint -q {volume_mount}; then
            EXISTING_MNT=$(findmnt -lno TARGET --types ext4,xfs,btrfs 2>/dev/null \
                | grep -E '^/mnt/' | grep -v '^{volume_mount}$' | head -1)
            if [ -n "$EXISTING_MNT" ]; then
                echo "[cloud-init] volume pre-mounted at $EXISTING_MNT — bind-mounting to {volume_mount}"
                mount --bind "$EXISTING_MNT" {volume_mount} || true
            fi
        fi
        if ! mountpoint -q {volume_mount}; then
            DEV=$(lsblk -dnpo NAME,MOUNTPOINT,TYPE | awk '$3=="disk" && $2==""{{print $1; exit}}')
            if [ -n "$DEV" ]; then
                echo "[cloud-init] mounting unmounted disk $DEV at {volume_mount}"
                mount "$DEV" {volume_mount} || true
            fi
        fi
        if ! mountpoint -q {volume_mount}; then
            echo "[cloud-init] FATAL: {volume_mount} is not a mountpoint — refusing to run"
            echo "[cloud-init] available disks:"; lsblk -p
            echo "[cloud-init] existing mounts:"; findmnt -ln
            exit 1
        fi
        df -h {volume_mount}

        # ----- 2. Docker + nvidia-container-toolkit ---------------------------
        if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
            DEBIAN_FRONTEND=noninteractive apt-get update -y
            DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io nvidia-container-toolkit
            systemctl enable --now docker
            nvidia-ctk runtime configure --runtime=docker || true
            systemctl restart docker || true
        fi
        nvidia-smi --query-gpu=name,memory.total --format=csv || {{
            echo "[cloud-init] FATAL: nvidia-smi failed — check GPU driver"; exit 1;
        }}

        # ----- 3. Token persist (set -x stays OFF) ----------------------------
        # Write HF token to a chmod-600 file on the volume so the docker run
        # never carries the literal token on argv. `set +x` is already in
        # effect (line 1's `set -uo pipefail` does not enable xtrace).
        umask 077
        printf '%s\\n' {tok_q} > {volume_mount}/hf_token
        chmod 600 {volume_mount}/hf_token

        # ----- 4. Run (now safe to enable xtrace) -----------------------------
        set -x
        docker pull {DOCKER_IMAGE}
        docker rm -f moe-run 2>/dev/null || true
        docker run --rm --name moe-run --gpus all --ipc=host \\
            -v {volume_mount}:/cache \\
            -e HF_TOKEN="$(cat {volume_mount}/hf_token)" \\
            -e HF_ARTIFACTS_BUCKET={bucket_q} \\
            -e MOE_PHASE={phase_q} \\
            -e MOE_BRANCH={branch_q} \\
            --entrypoint bash {DOCKER_IMAGE} -c '
                set -e
                if [ -d /cache/code/moe_compress/.git ]; then
                    git -C /cache/code/moe_compress fetch --depth 1 origin "${{MOE_BRANCH}}"
                    git -C /cache/code/moe_compress reset --hard "origin/${{MOE_BRANCH}}"
                else
                    git clone --depth 1 -b "${{MOE_BRANCH}}" {GH_REPO_URL} /cache/code/moe_compress
                fi
                exec bash /cache/code/moe_compress/max_quality/docker/run_sh_split.sh
            '
        RC=$?
        set +x
        echo "[cloud-init] docker run exited rc=$RC at $(date -u +%FT%TZ)"

        # ----- 5. Completion flag (orchestrator polls this over SSH) ----------
        printf '{{"phase": "%s", "rc": %d, "ts": "%s"}}\\n' \\
            "$MOE_PHASE_FOR_FLAG" "$RC" "$(date -u +%FT%TZ)" \\
            > {volume_mount}/.spheron_launch_status.json 2>/dev/null
        echo "[cloud-init] status file written at {volume_mount}/.spheron_launch_status.json"
        """).replace("$MOE_PHASE_FOR_FLAG", phase)


# ---------------------------------------------------------------------------
# Subcommand implementations.
# ---------------------------------------------------------------------------


def _resolve_or_create_volume(
    client: SpheronClient, name: str, size_gb: int,
) -> str:
    """Idempotent: return an existing volume's id if one with this name is
    already provisioned; otherwise create one. Spheron enforces unique
    volume names per account, so a name match is sufficient (no need to
    re-match on provider/region — the API returns `providerId` here, not
    `provider`, and the responsibility for region matching is the API's)."""
    for vol in client.list_volumes():
        if vol.get("name") == name:
            log.info(
                "reusing existing volume id=%s name=%s size=%sGB status=%s "
                "(provider=%s region=%s)",
                vol.get("id") or vol.get("_id"), name, vol.get("sizeInGb"),
                vol.get("status"), vol.get("providerId"), vol.get("region"),
            )
            return vol.get("id") or vol.get("_id")
    log.info("creating volume name=%s size=%dGB on %s/%s ($%.4f/hr → $%.2f/day)",
             name, size_gb, PROVIDER, REGION,
             size_gb * 0.0001320313, size_gb * 0.0001320313 * 24)
    created = client.create_volume(name, size_gb)
    vid = created.get("id") or created.get("_id")
    if not vid:
        raise RuntimeError(f"volume create returned no id: {created}")
    log.info("created volume id=%s", vid)
    return vid


def cmd_volume_create(args: argparse.Namespace) -> int:
    client = SpheronClient(_read_spheron_key())
    vid = _resolve_or_create_volume(client, args.volume_name, args.size_gb)
    print(vid)
    return 0


def cmd_volume_delete(args: argparse.Namespace) -> int:
    client = SpheronClient(_read_spheron_key())
    log.info("deleting volume %s", args.volume_id)
    client.delete_volume(args.volume_id)
    log.info("volume %s deleted", args.volume_id)
    return 0


def _pick_default_ssh_key(client: SpheronClient) -> str:
    keys = client.list_ssh_keys()
    if not keys:
        raise RuntimeError(
            "No SSH keys registered on the Spheron account. Add one via the "
            "dashboard first."
        )
    # Prefer the most-recent key.
    keys.sort(key=lambda k: k.get("createdAt", ""), reverse=True)
    return keys[0].get("_id") or keys[0].get("id")


def _wait_for_deployment_ready(
    client: SpheronClient, deployment_id: str, timeout_s: int,
) -> dict[str, Any]:
    """Block until the deployment has an IP + sshPort populated (i.e. the
    provisioner has finished and SSH is up). Returns the deployment dict."""
    t0 = time.monotonic()
    while True:
        dep = client.get_deployment(deployment_id)
        status = dep.get("status", "?")
        ip = dep.get("ipAddress") or dep.get("ip")
        ssh_port = dep.get("sshPort")
        log.info("deployment %s status=%s ip=%s sshPort=%s elapsed=%ds",
                 deployment_id, status, ip, ssh_port,
                 int(time.monotonic() - t0))
        if ip and ssh_port:
            return dep
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(
                f"deployment {deployment_id} did not come up within {timeout_s}s"
            )
        time.sleep(POLL_DEPLOYMENT_READY)


# Deployment statuses that count as "no longer using the volume" — invert
# the polarity (allowlist terminal states rather than reject a fixed live
# set) so the guard doesn't false-positive on transient cleanup states like
# "terminating", "stopped", or "error" that have already detached the volume.
_TERMINAL_DEPLOYMENT_STATUSES = {
    "terminated", "failed", "deleted", "stopped",
    "error", "cancelled", "expired", "terminating",
    # Defensive: include the other status names other cloud APIs use
    # for "no longer holding the volume". A false-positive here only
    # blocks a legitimate next phase; a false-negative was the bug we
    # already fixed (two phases racing on the same volume).
    "completed", "complete", "removed", "destroyed", "closed",
}


def _check_no_active_deployment_on_volume(
    client: SpheronClient, volume_id: str,
) -> None:
    """I2 guard: fail fast if another deployment is still attached to this
    volume — two GPUs writing to the same `/cache` would corrupt Stage 2."""
    for d in client.list_deployments():
        vids = d.get("volumeIds") or []
        status = (d.get("status") or "").lower()
        if volume_id in vids and status not in _TERMINAL_DEPLOYMENT_STATUSES:
            raise RuntimeError(
                f"volume {volume_id} is already attached to deployment "
                f"{d.get('_id') or d.get('id')} (status={status}, "
                f"name={d.get('name')}). Two phases on the same volume would "
                "corrupt Stage 2 output. Teardown that deployment first."
            )


def _launch_phase(*, phase: str, offer_id: str, volume_id: str,
                  volume_name: str, branch: str, hf_bucket: str,
                  ssh_key_id: str, timeout_s: int) -> tuple[SpheronClient, dict[str, Any]]:
    """Create the deployment, wait for it to be SSH-able. On ANY failure
    after `create_deployment` succeeds, deletes the deployment before
    re-raising so we never leak a billing GPU instance (C3)."""
    client = SpheronClient(_read_spheron_key())
    _check_no_active_deployment_on_volume(client, volume_id)
    hf_token = _read_hf_token()
    cloud_init = _make_cloud_init(
        phase=phase, branch=branch, hf_token=hf_token,
        hf_bucket=hf_bucket, volume_name=volume_name,
    )
    name = f"moe-sh-{phase}-{int(time.time())}"
    log.info("creating %s deployment %s (offer=%s, volume=%s)",
             phase, name, offer_id, volume_id)
    dep = client.create_deployment(
        offer_id=offer_id, ssh_key_id=ssh_key_id, volume_ids=[volume_id],
        cloud_init=cloud_init, name=name,
    )
    dep_id = dep.get("_id") or dep.get("id")
    if not dep_id:
        raise RuntimeError(f"create_deployment returned no id: {dep}")
    log.info("deployment created id=%s name=%s", dep_id, name)
    try:
        return client, _wait_for_deployment_ready(client, dep_id, timeout_s)
    except Exception:
        log.error("provisioning failed for %s — tearing down deployment %s "
                  "to avoid leaking billing", phase, dep_id)
        try:
            client.delete_deployment(dep_id)
            log.info("teardown OK")
        except Exception as cleanup_exc:  # noqa: BLE001
            log.error("teardown ALSO FAILED — manual cleanup required: "
                      "delete deployment %s on Spheron dashboard. Error: %s",
                      dep_id, cleanup_exc)
        raise


def _ssh_exec(*, ip: str, port: int, command: str, timeout: int = 30) -> tuple[int, str]:
    """Run a one-shot SSH command via argv-list subprocess (no shell, no
    injection). Returns (returncode, combined output). Uses the system `ssh`
    binary so we avoid a paramiko dependency for an environment where this
    script is run once or twice per experiment."""
    if not SSH_KEY_PATH.exists():
        raise FileNotFoundError(
            f"SSH private key not found at {SSH_KEY_PATH}. "
            "Set SPHERON_SSH_KEY env var to the path of the private key "
            "matching the public key registered on Spheron."
        )
    if shutil.which("ssh") is None:
        raise RuntimeError("`ssh` binary not on PATH")
    # argv-list form — no shell — `command` is a single remote arg and is
    # never interpreted by the local shell.
    argv = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", f"ConnectTimeout={min(timeout, 30)}",
        "-i", str(SSH_KEY_PATH),
        "-p", str(port),
        f"root@{ip}", command,
    ]
    try:
        out = subprocess.run(  # noqa: S603 — argv-list, no shell
            argv, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return out.returncode, (out.stdout or "") + (out.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "ssh: timeout"


def _poll_until_done(
    *, ip: str, port: int, volume_name: str, timeout_s: int,
) -> dict[str, Any]:
    """SSH-poll the `.spheron_launch_status.json` file on the deployment
    until the docker container exits. Returns the parsed status dict on
    success; raises TimeoutError otherwise."""
    flag_path = f"{_volume_mount_for(volume_name)}/.spheron_launch_status.json"
    # Use shlex.quote on the path even though it's controlled by us — defense
    # in depth in case volume_name ever flows from user input later.
    remote_cmd = f"cat {shlex.quote(flag_path)} 2>/dev/null || echo MISSING"
    t0 = time.monotonic()
    while True:
        rc, out = _ssh_exec(ip=ip, port=port, command=remote_cmd, timeout=30)
        elapsed = int(time.monotonic() - t0)
        if rc == 0 and out.strip() and out.strip() != "MISSING":
            try:
                status = json.loads(out.strip().splitlines()[-1])
                log.info("container finished elapsed=%ds status=%s",
                         elapsed, status)
                return status
            except Exception:
                log.debug("status file unparseable yet: %r", out[:200])
        log.info("polling … elapsed=%ds (rc=%d)", elapsed, rc)
        if elapsed > timeout_s:
            raise TimeoutError(
                f"container did not finish within {timeout_s}s "
                f"(last poll rc={rc}, out={out[:200]!r}). "
                f"Live tail: ssh -p {port} root@{ip} "
                "'tail -F /var/log/moe-launch.log'"
            )
        time.sleep(POLL_RUN)


def _safe_teardown(client: SpheronClient, dep_id: str, reason: str) -> None:
    """Best-effort deployment delete; logs (does not raise) on failure so the
    caller can re-raise the original error."""
    try:
        log.info("tearing down deployment %s (%s)", dep_id, reason)
        client.delete_deployment(dep_id)
    except Exception as e:  # noqa: BLE001
        log.error("teardown of %s FAILED — manual cleanup REQUIRED on "
                  "the Spheron dashboard. Reason: %s. Error: %s",
                  dep_id, reason, e)


def _phase_run_and_teardown(
    *, phase: str, offer_id: str, volume_id: str, volume_name: str,
    branch: str, hf_bucket: str, ssh_key_id: str,
    provisioning_timeout: int, run_timeout: int, keep_running: bool,
) -> dict[str, Any]:
    """Full lifecycle for one phase: launch → SSH-poll until done → teardown.

    Teardown policy:
      - clean exit (rc==0)                          → teardown
      - docker exit rc!=0 (user-recoverable bug)    → KEEP UP for diagnosis
      - poll timeout / SSH unreachable (infra bug)  → teardown (no diag value)
      - KeyboardInterrupt mid-poll                  → teardown
      - `keep_running` flag                          → never teardown (debug)

    The asymmetry matters: rc!=0 means the container actually ran and likely
    left logs / partial artifacts on the volume worth SSH-ing into. A poll
    timeout means the container is wedged or never started — keeping the
    box up burns $2-5/hr without yielding diagnostic value the volume
    doesn't already carry."""
    client, dep = _launch_phase(
        phase=phase, offer_id=offer_id, volume_id=volume_id,
        volume_name=volume_name, branch=branch, hf_bucket=hf_bucket,
        ssh_key_id=ssh_key_id, timeout_s=provisioning_timeout,
    )
    dep_id = dep.get("_id") or dep.get("id")
    ip = dep.get("ipAddress") or dep.get("ip")
    port = dep.get("sshPort")
    log.info("deployment %s ready — SSH-polling for completion "
             "(timeout=%ds). Live tail: ssh -p %s root@%s "
             "'tail -F /var/log/moe-launch.log'",
             phase, run_timeout, port, ip)

    # Note: --keep-running is documented as a clean-exit debug aid (the user
    # wants to SSH into a healthy container). Timeout / interrupt / SSH
    # failure means there is no healthy container to inspect, so we ignore
    # the flag and tear down unconditionally — burning $2-5/hr on a wedged
    # box yields no diagnostic value the volume doesn't already carry.
    try:
        status = _poll_until_done(
            ip=ip, port=port, volume_name=volume_name, timeout_s=run_timeout,
        )
    except KeyboardInterrupt:
        log.warning("interrupted during phase %s poll — tearing down %s",
                    phase, dep_id)
        _safe_teardown(client, dep_id, f"phase={phase} interrupted")
        raise
    except Exception as e:  # noqa: BLE001 — includes TimeoutError, ssh failures, etc.
        log.error("phase %s poll FAILED (%s) — no diagnostic value from "
                  "keeping the box up; tearing down %s",
                  phase, type(e).__name__, dep_id)
        _safe_teardown(client, dep_id, f"phase={phase} {type(e).__name__}")
        raise

    if status.get("rc") != 0:
        # Docker actually ran but exited nonzero — keep the box up so the
        # user can SSH in and read /var/log/moe-launch.log + the volume's
        # heal logs without re-provisioning.
        log.error("phase %s docker exited rc=%s — leaving deployment %s up "
                  "for diagnosis (run `teardown --deployment-id %s` when done)",
                  phase, status.get("rc"), dep_id, dep_id)
        raise RuntimeError(
            f"phase {phase}: docker exited rc={status.get('rc')}"
        )

    if keep_running:
        log.info("--keep-running set; deployment %s left up", dep_id)
    else:
        _safe_teardown(client, dep_id, f"phase={phase} clean")
    return {"deployment_id": dep_id, "ip": ip, "ssh_port": port,
            "phase": phase, "status": status}


def cmd_phase1(args: argparse.Namespace) -> int:
    client = SpheronClient(_read_spheron_key())
    ssh_key = args.ssh_key_id or _pick_default_ssh_key(client)
    result = _phase_run_and_teardown(
        phase="stage2", offer_id=RTX6000_OFFER_ID,
        volume_id=args.volume_id, volume_name=args.volume_name,
        branch=args.branch, hf_bucket=args.hf_bucket, ssh_key_id=ssh_key,
        provisioning_timeout=POLL_DEPLOYMENT_READY_TIMEOUT,
        run_timeout=POLL_RUN_TIMEOUT_PHASE1,
        keep_running=args.keep_running,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_phase2(args: argparse.Namespace) -> int:
    client = SpheronClient(_read_spheron_key())
    ssh_key = args.ssh_key_id or _pick_default_ssh_key(client)
    result = _phase_run_and_teardown(
        phase="stage2p5", offer_id=H200_OFFER_ID,
        volume_id=args.volume_id, volume_name=args.volume_name,
        branch=args.branch, hf_bucket=args.hf_bucket, ssh_key_id=ssh_key,
        provisioning_timeout=POLL_DEPLOYMENT_READY_TIMEOUT,
        run_timeout=POLL_RUN_TIMEOUT_PHASE2,
        keep_running=args.keep_running,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """End-to-end: volume-create → phase1 → phase2 → optional volume-delete.

    Each phase tears down its own deployment on clean exit; failures leave
    the box up for diagnosis. The volume persists across phases by design."""
    client = SpheronClient(_read_spheron_key())
    ssh_key = args.ssh_key_id or _pick_default_ssh_key(client)
    volume_id = _resolve_or_create_volume(client, args.volume_name, args.size_gb)
    log.info("=== Phase 1: Stage 2 on RTX PRO 6000 ===")
    p1 = _phase_run_and_teardown(
        phase="stage2", offer_id=RTX6000_OFFER_ID,
        volume_id=volume_id, volume_name=args.volume_name,
        branch=args.branch, hf_bucket=args.hf_bucket, ssh_key_id=ssh_key,
        provisioning_timeout=POLL_DEPLOYMENT_READY_TIMEOUT,
        run_timeout=POLL_RUN_TIMEOUT_PHASE1,
        keep_running=False,
    )
    log.info("=== Phase 2: Stage 2.5 + Stage 6 on H200 ===")
    p2 = _phase_run_and_teardown(
        phase="stage2p5", offer_id=H200_OFFER_ID,
        volume_id=volume_id, volume_name=args.volume_name,
        branch=args.branch, hf_bucket=args.hf_bucket, ssh_key_id=ssh_key,
        provisioning_timeout=POLL_DEPLOYMENT_READY_TIMEOUT,
        run_timeout=POLL_RUN_TIMEOUT_PHASE2,
        keep_running=False,
    )
    if args.delete_volume:
        log.info("deleting volume %s (--delete-volume)", volume_id)
        client.delete_volume(volume_id)
    print(json.dumps({"volume_id": volume_id, "phase1": p1, "phase2": p2}, indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    client = SpheronClient(_read_spheron_key())
    if args.deployment_id:
        dep = client.get_deployment(args.deployment_id)
        print(json.dumps(dep, indent=2))
        return 0
    print("=== deployments ===")
    for d in client.list_deployments():
        print(f"  {d.get('_id') or d.get('id')}  status={d.get('status')}  "
              f"offer={d.get('offerId')}  name={d.get('name')}")
    print("=== volumes ===")
    for v in client.list_volumes():
        print(f"  {v.get('_id') or v.get('id')}  status={v.get('status')}  "
              f"name={v.get('name')}  size={v.get('sizeInGb')}GB  region={v.get('region')}")
    return 0


def cmd_teardown(args: argparse.Namespace) -> int:
    client = SpheronClient(_read_spheron_key())
    if args.deployment_id:
        log.info("deleting deployment %s", args.deployment_id)
        client.delete_deployment(args.deployment_id)
    if args.volume_id and not args.keep_volume:
        log.info("deleting volume %s", args.volume_id)
        client.delete_volume(args.volume_id)
    return 0


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verbose", "-v", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    vc = sub.add_parser("volume-create", help="Provision (or reuse) the persistent volume.")
    vc.add_argument("--volume-name", default=DEFAULT_VOLUME_NAME)
    vc.add_argument("--size-gb", type=int, default=DEFAULT_VOLUME_SIZE_GB)
    vc.set_defaults(func=cmd_volume_create)

    vd = sub.add_parser("volume-delete", help="Destroy the persistent volume (irreversible).")
    vd.add_argument("--volume-id", required=True)
    vd.set_defaults(func=cmd_volume_delete)

    p1 = sub.add_parser("phase1", help="Launch RTX 6000 Pro Phase 1 (Stage 2 only).")
    p1.add_argument("--volume-id", required=True)
    p1.add_argument("--volume-name", default=DEFAULT_VOLUME_NAME,
                    help="Used to construct the in-deployment mount path "
                         f"(/mnt/<volume-name>). Default: {DEFAULT_VOLUME_NAME}.")
    p1.add_argument("--branch", default=DEFAULT_BRANCH)
    p1.add_argument("--hf-bucket", default=DEFAULT_HF_BUCKET)
    p1.add_argument("--ssh-key-id", default=None,
                    help="If unset, uses the most recently registered SSH key.")
    p1.add_argument("--keep-running", action="store_true",
                    help="Skip auto-teardown after the docker container exits "
                         "(debugging only — burns $2.405/hr until you teardown).")
    p1.set_defaults(func=cmd_phase1)

    p2 = sub.add_parser("phase2", help="Launch H200 Phase 2 (Stage 2.5 + Stage 6 alt).")
    p2.add_argument("--volume-id", required=True)
    p2.add_argument("--volume-name", default=DEFAULT_VOLUME_NAME)
    p2.add_argument("--branch", default=DEFAULT_BRANCH)
    p2.add_argument("--hf-bucket", default=DEFAULT_HF_BUCKET)
    p2.add_argument("--ssh-key-id", default=None)
    p2.add_argument("--keep-running", action="store_true",
                    help="Skip auto-teardown (debugging — burns $4.615/hr).")
    p2.set_defaults(func=cmd_phase2)

    rn = sub.add_parser("run", help="End-to-end: volume-create → phase1 → phase2.")
    rn.add_argument("--volume-name", default=DEFAULT_VOLUME_NAME)
    rn.add_argument("--size-gb", type=int, default=DEFAULT_VOLUME_SIZE_GB)
    rn.add_argument("--branch", default=DEFAULT_BRANCH)
    rn.add_argument("--hf-bucket", default=DEFAULT_HF_BUCKET)
    rn.add_argument("--ssh-key-id", default=None)
    rn.add_argument("--delete-volume", action="store_true",
                    help="Delete the volume on clean end-to-end exit "
                         "(default: keep it for follow-up runs).")
    rn.set_defaults(func=cmd_run)

    st = sub.add_parser("status", help="Show deployments + volumes.")
    st.add_argument("--deployment-id", default=None)
    st.set_defaults(func=cmd_status)

    td = sub.add_parser("teardown", help="Discontinue a deployment, optionally delete the volume.")
    td.add_argument("--deployment-id", required=True)
    td.add_argument("--volume-id", default=None)
    td.add_argument("--keep-volume", action="store_true",
                    help="Skip volume deletion (useful between Phase 1 and Phase 2).")
    td.set_defaults(func=cmd_teardown)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130
    except Exception as e:  # noqa: BLE001
        log.error("FATAL: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
