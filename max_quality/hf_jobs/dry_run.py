"""HF Jobs dry-run sanity check (no GPU cost).

Runs on ``cpu-basic`` flavor, mounts the same bucket, exercises:
- the ``HF_TOKEN`` secret is present,
- the bucket is writable at the mount point,
- the code repo snapshot_download succeeds,
- the config YAML parses and is valid.

Does NOT touch the 35 B model snapshot (would cost download bandwidth).
Submit it before the real pipeline run to catch config drift.
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub>=0.26.0",
#     "pyyaml>=6.0",
# ]
# ///

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    print("=== HF Jobs dry-run ===", flush=True)

    token = os.environ.get("HF_TOKEN")
    assert token, "HF_TOKEN secret not set"
    print("HF_TOKEN present:", "sk-" if token.startswith("sk-") else token[:3] + "…")

    mount = Path(os.environ.get("CACHE_MOUNT", "/mnt/cache"))
    assert mount.exists(), f"mount missing: {mount}"
    probe = mount / ".dry_run_probe"
    probe.write_text("ok")
    assert probe.read_text() == "ok"
    probe.unlink()
    print(f"bucket mount OK at {mount}")

    code_repo = os.environ.get("CODE_REPO", "pirola/moe-compress-code")
    from huggingface_hub import snapshot_download
    dest = mount / "code_dry_run"
    snapshot_download(
        code_repo, repo_type="dataset", local_dir=dest,
        allow_patterns=["configs/*", "src/moe_compress/__init__.py",
                        "requirements.txt", "hf_jobs/*.py"],
    )
    assert (dest / "configs/qwen36_35b_a3b_30pct.yaml").exists(), "config missing in code snapshot"
    print(f"code snapshot_download OK ({len(list(dest.rglob('*')))} files under {dest})")

    import yaml
    cfg = yaml.safe_load((dest / "configs/qwen36_35b_a3b_30pct.yaml").read_text())
    assert cfg["target"]["total_reduction_ratio"] > 0
    assert cfg["stage1_grape"]["min_experts_per_layer"] >= 9
    print("config parses + validator invariant holds")

    print("=== dry-run complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
