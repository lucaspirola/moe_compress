"""Cheap GPU probe — confirms whether torch sees CUDA on this flavor."""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     # Match the constraint in entrypoint.py so probes reflect the real run.
#     "torch>=2.5.0,<2.11.0",
# ]
# ///

import os
import subprocess
import sys


def main() -> int:
    print("=== GPU probe ===", flush=True)
    print("nvidia-smi:")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
            capture_output=True, text=True, timeout=10,
        )
        print(out.stdout or "(empty)")
        if out.stderr:
            print("stderr:", out.stderr, file=sys.stderr)
    except FileNotFoundError:
        print("nvidia-smi not on PATH")
    except Exception as err:                             # noqa: BLE001
        print(f"nvidia-smi error: {err}")

    print("Environment CUDA vars:")
    for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "NVIDIA_VISIBLE_DEVICES"):
        print(f"  {k} = {os.environ.get(k, '<unset>')}")

    import torch
    print(f"torch version          = {torch.__version__}")
    print(f"torch.version.cuda     = {torch.version.cuda}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  device {i}: {props.name} | {props.total_memory / 1e9:.1f} GB | capability {props.major}.{props.minor}")
    return 0 if torch.cuda.is_available() else 1


if __name__ == "__main__":
    sys.exit(main())
