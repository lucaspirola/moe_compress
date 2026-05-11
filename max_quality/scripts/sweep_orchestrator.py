#!/usr/bin/env python3
"""
Sweep orchestrator: price-chasing parallel ablation launcher.

Rules:
1. Find offer cheaper than current cheapest running instance
   → launch next A(n) on it immediately (parallel, don't kill existing)

2. Instance A(n) completes:
   - cheaper instance still running → just destroy it
   - this IS the cheapest (or only) running instance → destroy it AND
     immediately launch next A(n) on cheapest available offer right now

This pushes every ablation toward the cheapest GPU available at scheduling time.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

VASTAI = "/home/lucas/.venv-vastai/bin/vastai"
HF_TOKEN = Path("~/.cache/huggingface/token").expanduser().read_text().strip()
STATE_FILE = Path("/tmp/sweep_state.json")

ABLATIONS = ["A0","A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11"]


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "next_ablation_idx": 1,
        "running": {"36544987": {"ablation": "A0", "price": 3.7957}},
        "completed": [],
    }

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def cheapest_running_price(state):
    if not state["running"]:
        return 9999.0
    return min(float(v["price"]) for v in state["running"].values())

def cheapest_running_id(state):
    if not state["running"]:
        return None
    return min(state["running"], key=lambda k: float(state["running"][k]["price"]))

def get_offers():
    try:
        raw = subprocess.check_output([
            VASTAI, "search", "offers",
            "gpu_ram>=140 num_gpus=1 inet_up>=50 inet_down>=100 gpu_name!=A100_SXM4",
            "-o", "dph_total", "--raw"
        ], text=True, timeout=30, stderr=subprocess.DEVNULL)
        offers = json.loads(raw)
        offers.sort(key=lambda o: float(o.get("dph_total", 99)))
        return offers
    except Exception as e:
        print(f"[orch] offer search failed: {e}", flush=True)
        return []

def get_instances():
    try:
        raw = subprocess.check_output(
            [VASTAI, "show", "instances", "--raw"],
            text=True, timeout=30, stderr=subprocess.DEVNULL
        )
        data = json.loads(raw)
        items = data["instances"] if isinstance(data, dict) else data
        return {str(i["id"]): i for i in items}
    except Exception as e:
        print(f"[orch] instance list failed: {e}", flush=True)
        return None  # None = unknown, skip cycle

def is_complete(instance_id):
    try:
        logs = subprocess.check_output(
            [VASTAI, "logs", str(instance_id)],
            text=True, timeout=30, stderr=subprocess.DEVNULL
        )
        return ">>> RUN COMPLETE" in logs
    except Exception:
        return False

def launch_on_cheapest(ablation_id, offers):
    """Launch ablation on the cheapest available offer. Returns (instance_id, price) or (None, None)."""
    for offer in offers:
        price = float(offer.get("dph_total", 99))
        offer_id = offer["id"]
        print(f"[orch] launching {ablation_id} on offer {offer_id} @ ${price:.2f}/hr", flush=True)
        result = subprocess.run(
            ["bash", "-c",
             f"echo y | {VASTAI} create instance {offer_id} "
             f"--image ghcr.io/lucaspirola/moe-compress:latest "
             f"--disk 200 "
             f"--onstart-cmd '/usr/local/bin/bootstrap.sh' "
             f"--env \"-e HF_TOKEN={HF_TOKEN} -e ONLY={ablation_id} "
             f"-e PREFLIGHT_ONLY=0 -e UPLOAD_ON_SUCCESS=1\" "
             f"--ssh"],
            capture_output=True, text=True, timeout=60
        )
        out = result.stdout + result.stderr
        try:
            import ast
            data = ast.literal_eval(out[out.index("{"):out.rindex("}")+1])
            if data.get("new_contract"):
                new_iid = str(data["new_contract"])
                print(f"[orch] {ablation_id} started on instance {new_iid} @ ${price:.2f}/hr", flush=True)
                return new_iid, price
        except Exception:
            pass
        print(f"[orch] launch on offer {offer_id} failed: {out.strip()}", flush=True)
    return None, None

def destroy(instance_id):
    subprocess.run(
        ["bash", "-c", f"echo y | {VASTAI} destroy instance {instance_id}"],
        capture_output=True, timeout=30
    )
    print(f"[orch] destroyed instance {instance_id}", flush=True)


def main():
    state = load_state()
    save_state(state)
    print(
        f"[orch] started. next={ABLATIONS[state['next_ablation_idx']] if state['next_ablation_idx'] < len(ABLATIONS) else 'done'}, "
        f"running={list(state['running'].keys())}, "
        f"cheapest=${cheapest_running_price(state):.2f}",
        flush=True
    )

    while True:
        state = load_state()

        # ------------------------------------------------------------------ #
        # 0. Check if all done
        # ------------------------------------------------------------------ #
        if state["next_ablation_idx"] >= len(ABLATIONS) and not state["running"]:
            print("[orch] all ablations complete!", flush=True)
            break

        live_instances = get_instances()
        if live_instances is None:
            print("[orch] instance list unavailable — skipping cycle", flush=True)
            time.sleep(120)
            continue

        offers = get_offers()
        changed = False

        # ------------------------------------------------------------------ #
        # 1. Detect completions
        # ------------------------------------------------------------------ #
        for iid in list(state["running"].keys()):
            ablation = state["running"][iid]["ablation"]
            my_price = float(state["running"][iid]["price"])

            gone = iid not in live_instances
            done = gone or is_complete(iid)

            if not done:
                continue

            if gone:
                print(f"[orch] instance {iid} ({ablation}) disappeared unexpectedly", flush=True)
            else:
                print(f"[orch] {ablation} COMPLETE on {iid} @ ${my_price:.2f}/hr", flush=True)
                destroy(iid)

            state["running"].pop(iid)
            if ablation not in state["completed"]:
                state["completed"].append(ablation)
            changed = True

            # Rule 2: was this the cheapest (or only) running instance?
            # → immediately launch next ablation on cheapest available offer
            remaining_min = cheapest_running_price(state)  # after removing this one
            is_cheapest_gone = my_price <= remaining_min

            if is_cheapest_gone and state["next_ablation_idx"] < len(ABLATIONS):
                next_id = ABLATIONS[state["next_ablation_idx"]]
                print(f"[orch] cheapest instance completed — launching {next_id} immediately", flush=True)
                new_iid, price = launch_on_cheapest(next_id, offers)
                if new_iid:
                    state["running"][new_iid] = {"ablation": next_id, "price": price}
                    state["next_ablation_idx"] += 1

        if changed:
            save_state(state)

        # ------------------------------------------------------------------ #
        # 2. Launch next ablation if a cheaper offer exists
        # ------------------------------------------------------------------ #
        if state["next_ablation_idx"] < len(ABLATIONS):
            current_cheapest = cheapest_running_price(state)
            next_id = ABLATIONS[state["next_ablation_idx"]]

            for offer in offers:
                price = float(offer.get("dph_total", 99))
                if price < current_cheapest:
                    print(
                        f"[orch] offer ${price:.2f}/hr < running ${current_cheapest:.2f}/hr "
                        f"— launching {next_id} in parallel",
                        flush=True
                    )
                    new_iid, actual_price = launch_on_cheapest(next_id, [offer])
                    if new_iid:
                        state["running"][new_iid] = {"ablation": next_id, "price": actual_price}
                        state["next_ablation_idx"] += 1
                        save_state(state)
                        running_summary = [(v["ablation"], f"${v['price']:.2f}") for v in state["running"].values()]
                        print(f"[orch] now running: {running_summary}", flush=True)
                    break  # one launch per cycle; re-evaluate next cycle

        time.sleep(900)

    print("[orch] sweep done.", flush=True)


if __name__ == "__main__":
    main()
