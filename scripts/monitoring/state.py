"""Centralized state manager. Single source of truth for dashboard.

Polls VMs, pulls eval results, scrapes leaderboard. Writes to state.json.
Everything reads from state.json.
"""
import json
import subprocess
import re
import time
import threading
from pathlib import Path

STATE_FILE = Path(__file__).parent / "state.json"
ZONE = "europe-west1-b"

VMS = [
    {"name": "yolo-train", "label": "YOLOv8m proper-split", "logs": ["~/retrain.log"]},
    {"name": "yolo-train-2", "label": "YOLOv8x nc=356 (full data)", "logs": ["~/train_x_v2.log"]},
    {"name": "yolo-train-3", "label": "Eval Server", "logs": ["~/eval/auto_eval.log"]},
    {"name": "yolo-train-4", "label": "YOLOv8l GPT-recipe (proper split)", "logs": ["~/retrain_v2.log"]},
]

state = {
    "vms": [],
    "evals": [],
    "leaderboard": [
        {"rank":1,"team":"Experis","score":0.7802},
        {"rank":2,"team":"Aibo","score":0.78},
        {"rank":3,"team":"Fenrir's byte","score":0.7774},
        {"rank":4,"team":"Synthetic Synapses","score":0.7743},
        {"rank":5,"team":"prompt injection 1678","score":0.7734},
        {"rank":6,"team":"000110 000111","score":0.7716},
        {"rank":7,"team":"Guru Meditation","score":0.7694},
        {"rank":8,"team":"Ave Christus Rex","score":0.7687},
        {"rank":9,"team":"Team Ayfie","score":0.7658},
        {"rank":10,"team":"websecured.io","score":0.7657},
        {"rank":11,"team":"Paralov","score":0.7579},
    ],
    "submissions": [
        {"version": "v1", "model": "YOLOv8m nc=1", "test_score": 0.5583},
        {"version": "v2", "model": "YOLOv8m nc=356 + TTA", "test_score": 0.7413},
        {"version": "v4", "model": "YOLOv8l nc=1 + DINOv2", "test_score": 0.6863},
        {"version": "v5_multiscale", "model": "YOLOv8l nc=356 3-scale WBF", "test_score": 0.7579},
    ],
    "last_update": "",
    "best_test_score": 0.7579,
    "best_val_score": 0.9343,
    "leader_score": 0.7802,
}


def ssh_cmd(vm_name, cmd, timeout=15):
    try:
        r = subprocess.run(
            ["gcloud", "compute", "ssh", vm_name, f"--zone={ZONE}", f"--command={cmd}"],
            capture_output=True, text=True, timeout=timeout
        )
        return r.stdout.strip()
    except:
        return ""


def poll_vms():
    vms = []
    for vm in VMS:
        try:
            # Get training status
            log_checks = " || ".join([f'grep "all" {l} 2>/dev/null | grep -v "Error\\|File\\|torch\\|Traceback" | tail -1' for l in vm["logs"]])
            ep_checks = " || ".join([f'grep -c "all" {l} 2>/dev/null' for l in vm["logs"]])

            data = ssh_cmd(vm["name"], f"""
LAST=$({log_checks})
EP=$({ep_checks})
RUN=$(screen -ls 2>/dev/null | grep -c "\\.")
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
echo "$EP|$RUN|$GPU_USED|$GPU_TOTAL|$GPU_UTIL|$LAST"
""")
            parts = data.split("|", 5)
            epoch = 0
            try:
                epoch = int(parts[0].strip().split("\n")[-1])
            except:
                pass

            # Parse mAP from last line
            map50 = 0
            precision = 0
            recall = 0
            if len(parts) > 5:
                nums = re.findall(r"[\d.]+", parts[5])
                if len(nums) >= 6:
                    try:
                        precision = float(nums[2])
                        recall = float(nums[3])
                        map50 = float(nums[4])
                    except:
                        pass

            vms.append({
                "name": vm["name"],
                "label": vm["label"],
                "epoch": epoch,
                "running": parts[1].strip() != "0" if len(parts) > 1 else False,
                "gpu_used": int(parts[2].strip()) if len(parts) > 2 and parts[2].strip().isdigit() else 0,
                "gpu_total": int(parts[3].strip()) if len(parts) > 3 and parts[3].strip().isdigit() else 0,
                "gpu_util": int(parts[4].strip()) if len(parts) > 4 and parts[4].strip().isdigit() else 0,
                "map50": map50,
                "precision": precision,
                "recall": recall,
            })
        except Exception as e:
            vms.append({"name": vm["name"], "label": vm["label"], "error": str(e)[:50]})

    state["vms"] = vms


def poll_evals():
    """Pull eval results from eval VM."""
    try:
        data = ssh_cmd("yolo-train-3", """
for d in ~/eval/results/*/; do
  if [ -f "$d/scores.json" ]; then
    cat "$d/scores.json"
    echo "|||"
  fi
done
""", timeout=10)
        evals = []
        for chunk in data.split("|||"):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                r = json.loads(chunk)
                evals.append(r)
            except:
                pass

        evals.sort(key=lambda x: -x.get("combined", 0))
        state["evals"] = evals
        if evals:
            state["best_val_score"] = evals[0]["combined"]
    except:
        pass


def save_state():
    state["last_update"] = time.strftime("%H:%M:%S")
    STATE_FILE.write_text(json.dumps(state, indent=2))


def poll_loop():
    while True:
        poll_vms()
        poll_evals()
        save_state()
        time.sleep(15)


if __name__ == "__main__":
    print("State manager running. Writing to state.json every 15s.")
    poll_loop()
