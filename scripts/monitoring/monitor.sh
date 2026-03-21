#!/bin/bash
# Monitor all VMs every 5 min. Writes to monitor_status.json and monitor_alerts.txt.
ZONE=europe-west1-b
DIR=/Users/oscar/dev/ainm/object-detection

while true; do
  TS=$(date '+%H:%M:%S')
  ALERTS=""

  VM1=$(gcloud compute ssh yolo-train --zone=$ZONE --command='
    LAST=$(grep "all" ~/train_nc1l.log 2>/dev/null | grep -v "Error\|File\|torch\|Traceback" | tail -1 | xargs)
    EP=$(grep "all" ~/train_nc1l.log 2>/dev/null | grep -v "Error\|File\|torch\|Traceback" | wc -l | xargs)
    RUN=$(screen -ls 2>/dev/null | grep -c train)
    echo "$EP|$RUN|$LAST"
  ' 2>/dev/null)

  VM2=$(gcloud compute ssh yolo-train-2 --zone=$ZONE --command='
    LAST=$(grep "all" ~/train_l.log 2>/dev/null | grep -v "Error\|File\|torch\|Traceback" | tail -1 | xargs)
    EP=$(grep "all" ~/train_l.log 2>/dev/null | grep -v "Error\|File\|torch\|Traceback" | wc -l | xargs)
    RUN=$(screen -ls 2>/dev/null | grep -c train)
    echo "$EP|$RUN|$LAST"
  ' 2>/dev/null)

  VM3=$(gcloud compute ssh yolo-train-3 --zone=$ZONE --command='
    LAST=$(grep "all" ~/train.log 2>/dev/null | grep -v "Error\|File\|torch\|Traceback" | tail -1 | xargs)
    EP=$(grep "all" ~/train.log 2>/dev/null | grep -v "Error\|File\|torch\|Traceback" | wc -l | xargs)
    RUN=$(screen -ls 2>/dev/null | grep -c train)
    echo "$EP|$RUN|$LAST"
  ' 2>/dev/null)

  python3 << PYEOF
import json

vms_raw = {
    "VM1 YOLOv8l_nc1": "$VM1",
    "VM2 YOLOv8l_nc356": "$VM2",
    "VM3 YOLOv8x_nc356": "$VM3",
}

alerts = []
vms = []
for label, raw in vms_raw.items():
    parts = raw.split("|", 2)
    ep = int(parts[0]) if parts[0].strip().isdigit() else 0
    running = parts[1].strip() == "1" if len(parts) > 1 else False
    last = parts[2].strip() if len(parts) > 2 else ""

    # Extract mAP from last line (5th number)
    nums = [x for x in last.split() if x.replace(".", "").isdigit()]
    map50 = float(nums[4]) if len(nums) >= 5 else 0.0

    status = "TRAINING" if running else "IDLE"
    vms.append({"label": label, "epoch": ep, "map50": map50, "running": running})

    if not running and ep > 0:
        alerts.append(f"$TS ALERT: {label} is IDLE (crashed/finished at epoch {ep})")
    if map50 > 0.85:
        alerts.append(f"$TS HIGH: {label} hit mAP={map50:.3f} — DOWNLOAD WEIGHTS!")

    print(f"  {label:25s} ep={ep:3d} mAP={map50:.3f} [{status}]")

if alerts:
    print("  >>> ALERTS <<<")
    for a in alerts:
        print(f"  {a}")
    with open("$DIR/monitor_alerts.txt", "a") as f:
        for a in alerts:
            f.write(a + "\n")

with open("$DIR/monitor_status.json", "w") as f:
    json.dump({"timestamp": "$TS", "vms": vms, "alerts": alerts}, f, indent=2)
PYEOF

  echo "[$TS] Poll complete. Next in 5 min."
  sleep 300
done
