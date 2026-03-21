#!/bin/bash
# Overnight automation: monitors training VMs, builds and evals submissions automatically.
# Run with: nohup bash overnight.sh > overnight.log 2>&1 &

ZONE=europe-west1-b
EVAL_VM=yolo-train-3
DIR=/Users/oscar/dev/ainm/object-detection

echo "[$(date)] Overnight automation started"

while true; do
  echo ""
  echo "[$(date)] === Checking VMs ==="

  # Check VM2 (YOLOv8l nc=356)
  VM2_STATUS=$(gcloud compute ssh yolo-train-2 --zone=$ZONE --command='
    RUN=$(screen -ls 2>/dev/null | grep -c train)
    EP=$(grep "all" ~/train3.log 2>/dev/null | grep -v "Error\|File\|torch\|Traceback" | wc -l | xargs)
    MAP=$(grep "all" ~/train3.log 2>/dev/null | grep -v "Error\|File\|torch\|Traceback" | tail -1 | awk "{for(i=1;i<=NF;i++) if(\$i+0==\$i) {c++; if(c==5) print \$i}}")
    echo "$RUN|$EP|$MAP"
  ' 2>/dev/null)

  VM2_RUN=$(echo "$VM2_STATUS" | cut -d'|' -f1)
  VM2_EP=$(echo "$VM2_STATUS" | cut -d'|' -f2)
  VM2_MAP=$(echo "$VM2_STATUS" | cut -d'|' -f3)

  echo "[$(date)] VM2 YOLOv8l nc=356: running=$VM2_RUN epoch=$VM2_EP mAP=$VM2_MAP"

  # If VM2 finished or has a good checkpoint, build and eval a submission
  if [ "$VM2_RUN" = "0" ] && [ -n "$VM2_EP" ] && [ "$VM2_EP" -gt "10" ]; then
    echo "[$(date)] VM2 finished! Downloading weights and building submission..."

    # Check if we already evaluated this
    EVAL_NAME="v5_yolov8l_nc356_ep${VM2_EP}"
    ALREADY=$(gcloud compute ssh $EVAL_VM --zone=$ZONE --command="ls ~/eval/results/$EVAL_NAME/scores.json 2>/dev/null && echo yes || echo no" 2>/dev/null)

    if [ "$ALREADY" = "no" ]; then
      # Download weights
      gcloud compute scp yolo-train-2:~/runs/yolov8l_nc356_v2/weights/best.pt /tmp/yolov8l_nc356_best.pt --zone=$ZONE 2>/dev/null

      if [ -f /tmp/yolov8l_nc356_best.pt ]; then
        # Build submission on eval VM
        gcloud compute ssh $EVAL_VM --zone=$ZONE --command="mkdir -p ~/eval/submissions/$EVAL_NAME" 2>/dev/null
        gcloud compute scp /tmp/yolov8l_nc356_best.pt $EVAL_VM:~/eval/submissions/$EVAL_NAME/yolo_best.pt --zone=$ZONE 2>/dev/null
        gcloud compute scp $DIR/submission_v2/run.py $EVAL_VM:~/eval/submissions/$EVAL_NAME/run.py --zone=$ZONE 2>/dev/null

        echo "[$(date)] Submitted $EVAL_NAME for eval"
      fi
    else
      echo "[$(date)] $EVAL_NAME already evaluated"
    fi
  fi

  # Check VM1 (linear head)
  VM1_STATUS=$(gcloud compute ssh yolo-train --zone=$ZONE --command='
    if [ -f ~/runs/linear_head.pt ]; then echo "done"; else echo "training"; fi
  ' 2>/dev/null)
  echo "[$(date)] VM1 linear head: $VM1_STATUS"

  # Print eval results
  echo "[$(date)] === Current eval results ==="
  gcloud compute ssh $EVAL_VM --zone=$ZONE --command='
    for d in ~/eval/results/*/; do
      if [ -f "$d/scores.json" ]; then
        python3 -c "import json; d=json.load(open(\"${d}scores.json\")); print(f\"  {d.get(\"name\",\"?\"):<35s} {d[\"combined\"]:.4f}  det={d[\"det_mAP\"]:.4f}  cls={d[\"cls_mAP\"]:.4f}  {d[\"runtime\"]:.0f}s\")"
      fi
    done
  ' 2>/dev/null

  echo "[$(date)] Sleeping 5 minutes..."
  sleep 300
done
