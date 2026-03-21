#!/bin/bash
# Live training dashboard - polls both VMs every 30s
# Usage: bash dashboard.sh
ZONE=europe-west1-b

while true; do
  clear
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║              NM i AI 2026 — Training Dashboard                 ║"
  echo "║              $(date '+%Y-%m-%d %H:%M:%S')                              ║"
  echo "╠══════════════════════════════════════════════════════════════════╣"

  echo "║                                                                ║"
  echo "║  VM1 (yolo-train) — YOLOv8m nc=356                            ║"
  echo "║  ─────────────────────────────────────────────────────────────  ║"

  VM1=$(gcloud compute ssh yolo-train --zone=$ZONE --command='
    EPOCHS=$(grep -c "all" ~/train2.log 2>/dev/null || echo 0)
    LAST=$(grep "all" ~/train2.log 2>/dev/null | tail -1)
    GPU=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    SCREEN=$(screen -ls 2>/dev/null | grep -c train)
    echo "EPOCHS:$EPOCHS"
    echo "LAST:$LAST"
    echo "GPU:$GPU"
    echo "RUNNING:$SCREEN"
  ' 2>/dev/null)

  VM1_EPOCH=$(echo "$VM1" | grep "EPOCHS:" | cut -d: -f2)
  VM1_LAST=$(echo "$VM1" | grep "LAST:" | sed 's/LAST://' | xargs)
  VM1_GPU=$(echo "$VM1" | grep "GPU:" | cut -d: -f2)
  VM1_RUN=$(echo "$VM1" | grep "RUNNING:" | cut -d: -f2)

  if [ "$VM1_RUN" = "1" ]; then STATUS="TRAINING"; else STATUS="IDLE"; fi

  echo "║  Status: $STATUS | Epoch: $VM1_EPOCH/150"
  echo "║  GPU: ${VM1_GPU}MB"
  # Parse mAP from the last line
  if [ -n "$VM1_LAST" ]; then
    echo "║  Latest: $VM1_LAST"
  fi

  echo "║                                                                ║"
  echo "║  VM2 (yolo-train-2) — YOLOv8l nc=356 + DINOv2                 ║"
  echo "║  ─────────────────────────────────────────────────────────────  ║"

  VM2=$(gcloud compute ssh yolo-train-2 --zone=$ZONE --command='
    EPOCHS=$(grep -c "all" ~/train.log 2>/dev/null || echo 0)
    LAST=$(grep "all" ~/train.log 2>/dev/null | tail -1)
    GPU=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    SCREEN=$(screen -ls 2>/dev/null | grep -c train)
    DINOV2=$(grep "Saved.*embeddings" ~/train.log 2>/dev/null | tail -1)
    echo "EPOCHS:$EPOCHS"
    echo "LAST:$LAST"
    echo "GPU:$GPU"
    echo "RUNNING:$SCREEN"
    echo "DINOV2:$DINOV2"
  ' 2>/dev/null)

  VM2_EPOCH=$(echo "$VM2" | grep "EPOCHS:" | cut -d: -f2)
  VM2_LAST=$(echo "$VM2" | grep "LAST:" | sed 's/LAST://' | xargs)
  VM2_GPU=$(echo "$VM2" | grep "GPU:" | cut -d: -f2)
  VM2_RUN=$(echo "$VM2" | grep "RUNNING:" | cut -d: -f2)
  VM2_DINO=$(echo "$VM2" | grep "DINOV2:" | sed 's/DINOV2://')

  if [ "$VM2_RUN" = "1" ]; then STATUS="TRAINING"; else STATUS="IDLE"; fi

  echo "║  Status: $STATUS | Epoch: $VM2_EPOCH/200"
  echo "║  GPU: ${VM2_GPU}MB"
  if [ -n "$VM2_LAST" ]; then
    echo "║  Latest: $VM2_LAST"
  fi
  if [ -n "$VM2_DINO" ]; then
    echo "║  DINOv2: $VM2_DINO"
  fi

  echo "║                                                                ║"
  echo "╠══════════════════════════════════════════════════════════════════╣"
  echo "║  Competition score: 0.5583 (submission v1, detection-only)     ║"
  echo "║  Leader: 0.7618                                                ║"
  echo "║  Submissions remaining today: 2                                ║"
  echo "╚══════════════════════════════════════════════════════════════════╝"
  echo ""
  echo "  Refreshing in 30s... (Ctrl+C to exit)"
  sleep 30
done
