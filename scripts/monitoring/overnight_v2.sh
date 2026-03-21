#!/bin/bash
# Overnight automation v2 — runs until morning
ZONE=europe-west1-b
DIR=/Users/oscar/dev/ainm/object-detection
LOG=$DIR/overnight_v2.log

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a $LOG; }

log "=== OVERNIGHT V2 STARTED ==="
log "Target: val > 0.9625 (test > 0.7802)"
log "Current best val: 0.9343, test: 0.7579"

ROUND=0
while true; do
    ROUND=$((ROUND + 1))
    log ""
    log "=== ROUND $ROUND ==="

    # Check VM1 crop classifier
    VM1_CLS=$(gcloud compute ssh yolo-train --zone=$ZONE --command='
        if [ -f ~/runs/crop_classifier.pt ]; then
            SIZE=$(ls -lh ~/runs/crop_classifier.pt | awk "{print \$5}")
            ACC=$(grep "val_acc" ~/train_crop_cls.log 2>/dev/null | tail -1 | grep -o "val_acc=[0-9.]*")
            echo "DONE|$SIZE|$ACC"
        else
            LAST=$(tail -1 ~/train_crop_cls.log 2>/dev/null)
            echo "TRAINING|$LAST"
        fi
    ' 2>/dev/null)
    log "VM1 crop classifier: $VM1_CLS"

    # If crop classifier is done, download it
    if echo "$VM1_CLS" | grep -q "DONE"; then
        if [ ! -f $DIR/crop_classifier.pt ]; then
            log "Downloading crop classifier..."
            gcloud compute scp yolo-train:~/runs/crop_classifier.pt $DIR/crop_classifier.pt --zone=$ZONE 2>/dev/null
            log "Downloaded: $(ls -lh $DIR/crop_classifier.pt 2>/dev/null | awk '{print $5}')"
        fi
    fi

    # Check VM2 YOLOv8x training
    VM2_STATUS=$(gcloud compute ssh yolo-train-2 --zone=$ZONE --command='
        RUN=$(screen -ls 2>/dev/null | grep -c "\.")
        LAST=$(grep "all" ~/train_x_v2.log 2>/dev/null | grep -v "Error\|File\|torch\|Traceback" | tail -1 | xargs)
        EP=$(grep "all" ~/train_x_v2.log 2>/dev/null | grep -v "Error\|File\|torch\|Traceback" | wc -l | xargs)
        echo "$RUN|$EP|$LAST"
    ' 2>/dev/null)
    log "VM2 YOLOv8x: $VM2_STATUS"

    # If VM2 has a good checkpoint and we haven't evaluated it
    VM2_EP=$(echo "$VM2_STATUS" | cut -d'|' -f2)
    if [ "$VM2_EP" -gt "20" ] 2>/dev/null; then
        EVAL_NAME="v8_yolov8x_nc356_ep${VM2_EP}"
        ALREADY=$(gcloud compute ssh yolo-train-3 --zone=$ZONE --command="[ -f ~/eval/results/$EVAL_NAME/scores.json ] && echo yes || echo no" 2>/dev/null)
        if [ "$ALREADY" = "no" ]; then
            log "Submitting $EVAL_NAME for eval..."
            gcloud compute scp yolo-train-2:~/runs/yolov8x_nc356_v2/weights/best.pt /tmp/yolov8x_best.pt --zone=$ZONE 2>/dev/null
            if [ -f /tmp/yolov8x_best.pt ]; then
                gcloud compute ssh yolo-train-3 --zone=$ZONE --command="mkdir -p ~/eval/submissions/$EVAL_NAME" 2>/dev/null
                gcloud compute scp /tmp/yolov8x_best.pt yolo-train-3:~/eval/submissions/$EVAL_NAME/yolo_best.pt --zone=$ZONE 2>/dev/null
                gcloud compute scp $DIR/submission_v5_multiscale/run.py yolo-train-3:~/eval/submissions/$EVAL_NAME/run.py --zone=$ZONE 2>/dev/null
                log "Submitted $EVAL_NAME for eval"
            fi
        fi
    fi

    # Pull latest eval results
    log "=== EVAL RESULTS ==="
    gcloud compute ssh yolo-train-3 --zone=$ZONE --command='
        for d in ~/eval/results/*/; do
            [ -f "$d/scores.json" ] || continue
            python3 -c "import json; d=json.load(open(\"${d}scores.json\")); print(f\"  {d.get(chr(110)+chr(97)+chr(109)+chr(101),chr(63)):<35s} {d.get(chr(99)+chr(111)+chr(109)+chr(98)+chr(105)+chr(110)+chr(101)+chr(100),0):.4f}  det={d.get(chr(100)+chr(101)+chr(116)+chr(95)+chr(109)+chr(65)+chr(80),0):.4f}  cls={d.get(chr(99)+chr(108)+chr(115)+chr(95)+chr(109)+chr(65)+chr(80),0):.4f}  {d.get(chr(114)+chr(117)+chr(110)+chr(116)+chr(105)+chr(109)+chr(101),0):.0f}s\")"
        done | sort -rn -k2 | head -5
    ' 2>/dev/null | tee -a $LOG

    # Check if any eval beat our target
    BEST_VAL=$(gcloud compute ssh yolo-train-3 --zone=$ZONE --command='
        for d in ~/eval/results/*/; do
            [ -f "$d/scores.json" ] || continue
            python3 -c "import json; print(json.load(open(\"${d}scores.json\")).get(chr(99)+chr(111)+chr(109)+chr(98)+chr(105)+chr(110)+chr(101)+chr(100),0))"
        done | sort -rn | head -1
    ' 2>/dev/null)
    log "Best val score: $BEST_VAL"

    log "Sleeping 10 minutes..."
    sleep 600
done
