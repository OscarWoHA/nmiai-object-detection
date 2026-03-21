#!/bin/bash
# AUTOPILOT: Poll VMs, auto-download weights, auto-eval, auto-build submissions
# Runs for 8 hours (48 rounds of 10 min)

ZONE=europe-west1-b
DIR=/Users/oscar/dev/ainm/object-detection
LOG=$DIR/autopilot.log
EVAL_VM=yolo-train-3

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a $LOG; }

log "=== AUTOPILOT STARTED — running for 8 hours ==="

for ROUND in $(seq 1 48); do
    log ""
    log "=== ROUND $ROUND/48 ==="

    # --- Poll all VMs ---
    for VM_ENTRY in "yolo-train:retrain.log:vm1_yolov8m_proper" "yolo-train-2:train_x_v2.log:vm2_yolov8x" "yolo-train-4:retrain_v2.log:vm4_yolov8l_gpt"; do
        IFS=: read -r VM LOGFILE LABEL <<< "$VM_ENTRY"
        
        STATUS=$(gcloud compute ssh $VM --zone=$ZONE --command="
            RUN=\$(screen -ls 2>/dev/null | grep -c '\\.')
            LAST=\$(grep 'all' ~/$LOGFILE 2>/dev/null | grep -v 'Error\|File\|torch\|Traceback' | tail -1 | xargs)
            EP=\$(grep 'all' ~/$LOGFILE 2>/dev/null | grep -v 'Error\|File\|torch\|Traceback' | wc -l | xargs)
            BEST_FILE=''
            for d in ~/runs/*/weights/best.pt; do
                [ -f \"\$d\" ] && BEST_FILE=\$d
            done
            echo \"\$RUN|\$EP|\$BEST_FILE|\$LAST\"
        " 2>/dev/null)
        
        VM_RUN=$(echo "$STATUS" | cut -d'|' -f1)
        VM_EP=$(echo "$STATUS" | cut -d'|' -f2)
        VM_BEST=$(echo "$STATUS" | cut -d'|' -f3)
        VM_LAST=$(echo "$STATUS" | cut -d'|' -f4-)
        
        log "$LABEL: running=$VM_RUN ep=$VM_EP"
        
        # Extract mAP from last line
        MAP=$(echo "$VM_LAST" | awk '{for(i=1;i<=NF;i++) if($i+0==$i) {c++; if(c==5) {print $i; exit}}}' 2>/dev/null)
        [ -n "$MAP" ] && log "  mAP@0.5: $MAP"
        
        # If training finished and epoch > 20, download and eval
        if [ "$VM_RUN" = "0" ] && [ -n "$VM_EP" ] && [ "$VM_EP" -gt "20" ] 2>/dev/null; then
            EVAL_NAME="${LABEL}_ep${VM_EP}"
            ALREADY=$(gcloud compute ssh $EVAL_VM --zone=$ZONE --command="[ -f ~/eval/results/$EVAL_NAME/scores.json ] && echo yes || echo no" 2>/dev/null)
            
            if [ "$ALREADY" = "no" ] && [ -n "$VM_BEST" ]; then
                log "  FINISHED! Downloading weights and submitting for eval..."
                gcloud compute scp ${VM}:${VM_BEST} /tmp/${LABEL}_best.pt --zone=$ZONE 2>/dev/null
                
                if [ -f /tmp/${LABEL}_best.pt ]; then
                    SIZE=$(ls -lh /tmp/${LABEL}_best.pt | awk '{print $5}')
                    log "  Downloaded: $SIZE"
                    
                    # Submit to eval with multiscale run.py
                    gcloud compute ssh $EVAL_VM --zone=$ZONE --command="mkdir -p ~/eval/submissions/$EVAL_NAME" 2>/dev/null
                    gcloud compute scp /tmp/${LABEL}_best.pt $EVAL_VM:~/eval/submissions/$EVAL_NAME/yolo_best.pt --zone=$ZONE 2>/dev/null
                    gcloud compute scp $DIR/submission_v5_multiscale/run.py $EVAL_VM:~/eval/submissions/$EVAL_NAME/run.py --zone=$ZONE 2>/dev/null
                    
                    log "  Submitted $EVAL_NAME for eval"
                fi
            fi
        fi
        
        # If VM crashed, log alert
        if [ "$VM_RUN" = "0" ] && [ -n "$VM_EP" ] && [ "$VM_EP" -lt "20" ] 2>/dev/null; then
            log "  WARNING: $LABEL appears idle with only $VM_EP epochs — may have crashed!"
        fi
    done

    # --- Check eval results ---
    log "--- Eval Results ---"
    gcloud compute ssh $EVAL_VM --zone=$ZONE --command='
        for d in ~/eval/results/*/; do
            [ -f "$d/scores.json" ] || continue
            python3 -c "import json; d=json.load(open(\"${d}scores.json\")); print(f\"  {d.get(chr(110)+chr(97)+chr(109)+chr(101),chr(63)):<35s} {d.get(chr(99)+chr(111)+chr(109)+chr(98)+chr(105)+chr(110)+chr(101)+chr(100),0):.4f}  det={d.get(chr(100)+chr(101)+chr(116)+chr(95)+chr(109)+chr(65)+chr(80),0):.4f}  cls={d.get(chr(99)+chr(108)+chr(115)+chr(95)+chr(109)+chr(65)+chr(80),0):.4f}  {d.get(chr(114)+chr(117)+chr(110)+chr(116)+chr(105)+chr(109)+chr(101),0):.0f}s\")"
        done | sort -rn -k2 | head -5
    ' 2>/dev/null | tee -a $LOG

    # --- Update state.json for dashboard ---
    python3 -c "
import json, subprocess
s = json.loads(open('$DIR/state.json').read())
s['last_update'] = '$(date +%H:%M:%S)'
open('$DIR/state.json','w').write(json.dumps(s, indent=2))
" 2>/dev/null

    log "Sleeping 10 min (round $ROUND/48)..."
    sleep 600
done

log "=== AUTOPILOT FINISHED (8 hours) ==="
