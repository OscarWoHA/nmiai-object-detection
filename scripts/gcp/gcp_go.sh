#!/bin/bash
# Clean VM, upload training script, then SSH in to run it
set -e
ZONE=europe-west1-b
VM=yolo-train

echo "=== Cleaning up previous data on VM ==="
gcloud compute ssh $VM --zone=$ZONE -- 'rm -rf ~/data ~/runs ~/gcp_train.sh'

echo "=== Uploading training script ==="
gcloud compute scp /Users/oscar/dev/ainm/object-detection/gcp_train.sh $VM:~ --zone=$ZONE

echo "=== SSH-ing in — run: bash ~/gcp_train.sh ==="
gcloud compute ssh $VM --zone=$ZONE
