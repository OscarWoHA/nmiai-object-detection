"""v13: 2-model class-AGNOSTIC ensemble with weighted class voting.

CRITICAL FIX: Standard WBF won't merge boxes with different class labels,
creating duplicate detections that destroy det_mAP on test set.

Solution: Merge ALL boxes as class 0, then vote on class from overlapping raw detections.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image


def compute_iou_norm(box1, box2):
    """IoU between two [x1,y1,x2,y2] normalized boxes."""
    xa, ya = max(box1[0], box2[0]), max(box1[1], box2[1])
    xb, yb = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0


def run_model(model, img_path, device, w, h, imgsz=1280, conf=0.01):
    results = model(str(img_path), device=device, verbose=False,
                    imgsz=imgsz, conf=conf, iou=0.5, max_det=500)
    boxes, scores, labels = [], [], []
    for r in results:
        if r.boxes is None: continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            boxes.append([x1/w, y1/h, x2/w, y2/h])
            scores.append(float(r.boxes.conf[i].item()))
            labels.append(int(r.boxes.cls[i].item()))
    return boxes, scores, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = Path(__file__).parent

    model_a = YOLO(str(script_dir / "model_a.pt"))
    model_b = YOLO(str(script_dir / "model_b.pt"))

    predictions = []
    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])

        img = Image.open(img_path)
        w, h = img.size

        # Collect ALL raw detections with their class labels
        raw_detections = []  # (box_norm, score, class_id, weight)

        # Class-agnostic boxes for WBF (all labels = 0)
        all_boxes_agnostic = []
        all_scores_agnostic = []
        all_labels_agnostic = []
        wbf_weights = []

        configs = [
            (model_a, [(800, 0.02, 0.8), (1088, 0.015, 1.2), (1280, 0.01, 2.0), (1440, 0.01, 1.2)], 1.5),
            (model_b, [(1088, 0.015, 1.0), (1280, 0.01, 1.5), (1440, 0.01, 1.0)], 1.0),
        ]

        for model, scales, model_weight in configs:
            for imgsz, conf, scale_wt in scales:
                boxes, scores, labels = run_model(model, img_path, device, w, h, imgsz, conf)
                combined_wt = model_weight * scale_wt

                # Store raw detections for class voting later
                for b, s, l in zip(boxes, scores, labels):
                    raw_detections.append((b, s, l, combined_wt))

                if boxes:
                    all_boxes_agnostic.append(np.array(boxes, dtype=np.float32))
                    all_scores_agnostic.append(np.array(scores, dtype=np.float32))
                    # ALL ZEROS — class agnostic merge
                    all_labels_agnostic.append(np.zeros(len(boxes), dtype=np.float32))
                    wbf_weights.append(combined_wt)

        if not all_boxes_agnostic:
            continue

        # Step 1: Class-AGNOSTIC WBF merge (prevents duplicate boxes)
        merged_boxes, merged_scores, _ = weighted_boxes_fusion(
            all_boxes_agnostic, all_scores_agnostic, all_labels_agnostic,
            weights=wbf_weights, iou_thr=0.5, skip_box_thr=0.005, conf_type='max',
        )

        # Step 2: Vote on class for each merged box
        for i in range(len(merged_boxes)):
            mbox = merged_boxes[i]
            det_score = float(merged_scores[i])

            # Find all raw detections overlapping this merged box
            class_votes = {}
            for raw_box, raw_score, raw_cls, raw_wt in raw_detections:
                iou = compute_iou_norm(mbox, raw_box)
                if iou >= 0.45:
                    vote = raw_score * raw_wt
                    class_votes[raw_cls] = class_votes.get(raw_cls, 0) + vote

            if not class_votes:
                continue

            best_cls = max(class_votes, key=class_votes.get)

            predictions.append({
                "image_id": image_id,
                "category_id": best_cls,
                "bbox": [
                    round(mbox[0]*w, 1), round(mbox[1]*h, 1),
                    round((mbox[2]-mbox[0])*w, 1), round((mbox[3]-mbox[1])*h, 1),
                ],
                "score": round(det_score, 4),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Wrote {len(predictions)} predictions")


if __name__ == "__main__":
    main()
