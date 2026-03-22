"""v7: Agnostic WBF + class voting.

Per GPT-5.4: merge boxes ignoring class labels (avoids duplicate boxes when
different scales predict different classes for the same product). Then vote
on class from all contributing detections weighted by confidence * scale_weight.

This directly fixes the megascale failure mode where class-aware WBF creates
duplicates when scales disagree on class.
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


def compute_iou(box1, box2):
    """IoU between two [x1,y1,x2,y2] normalized boxes."""
    xa, ya = max(box1[0], box2[0]), max(box1[1], box2[1])
    xb, yb = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = Path(__file__).parent
    model = YOLO(str(script_dir / "yolo_best.pt"))

    predictions = []
    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])

        img = Image.open(img_path)
        w, h = img.size

        # Collect raw detections from multiple scales
        # Store: (box_norm, score, class_id, scale_weight)
        raw_detections = []

        scales = [
            (960,  0.02, 1.0),
            (1280, 0.01, 2.0),
            (1536, 0.02, 1.5),
        ]

        all_boxes_agnostic = []
        all_scores_agnostic = []
        all_labels_agnostic = []  # all zeros for class-agnostic fusion
        scale_weights = []

        for imgsz, conf, weight in scales:
            results = model(str(img_path), device=device, verbose=False,
                           imgsz=imgsz, conf=conf, iou=0.5, max_det=500)

            boxes, scores = [], []
            for r in results:
                if r.boxes is None:
                    continue
                for i in range(len(r.boxes)):
                    x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                    box_norm = [x1/w, y1/h, x2/w, y2/h]
                    score = float(r.boxes.conf[i].item())
                    cls_id = int(r.boxes.cls[i].item())
                    raw_detections.append((box_norm, score, cls_id, weight))
                    boxes.append(box_norm)
                    scores.append(score)

            if boxes:
                all_boxes_agnostic.append(np.array(boxes, dtype=np.float32))
                all_scores_agnostic.append(np.array(scores, dtype=np.float32))
                all_labels_agnostic.append(np.zeros(len(boxes), dtype=np.float32))
                scale_weights.append(weight)

        if not all_boxes_agnostic:
            continue

        # Step 1: Class-AGNOSTIC WBF merge (all labels = 0)
        merged_boxes, merged_scores, _ = weighted_boxes_fusion(
            all_boxes_agnostic, all_scores_agnostic, all_labels_agnostic,
            weights=scale_weights, iou_thr=0.55, skip_box_thr=0.01, conf_type='max',
        )

        # Step 2: For each merged box, vote on class from contributing raw detections
        for i in range(len(merged_boxes)):
            mbox = merged_boxes[i]
            det_score = float(merged_scores[i])

            # Find all raw detections that overlap with this merged box
            class_votes = {}  # class_id -> total weighted score
            for raw_box, raw_score, raw_cls, raw_weight in raw_detections:
                iou = compute_iou(mbox, raw_box)
                if iou >= 0.5:
                    vote_strength = raw_score * raw_weight
                    class_votes[raw_cls] = class_votes.get(raw_cls, 0) + vote_strength

            if not class_votes:
                continue

            # Winner takes all
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
