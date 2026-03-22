"""v6 multi-scale + spatial row voting.

Run at 3 scales + WBF, then cluster detections into shelf rows
and use neighbor consensus to fix misclassifications.
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
from collections import Counter


def row_voting(predictions, y_tolerance=0.03):
    """Cluster predictions by y-center into rows, then use majority voting
    to fix misclassifications. Products repeat in horizontal blocks."""
    if len(predictions) < 3:
        return predictions

    # Sort by y-center
    for p in predictions:
        p["_cy"] = (p["bbox"][1] + p["bbox"][3] / 2)

    sorted_preds = sorted(predictions, key=lambda p: p["_cy"])

    # Cluster into rows using y-center gaps
    rows = []
    current_row = [sorted_preds[0]]
    img_h = max(p["bbox"][1] + p["bbox"][3] for p in predictions)

    for p in sorted_preds[1:]:
        if abs(p["_cy"] - current_row[-1]["_cy"]) < y_tolerance * img_h:
            current_row.append(p)
        else:
            rows.append(current_row)
            current_row = [p]
    rows.append(current_row)

    # Within each row, find repeated classes and fix outliers
    for row in rows:
        if len(row) < 3:
            continue

        # Sort by x position
        row.sort(key=lambda p: p["bbox"][0])

        # Look at sliding windows of 5 for consensus
        for i in range(len(row)):
            window = row[max(0, i-2):min(len(row), i+3)]
            if len(window) < 3:
                continue

            cats = [p["category_id"] for p in window]
            cat_counts = Counter(cats)
            most_common_cat, most_common_count = cat_counts.most_common(1)[0]

            # If this prediction disagrees with majority AND its confidence is low
            current = row[i]
            if (current["category_id"] != most_common_cat and
                most_common_count >= 3 and
                current["score"] < 0.5):
                current["category_id"] = most_common_cat
                current["score"] *= 0.9  # slight penalty for being overridden

    # Clean up temp field
    for p in predictions:
        p.pop("_cy", None)

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = Path(__file__).parent
    model = YOLO(str(script_dir / "yolo_best.pt"))

    all_predictions = []
    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])

        img = Image.open(img_path)
        w, h = img.size

        all_boxes, all_scores, all_labels = [], [], []

        for imgsz, conf, weight in [(1280, 0.01, 2.0), (960, 0.02, 1.0), (1536, 0.02, 1.5)]:
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
            if boxes:
                all_boxes.append(np.array(boxes, dtype=np.float32))
                all_scores.append(np.array(scores, dtype=np.float32))
                all_labels.append(np.array(labels, dtype=np.float32))

        if not all_boxes:
            continue

        weights = [2.0, 1.0, 1.5][:len(all_boxes)]
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=weights, iou_thr=0.55, skip_box_thr=0.01, conf_type='max',
        )

        img_preds = []
        for i in range(len(merged_boxes)):
            box = merged_boxes[i]
            img_preds.append({
                "image_id": image_id,
                "category_id": int(merged_labels[i]),
                "bbox": [
                    round(box[0]*w, 1), round(box[1]*h, 1),
                    round((box[2]-box[0])*w, 1), round((box[3]-box[1])*h, 1),
                ],
                "score": round(float(merged_scores[i]), 4),
            })

        # Apply spatial row voting
        img_preds = row_voting(img_preds)
        all_predictions.extend(img_preds)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_predictions, f)
    print(f"Wrote {len(all_predictions)} predictions")


if __name__ == "__main__":
    main()
