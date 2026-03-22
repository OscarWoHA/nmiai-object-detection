"""v5 horizontal stripes — exploit shelf geometry.

Shelves are wide (2000x1500). Square resize to 1280 wastes vertical resolution.
Instead: run full image + 3 horizontal bands at higher effective resolution.
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
        img_np = np.array(img)

        all_boxes, all_scores, all_labels = [], [], []

        # Pass 1: Full image at 1280
        results = model(str(img_path), device=device, verbose=False,
                       imgsz=1280, conf=0.01, iou=0.5, max_det=500)
        boxes1, scores1, labels1 = [], [], []
        for r in results:
            if r.boxes is None: continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                boxes1.append([x1/w, y1/h, x2/w, y2/h])
                scores1.append(float(r.boxes.conf[i].item()))
                labels1.append(int(r.boxes.cls[i].item()))
        if boxes1:
            all_boxes.append(np.array(boxes1, dtype=np.float32))
            all_scores.append(np.array(scores1, dtype=np.float32))
            all_labels.append(np.array(labels1, dtype=np.float32))

        # Pass 2: Horizontal stripes (exploit shelf width)
        # 3 overlapping horizontal bands, each ~60% of image height
        stripe_height = int(h * 0.6)
        stripe_overlap = int(h * 0.15)
        stripe_starts = [0, (h - stripe_height) // 2, h - stripe_height]

        for y_start in stripe_starts:
            y_end = min(y_start + stripe_height, h)
            stripe = img_np[y_start:y_end, :, :]  # full width, partial height

            results = model(stripe, device=device, verbose=False,
                           imgsz=1280, conf=0.02, iou=0.5, max_det=300)

            boxes_s, scores_s, labels_s = [], [], []
            for r in results:
                if r.boxes is None: continue
                for i in range(len(r.boxes)):
                    bx1, by1, bx2, by2 = r.boxes.xyxy[i].tolist()
                    # Map back to full image coords (normalized)
                    boxes_s.append([bx1/w, (by1+y_start)/h, bx2/w, (by2+y_start)/h])
                    scores_s.append(float(r.boxes.conf[i].item()))
                    labels_s.append(int(r.boxes.cls[i].item()))

            if boxes_s:
                all_boxes.append(np.array(boxes_s, dtype=np.float32))
                all_scores.append(np.array(scores_s, dtype=np.float32))
                all_labels.append(np.array(labels_s, dtype=np.float32))

        if not all_boxes:
            continue

        # WBF merge — weight full image higher
        weights = [2.0] + [1.0] * (len(all_boxes) - 1)
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=weights, iou_thr=0.55, skip_box_thr=0.01, conf_type='max',
        )

        for i in range(len(merged_boxes)):
            box = merged_boxes[i]
            predictions.append({
                "image_id": image_id,
                "category_id": int(merged_labels[i]),
                "bbox": [
                    round(box[0]*w, 1), round(box[1]*h, 1),
                    round((box[2]-box[0])*w, 1), round((box[3]-box[1])*h, 1),
                ],
                "score": round(float(merged_scores[i]), 4),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Wrote {len(predictions)} predictions")


if __name__ == "__main__":
    main()
