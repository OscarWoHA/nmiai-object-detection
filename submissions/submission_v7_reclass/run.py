"""v7 selective reclassification: 3-scale WBF + re-run YOLO on ambiguous crops.

For low-confidence detections (score < 0.4), crop with 20% context,
resize, re-run YOLO. If new result has higher confidence for the same box,
replace the classification. Only reclassify ambiguous boxes to save runtime.
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

        # Identify ambiguous boxes for reclassification
        ambiguous_indices = []
        for i in range(len(merged_boxes)):
            if merged_scores[i] < 0.4:
                ambiguous_indices.append(i)

        # Reclassify ambiguous boxes by cropping with context and re-running
        if ambiguous_indices:
            crops = []
            for idx in ambiguous_indices:
                box = merged_boxes[idx]
                # 20% context padding
                bw = box[2] - box[0]
                bh = box[3] - box[1]
                pad = 0.20
                x1 = max(0, int((box[0] - bw*pad) * w))
                y1 = max(0, int((box[1] - bh*pad) * h))
                x2 = min(w, int((box[2] + bw*pad) * w))
                y2 = min(h, int((box[3] + bh*pad) * h))
                crop = img_np[y1:y2, x1:x2]
                if crop.shape[0] < 32 or crop.shape[1] < 32:
                    continue
                crops.append((idx, crop, x1, y1, x2, y2))

            # Run model on crops
            for idx, crop, cx1, cy1, cx2, cy2 in crops:
                results = model(crop, device=device, verbose=False,
                               imgsz=640, conf=0.05, iou=0.5, max_det=10)
                # Find the best detection that overlaps with our target box
                orig_box = merged_boxes[idx]
                crop_w = cx2 - cx1
                crop_h = cy2 - cy1
                best_conf = merged_scores[idx]
                best_cls = int(merged_labels[idx])

                for r in results:
                    if r.boxes is None: continue
                    for j in range(len(r.boxes)):
                        bx1, by1, bx2, by2 = r.boxes.xyxy[j].tolist()
                        # Convert crop coords to image coords
                        abs_box = [(bx1+cx1)/w, (by1+cy1)/h, (bx2+cx1)/w, (by2+cy1)/h]
                        # Check overlap with original box
                        xa = max(orig_box[0], abs_box[0])
                        ya = max(orig_box[1], abs_box[1])
                        xb = min(orig_box[2], abs_box[2])
                        yb = min(orig_box[3], abs_box[3])
                        inter = max(0, xb-xa) * max(0, yb-ya)
                        a1 = (orig_box[2]-orig_box[0]) * (orig_box[3]-orig_box[1])
                        a2 = (abs_box[2]-abs_box[0]) * (abs_box[3]-abs_box[1])
                        iou = inter/(a1+a2-inter) if (a1+a2-inter) > 0 else 0

                        if iou > 0.3 and float(r.boxes.conf[j].item()) > best_conf:
                            best_conf = float(r.boxes.conf[j].item())
                            best_cls = int(r.boxes.cls[j].item())

                merged_labels[idx] = best_cls
                merged_scores[idx] = max(merged_scores[idx], best_conf * 0.9)

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
