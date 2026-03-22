"""v17: Bug-fixed 2-model ensemble.

Fixes identified by GPT-5.4 and Gemini debugging:
1. max_det=500→2000 (dense shelves can have 800+ products)
2. WBF iou_thr=0.5→0.65 (adjacent products overlap, don't merge them)
3. YOLO NMS iou=0.5→0.7 (same reason, don't suppress adjacent products)
4. conf_type='avg'→'max' (preserve high-confidence single-model detections)
5. Clamp boxes to image bounds
6. No rounding of scores (preserve ranking precision)
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


def run_model(model, img_path, device, w, h, imgsz=1280, conf=0.01):
    results = model(str(img_path), device=device, verbose=False,
                    imgsz=imgsz, conf=conf,
                    iou=0.7,       # FIX: was 0.5, suppressed adjacent products
                    max_det=2000)  # FIX: was 500, capped recall on dense shelves
    boxes, scores, labels = [], [], []
    for r in results:
        if r.boxes is None: continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            # FIX: clamp to image bounds
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
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

        all_boxes, all_scores, all_labels, weights = [], [], [], []

        # Model A at sweep-optimized scales
        for imgsz, conf, wt in [(800, 0.02, 0.8), (1088, 0.015, 1.2), (1280, 0.01, 2.0), (1440, 0.01, 1.2)]:
            boxes, scores, labels = run_model(model_a, img_path, device, w, h, imgsz, conf)
            if boxes:
                all_boxes.append(np.array(boxes, dtype=np.float32))
                all_scores.append(np.array(scores, dtype=np.float32))
                all_labels.append(np.array(labels, dtype=np.float32))
                weights.append(wt * 1.5)

        # Model B at key scales
        for imgsz, conf, wt in [(1088, 0.015, 1.0), (1280, 0.01, 1.5), (1440, 0.01, 1.0)]:
            boxes, scores, labels = run_model(model_b, img_path, device, w, h, imgsz, conf)
            if boxes:
                all_boxes.append(np.array(boxes, dtype=np.float32))
                all_scores.append(np.array(scores, dtype=np.float32))
                all_labels.append(np.array(labels, dtype=np.float32))
                weights.append(wt)

        if not all_boxes:
            continue

        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=weights,
            iou_thr=0.65,          # FIX: was 0.5, adjacent products were being merged
            skip_box_thr=0.005,
            conf_type='max',       # FIX: was 'avg', was destroying high-conf single-model detections
        )

        for i in range(len(merged_boxes)):
            box = merged_boxes[i]
            # FIX: clamp fused boxes to valid range
            bx1 = max(0, box[0]) * w
            by1 = max(0, box[1]) * h
            bx2 = min(1, box[2]) * w
            by2 = min(1, box[3]) * h
            bw = bx2 - bx1
            bh = by2 - by1
            if bw < 1 or bh < 1:
                continue

            predictions.append({
                "image_id": image_id,
                "category_id": int(merged_labels[i]),
                "bbox": [round(bx1, 1), round(by1, 1), round(bw, 1), round(bh, 1)],
                "score": float(merged_scores[i]),  # FIX: no rounding, preserve ranking
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Wrote {len(predictions)} predictions")


if __name__ == "__main__":
    main()
