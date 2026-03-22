"""v10: Same YOLOv8l model, sweep-optimized WBF parameters.

Sweep found: 4 scales (800,1088,1280,1440), wbf_iou=0.5, conf_type=avg, skip=0.005
Val: 0.9403 (up from 0.9343 with old params)
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

        all_boxes, all_scores, all_labels = [], [], []

        # Sweep-optimized 4-scale config
        scales = [
            (800,  0.02, 0.8),
            (1088, 0.015, 1.2),
            (1280, 0.01, 2.0),
            (1440, 0.01, 1.2),
        ]

        for imgsz, conf, weight in scales:
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

        weights = [0.8, 1.2, 2.0, 1.2][:len(all_boxes)]
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=weights, iou_thr=0.5, skip_box_thr=0.005, conf_type='avg',
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
