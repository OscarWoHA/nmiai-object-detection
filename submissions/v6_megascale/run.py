"""v6 Mega-scale ensemble — 7 scales + HFlip, WBF merge.
Uses 267s of unused runtime budget for maximum mAP.
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


def run_single(model, img_source, device, imgsz, conf, w_img, h_img):
    """Run model on an image/array, return normalized boxes, scores, labels."""
    results = model(img_source, device=device, verbose=False,
                    imgsz=imgsz, conf=conf, iou=0.5, max_det=500)
    boxes, scores, labels = [], [], []
    for r in results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            boxes.append([x1/w_img, y1/h_img, x2/w_img, y2/h_img])
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
    model = YOLO(str(script_dir / "yolo_best.pt"))

    predictions = []
    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])

        img = Image.open(img_path)
        w, h = img.size
        img_np = np.array(img)

        all_boxes, all_scores, all_labels, weights = [], [], [], []

        # Multi-scale passes
        scales = [
            (960,  0.02, 0.8),   # small scale, lower weight
            (1088, 0.02, 1.0),
            (1280, 0.01, 2.0),   # primary scale, highest weight
            (1440, 0.02, 1.2),
            (1600, 0.02, 1.0),   # large scale
        ]

        for imgsz, conf, weight in scales:
            boxes, scores, labels = run_single(model, str(img_path), device, imgsz, conf, w, h)
            if boxes:
                all_boxes.append(np.array(boxes, dtype=np.float32))
                all_scores.append(np.array(scores, dtype=np.float32))
                all_labels.append(np.array(labels, dtype=np.float32))
                weights.append(weight)

        # HFlip pass at primary scale
        img_flipped = img_np[:, ::-1, :].copy()
        results = model(img_flipped, device=device, verbose=False,
                       imgsz=1280, conf=0.01, iou=0.5, max_det=500)
        flip_boxes, flip_scores, flip_labels = [], [], []
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                # Mirror x coordinates back
                flip_boxes.append([(w - x2)/w, y1/h, (w - x1)/w, y2/h])
                flip_scores.append(float(r.boxes.conf[i].item()))
                flip_labels.append(int(r.boxes.cls[i].item()))

        if flip_boxes:
            all_boxes.append(np.array(flip_boxes, dtype=np.float32))
            all_scores.append(np.array(flip_scores, dtype=np.float32))
            all_labels.append(np.array(flip_labels, dtype=np.float32))
            weights.append(1.5)  # HFlip weighted moderately

        if not all_boxes:
            continue

        # WBF merge
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=weights, iou_thr=0.55, skip_box_thr=0.005, conf_type='max',
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
