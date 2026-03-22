"""NM i AI 2026 — v3: YOLOv8m nc=356 with tiled inference + WBF merging.

Improvements over v2:
- Tiled inference at 640px to catch small products
- Full image at 1280px for context
- WBF to merge overlapping predictions
- TTA on full image pass
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


def run_full_image(model, img_path, device):
    """Run detection on full image at 1280 with TTA."""
    results = model(
        str(img_path),
        device=device,
        verbose=False,
        imgsz=1280,
        conf=0.1,
        iou=0.5,
        max_det=500,
        augment=True,
    )

    boxes, scores, labels = [], [], []
    img = Image.open(img_path)
    w, h = img.size

    for r in results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
            scores.append(float(r.boxes.conf[i].item()))
            labels.append(int(r.boxes.cls[i].item()))

    return boxes, scores, labels, w, h


def run_tiles(model, img_path, device, tile_size=960, overlap=0.25):
    """Run detection on overlapping tiles for small object detection."""
    img = Image.open(img_path)
    w, h = img.size
    img_np = np.array(img)

    stride = int(tile_size * (1 - overlap))
    boxes, scores, labels = [], [], []

    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1 = min(x0, w - tile_size) if x0 + tile_size > w else x0
            y1 = min(y0, h - tile_size) if y0 + tile_size > h else y0
            x2 = x1 + tile_size
            y2 = y1 + tile_size

            # Clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            tile = img_np[y1:y2, x1:x2]
            if tile.shape[0] < 100 or tile.shape[1] < 100:
                continue

            results = model(
                tile,
                device=device,
                verbose=False,
                imgsz=640,
                conf=0.2,
                iou=0.5,
                max_det=300,
            )

            for r in results:
                if r.boxes is None:
                    continue
                for i in range(len(r.boxes)):
                    bx1, by1, bx2, by2 = r.boxes.xyxy[i].tolist()
                    # Convert tile coords to image coords, normalize
                    boxes.append([
                        (bx1 + x1) / w,
                        (by1 + y1) / h,
                        (bx2 + x1) / w,
                        (by2 + y1) / h,
                    ])
                    scores.append(float(r.boxes.conf[i].item()))
                    labels.append(int(r.boxes.cls[i].item()))

    return boxes, scores, labels


def merge_predictions(all_boxes, all_scores, all_labels, iou_thr=0.55):
    """Merge predictions from multiple passes using WBF."""
    if not any(len(b) > 0 for b in all_boxes):
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    # WBF expects list of arrays per model
    boxes_list = []
    scores_list = []
    labels_list = []

    for boxes, scores, labels in zip(all_boxes, all_scores, all_labels):
        if len(boxes) > 0:
            boxes_list.append(np.array(boxes, dtype=np.float32))
            scores_list.append(np.array(scores, dtype=np.float32))
            labels_list.append(np.array(labels, dtype=np.float32))
        else:
            boxes_list.append(np.zeros((0, 4), dtype=np.float32))
            scores_list.append(np.zeros(0, dtype=np.float32))
            labels_list.append(np.zeros(0, dtype=np.float32))

    # WBF with higher weight for full-image pass (index 0)
    weights = [2, 1]  # full image weighted more than tiles

    merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=0.05,
    )

    return merged_boxes, merged_scores, merged_labels.astype(int)


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

        # Pass 1: Full image with TTA
        full_boxes, full_scores, full_labels, w, h = run_full_image(
            model, img_path, device
        )

        # Pass 2: Tiled inference for small objects
        tile_boxes, tile_scores, tile_labels = run_tiles(
            model, img_path, device, tile_size=960, overlap=0.25
        )

        # Merge with WBF
        merged_boxes, merged_scores, merged_labels = merge_predictions(
            [full_boxes, tile_boxes],
            [full_scores, tile_scores],
            [full_labels, tile_labels],
            iou_thr=0.55,
        )

        # Convert to output format
        for i in range(len(merged_boxes)):
            box = merged_boxes[i]
            x1 = box[0] * w
            y1 = box[1] * h
            x2 = box[2] * w
            y2 = box[3] * h

            predictions.append({
                "image_id": image_id,
                "category_id": int(merged_labels[i]),
                "bbox": [
                    round(x1, 1),
                    round(y1, 1),
                    round(x2 - x1, 1),
                    round(y2 - y1, 1),
                ],
                "score": round(float(merged_scores[i]), 4),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions for {len(set(p['image_id'] for p in predictions))} images")


if __name__ == "__main__":
    main()
