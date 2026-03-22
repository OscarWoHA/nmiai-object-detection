"""v8 hybrid: YOLOv8l nc=356 detection + mobilenetv3 crop reclassification.

Strategy: Use YOLO at 3 scales for detection (proven 0.966 det mAP).
For each detection, if YOLO confidence is low (<0.6), crop the product
and reclassify with the mobilenetv3 classifier trained on GT crops.
Blend YOLO and classifier predictions weighted by confidence.

This targets the classification bottleneck (0.861) without touching detection.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image
import timm
from timm.data import resolve_data_config, create_transform


def load_crop_classifier(weights_path, num_classes, device):
    """Load mobilenetv3 crop classifier."""
    model = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=num_classes)
    state_dict = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    data_config = resolve_data_config(model=model)
    transform = create_transform(**data_config, is_training=False)
    return model, transform


def classify_crops(classifier, transform, crops, device, batch_size=128):
    """Run crop classifier, return (class_ids, confidences) for each crop."""
    if not crops:
        return [], []

    all_cls_ids = []
    all_confs = []

    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size]
        tensors = torch.stack([transform(c) for c in batch]).to(device)
        with torch.no_grad():
            logits = classifier(tensors)
            probs = F.softmax(logits, dim=1)
            confs, cls_ids = probs.max(dim=1)
            all_cls_ids.extend(cls_ids.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())

    return all_cls_ids, all_confs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = Path(__file__).parent

    # Load YOLO detector
    yolo = YOLO(str(script_dir / "yolo_best.pt"))

    # Load crop classifier
    classifier, cls_transform = load_crop_classifier(
        script_dir / "crop_classifier.pt", num_classes=356, device=device
    )

    predictions = []
    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])

        img = Image.open(img_path)
        w, h = img.size

        # Multi-scale detection (proven best approach)
        all_boxes, all_scores, all_labels = [], [], []
        for imgsz, conf, weight in [(1280, 0.01, 2.0), (960, 0.02, 1.0), (1536, 0.02, 1.5)]:
            results = yolo(str(img_path), device=device, verbose=False,
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

        # Identify which boxes need reclassification
        # Low-confidence boxes benefit most from crop classifier
        reclass_threshold = 0.6
        crops_to_classify = []
        reclass_indices = []

        for i in range(len(merged_boxes)):
            box = merged_boxes[i]
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            # Add 10% padding for context
            bw, bh = x2 - x1, y2 - y1
            pad = 0.10
            x1 = max(0, int(x1 - bw * pad))
            y1 = max(0, int(y1 - bh * pad))
            x2 = min(w, int(x2 + bw * pad))
            y2 = min(h, int(y2 + bh * pad))

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            crop = img.crop((x1, y1, x2, y2)).convert("RGB")
            # Pad to square
            cw, ch = crop.size
            if cw != ch:
                side = max(cw, ch)
                padded = Image.new("RGB", (side, side), (128, 128, 128))
                padded.paste(crop, ((side - cw) // 2, (side - ch) // 2))
                crop = padded

            if merged_scores[i] < reclass_threshold:
                crops_to_classify.append(crop)
                reclass_indices.append(i)

        # Run crop classifier on ambiguous detections
        if crops_to_classify:
            cls_ids, cls_confs = classify_crops(classifier, cls_transform, crops_to_classify, device)

            for j, idx in enumerate(reclass_indices):
                yolo_conf = merged_scores[idx]
                crop_conf = cls_confs[j]
                crop_cls = cls_ids[j]
                yolo_cls = int(merged_labels[idx])

                # Blend: if crop classifier is more confident, use its prediction
                # Otherwise keep YOLO's prediction
                if crop_conf > yolo_conf * 1.2:
                    merged_labels[idx] = crop_cls
                elif crop_conf > 0.5 and yolo_conf < 0.4:
                    merged_labels[idx] = crop_cls

        # Build predictions
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
