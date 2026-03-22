"""v8 hybrid ALL: YOLO detects, mobilenetv3 classifies EVERY box.

Instead of only reclassifying low-confidence boxes, run the crop classifier
on ALL detections and blend YOLO + classifier predictions.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = Path(__file__).parent

    yolo = YOLO(str(script_dir / "yolo_best.pt"))

    # Load crop classifier
    classifier = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=356)
    classifier.load_state_dict(torch.load(str(script_dir / "crop_classifier.pt"), map_location="cpu"))
    classifier = classifier.to(device).eval()
    cls_config = resolve_data_config(model=classifier)
    cls_transform = create_transform(**cls_config, is_training=False)

    predictions = []
    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])

        img = Image.open(img_path)
        w, h = img.size

        # Multi-scale YOLO detection
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

        weights_list = [2.0, 1.0, 1.5][:len(all_boxes)]
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=weights_list, iou_thr=0.55, skip_box_thr=0.01, conf_type='max',
        )

        # Crop ALL detections for classifier
        crops = []
        valid_indices = []
        for i in range(len(merged_boxes)):
            box = merged_boxes[i]
            bw = (box[2] - box[0]) * w
            bh = (box[3] - box[1]) * h
            pad = 0.10
            x1 = max(0, int(box[0] * w - bw * pad))
            y1 = max(0, int(box[1] * h - bh * pad))
            x2 = min(w, int(box[2] * w + bw * pad))
            y2 = min(h, int(box[3] * h + bh * pad))
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            crop = img.crop((x1, y1, x2, y2)).convert("RGB")
            cw, ch = crop.size
            if cw != ch:
                side = max(cw, ch)
                padded = Image.new("RGB", (side, side), (128, 128, 128))
                padded.paste(crop, ((side - cw) // 2, (side - ch) // 2))
                crop = padded
            crops.append(crop)
            valid_indices.append(i)

        # Batch classify all crops
        if crops:
            all_cls_ids = []
            all_cls_confs = []
            for batch_start in range(0, len(crops), 128):
                batch = crops[batch_start:batch_start + 128]
                tensors = torch.stack([cls_transform(c) for c in batch]).to(device)
                with torch.no_grad():
                    logits = classifier(tensors)
                    probs = F.softmax(logits, dim=1)
                    confs, cls_ids = probs.max(dim=1)
                    all_cls_ids.extend(cls_ids.cpu().tolist())
                    all_cls_confs.extend(confs.cpu().tolist())

            # Blend YOLO and classifier predictions
            for j, idx in enumerate(valid_indices):
                yolo_cls = int(merged_labels[idx])
                yolo_conf = float(merged_scores[idx])
                crop_cls = all_cls_ids[j]
                crop_conf = all_cls_confs[j]

                # Weighted blend:
                # - High YOLO conf (>0.7): trust YOLO (it saw context)
                # - Low YOLO conf (<0.4): trust classifier
                # - Middle: blend
                if yolo_conf > 0.7:
                    final_cls = yolo_cls
                elif yolo_conf < 0.4:
                    final_cls = crop_cls if crop_conf > 0.3 else yolo_cls
                else:
                    # Both have moderate confidence — use classifier if more confident
                    final_cls = crop_cls if crop_conf > yolo_conf else yolo_cls

                merged_labels[idx] = final_cls

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
