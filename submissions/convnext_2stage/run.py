"""Two-stage: YOLOv8 detects (class-agnostic boxes) + ConvNeXt classifies crops.

Stage 1: YOLOv8l at multiple scales → boxes via WBF
Stage 2: Crop each detection from ORIGINAL high-res image → ConvNeXt → class

The ConvNeXt sees 224x224 crops of individual products at full resolution,
so it can read text, logos, and fine details that YOLO misses at 1280.
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


def run_detector(model, img_path, device, w, h, imgsz=1280, conf=0.01):
    results = model(str(img_path), device=device, verbose=False,
                    imgsz=imgsz, conf=conf, iou=0.5, max_det=500)
    boxes, scores = [], []
    for r in results:
        if r.boxes is None: continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            boxes.append([x1/w, y1/h, x2/w, y2/h])
            scores.append(float(r.boxes.conf[i].item()))
    return boxes, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = Path(__file__).parent

    # Stage 1: YOLO detector (uses class predictions too as prior)
    detector = YOLO(str(script_dir / "detector.pt"))

    # Stage 2: ConvNeXt classifier
    classifier = timm.create_model("convnext_small.fb_in22k_ft_in1k", pretrained=False, num_classes=356)
    cls_sd = torch.load(str(script_dir / "classifier.pt"), map_location="cpu")
    classifier.load_state_dict(cls_sd)
    classifier = classifier.to(device).eval()
    cls_config = resolve_data_config(model=classifier)
    cls_transform = create_transform(**cls_config, is_training=False)

    predictions = []
    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"): continue
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(img_path)
        w, h = img.size

        # Multi-scale detection
        all_boxes, all_scores, all_labels, weights = [], [], [], []
        for imgsz, conf, wt in [(960, 0.02, 1.0), (1280, 0.01, 2.0), (1440, 0.01, 1.5)]:
            results = detector(str(img_path), device=device, verbose=False,
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
                weights.append(wt)

        if not all_boxes: continue
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=weights, iou_thr=0.5, skip_box_thr=0.005, conf_type='avg',
        )

        # Stage 2: Classify each crop with ConvNeXt
        crops = []
        valid_indices = []
        for i in range(len(merged_boxes)):
            box = merged_boxes[i]
            bw, bh = (box[2]-box[0])*w, (box[3]-box[1])*h
            pad = 0.12
            x1 = max(0, int(box[0]*w - bw*pad))
            y1 = max(0, int(box[1]*h - bh*pad))
            x2 = min(w, int(box[2]*w + bw*pad))
            y2 = min(h, int(box[3]*h + bh*pad))
            if x2-x1 < 10 or y2-y1 < 10: continue
            crop = img.crop((x1,y1,x2,y2)).convert("RGB")
            cw, ch = crop.size
            if cw != ch:
                side = max(cw, ch)
                padded = Image.new("RGB", (side, side), (128, 128, 128))
                padded.paste(crop, ((side-cw)//2, (side-ch)//2))
                crop = padded
            crops.append(crop)
            valid_indices.append(i)

        # Batch classify
        if crops:
            yolo_labels = [int(merged_labels[i]) for i in valid_indices]
            yolo_confs = [float(merged_scores[i]) for i in valid_indices]

            for batch_start in range(0, len(crops), 64):
                batch = crops[batch_start:batch_start+64]
                tensors = torch.stack([cls_transform(c) for c in batch]).to(device)
                with torch.no_grad():
                    logits = classifier(tensors)
                    probs = F.softmax(logits, dim=1)
                    cls_confs, cls_ids = probs.max(dim=1)

                for j in range(len(batch)):
                    idx = valid_indices[batch_start + j]
                    yolo_cls = int(merged_labels[idx])
                    yolo_conf = float(merged_scores[idx])
                    conv_cls = cls_ids[j].item()
                    conv_conf = cls_confs[j].item()

                    # Blend: trust ConvNeXt for classification, YOLO for detection
                    # If ConvNeXt is confident, use its class
                    # If not, fall back to YOLO's class
                    if conv_conf > 0.5:
                        final_cls = conv_cls
                    elif conv_conf > yolo_conf * 0.8:
                        final_cls = conv_cls
                    else:
                        final_cls = yolo_cls

                    box = merged_boxes[idx]
                    predictions.append({
                        "image_id": image_id,
                        "category_id": final_cls,
                        "bbox": [round(box[0]*w,1), round(box[1]*h,1),
                                 round((box[2]-box[0])*w,1), round((box[3]-box[1])*h,1)],
                        "score": float(merged_scores[idx]),
                    })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f: json.dump(predictions, f)
    print(f"Wrote {len(predictions)} predictions")

if __name__ == "__main__": main()
