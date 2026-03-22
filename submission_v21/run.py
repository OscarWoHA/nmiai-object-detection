"""v21: Stage-2 original + stage-2 copypaste with sweep-optimized WBF.

Sweep found: model_b weighted 2x, iou_thr=0.45, conf_type=max
Model A: original YOLOv8l stage-2 fine-tuned at 1536 (.pt, multi-scale)
Model B: copy-paste YOLOv8l stage-2 fine-tuned at 1536 (.pt)
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
                    imgsz=imgsz, conf=conf, iou=0.5, max_det=500)
    boxes, scores, labels = [], [], []
    for r in results:
        if r.boxes is None: continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
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
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"): continue
        image_id = int(img_path.stem.split("_")[-1])
        img = Image.open(img_path)
        w, h = img.size
        all_boxes, all_scores, all_labels, weights = [], [], [], []

        # Model A at 3 scales (sweep-optimized: weight=1.0)
        for imgsz, conf, wt in [(960, 0.02, 1.0), (1280, 0.01, 2.0), (1440, 0.01, 1.5)]:
            boxes, scores, labels = run_model(model_a, img_path, device, w, h, imgsz, conf)
            if boxes:
                all_boxes.append(np.array(boxes, dtype=np.float32))
                all_scores.append(np.array(scores, dtype=np.float32))
                all_labels.append(np.array(labels, dtype=np.float32))
                weights.append(wt * 1.0)  # sweep: a_wt=1.0

        # Model B at 3 scales (sweep-optimized: weight=2.0)
        for imgsz, conf, wt in [(960, 0.02, 1.0), (1280, 0.01, 2.0), (1440, 0.01, 1.5)]:
            boxes, scores, labels = run_model(model_b, img_path, device, w, h, imgsz, conf)
            if boxes:
                all_boxes.append(np.array(boxes, dtype=np.float32))
                all_scores.append(np.array(scores, dtype=np.float32))
                all_labels.append(np.array(labels, dtype=np.float32))
                weights.append(wt * 2.0)  # sweep: b_wt=2.0

        if not all_boxes: continue
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=weights, iou_thr=0.45, skip_box_thr=0.005, conf_type='max',
        )
        for i in range(len(merged_boxes)):
            box = merged_boxes[i]
            predictions.append({
                "image_id": image_id,
                "category_id": int(merged_labels[i]),
                "bbox": [round(box[0]*w,1), round(box[1]*h,1),
                         round((box[2]-box[0])*w,1), round((box[3]-box[1])*h,1)],
                "score": float(merged_scores[i]),
            })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f: json.dump(predictions, f)
    print(f"Wrote {len(predictions)} predictions")

if __name__ == "__main__": main()
