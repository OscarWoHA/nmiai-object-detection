"""v7 two-model ensemble: YOLOv8l nc=356 (.pt) + YOLOv8l nc=356 ONNX FP16.

Two differently-trained models, WBF merge. Different training runs
produce different errors — ensemble reduces both.
Total: 85MB + 169MB = 254MB (under 420MB limit).
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
import onnxruntime as ort


def run_yolo(model, img_path, device, w, h, imgsz=1280, conf=0.01):
    """Run YOLO .pt model."""
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


def run_onnx(session, img_path, w, h, imgsz=1280, conf=0.01):
    """Run ONNX model with manual pre/post processing."""
    img = Image.open(img_path).convert("RGB")

    # Letterbox resize
    scale = min(imgsz / w, imgsz / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    pad_w = (imgsz - new_w) // 2
    pad_h = (imgsz - new_h) // 2
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    canvas.paste(img_resized, (pad_w, pad_h))

    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: arr})

    # Parse output: (1, nc+4, num_boxes) -> transpose to (num_boxes, nc+4)
    output = outputs[0][0].T

    boxes, scores, labels = [], [], []
    for i in range(len(output)):
        cls_probs = output[i, 4:]
        max_prob = cls_probs.max()
        if max_prob < conf:
            continue

        cls_id = int(cls_probs.argmax())
        cx, cy, bw, bh = output[i, :4]

        x1 = (cx - bw/2 - pad_w) / scale
        y1 = (cy - bh/2 - pad_h) / scale
        x2 = (cx + bw/2 - pad_w) / scale
        y2 = (cy + bh/2 - pad_h) / scale

        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        boxes.append([x1/w, y1/h, x2/w, y2/h])
        scores.append(float(max_prob))
        labels.append(cls_id)

    return boxes, scores, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = Path(__file__).parent

    # Load both models
    yolo_model = YOLO(str(script_dir / "yolo_best.pt"))
    onnx_session = ort.InferenceSession(
        str(script_dir / "model.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    predictions = []
    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])

        img = Image.open(img_path)
        w, h = img.size

        all_boxes, all_scores, all_labels = [], [], []

        # Model 1: YOLO .pt at 1280
        b1, s1, l1 = run_yolo(yolo_model, img_path, device, w, h, imgsz=1280, conf=0.01)
        if b1:
            all_boxes.append(np.array(b1, dtype=np.float32))
            all_scores.append(np.array(s1, dtype=np.float32))
            all_labels.append(np.array(l1, dtype=np.float32))

        # Model 2: ONNX at 1280
        b2, s2, l2 = run_onnx(onnx_session, img_path, w, h, imgsz=1280, conf=0.01)
        if b2:
            all_boxes.append(np.array(b2, dtype=np.float32))
            all_scores.append(np.array(s2, dtype=np.float32))
            all_labels.append(np.array(l2, dtype=np.float32))

        if not all_boxes:
            continue

        # WBF merge with equal weights
        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=[2.0, 1.5], iou_thr=0.55, skip_box_thr=0.01, conf_type='max',
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
