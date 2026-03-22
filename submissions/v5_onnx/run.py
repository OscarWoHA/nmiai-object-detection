"""v5 ONNX FP16 — bigger YOLOv8l nc=356 model exported to ONNX half precision.
Uses onnxruntime with CUDAExecutionProvider for GPU inference.
Multi-scale self-ensemble at 960, 1280, 1536.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort


def preprocess(img_path, imgsz=1280):
    """Letterbox + normalize for YOLO ONNX input."""
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    # Compute scale
    scale = min(imgsz / orig_w, imgsz / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Letterbox (pad to imgsz x imgsz)
    pad_w = (imgsz - new_w) // 2
    pad_h = (imgsz - new_h) // 2
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    canvas.paste(img_resized, (pad_w, pad_h))

    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

    return arr, orig_w, orig_h, scale, pad_w, pad_h


def postprocess(outputs, orig_w, orig_h, scale, pad_w, pad_h, conf_thr=0.01):
    """Parse YOLO ONNX output (1, 360, 8400) -> list of (x1,y1,x2,y2,conf,cls)."""
    # Output shape: (1, nc+4, num_boxes) = (1, 360, 8400)
    output = outputs[0][0]  # (360, 8400)
    # Transpose to (8400, 360)
    output = output.T

    boxes_xywh = output[:, :4]
    class_probs = output[:, 4:]

    results = []
    for i in range(len(boxes_xywh)):
        cls_scores = class_probs[i]
        max_score = cls_scores.max()
        if max_score < conf_thr:
            continue

        cls_id = int(cls_scores.argmax())
        cx, cy, w, h = boxes_xywh[i]

        # Remove letterbox padding and scale
        x1 = (cx - w/2 - pad_w) / scale
        y1 = (cy - h/2 - pad_h) / scale
        x2 = (cx + w/2 - pad_w) / scale
        y2 = (cy + h/2 - pad_h) / scale

        # Clamp
        x1 = max(0, min(orig_w, x1))
        y1 = max(0, min(orig_h, y1))
        x2 = max(0, min(orig_w, x2))
        y2 = max(0, min(orig_h, y2))

        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        results.append((x1, y1, x2, y2, float(max_score), cls_id))

    return results


def nms(detections, iou_thr=0.5):
    """Simple NMS on detections list."""
    if not detections:
        return []
    dets = sorted(detections, key=lambda x: -x[4])
    keep = []
    for d in dets:
        overlap = False
        for k in keep:
            # IoU
            xa = max(d[0], k[0]); ya = max(d[1], k[1])
            xb = min(d[2], k[2]); yb = min(d[3], k[3])
            inter = max(0, xb-xa) * max(0, yb-ya)
            a1 = (d[2]-d[0])*(d[3]-d[1])
            a2 = (k[2]-k[0])*(k[3]-k[1])
            iou = inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0
            if iou > iou_thr:
                overlap = True
                break
        if not overlap:
            keep.append(d)
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    session = ort.InferenceSession(
        str(script_dir / "model.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    predictions = []
    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])

        all_dets = []
        for imgsz in [1280]:
            inp, ow, oh, sc, pw, ph = preprocess(img_path, imgsz)
            outputs = session.run(None, {input_name: inp})
            dets = postprocess(outputs, ow, oh, sc, pw, ph, conf_thr=0.01)
            all_dets.extend(dets)

        # NMS
        final = nms(all_dets, iou_thr=0.5)

        for x1, y1, x2, y2, conf, cls_id in final:
            predictions.append({
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": [round(x1,1), round(y1,1), round(x2-x1,1), round(y2-y1,1)],
                "score": round(conf, 4),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"Wrote {len(predictions)} predictions")


if __name__ == "__main__":
    main()
