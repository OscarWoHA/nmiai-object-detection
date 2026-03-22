"""NM i AI 2026 — YOLOv8m nc=356 with TTA and aggressive detection."""
import argparse
import json
from pathlib import Path

import torch
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = Path(__file__).parent

    # Load multi-class model (handles both detection + classification)
    model = YOLO(str(script_dir / "yolo_best.pt"))

    predictions = []

    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        image_id = int(img_path.stem.split("_")[-1])

        # Run with TTA (augment=True) for better accuracy
        results = model(
            str(img_path),
            device=device,
            verbose=False,
            imgsz=1280,
            conf=0.05,       # very low conf — catch everything, let score sort it
            iou=0.5,
            max_det=500,
            augment=True,    # TTA: flips + scales
        )

        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(r.boxes.cls[i].item()),
                    "bbox": [
                        round(x1, 1),
                        round(y1, 1),
                        round(x2 - x1, 1),
                        round(y2 - y1, 1),
                    ],
                    "score": round(float(r.boxes.conf[i].item()), 4),
                })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions for {len(set(p['image_id'] for p in predictions))} images")


if __name__ == "__main__":
    main()
