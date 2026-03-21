# NM i AI 2026 - NorgesGruppen Object Detection

## Competition Overview
Norwegian Championship for AI. Task: detect and classify grocery products on store shelves.
**Deadline**: 3 days from 2026-03-19. **Max 3 submissions/day**.

## Scoring
`Score = 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5`
- Detection: did you find products? (IoU >= 0.5, category ignored) — 70%
- Classification: did you identify the right product? (IoU >= 0.5 AND correct category_id) — 30%
- Detection-only (all category_id=0) caps at 70%.

## Dataset
- **Shelf images**: 248 images (2000x1500), ~22,700 COCO-format annotations, 356 product categories (IDs 0-355, plus 356=unknown_product)
- **Avg ~92 annotations per image** (very crowded shelves)
- **Reference images**: 327 products with multi-angle photos (main, front, back, left, right, top, bottom) organized by barcode in `images_multiangle/`
- **Category imbalance**: most common has 422 annotations, many categories have only 1-5 examples
- Annotations: `annotations.json` in COCO format, bbox is [x, y, width, height]

## Sandbox Environment
- Python 3.11, NVIDIA L4 GPU (24GB VRAM), 4 vCPU, 8GB RAM
- **Timeout**: 300 seconds, **No network access**
- Pre-installed: ultralytics==8.1.0, torch==2.6.0+cu124, torchvision==0.21.0+cu124, onnxruntime-gpu==1.20.0, timm==0.9.12, supervision==0.18.0, ensemble-boxes==1.0.9
- **CRITICAL**: Pin ultralytics==8.1.0 for .pt weights compatibility

## Submission Format
- ZIP with `run.py` at root + model weights
- **Max 420MB** (uncompressed), max 3 weight files
- `run.py --input /data/images --output /output/predictions.json`
- Output: JSON array of `{image_id, category_id, bbox: [x,y,w,h], score}`
- image_id extracted from filename: img_00042.jpg -> 42
- **Security**: no `import os`, `subprocess`, `socket`, `eval`, `exec`. Use `pathlib`.

## Key Constraints
- Must use `pathlib` not `os` for file ops
- YOLOv8 with ultralytics==8.1.0 is the safest path (pre-installed in sandbox)
- ONNX export with opset<=20 is universal fallback
- FP16 recommended for L4 GPU (smaller + faster)
- Process images one at a time to stay within 8GB memory

## Project Structure
```
object-detection/
  shelf_images_with_coco_annotations/
    annotations.json          # COCO format annotations
    images/                   # 248 shelf images (img_XXXXX.jpg)
  images_multiangle/
    metadata.json             # Product info + annotation counts
    {barcode}/                # 327 product folders
      main.jpg, front.jpg, back.jpg, left.jpg, right.jpg, ...
```

## Development Notes
- Local Python is 3.14 — use a venv with Python 3.11 for training
- Always validate submission zip structure before uploading
- Test locally before submitting (only 3 attempts/day)
