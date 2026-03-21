"""NM i AI 2026 — Two-stage grocery product detection.

Stage 1: YOLOv8 class-agnostic detection (find all products)
Stage 2: DINOv2 embedding + k-NN classification (identify products)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import timm
from timm.data import resolve_data_config, create_transform
from PIL import Image
from ultralytics import YOLO


def load_detector(weights_path, device):
    """Load YOLOv8 detection model."""
    model = YOLO(str(weights_path))
    return model


def load_classifier(weights_path, device):
    """Load DINOv2-Small feature extractor."""
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=0,
    )
    state_dict = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device).half().eval()

    data_config = resolve_data_config(model=model)
    transform = create_transform(**data_config, is_training=False)
    return model, transform


def embed_crops(model, transform, crops, device, batch_size=64):
    """Compute DINOv2 embeddings for a list of PIL Image crops."""
    if not crops:
        return np.zeros((0, 384), dtype=np.float16)

    all_embeddings = []
    for i in range(0, len(crops), batch_size):
        batch_imgs = crops[i : i + batch_size]
        batch_tensors = torch.stack([transform(img) for img in batch_imgs])
        batch_tensors = batch_tensors.to(device).half()

        with torch.no_grad():
            embs = model(batch_tensors)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embeddings.append(embs.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def classify_crops(crop_embeddings, ref_embeddings, ref_category_ids, threshold=0.15):
    """Classify crops by cosine similarity to reference embeddings."""
    if len(crop_embeddings) == 0:
        return [], []

    # ref_embeddings: (N_ref, 384), crop_embeddings: (N_crops, 384)
    similarities = crop_embeddings @ ref_embeddings.T  # (N_crops, N_ref)
    best_idx = similarities.argmax(axis=1)
    best_scores = similarities[np.arange(len(best_idx)), best_idx]

    category_ids = []
    confidences = []
    for i, (idx, score) in enumerate(zip(best_idx, best_scores)):
        if score >= threshold:
            category_ids.append(int(ref_category_ids[idx]))
            confidences.append(float(score))
        else:
            category_ids.append(0)  # fallback to first category
            confidences.append(0.0)

    return category_ids, confidences


def run_tiled_detection(model, image_path, tile_size=640, overlap=0.2, device="cuda"):
    """Run detection on overlapping tiles for better small object detection."""
    img = Image.open(image_path)
    w, h = img.size

    stride = int(tile_size * (1 - overlap))
    all_boxes = []
    all_scores = []

    # Generate tile coordinates
    tiles = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            tiles.append((x1, y1, x2, y2))

    # Also run full image
    results_full = model(str(image_path), device=device, verbose=False,
                         imgsz=1280, conf=0.15, iou=0.5, max_det=500)

    for r in results_full:
        if r.boxes is not None and len(r.boxes):
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                all_boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
                all_scores.append(float(r.boxes.conf[i].item()))

    # Run on tiles
    img_np = np.array(img)
    for tx1, ty1, tx2, ty2 in tiles:
        tile = img_np[ty1:ty2, tx1:tx2]
        results = model(tile, device=device, verbose=False,
                       imgsz=640, conf=0.15, iou=0.5, max_det=300)
        for r in results:
            if r.boxes is not None and len(r.boxes):
                for i in range(len(r.boxes)):
                    bx1, by1, bx2, by2 = r.boxes.xyxy[i].tolist()
                    # Convert tile coords to image coords, then normalize
                    all_boxes.append([
                        (bx1 + tx1) / w, (by1 + ty1) / h,
                        (bx2 + tx1) / w, (by2 + ty1) / h
                    ])
                    all_scores.append(float(r.boxes.conf[i].item()))

    return all_boxes, all_scores, img


def merge_boxes_wbf(boxes, scores, iou_thr=0.5, skip_box_thr=0.01):
    """Merge overlapping boxes using Weighted Box Fusion."""
    from ensemble_boxes import weighted_boxes_fusion

    if not boxes:
        return np.zeros((0, 4)), np.zeros(0)

    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)
    labels_arr = np.zeros(len(boxes), dtype=np.float32)

    merged_boxes, merged_scores, _ = weighted_boxes_fusion(
        [boxes_arr], [scores_arr], [labels_arr],
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )

    return merged_boxes, merged_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = Path(__file__).parent

    # Load models
    detector = load_detector(script_dir / "yolo_best.pt", device)
    classifier, cls_transform = load_classifier(script_dir / "dinov2_vits14.pt", device)

    # Load reference embeddings and category mapping
    ref_data = torch.load(str(script_dir / "reference_embeddings.pt"), map_location="cpu")
    ref_embeddings = ref_data["embeddings"]  # (N, 384)
    ref_category_ids = ref_data["category_ids"]  # (N,)

    predictions = []
    input_dir = Path(args.input)

    for img_path in sorted(input_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        image_id = int(img_path.stem.split("_")[-1])

        # Stage 1: Tiled detection
        boxes_norm, scores, img = run_tiled_detection(
            detector, img_path, tile_size=640, overlap=0.2, device=device
        )

        # Merge overlapping boxes with WBF
        merged_boxes, merged_scores = merge_boxes_wbf(
            boxes_norm, scores, iou_thr=0.5, skip_box_thr=0.1
        )

        if len(merged_boxes) == 0:
            continue

        w, h = img.size

        # Stage 2: Crop and classify
        crops = []
        valid_indices = []
        for i, (box, score) in enumerate(zip(merged_boxes, merged_scores)):
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            # Ensure valid crop
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            crop = img.crop((x1, y1, x2, y2)).convert("RGB")
            crops.append(crop)
            valid_indices.append(i)

        # Embed crops and classify
        crop_embs = embed_crops(classifier, cls_transform, crops, device)
        cat_ids, cls_scores = classify_crops(
            crop_embs, ref_embeddings, ref_category_ids
        )

        # Build predictions
        for j, idx in enumerate(valid_indices):
            box = merged_boxes[idx]
            det_score = float(merged_scores[idx])

            # Convert normalized xyxy to COCO [x, y, w, h] pixels
            x1 = round(box[0] * w, 1)
            y1 = round(box[1] * h, 1)
            bw = round((box[2] - box[0]) * w, 1)
            bh = round((box[3] - box[1]) * h, 1)

            predictions.append({
                "image_id": image_id,
                "category_id": cat_ids[j],
                "bbox": [x1, y1, bw, bh],
                "score": round(det_score, 3),
            })

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions")


if __name__ == "__main__":
    main()
