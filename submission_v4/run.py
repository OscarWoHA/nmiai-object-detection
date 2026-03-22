"""NM i AI 2026 — v4: Two-stage detection + classification.

Stage 1: YOLOv8 nc=1 detector (full image + tiles + WBF)
Stage 2: DINOv2 classification (per-angle + shelf prototypes, score calibration)

Key improvements over v2 (0.7413):
- nc=1 detector has higher recall than nc=356
- Per-angle prototypes (not averaged) for better matching
- Shelf GT crop prototypes for domain adaptation
- Score calibration: final_score = f(det_conf, cls_conf)
- Tiled inference for small products
- Box expansion for better IoU
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
import timm
from timm.data import resolve_data_config, create_transform


def load_detector(weights_path, device):
    return YOLO(str(weights_path))


def load_classifier(weights_path, device):
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    state_dict = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).half().eval()
    data_config = resolve_data_config(model=model)
    transform = create_transform(**data_config, is_training=False)
    return model, transform


def embed_crops(model, transform, crops, device, batch_size=64):
    if not crops:
        return np.zeros((0, 384), dtype=np.float32)
    all_embs = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size]
        tensors = torch.stack([transform(img) for img in batch]).to(device).half()
        with torch.no_grad():
            embs = model(tensors)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0)


def classify_crops(crop_embs, ref_embs, ref_cats):
    """Per-angle + top-k class aggregation (not single nearest)."""
    if len(crop_embs) == 0:
        return [], []

    # Cosine similarity: (n_crops, n_refs)
    sims = crop_embs @ ref_embs.T

    unique_cats = sorted(set(ref_cats.tolist()))
    cat_to_idx = {}
    for i, c in enumerate(ref_cats):
        cat_to_idx.setdefault(int(c), []).append(i)

    category_ids = []
    cls_scores = []

    for i in range(len(crop_embs)):
        row = sims[i]
        best_cat = -1
        best_score = -1
        second_score = -1

        for cat, indices in cat_to_idx.items():
            cat_sims = row[indices]
            # Score = 0.7 * max + 0.3 * mean(top3) per GPT-5.4
            top_k = np.sort(cat_sims)[::-1][:3]
            score = 0.7 * top_k[0] + 0.3 * np.mean(top_k)

            if score > best_score:
                second_score = best_score
                best_score = score
                best_cat = cat
            elif score > second_score:
                second_score = score

        category_ids.append(best_cat)
        # Margin-aware confidence (handle negative cosine sims)
        margin = best_score - second_score if second_score > -1 else 0.0
        cls_scores.append((best_score, max(0, margin)))

    return category_ids, cls_scores


def run_detection(model, img_path, device):
    """Full image + tiled inference, merged with WBF."""
    img = Image.open(img_path)
    w, h = img.size
    img_np = np.array(img)

    all_boxes_lists = []
    all_scores_lists = []
    all_labels_lists = []

    # Pass A: Full image with TTA
    results = model(str(img_path), device=device, verbose=False,
                    imgsz=1280, conf=0.01, iou=0.6, max_det=800, augment=True)
    boxes_a, scores_a, labels_a = [], [], []
    for r in results:
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            boxes_a.append([x1 / w, y1 / h, x2 / w, y2 / h])
            scores_a.append(float(r.boxes.conf[i].item()))
            labels_a.append(0)

    # Pass B: Tiles
    tile_size = 1024
    overlap = 0.20
    stride = int(tile_size * (1 - overlap))
    boxes_b, scores_b, labels_b = [], [], []

    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1t = min(x0, max(0, w - tile_size))
            y1t = min(y0, max(0, h - tile_size))
            x2t = min(x1t + tile_size, w)
            y2t = min(y1t + tile_size, h)

            tile = img_np[y1t:y2t, x1t:x2t]
            if tile.shape[0] < 100 or tile.shape[1] < 100:
                continue

            results = model(tile, device=device, verbose=False,
                           imgsz=1024, conf=0.10, iou=0.55, max_det=300)
            for r in results:
                if r.boxes is None:
                    continue
                for i in range(len(r.boxes)):
                    bx1, by1, bx2, by2 = r.boxes.xyxy[i].tolist()
                    # Drop detections near tile borders (hallucinations)
                    cx = (bx1 + bx2) / 2
                    cy = (by1 + by2) / 2
                    border_margin = 24
                    if (cx < border_margin and x1t > 0) or \
                       (cx > (x2t - x1t) - border_margin and x2t < w) or \
                       (cy < border_margin and y1t > 0) or \
                       (cy > (y2t - y1t) - border_margin and y2t < h):
                        continue

                    boxes_b.append([
                        (bx1 + x1t) / w, (by1 + y1t) / h,
                        (bx2 + x1t) / w, (by2 + y1t) / h,
                    ])
                    scores_b.append(float(r.boxes.conf[i].item()))
                    labels_b.append(0)

    # WBF merge
    boxes_list = []
    scores_list = []
    labels_list = []
    weights = []

    if boxes_a:
        boxes_list.append(np.array(boxes_a, dtype=np.float32))
        scores_list.append(np.array(scores_a, dtype=np.float32))
        labels_list.append(np.zeros(len(boxes_a), dtype=np.float32))
        weights.append(2)

    if boxes_b:
        boxes_list.append(np.array(boxes_b, dtype=np.float32))
        scores_list.append(np.array(scores_b, dtype=np.float32))
        labels_list.append(np.zeros(len(boxes_b), dtype=np.float32))
        weights.append(1)

    if not boxes_list:
        return np.zeros((0, 4)), np.zeros(0), img

    merged_boxes, merged_scores, _ = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        weights=weights, iou_thr=0.55, skip_box_thr=0.03,
        conf_type='max',  # don't penalize good full-image dets when tiles miss
    )

    # Return TIGHT boxes for submission (expansion only for classifier crops later)
    return merged_boxes, merged_scores, img


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

    # Load prototype bank
    bank = torch.load(str(script_dir / "prototype_bank.pt"), map_location="cpu")
    ref_embs = bank["embeddings"].astype(np.float32)
    ref_cats = bank["category_ids"].astype(np.int64)
    # Normalize (FP16 might have denormed)
    norms = np.linalg.norm(ref_embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    ref_embs = ref_embs / norms

    predictions = []

    for img_path in sorted(Path(args.input).iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        image_id = int(img_path.stem.split("_")[-1])

        # Stage 1: Detection
        boxes_norm, det_scores, img = run_detection(detector, img_path, device)
        if len(boxes_norm) == 0:
            continue

        w, h = img.size

        # Stage 2: Crop and classify
        crops = []
        valid_indices = []
        for i in range(len(boxes_norm)):
            box = boxes_norm[i]
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

            # Add 8% padding for classifier
            bw = x2 - x1
            bh = y2 - y1
            pad = 0.08
            x1 = max(0, int(x1 - bw * pad))
            y1 = max(0, int(y1 - bh * pad))
            x2 = min(w, int(x2 + bw * pad))
            y2 = min(h, int(y2 + bh * pad))

            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            crop = img.crop((x1, y1, x2, y2)).convert("RGB")
            # Pad to square to avoid CenterCrop destroying tall/narrow products
            cw, ch = crop.size
            if cw != ch:
                side = max(cw, ch)
                padded = Image.new("RGB", (side, side), (128, 128, 128))
                padded.paste(crop, ((side - cw) // 2, (side - ch) // 2))
                crop = padded
            crops.append(crop)
            valid_indices.append(i)

        # Embed and classify
        crop_embs = embed_crops(classifier, cls_transform, crops, device)
        cat_ids, cls_scores = classify_crops(crop_embs, ref_embs, ref_cats)

        # Build predictions with calibrated scores
        for j, idx in enumerate(valid_indices):
            box = boxes_norm[idx]
            det_conf = float(det_scores[idx])
            cls_conf, margin = cls_scores[j]

            # Score calibration: rank by P(det) * (0.7 + 0.3 * P(cls|det))
            p_cls = max(0, min(1, (cls_conf - 0.1) / 0.6))  # normalize cosine sim to ~[0,1]
            if margin < 0.03:
                p_cls *= 0.5  # low margin = uncertain classification
            final_score = det_conf * (0.7 + 0.3 * p_cls)

            # Convert to COCO [x, y, w, h] pixels
            x1 = round(box[0] * w, 1)
            y1 = round(box[1] * h, 1)
            bw = round((box[2] - box[0]) * w, 1)
            bh = round((box[3] - box[1]) * h, 1)

            predictions.append({
                "image_id": image_id,
                "category_id": cat_ids[j],
                "bbox": [x1, y1, bw, bh],
                "score": round(final_score, 4),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions")


if __name__ == "__main__":
    main()
