"""Evaluate predictions against ground truth using competition scoring.

Score = 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def compute_iou(box1, box2):
    """IoU between two [x, y, w, h] boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0


def compute_ap(precisions, recalls):
    """Compute AP using 101-point interpolation (COCO style)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    recall_points = np.linspace(0, 1, 101)
    ap = 0
    for t in recall_points:
        precs = mpre[mrec >= t]
        ap += (max(precs) if len(precs) > 0 else 0)
    return ap / 101


def compute_map(predictions, ground_truths, check_category=False, iou_threshold=0.5):
    """Compute mAP@0.5 for detection or classification."""
    # Group by image
    pred_by_img = defaultdict(list)
    gt_by_img = defaultdict(list)

    for p in predictions:
        pred_by_img[p["image_id"]].append(p)
    for g in ground_truths:
        gt_by_img[g["image_id"]].append(g)

    all_image_ids = set(list(pred_by_img.keys()) + list(gt_by_img.keys()))

    # For detection mAP, treat all as single class
    # For classification mAP, compute per-category then average

    if not check_category:
        # Detection: single class AP
        all_preds = []
        total_gt = 0

        for img_id in all_image_ids:
            preds = sorted(pred_by_img[img_id], key=lambda x: -x["score"])
            gts = list(gt_by_img[img_id])
            total_gt += len(gts)
            matched = [False] * len(gts)

            for pred in preds:
                best_iou = 0
                best_j = -1
                for j, gt in enumerate(gts):
                    if matched[j]:
                        continue
                    iou = compute_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j

                if best_iou >= iou_threshold and best_j >= 0:
                    matched[best_j] = True
                    all_preds.append((pred["score"], True))
                else:
                    all_preds.append((pred["score"], False))

        all_preds.sort(key=lambda x: -x[0])
        tp = 0
        precisions = []
        recalls = []
        for i, (score, is_tp) in enumerate(all_preds):
            if is_tp:
                tp += 1
            precisions.append(tp / (i + 1))
            recalls.append(tp / total_gt if total_gt > 0 else 0)

        return compute_ap(precisions, recalls) if total_gt > 0 else 0

    else:
        # Classification: per-category AP
        categories = set(g["category_id"] for g in ground_truths)
        aps = []

        for cat_id in categories:
            cat_preds = []
            total_gt = 0

            for img_id in all_image_ids:
                preds = sorted(
                    [p for p in pred_by_img[img_id] if p["category_id"] == cat_id],
                    key=lambda x: -x["score"]
                )
                gts = [g for g in gt_by_img[img_id] if g["category_id"] == cat_id]
                total_gt += len(gts)
                matched = [False] * len(gts)

                for pred in preds:
                    best_iou = 0
                    best_j = -1
                    for j, gt in enumerate(gts):
                        if matched[j]:
                            continue
                        iou = compute_iou(pred["bbox"], gt["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_j = j

                    if best_iou >= iou_threshold and best_j >= 0:
                        matched[best_j] = True
                        cat_preds.append((pred["score"], True))
                    else:
                        cat_preds.append((pred["score"], False))

            if total_gt == 0:
                continue

            cat_preds.sort(key=lambda x: -x[0])
            tp = 0
            precisions = []
            recalls = []
            for i, (score, is_tp) in enumerate(cat_preds):
                if is_tp:
                    tp += 1
                precisions.append(tp / (i + 1))
                recalls.append(tp / total_gt)

            aps.append(compute_ap(precisions, recalls))

        return np.mean(aps) if aps else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--annotations", default="shelf_images_with_coco_annotations/annotations.json")
    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)
    with open(args.annotations) as f:
        coco = json.load(f)

    # Convert COCO annotations to same format as predictions
    ground_truths = []
    for ann in coco["annotations"]:
        ground_truths.append({
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
        })

    # Only evaluate on images that exist in predictions
    pred_imgs = set(p["image_id"] for p in predictions)
    ground_truths = [g for g in ground_truths if g["image_id"] in pred_imgs]

    det_map = compute_map(predictions, ground_truths, check_category=False)
    cls_map = compute_map(predictions, ground_truths, check_category=True)
    score = 0.7 * det_map + 0.3 * cls_map

    print(f"Detection mAP@0.5:       {det_map:.4f}")
    print(f"Classification mAP@0.5:  {cls_map:.4f}")
    print(f"Combined score:          {score:.4f}")
    print(f"  (0.7 * {det_map:.4f} + 0.3 * {cls_map:.4f})")
    print(f"Predictions: {len(predictions)}, GT: {len(ground_truths)}, Images: {len(pred_imgs)}")


if __name__ == "__main__":
    main()
