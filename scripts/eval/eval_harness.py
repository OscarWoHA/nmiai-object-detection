"""Evaluation harness that mimics the competition scoring exactly.

Usage:
  # Step 1: Create the val split (once)
  python eval_harness.py split

  # Step 2: Retrain your model on train-only data
  # (or use the full-data model and accept slight optimistic bias)

  # Step 3: Evaluate a submission directory
  python eval_harness.py eval --submission ./submission_v2 --name v2_baseline

  # Step 4: Compare all evaluated submissions
  python eval_harness.py compare
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter
import random
import numpy as np

ROOT = Path(__file__).parent
COCO_ANN = ROOT / "shelf_images_with_coco_annotations" / "annotations.json"
IMG_DIR = ROOT / "shelf_images_with_coco_annotations" / "images"
EVAL_DIR = ROOT / "eval_workspace"
SPLITS_DIR = EVAL_DIR / "splits"
RESULTS_DIR = EVAL_DIR / "results"

random.seed(42)
np.random.seed(42)


def load_coco():
    with open(COCO_ANN) as f:
        return json.load(f)


# ============================================================
# SPLIT STRATEGY
# ============================================================
def create_splits():
    """Create a proper stratified train/val split.

    Strategy (from ML best practices for imbalanced multi-label):
    - Use iterative stratification: each image has multiple categories
    - Ensure every category with >=2 examples has at least 1 in val
    - Target 20% val split (50 images) for statistical significance
    - Categories with exactly 1 example go to train (can't split)
    - Seed is fixed for reproducibility
    """
    coco = load_coco()
    images = coco["images"]
    annotations = coco["annotations"]

    # Group annotations by image
    img_anns = defaultdict(list)
    for ann in annotations:
        img_anns[ann["image_id"]].append(ann)

    # Category counts across dataset
    cat_counts = Counter(ann["category_id"] for ann in annotations)

    # Categories that appear in each image
    img_cats = {}
    for img in images:
        img_cats[img["id"]] = set(
            ann["category_id"] for ann in img_anns[img["id"]]
        )

    # Iterative stratification
    all_ids = [img["id"] for img in images]
    val_ids = set()
    train_ids = set()

    # Track how many val examples each category has
    cat_val_counts = Counter()

    # First pass: for each category with >=2 examples, ensure at least 1 image in val
    # Sort categories by count (rarest first) to prioritize rare class representation
    sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1])

    for cat_id, count in sorted_cats:
        if count < 2:
            continue  # singleton categories can't be split
        if cat_val_counts[cat_id] > 0:
            continue  # already have val coverage

        # Find images containing this category that aren't already assigned
        candidate_imgs = [
            img_id for img_id in all_ids
            if img_id not in val_ids and img_id not in train_ids
            and cat_id in img_cats[img_id]
        ]

        if not candidate_imgs:
            # All images with this cat are assigned; find one in train to move
            candidate_imgs = [
                img_id for img_id in train_ids
                if cat_id in img_cats[img_id]
            ]

        if candidate_imgs:
            # Pick the image that covers the most uncovered categories
            best_img = max(
                candidate_imgs,
                key=lambda img_id: len(
                    img_cats[img_id] - set(c for c in cat_val_counts if cat_val_counts[c] > 0)
                )
            )
            val_ids.add(best_img)
            train_ids.discard(best_img)
            for c in img_cats[best_img]:
                cat_val_counts[c] += len([a for a in img_anns[best_img] if a["category_id"] == c])

    # Second pass: fill up to ~20% val from remaining unassigned images
    unassigned = [i for i in all_ids if i not in val_ids and i not in train_ids]
    random.shuffle(unassigned)

    target_val = int(len(all_ids) * 0.20)
    while len(val_ids) < target_val and unassigned:
        img_id = unassigned.pop()
        val_ids.add(img_id)
        for c in img_cats[img_id]:
            cat_val_counts[c] += len([a for a in img_anns[img_id] if a["category_id"] == c])

    # Everything else is train
    train_ids = set(all_ids) - val_ids

    # Save splits
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Create val image directory (symlinks)
    val_img_dir = SPLITS_DIR / "val_images"
    val_img_dir.mkdir(parents=True, exist_ok=True)
    train_img_dir = SPLITS_DIR / "train_images"
    train_img_dir.mkdir(parents=True, exist_ok=True)

    img_map = {img["id"]: img for img in images}
    for img_id in val_ids:
        src = IMG_DIR / img_map[img_id]["file_name"]
        dst = val_img_dir / img_map[img_id]["file_name"]
        if not dst.exists():
            dst.symlink_to(src.resolve())

    for img_id in train_ids:
        src = IMG_DIR / img_map[img_id]["file_name"]
        dst = train_img_dir / img_map[img_id]["file_name"]
        if not dst.exists():
            dst.symlink_to(src.resolve())

    # Create val ground truth in COCO format
    val_annotations = [ann for ann in annotations if ann["image_id"] in val_ids]
    val_images = [img for img in images if img["id"] in val_ids]
    val_coco = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco["categories"],
    }
    with open(SPLITS_DIR / "val_annotations.json", "w") as f:
        json.dump(val_coco, f)

    # Save split info
    split_info = {
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "seed": 42,
    }
    with open(SPLITS_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Stats
    val_cat_coverage = len([c for c in cat_counts if cat_val_counts.get(c, 0) > 0])
    print(f"Split created:")
    print(f"  Train: {len(train_ids)} images")
    print(f"  Val:   {len(val_ids)} images")
    print(f"  Val annotations: {len(val_annotations)}")
    print(f"  Val category coverage: {val_cat_coverage}/{len(cat_counts)} categories")
    print(f"  Categories with 0 val examples: {len(cat_counts) - val_cat_coverage}")
    print(f"  Saved to: {SPLITS_DIR}")


# ============================================================
# mAP COMPUTATION
# ============================================================
def compute_ap(recalls, precisions):
    """Compute AP using 101-point COCO-style interpolation."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # 101-point interpolation
    recall_points = np.linspace(0, 1, 101)
    ap = 0.0
    for t in recall_points:
        precs_at_recall = mpre[mrec >= t]
        ap += max(precs_at_recall) if len(precs_at_recall) > 0 else 0.0
    return ap / 101.0


def compute_iou(box1, box2):
    """IoU between two COCO [x,y,w,h] boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def compute_competition_score(predictions, ground_truths):
    """Compute the exact competition metric:
    Score = 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5

    Detection mAP: IoU >= 0.5, category IGNORED (all treated as one class)
    Classification mAP: IoU >= 0.5 AND category must match (per-category AP, then mean)
    """
    # Group by image
    pred_by_img = defaultdict(list)
    gt_by_img = defaultdict(list)
    for p in predictions:
        pred_by_img[p["image_id"]].append(p)
    for g in ground_truths:
        gt_by_img[g["image_id"]].append(g)

    all_img_ids = set(list(pred_by_img.keys()) + list(gt_by_img.keys()))

    # ---- Detection mAP (class-agnostic) ----
    det_preds = []  # (score, is_tp)
    det_total_gt = 0

    for img_id in all_img_ids:
        preds = sorted(pred_by_img[img_id], key=lambda x: -x["score"])
        gts = list(gt_by_img[img_id])
        det_total_gt += len(gts)
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

            if best_iou >= 0.5 and best_j >= 0:
                matched[best_j] = True
                det_preds.append((pred["score"], True))
            else:
                det_preds.append((pred["score"], False))

    # Sort by score descending
    det_preds.sort(key=lambda x: -x[0])
    tp = 0
    det_precisions, det_recalls = [], []
    for i, (score, is_tp) in enumerate(det_preds):
        if is_tp:
            tp += 1
        det_precisions.append(tp / (i + 1))
        det_recalls.append(tp / det_total_gt if det_total_gt > 0 else 0)

    det_map = compute_ap(det_recalls, det_precisions) if det_total_gt > 0 else 0.0

    # ---- Classification mAP (per-category) ----
    gt_categories = set(g["category_id"] for g in ground_truths)
    cls_aps = []

    for cat_id in gt_categories:
        cat_preds = []
        cat_total_gt = 0

        for img_id in all_img_ids:
            preds = sorted(
                [p for p in pred_by_img[img_id] if p["category_id"] == cat_id],
                key=lambda x: -x["score"]
            )
            gts = [g for g in gt_by_img[img_id] if g["category_id"] == cat_id]
            cat_total_gt += len(gts)
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

                if best_iou >= 0.5 and best_j >= 0:
                    matched[best_j] = True
                    cat_preds.append((pred["score"], True))
                else:
                    cat_preds.append((pred["score"], False))

        if cat_total_gt == 0:
            continue

        cat_preds.sort(key=lambda x: -x[0])
        tp = 0
        precisions, recalls = [], []
        for i, (score, is_tp) in enumerate(cat_preds):
            if is_tp:
                tp += 1
            precisions.append(tp / (i + 1))
            recalls.append(tp / cat_total_gt)

        cls_aps.append(compute_ap(recalls, precisions))

    cls_map = np.mean(cls_aps) if cls_aps else 0.0

    combined = 0.7 * det_map + 0.3 * cls_map

    return {
        "combined_score": round(combined, 4),
        "detection_mAP": round(det_map, 4),
        "classification_mAP": round(cls_map, 4),
        "num_predictions": len(predictions),
        "num_gt": len(ground_truths),
        "num_gt_categories": len(gt_categories),
        "num_cls_aps": len(cls_aps),
    }


# ============================================================
# RUN SUBMISSION
# ============================================================
def evaluate_submission(submission_dir, name=None):
    """Run a submission's run.py on val images and score it."""
    submission_dir = Path(submission_dir)
    name = name or submission_dir.name

    # Check split exists
    val_img_dir = SPLITS_DIR / "val_images"
    val_ann_path = SPLITS_DIR / "val_annotations.json"
    if not val_img_dir.exists():
        print("ERROR: No val split found. Run: python eval_harness.py split")
        return None

    # Check run.py exists
    run_py = submission_dir / "run.py"
    if not run_py.exists():
        print(f"ERROR: No run.py in {submission_dir}")
        return None

    # Create output directory
    result_dir = RESULTS_DIR / name
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "predictions.json"

    # Run the submission
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"Submission: {submission_dir}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, str(run_py.resolve()),
        "--input", str(val_img_dir.resolve()),
        "--output", str(output_path.resolve()),
    ]

    print(f"Running: {' '.join(cmd)}")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"FAILED (exit code {result.returncode})")
            print(f"stderr: {result.stderr[-500:]}")
            return None

        print(f"Completed in {elapsed:.1f}s")
        if result.stdout:
            print(f"stdout: {result.stdout[-200:]}")

    except subprocess.TimeoutExpired:
        print("TIMEOUT after 600s")
        return None

    # Load predictions
    if not output_path.exists():
        print("ERROR: No predictions.json generated")
        return None

    with open(output_path) as f:
        predictions = json.load(f)

    # Load ground truth
    with open(val_ann_path) as f:
        val_coco = json.load(f)
    ground_truths = [
        {"image_id": ann["image_id"], "category_id": ann["category_id"], "bbox": ann["bbox"]}
        for ann in val_coco["annotations"]
    ]

    # Score
    scores = compute_competition_score(predictions, ground_truths)
    scores["runtime_seconds"] = round(elapsed, 1)
    scores["name"] = name

    # Save
    with open(result_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    print(f"\nResults for {name}:")
    print(f"  Combined Score:       {scores['combined_score']:.4f}")
    print(f"  Detection mAP@0.5:   {scores['detection_mAP']:.4f}")
    print(f"  Classification mAP:  {scores['classification_mAP']:.4f}")
    print(f"  Runtime:             {scores['runtime_seconds']}s")
    print(f"  Predictions:         {scores['num_predictions']}")
    print(f"  Val GT:              {scores['num_gt']} ({scores['num_gt_categories']} categories)")

    return scores


# ============================================================
# COMPARE
# ============================================================
def compare_results():
    """Compare all evaluated submissions."""
    if not RESULTS_DIR.exists():
        print("No results yet. Run eval first.")
        return

    results = []
    for d in sorted(RESULTS_DIR.iterdir()):
        scores_file = d / "scores.json"
        if scores_file.exists():
            with open(scores_file) as f:
                results.append(json.load(f))

    if not results:
        print("No results found.")
        return

    results.sort(key=lambda x: -x["combined_score"])

    print(f"\n{'='*80}")
    print(f"{'Name':<25s} {'Combined':>8s} {'Det mAP':>8s} {'Cls mAP':>8s} {'Runtime':>8s} {'Preds':>6s}")
    print(f"{'='*80}")
    for r in results:
        print(f"{r['name']:<25s} {r['combined_score']:>8.4f} {r['detection_mAP']:>8.4f} {r['classification_mAP']:>8.4f} {r['runtime_seconds']:>7.1f}s {r['num_predictions']:>6d}")
    print(f"{'='*80}")

    if len(results) > 1:
        best = results[0]
        for r in results[1:]:
            delta = best["combined_score"] - r["combined_score"]
            print(f"  {best['name']} beats {r['name']} by {delta:+.4f}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("split", help="Create train/val split")

    eval_p = sub.add_parser("eval", help="Evaluate a submission")
    eval_p.add_argument("--submission", required=True, help="Path to submission directory")
    eval_p.add_argument("--name", help="Name for this evaluation run")

    sub.add_parser("compare", help="Compare all evaluated submissions")

    args = parser.parse_args()

    if args.command == "split":
        create_splits()
    elif args.command == "eval":
        evaluate_submission(args.submission, args.name)
    elif args.command == "compare":
        compare_results()
    else:
        parser.print_help()
