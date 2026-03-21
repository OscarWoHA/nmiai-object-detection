"""Train YOLOv8 locally. Monkey-patches torch.load for compatibility."""
import json
import torch
import random
from pathlib import Path
from collections import Counter, defaultdict

# Fix torch.load weights_only issue
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'weights_only': False})

from ultralytics import YOLO

ROOT = Path(__file__).parent
COCO_ANN = ROOT / "shelf_images_with_coco_annotations" / "annotations.json"
IMG_DIR = ROOT / "shelf_images_with_coco_annotations" / "images"

random.seed(42)


def prepare_yolo_data(nc1=True):
    """Convert COCO to YOLO format, return data.yaml path."""
    with open(COCO_ANN) as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]
    img_map = {img["id"]: img for img in images}

    # Stratified split — keep rare categories in train
    img_cats = defaultdict(set)
    for ann in annotations:
        img_cats[ann["image_id"]].add(ann["category_id"])

    cat_counts = Counter(ann["category_id"] for ann in annotations)
    rare_cats = {c for c, n in cat_counts.items() if n < 10}
    rare_images = {img_id for img_id, cats in img_cats.items() if cats & rare_cats}

    all_ids = [img["id"] for img in images]
    non_rare = [i for i in all_ids if i not in rare_images]
    random.shuffle(non_rare)
    val_count = min(max(1, int(len(all_ids) * 0.15)), len(non_rare))
    val_ids = set(non_rare[:val_count])
    train_ids = set(all_ids) - val_ids

    # Build labels
    mode = "nc1" if nc1 else "nc356"
    ds_dir = ROOT / f"yolo_{mode}"

    labels = defaultdict(list)
    for ann in annotations:
        img = img_map[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        cx = max(0, min(1, (x + w / 2) / img["width"]))
        cy = max(0, min(1, (y + h / 2) / img["height"]))
        nw = max(0, min(1, w / img["width"]))
        nh = max(0, min(1, h / img["height"]))
        cls_id = 0 if nc1 else ann["category_id"]
        labels[ann["image_id"]].append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    import os
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        lbl_dir = ds_dir / split / "labels"
        img_dir = ds_dir / split / "images"
        lbl_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)

        for img_id in ids:
            info = img_map[img_id]
            src = IMG_DIR / info["file_name"]
            dst = img_dir / info["file_name"]
            if not dst.exists() and src.exists():
                os.symlink(str(src.resolve()), str(dst))
            lbl_file = lbl_dir / (Path(info["file_name"]).stem + ".txt")
            lines = labels.get(img_id, [])
            lbl_file.write_text("\n".join(lines) + "\n" if lines else "")

    # Write data.yaml
    if nc1:
        yaml_content = f"path: {ds_dir.resolve()}\ntrain: train/images\nval: val/images\n\nnc: 1\nnames: ['product']\n"
    else:
        cat_names = [c["name"] for c in sorted(categories, key=lambda c: c["id"])]
        yaml_content = f"path: {ds_dir.resolve()}\ntrain: train/images\nval: val/images\n\nnc: {len(cat_names)}\nnames: {cat_names}\n"

    yaml_path = ds_dir / "data.yaml"
    yaml_path.write_text(yaml_content)

    print(f"Prepared {mode}: {len(train_ids)} train, {len(val_ids)} val")
    return str(yaml_path)


def train(nc1=True, model_size="n", epochs=50, imgsz=640):
    data_yaml = prepare_yolo_data(nc1=nc1)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = YOLO(f"yolov8{model_size}.pt")

    results = model.train(
        data=data_yaml,
        imgsz=imgsz,
        batch=4,
        epochs=epochs,
        patience=15,
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        weight_decay=0.0005,
        max_det=300,
        project=str(ROOT / "runs"),
        name=f"yolov8{model_size}_{'nc1' if nc1 else 'nc356'}",
        save=True,
        plots=True,
        device=device,
    )

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc356", action="store_true", help="Multi-class (default: nc1)")
    parser.add_argument("--model", default="n", choices=["n", "s", "m", "l"], help="YOLOv8 size")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    train(nc1=not args.nc356, model_size=args.model, epochs=args.epochs, imgsz=args.imgsz)
