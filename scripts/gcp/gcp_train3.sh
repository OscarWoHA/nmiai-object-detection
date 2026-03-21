#!/bin/bash
# VM3: RT-DETR-l + YOLOv8x nc=356
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Installing packages ==="
sudo apt-get update -qq && sudo apt-get install -y -qq unzip libgl1-mesa-glx libglib2.0-0
pip install ultralytics==8.1.0 timm==0.9.12 -q

echo "=== Downloading data ==="
mkdir -p ~/data && cd ~/data

curl -L -o coco_dataset.zip 'https://storage.googleapis.com/nmaichamps-participant-data/norgesgruppen-data/NM_NGD_coco_dataset.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=1065880881946-compute%40developer.gserviceaccount.com%2F20260319%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260319T205602Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host&X-Goog-Signature=d2bf6746362c91da63ec5903314ef57db444134f3033a141830aed03dac15f1f28edfd8f412afbacdc6d2bb098acae56c930499719fc135f0185caaf9f9ea38bd8a4569d3308a43a510b349ba4e1763793cf4a7a569b62d795ffd00182ff1909df3c244c79892bffae640ad279d7d4a70e5671ad927dcb87af867ffca6f2850598fad33b7541850f5f8bd7b022aea803412f09df2eb035c6d9b15309d23c7e38cb76cbbb48064f0805461e442fdd8aa9cdedd8219066653c096fcd2c6425d2fdab5e167c148d3a2e63965b455a8c9aa6bf0b4eaee8311b53f7388e9a150deb7dee33c2dc5060c9d25f4a5a12a45eba5d8ac302078e7f008f2976edbfc2b1ffd7'

curl -L -o product_images.zip 'https://storage.googleapis.com/nmaichamps-participant-data/norgesgruppen-data/NM_NGD_product_images.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=1065880881946-compute%40developer.gserviceaccount.com%2F20260319%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260319T205602Z&X-Goog-Expires=3600&X-Goog-SignedHeaders=host&X-Goog-Signature=d4fbc78b6921201f66698e4e457a6e49b7fe961dbaf812afa50585bad5ff29f54a3e004c839cb4d75c0c0126c0767c22f32df71ff468a5b5417124120bfeeb86ea62fc2809d958be8d3074b32fb08396ef280f023c74db398928fdbbd2e32f0863bbffb4b22c657fe531c3c7120d0d3261d6c03e7e81029fa5aa5f24ac7a11092d59fe7b9a30a0d5bd1a5d7bb6feaaa7ff198ab18b98190dc13196ba4826aa92a4a4c0bacc04a40f3f42c9543ce7aa3c60b7b31ba4db160e2486e371bcf813afb886c6fc9834181d7b749fa6e715b90e1b5c9f35a7b65b5fec4a1314b1409d6213d2bdbce0cdb1be6bd1f7c8d0ef14acb9d165d2a1de814c0e0af8b65bbb99bd'

echo "=== Unpacking ==="
unzip -qo coco_dataset.zip
unzip -qo product_images.zip

echo "=== Converting COCO to YOLO ==="
python3 << 'PYEOF'
import json, random, os
from pathlib import Path
from collections import Counter, defaultdict

random.seed(42)
DATA = Path(os.path.expanduser("~/data"))
ann_path = list(DATA.rglob("annotations.json"))[0]
img_candidates = [d for d in DATA.rglob("images") if d.is_dir() and any(d.glob("*.jpg"))]
img_dir = img_candidates[0]

with open(ann_path) as f:
    data = json.load(f)

images, annotations, categories = data["images"], data["annotations"], data["categories"]
img_map = {img["id"]: img for img in images}

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

labels = defaultdict(list)
for ann in annotations:
    img = img_map[ann["image_id"]]
    x, y, w, h = ann["bbox"]
    cx = max(0, min(1, (x + w/2) / img["width"]))
    cy = max(0, min(1, (y + h/2) / img["height"]))
    nw = max(0, min(1, w / img["width"]))
    nh = max(0, min(1, h / img["height"]))
    labels[ann["image_id"]].append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

for split, ids in [("train", train_ids), ("val", val_ids)]:
    lbl_dir = DATA / "labels_nc356" / split / "labels"
    img_dst = DATA / "labels_nc356" / split / "images"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    img_dst.mkdir(parents=True, exist_ok=True)
    for img_id in ids:
        info = img_map[img_id]
        src = img_dir / info["file_name"]
        dst = img_dst / info["file_name"]
        if not dst.exists() and src.exists():
            os.symlink(str(src), str(dst))
        lbl_file = lbl_dir / (Path(info["file_name"]).stem + ".txt")
        lines = labels.get(img_id, [])
        lbl_file.write_text("\n".join(lines) + "\n" if lines else "")

cat_names = [c["name"] for c in sorted(categories, key=lambda c: c["id"])]
with open(DATA / "data_nc356.yaml", "w") as f:
    f.write(f"path: {DATA}\ntrain: labels_nc356/train/images\nval: labels_nc356/val/images\n\nnc: {len(cat_names)}\nnames: {cat_names}\n")
print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
PYEOF

echo "=== Training YOLOv8x nc=356 ==="
python3 -c "
import torch; _L=torch.load; torch.load=lambda *a,**k: _L(*a,**{**k,'weights_only':False})
from ultralytics import YOLO
model = YOLO('yolov8x.pt')
model.train(
    data='$HOME/data/data_nc356.yaml',
    imgsz=1280, batch=2, epochs=200, patience=40,
    lr0=0.001, lrf=0.01, cos_lr=True, warmup_epochs=5,
    mosaic=1.0, close_mosaic=15, mixup=0.15,
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.5,
    translate=0.1, scale=0.4, fliplr=0.5, flipud=0.0, degrees=0.0,
    erasing=0.1, weight_decay=0.0005, max_det=500,
    project='$HOME/runs', name='yolov8x_nc356',
    save=True, plots=True, device=0,
)
"

echo "=== DONE ==="
