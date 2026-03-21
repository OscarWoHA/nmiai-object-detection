#!/bin/bash
# VM4: Compute DINOv2 prototype bank (per-angle + shelf GT crops)
set -e

echo "=== Installing packages ==="
sudo apt-get update -qq && sudo apt-get install -y -qq unzip libgl1-mesa-glx libglib2.0-0
pip install timm==0.9.12 -q

echo "=== Copying data from VM1 ==="
# VM1 has the data already unpacked
mkdir -p ~/data
gcloud compute scp --internal-ip yolo-train:~/data/train ~/data/train --zone=europe-west1-b --recurse 2>/dev/null || true

echo "=== Downloading data (fallback) ==="
# If copy failed, we need the URLs. User must paste them.
if [ ! -f ~/data/train/annotations.json ]; then
  echo "ERROR: Could not copy from VM1. Need download URLs."
  echo "Paste coco_dataset URL as first arg, product_images URL as second arg"
  exit 1
fi

echo "=== Computing prototype bank ==="
python3 << 'PYEOF'
import json, numpy as np, torch, timm
from timm.data import resolve_data_config, create_transform
from PIL import Image
from pathlib import Path
from collections import Counter, defaultdict
import os

DATA = Path(os.path.expanduser("~/data"))
RUNS = Path(os.path.expanduser("~/runs"))
RUNS.mkdir(exist_ok=True)

# Find files
ann_path = list(DATA.rglob("annotations.json"))[0]
img_candidates = [d for d in DATA.rglob("images") if d.is_dir() and any(d.glob("*.jpg"))]
img_dir = img_candidates[0]

with open(ann_path) as f:
    coco = json.load(f)
img_map = {img["id"]: img for img in coco["images"]}

# Find product image dirs
product_dirs = []
for d in DATA.rglob("*"):
    if d.is_dir() and any(d.glob("main.jpg")):
        product_dirs.append(d)
if not product_dirs:
    # Try flat structure
    for d in DATA.iterdir():
        if d.is_dir() and d.name not in ("train", "labels_nc1", "labels_nc356", "runs"):
            if any(d.glob("*.jpg")):
                product_dirs.append(d)
print(f"Found {len(product_dirs)} product dirs")

# Build barcode -> category mapping
cat_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
meta_files = list(DATA.rglob("metadata.json"))
barcode_to_cat = {}
if meta_files:
    with open(meta_files[0]) as f:
        meta = json.load(f)
    for product in meta["products"]:
        bc = product["product_code"]
        name = product["product_name"]
        if name in cat_name_to_id:
            barcode_to_cat[bc] = cat_name_to_id[name]
        else:
            for cn, ci in cat_name_to_id.items():
                if cn.strip().upper() == name.strip().upper():
                    barcode_to_cat[bc] = ci
                    break
print(f"Barcode mapping: {len(barcode_to_cat)} products")

# Load DINOv2
print("Loading DINOv2-ViT-S/14...")
model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
model = model.half().cuda().eval()
data_config = resolve_data_config(model=model)
transform = create_transform(**data_config, is_training=False)

# Save weights
torch.save(model.cpu().state_dict(), str(RUNS / "dinov2_vits14.pt"))
model = model.cuda()
print(f"DINOv2 weights: {(RUNS / 'dinov2_vits14.pt').stat().st_size / 1e6:.1f} MB")

def embed_image(img):
    tensor = transform(img).unsqueeze(0).cuda().half()
    with torch.no_grad():
        emb = model(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()[0]

# === 1. Per-angle reference embeddings (don't average!) ===
print("\n=== Per-angle reference embeddings ===")
ref_embs, ref_cats = [], []
for d in sorted(product_dirs):
    bc = d.name
    if bc not in barcode_to_cat:
        continue
    cat_id = barcode_to_cat[bc]
    for img_path in sorted(d.glob("*.jpg")):
        try:
            emb = embed_image(Image.open(img_path).convert("RGB"))
            ref_embs.append(emb)
            ref_cats.append(cat_id)
        except Exception as e:
            pass
print(f"Reference: {len(ref_embs)} embeddings (per-angle)")

# === 2. Shelf GT crop prototypes ===
print("\n=== Shelf GT crop prototypes ===")
cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])
anns_by_cat = defaultdict(list)
for ann in coco["annotations"]:
    anns_by_cat[ann["category_id"]].append(ann)

crop_embs, crop_cats = [], []
for cat_id, anns in anns_by_cat.items():
    max_crops = 12 if cat_counts[cat_id] < 15 else 8
    sorted_anns = sorted(anns, key=lambda a: -a["area"])[:max_crops]
    for ann in sorted_anns:
        info = img_map[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        if w < 20 or h < 20 or w * h < 400:
            continue
        ratio = w / h if h > 0 else 0
        if ratio < 0.2 or ratio > 5.0:
            continue
        pad = 0.08
        x1 = max(0, x - w * pad)
        y1 = max(0, y - h * pad)
        x2 = min(info["width"], x + w + w * pad)
        y2 = min(info["height"], y + h + h * pad)
        try:
            img = Image.open(img_dir / info["file_name"]).convert("RGB")
            crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
            emb = embed_image(crop)
            crop_embs.append(emb)
            crop_cats.append(cat_id)
        except:
            pass
print(f"Shelf crops: {len(crop_embs)} embeddings")

# === 3. Save combined bank ===
all_embs = ref_embs + crop_embs
all_cats = ref_cats + crop_cats
print(f"\nTotal bank: {len(all_embs)} embeddings, {len(set(all_cats))} categories")

bank = {
    "embeddings": np.array(all_embs, dtype=np.float16),
    "category_ids": np.array(all_cats, dtype=np.int16),
    "n_reference": len(ref_embs),
    "n_shelf_crops": len(crop_embs),
}
torch.save(bank, str(RUNS / "prototype_bank.pt"))
print(f"Saved: {(RUNS / 'prototype_bank.pt').stat().st_size / 1e6:.1f} MB")
PYEOF

echo "=== DONE ==="
