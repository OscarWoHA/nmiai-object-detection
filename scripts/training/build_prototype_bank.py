"""Build DINOv2 prototype bank with per-angle embeddings + shelf GT crop prototypes.

Per GPT-5.4 advice: don't average angles, keep each separately.
Add shelf GT crop embeddings for domain adaptation.
"""
import json
import numpy as np
import torch
import timm
from timm.data import resolve_data_config, create_transform
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).parent
COCO_ANN = ROOT / "shelf_images_with_coco_annotations" / "annotations.json"
IMG_DIR = ROOT / "shelf_images_with_coco_annotations" / "images"
MULTIANGLE_DIR = ROOT / "images_multiangle"
BARCODE_MAP = ROOT / "barcode_to_category.json"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# Load DINOv2-Small
print("Loading DINOv2-ViT-S/14...")
model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
model = model.to(device).eval()

data_config = resolve_data_config(model=model)
transform = create_transform(**data_config, is_training=False)

# Save weights for submission
weights_path = ROOT / "dinov2_vits14.pt"
torch.save(model.cpu().state_dict(), str(weights_path))
model = model.to(device)
print(f"DINOv2 weights saved: {weights_path.stat().st_size / 1e6:.1f} MB")


def embed_image(img):
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]


# Load mappings
barcode_to_cat = json.loads(BARCODE_MAP.read_text())
with open(COCO_ANN) as f:
    coco = json.load(f)

img_map = {img["id"]: img for img in coco["images"]}

# === 1. Per-angle reference embeddings (don't average!) ===
print("\n=== Computing per-angle reference embeddings ===")
ref_embeddings = []
ref_category_ids = []

product_dirs = sorted([d for d in MULTIANGLE_DIR.iterdir() if d.is_dir()])
for d in product_dirs:
    barcode = d.name
    if barcode not in barcode_to_cat:
        continue
    cat_id = barcode_to_cat[barcode]

    for img_path in sorted(d.glob("*.jpg")):
        try:
            img = Image.open(img_path).convert("RGB")
            emb = embed_image(img)
            ref_embeddings.append(emb)
            ref_category_ids.append(cat_id)
        except Exception as e:
            print(f"  Error: {img_path}: {e}")

print(f"Reference embeddings: {len(ref_embeddings)} (per-angle, not averaged)")

# === 2. Shelf GT crop prototypes ===
print("\n=== Computing shelf GT crop prototypes ===")
from collections import Counter, defaultdict

cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])
crop_embeddings = []
crop_category_ids = []

# Group annotations by category
anns_by_cat = defaultdict(list)
for ann in coco["annotations"]:
    anns_by_cat[ann["category_id"]].append(ann)

for cat_id, anns in anns_by_cat.items():
    # For rare classes (<15 examples), use all crops
    # For frequent classes, use up to 8 best crops (largest area)
    max_crops = 12 if cat_counts[cat_id] < 15 else 8
    sorted_anns = sorted(anns, key=lambda a: -a["area"])[:max_crops]

    for ann in sorted_anns:
        img_info = img_map[ann["image_id"]]
        x, y, w, h = ann["bbox"]

        # Skip tiny crops
        if w < 20 or h < 20 or w * h < 400:
            continue

        # Aspect ratio filter
        ratio = w / h if h > 0 else 0
        if ratio < 0.2 or ratio > 5.0:
            continue

        # Add 8% padding (GPT-5.4 recommended)
        pad = 0.08
        x1 = max(0, x - w * pad)
        y1 = max(0, y - h * pad)
        x2 = min(img_info["width"], x + w + w * pad)
        y2 = min(img_info["height"], y + h + h * pad)

        try:
            img = Image.open(IMG_DIR / img_info["file_name"]).convert("RGB")
            crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
            emb = embed_image(crop)
            crop_embeddings.append(emb)
            crop_category_ids.append(cat_id)
        except Exception as e:
            pass

print(f"Shelf crop prototypes: {len(crop_embeddings)}")

# === 3. Combine all prototypes ===
all_embeddings = ref_embeddings + crop_embeddings
all_category_ids = ref_category_ids + crop_category_ids

print(f"\nTotal prototype bank: {len(all_embeddings)} embeddings")
print(f"Categories covered: {len(set(all_category_ids))}")

# Save
bank = {
    "embeddings": np.array(all_embeddings, dtype=np.float16),  # FP16 to save space
    "category_ids": np.array(all_category_ids, dtype=np.int16),
    "n_reference": len(ref_embeddings),
    "n_shelf_crops": len(crop_embeddings),
}

out_path = ROOT / "prototype_bank.pt"
torch.save(bank, str(out_path))
print(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
